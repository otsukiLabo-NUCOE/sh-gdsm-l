#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 4 harness: end-to-end validation of SH-GD/SM-L against a real,
locally hosted LLM agent.

Target model
------------
    NVIDIA-Nemotron-Nano-9B-v2-Japanese-Q8_0.gguf

served by llama.cpp's OpenAI-compatible server, e.g.

    llama-server -m /path/to/NVIDIA-Nemotron-Nano-9B-v2-Japanese-Q8_0.gguf \
                 --host 127.0.0.1 --port 8080 \
                 -c 8192 -ngl 99 --jinja

Then

    python llm_experiment.py --episodes 200 --seeds 5 --out llm_results.json

NOTHING IS SIMULATED HERE. Every tool call is issued by the model, every
schema validation is a real jsonschema validation, and every failure the agent
sees is a real failure. The twelve tools mirror the task universe of
Experiments 1-3 so that the two sets of results are directly comparable; the
*payloads* are synthetic (a stub market-data service), which is what makes the
ground-truth relation strengths knowable while the agent's behaviour is real.

Arms
----
  A  llm-zeroshot   : on failure the model is handed the tool list, the current
                      schema set and the failure reason, and picks the next
                      tool itself. No learned relation graph. This is the
                      baseline that operational agent frameworks implement.
  B  llm-reflect    : as A, plus up to 3 reflect-and-retry attempts on the same
                      tool before giving up (the Google ADK pattern).
  C  shgdsm         : every decision (initial and after a failure) is made by
                      the SH-GD/SM-L state machine from the learned relation
                      graph (manuscript Eq.15/Eq.16). No model is called.
  D  shgdsm-hinted  : as C, but the learned rel table is also injected into the
                      model's context, so the model can use it in its own
                      planning.

Reported metrics
----------------
  completion rate, 95% CI over seeds
  mean steps to completion
  malformed tool-call rate (calls rejected before execution)
  wall-clock seconds and prompt/completion tokens per episode
  for C and D: claimed P(Goal) at S0 and the calibration error against the
  measured completion rate

Author: (revision support)
License: MIT
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None

sys.path.insert(0, ".")
import shgdsm as M  # the model implementation shared with Experiments 1-3

DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_MODEL = "NVIDIA-Nemotron-Nano-9B-v2-Japanese-Q8_0"
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"

# ---------------------------------------------------------------------------
# JSON Schemas for the twelve schemas of the task universe
# ---------------------------------------------------------------------------

SCHEMA_DEFS: Dict[str, dict] = {
    "instruction": {
        "type": "object",
        "required": ["goal"],
        "properties": {"goal": {"type": "string"}},
    },
    "stock_raw": {
        "type": "object",
        "required": ["ticker", "period", "bars"],
        "properties": {
            "ticker": {"type": "string"},
            "period": {"type": "string"},
            "bars": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["date", "close"],
                    "properties": {"date": {"type": "string"},
                                   "close": {"type": "number"}},
                },
            },
        },
    },
    "econ_raw": {
        "type": "object",
        "required": ["indicator", "period", "series"],
        "properties": {
            "indicator": {"type": "string"},
            "period": {"type": "string"},
            "series": {
                "type": "array", "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["date", "value"],
                    "properties": {"date": {"type": "string"},
                                   "value": {"type": "number"}},
                },
            },
        },
    },
    # market_raw is a generic feed: it MAY carry the fields stock_raw/econ_raw
    # need, and it may not. That is what makes rho* a frequency, not a distance.
    "market_raw": {
        "type": "object",
        "required": ["feed", "records"],
        "properties": {
            "feed": {"type": "string"},
            "records": {"type": "array", "minItems": 1, "items": {"type": "object"}},
        },
    },
    "news_raw": {
        "type": "object",
        "required": ["articles"],
        "properties": {
            "articles": {
                "type": "array", "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["title", "body"],
                    "properties": {"title": {"type": "string"},
                                   "body": {"type": "string"}},
                },
            },
        },
    },
    "stock_clean": {
        "type": "object",
        "required": ["ticker", "returns"],
        "properties": {"ticker": {"type": "string"},
                       "returns": {"type": "array", "minItems": 1,
                                   "items": {"type": "number"}}},
    },
    "econ_clean": {
        "type": "object",
        "required": ["indicator", "normalized"],
        "properties": {"indicator": {"type": "string"},
                       "normalized": {"type": "array", "minItems": 1,
                                      "items": {"type": "number"}}},
    },
    # news_summary requires summary PROSE
    "news_summary": {
        "type": "object",
        "required": ["summary_text"],
        "properties": {"summary_text": {"type": "string", "minLength": 80}},
    },
    # news_digest looks similar but carries only headlines -> fails the above
    "news_digest": {
        "type": "object",
        "required": ["headlines"],
        "properties": {"headlines": {"type": "array", "minItems": 1,
                                     "items": {"type": "string"}}},
    },
    "report_draft": {
        "type": "object",
        "required": ["sections"],
        "properties": {"sections": {"type": "array", "minItems": 3,
                                    "items": {"type": "string"}}},
    },
    "final_report": {
        "type": "object",
        "required": ["title", "body"],
        "properties": {"title": {"type": "string"},
                       "body": {"type": "string", "minLength": 200}},
    },
}


def validate(schema_name: str, payload: Any) -> Tuple[bool, str]:
    """Real schema validation. This is the signal that separates failure type
    (a) execution error from failure type (b) schema mismatch."""
    if jsonschema is None:
        raise RuntimeError("pip install jsonschema")
    try:
        jsonschema.validate(payload, SCHEMA_DEFS[schema_name])
        return True, ""
    except jsonschema.ValidationError as e:  # type: ignore[attr-defined]
        return False, f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"


# ---------------------------------------------------------------------------
# Stub services. The *instance-level* variability here is the ground truth:
# rho* is never written down as a constant, it emerges from these generators.
# ---------------------------------------------------------------------------


def _bars(rng: random.Random, n: int, with_close: bool = True) -> List[dict]:
    out = []
    for i in range(n):
        rec: Dict[str, Any] = {"date": f"2026-09-{i+1:02d}"}
        if with_close:
            rec["close"] = round(1000 + rng.gauss(0, 25), 2)
        out.append(rec)
    return out


# Instance-level properties of real feeds. These -- not a table of constants --
# are what induce rho*. The induced values are MEASURED (--measure-rho) and
# reported, never assumed.
P_MARKET_HAS_CLOSE = 0.904   # generic feed carries a close-equivalent field
P_MARKET_HAS_VALUE = 0.744   # generic feed carries an index-value field
P_MARKET_DAILY = 0.95        # returned at daily rather than intraday grain
P_MARKET_FULL_PERIOD = 0.99  # covers the requested period
P_ECON_DAILY = 0.30          # an indicator series that happens to be daily
P_STOCK_ADJUSTED = 0.25      # price series carrying an index-comparable level
P_DIGEST_LONG = 0.15         # a digest whose headlines amount to prose


def make_market_raw(rng: random.Random) -> dict:
    """Generic feed. Fields are present only some of the time, at a granularity
    that is sometimes wrong, over a period that is sometimes short. The fraction
    of instances that then validate as stock_raw / econ_raw *is* rho*."""
    recs: List[Dict[str, Any]] = []
    has_close = rng.random() < P_MARKET_HAS_CLOSE
    has_value = rng.random() < P_MARKET_HAS_VALUE
    daily = rng.random() < P_MARKET_DAILY
    long_enough = rng.random() < P_MARKET_FULL_PERIOD
    n = 20 if long_enough else 1
    for i in range(n):
        r: Dict[str, Any] = {"date" if daily else "timestamp": f"2026-09-{i+1:02d}"}
        if has_close:
            r["close"] = round(1000 + rng.gauss(0, 25), 2)
        if has_value:
            r["value"] = round(100 + rng.gauss(0, 3), 3)
        recs.append(r)
    return {"feed": "generic-market-feed", "records": recs}


def coerce(x_schema: str, s_schema: str, payload: dict) -> Optional[dict]:
    """What the runtime actually does when schema x is handed to a requirement
    s: a best-effort field projection, no more. Whether the projection then
    validates is decided by validate(), not by a constant."""
    if x_schema == s_schema:
        return payload
    if x_schema == "market_raw" and s_schema == "stock_raw":
        recs = payload.get("records", [])
        bars = [{"date": r["date"], "close": r["close"]}
                for r in recs if "date" in r and "close" in r]
        return {"ticker": "GENERIC", "period": "2026-09", "bars": bars}
    if x_schema == "market_raw" and s_schema == "econ_raw":
        recs = payload.get("records", [])
        ser = [{"date": r["date"], "value": r["value"]}
               for r in recs if "date" in r and "value" in r]
        return {"indicator": "GENERIC", "period": "2026-09", "series": ser}
    if x_schema == "econ_raw" and s_schema == "stock_raw":
        # an indicator series can stand in for a price series only when it is
        # at daily grain; otherwise the projection has no close to offer and
        # the validator -- not this function -- rejects it
        ser = payload.get("series", [])
        daily = payload.get("granularity") == "daily"
        bars = [{"date": r["date"], **({"close": r["value"]} if daily else {})}
                for r in ser]
        return {"ticker": "PROXY", "period": "2026-09", "bars": bars}
    if x_schema == "stock_raw" and s_schema == "econ_raw":
        bars = payload.get("bars", [])
        adj = bool(payload.get("adjusted"))
        ser = [{"date": r["date"], **({"value": r["close"]} if adj else {})}
               for r in bars]
        return {"indicator": "PROXY", "period": "2026-09", "series": ser}
    if x_schema == "news_digest" and s_schema == "news_summary":
        heads = payload.get("headlines", [])
        return {"summary_text": " / ".join(heads)}      # usually < 80 chars
    return None


TOOLS_SPEC = [
    ("FetchStockA", [], "stock_raw", 0.80),
    ("FetchStockB", [], "stock_raw", 0.60),
    ("FetchEcon", [], "econ_raw", 0.70),
    ("FetchMarketFeed", [], "market_raw", 0.75),
    ("FetchNews", [], "news_raw", 0.90),
    ("FetchNewsAlt", [], "news_raw", 0.45),
    ("CleanStock", ["stock_raw"], "stock_clean", 0.95),
    ("CleanEcon", ["econ_raw"], "econ_clean", 0.90),
    ("SummarizeNews", ["news_raw"], "news_summary", 0.85),
    ("SummarizeNewsTrap", ["news_raw"], "news_digest", 0.90),
    ("ComposeReport", ["stock_clean", "econ_clean", "news_summary"], "report_draft", 0.85),
    ("FinalizeReport", ["report_draft"], "final_report", 0.90),
]


@dataclass
class FaultEnv:
    """Fault injection for Experiment 6 (recovery from disruptions).

    outage     tools that return 503 on every call (permanent outage)
    drift      tools whose output silently changes format (stock bars carry
               'price' instead of 'close'), so the output no longer validates
    transient  tool -> number of initial calls per episode that return 503
    persistent_errors
               a natural 503 keeps that service down for the rest of the
               episode (the reading under which F in Eq.8 is never re-run);
               False gives i.i.d. per-call errors, under which retry pays.

    The object is per-episode state: call fresh() at the start of an episode.
    With env=None every function below behaves exactly as in Experiment 4.
    """
    outage: frozenset = frozenset()
    drift: frozenset = frozenset()
    transient: Dict[str, int] = field(default_factory=dict)
    persistent_errors: bool = True
    down: set = field(default_factory=set)
    calls: Dict[str, int] = field(default_factory=dict)

    def fresh(self) -> "FaultEnv":
        return FaultEnv(self.outage, self.drift, dict(self.transient),
                        self.persistent_errors)

    def affected(self) -> frozenset:
        return frozenset(self.outage) | frozenset(self.drift) | frozenset(self.transient)


def _drift(payload: dict) -> dict:
    bars = [{"date": b["date"], "price": b["close"]} for b in payload["bars"]]
    return {**payload, "bars": bars}


def run_tool(name: str, inputs: Dict[str, dict], rng: random.Random,
             env: Optional[FaultEnv] = None
             ) -> Tuple[Optional[str], Optional[dict], str]:
    """-> (output_schema, payload, error). error != '' means execution failed."""
    if env is not None:
        env.calls[name] = env.calls.get(name, 0) + 1
        if name in env.outage or name in env.down:
            return None, None, "ExecutionError: upstream service returned 503"
        if env.calls[name] <= env.transient.get(name, 0):
            return None, None, "ExecutionError: upstream service returned 503"
    out_schema, payload, err = _run_tool(name, rng)
    if env is not None:
        if err and env.persistent_errors:
            env.down.add(name)
        if not err and name in env.drift and out_schema == "stock_raw":
            payload = _drift(payload)
    return out_schema, payload, err


def _run_tool(name: str, rng: random.Random
              ) -> Tuple[Optional[str], Optional[dict], str]:
    spec = {t[0]: t for t in TOOLS_SPEC}[name]
    _, _, out_schema, p = spec
    if rng.random() >= p:
        return None, None, "ExecutionError: upstream service returned 503"
    if name == "FetchMarketFeed":
        return out_schema, make_market_raw(rng), ""
    if out_schema == "stock_raw":
        return out_schema, {"ticker": "7203", "period": "2026-09",
                            "adjusted": rng.random() < P_STOCK_ADJUSTED,
                            "bars": _bars(rng, 20)}, ""
    if out_schema == "econ_raw":
        return out_schema, {"indicator": "CPI", "period": "2026-09",
                            "granularity": "daily" if rng.random() < P_ECON_DAILY
                                           else "monthly",
                            "series": [{"date": f"2026-09-{i+1:02d}",
                                        "value": round(100 + rng.gauss(0, 2), 3)}
                                       for i in range(20)]}, ""
    if out_schema == "news_raw":
        return out_schema, {"articles": [{"title": f"headline {i}",
                                          "body": "body text " * 20}
                                         for i in range(5)]}, ""
    if out_schema == "stock_clean":
        return out_schema, {"ticker": "7203",
                            "returns": [round(rng.gauss(0, .01), 5) for _ in range(19)]}, ""
    if out_schema == "econ_clean":
        return out_schema, {"indicator": "CPI",
                            "normalized": [round(rng.gauss(0, 1), 4) for _ in range(19)]}, ""
    if out_schema == "news_summary":
        return out_schema, {"summary_text": "Summary prose. " * 30}, ""
    if out_schema == "news_digest":
        long_ = rng.random() < P_DIGEST_LONG
        heads = ([f"Headline {i}: " + "excerpt of the full story " * 4 for i in range(5)]
                 if long_ else [f"H{i}" for i in range(5)])
        return out_schema, {"headlines": heads}, ""
    if out_schema == "report_draft":
        return out_schema, {"sections": ["Equities", "Economic indicators", "News"]}, ""
    if out_schema == "final_report":
        return out_schema, {"title": "Economic report", "body": "Body text. " * 100}, ""
    return None, None, "ExecutionError: unknown tool"


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


@dataclass
class LlamaClient:
    """Any OpenAI-compatible chat-completions endpoint: llama.cpp's
    llama-server (backend 'llama') or a hosted / vLLM endpoint (backend
    'openai', key read from OPENAI_API_KEY). Wall-clock latency of every call is
    recorded, so tau_LLM in Eq.20 is measured rather than assumed."""
    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: float = 600.0
    api_key: Optional[str] = None
    # Nemotron-Nano-v2 reasons by default; '/no_think' in the system turn
    # switches that off so that one decision is one short JSON reply.
    no_think: bool = True
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    latency_s: float = 0.0

    def chat(self, messages: List[dict]) -> str:
        if self.no_think:
            messages = [{"role": "system", "content": "/no_think"}] + list(messages)
        req_body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.no_think:
            # llama.cpp --jinja: the chat template honours this flag; the
            # '/no_think' system turn alone is not always respected
            req_body["chat_template_kwargs"] = {"enable_thinking": False}
        body = json.dumps(req_body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.endpoint, data=body, headers=headers)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise SystemExit(
                f"\ncannot reach llama.cpp at {self.endpoint} ({e}).\n"
                f"start it with:\n"
                f"  llama-server -m NVIDIA-Nemotron-Nano-9B-v2-Japanese-Q8_0.gguf "
                f"--host 127.0.0.1 --port 8080 -c 8192 -ngl 99 --jinja\n")
        self.latency_s += time.perf_counter() - t0
        self.calls += 1
        u = payload.get("usage", {})
        self.prompt_tokens += int(u.get("prompt_tokens", 0))
        self.completion_tokens += int(u.get("completion_tokens", 0))
        text = payload["choices"][0]["message"].get("content") or ""
        return _THINK_RE.sub("", text)


@dataclass
class AnthropicClient:
    """Claude as the model-delegated agent (Experiments 4 and 6, real-LLM arms).

    Same interface as LlamaClient. Uses the official SDK
    (pip install anthropic); credentials are resolved from the environment
    (ANTHROPIC_API_KEY or an `ant auth login` profile). Server-side refusal
    fallbacks are enabled so that a policy decline does not silently end an
    episode. Effort is set explicitly because the default differs by model.
    """
    model: str = DEFAULT_CLAUDE_MODEL
    max_tokens: int = 16000
    effort: str = "medium"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    latency_s: float = 0.0
    _client: Any = None

    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    refusals: int = 0

    def chat(self, messages: List[dict]) -> str:
        if self._client is None:
            import anthropic
            ensure_anthropic_key()
            self._client = anthropic.Anthropic(max_retries=8)
        t0 = time.perf_counter()
        resp = self._client.beta.messages.create(
            model=self.model, max_tokens=self.max_tokens, messages=messages,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        self.latency_s += time.perf_counter() - t0
        self.calls += 1
        u = resp.usage
        self.prompt_tokens += int(u.input_tokens)
        self.completion_tokens += int(u.output_tokens)
        self.cache_read_tokens += int(getattr(u, "cache_read_input_tokens", 0) or 0)
        self.cache_write_tokens += int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        if resp.stop_reason == "refusal":
            self.refusals += 1
            return ""                       # counted as a malformed action
        return "".join(b.text for b in resp.content if b.type == "text")


def ensure_anthropic_key() -> None:
    """ANTHROPIC_API_KEY from the process, else from the Windows user
    environment (setx) when the parent process predates it. Never printed."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.name != "nt":
        return
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            os.environ["ANTHROPIC_API_KEY"] = winreg.QueryValueEx(k, "ANTHROPIC_API_KEY")[0]
    except OSError:
        pass


# USD per million tokens (input, output, cache read, cache write 5 min)
PRICES = {"claude-sonnet-5-5": (2.0, 10.0, 0.10, 2.5),
          "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0)}


def client_cost(c) -> float:
    p = PRICES.get(getattr(c, "model", ""), None)
    if p is None:
        return 0.0
    return (c.prompt_tokens * p[0] + c.completion_tokens * p[1]
            + getattr(c, "cache_read_tokens", 0) * p[2]
            + getattr(c, "cache_write_tokens", 0) * p[3]) / 1e6


EFFORT = "medium"


def make_client(backend: str, endpoint: str, model: Optional[str],
                no_think: bool = True):
    if backend == "anthropic":
        return AnthropicClient(model=model or DEFAULT_CLAUDE_MODEL, effort=EFFORT)
    if backend == "openai":
        return LlamaClient(endpoint, model or "gpt-5", api_key=os.environ.get("OPENAI_API_KEY"),
                           no_think=False)
    return LlamaClient(endpoint, model or DEFAULT_MODEL, no_think=no_think)


PROMPT_LANG = "en"

SYSTEM_PROMPT_EN = """You are an agent that writes an economic report by calling tools.
Goal schema: final_report

Available tools with their input and output schemas:
{tools}

Schemas currently held: {sigma}
Tools already executed successfully: {done}
Tools that have already failed: {failed}
{extra}
Choose exactly ONE tool to run next and output ONLY the following JSON.
Do not output any explanation, preamble or code fence.

{{"tool": "<tool name>", "inputs": {{"<input schema name>": "<held schema to pass in its place>"}}}}

In "inputs", map every input schema the tool requires to one of the schemas you
currently hold. If no exact match is held, you may pass one that you think can
substitute for it.
"""

SYSTEM_PROMPT_JA = """あなたはツールを使って経済レポートを作成するエージェントです。
目標スキーマ: final_report

利用可能なツールと、その入力スキーマ・出力スキーマ:
{tools}

現在保持しているスキーマ: {sigma}
成功済みのツール: {done}
これまでに失敗したツール: {failed}
{extra}
次に実行すべきツールを1つだけ選び、次のJSONのみを出力してください。
説明文・前置き・コードブロックは一切出力しないこと。

{{"tool": "<ツール名>", "inputs": {{"<入力スキーマ名>": "<代わりに流すスキーマ名>"}}}}

inputs には、そのツールが要求する各入力スキーマに対して、現在保持しているスキーマの
どれを流すかを指定します。完全一致が無い場合は、代わりに流せそうなものを指定できます。
"""


MSG = {
    "en": dict(
        goal="Write an economic report",
        tool_line="- {name}: inputs={ins} output={out}", none="(none)",
        last_failure="\nPrevious failure: {err}\n",
        hint="\nSchema substitutability learned from past execution logs:\n{hint}\n",
        malformed="Your previous output could not be parsed as JSON. Output the JSON only.",
        already_failed="{name} has already failed. Choose a different tool.",
        reflect="{name} failed: {err}\nState the cause in one sentence; the same tool will be retried.",
        not_learned="  (nothing learned yet)",
    ),
    "ja": dict(
        goal="経済レポートを作成する",
        tool_line="- {name}: 入力={ins} 出力={out}", none="(なし)",
        last_failure="\n直前の失敗: {err}\n",
        hint="\n過去ログから学習したスキーマ代替可能性:\n{hint}\n",
        malformed="前回の出力はJSONとして解釈できませんでした。JSONのみを出力してください。",
        already_failed="{name} は既に失敗しています。別のツールを選んでください。",
        reflect="{name} が失敗しました: {err}\n原因を1文で述べ、同じツールを再試行します。",
        not_learned="  (まだ学習されていません)",
    ),
}


def msg(key: str, **kw) -> str:
    return MSG[PROMPT_LANG][key].format(**kw)


def system_prompt() -> str:
    return SYSTEM_PROMPT_EN if PROMPT_LANG == "en" else SYSTEM_PROMPT_JA


def instruction_payload() -> dict:
    return {"goal": msg("goal")}


def tools_block() -> str:
    lines = []
    for name, ins, out, _ in TOOLS_SPEC:
        lines.append(msg("tool_line", name=name, ins=ins or [msg("none")], out=out))
    return "\n".join(lines)


def parse_action(text: str) -> Optional[Tuple[str, Dict[str, str]]]:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t.split("\n", 1)[-1] if "\n" in t else t
    i = t.find("{")
    if i < 0:
        return None
    # tolerate surplus closing braces or trailing text: take the first prefix
    # that parses as a JSON object
    obj = None
    for j in [k for k in range(i + 1, len(t)) if t[k] == "}"]:
        try:
            obj = json.loads(t[i:j + 1])
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(obj, dict):
        return None
    name = obj.get("tool")
    if name not in {s[0] for s in TOOLS_SPEC}:
        return None
    ins = obj.get("inputs") or {}
    if not isinstance(ins, dict):
        return None
    return name, {str(k): str(v) for k, v in ins.items()}


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    completed: bool
    steps: int
    malformed: int
    seconds: float
    observations: List[Tuple[str, str, bool]] = field(default_factory=list)
    # one record per tool execution (Experiment 6):
    #   (tool, ok, error, schemas held afterwards, decision calls so far)
    trace: List[Tuple[str, bool, str, frozenset, int]] = field(default_factory=list)


def _apply(name: str, sub: Dict[str, str], store: Dict[str, dict],
           rng: random.Random, env: Optional[FaultEnv] = None
           ) -> Tuple[bool, List[Tuple[str, str, bool]], str]:
    """Execute one tool call against the stub services with real validation."""
    spec = {t[0]: t for t in TOOLS_SPEC}[name]
    _, needed, _, _ = spec
    obs: List[Tuple[str, str, bool]] = []
    for s in needed:
        x = sub.get(s, s)
        if x not in store:
            return False, obs, f"SchemaMissing: {x} is not held"
        payload = coerce(x, s, store[x])
        if payload is None:
            if x != s:
                obs.append((x, s, False))
            return False, obs, f"SchemaMismatch: no projection from {x} to {s}"
        ok, why = validate(s, payload)
        if x != s:
            obs.append((x, s, ok))
        if not ok:
            return False, obs, f"SchemaMismatch: {x} -> {s}: {why}"
    out_schema, out_payload, err = run_tool(name, {}, rng, env)
    if err:
        return False, obs, err
    ok, why = validate(out_schema, out_payload)
    if not ok:
        return False, obs, f"SchemaMismatch(output): {out_schema}: {why}"
    store[out_schema] = out_payload
    return True, obs, ""


def exec_outcome(ok: bool, err: str) -> Optional[bool]:
    """Execution success as seen by the runtime: None when the tool never ran
    (an input failed validation), False on an execution error, else True."""
    if ok or err.startswith("SchemaMismatch(output)"):
        return True
    if err.startswith("ExecutionError"):
        return False
    return None


def episode_llm(client: "LlamaClient", rng: random.Random, reflect: int = 0,
                rel_hint: Optional[str] = None,
                env: Optional[FaultEnv] = None) -> EpisodeResult:
    t0 = time.time()
    store: Dict[str, dict] = {"instruction": instruction_payload()}
    failed: List[str] = []
    done: List[str] = []
    obs_all: List[Tuple[str, str, bool]] = []
    trace: List[Tuple[str, bool, str, frozenset, int]] = []
    calls0 = client.calls
    malformed = 0
    steps = 0
    last_error = ""

    def apply(name, sub):
        ok_, obs_, err_ = _apply(name, sub, store, rng, env)
        trace.append((name, ok_, err_, frozenset(store), client.calls - calls0))
        return ok_, obs_, err_

    for _ in range(24):
        if "final_report" in store:
            break
        extra = ""
        if last_error:
            extra += msg("last_failure", err=last_error)
        if rel_hint:
            extra += msg("hint", hint=rel_hint)
        prompt = system_prompt().format(tools=tools_block(),
                                   sigma=sorted(store), failed=failed or msg("none"),
                                   done=done or msg("none"),
                                   extra=extra)
        out = client.chat([{"role": "user", "content": prompt}])
        act = parse_action(out)
        if act is None:
            malformed += 1
            last_error = msg("malformed")
            continue
        name, sub = act
        if name in failed:
            last_error = msg("already_failed", name=name)
            continue
        steps += 1
        ok, obs, err = apply(name, sub)
        obs_all.extend(obs)
        if ok:
            last_error = ""
            if name not in done:
                done.append(name)
            continue
        tries = 0
        while (not ok) and tries < reflect:
            tries += 1
            steps += 1
            # ADK Reflect-and-Retry: the model reflects on the error before
            # every retry, which is one more model call
            client.chat([{"role": "user", "content":
                          msg("reflect", name=name, err=err)}])
            ok, obs, err = apply(name, sub)
            obs_all.extend(obs)
        if ok:
            last_error = ""
            if name not in done:
                done.append(name)
        else:
            failed.append(name)
            last_error = err
    return EpisodeResult("final_report" in store, steps, malformed,
                         time.time() - t0, obs_all, trace)


def episode_shgdsm(belief: M.RelBelief, cfg: M.Config, rng: random.Random,
                   explore: bool = False) -> EpisodeResult:
    """Arm C: recovery decided by Eq.15/Eq.16, no LLM call on the failure path.

    explore=True draws one rel sample per episode (Eq.17) and is used only for
    the training episodes; evaluation always uses the posterior mean.
    """
    t0 = time.time()
    rel = belief.snapshot_sample(rng) if explore else belief.snapshot_mean()
    pol = M.MyopicPolicy(rel, M.explore_cfg(cfg) if explore else cfg)
    store: Dict[str, dict] = {"instruction": instruction_payload()}
    st = M.State()
    obs_all: List[Tuple[str, str, bool]] = []
    steps = 0
    for _ in range(24):
        if "final_report" in store:
            break
        i = pol.act(st)
        if i is None:
            break
        t = M.TASKS[i]
        sigma = st.sigma()
        sub = {s: (pol.est.best_substitution(s, sigma)[0] or s) for s in t.inputs}
        steps += 1
        ok, obs, err = _apply(t.name, sub, store, rng)
        obs_all.extend(obs)
        for x, s, passed in obs:
            belief.observe(x, s, passed)
        eo = exec_outcome(ok, err)
        if eo is not None and hasattr(belief, "observe_exec"):
            belief.observe_exec(t.name, eo)
        st = st.with_success(i) if ok else st.with_failure(i)
    return EpisodeResult("final_report" in store, steps, 0,
                         time.time() - t0, obs_all)


def rel_hint_text(belief: M.RelBelief, cfg: M.Config) -> str:
    rows = []
    for (x, s) in M.LEARNABLE_PAIRS:
        m = belief.mean(x, s)
        if belief.n(x, s) > 0 and m >= cfg.theta:
            rows.append(f"  {x} -> {s}: {m:.3f} (n={belief.n(x, s)})")
    return "\n".join(rows) or msg("not_learned")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def ci95(xs: Sequence[float]) -> Tuple[float, float]:
    n = len(xs)
    m = statistics.fmean(xs)
    if n < 2:
        return m, 0.0
    return m, 1.96 * statistics.stdev(xs) / math.sqrt(n)


def main() -> None:
    global PROMPT_LANG
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=None,
                    help=f"chat-completions URL (default: {DEFAULT_ENDPOINT} for llama, "
                         "https://api.openai.com/v1/chat/completions for openai)")
    ap.add_argument("--model", default=None,
                    help=f"default: {DEFAULT_MODEL} (llama) / {DEFAULT_CLAUDE_MODEL} (anthropic)")
    ap.add_argument("--backend", choices=["llama", "openai", "anthropic"], default="llama")
    ap.add_argument("--lang", choices=["en", "ja"], default="en",
                    help="prompt language (the manuscript reports 'en')")
    ap.add_argument("--think", action="store_true",
                    help="let Nemotron reason before answering (default: /no_think)")
    ap.add_argument("--episodes", type=int, default=200,
                    help="evaluation episodes per arm per seed")
    ap.add_argument("--train-episodes", type=int, default=1600,
                    help="episodes used to learn rel for arms C and D")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-list", default=None,
                    help="comma-separated seeds (overrides --seeds), e.g. 3,4")
    ap.add_argument("--effort", default="medium",
                    help="Claude effort level (anthropic backend)")
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--estimator", choices=["lcb", "mean"], default="lcb",
                    help="planning-time point estimate for arms C/D (Eq.19 / Eq.4)")
    ap.add_argument("--out", default="llm_results.json")
    ap.add_argument("--measure-rho", type=int, default=0, metavar="N",
                    help="measure the rho* induced by the stub services over N "
                         "instances per pair, print it, and exit")
    args = ap.parse_args()
    PROMPT_LANG = args.lang
    global EFFORT
    EFFORT = args.effort
    seed_list = ([int(x) for x in args.seed_list.split(",")] if args.seed_list
                 else list(range(1, args.seeds + 1)))
    if args.endpoint is None:
        args.endpoint = ("https://api.openai.com/v1/chat/completions"
                         if args.backend == "openai" else DEFAULT_ENDPOINT)

    if args.measure_rho:
        rng = random.Random(12345)
        pairs = [("market_raw", "stock_raw"), ("market_raw", "econ_raw"),
                 ("econ_raw", "stock_raw"), ("stock_raw", "econ_raw"),
                 ("news_digest", "news_summary")]
        print(f"induced rho* over {args.measure_rho} instances per pair:")
        measured = []
        for x, s_ in pairs:
            ok = n = 0
            while n < args.measure_rho:
                _, payload, err = run_tool(
                    {"market_raw": "FetchMarketFeed", "econ_raw": "FetchEcon",
                     "stock_raw": "FetchStockA",
                     "news_digest": "SummarizeNewsTrap"}[x], {}, rng)
                if err:
                    continue          # execution error: not a rho* observation
                n += 1
                proj = coerce(x, s_, payload)
                if proj is None:
                    continue
                good, _ = validate(s_, proj)
                ok += good
            print(f"  {x:<12} -> {s_:<13} {ok/n:.4f}")
            measured.append([f"{x} -> {s_}", ok / n, M.REF_RHO[(x, s_)]])
        with open("rho_measured.json", "w", encoding="utf-8") as f:
            json.dump(dict(n=args.measure_rho, rows=measured), f, indent=2)
        return

    if jsonschema is None:
        raise SystemExit("pip install jsonschema")

    cfg = M.Config(**({"point_estimate": "lcb", "theta": 0.01}
                      if args.estimator == "lcb" else {}))
    arms = [a.strip().upper() for a in args.arms.split(",")]
    results: Dict[str, dict] = {}
    per_seed: Dict[str, List[dict]] = {}
    if os.path.exists(args.out):          # resume: keep finished (arm, seed) cells
        try:
            prev = json.load(open(args.out, encoding="utf-8"))
            per_seed = {a: [r for r in rs if r.get("seed") in seed_list]
                        for a, rs in prev.get("per_seed", {}).items()}
            print("resume:", {a: [r["seed"] for r in rs] for a, rs in per_seed.items()}, flush=True)
        except (OSError, ValueError):
            per_seed = {}

    def dump() -> None:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(dict(config=vars(args), results=results, per_seed=per_seed), f,
                      ensure_ascii=False, indent=2)

    for arm in arms:
        rates, steps_, malf, secs, claims, lat, calls_ep = [], [], [], [], [], [], []
        for sd in seed_list:
            rng = random.Random(10_000 + sd)
            client = make_client(args.backend, args.endpoint, args.model,
                                 no_think=not args.think)
            belief = M.RelBelief(cfg)
            if arm in ("C", "D"):
                trng = random.Random(20_000 + sd)
                for _ in range(args.train_episodes):
                    episode_shgdsm(belief, cfg, trng, explore=True)
                claims.append(M.Estimator(belief.snapshot_mean(), cfg).p_goal(M.State()))
            old = next((r for r in per_seed.get(arm, []) if r["seed"] == sd), None)
            if old is not None:
                rates.append(old["completion"])
                steps_.append(old["steps"])
                malf.append(old["malformed"])
                secs.append(old.get("seconds_per_episode", 0.0))
                calls_ep.append(old["llm_calls"] / args.episodes)
                if old["llm_calls"]:
                    lat.append(old["mean_call_latency_s"])
                continue
            done = 0
            s_acc = m_acc = 0
            t0 = time.time()
            for _ in range(args.episodes):
                if arm == "A":
                    r = episode_llm(client, rng, reflect=0)
                elif arm == "B":
                    r = episode_llm(client, rng, reflect=3)
                elif arm == "C":
                    r = episode_shgdsm(belief, cfg, rng)
                elif arm == "D":
                    r = episode_llm(client, rng, reflect=0,
                                    rel_hint=rel_hint_text(belief, cfg))
                else:
                    raise SystemExit(f"unknown arm {arm}")
                done += int(r.completed)
                s_acc += r.steps
                m_acc += r.malformed
            rates.append(done / args.episodes)
            steps_.append(s_acc / args.episodes)
            malf.append(m_acc / max(1, args.episodes))
            secs.append((time.time() - t0) / args.episodes)
            calls_ep.append(client.calls / args.episodes)
            if client.calls:
                lat.append(client.latency_s / client.calls)
            row = dict(seed=sd, completion=done / args.episodes,
                       steps=s_acc / args.episodes, malformed=m_acc / args.episodes,
                       llm_calls=client.calls, prompt_tokens=client.prompt_tokens,
                       completion_tokens=client.completion_tokens,
                       mean_call_latency_s=(client.latency_s / client.calls
                                            if client.calls else 0.0),
                       cache_read_tokens=getattr(client, "cache_read_tokens", 0),
                       refusals=getattr(client, "refusals", 0),
                       cost_usd=client_cost(client),
                       seconds_per_episode=secs[-1])
            per_seed.setdefault(arm, []).append(row)
            print(f"arm {arm} seed {sd}: " + json.dumps(row), flush=True)
            dump()                                # partial results survive a crash
        rm, rh = ci95(rates)
        entry = dict(arm=arm, completion=rm, completion_ci=rh,
                     steps=statistics.fmean(steps_),
                     malformed=statistics.fmean(malf),
                     seconds_per_episode=statistics.fmean(secs),
                     llm_calls_per_episode=statistics.fmean(calls_ep),
                     mean_call_latency_s=statistics.fmean(lat) if lat else 0.0)
        if claims:
            cm, ch = ci95(claims)
            entry.update(claim=cm, claim_ci=ch, calibration_gap=cm - rm)
        results[arm] = entry
        dump()

    print("\n== Experiment 4 ==")
    for arm, e in results.items():
        line = (f"  {arm}: completion={100*e['completion']:.2f}%"
                f"+-{100*e['completion_ci']:.2f}  steps={e['steps']:.2f}  "
                f"malformed={e['malformed']:.3f}/ep  calls/ep={e['llm_calls_per_episode']:.2f}  "
                f"latency={e['mean_call_latency_s']:.2f}s/call")
        if "claim" in e:
            line += (f"  claim={100*e['claim']:.2f}%  "
                     f"gap={100*e['calibration_gap']:+.2f}pt")
        print(line)
    dump()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
