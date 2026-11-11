#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 8 (pilot): SH-GD/SM-L on the external ToolMaze benchmark [25].

ToolMaze (Zhu et al., arXiv:2606.05806) runs a function-calling agent in a
sandbox that injects tool perturbations and scores the trace with a rule-based
judge (no LLM judge). Its tools need natural-language argument filling, so
SH-GD/SM-L cannot act alone; we evaluate the division of roles proposed in
Sec. VI-E of the manuscript:

  llm      ToolMaze's own OpenAI-compatible function-calling agent, unchanged
           (the model decides everything, including recovery).
  hybrid   the same agent, but every tool result is checked by the state
           machine: an explicit error or an output that violates the tool's
           declared output schema marks the tool as failed (set F); the
           replacement is chosen by str(e, s) = p_hat_e * rel_LCB(sig(e) -> sig(f))
           over the task's remaining tools (Eq. 6, 8, 16, 19), and the next
           model turn is restricted to that tool, the model only fills in its
           arguments. If no candidate clears the pruning constant, the model
           is told that no alternative is known and decides itself.

Schema identity: the signature of a tool's declared output schema (field names
and types, from tools/definitions/*.yaml), so the learned relation is between
schemas, not tool names.

Prior knowledge: rel is learned before evaluation from execution logs of the
C1/C2 tasks (disjoint from the C3/C4 test tasks): wherever two valid paths of
a task differ by one alternative tool for the same sub-task, the logged
successful execution of each is a pass observation for the other's schema.
During evaluation rel and p_hat keep learning online from the hybrid agent's
own substitutions (pass = the substitute returned a schema-valid, non-error
output). The model under test and its decoding settings are identical in
both arms.

The ToolMaze code is used read-only from ../IEEE_Access/external (not
redistributed; the repository carries no licence).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
TM_ROOT = os.path.join(HERE, "..", "IEEE_Access", "external", "ToolMaze_code", "ToolMaze-main")
TM_DATA = os.path.join(HERE, "..", "IEEE_Access", "external", "ToolMaze_data")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(TM_ROOT))

import yaml  # noqa: E402
import shgdsm as M  # noqa: E402
from evaluation.agents.openai_agent import OpenAIAgent  # noqa: E402
from evaluation.core.sandbox import ExecutionEngine  # noqa: E402
from evaluation.core.judge import JudgeSystem  # noqa: E402

JTYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
          "array": list, "object": dict}


# ---------------------------------------------------------------------------
# Tool catalogue: output-schema signatures
# ---------------------------------------------------------------------------

def load_catalogue() -> Dict[str, dict]:
    cat: Dict[str, dict] = {}
    for f in glob.glob(os.path.join(TM_ROOT, "tools", "definitions", "*.yaml")):
        for t in (yaml.safe_load(open(f, encoding="utf-8")) or {}).get("tools", []) or []:
            out = ((t.get("paradigms") or {}).get("mcp") or {}).get("output_schema") or {}
            cat[t["name"]] = dict(output_schema=out,
                                  sig="{" + ",".join(f"{k}:{(v or {}).get('type','any')}"
                                                     for k, v in sorted(out.items())) + "}")
    return cat


CAT: Dict[str, dict] = {}


def sig(tool: str) -> str:
    return CAT.get(tool, {}).get("sig", "tool:" + tool)


def output_ok(tool: str, result: Any) -> bool:
    """Schema validation of a tool result against its declared output schema:
    no error field, and every declared field present with the declared type."""
    if not isinstance(result, dict) or "error" in result or result.get("status") == "error":
        return False
    schema = CAT.get(tool, {}).get("output_schema") or {}
    for k, v in schema.items():
        if k not in result:
            return False
        ty = JTYPES.get((v or {}).get("type"))
        if ty and result[k] is not None and not isinstance(result[k], ty):
            return False
    return True


# ---------------------------------------------------------------------------
# Belief over schema pairs (reuses shgdsm.RelBelief: Beta posterior, LCB)
# ---------------------------------------------------------------------------

def make_belief() -> M.RelBelief:
    return M.RelBelief(M.cfg_lcb())


def pretrain_from_logs(belief: M.RelBelief, categories=("c1", "c2")) -> dict:
    n_pairs = n_exec = 0
    for c in categories:
        for f in sorted(glob.glob(os.path.join(TM_DATA, "perturbed_tasks", c, "*_P0.json"))):
            d = json.load(open(f, encoding="utf-8"))
            paths = d.get("valid_paths") or []
            for p in paths:
                for s in p.get("execution_trace", []):
                    if s.get("status") == "success":
                        belief.observe_exec(s["tool_name"], True)
                        n_exec += 1
        # alternative branches of the same DAG feed the same downstream node, and
        # both were executed successfully in the logged traces: each is a pass
        # observation of substituting for the other (aligned by position)
        for f in sorted(glob.glob(os.path.join(TM_DATA, "templates", c, "*.json"))):
            d = json.load(open(f, encoding="utf-8"))
            node_tool = {n["id"]: n["tool_name"] for n in d.get("dag", {}).get("nodes", [])}
            for g in d.get("alternative_groups", []) or []:
                seqs = [[node_tool.get(n) for n in p.get("node_ids", [])] for p in g.get("paths", [])]
                for i in range(len(seqs)):
                    for j in range(len(seqs)):
                        if i == j:
                            continue
                        for a, b in zip(seqs[i], seqs[j]):
                            if a and b and a != b:
                                belief.observe(sig(a), sig(b), True)
                                n_pairs += 1
    return dict(pass_observations=n_pairs, exec_observations=n_exec)


def lcb(belief: M.RelBelief, x: str, s: str) -> float:
    return 1.0 if x == s else belief.lcb(x, s)


def exec_rel(belief: M.RelBelief, tool: str) -> float:
    a0, b0 = 0.9 * belief.cfg.exec_prior_n, 0.1 * belief.cfg.exec_prior_n
    a, b = belief.ea.get(tool, 0), belief.eb.get(tool, 0)
    return (a + a0) / (a + b + a0 + b0)


# ---------------------------------------------------------------------------
# Claude backend (revision, E-L)
# ---------------------------------------------------------------------------
# ToolMaze's OpenAIAgent keeps its history in chat-completions form; the
# agent logic (system prompts, the hybrid state machine) is left unchanged and
# only the transport is replaced: each request is translated to the Anthropic
# Messages API (official SDK) and the reply back. Claude's own content blocks
# (thinking included) are kept on the assistant turn and replayed unchanged.

PRICES = {"claude-sonnet-5-5": (2.0, 10.0, 0.10, 2.5),
          "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0)}


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _ClaudeMsg:
    def __init__(self, text, tool_calls, blocks):
        self.content = text
        self.tool_calls = tool_calls or None
        self.reasoning_content = None
        self._blocks = blocks

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant", "content": self.content, "_anthropic_content": self._blocks}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


class ClaudeShim:
    """Exposes .chat.completions.create(**openai_kwargs) backed by Claude."""

    def __init__(self, model, effort="medium"):
        import anthropic
        from llm_experiment import ensure_anthropic_key
        ensure_anthropic_key()
        self._c = anthropic.Anthropic(max_retries=8)
        self.model, self.effort = model, effort
        self.usage = dict(input=0, output=0, cache_read=0, cache_write=0, refusals=0)
        self.chat = _Obj(completions=_Obj(create=self.create))

    @staticmethod
    def _convert(messages):
        system, out = [], []
        for m in messages:
            r = m.get("role")
            if r == "system":
                system.append(m.get("content") or "")
            elif r == "user":
                out.append({"role": "user", "content": m.get("content") or ""})
            elif r == "assistant":
                if m.get("_anthropic_content") is not None:
                    out.append({"role": "assistant", "content": m["_anthropic_content"]})
                    continue
                blocks = []
                if (m.get("content") or "").strip():
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls") or []:
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"],
                                   "input": json.loads(tc["function"].get("arguments") or "{}")})
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "."}]})
            elif r == "tool":
                out.append({"role": "user", "content": [{"type": "tool_result",
                                                          "tool_use_id": m["tool_call_id"],
                                                          "content": m.get("content") or ""}]})
        return "\n\n".join(system), out

    def create(self, model=None, messages=None, tools=None, max_tokens=None, **_ignored):
        system, msgs = self._convert(messages or [])
        ctools = [{"name": t["function"]["name"],
                   "description": t["function"].get("description", ""),
                   "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}}
                  for t in (tools or [])]
        kw = dict(model=self.model, max_tokens=16000, system=system, messages=msgs,
                  output_config={"effort": self.effort},
                  cache_control={"type": "ephemeral"},
                  betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        if ctools:
            kw.update(tools=ctools, tool_choice={"type": "auto", "disable_parallel_tool_use": True})
        resp = self._c.beta.messages.create(**kw)
        u = resp.usage
        self.usage["input"] += int(u.input_tokens)
        self.usage["output"] += int(u.output_tokens)
        self.usage["cache_read"] += int(getattr(u, "cache_read_input_tokens", 0) or 0)
        self.usage["cache_write"] += int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        if resp.stop_reason == "refusal":
            self.usage["refusals"] += 1
        blocks = [b.model_dump(exclude_none=True) for b in resp.content]
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [{"id": b.id, "type": "function",
                  "function": {"name": b.name, "arguments": json.dumps(b.input, ensure_ascii=False)}}
                 for b in resp.content if b.type == "tool_use"]
        msg = _ClaudeMsg(text, calls, blocks)
        return _Obj(choices=[_Obj(message=msg)],
                    usage=_Obj(prompt_tokens=int(u.input_tokens), completion_tokens=int(u.output_tokens)))

    def cost(self):
        p = PRICES.get(self.model, (0, 0, 0, 0))
        u = self.usage
        return (u["input"] * p[0] + u["output"] * p[1] + u["cache_read"] * p[2]
                + u["cache_write"] * p[3]) / 1e6


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class LocalAgent(OpenAIAgent):
    """ToolMaze's OpenAI agent against llama.cpp, reasoning switched off, call
    latency recorded. Behaviour otherwise unchanged."""

    def __init__(self, *a, no_think: bool = True, claude=None, **kw):
        super().__init__(*a, **kw)
        self.calls = 0
        self.latency = 0.0
        self.claude = claude
        if claude is not None:
            self.client = claude
            no_think = False
        create = self.client.chat.completions.create

        def wrapped(**kwargs):
            if no_think:
                kwargs.setdefault("extra_body", {})["chat_template_kwargs"] = {"enable_thinking": False}
            kwargs["timeout"] = 600.0
            t0 = time.perf_counter()
            try:
                return create(**kwargs)
            finally:
                self.latency += time.perf_counter() - t0
                self.calls += 1
        self.client.chat.completions.create = wrapped


class HybridAgent(LocalAgent):
    """LLM fills arguments; the SH-GD/SM-L state machine decides recovery."""

    def __init__(self, belief: M.RelBelief, *a, theta: float = 0.01, handover: bool = False, **kw):
        super().__init__(*a, **kw)
        self.handover = handover
        self.belief = belief
        self.theta = theta
        self.failed: set = set()
        self.forced: Optional[str] = None
        self.pending: Optional[Tuple[str, str]] = None   # (substitute, failed tool)
        self.interventions: List[dict] = []
        self._all_tools: List[dict] = []

    def initialize(self, task_description, tool_definitions):
        super().initialize(task_description, tool_definitions)
        self._all_tools = list(self.tools)
        self.failed, self.forced, self.pending, self.interventions = set(), None, None, []

    def step(self, user_message=None):
        if self.forced:
            self.tools = [t for t in self._all_tools if t["function"]["name"] == self.forced] or self._all_tools
        else:
            self.tools = self._all_tools
        try:
            return super().step(user_message)
        finally:
            self.tools = self._all_tools
            self.forced = None

    def _choose(self, f: str) -> Tuple[Optional[str], List[Tuple[str, float]]]:
        s = sig(f)
        ranked = []
        for t in self._all_tools:
            name = t["function"]["name"]
            if name == f or name in self.failed:
                continue
            v = exec_rel(self.belief, name) * lcb(self.belief, sig(name), s)       # Eq. 6
            if v >= self.theta:                                                     # Eq. 8 / 16
                ranked.append((name, v))
        ranked.sort(key=lambda r: -r[1])
        return (ranked[0][0] if ranked else None), ranked[:5]

    def receive_tool_result(self, tool_name, result, tool_call_index=0):
        super().receive_tool_result(tool_name, result, tool_call_index)
        ok = output_ok(tool_name, result)
        is_exec_error = isinstance(result, dict) and "error" in result
        if is_exec_error:
            self.belief.observe_exec(tool_name, False)
        elif ok:
            self.belief.observe_exec(tool_name, True)
        if self.pending and self.pending[0] == tool_name:       # outcome of our substitution
            self.belief.observe(sig(tool_name), sig(self.pending[1]), ok)
            self.pending = None
        if ok:
            return
        self.failed.add(tool_name)
        alt, ranked = self._choose(tool_name)
        rec = dict(failed=tool_name, chosen=alt, ranked=ranked)
        self.interventions.append(rec)
        if alt:
            self.forced, self.pending = alt, (alt, tool_name)
            note = (f"[state monitor] {tool_name} failed and is treated as unavailable for the rest "
                    f"of this task. Call {alt} instead to obtain the equivalent information; fill in "
                    f"its arguments from the request and earlier results.")
        elif self.handover:
            # M-H: no learned candidate -> the decision is handed to the model
            # without steering it (the outcome is still learned from)
            note = (f"[state monitor] {tool_name} failed. No learned substitute is available; "
                    f"decide yourself how to obtain the information it should have provided.")
        else:
            note = (f"[state monitor] {tool_name} failed and is treated as unavailable for the rest "
                    f"of this task. No alternative tool with evidence of substituting for it is known.")
        self.messages.append({"role": "user", "content": note})


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(task: dict, arm: str, belief: Optional[M.RelBelief], args) -> dict:
    kw = dict(model=args.model, api_key="none", base_url=args.base_url,
              temperature=args.temperature, max_tokens=1024)
    if args.backend == "anthropic":
        kw["claude"] = ClaudeShim(args.model, args.effort)
    agent = (HybridAgent(belief, handover=(arm == "hybrid-h"), **kw) if arm.startswith("hybrid")
             else LocalAgent(**kw))
    t0 = time.time()
    engine = ExecutionEngine(task, agent, tools_dir=os.path.join(TM_ROOT, "tools"))
    logger, tokens = engine.run(max_rounds=args.max_rounds)
    inference = logger.to_dict(token_usage=tokens)
    judge = JudgeSystem(llm_client=None, template_dir=os.path.join(TM_DATA, "templates"))
    j = judge.judge(task, inference)
    rounds = [m for m in inference.get("messages", []) if m.get("role") == "tool"]
    hit = any((m.get("metadata") or {}).get("perturbation_status") == "perturbed"
              for m in inference.get("messages", []))
    return dict(task_id=task["task_id"], mode=task.get("perturbation_mode"), arm=arm,
                passed=bool(j.get("pass")), hit=hit, tool_calls=len(rounds),
                llm_calls=agent.calls, latency_per_call=agent.latency / max(1, agent.calls),
                seconds=time.time() - t0, tokens=tokens,
                interventions=getattr(agent, "interventions", None),
                cost_usd=(agent.claude.cost() if getattr(agent, "claude", None) else 0.0),
                claude_usage=(agent.claude.usage if getattr(agent, "claude", None) else None),
                failure_reason=j.get("failure_reason"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="c3,c4")
    ap.add_argument("--modes", default="P2")
    ap.add_argument("--tasks-per-cell", type=int, default=12)
    ap.add_argument("--arms", default="llm,hybrid")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="NVIDIA-Nemotron-Nano-9B-v2-Japanese-Q8_0")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--backend", choices=["llama", "anthropic"], default="llama")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-rounds", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results_toolmaze_pilot.json")
    args = ap.parse_args()
    CAT.update(load_catalogue())
    belief = make_belief()
    pre = pretrain_from_logs(belief)
    print("catalogue tools:", len(CAT), "| pretraining:", pre, flush=True)
    rng = random.Random(args.seed)
    tasks = []
    for c in args.categories.split(","):
        for m in args.modes.split(","):
            fs = sorted(glob.glob(os.path.join(TM_DATA, "perturbed_tasks", c, f"*_{m}.json")))
            rng.shuffle(fs)
            tasks += [json.load(open(f, encoding="utf-8")) for f in fs[: args.tasks_per_cell]]
    rows = []
    if os.path.exists(args.out):
        rows = json.load(open(args.out, encoding="utf-8")).get("runs", [])
    done = {(r["task_id"], r["mode"], r["arm"]) for r in rows}
    for t in tasks:
        for arm in args.arms.split(","):
            if (t["task_id"], t.get("perturbation_mode"), arm) in done:
                continue
            try:
                r = run_one(t, arm, belief, args)
            except Exception as e:  # keep going; record the failure
                r = dict(task_id=t["task_id"], mode=t.get("perturbation_mode"), arm=arm, passed=False,
                         error=repr(e))
            rows.append(r)
            print(json.dumps({k: r.get(k) for k in ("task_id", "mode", "arm", "passed", "hit", "tool_calls",
                                                   "llm_calls", "seconds", "cost_usd", "error")}), flush=True)
            summary = {}
            for arm_ in args.arms.split(","):
                rs = [x for x in rows if x["arm"] == arm_ and "error" not in x]
                if rs:
                    hits = [x for x in rs if x.get("hit")]
                    summary[arm_] = dict(n=len(rs), tsr=statistics.fmean(x["passed"] for x in rs),
                                         prr=(statistics.fmean(x["passed"] for x in hits) if hits else None),
                                         llm_calls=statistics.fmean(x["llm_calls"] for x in rs),
                                         tool_calls=statistics.fmean(x["tool_calls"] for x in rs),
                                         latency=statistics.fmean(x["latency_per_call"] for x in rs),
                                         cost_usd=sum(x.get("cost_usd", 0.0) for x in rs))
            json.dump(dict(config=vars(args), pretraining=pre, summary=summary, runs=rows),
                      open(args.out, "w", encoding="utf-8"), indent=2, default=str)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
