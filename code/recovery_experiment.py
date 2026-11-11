#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 6: recovery efficiency under injected disruptions.

Question
--------
When a tool fails or the environment deviates from what the plan assumed,
how fast -- in tool executions, in failed executions, in model calls and in
wall-clock time -- does each agent find an alternative path and restore the
broken lineage?  This is the situation the SH-GD/SM-L state machine is built
for, and the one where it should differ from a *model-delegated* agent, i.e.
one that hands the error back to the LLM and lets the LLM pick the next tool
(Claude Agent SDK [27]; Google ADK Reflect-and-Retry [28]).

Environment
-----------
The Experiment 4 environment: twelve tools, real JSON Schema validation with
`jsonschema`, stub services whose instance-level variability induces rho*.
On top of it, FaultEnv (llm_experiment.py) injects six disruption scenarios
that are unknown to every agent at the start of the episode.

Arms
----
  C    SH-GD/SM-L   : learned rel (1,600 Thompson training episodes in the
                      undisrupted environment), posterior lower confidence
                      bound (Eq.19, --estimator lcb, default) or posterior mean
                      (Eq.4, --estimator mean), Eq.15/Eq.16.
                      Recovery decisions are computed; no model is called.
  A    delegated    : decision-rule PROXY of a model-delegated agent (Claude
                      Agent SDK style: the error is returned to the model, the
                      model picks the next tool). One model call per decision.
  B    delegated+RR : A plus ADK Reflect-and-Retry: a failed tool is retried up
                      to 3 times, one reflection call per retry.
  A*   delegated,
       ideal        : A with the TRUE rho* in place of surface plausibility --
                      the best any model-delegated agent could do without
                      experience. Still one model call per decision.
  C+R  SH-GD/SM-L+RR: C plus the same retry rule (extension, reported for
                      the transient scenario and the i.i.d.-error variant).

What the proxy is, and is not
-----------------------------
A, B and A* are NOT language models. They are decision rules that stand in
for one, built to be *favourable* to the baseline:
  * they use exactly the same Eq.15 planner as SH-GD/SM-L, with the same
    declared success rates p_e and grades -- so they plan as well as the
    proposed model does, and differ ONLY in what they know about schema
    substitutability and in having no memory across episodes;
  * substitutability comes from the surface of the schema names, the way a
    model reads a tool list: a substitution x -> s is plausible iff x and s
    are at the same processing stage and either share a domain token or x is
    a generic source (market_raw). Plausible pairs get rel = c (default 0.5,
    "might work"), all others 0;
  * within an episode they learn perfectly from the validator's error
    message: a substitution that failed is set to 0, one that passed to 1;
  * decision noise eps (a random ready tool instead of the planned one) and
    the malformed-output rate are 0 in the main table and swept separately.
The real-LLM version of arms A and B is the --llm mode below (llama.cpp,
any OpenAI-compatible endpoint, or Claude); arm C is re-run there on the same
random streams so that the comparison is paired.

Metrics (per disrupted episode)
-------------------------------
  disruption  first failed execution of a tool the scenario targets
  recovery    the first later point at which every target schema of the
              scenario (the merge-point inputs the failed lineage feeds) is held
  steps       tool executions from the disruption (excl.) to recovery (incl.)
  wasted      failed executions in that window
  calls       model calls in that window (C: 0 by construction)
  time        steps * tau_tool + calls * tau_llm + measured decision compute
Plus the completion rate over all episodes.

Author: (revision support)
License: MIT
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

sys.path.insert(0, ".")
import shgdsm as M
import llm_experiment as L

# ---------------------------------------------------------------------------
# Disruption scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    outage: FrozenSet[str] = frozenset()
    drift: FrozenSet[str] = frozenset()
    transient: Tuple[Tuple[str, int], ...] = ()
    targets: FrozenSet[str] = frozenset()

    def env(self, persistent: bool) -> L.FaultEnv:
        return L.FaultEnv(frozenset(self.outage), frozenset(self.drift),
                          dict(self.transient), persistent)


SCENARIOS: Tuple[Scenario, ...] = (
    Scenario("S1", "FetchEcon permanent outage (no same-type alternative)",
             outage=frozenset({"FetchEcon"}), targets=frozenset({"econ_clean"})),
    Scenario("S2", "FetchStockA permanent outage (same-type alternative exists)",
             outage=frozenset({"FetchStockA"}), targets=frozenset({"stock_clean"})),
    Scenario("S3", "stock vendor API change: FetchStockA/B output drifts",
             drift=frozenset({"FetchStockA", "FetchStockB"}),
             targets=frozenset({"stock_clean"})),
    Scenario("S4", "SummarizeNews permanent outage (only a look-alike remains)",
             outage=frozenset({"SummarizeNews"}), targets=frozenset({"news_summary"})),
    Scenario("S5", "FetchEcon transient outage (first 2 calls fail)",
             transient=(("FetchEcon", 2),), targets=frozenset({"econ_clean"})),
    Scenario("S6", "compound: FetchEcon outage + stock vendor drift",
             outage=frozenset({"FetchEcon"}),
             drift=frozenset({"FetchStockA", "FetchStockB"}),
             targets=frozenset({"econ_clean", "stock_clean"})),
)
SCN = {s.key: s for s in SCENARIOS}
SCN_INDEX = {s.key: i for i, s in enumerate(SCENARIOS)}

# ---------------------------------------------------------------------------
# Surface plausibility: what a model can read off the tool list
# ---------------------------------------------------------------------------

STAGES = (
    {"stock_raw", "econ_raw", "market_raw", "news_raw"},
    {"stock_clean", "econ_clean", "news_summary", "news_digest"},
)
GENERIC = {"market_raw"}


def surface_plausible(x: str, s: str, mode: str = "token") -> bool:
    """mode='token': same stage and (shared domain token or generic source).
    mode='stage': same stage only -- a careless reading of the tool list."""
    if x == s:
        return True
    if not any(x in st and s in st for st in STAGES):
        return False
    if mode == "stage":
        return True
    shared = set(x.split("_")[:-1]) & set(s.split("_")[:-1])
    return bool(shared) or x in GENERIC


def surface_rel_table(c: float, mode: str = "token") -> Dict[Tuple[str, str], float]:
    d = {p: (c if surface_plausible(*p, mode) else 0.0) for p in M.LEARNABLE_PAIRS}
    for s in M.SCHEMAS:
        d[(s, s)] = 1.0
    return d


# ---------------------------------------------------------------------------
# Episode runners. Every runner returns an EpisodeResult with a trace:
#   (tool, ok, error, schemas held afterwards, model calls so far)
# ---------------------------------------------------------------------------

MAX_DECISIONS = 40


def _tool_side(err: str) -> bool:
    """A failure a retry can plausibly fix (not a wrong input argument)."""
    return err.startswith("ExecutionError") or err.startswith("SchemaMismatch(output)")


def episode_shgdsm(belief: M.RelBelief, cfg: M.Config, rng: random.Random,
                   env: L.FaultEnv, retry: int = 0, learn: bool = True
                   ) -> Tuple[L.EpisodeResult, float, int]:
    """Arm C (and C+R). Returns (result, decision seconds, decisions)."""
    t0 = time.perf_counter()
    pol = M.MyopicPolicy(belief.snapshot_mean(), cfg)
    dec_s = time.perf_counter() - t0
    n_dec = 0
    store = {"instruction": L.instruction_payload()}
    st = M.State()
    trace = []
    steps = 0
    for _ in range(MAX_DECISIONS):
        if "final_report" in store:
            break
        t1 = time.perf_counter()
        i = pol.act(st)
        if i is not None:
            t = M.TASKS[i]
            sigma = st.sigma()
            sub = {s: (pol.est.best_substitution(s, sigma)[0] or s) for s in t.inputs}
        dec_s += time.perf_counter() - t1
        n_dec += 1
        if i is None:
            break
        ok, obs, err = L._apply(t.name, sub, store, rng, env)
        steps += 1
        trace.append((t.name, ok, err, frozenset(store), 0))
        tries = 0
        while not ok and tries < retry and _tool_side(err):
            tries += 1
            ok, obs, err = L._apply(t.name, sub, store, rng, env)
            steps += 1
            trace.append((t.name, ok, err, frozenset(store), 0))
        if learn:
            for x, s, passed in obs:
                belief.observe(x, s, passed)
            eo = L.exec_outcome(ok, err)
            if eo is not None:
                belief.observe_exec(t.name, eo)
        st = st.with_success(i) if ok else st.with_failure(i)
    return (L.EpisodeResult("final_report" in store, steps, 0, 0.0, [], trace),
            dec_s, n_dec)


def episode_delegated(rel0: Dict[Tuple[str, str], float], cfg: M.Config,
                      rng: random.Random, arng: random.Random, env: L.FaultEnv,
                      retry: int = 0, eps: float = 0.0, malformed: float = 0.0
                      ) -> L.EpisodeResult:
    """Arms A, B, A*: proxy of a model-delegated agent (see module docstring)."""
    rel = dict(rel0)                     # no memory across episodes
    pol = M.MyopicPolicy(rel, cfg)
    store = {"instruction": L.instruction_payload()}
    st = M.State()
    trace = []
    calls = steps = n_malformed = 0
    for _ in range(MAX_DECISIONS):
        if "final_report" in store:
            break
        calls += 1                        # the model is asked for the next action
        while malformed and arng.random() < malformed:
            calls += 1                    # unparseable reply -> ask again
            n_malformed += 1
        i = pol.act(st)
        if eps and i is not None and arng.random() < eps:
            ready = pol.est.ready(st)
            i = arng.choice(ready)
        if i is None:
            break                         # the model declares it cannot proceed
        t = M.TASKS[i]
        sigma = st.sigma()
        sub = {s: (pol.est.best_substitution(s, sigma)[0] or s) for s in t.inputs}
        ok, obs, err = L._apply(t.name, sub, store, rng, env)
        steps += 1
        trace.append((t.name, ok, err, frozenset(store), calls))
        tries = 0
        while not ok and tries < retry and _tool_side(err):
            tries += 1
            calls += 1                    # reflection before the retry
            ok, obs, err = L._apply(t.name, sub, store, rng, env)
            steps += 1
            trace.append((t.name, ok, err, frozenset(store), calls))
        changed = False
        for x, s, passed in obs:          # in-context learning from the error text
            v = 1.0 if passed else 0.0
            if rel.get((x, s)) != v:
                rel[(x, s)] = v
                changed = True
        if changed:
            pol = M.MyopicPolicy(rel, cfg)
        st = st.with_success(i) if ok else st.with_failure(i)
    return L.EpisodeResult("final_report" in store, steps, n_malformed, 0.0, [], trace)


# ---------------------------------------------------------------------------
# Recovery metrics
# ---------------------------------------------------------------------------


def recovery_of(trace, affected: FrozenSet[str], targets: FrozenSet[str]) -> Optional[dict]:
    d = next((k for k, r in enumerate(trace) if r[0] in affected and not r[1]), None)
    if d is None:
        return None
    rec = next((j for j in range(d + 1, len(trace)) if targets <= trace[j][3]), None)
    end = rec if rec is not None else len(trace) - 1
    window = trace[d + 1:end + 1]
    return dict(recovered=rec is not None,
                steps=(rec - d) if rec is not None else None,
                wasted=sum(1 for r in window if not r[1]),
                calls=trace[end][4] - trace[d][4])


@dataclass
class Acc:
    episodes: int = 0
    completed: int = 0
    steps_total: int = 0
    calls_total: int = 0
    disrupted: int = 0
    recovered: int = 0
    completed_given_disrupted: int = 0
    rec_steps: List[int] = field(default_factory=list)
    rec_wasted: List[int] = field(default_factory=list)
    rec_calls: List[int] = field(default_factory=list)
    unrec_wasted: List[int] = field(default_factory=list)
    unrec_calls: List[int] = field(default_factory=list)
    dec_seconds: float = 0.0
    decisions: int = 0

    def add(self, res: L.EpisodeResult, scn: Scenario, persistent: bool) -> None:
        self.episodes += 1
        self.completed += int(res.completed)
        self.steps_total += res.steps
        self.calls_total += res.trace[-1][4] if res.trace else 0
        m = recovery_of(res.trace, scn.env(persistent).affected(), scn.targets)
        if m is None:
            return
        self.disrupted += 1
        self.completed_given_disrupted += int(res.completed)
        if m["recovered"]:
            self.recovered += 1
            self.rec_steps.append(m["steps"])
            self.rec_wasted.append(m["wasted"])
            self.rec_calls.append(m["calls"])
        else:
            self.unrec_wasted.append(m["wasted"])
            self.unrec_calls.append(m["calls"])

    def summary(self) -> dict:
        f = lambda xs: statistics.fmean(xs) if xs else float("nan")
        return dict(
            completion=self.completed / max(1, self.episodes),
            disrupted_share=self.disrupted / max(1, self.episodes),
            recovery_rate=self.recovered / max(1, self.disrupted),
            completion_given_disrupted=self.completed_given_disrupted / max(1, self.disrupted),
            steps_to_recover=f(self.rec_steps),
            wasted_to_recover=f(self.rec_wasted),
            calls_to_recover=f(self.rec_calls),
            wasted_if_unrecovered=f(self.unrec_wasted),
            calls_if_unrecovered=f(self.unrec_calls),
            steps_per_episode=self.steps_total / max(1, self.episodes),
            calls_per_episode=self.calls_total / max(1, self.episodes),
            decision_ms=1000.0 * self.dec_seconds / max(1, self.decisions),
        )


# ---------------------------------------------------------------------------
# One (seed, scenario, persistence) cell -- all arms, common random numbers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    seed: int
    scenario: str
    persistent: bool
    episodes: int
    train: int
    arms: Tuple[str, ...]
    c_surface: float = 0.5
    eps: float = 0.0
    malformed: float = 0.0
    surface_mode: str = "token"
    estimator: str = "lcb"


_BELIEFS: Dict[int, Tuple[dict, dict]] = {}


def cfg_for(estimator: str) -> M.Config:
    return (M.Config(point_estimate="lcb", theta=0.01) if estimator == "lcb"
            else M.Config())


def train_seed(args: Tuple) -> Tuple[int, Tuple[dict, dict]]:
    """Thompson training in the UNDISRUPTED environment (same as Experiment 4).
    Training is identical for both estimators (exploration always samples the
    posterior, Eq.17); only the planning-time point estimate differs."""
    seed, n = args[0], args[1]
    cfg = cfg_for(args[2] if len(args) > 2 else "lcb")
    b = M.RelBelief(cfg)
    trng = random.Random(20_000 + seed)
    for _ in range(n):
        L.episode_shgdsm(b, cfg, trng, explore=True)
    return seed, (b.a, b.b)


def _init_worker(beliefs: Dict[int, Tuple[dict, dict]]) -> None:
    _BELIEFS.update(beliefs)


def trained_belief(cfg: M.Config, seed: int, n: int) -> M.RelBelief:
    """A private copy: C keeps learning online during evaluation."""
    if seed not in _BELIEFS:
        _BELIEFS[seed] = train_seed((seed, n, cfg.point_estimate))[1]
    a, b = _BELIEFS[seed]
    out = M.RelBelief(cfg)
    out.a, out.b = dict(a), dict(b)
    return out


def run_job(job: Job) -> Tuple[Job, Dict[str, dict]]:
    cfg = cfg_for(job.estimator)
    scn = SCN[job.scenario]
    out: Dict[str, dict] = {}
    for arm in job.arms:
        acc = Acc()
        rng = random.Random(10_000 + 100 * job.seed + SCN_INDEX[job.scenario])
        arng = random.Random(30_000 + job.seed)
        if arm in ("C", "C+R"):
            belief = trained_belief(cfg, job.seed, job.train)
        rel_s = surface_rel_table(job.c_surface, job.surface_mode)
        rel_o = M.oracle_rel_table()
        for _ in range(job.episodes):
            env = scn.env(job.persistent).fresh()
            if arm == "C":
                r, ds, nd = episode_shgdsm(belief, cfg, rng, env)
                acc.dec_seconds += ds; acc.decisions += nd
            elif arm == "C+R":
                r, ds, nd = episode_shgdsm(belief, cfg, rng, env, retry=3)
                acc.dec_seconds += ds; acc.decisions += nd
            elif arm == "A":
                r = episode_delegated(rel_s, cfg, rng, arng, env, 0, job.eps, job.malformed)
            elif arm == "B":
                r = episode_delegated(rel_s, cfg, rng, arng, env, 3, job.eps, job.malformed)
            elif arm == "A*":
                r = episode_delegated(rel_o, cfg, rng, arng, env, 0, job.eps, job.malformed)
            else:
                raise ValueError(arm)
            acc.add(r, scn, job.persistent)
        out[arm] = acc.summary()
    return job, out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def ci95(xs: Sequence[float]) -> Tuple[float, float]:
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan"), float("nan")
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


def aggregate(cells: List[Tuple[Job, Dict[str, dict]]]) -> Dict[str, Dict[str, dict]]:
    by: Dict[Tuple[str, str], List[dict]] = {}
    for job, res in cells:
        for arm, s in res.items():
            by.setdefault((job.scenario, arm), []).append(s)
    out: Dict[str, Dict[str, dict]] = {}
    for (scn, arm), rows in by.items():
        agg = {}
        for k in rows[0]:
            m, h = ci95([r[k] for r in rows])
            agg[k] = m
            agg[k + "_ci"] = h
        agg["seeds"] = len(rows)
        out.setdefault(scn, {})[arm] = agg
    return out


def time_to_recover(row: dict, tau_tool: float, tau_llm: float) -> float:
    return (row["steps_to_recover"] * tau_tool + row["calls_to_recover"] * tau_llm
            + row["steps_to_recover"] * row["decision_ms"] / 1000.0)


# ---------------------------------------------------------------------------
# Real-LLM mode (arms A / B with an actual model); not run in the manuscript
# ---------------------------------------------------------------------------


HANDOVER_U = 0.02      # hand over when the best utility (Eq.15) is below this
MAX_HANDOVERS = 6


def episode_handover(belief: M.RelBelief, cfg: M.Config, rng: random.Random,
                     env: L.FaultEnv, client, learn: bool = True
                     ) -> Tuple[L.EpisodeResult, int]:
    """M-H (revision): arm C, but when no candidate clears the pruning
    constant or the best utility is below HANDOVER_U, ONE decision is handed
    to the LLM (same prompt as arms A/B, with the learned hint). The executed
    result, whoever chose it, is fed back into the posteriors.
    Returns (result, number of hand-overs)."""
    pol = M.MyopicPolicy(belief.snapshot_mean(), cfg)
    store = {"instruction": L.instruction_payload()}
    st = M.State()
    trace, steps, handovers, malformed = [], 0, 0, 0
    calls0 = client.calls
    done_names, failed_names, last_error = [], [], ""
    for _ in range(MAX_DECISIONS):
        if "final_report" in store:
            break
        rows = pol.scores(st)
        i = rows[0][0] if rows else None
        if i is not None and rows[0][4] >= HANDOVER_U:
            t = M.TASKS[i]
            sigma = st.sigma()
            name = t.name
            sub = {s_: (pol.est.best_substitution(s_, sigma)[0] or s_) for s_ in t.inputs}
        else:
            if handovers >= MAX_HANDOVERS:
                break
            handovers += 1
            extra = (L.msg("last_failure", err=last_error) if last_error else "")
            extra += L.msg("hint", hint=L.rel_hint_text(belief, cfg))
            prompt = L.system_prompt().format(tools=L.tools_block(), sigma=sorted(store),
                                              failed=failed_names or L.msg("none"),
                                              done=done_names or L.msg("none"), extra=extra)
            act = L.parse_action(client.chat([{"role": "user", "content": prompt}]))
            if act is None:
                malformed += 1
                last_error = L.msg("malformed")
                continue
            name, sub = act
            if name not in M.TASK_INDEX or (st.used() >> M.TASK_INDEX[name]) & 1:
                last_error = L.msg("already_failed", name=name)
                continue
            i = M.TASK_INDEX[name]
        ok, obs, err = L._apply(name, sub, store, rng, env)
        steps += 1
        trace.append((name, ok, err, frozenset(store), client.calls - calls0))
        if learn:
            for x, s_, passed in obs:
                belief.observe(x, s_, passed)
            eo = L.exec_outcome(ok, err)
            if eo is not None:
                belief.observe_exec(name, eo)
        if ok:
            done_names.append(name)
            last_error = ""
            st = st.with_success(i)
        else:
            failed_names.append(name)
            last_error = err
            st = st.with_failure(i)
    return (L.EpisodeResult("final_report" in store, steps, malformed, 0.0, [], trace), handovers)


def run_llm(args) -> None:
    """Arms A/B driven by a real model, with arm C run on the SAME random
    streams (common random numbers) so that the comparison is paired.
    The measured mean latency per model call is reported as tau_LLM, so Eq.20
    is evaluated with a measured rather than an assumed value."""
    cfg = _cfg(args)
    results: Dict[str, Dict[str, dict]] = {}
    lat: Dict[str, List[float]] = {}

    def dump() -> None:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(dict(mode="llm", backend=args.backend, model=args.model,
                           lang=args.lang, seeds=args.llm_seeds, episodes=args.episodes,
                           persistent=not args.iid, results=results,
                           mean_call_latency_s={k: (statistics.fmean(v) if v else None)
                                                for k, v in lat.items()}),
                      f, ensure_ascii=False, indent=2, allow_nan=True)

    for key in args.scenarios.split(","):
        scn = SCN[key]
        arms = [a_ for a_ in (("C", None), ("A", 0), ("B", 3), ("C+H", None))
                if a_[0] in args.llm_arms.split(",")]
        for arm, reflect in arms:
            cells = []
            for sd in range(args.llm_seed_start, args.llm_seed_start + args.llm_seeds):
                acc = Acc()
                rng = random.Random(10_000 + 100 * sd + SCN_INDEX[key])
                hand = 0
                if arm in ("C", "C+H"):
                    belief = trained_belief(cfg, sd, args.train_episodes)
                if arm != "C":
                    client = L.make_client(args.backend, args.endpoint, args.model,
                                           no_think=not args.think)
                for _ in range(args.episodes):
                    env = scn.env(not args.iid).fresh()
                    if arm == "C":
                        r, ds, nd = episode_shgdsm(belief, cfg, rng, env)
                        acc.dec_seconds += ds; acc.decisions += nd
                    elif arm == "C+H":
                        r, h = episode_handover(belief, cfg, rng, env, client)
                        hand += h
                    else:
                        r = L.episode_llm(client, rng, reflect=reflect, env=env)
                    acc.add(r, scn, not args.iid)
                if arm != "C" and client.calls:
                    lat.setdefault(f"{arm}", []).append(client.latency_s / client.calls)
                cells.append(acc.summary())
                cells[-1]["handovers_per_episode"] = hand / args.episodes
                if arm != "C":
                    cells[-1]["llm_calls_total"] = client.calls
                    cells[-1]["cost_usd"] = L.client_cost(client)
                print(key, arm, sd, json.dumps(cells[-1], ensure_ascii=False), flush=True)
            agg = {}
            for k in cells[0]:
                m, h = ci95([c[k] for c in cells])
                agg[k], agg[k + "_ci"] = m, h
            agg["cells"] = cells
            results.setdefault(key, {})[f"{arm}-llm" if arm not in ("C", "C+H") else arm] = agg
            dump()
    print("wrote", args.out)


def _cfg(args) -> M.Config:
    if getattr(args, "estimator", "lcb") == "lcb":
        return M.Config(point_estimate="lcb", theta=0.01)
    return M.Config()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MAIN_ARMS = ("C", "A", "B", "A*", "C+R")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--train-episodes", type=int, default=1600)
    ap.add_argument("--scenarios", default=",".join(s.key for s in SCENARIOS))
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--out", default="results_recovery.json")
    ap.add_argument("--skip-sensitivity", action="store_true")
    # real-LLM mode
    ap.add_argument("--llm", action="store_true",
                    help="run arms A/B with a real model instead of the proxy")
    ap.add_argument("--backend", choices=["llama", "openai", "anthropic"], default="llama")
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--lang", choices=["en", "ja"], default="en")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--llm-seeds", type=int, default=3)
    ap.add_argument("--llm-seed-start", type=int, default=1)
    ap.add_argument("--llm-arms", default="C,A,B",
                    help="subset of C,A,B,C+H (C+H: hand-over extension, M-H)")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--estimator", choices=["lcb", "mean"], default="lcb")
    ap.add_argument("--iid", action="store_true",
                    help="i.i.d. per-call execution errors instead of persistent")
    args = ap.parse_args()

    L.PROMPT_LANG = args.lang
    L.EFFORT = args.effort
    if args.endpoint is None:
        args.endpoint = ("https://api.openai.com/v1/chat/completions"
                         if args.backend == "openai" else L.DEFAULT_ENDPOINT)
    if args.llm:
        run_llm(args)
        return

    keys = args.scenarios.split(",")
    seeds = range(1, args.seeds + 1)
    jobs: List[Job] = []
    # main: persistent errors, all arms, favourable proxy (eps = 0)
    for sd in seeds:
        for k in keys:
            jobs.append(Job(sd, k, True, args.episodes, args.train_episodes, MAIN_ARMS,
                            estimator=args.estimator))
    groups = {"main": len(jobs)}
    if not args.skip_sensitivity:
        # i.i.d. per-call execution errors (retry now pays in general)
        for sd in seeds:
            for k in keys:
                jobs.append(Job(sd, k, False, args.episodes, args.train_episodes,
                                ("C", "A", "B", "C+R"), estimator=args.estimator))
        groups["iid"] = len(jobs)
        # decision noise and malformed output for the delegated proxy
        for eps, mal in ((0.1, 0.05), (0.2, 0.10)):
            for sd in seeds:
                for k in keys:
                    jobs.append(Job(sd, k, True, args.episodes, args.train_episodes,
                                    ("A",), eps=eps, malformed=mal))
            groups[f"noise{eps}"] = len(jobs)
        # surface plausibility level c
        for c in (0.3, 0.9):
            for sd in seeds:
                for k in keys:
                    jobs.append(Job(sd, k, True, args.episodes, args.train_episodes,
                                    ("A",), c_surface=c))
            groups[f"c{c}"] = len(jobs)
        for sd in seeds:
            for k in keys:
                jobs.append(Job(sd, k, True, args.episodes, args.train_episodes,
                                ("A",), surface_mode="stage"))

    t0 = time.time()
    print(f"{len(jobs)} cells on {args.workers} workers", flush=True)
    cells: List[Tuple[Job, Dict[str, dict]]] = []
    with mp.Pool(args.workers) as pool:
        beliefs = dict(pool.map(train_seed, [(sd, args.train_episodes, args.estimator)
                                             for sd in seeds]))
    print(f"trained {len(beliefs)} beliefs  {time.time() - t0:.0f}s", flush=True)
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(beliefs,)) as pool:
        for n, cell in enumerate(pool.imap_unordered(run_job, jobs, chunksize=1), 1):
            cells.append(cell)
            if n % 10 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    def pick(pred):
        return aggregate([c for c in cells if pred(c[0])])

    main_res = pick(lambda j: j.arms == MAIN_ARMS)
    payload = dict(
        config=dict(seeds=args.seeds, episodes=args.episodes, estimator=args.estimator,
                    train_episodes=args.train_episodes,
                    surface_pairs=sorted(f"{x}->{s}" for (x, s) in M.LEARNABLE_PAIRS
                                         if surface_plausible(x, s)),
                    scenarios={s.key: s.label for s in SCENARIOS}),
        main=main_res,
    )
    if not args.skip_sensitivity:
        payload["iid"] = pick(lambda j: not j.persistent)
        payload["noise"] = {
            f"eps={e},malformed={m}": pick(lambda j, e=e: j.eps == e)
            for e, m in ((0.1, 0.05), (0.2, 0.10))}
        payload["surface_c"] = {
            f"c={c}": pick(lambda j, c=c: j.c_surface == c and j.eps == 0 and j.arms == ("A",))
            for c in (0.3, 0.9)}
        payload["surface_stage"] = pick(lambda j: j.surface_mode == "stage")

    print("\n== Experiment 6: recovery (persistent errors, favourable proxy) ==")
    for k in keys:
        print(f"[{k}] {SCN[k].label}")
        for arm in MAIN_ARMS:
            r = main_res[k][arm]
            print(f"   {arm:<4} compl={100*r['completion']:6.2f}%  "
                  f"rec={100*r['recovery_rate']:6.2f}%  "
                  f"steps={r['steps_to_recover']:.2f}  wasted={r['wasted_to_recover']:.2f}  "
                  f"calls={r['calls_to_recover']:.2f}  "
                  f"calls/ep={r['calls_per_episode']:.2f}  dec={r['decision_ms']:.2f}ms  "
                  f"T(0.5,2)={time_to_recover(r, 0.5, 2.0):.2f}s")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=True)
    print(f"\nwrote {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
