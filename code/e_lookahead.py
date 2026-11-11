#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-H (revision 2026-11): one-step look-ahead (Eq.15) against k-step
expectimax and policy rollout, in universes that need long recovery chains.

Policies (all plan on the same learned rel / p_hat snapshot, LCB estimator)
  k=1       Eq.15 (the proposed rule, MyopicPolicy)
  k=2, k=3  expectimax over k real decisions, leaves valued by Eq.9-12
  rollout   for every ready task, N simulated continuations in the agent's own
            model with the k=1 rule as base policy (one-step policy improvement)

Universes
  detour-L  the dedicated fetcher of a required raw schema is unreliable
            (p=0.5); recovery needs an L-step conversion chain
            feed -> m1 -> ... -> mL ~ a_raw (rel 0.9), while a look-alike
            one-step alternative is attractive but weak (rel 0.3).
            L in {1, 2, 3, 4}; longest generation chain = L + 4.
  reference the hand-built universe of Experiments 1-6
  K=8       random universes (envgen), Monte Carlo
Depth cap D of the value recursion: the longest chain (adequate) and 3
(too short), so that the effect of the horizon is separated from that of
look-ahead.

    python e_lookahead.py --out results_rev_EH.json --resume
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
from typing import Dict, Optional, Tuple

sys.path.insert(0, ".")
import shgdsm as M
import envgen as E

SEEDS = list(range(1, 11))
POLICIES = ["k=1", "k=2", "k=3", "rollout"]


# ---------------------------------------------------------------------------
# universes
# ---------------------------------------------------------------------------

def detour_universe(L: int):
    T = M.Task
    m = [f"m{j}" for j in range(L + 1)]
    schemas = ["instruction", "a_raw", "a_clean", "a_alt", "b_raw", "b_clean",
               "report_draft", "final_report"] + m
    tasks = [T("FetchA", ("instruction",), ("a_raw",), 0.50, ("observed",)),
             T("FetchAltA", ("instruction",), ("a_alt",), 0.95, ("declared",)),
             T("CleanA", ("a_raw",), ("a_clean",), 0.95, ("observed",)),
             T("FetchB", ("instruction",), ("b_raw",), 0.90, ("observed",)),
             T("CleanB", ("b_raw",), ("b_clean",), 0.95, ("observed",)),
             T("Compose", ("a_clean", "b_clean"), ("report_draft",), 0.90, ("observed", "observed")),
             T("Finalize", ("report_draft",), ("final_report",), 0.95, ("observed",)),
             T("FetchFeed", ("instruction",), ("m0",), 0.90, ("observed",))]
    for j in range(1, L + 1):
        tasks.append(T(f"Convert{j}", (m[j - 1],), (m[j],), 0.90, ("observed",)))
    rho = {(m[L], "a_raw"): 0.90, ("a_alt", "a_raw"): 0.30}
    return schemas, tasks, rho, L + 4


def use_universe(name: str, u: int = 1) -> int:
    if name == "reference":
        M.use_reference_universe()
        return 4
    if name.startswith("detour-"):
        s, t, r, depth = detour_universe(int(name.split("-")[1]))
        M.set_universe(s, t, r)
        return depth
    if name == "K=8":
        U = E.generate(8, u)
        M.set_universe(U["schemas"], U["tasks"], U["rho"])
        return U["depth"]
    raise ValueError(name)


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------

class KStepPolicy(M.Policy):
    """Expectimax over k decisions; leaves valued by the Eq.9-12 estimate."""

    def __init__(self, rel, cfg: M.Config, k: int):
        self.est = M.Estimator(rel, cfg)
        self.cfg, self.k = cfg, k
        self._v: Dict[Tuple[int, int, int], float] = {}
        self._act: Dict[Tuple[int, int], Optional[int]] = {}

    def value(self, st: M.State, k: int) -> float:
        if st.reached_goal():
            return 1.0
        if k == 0:
            return self.est.p_goal(st)
        key = (st.done, st.failed, k)
        if key in self._v:
            return self._v[key]
        best = self.est.p_goal(st) if not self.est.ready(st) else -1e9
        for i in self.est.ready(st):
            q = self.est.p_exec(i, st)
            u = (q * self.value(st.with_success(i), k - 1)
                 + (1 - q) * self.value(st.with_failure(i), k - 1) - self.cfg.lam * M.TASKS[i].cost)
            best = max(best, u)
        self._v[key] = best
        return best

    def act(self, st: M.State) -> Optional[int]:
        key = (st.done, st.failed)
        if key in self._act:
            return self._act[key]
        best, bu = None, -1e9
        for i in self.est.ready(st):
            q = self.est.p_exec(i, st)
            u = (q * self.value(st.with_success(i), self.k - 1)
                 + (1 - q) * self.value(st.with_failure(i), self.k - 1) - self.cfg.lam * M.TASKS[i].cost)
            if u > bu + 1e-12 or (abs(u - bu) <= 1e-12 and best is not None
                                  and M.TASKS[i].name < M.TASKS[best].name):
                best, bu = i, u
        self._act[key] = best
        return best


class RolloutPolicy(M.Policy):
    """One-step policy improvement with simulated continuations (agent model)."""

    def __init__(self, rel, cfg: M.Config, n: int = 24, seed: int = 0):
        self.base = M.MyopicPolicy(rel, cfg)
        self.est = self.base.est
        self.cfg, self.n = cfg, n
        self.rng = random.Random(seed)
        self._act: Dict[Tuple[int, int], Optional[int]] = {}

    def _sim(self, st: M.State) -> float:
        for _ in range(M.N_TASKS):
            if st.reached_goal():
                return 1.0
            i = self.base.act(st)
            if i is None:
                return 0.0
            st = st.with_success(i) if self.rng.random() < self.est.p_exec(i, st) else st.with_failure(i)
        return float(st.reached_goal())

    def act(self, st: M.State) -> Optional[int]:
        key = (st.done, st.failed)
        if key in self._act:
            return self._act[key]
        best, bu = None, -1e9
        for i in self.est.ready(st):
            q = self.est.p_exec(i, st)
            s1, s0 = st.with_success(i), st.with_failure(i)
            v1 = statistics.fmean(self._sim(s1) for _ in range(self.n))
            v0 = statistics.fmean(self._sim(s0) for _ in range(self.n))
            u = q * v1 + (1 - q) * v0 - self.cfg.lam * M.TASKS[i].cost
            if u > bu:
                best, bu = i, u
        self._act[key] = best
        return best


def make_policy(name: str, rel, cfg: M.Config, seed: int):
    if name == "k=1":
        return M.MyopicPolicy(rel, cfg)
    if name.startswith("k="):
        return KStepPolicy(rel, cfg, int(name[2:]))
    if name == "rollout":
        return RolloutPolicy(rel, cfg, seed=seed)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------

def cell(args):
    uni, cap_kind, seed, train_eps, eval_eps = args
    sys.setrecursionlimit(200000)
    depth = use_universe(uni, seed)
    D = depth if cap_kind == "adequate" else 3
    cfg = M.cfg_lcb(max_depth=D)
    b = M.train(M.cfg_lcb(max_depth=depth), train_eps, seed)   # learning itself is not capped
    snap = b.snapshot_mean()
    out = []
    exact = M.N_TASKS <= 16
    for pname in POLICIES:
        pol = make_policy(pname, snap, cfg, seed)
        t0 = time.perf_counter()
        if exact:
            r = M.evaluate_policy_full(pol, cfg)
        else:
            r = M.evaluate_policy_mc(pol, cfg, eval_eps, random.Random(77 + seed))
        dt = time.perf_counter() - t0
        out.append(dict(universe=uni, cap=cap_kind, depth_cap=D, longest_chain=depth, seed=seed,
                        policy=pname, real=r["real"], claim=r["claim"], ece=r["ece"],
                        eval_seconds=dt, exact=exact))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--train-episodes", type=int, default=800)
    ap.add_argument("--eval-episodes", type=int, default=400)
    ap.add_argument("--out", default="results_rev_EH.json")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    rows = []
    if a.resume and os.path.exists(a.out):
        rows = json.load(open(a.out, encoding="utf-8")).get("rows", [])
    done = {(r["universe"], r["cap"], r["seed"]) for r in rows}
    unis = ["detour-1", "detour-2", "detour-3", "detour-4", "reference", "K=8"]
    jobs = [(u, c, sd, a.train_episodes, a.eval_episodes) for u in unis
            for c in ("adequate", "short") for sd in (SEEDS if u != "K=8" else SEEDS[:6])
            if (u, c, sd) not in done]
    t0 = time.time()
    print(f"{len(jobs)} cells ({len(done)} kept)", flush=True)
    with mp.Pool(a.workers, maxtasksperchild=4) as pool:
        for rr in pool.imap_unordered(cell, jobs):
            rows += rr
            json.dump(dict(rows=rows), open(a.out, "w", encoding="utf-8"), indent=1)
            r0 = rr[0]
            print(f"{r0['universe']:<10} {r0['cap']:<8} seed={r0['seed']:>2} " +
                  " ".join(f"{r['policy']}={100*r['real']:.1f}" for r in rr) +
                  f" ({time.time()-t0:.0f}s)", flush=True)
    # summary
    summ = {}
    for u in unis:
        for c in ("adequate", "short"):
            for p in POLICIES:
                xs = [r for r in rows if r["universe"] == u and r["cap"] == c and r["policy"] == p]
                if xs:
                    summ[f"{u} | {c} | {p}"] = dict(
                        n=len(xs), real=statistics.fmean(x["real"] for x in xs),
                        ece=statistics.fmean(x["ece"] for x in xs),
                        eval_seconds=statistics.fmean(x["eval_seconds"] for x in xs))
    json.dump(dict(rows=rows, summary=summ), open(a.out, "w", encoding="utf-8"), indent=1)
    for k, v in summ.items():
        print(f"{k:<34} real={100*v['real']:.2f} ECE={v['ece']:.3f} t={v['eval_seconds']:.1f}s")


if __name__ == "__main__":
    main()
