#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M-R (revision 2026-11): retry as a learned action inside the transition rule.

Fault model (reference universe). When a tool raises an execution error, the
fault is transient with probability tau(tool) (a retry is a fresh draw of the
nominal execution success) and persistent otherwise (every retry in the same
episode fails):
    FetchStockA 0.9, FetchMarketFeed 0.9, FetchNews 0.8   (rate limits, timeouts)
    FetchEcon 0.1, FetchStockB 0.1                         (outages, auth errors)
    every other tool 0.5
The episode has a budget of B = 24 tool calls (retries count); retry outcomes
are logged only into the retry posterior, so p_hat stays a first-attempt rate.

Arms
  no-retry      the specified rule: a failed tool joins F (Sec. III-E)
  fixed-retry   retry every execution error up to 2 times, then F
                (Reflect-and-Retry style, no learning)
  learned+budget  as learned, and a retry is made only if the calls left
                after it still cover the shortest remaining plan
  learned       M-R: one Beta posterior per tool for "a retry after an
                execution error succeeds", learned from the logs. After an
                execution error of e the rule compares
                  U_retry = r_e * V(S + e) + (1 - r_e) * V(S + F{e}) - lambda
                  U_alt   = max over ready e' != e of Eq.15 in S + F{e}
                (V: Eq.9-12) and retries when U_retry > U_alt (at most 3
                retries per tool). Training uses a Thompson draw of r_e.
Measured: completion, tool calls per episode, retries per episode.
10 seeds x 1,600 training episodes, 4,000 evaluation episodes (Monte Carlo).

    python e_retry.py --out results_rev_MR.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import statistics
import sys
import time

sys.path.insert(0, ".")
import shgdsm as M

SEEDS = list(range(1, 11))
TAU = {"FetchStockA": 0.9, "FetchMarketFeed": 0.9, "FetchNews": 0.8,
       "FetchEcon": 0.1, "FetchStockB": 0.1}
BUDGET = 24
BUDGETS = [12, 16, 24]
MAX_RETRY = 3
ARMS = ["no-retry", "fixed-retry", "learned", "learned+budget"]


def min_calls(st, est, limit=12):
    """Fewest tool calls that can still produce the goal from st (failed
    tools excluded; a substitution with rel >= theta costs nothing)."""
    sigma, used = st.sigma(), st.used()
    memo = {}

    def c(s, d):
        if s in sigma:
            return 0
        if d > limit:
            return 10 ** 6
        key = (s, d)
        if key in memo:
            return memo[key]
        memo[key] = 10 ** 6
        best = 10 ** 6
        for i, t in enumerate(M.TASKS):
            if not (used >> i) & 1 and s in t.outputs:
                best = min(best, 1 + sum(c(x, d + 1) for x in t.inputs))
        for x in M.SCHEMAS:
            if x != s and est.rel.get((x, s), 0.0) >= est.cfg.theta:
                best = min(best, c(x, d + 1))
        memo[key] = best
        return best
    return c(M.GOAL, 0)


def episode(pol, cfg, rng, arm, retry_counts, learn=False, explore=False, belief=None,
            budget=None):
    budget = budget or BUDGET
    st = M.State()
    calls = retries = 0
    while calls < budget and not st.reached_goal():
        i = pol.act(st)
        if i is None:
            break
        t = M.TASKS[i]
        sigma = st.sigma()
        subs = [(pol.est.best_substitution(s, sigma)[0], s) for s in t.inputs]
        p_exec = t.p * (t.w(cfg.grade_w) if cfg.env_realizes_w else 1.0)
        calls += 1
        ok_exec = rng.random() < p_exec
        if not ok_exec:
            transient = rng.random() < TAU.get(t.name, 0.5)
            if belief is not None:
                belief.observe_exec(t.name, False)
            n_try = 0
            while not ok_exec and calls < budget and n_try < MAX_RETRY:
                if arm == "no-retry":
                    break
                if arm == "fixed-retry":
                    if n_try >= 2:
                        break
                elif arm.startswith("learned"):         # learned utility comparison
                    a, b = retry_counts.get(t.name, (0.0, 0.0))
                    r = rng.betavariate(a + 1, b + 1) if explore else (a + 1) / (a + b + 2)
                    s_ok, s_f = st.with_success(i), st.with_failure(i)
                    u_retry = r * pol.est.p_goal(s_ok) + (1 - r) * pol.est.p_goal(s_f) - cfg.lam
                    alts = [row[4] for row in pol.scores(s_f)] if hasattr(pol, "scores") else []
                    u_alt = max(alts) if alts else pol.est.p_goal(s_f)
                    if u_retry <= u_alt:
                        break
                    if arm == "learned+budget" and                             budget - calls - 1 < min_calls(st.with_success(i), pol.est):
                        break
                n_try += 1
                retries += 1
                calls += 1
                ok_exec = transient and rng.random() < p_exec
                if learn:
                    c = retry_counts.setdefault(t.name, [0.0, 0.0])
                    c[0 if ok_exec else 1] += 1
            if not ok_exec:
                st = st.with_failure(i)
                continue
        verdicts, ok = [], True
        for x, s in subs:
            passed = rng.random() < M.rho_star(x, s)
            verdicts.append((x, s, passed))
            ok &= passed
        if belief is not None:
            if ok_exec:
                M.observe_step(belief, cfg, rng, t.name, verdicts, ok)
        st = st.with_success(i) if ok else st.with_failure(i)
    return float(st.reached_goal()), calls, retries


def cell(args):
    seed, train_eps, eval_eps, budget = args
    M.use_reference_universe()
    cfg = M.cfg_lcb()
    out = []
    for arm in ARMS:
        rng = random.Random(seed)
        b = M.RelBelief(cfg)
        rc = {}
        for _ in range(train_eps):
            pol = M.MyopicPolicy(b.snapshot_sample(rng), M.explore_cfg(cfg), "train")
            episode(pol, cfg, rng, arm, rc, learn=True, explore=True, belief=b, budget=budget)
        pol = M.MyopicPolicy(b.snapshot_mean(), cfg)
        erng = random.Random(4242 + seed)
        res = [episode(pol, cfg, erng, arm, rc, budget=budget) for _ in range(eval_eps)]
        out.append(dict(seed=seed, arm=arm, budget=budget,
                        real=statistics.fmean(r[0] for r in res),
                        calls=statistics.fmean(r[1] for r in res),
                        retries=statistics.fmean(r[2] for r in res),
                        claim=pol.est.p_goal(M.State()),
                        retry_success={k: (v[0] + 1) / (v[0] + v[1] + 2) for k, v in rc.items()}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--train-episodes", type=int, default=1600)
    ap.add_argument("--eval-episodes", type=int, default=4000)
    ap.add_argument("--out", default="results_rev_MR.json")
    a = ap.parse_args()
    t0 = time.time()
    rows = []
    with mp.Pool(a.workers) as pool:
        for rr in pool.imap_unordered(cell, [(s, a.train_episodes, a.eval_episodes, B)
                                             for B in BUDGETS for s in SEEDS]):
            rows += rr
            json.dump(dict(rows=rows), open(a.out, "w", encoding="utf-8"), indent=1)
    from scipy.stats import wilcoxon
    summ = {}
    for B in BUDGETS:
      base = {r["seed"]: r for r in rows if r["arm"] == "no-retry" and r["budget"] == B}
      for arm in ARMS:
        xs = {r["seed"]: r for r in rows if r["arm"] == arm and r["budget"] == B}
        d = [xs[s]["real"] - base[s]["real"] for s in sorted(xs)]
        summ[f"B={B} | {arm}"] = dict(real=statistics.fmean(x["real"] for x in xs.values()),
                         calls=statistics.fmean(x["calls"] for x in xs.values()),
                         retries=statistics.fmean(x["retries"] for x in xs.values()),
                         diff_vs_no_retry=statistics.fmean(d),
                         wilcoxon_p=(float(wilcoxon(d).pvalue) if arm != "no-retry" else None))
    json.dump(dict(rows=rows, summary=summ), open(a.out, "w", encoding="utf-8"), indent=1)
    for k, v in summ.items():
        print(f"{k:<12} real={100*v['real']:.2f} calls={v['calls']:.2f} retries={v['retries']:.2f} "
              f"diff={100*v['diff_vs_no_retry']:+.2f} p={v['wilcoxon_p']}")
    print(f"wrote {a.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
