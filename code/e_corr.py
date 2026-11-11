#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-F (revision 2026-11): correlated failures (provider outages).

Reference universe. Tools are grouped by provider:
    provider X: FetchStockA, FetchEcon, FetchNews       (primary data vendor)
    provider Y: FetchStockB, FetchMarketFeed, FetchNewsAlt
    (processing tasks are local and never part of an outage)
In each episode provider X is down with probability q (an outage): every call
to one of its tools is an execution error for the whole episode. Outside an
outage the tools behave as in the nominal universe. q in {0, 0.1, 0.2, 0.4}.

Learners / planners
  independent     SH-GD/SM-L as specified (failures assumed independent)
  provider-aware  same learner plus one ratio per provider, learned from the
                  logs: observed / expected execution successes of tools
                  called after another tool of the same provider has failed in
                  the same episode (expected = sum of their p_hat). After such
                  a failure the planner re-plans with the provider's remaining
                  p_hat multiplied by that ratio (capped at 1).
  fixed plan      learned rel, plan fixed at S0 (reference)
Measured: realized completion, claim, ECE (Monte Carlo, 4,000 episodes per
seed), 10 seeds x 1,600 training episodes (training under the same q).

    python e_corr.py --out results_rev_EF.json
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

sys.path.insert(0, ".")
import shgdsm as M

SEEDS = list(range(1, 11))
QS = [0.0, 0.1, 0.2, 0.4]
PROVIDER = {"FetchStockA": "X", "FetchEcon": "X", "FetchNews": "X",
            "FetchStockB": "Y", "FetchMarketFeed": "Y", "FetchNewsAlt": "Y"}
ARMS = ["independent", "provider-aware", "fixed plan"]


def conditioned_policy(pol, cfg, corr, prov):
    """Copy of pol whose p_hat for the provider's tools is the learned
    conditional reliability after a failure of a provider-mate."""
    obs, exp_ = corr.get(prov, (0.0, 0.0))
    lift = min(1.0, (obs + 1.0) / (exp_ + 1.0))      # observed / expected successes
    snap = M.Snapshot(pol.est.rel)
    snap.phat = {t.name: (pol.est.phat[i] * lift if PROVIDER.get(t.name) == prov else pol.est.phat[i])
                 for i, t in enumerate(M.TASKS)}
    return M.MyopicPolicy(snap, cfg)


def episode(pol, cfg, rng, q, belief=None, aware=False, record=None, corr=None):
    """One episode in the outage environment. Learns when belief is given;
    corr (dict provider -> [ok, fail]) collects conditional outcomes."""
    down = "X" if rng.random() < q else None
    failed_prov = set()
    st = M.State()
    preds = []
    for _ in range(M.N_TASKS):
        if st.reached_goal():
            break
        i = pol.act(st)
        if i is None:
            break
        if record is not None:
            preds.append(pol.est.p_goal(st))
        t = M.TASKS[i]
        sigma = st.sigma()
        subs = [(pol.est.best_substitution(s, sigma)[0], s) for s in t.inputs]
        p_exec = 0.0 if (down and PROVIDER.get(t.name) == down) else \
            t.p * (t.w(cfg.grade_w) if cfg.env_realizes_w else 1.0)
        prov = PROVIDER.get(t.name)
        exec_ok = rng.random() < p_exec
        if corr is not None and prov in failed_prov:
            c = corr.setdefault(prov, [0.0, 0.0])      # [observed, expected] successes
            c[0] += float(exec_ok)
            c[1] += (belief.exec_mean(t) if belief is not None else pol.est.phat[i])
        if not exec_ok:
            if belief is not None:
                belief.observe_exec(t.name, False)
            st = st.with_failure(i)
            if prov is not None and prov not in failed_prov:
                failed_prov.add(prov)
                if aware:
                    pol = conditioned_policy(pol, cfg, aware, prov)
            continue
        verdicts, ok = [], True
        for x, s in subs:
            passed = rng.random() < M.rho_star(x, s)
            verdicts.append((x, s, passed))
            ok &= passed
        if belief is not None:
            M.observe_step(belief, cfg, rng, t.name, verdicts, ok)
        st = st.with_success(i) if ok else st.with_failure(i)
    y = 1.0 if st.reached_goal() else 0.0
    if record is not None:
        record.extend((1.0, p, y) for p in preds)
    return y


def cell(args):
    q, seed, train_eps, eval_eps = args
    M.use_reference_universe()
    cfg = M.cfg_lcb()
    rng = random.Random(seed)
    b = M.RelBelief(cfg)
    corr = {}
    for _ in range(train_eps):
        snap = b.snapshot_sample(rng)
        pol = M.MyopicPolicy(snap, M.explore_cfg(cfg), "train")
        episode(pol, cfg, rng, q, belief=b, corr=corr)
    snap = b.snapshot_mean()
    out = []
    for arm in ARMS:
        pol = M.FixedPlanPolicy(snap, cfg) if arm == "fixed plan" else M.MyopicPolicy(snap, cfg)
        erng = random.Random(4242 + seed)
        pts = []
        wins = sum(episode(pol, cfg, erng, q, aware=(corr if arm == "provider-aware" else False),
                           record=pts) for _ in range(eval_eps))
        real = wins / eval_eps
        claim = pol.est.p_goal(M.State())
        out.append(dict(q=q, seed=seed, arm=arm, real=real, claim=claim, gap=claim - real,
                        ece=M._ece(pts), corr={k: list(v) for k, v in corr.items()}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--train-episodes", type=int, default=1600)
    ap.add_argument("--eval-episodes", type=int, default=4000)
    ap.add_argument("--out", default="results_rev_EF.json")
    a = ap.parse_args()
    jobs = [(q, sd, a.train_episodes, a.eval_episodes) for q in QS for sd in SEEDS]
    rows = []
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for rr in pool.imap_unordered(cell, jobs):
            rows += rr
            json.dump(dict(rows=rows), open(a.out, "w", encoding="utf-8"), indent=1)
    from scipy.stats import wilcoxon
    summ = {}
    for q in QS:
        base = {r["seed"]: r for r in rows if r["q"] == q and r["arm"] == "independent"}
        for arm in ARMS:
            xs = {r["seed"]: r for r in rows if r["q"] == q and r["arm"] == arm}
            d = [xs[s]["real"] - base[s]["real"] for s in sorted(xs)]
            p = (float(wilcoxon(d).pvalue) if arm != "independent" and any(abs(v) > 1e-12 for v in d)
                 else None)
            summ[f"q={q} | {arm}"] = dict(
                real=statistics.fmean(x["real"] for x in xs.values()),
                gap=statistics.fmean(x["gap"] for x in xs.values()),
                ece=statistics.fmean(x["ece"] for x in xs.values()),
                diff_vs_independent=statistics.fmean(d), wilcoxon_p=p)
    json.dump(dict(rows=rows, summary=summ), open(a.out, "w", encoding="utf-8"), indent=1)
    for k, v in summ.items():
        print(f"{k:<28} real={100*v['real']:.2f} gap={100*v['gap']:+.2f} ECE={v['ece']:.3f} "
              f"diff={100*v['diff_vs_independent']:+.2f} p={v['wilcoxon_p']}")
    print(f"wrote {a.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
