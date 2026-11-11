#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-D (revision 2026-11): non-stationarity and production calibration monitoring.

Reference universe. The learner trains 800 episodes in the nominal environment,
then the environment changes and training continues to 2,000 episodes. Every
100 episodes the current planning snapshot is evaluated
  * exactly (DP) in the CURRENT environment: real, claim, gap, ECE, and
  * from a window of 100 deployed episodes (Monte Carlo), as an operator
    would see it: windowed ECE over (claimed value, outcome) pairs and a
    one-sided binomial test of the claimed start value against the observed
    successes (alarm at p < 0.01).

Scenarios (change at episode 800)
  drift     FetchStockA p 0.80 -> 0.30 (main source degrades)
  drift-news FetchNews p 0.90 -> 0.20
  withdraw  FetchEcon removed
  add       FetchEconB added (instruction -> econ_raw, p 0.85, declared)
  overlap   rel*(market_raw -> stock_raw) 0.85 -> 0.20 (format change)
Learners: forget gamma in {1.0, 0.995, 0.98} (per-episode count discount).

    python e_drift.py --out results_rev_ED.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import random
import statistics
import sys
import time

sys.path.insert(0, ".")
import shgdsm as M

SEEDS = list(range(1, 11))
CHANGE_AT = 800
TOTAL = 2000
EVERY = 100
WINDOW = 100
GAMMAS = [1.0, 0.995, 0.98]
ALARM_KEEP = 0.1     # "alarm": no forgetting; an over-claim alarm keeps 10% of the counts
CHECKPOINTS = sorted(set(range(EVERY, TOTAL + 1, EVERY))
                     | set(range(CHANGE_AT + 20, CHANGE_AT + 401, 20)))
SCENARIOS = ["drift", "drift-news", "withdraw", "add", "overlap"]


def apply_change(name: str) -> None:
    schemas, tasks, rho, goal, init = M.get_universe()
    tasks, rho = list(tasks), dict(rho)
    if name == "drift":
        tasks = [dataclasses.replace(t, p=0.30) if t.name == "FetchStockA" else t for t in tasks]
    elif name == "drift-news":
        tasks = [dataclasses.replace(t, p=0.20) if t.name == "FetchNews" else t for t in tasks]
    elif name == "withdraw":
        tasks = [t for t in tasks if t.name != "FetchEcon"]
    elif name == "add":
        tasks.append(M.Task("FetchEconB", ("instruction",), ("econ_raw",), 0.85, ("declared",)))
    elif name == "overlap":
        rho[("market_raw", "stock_raw")] = 0.20
    else:
        raise ValueError(name)
    M.set_universe(schemas, tasks, rho, goal, init)


def binom_p_upper(k: int, n: int, p: float) -> float:
    """P(X <= k | n, p): small when the claim p is too high for k successes."""
    from scipy.stats import binom
    return float(binom.cdf(k, n, p))


def binom_p_lower(k: int, n: int, p: float) -> float:
    from scipy.stats import binom
    return float(binom.sf(k - 1, n, p))


def cell(args):
    scen, gamma, seed = args
    M.use_reference_universe()
    cfg = M.cfg_lcb(forget=(1.0 if gamma == "alarm" else gamma))
    rng = random.Random(seed)
    mon_rng = random.Random(10_000 + seed)
    b = M.RelBelief(cfg)
    out = []
    for ep in range(1, TOTAL + 1):
        if ep == CHANGE_AT + 1:
            apply_change(scen)
        M.run_episode(b, cfg, rng, explore=True)
        if ep in CHECKPOINTS:
            snap = b.snapshot_mean()
            r = M.full_eval(snap, cfg)
            mc = M.evaluate_policy_mc(M.MyopicPolicy(snap, cfg), cfg, WINDOW, mon_rng)
            k = round(mc["real"] * WINDOW)
            p_over = binom_p_upper(k, WINDOW, min(max(mc["claim"], 1e-9), 1 - 1e-9))
            p_under = binom_p_lower(k, WINDOW, min(max(mc["claim"], 1e-9), 1 - 1e-9))
            out.append(dict(scenario=scen, gamma=gamma, seed=seed, episode=ep,
                            real=r["real"], claim=r["claim"], gap=r["claim"] - r["real"],
                            ece=r["ece"], mon_ece=mc["ece"], mon_success=mc["real"],
                            alarm_over=p_over < 0.01, alarm_under=p_under < 0.01))
            if gamma == "alarm" and p_over < 0.01:
                b.decay(ALARM_KEEP)          # alarm-triggered forgetting
    return out


def summarize(rows):
    summ = {}
    for scen in SCENARIOS:
        for g in GAMMAS:
            rs = [r for r in rows if r["scenario"] == scen and r["gamma"] == g]
            if not rs:
                continue
            curve = []
            for ep in sorted({r["episode"] for r in rs}):
                xs = [r for r in rs if r["episode"] == ep]
                f = lambda k: statistics.fmean(x[k] for x in xs)
                curve.append(dict(episode=ep, real=f("real"), claim=f("claim"), gap=f("gap"),
                                  ece=f("ece"), mon_ece=f("mon_ece"),
                                  alarm_rate=statistics.fmean(
                                      float(x["alarm_over"] or x["alarm_under"]) for x in xs)))
            # pre-change false alarm rate, detection delay (first alarm after change)
            pre = [r for r in rs if r["episode"] <= CHANGE_AT]
            fa = statistics.fmean(float(r["alarm_over"] or r["alarm_under"]) for r in pre)
            delays = []
            for sd in SEEDS:
                post = sorted((r for r in rs if r["seed"] == sd and r["episode"] > CHANGE_AT),
                              key=lambda r: r["episode"])
                hit = next((r["episode"] - CHANGE_AT for r in post
                            if r["alarm_over"] or r["alarm_under"]), None)
                delays.append(hit)
            last = [r for r in rs if r["episode"] == TOTAL]
            summ[f"{scen} | gamma={g}"] = dict(
                curve=curve, false_alarm_pre=fa,
                detected=sum(d is not None for d in delays) / len(delays),
                delay_median=(statistics.median([d for d in delays if d is not None])
                              if any(d is not None for d in delays) else None),
                final_real=statistics.fmean(r["real"] for r in last),
                final_gap=statistics.fmean(r["gap"] for r in last),
                final_ece=statistics.fmean(r["ece"] for r in last),
                # mean |gap| over the 400 episodes right after the change
                post_abs_gap=statistics.fmean(abs(r["gap"]) for r in rs
                                              if CHANGE_AT < r["episode"] <= CHANGE_AT + 400))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--gammas", default=None, help='e.g. "alarm" or "1.0,0.995"')
    ap.add_argument("--out", default="results_rev_ED.json")
    a = ap.parse_args()
    t0 = time.time()
    global GAMMAS
    if a.gammas:
        GAMMAS = [g if g == "alarm" else float(g) for g in a.gammas.split(",")]
    jobs = [(s, g, sd) for s in SCENARIOS for g in GAMMAS for sd in SEEDS[:a.seeds]]
    rows = []
    with mp.Pool(a.workers) as pool:
        for n, rr in enumerate(pool.imap_unordered(cell, jobs), 1):
            rows += rr
            if n % 10 == 0:
                print(f"{n}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    summ = summarize(rows)
    json.dump(dict(rows=rows, summary=summ), open(a.out, "w", encoding="utf-8"), indent=1)
    for k, v in summ.items():
        print(f"{k:<26} final real={100*v['final_real']:.1f} gap={100*v['final_gap']:+.1f} "
              f"ECE={v['final_ece']:.3f} post|gap|={100*v['post_abs_gap']:.1f} "
              f"FA={v['false_alarm_pre']:.2f} det={v['detected']:.1f} delay={v['delay_median']}")
    print(f"wrote {a.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
