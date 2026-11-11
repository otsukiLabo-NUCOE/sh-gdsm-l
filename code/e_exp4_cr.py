#!/usr/bin/env python3
"""Experiment 4 environment (i.i.d. per-call execution errors, real JSON Schema
validation): arm C with and without the retry rule (C+R, up to 3 retries of an
execution error), 10 seeds x 500 episodes, no model calls. Revision 2026-11."""
import json, random, statistics, sys
sys.path.insert(0, ".")
import shgdsm as M
import llm_experiment as L
import recovery_experiment as R

cfg = M.Config(point_estimate="lcb", theta=0.01)
out = {}
for arm, retry in (("C", 0), ("C+R", 3)):
    rates, steps = [], []
    for sd in range(1, 11):
        belief = M.RelBelief(cfg)
        trng = random.Random(20_000 + sd)
        for _ in range(1600):
            L.episode_shgdsm(belief, cfg, trng, explore=True)
        rng = random.Random(10_000 + sd)
        res = [R.episode_shgdsm(belief, cfg, rng, None, retry=retry, learn=False)[0] for _ in range(500)]
        rates.append(statistics.fmean(r.completed for r in res))
        steps.append(statistics.fmean(r.steps for r in res))
    m = statistics.fmean(rates); h = 1.96 * statistics.stdev(rates) / len(rates) ** 0.5
    out[arm] = dict(completion=m, ci=h, steps=statistics.fmean(steps), per_seed=rates)
    print(arm, f"{100*m:.1f}+-{100*h:.1f}", "steps", round(statistics.fmean(steps), 2), flush=True)
json.dump(out, open("results_rev_exp4_CR.json", "w"), indent=1)
