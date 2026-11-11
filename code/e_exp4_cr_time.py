#!/usr/bin/env python3
"""Experiment 4/5 environment, arm C+R: decision time of the transition rule and
reproducibility of its decisions. The trained belief is frozen (learn=False) and
every episode is replayed with the same random stream; identical step counts and
outcomes in both passes show that the rule takes the same decision in the same
state. No model calls. Revision 2026-11."""
import json, random, statistics, sys
sys.path.insert(0, ".")
import shgdsm as M
import llm_experiment as L
import recovery_experiment as R

cfg = M.Config(point_estimate="lcb", theta=0.01)
SEEDS, EPS = range(1, 11), 500
dec_s, dec_n, ep_s, same, total = 0.0, 0, [], 0, 0
for sd in SEEDS:
    belief = M.RelBelief(cfg)
    trng = random.Random(20_000 + sd)
    for _ in range(1600):
        L.episode_shgdsm(belief, cfg, trng, explore=True)
    passes = []
    for _ in range(2):
        rng = random.Random(10_000 + sd)
        out = []
        for _ in range(EPS):
            r, ds, nd = R.episode_shgdsm(belief, cfg, rng, None, retry=3, learn=False)
            out.append((r.completed, r.steps))
            dec_s += ds; dec_n += nd; ep_s.append(ds)
        passes.append(out)
    same += sum(a == b for a, b in zip(*passes)); total += EPS
res = dict(decision_ms=1000 * dec_s / max(1, dec_n), episode_ms=1000 * statistics.fmean(ep_s),
           reproduced=same / total, episodes=total)
print(res)
json.dump(res, open("results_rev_exp4_CR_time.json", "w"), indent=1)
