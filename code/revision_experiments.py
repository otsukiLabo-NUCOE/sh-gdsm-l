#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revision experiments (2026-11) for Access-2026-52092.

  E-N  degraded supervision: validator accuracy, missing verdicts, outcome-only
  E-A  ablations: schema learning, reliability learning, LCB, look-ahead, dynamics
  E-S  paired statistical tests (Wilcoxon signed-rank) against the full model

Reference universe: 10 seeds x 1,600 training episodes, exact DP evaluation.
Random universes (envgen.py): 8 universes, 800 training episodes, Monte Carlo.

    python revision_experiments.py --part EN,EA --out results_rev_EN_EA.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import statistics
import sys
import time
from typing import Dict, List

sys.path.insert(0, ".")
import shgdsm as M
import envgen as E

SEEDS = list(range(1, 11))


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

def noise_settings() -> Dict[str, dict]:
    return {
        "acc=1.0 (nominal)": dict(),
        "acc=0.9": dict(val_acc=0.9),
        "acc=0.7": dict(val_acc=0.7),
        "acc=0.5": dict(val_acc=0.5),
        "missing=0.2": dict(val_missing=0.2),
        "missing=0.5": dict(val_missing=0.5),
        "missing=0.8": dict(val_missing=0.8),
        "outcome-only": dict(signal="outcome_only"),
    }


def corrected_settings() -> Dict[str, dict]:
    """ENC: the same degradations, learner with the matching estimator."""
    return {
        "acc=1.0 (nominal)": dict(),
        "acc=0.9 +noise model": dict(val_acc=0.9, val_correct=True),
        "acc=0.7 +noise model": dict(val_acc=0.7, val_correct=True),
        "acc=0.7 +noise model (assumed 0.8)": dict(val_acc=0.7, val_correct=True, val_acc_hat=0.8),
        "acc=0.7 +noise model (assumed 0.6)": dict(val_acc=0.7, val_correct=True, val_acc_hat=0.6),
        "missing=0.5 +soft attribution": dict(val_missing=0.5, attribution="soft"),
        "missing=0.8 +soft attribution": dict(val_missing=0.8, attribution="soft"),
        "outcome-only +soft attribution": dict(signal="outcome_only", attribution="soft"),
    }


def settings_for(part: str) -> Dict[str, dict]:
    return corrected_settings() if part == "ENC" else noise_settings()


ABLATIONS = ["full", "-schema learning", "-reliability learning", "-LCB (mean, theta=0.12)",
             "-LCB (mean, theta=0)", "-look-ahead (greedy)", "-dynamic (fixed plan)"]


def ablation_policy(name: str, b: M.RelBelief, cfg: M.Config):
    """-> (policy, evaluation config)"""
    if name == "full":
        return M.MyopicPolicy(b.snapshot_mean(), cfg), cfg
    if name == "-schema learning":
        return M.MyopicPolicy(M.with_phat(M.exact_rel_table(), b.snapshot_mean().phat), cfg), cfg
    if name == "-reliability learning":
        c = M.cfg_replace(cfg, learn_exec=False)
        return M.MyopicPolicy(M.clone_belief(b, c).snapshot_mean(), c), c
    if name.startswith("-LCB (mean, theta=0.12)"):
        c = M.cfg_replace(cfg, point_estimate="mean", theta=0.12)
        return M.MyopicPolicy(M.clone_belief(b, c).snapshot_mean(), c), c
    if name.startswith("-LCB (mean, theta=0)"):
        c = M.cfg_replace(cfg, point_estimate="mean", theta=0.0)
        return M.MyopicPolicy(M.clone_belief(b, c).snapshot_mean(), c), c
    if name.startswith("-look-ahead"):
        return M.GreedyPolicy(b.snapshot_mean(), cfg), cfg
    if name.startswith("-dynamic"):
        return M.FixedPlanPolicy(b.snapshot_mean(), cfg), cfg
    raise ValueError(name)


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------

def ref_noise_cell(args):
    label, kw, seed, part = (args + ("EN",))[:4]
    M.use_reference_universe()
    cfg = M.cfg_lcb(**kw)
    b = M.train(cfg, 1600, seed)
    r = M.full_eval(b.snapshot_mean(), cfg)
    err = b.rel_errors()
    return dict(part=part, universe="reference", setting=label, seed=seed, real=r["real"],
                claim=r["claim"], gap=r["claim"] - r["real"], ece=r["ece"], brier=r["brier"],
                rel_err_nonzero=err["nonzero"], rel_err_observed=err["observed"])


def ref_ablation_cell(seed):
    M.use_reference_universe()
    cfg = M.cfg_lcb()
    b = M.train(cfg, 1600, seed)
    out = []
    for name in ABLATIONS:
        pol, c = ablation_policy(name, b, cfg)
        r = M.evaluate_policy_full(pol, c)
        out.append(dict(part="EA", universe="reference", setting=name, seed=seed, real=r["real"],
                        claim=r["claim"], gap=r["claim"] - r["real"], ece=r["ece"], brier=r["brier"]))
    return out


def rand_cell(args):
    K, u, train_eps, eval_eps, parts = args
    sys.setrecursionlimit(100000)
    U = E.generate(K, u)
    M.set_universe(U["schemas"], U["tasks"], U["rho"])
    out = []
    for part in [q for q in ("EN", "ENC") if q in parts]:
        for label, kw in settings_for(part).items():
            cfg = M.cfg_lcb(max_depth=U["depth"], **kw)
            b = M.train(cfg, train_eps, 1000 + u)
            r = M.evaluate_policy_mc(M.MyopicPolicy(b.snapshot_mean(), cfg), cfg, eval_eps,
                                     random.Random(77 + u))
            out.append(dict(part=part, universe=f"K={K}", setting=label, seed=u, real=r["real"],
                            claim=r["claim"], gap=r["claim"] - r["real"], ece=r["ece"], brier=r["brier"]))
    if "EA" in parts:
        cfg = M.cfg_lcb(max_depth=U["depth"])
        b = M.train(cfg, train_eps, 1000 + u)
        for name in ABLATIONS:
            pol, c = ablation_policy(name, b, cfg)
            if name.startswith("-LCB"):
                c = M.cfg_replace(c, max_depth=U["depth"])
            r = M.evaluate_policy_mc(pol, c, eval_eps, random.Random(77 + u))
            out.append(dict(part="EA", universe=f"K={K}", setting=name, seed=u, real=r["real"],
                            claim=r["claim"], gap=r["claim"] - r["real"], ece=r["ece"], brier=r["brier"]))
    return out


# ---------------------------------------------------------------------------
# statistics (E-S)
# ---------------------------------------------------------------------------

def ci95(xs):
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, 1.96 * statistics.stdev(xs) / len(xs) ** 0.5


def bootstrap_ci(diffs, n=10000, seed=0):
    rng = random.Random(seed)
    k = len(diffs)
    ms = sorted(statistics.fmean(rng.choice(diffs) for _ in range(k)) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def summarize(rows: List[dict], reference_setting: Dict[str, str]) -> Dict:
    from scipy.stats import wilcoxon
    out: Dict = {}
    for part in sorted({r["part"] for r in rows}):
        for uni in sorted({r["universe"] for r in rows if r["part"] == part}):
            rs = [r for r in rows if r["part"] == part and r["universe"] == uni]
            base_name = reference_setting[part]
            base = {r["seed"]: r for r in rs if r["setting"] == base_name}
            table = []
            for setting in dict.fromkeys(r["setting"] for r in rs):
                cur = {r["seed"]: r for r in rs if r["setting"] == setting}
                seeds = sorted(set(base) & set(cur))
                row = dict(setting=setting, n=len(seeds))
                for k in ("real", "gap", "ece", "brier"):
                    m, h = ci95([cur[s][k] for s in seeds])
                    row[k], row[k + "_ci"] = m, h
                if setting != base_name:
                    d = [cur[s]["real"] - base[s]["real"] for s in seeds]
                    row["diff_real"] = statistics.fmean(d)
                    row["diff_real_boot95"] = bootstrap_ci(d)
                    try:
                        row["wilcoxon_p"] = (float(wilcoxon(d).pvalue)
                                             if any(abs(x) > 1e-12 for x in d) else 1.0)
                    except ValueError:
                        row["wilcoxon_p"] = 1.0
                table.append(row)
            out[f"{part} | {uni}"] = table
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="EN,EA")
    ap.add_argument("--sizes", default="8")
    ap.add_argument("--universes", type=int, default=8)
    ap.add_argument("--train-episodes", type=int, default=800)
    ap.add_argument("--eval-episodes", type=int, default=600)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="results_rev_EN_EA.json")
    a = ap.parse_args()
    parts = a.part.split(",")
    t0 = time.time()
    rows: List[dict] = []
    jobs_ref_n = [(lab, kw, sd, part) for part in ("EN", "ENC") if part in parts
                  for lab, kw in settings_for(part).items() for sd in SEEDS]
    jobs_rand = [(int(K), u, a.train_episodes, a.eval_episodes, parts)
                 for K in a.sizes.split(",") for u in range(1, a.universes + 1)]
    with mp.Pool(a.workers) as pool:
        if jobs_ref_n:
            for r in pool.imap_unordered(ref_noise_cell, jobs_ref_n):
                rows.append(r)
            print(f"reference noise done ({time.time()-t0:.0f}s)", flush=True)
        if "EA" in parts:
            for rr in pool.imap_unordered(ref_ablation_cell, SEEDS):
                rows += rr
            print(f"reference ablation done ({time.time()-t0:.0f}s)", flush=True)
        for n, rr in enumerate(pool.imap_unordered(rand_cell, jobs_rand), 1):
            rows += rr
            print(f"random universe {n}/{len(jobs_rand)} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(dict(rows=rows), open(a.out, "w", encoding="utf-8"), indent=1)
    summ = summarize(rows, {"EN": "acc=1.0 (nominal)", "ENC": "acc=1.0 (nominal)", "EA": "full"})
    json.dump(dict(config=vars(a), rows=rows, summary=summ), open(a.out, "w", encoding="utf-8"), indent=1)
    for key, table in summ.items():
        print("==", key)
        for r in table:
            extra = (f"  diff={100*r['diff_real']:+.2f}pt p={r['wilcoxon_p']:.3g}"
                     if "diff_real" in r else "")
            print(f"   {r['setting']:<28} real={100*r['real']:6.2f}  gap={100*r['gap']:+6.2f}  "
                  f"ECE={r['ece']:.3f}{extra}")
    print(f"wrote {a.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
