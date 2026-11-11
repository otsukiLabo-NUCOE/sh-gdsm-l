#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-B (revision 2026-11): additional baselines.

  SH-GD/SM-L    the proposed model (LCB, learned rel and p_hat)
  tool-bandit   tool-level Bayesian selection in the style of [22][23]: one
                Beta posterior per TOOL for "a call of this tool succeeds"
                (execution and validation together), Thompson sampling during
                training; substitutes are admitted by name similarity (same
                schema suffix, e.g. *_raw -> *_raw) and their effect is only
                visible through the tool's own success rate, so nothing is
                learned about schema pairs
  similarity    description/name-similarity substitution, no learning
                (rel = 0.5 for a plausible pair, declared p_e * w(g))
  exact-match   identity matches only, learned p_hat
  oracle        true rel* and p_hat with the same one-step rule
  optimal       exact optimal policy under the true model (reference universe
                only; the definition of "Oracle" in the manuscript)
Reference universe: exact DP, 10 seeds x 1,600 training episodes.
Random universes K=8: Monte Carlo, 8 universes x 800 training episodes, 2,000
evaluation episodes; plus the transfer test (a dedicated fetcher withdrawn
after training, beliefs reused).

    python e_baselines.py --out results_rev_EB.json
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
import envgen as E

SEEDS = list(range(1, 11))
ARMS = ["SH-GD/SM-L", "tool-bandit", "similarity", "exact-match", "oracle", "optimal"]


def suffix(s: str) -> str:
    return s.rsplit("_", 1)[-1]


def plausible(x: str, s: str, kind: str) -> bool:
    if x == s:
        return True
    if kind == "K=8":
        if x.startswith("g") and x.endswith("_raw") and s.endswith("_raw"):
            return True
        return x.endswith("_digest") and s.endswith("_clean") and x.split("_")[0] == s.split("_")[0]
    return suffix(x) == suffix(s) and x != "instruction"


def gate_table(kind: str, val: float) -> M.Snapshot:
    return M._table(lambda x, s: val if plausible(x, s, kind) else 0.0)


def train_bandit(cfg, kind, episodes, seed):
    """Per-tool Beta(1,1) on end-to-end call success."""
    rng = random.Random(seed)
    a = {t.name: 0.0 for t in M.TASKS}
    b = {t.name: 0.0 for t in M.TASKS}
    gate = gate_table(kind, 0.99)        # identity (1.0) preferred over a look-alike
    for _ in range(episodes):
        snap = M.Snapshot(gate)
        snap.phat = {t.name: rng.betavariate(a[t.name] + 1, b[t.name] + 1) for t in M.TASKS}
        pol = M.MyopicPolicy(snap, M.explore_cfg(cfg), "train")
        st = M.State()
        for _ in range(M.N_TASKS):
            if st.reached_goal():
                break
            i = pol.act(st)
            if i is None:
                break
            qe, qv, _ = M.true_exec_prob(i, st, pol.est, cfg)
            ok = rng.random() < qe and rng.random() < qv
            (a if ok else b)[M.TASKS[i].name] += 1
            st = st.with_success(i) if ok else st.with_failure(i)
    snap = M.Snapshot(gate)
    snap.phat = {t.name: (a[t.name] + 1) / (a[t.name] + b[t.name] + 2) for t in M.TASKS}
    return snap


def policies(kind, cfg, belief, bandit_snap):
    snap = belief.snapshot_mean()
    sim = gate_table(kind, 0.5)
    return {"SH-GD/SM-L": M.MyopicPolicy(snap, cfg),
            "tool-bandit": M.MyopicPolicy(bandit_snap, cfg),
            "similarity": M.MyopicPolicy(sim, cfg),
            "exact-match": M.MyopicPolicy(M.with_phat(M.exact_rel_table(), snap.phat), cfg),
            "oracle": M.MyopicPolicy(M.with_phat(M.oracle_rel_table(), M.true_phat(cfg)), cfg)}


def ref_cell(seed):
    M.use_reference_universe()
    cfg = M.cfg_lcb()
    b = M.train(cfg, 1600, seed)
    bs = train_bandit(cfg, "reference", 1600, seed)
    out = []
    for name, pol in policies("reference", cfg, b, bs).items():
        r = M.evaluate_policy_full(pol, cfg)
        out.append(dict(universe="reference", seed=seed, arm=name, real=r["real"], claim=r["claim"],
                        ece=r["ece"]))
    out.append(dict(universe="reference", seed=seed, arm="optimal", real=M.optimal_value(cfg),
                    claim=None, ece=None))
    # transfer: FetchEcon withdrawn, beliefs reused
    saved = M.get_universe()
    M.set_universe(saved[0], [t for t in saved[1] if t.name != "FetchEcon"], saved[2], saved[3], saved[4])
    bs2 = M.Snapshot(bs)
    bs2.phat = {t.name: bs.phat[t.name] for t in M.TASKS}
    for name, pol in policies("reference", cfg, b, bs2).items():
        r = M.evaluate_policy_full(pol, cfg)
        out.append(dict(universe="reference (transfer)", seed=seed, arm=name, real=r["real"],
                        claim=r["claim"], ece=r["ece"]))
    out.append(dict(universe="reference (transfer)", seed=seed, arm="optimal",
                    real=M.optimal_value(cfg), claim=None, ece=None))
    M.set_universe(*saved)
    return out


def rand_cell(u):
    sys.setrecursionlimit(100000)
    U = E.generate(8, u)
    M.set_universe(U["schemas"], U["tasks"], U["rho"])
    cfg = M.cfg_lcb(max_depth=U["depth"])
    b = M.train(cfg, 800, 1000 + u)
    bs = train_bandit(cfg, "K=8", 800, 1000 + u)
    out = []
    for name, pol in policies("K=8", cfg, b, bs).items():
        r = M.evaluate_policy_mc(pol, cfg, 2000, random.Random(77 + u))
        out.append(dict(universe="K=8", seed=u, arm=name, real=r["real"], claim=r["claim"], ece=r["ece"]))
    saved = M.get_universe()
    M.set_universe(saved[0], [t for t in saved[1] if t.name not in U["withdraw"]], saved[2])
    bs2 = M.Snapshot(bs)
    bs2.phat = {t.name: bs.phat[t.name] for t in M.TASKS}
    for name, pol in policies("K=8", cfg, b, bs2).items():
        r = M.evaluate_policy_mc(pol, cfg, 2000, random.Random(78 + u))
        out.append(dict(universe="K=8 (transfer)", seed=u, arm=name, real=r["real"], claim=r["claim"],
                        ece=r["ece"]))
    M.set_universe(*saved)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="results_rev_EB.json")
    a = ap.parse_args()
    t0 = time.time()
    rows = []
    with mp.Pool(a.workers) as pool:
        for rr in pool.imap_unordered(ref_cell, SEEDS):
            rows += rr
            json.dump(dict(rows=rows), open(a.out, "w", encoding="utf-8"), indent=1)
        print(f"reference done ({time.time()-t0:.0f}s)", flush=True)
        for rr in pool.imap_unordered(rand_cell, range(1, 9)):
            rows += rr
            json.dump(dict(rows=rows), open(a.out, "w", encoding="utf-8"), indent=1)
    from scipy.stats import wilcoxon
    summ = {}
    for uni in dict.fromkeys(r["universe"] for r in rows):
        base = {r["seed"]: r for r in rows if r["universe"] == uni and r["arm"] == "SH-GD/SM-L"}
        for arm in ARMS:
            xs = {r["seed"]: r for r in rows if r["universe"] == uni and r["arm"] == arm}
            if not xs:
                continue
            d = [xs[s]["real"] - base[s]["real"] for s in sorted(xs)]
            eces = [x["ece"] for x in xs.values() if x["ece"] is not None]
            summ[f"{uni} | {arm}"] = dict(
                real=statistics.fmean(x["real"] for x in xs.values()),
                ece=statistics.fmean(eces) if eces else None,
                diff_vs_proposed=statistics.fmean(d),
                wilcoxon_p=(float(wilcoxon(d).pvalue)
                            if arm != "SH-GD/SM-L" and any(abs(v) > 1e-12 for v in d) else None))
    json.dump(dict(rows=rows, summary=summ), open(a.out, "w", encoding="utf-8"), indent=1)
    for k, v in summ.items():
        e = f"{v['ece']:.3f}" if v["ece"] is not None else "  -  "
        print(f"{k:<38} real={100*v['real']:.2f} ECE={e} diff={100*v['diff_vs_proposed']:+.2f} "
              f"p={v['wilcoxon_p']}")
    print(f"wrote {a.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
