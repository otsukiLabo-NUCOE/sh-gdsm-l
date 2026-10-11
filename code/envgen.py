#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 7: randomly generated task universes of increasing size.

Why
---
Experiments 1-6 use one hand-built universe of 12 tasks whose rho* was set by
the authors. Here the universe itself is drawn at random, so that the result
does not depend on one design:

  * K lineages. Lineage k has a raw schema bK_raw, a clean schema bK_clean
    and, with probability 1/2, a third stage bK_feat. The report needs
    R = min(K, 4) of them; the other K - R are auxiliary lineages (tools that
    exist in the catalogue but are not needed by this goal).
  * Each lineage has 1-2 dedicated fetchers (instruction -> bK_raw) and 1-2
    processing tasks per stage.
  * G = max(1, K // 3) generic feeds gJ_raw; each substitutes for 1-3 required
    raw schemas with rho* ~ U(0.5, 0.9).
  * Each auxiliary raw schema substitutes, with probability 0.6, for one
    required raw schema with rho* ~ U(0.3, 0.8); weak cross substitutions
    among required lineages have rho* ~ U(0.1, 0.4).
  * Growing K therefore grows the catalogue, the alternatives and the
    substitution search space (|S|^2 pairs), not the number of AND inputs.
  * Look-alike traps: with probability 1/2 a branch has a task
    bK_raw -> bK_digest with a high declared success rate and
    rho*(bK_digest -> bK_clean) ~ U(0.05, 0.2).
  * Branch outputs merge hierarchically (groups of <= 3) into section schemas,
    then into report_draft and final_report (AND junctions).
  * Execution probabilities: fetchers p ~ U(0.6, 0.95), processing tasks
    p ~ U(0.85, 0.99); grades observed/declared/inferred with probabilities
    0.7/0.2/0.1 (per-branch reliability stays high enough that completion does
    not vanish as K grows).

Because exact DP is infeasible beyond ~16 tasks, realized completion is
estimated by Monte Carlo; ECE and Brier are computed from (claimed value at a
decision state, eventual episode outcome) pairs.

Arms
----
  lcb        SH-GD/SM-L, posterior LCB (Eq.19), learned p_hat
  mean       SH-GD/SM-L, posterior mean + theta = 0.12 (Eq.4, Eq.18)
  exact      exact identity match only (learned p_hat), dynamic
  fixed      learned rel (LCB), plan fixed at S0 (-dynamic)
  surface    delegated-agent proxy: substitutability read from schema names
             (generic feed -> any raw; same-branch digest -> clean), rel = 0.5,
             no memory across episodes
  oracle     true rho* and true p_hat, same one-step policy (reference)
Transfer: after training, one dedicated fetcher of a branch covered by a
generic feed is withdrawn and the learned beliefs are reused unchanged.

License: MIT
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, ".")
import shgdsm as M

GRADES = ("observed", "declared", "inferred")


def generate(K: int, seed: int, G: Optional[int] = None) -> dict:
    rng = random.Random(seed)
    G = G if G is not None else max(1, K // 3)
    schemas = ["instruction"]
    tasks: List[M.Task] = []
    rho: Dict[Tuple[str, str], float] = {}
    finals = []

    def g() -> str:
        return rng.choices(GRADES, weights=(0.7, 0.2, 0.1))[0]

    def p(lo=0.8, hi=0.99) -> float:
        return round(rng.uniform(lo, hi), 3)

    R = min(K, 4)                         # lineages the report actually needs
    branch_has_alt_source: Dict[int, bool] = {}
    for k in range(K):
        raw, clean = f"b{k}_raw", f"b{k}_clean"
        schemas += [raw, clean]
        for a in range(rng.randint(1, 2)):
            tasks.append(M.Task(f"Fetch_b{k}_{a}", ("instruction",), (raw,), p(0.6, 0.95), (g(),)))
        for a in range(rng.randint(1, 2)):
            tasks.append(M.Task(f"Clean_b{k}_{a}", (raw,), (clean,), p(0.85, 0.99), (g(),)))
        last = clean
        if rng.random() < 0.5:
            feat = f"b{k}_feat"
            schemas.append(feat)
            tasks.append(M.Task(f"Feat_b{k}", (clean,), (feat,), p(0.85, 0.99), (g(),)))
            last = feat
        if rng.random() < 0.5:
            dig = f"b{k}_digest"
            schemas.append(dig)
            tasks.append(M.Task(f"Trap_b{k}", (raw,), (dig,), p(0.85, 0.98), (g(),)))
            rho[(dig, clean)] = round(rng.uniform(0.05, 0.2), 3)
        if k < R:
            finals.append(last)
        branch_has_alt_source[k] = False
    for j in range(G):
        gr = f"g{j}_raw"
        schemas.append(gr)
        tasks.append(M.Task(f"FetchFeed_g{j}", ("instruction",), (gr,), p(0.6, 0.9), (g(),)))
        for k in rng.sample(range(R), min(R, rng.randint(1, 3))):
            rho[(gr, f"b{k}_raw")] = round(rng.uniform(0.5, 0.9), 3)
            branch_has_alt_source[k] = True
    # auxiliary lineages (k >= R) are not needed by the report, but their raw
    # schema can partially substitute for one required raw schema
    for k in range(R, K):
        if rng.random() < 0.6:
            j = rng.randrange(R)
            rho[(f"b{k}_raw", f"b{j}_raw")] = round(rng.uniform(0.3, 0.8), 3)
            branch_has_alt_source[j] = True
    # weak cross substitutions among required lineages
    for _ in range(max(1, R // 2)):
        if R > 1:
            i, j = rng.sample(range(R), 2)
            rho[(f"b{i}_raw", f"b{j}_raw")] = round(rng.uniform(0.1, 0.4), 3)
    # hierarchical AND merge
    level, n = finals, 0
    while len(level) > 3:
        nxt = []
        for c in range(0, len(level), 3):
            grp = tuple(level[c:c + 3])
            sec = f"section{n}"
            n += 1
            schemas.append(sec)
            tasks.append(M.Task(f"Merge_{sec}", grp, (sec,), p(0.85, 0.98), tuple("observed" for _ in grp)))
            nxt.append(sec)
        level = nxt
    schemas += ["report_draft", "final_report"]
    tasks.append(M.Task("ComposeReport", tuple(level), ("report_draft",), p(0.85, 0.98),
                        tuple("observed" for _ in level)))
    tasks.append(M.Task("FinalizeReport", ("report_draft",), ("final_report",), p(0.85, 0.98),
                        ("observed",)))
    # a dedicated fetcher that can be withdrawn while a generic feed still covers the branch
    cands = [k for k in range(R) if branch_has_alt_source[k]]
    wk = rng.choice(cands) if cands else 0
    withdraw = [t.name for t in tasks if t.name.startswith(f"Fetch_b{wk}_")]
    # longest production chain instruction -> final_report (sets the recursion
    # depth cap, as the reference universe's cap of 4 equals its longest chain)
    depth = {"instruction": 0}
    for t in tasks:                       # tasks are generated in topological order
        d = 1 + max(depth.get(s, 0) for s in t.inputs)
        for o in t.outputs:
            depth[o] = max(depth.get(o, 0), d)
    return dict(schemas=schemas, tasks=tasks, rho=rho, withdraw=withdraw, K=K, seed=seed,
                depth=depth["final_report"])


def surface_table(c: float = 0.5) -> M.Snapshot:
    def plaus(x: str, s: str) -> bool:
        if x.startswith("g") and x.endswith("_raw") and s.endswith("_raw"):
            return True
        if x.endswith("_digest") and s.endswith("_clean") and x.split("_")[0] == s.split("_")[0]:
            return True
        return False
    return M._table(lambda x, s: c if plaus(x, s) else 0.0)


def run_cell(args: Tuple[int, int, int, int]) -> dict:
    K, useed, train_eps, eval_eps = args
    U = generate(K, useed)
    M.set_universe(U["schemas"], U["tasks"], U["rho"])
    cfg = M.cfg_lcb(max_depth=U["depth"])
    cm = M.cfg_mean(max_depth=U["depth"])
    t0 = time.time()
    b = M.train(cfg, train_eps, 1000 + useed)
    t_train = time.time() - t0
    rng = random.Random(7 + useed)
    snap = b.snapshot_mean()
    res: Dict[str, dict] = {}

    def ev(name: str, pol: M.Policy, c: M.Config) -> None:
        r = M.evaluate_policy_mc(pol, c, eval_eps, random.Random(rng.random()))
        res[name] = r

    ev("lcb", M.MyopicPolicy(snap, cfg), cfg)
    ev("mean", M.MyopicPolicy(M.clone_belief(b, cm).snapshot_mean(), cm), cm)
    ev("exact", M.MyopicPolicy(M.with_phat(M.exact_rel_table(), snap.phat), cfg), cfg)
    ev("fixed", M.FixedPlanPolicy(snap, cfg), cfg)
    ev("surface", M.MyopicPolicy(surface_table(), cfg), cfg)
    ev("oracle", M.MyopicPolicy(M.with_phat(M.oracle_rel_table(), M.true_phat(cfg)), cfg), cfg)
    errs = b.rel_errors(use_point=True)
    # transfer: withdraw the dedicated fetchers of one covered branch
    saved = M.get_universe()
    M.set_universe(saved[0], [t for t in saved[1] if t.name not in U["withdraw"]], saved[2])
    tr: Dict[str, dict] = {}
    snap2 = b.snapshot_mean()
    for name, pol, c in (
            ("lcb", M.MyopicPolicy(snap2, cfg), cfg),
            ("mean", M.MyopicPolicy(M.clone_belief(b, cm).snapshot_mean(), cm), cm),
            ("exact", M.MyopicPolicy(M.with_phat(M.exact_rel_table(), snap2.phat), cfg), cfg),
            ("surface", M.MyopicPolicy(surface_table(), cfg), cfg),
            ("oracle", M.MyopicPolicy(M.with_phat(M.oracle_rel_table(), M.true_phat(cfg)), cfg), cfg)):
        tr[name] = M.evaluate_policy_mc(pol, c, eval_eps, random.Random(rng.random()))
    M.set_universe(*saved)
    return dict(K=K, universe=useed, n_tasks=len(U["tasks"]), n_schemas=len(U["schemas"]),
                n_nonzero_pairs=len(U["rho"]), train_seconds=t_train,
                results=res, transfer=tr, rel_err=errs, withdrawn=U["withdraw"])


def summarize(cells: List[dict]) -> dict:
    out: Dict[str, dict] = {}
    for K in sorted({c["K"] for c in cells}):
        cs = [c for c in cells if c["K"] == K]
        row = dict(K=K, universes=len(cs),
                   n_tasks=statistics.fmean(c["n_tasks"] for c in cs),
                   n_schemas=statistics.fmean(c["n_schemas"] for c in cs))
        for part in ("results", "transfer"):
            arms = cs[0][part].keys()
            row[part] = {}
            for a in arms:
                d = {}
                for k in ("real", "claim", "ece", "brier"):
                    m, h = M.ci95([c[part][a][k] for c in cs])
                    d[k], d[k + "_ci"] = m, h
                m, h = M.ci95([c[part][a]["claim"] - c[part][a]["real"] for c in cs])
                d["gap"], d["gap_ci"] = m, h
                m, h = M.ci95([c[part][a]["real"] / max(1e-9, c[part]["oracle"]["real"])
                               for c in cs if c[part]["oracle"]["real"] > 0.01])
                d["ratio_to_oracle"], d["ratio_to_oracle_ci"] = m, h
                row[part][a] = d
        for k in ("all", "nonzero", "observed"):
            row["rel_err_" + k] = M.ci95([c["rel_err"][k] for c in cs])[0]
        out[str(K)] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="3,6,10,16")
    ap.add_argument("--universes", type=int, default=10)
    ap.add_argument("--train-episodes", type=int, default=1600)
    ap.add_argument("--eval-episodes", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--out", default="results_envgen.json")
    ap.add_argument("--resume", action="store_true", help="keep finished cells in --out")
    args = ap.parse_args()
    sys.setrecursionlimit(100000)
    jobs = [(K, u, args.train_episodes, args.eval_episodes)
            for K in [int(x) for x in args.sizes.split(",")]
            for u in range(1, args.universes + 1)]
    t0 = time.time()
    cells = []
    if args.resume and os.path.exists(args.out):
        # keep finished cells, run only the missing (K, universe) pairs
        cells = json.load(open(args.out, encoding="utf-8")).get("cells", [])
        have = {(c["K"], c["universe"]) for c in cells}
        jobs = [j for j in jobs if (j[0], j[1]) not in have]
        print(f"resuming: {len(cells)} cells kept, {len(jobs)} to run", flush=True)
    with mp.Pool(args.workers) as pool:
        for n, c in enumerate(pool.imap_unordered(run_cell, jobs), 1):
            cells.append(c)
            print(f"[{n}/{len(jobs)}] K={c['K']} u={c['universe']} tasks={c['n_tasks']} "
                  + " ".join(f"{a}={100*r['real']:.1f}%" for a, r in c["results"].items())
                  + f"  ({time.time()-t0:.0f}s)", flush=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(M._jsonable(dict(config=vars(args), cells=cells,
                                           summary=summarize(cells))), f, indent=2)
    print(json.dumps(M._jsonable(summarize(cells)), indent=1)[:4000])


if __name__ == "__main__":
    main()
