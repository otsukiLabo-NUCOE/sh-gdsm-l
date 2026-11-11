#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-X (revision 2026-11): scalability of the planner.

Random universes (envgen.py) from ~40 to ~500 tools. For every universe and
every depth cap D of the recursive value estimate (Eq.9-12) we measure
  * decision latency of one act() (ms, mean and 95th percentile),
  * the size of the memo table and the peak working set of the process,
  * the realized completion (Monte Carlo) relative to the deepest cap, i.e.
    what a shallower cap costs,
  * the number of schema pairs actually stored after a short training run
    (sparse storage) against |S|^2.
The policy plans on the true rel / p_hat table (oracle), so that the timing
reflects the planner, not the state of learning; the stored-pair count comes
from 50 training episodes with the learner itself.

Every cell is written to --out as soon as it finishes (resume with --resume).

    python e_scale.py --out results_rev_EX.json --resume
"""
from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import time

sys.path.insert(0, ".")
import shgdsm as M
import envgen as E

SIZES = [8, 16, 24, 32, 48, 64, 80, 110]
UNIVERSES = [1, 2, 3]
DEPTHS = [2, 3, 4, 6]
TIME_CAP = 900.0          # seconds of evaluation per cell


def peak_mb() -> float:
    if os.name != "nt":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    f = ctypes.windll.psapi.GetProcessMemoryInfo
    f.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    f(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2 ** 20


def cell(args):
    K, u, D, eval_eps = args
    sys.setrecursionlimit(200000)
    U = E.generate(K, u)
    M.set_universe(U["schemas"], U["tasks"], U["rho"])
    cfg = M.cfg_lcb(max_depth=D)
    rel = M.with_phat(M.oracle_rel_table(), M.true_phat(cfg))
    pol = M.MyopicPolicy(rel, cfg)
    rng = random.Random(500 + u)
    lat, wins, eps, max_memo = [], 0, 0, 0
    t_start = time.time()
    while eps < eval_eps and time.time() - t_start < TIME_CAP:
        st = M.State()
        for _ in range(M.N_TASKS):
            if st.reached_goal():
                break
            t0 = time.perf_counter()
            i = pol.act(st)
            lat.append(time.perf_counter() - t0)
            max_memo = max(max_memo, len(pol.est._memo))
            if i is None:
                break
            qe, qv, _ = M.true_exec_prob(i, st, pol.est, cfg)
            ok = rng.random() < qe and rng.random() < qv
            st = st.with_success(i) if ok else st.with_failure(i)
        wins += int(st.reached_goal())
        eps += 1
    out = dict(K=K, universe=u, depth_cap=D, n_tools=len(U["tasks"]), n_schemas=len(U["schemas"]),
               longest_chain=U["depth"], episodes=eps, decisions=len(lat),
               completion=wins / max(1, eps),
               latency_ms_mean=1000 * statistics.fmean(lat) if lat else None,
               latency_ms_p95=(1000 * sorted(lat)[int(0.95 * (len(lat) - 1))] if lat else None),
               memo_entries_max=max_memo, peak_mb=peak_mb())
    if D == 3:                     # sparse pair storage after 50 training episodes
        lc = M.cfg_lcb(max_depth=3)
        b = M.RelBelief(lc)
        trng = random.Random(900 + u)
        t0 = time.time()
        for _ in range(50):
            M.run_episode(b, lc, trng, explore=True)
        out.update(train_s_per_episode=(time.time() - t0) / 50,
                   stored_pairs=len(set(b.a) | set(b.b)),
                   possible_pairs=len(M.LEARNABLE_PAIRS))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=40)
    ap.add_argument("--out", default="results_rev_EX.json")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    rows = []
    if a.resume and os.path.exists(a.out):
        rows = json.load(open(a.out, encoding="utf-8")).get("rows", [])
    done = {(r["K"], r["universe"], r["depth_cap"]) for r in rows}
    jobs = [(K, u, D, a.eval_episodes) for K in SIZES for u in UNIVERSES for D in DEPTHS
            if (K, u, D) not in done]
    t0 = time.time()
    print(f"{len(jobs)} cells to run ({len(done)} kept)", flush=True)
    with mp.Pool(a.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(cell, jobs):
            rows.append(r)
            json.dump(dict(rows=rows), open(a.out, "w", encoding="utf-8"), indent=1)
            print(f"K={r['K']:>3} u={r['universe']} D={r['depth_cap']} tools={r['n_tools']:>3} "
                  f"lat={r['latency_ms_mean']:.1f}ms p95={r['latency_ms_p95']:.1f}ms "
                  f"completion={r['completion']:.2f} eps={r['episodes']} peak={r['peak_mb']:.0f}MB "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
