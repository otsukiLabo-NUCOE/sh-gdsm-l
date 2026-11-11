#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-K (revision 2026-11): context-conditioned relation strength.

Whether x can stand in for s may depend on the consumer: a generic price feed
can replace the dedicated one for a chart but not for a volume analysis.
Every input slot is given its own schema name s@consumer (identity
rel(s -> s@consumer) = 1 is known, not learned), so that a pair is learned per
(x, s, consumer). Three estimators of rel(x -> s@c):
  free     context-free (the specified model): counts pooled over all
           consumers of s
  ctx      per consumer only
  backoff  per consumer, with a prior centred on the pooled estimate:
           Beta(a_c + n0 * m_pool, b_c + n0 * (1 - m_pool)), n0 = 5
Universes (exact DP)
  ctx-strong  rel*(g_raw -> p_raw@Chart) = 0.9, rel*(g_raw -> p_raw@Vol) = 0.1;
              the dedicated fetcher of p_raw is unreliable (p = 0.5); Vol has
              its own alternative route (FetchVolAlt -> VolFromAlt)
  ctx-strong-weak-alt  as ctx-strong, but the Vol alternative route is weaker
              (0.65 x 0.65), so a pooled estimate (~0.5) sends Vol to g_raw
  ctx-none    both consumers 0.5 (no context effect): what splitting costs
10 seeds x 1,600 training episodes, LCB planning.

    python e_context.py --out results_rev_EK.json
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
MODES = ["free", "ctx", "backoff"]
N0 = 5.0


def base(s: str) -> str:
    return s.split("@", 1)[0]


def universe(kind: str):
    T = M.Task
    alt = 0.65 if kind.endswith("weak-alt") else 0.80
    schemas = ["instruction", "p_raw", "g_raw", "p_raw@Chart", "p_raw@Vol", "va_raw",
               "chart", "vol", "report_draft", "final_report"]
    tasks = [T("FetchP", ("instruction",), ("p_raw",), 0.50, ("observed",)),
             T("FetchG", ("instruction",), ("g_raw",), 0.90, ("observed",)),
             T("Chart", ("p_raw@Chart",), ("chart",), 0.95, ("observed",)),
             T("Vol", ("p_raw@Vol",), ("vol",), 0.95, ("observed",)),
             T("FetchVolAlt", ("instruction",), ("va_raw",), alt, ("observed",)),
             T("VolFromAlt", ("va_raw",), ("vol",), alt, ("observed",)),
             T("Compose", ("chart", "vol"), ("report_draft",), 0.90, ("observed", "observed")),
             T("Finalize", ("report_draft",), ("final_report",), 0.95, ("observed",))]
    hi, lo = (0.9, 0.1) if kind.startswith("ctx-strong") else (0.5, 0.5)
    rho = {("p_raw", "p_raw@Chart"): 1.0, ("p_raw", "p_raw@Vol"): 1.0,
           ("g_raw", "p_raw@Chart"): hi, ("g_raw", "p_raw@Vol"): lo}
    return schemas, tasks, rho


class CtxBelief(M.RelBelief):
    def __init__(self, cfg, mode: str):
        super().__init__(cfg)
        self.mode = mode

    def _known(self, x, s):
        return x == s or x == base(s)

    def observe(self, x, s, passed):
        if self._known(x, s):
            return
        super().observe(x, s, passed)

    def _pool(self, x, s):
        a = b = 0.0
        for (x2, s2), v in self.a.items():
            if x2 == x and base(s2) == base(s):
                a += v
        for (x2, s2), v in self.b.items():
            if x2 == x and base(s2) == base(s):
                b += v
        return a, b

    def _ab(self, x, s):
        own = (self.a.get((x, s), 0), self.b.get((x, s), 0))
        if self.mode == "ctx":
            return own
        pa, pb = self._pool(x, s)
        if self.mode == "free":
            return pa, pb
        m = (pa + self.cfg.alpha) / (pa + pb + self.cfg.alpha + self.cfg.beta)
        # backoff: replace the Beta(alpha, beta) prior by n0 pseudo-counts at m
        return (own[0] + N0 * m - self.cfg.alpha, own[1] + N0 * (1 - m) - self.cfg.beta)

    def mean(self, x, s):
        return 1.0 if self._known(x, s) else super().mean(x, s)

    def lcb(self, x, s):
        if self._known(x, s):
            return 1.0
        a, b = self._ab(x, s)
        return M._beta_ppf(self.cfg.lcb_delta, max(a + self.cfg.alpha, 1e-3), max(b + self.cfg.beta, 1e-3))

    def sample(self, x, s, rng):
        if self._known(x, s):
            return 1.0
        a, b = self._ab(x, s)
        return rng.betavariate(max(a + self.cfg.alpha, 1e-3), max(b + self.cfg.beta, 1e-3))

    def _snap(self, f, ef):
        d = M.Snapshot()
        for p in M.LEARNABLE_PAIRS:
            d[p] = f(*p)
        for s in M.SCHEMAS:
            d[(s, s)] = 1.0
        d.phat = {t.name: ef(t) for t in M.TASKS}
        return d


def cell(args):
    kind, seed, train_eps = args
    s, t, r = universe(kind)
    M.set_universe(s, t, r)
    cfg = M.cfg_lcb(max_depth=5)
    out = []
    for mode in MODES:
        rng = random.Random(seed)
        b = CtxBelief(cfg, mode)
        for _ in range(train_eps):
            M.run_episode(b, cfg, rng, explore=True)
        res = M.full_eval(b.snapshot_mean(), cfg)
        out.append(dict(universe=kind, seed=seed, mode=mode, real=res["real"], claim=res["claim"],
                        gap=res["claim"] - res["real"], ece=res["ece"],
                        rel_chart=b.mean("g_raw", "p_raw@Chart"), rel_vol=b.mean("g_raw", "p_raw@Vol")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--train-episodes", type=int, default=1600)
    ap.add_argument("--out", default="results_rev_EK.json")
    a = ap.parse_args()
    t0 = time.time()
    rows = []
    jobs = [(k, sd, a.train_episodes) for k in ("ctx-strong", "ctx-strong-weak-alt", "ctx-none") for sd in SEEDS]
    with mp.Pool(a.workers) as pool:
        for rr in pool.imap_unordered(cell, jobs):
            rows += rr
            json.dump(dict(rows=rows), open(a.out, "w", encoding="utf-8"), indent=1)
    from scipy.stats import wilcoxon
    summ = {}
    for k in ("ctx-strong", "ctx-strong-weak-alt", "ctx-none"):
        base_ = {r["seed"]: r for r in rows if r["universe"] == k and r["mode"] == "free"}
        for mode in MODES:
            xs = {r["seed"]: r for r in rows if r["universe"] == k and r["mode"] == mode}
            d = [xs[s]["real"] - base_[s]["real"] for s in sorted(xs)]
            f = lambda key: statistics.fmean(x[key] for x in xs.values())
            summ[f"{k} | {mode}"] = dict(real=f("real"), gap=f("gap"), ece=f("ece"),
                                         rel_chart=f("rel_chart"), rel_vol=f("rel_vol"),
                                         diff_vs_free=statistics.fmean(d),
                                         wilcoxon_p=(float(wilcoxon(d).pvalue)
                                                     if mode != "free" and any(abs(v) > 1e-12 for v in d)
                                                     else None))
    json.dump(dict(rows=rows, summary=summ), open(a.out, "w", encoding="utf-8"), indent=1)
    for k, v in summ.items():
        print(f"{k:<22} real={100*v['real']:.2f} gap={100*v['gap']:+.2f} ECE={v['ece']:.3f} "
              f"rel(chart)={v['rel_chart']:.2f} rel(vol)={v['rel_vol']:.2f} "
              f"diff={100*v['diff_vs_free']:+.2f} p={v['wilcoxon_p']}")
    print(f"wrote {a.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
