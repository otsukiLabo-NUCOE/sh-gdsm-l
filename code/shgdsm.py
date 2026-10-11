#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-GD/SM-L  reference implementation  (rev.4)
=============================================
Changes relative to rev.3 (shgdsm_rev3.py):

  (R1) The planning-time point estimate defaults to the posterior lower
       confidence bound (manuscript Eq.19, delta = 0.05); the posterior mean
       with theta = 0.12 (Eq.4 + Eq.18) is kept as the comparison setting.
  (R2) Execution reliability p_hat_e is learned from execution outcomes
       (Beta posterior per task, prior centred on the declared p_e * w(g)
       with strength cfg.exec_prior_n). w(g) is therefore only a prior, and
       the calibration of the exact-match model is no longer zero by
       construction (manuscript Sec. III-C).
  (R3) Calibration is measured over every decision state, not only at S0:
       reach-weighted expected calibration error (ECE) and Brier score,
       computed exactly by DP for small universes and by Monte Carlo for
       large ones.
  (R4) rel estimation error is reported over all pairs, over pairs with a
       non-zero true value, and over observed pairs.
  (R5) The task universe is swappable (set_universe), so that the same model
       code runs on the randomly generated universes of envgen.py.

Equation numbers in comments refer to the manuscript (IEEE Access version).

All probabilities reported as "realized" on the reference universe are
computed by exact dynamic programming over the reachable state space.

License: MIT
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 1. Task universe (reference universe of Experiments 1-3; swappable)
# ---------------------------------------------------------------------------

GRADE_W_DEFAULT = {"observed": 1.00, "declared": 0.80, "inferred": 0.50}


@dataclass(frozen=True)
class Task:
    name: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    p: float                      # true execution success probability
    grades: Tuple[str, ...]       # one grade per input dependency edge
    cost: float = 1.0

    def w(self, gw: Dict[str, float]) -> float:
        prod = 1.0
        for g in self.grades:
            prod *= gw[g]
        return prod

    def p_hat(self, gw: Dict[str, float]) -> float:
        """Grade-discounted declared execution probability (Eq.2)."""
        return self.p * self.w(gw)


REF_SCHEMAS = (
    "instruction",
    "stock_raw", "econ_raw", "market_raw", "news_raw",
    "stock_clean", "econ_clean", "news_summary", "news_digest",
    "report_draft", "final_report",
)
REF_TASKS: Tuple[Task, ...] = (
    Task("FetchStockA",       ("instruction",), ("stock_raw",),    0.80, ("observed",)),
    Task("FetchStockB",       ("instruction",), ("stock_raw",),    0.60, ("declared",)),
    Task("FetchEcon",         ("instruction",), ("econ_raw",),     0.70, ("observed",)),
    Task("FetchMarketFeed",   ("instruction",), ("market_raw",),   0.75, ("inferred",)),
    Task("FetchNews",         ("instruction",), ("news_raw",),     0.90, ("observed",)),
    Task("FetchNewsAlt",      ("instruction",), ("news_raw",),     0.45, ("inferred",)),
    Task("CleanStock",        ("stock_raw",),   ("stock_clean",),  0.95, ("observed",)),
    Task("CleanEcon",         ("econ_raw",),    ("econ_clean",),   0.90, ("observed",)),
    Task("SummarizeNews",     ("news_raw",),    ("news_summary",), 0.85, ("observed",)),
    Task("SummarizeNewsTrap", ("news_raw",),    ("news_digest",),  0.90, ("declared",)),
    Task("ComposeReport",     ("stock_clean", "econ_clean", "news_summary"),
                              ("report_draft",), 0.85, ("observed", "observed", "observed")),
    Task("FinalizeReport",    ("report_draft",), ("final_report",), 0.90, ("observed",)),
)
REF_RHO: Dict[Tuple[str, str], float] = {
    ("market_raw",  "stock_raw"):    0.85,
    ("market_raw",  "econ_raw"):     0.70,
    ("econ_raw",    "stock_raw"):    0.30,
    ("stock_raw",   "econ_raw"):     0.25,
    ("news_digest", "news_summary"): 0.15,
}

# module-level view of the current universe (rebound by set_universe)
SCHEMAS: Tuple[str, ...] = ()
TASKS: Tuple[Task, ...] = ()
RHO_STAR: Dict[Tuple[str, str], float] = {}
GOAL = "final_report"
INITIAL: FrozenSet[str] = frozenset({"instruction"})
TASK_INDEX: Dict[str, int] = {}
N_TASKS = 0
LEARNABLE_PAIRS: Tuple[Tuple[str, str], ...] = ()
_SIGMA_CACHE: Dict[int, FrozenSet[str]] = {}


def set_universe(schemas: Sequence[str], tasks: Sequence[Task],
                 rho: Dict[Tuple[str, str], float], goal: str = "final_report",
                 initial: FrozenSet[str] = frozenset({"instruction"})) -> None:
    global SCHEMAS, TASKS, RHO_STAR, GOAL, INITIAL, TASK_INDEX, N_TASKS, LEARNABLE_PAIRS
    SCHEMAS = tuple(schemas)
    TASKS = tuple(tasks)
    RHO_STAR = dict(rho)
    GOAL = goal
    INITIAL = frozenset(initial)
    TASK_INDEX = {t.name: i for i, t in enumerate(TASKS)}
    N_TASKS = len(TASKS)
    LEARNABLE_PAIRS = tuple((x, s) for x in SCHEMAS for s in SCHEMAS if x != s)
    _SIGMA_CACHE.clear()


def get_universe() -> tuple:
    return (SCHEMAS, TASKS, RHO_STAR, GOAL, INITIAL)


def use_reference_universe() -> None:
    set_universe(REF_SCHEMAS, REF_TASKS, REF_RHO)


use_reference_universe()


def rho_star(x: str, s: str) -> float:
    if x == s:
        return 1.0
    return RHO_STAR.get((x, s), 0.0)


# ---------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    alpha: float = 1.0
    beta: float = 9.0
    # theta: under the LCB estimator (default) it is only a pruning constant;
    # under the posterior mean it must satisfy Eq.18 (theta > alpha/(alpha+beta)).
    theta: float = 0.01
    lam: float = 0.05                 # cost coefficient lambda (Eq.15)
    grade_w: Dict[str, float] = field(default_factory=lambda: dict(GRADE_W_DEFAULT))
    # Environment semantics of w(g) (Sec. III-C). With learn_exec=True the
    # agent's p_hat is learned, so either reading is calibrated after training.
    env_realizes_w: bool = True
    max_depth: int = 4
    top_k: int = 4
    theta_in_eq5: bool = True
    # Planning-time point estimate of rel: "lcb" (Eq.19, default) or "mean" (Eq.4)
    point_estimate: str = "lcb"
    lcb_delta: float = 0.05
    # (R2) learned execution reliability
    learn_exec: bool = True
    exec_prior_n: float = 10.0
    # Candidate threshold applied to Thompson SAMPLES during training (Eq.17).
    # Identical for both estimators, so the two differ only at planning time.
    theta_explore: float = 0.12


def cfg_mean(**kw) -> Config:
    """The rev.3 setting: posterior mean + theta = 0.12 (Eq.4, Eq.18)."""
    base = dict(point_estimate="mean", theta=0.12)
    base.update(kw)
    return Config(**base)


def cfg_lcb(**kw) -> Config:
    base = dict(point_estimate="lcb", theta=0.01)
    base.update(kw)
    return Config(**base)


def explore_cfg(cfg: Config) -> Config:
    """Planning configuration used on a Thompson sample (training only)."""
    return cfg_replace(cfg, theta=cfg.theta_explore)


def cfg_replace(cfg: Config, **kw) -> Config:
    d = dict(cfg.__dict__)
    d.update(kw)
    return Config(**d)


# ---------------------------------------------------------------------------
# 3. Beliefs: relation strength (Eq.4/17/19) and execution reliability (Eq.3)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1 << 18)
def _beta_ppf(q: float, a: float, b: float) -> float:
    from scipy.stats import beta as _b
    return float(_b.ppf(q, a, b))


class Snapshot(dict):
    """A frozen rel table {(x, s): value}; .phat carries the execution
    reliability table {task_name: value} (None -> declared p_e * w(g))."""
    phat: Optional[Dict[str, float]] = None


class RelBelief:
    """Beta posteriors over rel(x -> s) for every learnable schema pair, and
    over the execution success of every task."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.a: Dict[Tuple[str, str], int] = {}
        self.b: Dict[Tuple[str, str], int] = {}
        self.ea: Dict[str, int] = {}      # execution successes per task
        self.eb: Dict[str, int] = {}      # execution failures per task

    # -- observations -------------------------------------------------------
    def observe(self, x: str, s: str, passed: bool) -> None:
        if x == s:
            return
        d = self.a if passed else self.b
        d[(x, s)] = d.get((x, s), 0) + 1

    def observe_exec(self, task: str, ok: bool) -> None:
        d = self.ea if ok else self.eb
        d[task] = d.get(task, 0) + 1

    def n(self, x: str, s: str) -> int:
        return self.a.get((x, s), 0) + self.b.get((x, s), 0)

    def _ab(self, x: str, s: str) -> Tuple[int, int]:
        return self.a.get((x, s), 0), self.b.get((x, s), 0)

    # -- rel point estimates -------------------------------------------------
    def mean(self, x: str, s: str) -> float:
        """Eq.4: posterior mean."""
        if x == s:
            return 1.0
        a, b = self._ab(x, s)
        return (a + self.cfg.alpha) / (a + b + self.cfg.alpha + self.cfg.beta)

    def lcb(self, x: str, s: str) -> float:
        """Eq.19: posterior lower confidence bound at level cfg.lcb_delta."""
        if x == s:
            return 1.0
        a, b = self._ab(x, s)
        return _beta_ppf(self.cfg.lcb_delta, a + self.cfg.alpha, b + self.cfg.beta)

    def sample(self, x: str, s: str, rng: random.Random) -> float:
        """Eq.17: one draw from the posterior (Thompson sampling)."""
        if x == s:
            return 1.0
        a, b = self._ab(x, s)
        return rng.betavariate(a + self.cfg.alpha, b + self.cfg.beta)

    def point(self, x: str, s: str) -> float:
        return self.lcb(x, s) if self.cfg.point_estimate == "lcb" else self.mean(x, s)

    # -- execution reliability ---------------------------------------------
    def _exec_prior(self, t: Task) -> Tuple[float, float]:
        m = min(max(t.p_hat(self.cfg.grade_w), 1e-3), 1 - 1e-3)
        n0 = self.cfg.exec_prior_n
        return m * n0, (1 - m) * n0

    def exec_mean(self, t: Task) -> float:
        if not self.cfg.learn_exec:
            return t.p_hat(self.cfg.grade_w)
        a0, b0 = self._exec_prior(t)
        a, b = self.ea.get(t.name, 0), self.eb.get(t.name, 0)
        return (a + a0) / (a + b + a0 + b0)

    def exec_sample(self, t: Task, rng: random.Random) -> float:
        if not self.cfg.learn_exec:
            return t.p_hat(self.cfg.grade_w)
        a0, b0 = self._exec_prior(t)
        return rng.betavariate(self.ea.get(t.name, 0) + a0, self.eb.get(t.name, 0) + b0)

    # -- frozen snapshots --------------------------------------------------
    def _snap(self, f, ef) -> Snapshot:
        d = Snapshot()
        # unobserved pairs share one value; compute it once
        un = None
        for p in LEARNABLE_PAIRS:
            if p in self.a or p in self.b:
                d[p] = f(*p)
            else:
                if un is None:
                    un = f(*p)
                d[p] = un
        for s in SCHEMAS:
            d[(s, s)] = 1.0
        d.phat = {t.name: ef(t) for t in TASKS}
        return d

    def snapshot_mean(self) -> Snapshot:
        """Planning snapshot with the configured point estimate (name kept for
        backward compatibility with the rev.3 harnesses)."""
        return self._snap(self.point, self.exec_mean)

    def snapshot_sample(self, rng: random.Random) -> Snapshot:
        d = Snapshot()
        for p in LEARNABLE_PAIRS:
            d[p] = self.sample(*p, rng)
        for s in SCHEMAS:
            d[(s, s)] = 1.0
        d.phat = {t.name: self.exec_sample(t, rng) for t in TASKS}
        return d

    # -- estimation error (R4) ---------------------------------------------
    def rel_errors(self, use_point: bool = False) -> Dict[str, float]:
        f = self.point if use_point else self.mean
        allp = [abs(f(x, s) - rho_star(x, s)) for (x, s) in LEARNABLE_PAIRS]
        nz = [abs(f(x, s) - rho_star(x, s)) for (x, s) in LEARNABLE_PAIRS
              if rho_star(x, s) > 0]
        ob = [abs(f(x, s) - rho_star(x, s)) for (x, s) in LEARNABLE_PAIRS
              if self.n(x, s) > 0]
        fm = lambda xs: statistics.fmean(xs) if xs else float("nan")
        return dict(all=fm(allp), nonzero=fm(nz), observed=fm(ob), n_observed=len(ob))

    def mean_abs_error(self) -> float:
        return self.rel_errors()["all"]


def _table(f) -> Snapshot:
    d = Snapshot()
    for p in LEARNABLE_PAIRS:
        d[p] = f(*p)
    for s in SCHEMAS:
        d[(s, s)] = 1.0
    return d


def exact_rel_table() -> Snapshot:
    """rel in {0,1}: nominal identity only (the '-relation' ablation)."""
    return _table(lambda x, s: 0.0)


def oracle_rel_table() -> Snapshot:
    return _table(rho_star)


def prior_rel_table(cfg: Config) -> Snapshot:
    m = cfg.alpha / (cfg.alpha + cfg.beta)
    if cfg.point_estimate == "lcb":
        m = _beta_ppf(cfg.lcb_delta, cfg.alpha, cfg.beta)
    return _table(lambda x, s: m)


def with_phat(rel: Snapshot, phat: Optional[Dict[str, float]]) -> Snapshot:
    out = Snapshot(rel)
    out.phat = phat
    return out


def true_phat(cfg: Config) -> Dict[str, float]:
    """The environment's actual execution reliability (oracle p_hat)."""
    return {t.name: t.p * (t.w(cfg.grade_w) if cfg.env_realizes_w else 1.0) for t in TASKS}


# ---------------------------------------------------------------------------
# 4. State
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class State:
    done: int = 0      # bitmask of successfully executed tasks
    failed: int = 0    # bitmask of failed tasks (F)

    def sigma(self) -> FrozenSet[str]:
        hit = _SIGMA_CACHE.get(self.done)
        if hit is not None:
            return hit
        acc = set(INITIAL)
        d, i = self.done, 0
        while d:
            if d & 1:
                acc.update(TASKS[i].outputs)
            d >>= 1
            i += 1
        val = frozenset(acc)
        if len(_SIGMA_CACHE) > 200_000:      # bounded: large universes
            _SIGMA_CACHE.clear()
        _SIGMA_CACHE[self.done] = val
        return val

    def used(self) -> int:
        return self.done | self.failed

    def with_success(self, i: int) -> "State":
        return State(self.done | (1 << i), self.failed)

    def with_failure(self, i: int) -> "State":
        return State(self.done, self.failed | (1 << i))

    def reached_goal(self) -> bool:
        return GOAL in self.sigma()


# ---------------------------------------------------------------------------
# 5. Claimed completion probability (Eq.5 - Eq.11)
# ---------------------------------------------------------------------------


class Estimator:
    """Computes the agent's claimed P(Goal) from a frozen rel table."""

    def __init__(self, rel: Dict[Tuple[str, str], float], cfg: Config):
        self.rel = rel
        self.cfg = cfg
        self.gw = cfg.grade_w
        ph = getattr(rel, "phat", None)
        self.phat: List[float] = [
            (ph[t.name] if ph is not None and t.name in ph else t.p_hat(self.gw))
            for t in TASKS]
        self._memo: Dict[Tuple, float] = {}
        self._masks: Dict[Tuple[str, int], int] = {}
        self._cand: Dict[str, Tuple[Tuple[int, float], ...]] = {}
        for s in SCHEMAS:
            rows = []
            for i, t in enumerate(TASKS):
                v = self.phat[i] * max(rel[(o, s)] for o in t.outputs)
                if v >= cfg.theta:
                    rows.append((i, v))
            rows.sort(key=lambda r: -r[1])
            self._cand[s] = tuple(rows[: cfg.top_k])

    def rho(self, t: Task, s: str) -> float:                     # Eq.5
        return max(self.rel[(o, s)] for o in t.outputs)

    def str_(self, t: Task, s: str) -> float:                    # Eq.6
        return self.phat[TASK_INDEX[t.name]] * self.rho(t, s)

    def p_from_sigma(self, s: str, sigma: FrozenSet[str]) -> float:   # Eq.7
        th = self.cfg.theta if self.cfg.theta_in_eq5 else 0.0
        best = 0.0
        for x in sigma:
            v = self.rel[(x, s)]
            if v >= th and v > best:
                best = v
        return best

    def _mask(self, s: str, depth: int) -> int:
        """Bitmask of every task that can appear in the recursion below
        P(s; V) at this depth. Only those tasks' used/unused status can change
        the value, so the memo key is restricted to them (exact, and it lets
        states that differ elsewhere share work)."""
        key = (s, depth)
        hit = self._masks.get(key)
        if hit is not None:
            return hit
        self._masks[key] = 0                       # cycle guard
        m = 0
        if depth > 0:
            for i, _ in self._cand[s]:
                m |= 1 << i
                for s2 in TASKS[i].inputs:
                    m |= self._mask(s2, depth - 1)
        self._masks[key] = m
        return m

    def p_schema(self, s: str, st: State, visited: FrozenSet[str], depth: int) -> float:
        """Eq.9 with the visited-set base case Eq.11."""
        key = (s, st.sigma(), st.used() & self._mask(s, depth), visited, depth)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        base = self.p_from_sigma(s, st.sigma())
        if s in visited or depth <= 0 or base >= 1.0:
            self._memo[key] = base
            return base
        prod = 1.0
        nxt = visited | {s}
        used = st.used()
        for i, st_es in self._cand[s]:
            if used >> i & 1:
                continue
            p_in = self.p_inputs(TASKS[i], st, nxt, depth - 1)
            prod *= (1.0 - st_es * p_in)
        val = 1.0 - (1.0 - base) * prod
        if len(self._memo) > 300_000:       # bounded cache (values are exact either way)
            self._memo.clear()
        self._memo[key] = val
        return val

    def p_inputs(self, t: Task, st: State, visited: FrozenSet[str], depth: int) -> float:
        prod = 1.0                                               # Eq.10
        for s in t.inputs:
            prod *= self.p_schema(s, st, visited, depth)
            if prod == 0.0:
                break
        return prod

    def p_goal(self, st: State) -> float:                        # Eq.12
        if st.reached_goal():
            return 1.0
        return self.p_schema(GOAL, st, frozenset(), self.cfg.max_depth)

    def best_substitution(self, s: str, sigma: FrozenSet[str]) -> Tuple[Optional[str], float]:
        best_x, best_v = None, 0.0
        for x in sorted(sigma):
            v = self.rel[(x, s)]
            if v > best_v:
                best_x, best_v = x, v
        return best_x, best_v

    def ready(self, st: State) -> List[int]:
        sigma = st.sigma()
        out = []
        used = st.used()
        for i, t in enumerate(TASKS):
            if used >> i & 1:
                continue
            ok = True
            for s in t.inputs:
                _, v = self.best_substitution(s, sigma)
                if v < self.cfg.theta:
                    ok = False
                    break
            if ok:
                out.append(i)
        return out

    def p_exec(self, i: int, st: State) -> float:                # Eq.14
        t = TASKS[i]
        sigma = st.sigma()
        v = self.phat[i]
        for s in t.inputs:
            _, r = self.best_substitution(s, sigma)
            v *= r
        return v


# ---------------------------------------------------------------------------
# 6. Policies
# ---------------------------------------------------------------------------


class Policy:
    name = "policy"

    def act(self, st: State) -> Optional[int]:
        raise NotImplementedError


class MyopicPolicy(Policy):
    """Eq.15: one-step lookahead on the Eq.9/Eq.10 value estimate."""

    def __init__(self, rel, cfg: Config, name: str = "myopic"):
        self.est = Estimator(rel, cfg)
        self.cfg = cfg
        self.name = name
        self._act: Dict[Tuple[int, int], Optional[int]] = {}

    def scores(self, st: State) -> List[Tuple[int, float, float, float, float]]:
        rows = []
        for i in self.est.ready(st):
            t = TASKS[i]
            q = self.est.p_exec(i, st)
            u = (q * self.est.p_goal(st.with_success(i))
                 + (1.0 - q) * self.est.p_goal(st.with_failure(i))
                 - self.cfg.lam * t.cost)
            rows.append((i, q, self.est.phat[i], q, u))
        rows.sort(key=lambda r: (-r[4], TASKS[r[0]].name))
        return rows

    def act(self, st: State) -> Optional[int]:
        key = (st.done, st.failed)
        if key in self._act:
            return self._act[key]
        rows = self.scores(st)
        a = rows[0][0] if rows else None
        self._act[key] = a
        return a


class FixedPlanPolicy(Policy):
    """'-dynamic': a plan is computed once at S0 and executed in order."""

    def __init__(self, rel, cfg: Config, name: str = "fixed"):
        self.inner = MyopicPolicy(rel, cfg, name)
        self.est = self.inner.est
        self.name = name
        self.plan = self._build()

    def _build(self) -> List[int]:
        st = State()
        plan: List[int] = []
        for _ in range(N_TASKS):
            if st.reached_goal():
                break
            i = self.inner.act(st)
            if i is None:
                break
            plan.append(i)
            st = st.with_success(i)
        return plan

    def act(self, st: State) -> Optional[int]:
        for i in self.plan:
            if st.used() >> i & 1:
                continue
            return i
        return None


class SinglePathPolicy(Policy):
    """'PropertyGraph (naive)': one exact-match path, no redundancy, no retry."""

    def __init__(self, cfg: Config, name: str = "single-path"):
        self.inner = FixedPlanPolicy(with_phat(exact_rel_table(), None), cfg, name)
        self.est = self.inner.est
        self.name = name
        self.plan = self.inner.plan

    def act(self, st: State) -> Optional[int]:
        for i in self.plan:
            if st.done >> i & 1:
                continue
            if st.failed >> i & 1:
                return None
            return i
        return None

    def claim(self, cfg: Config) -> float:
        v = 1.0
        for i in self.plan:
            v *= TASKS[i].p_hat(cfg.grade_w)
        return v


class CategoryMatchPolicy(Policy):
    """'RDF (naive)': coarse category match, no downstream schema validation.
    Categories are the processing stages: *_raw, *_clean/summary/digest, rest."""

    def __init__(self, cfg: Config, name: str = "rdf-naive"):
        def cat(x: str) -> str:
            if x.endswith("_raw"):
                return "raw"
            if x.endswith(("_clean", "_summary", "_digest")):
                return "clean"
            return x
        rel = _table(lambda x, s: 1.0 if cat(x) == cat(s) else 0.0)
        self.inner = MyopicPolicy(rel, cfg, name)
        self.est = self.inner.est
        self.name = name

    def act(self, st: State) -> Optional[int]:
        return self.inner.act(st)


# ---------------------------------------------------------------------------
# 7. Evaluation under rho*: exact DP (small universes) and Monte Carlo
# ---------------------------------------------------------------------------


def true_exec_prob(i: int, st: State, est: Estimator, cfg: Config,
                   true_p: Optional[Dict[str, float]] = None
                   ) -> Tuple[float, float, List[Tuple[str, str]]]:
    """-> (P(execution succeeds), P(all inputs validate | executed), pairs).
    true_p overrides the environment's execution reliability (used when the
    agent's w(g) prior deliberately differs from the environment's)."""
    t = TASKS[i]
    sigma = st.sigma()
    if true_p is not None:
        q_exec = true_p[t.name]
    else:
        q_exec = t.p * (t.w(cfg.grade_w) if cfg.env_realizes_w else 1.0)
    q_val = 1.0
    pairs: List[Tuple[str, str]] = []
    for s in t.inputs:
        x, _ = est.best_substitution(s, sigma)
        if x is None:
            return 0.0, 0.0, []
        pairs.append((x, s))
        q_val *= rho_star(x, s)
    return q_exec, q_val, pairs


def _ece(points: List[Tuple[float, float, float]], bins: int = 10) -> float:
    """points = [(weight, predicted, realized)] -> weighted ECE."""
    tot = sum(w for w, _, _ in points)
    if tot <= 0:
        return float("nan")
    acc = [[0.0, 0.0, 0.0] for _ in range(bins)]
    for w, p, y in points:
        k = min(bins - 1, int(p * bins))
        acc[k][0] += w
        acc[k][1] += w * p
        acc[k][2] += w * y
    return sum(abs(b[1] - b[2]) for b in acc if b[0] > 0) / tot


def evaluate_policy_full(policy: Policy, cfg: Config,
                         true_p: Optional[Dict[str, float]] = None) -> dict:
    """Exact DP: realized completion from S0, and reach-weighted calibration
    of the claimed value at every decision state (R3)."""
    est = policy.est
    memo: Dict[Tuple[int, int], float] = {}

    def V(st: State) -> float:
        if st.reached_goal():
            return 1.0
        key = (st.done, st.failed)
        hit = memo.get(key)
        if hit is not None:
            return hit
        i = policy.act(st)
        if i is None:
            memo[key] = 0.0
            return 0.0
        qe, qv, _ = true_exec_prob(i, st, est, cfg, true_p)
        q = qe * qv
        v = q * V(st.with_success(i)) + (1.0 - q) * V(st.with_failure(i))
        memo[key] = v
        return v

    v0 = V(State())
    # forward pass: reach probabilities of decision states
    reach: Dict[Tuple[int, int], float] = {(0, 0): 1.0}
    order = sorted(memo.keys(), key=lambda k: bin(k[0] | k[1]).count("1"))
    pts: List[Tuple[float, float, float]] = []
    brier = 0.0
    tot = 0.0
    for key in order:
        r = reach.get(key, 0.0)
        if r <= 0:
            continue
        st = State(*key)
        v = memo[key]
        p = est.p_goal(st)
        pts.append((r, p, v))
        brier += r * ((p - v) ** 2 + v * (1 - v))
        tot += r
        i = policy.act(st)
        if i is None:
            continue
        qe, qv, _ = true_exec_prob(i, st, est, cfg, true_p)
        q = qe * qv
        for nxt, pr in ((st.with_success(i), q), (st.with_failure(i), 1 - q)):
            if pr > 0 and not nxt.reached_goal():
                k2 = (nxt.done, nxt.failed)
                reach[k2] = reach.get(k2, 0.0) + r * pr
    return dict(real=v0, claim=est.p_goal(State()), ece=_ece(pts),
                brier=brier / tot if tot else float("nan"), states=len(pts))


def evaluate_policy(policy: Policy, cfg: Config) -> float:
    return evaluate_policy_full(policy, cfg)["real"]


def evaluate_policy_mc(policy: Policy, cfg: Config, episodes: int,
                       rng: random.Random, max_steps: Optional[int] = None) -> dict:
    """Monte Carlo evaluation for universes too large for exact DP. Every
    decision state contributes (claimed value, episode outcome) to ECE/Brier."""
    est = policy.est
    wins = 0
    pts: List[Tuple[float, float, float]] = []
    steps_tot = 0
    max_steps = max_steps or N_TASKS
    for _ in range(episodes):
        st = State()
        preds: List[float] = []
        for _ in range(max_steps):
            if st.reached_goal():
                break
            i = policy.act(st)
            if i is None:
                break
            preds.append(est.p_goal(st))
            qe, qv, _ = true_exec_prob(i, st, est, cfg)
            ok = rng.random() < qe and rng.random() < qv
            st = st.with_success(i) if ok else st.with_failure(i)
            steps_tot += 1
        y = 1.0 if st.reached_goal() else 0.0
        wins += int(y)
        for p in preds:
            pts.append((1.0, p, y))
    brier = statistics.fmean([(p - y) ** 2 for _, p, y in pts]) if pts else float("nan")
    return dict(real=wins / episodes, claim=est.p_goal(State()), ece=_ece(pts),
                brier=brier, steps=steps_tot / episodes)


def optimal_value(cfg: Config) -> float:
    """Exact optimal policy under known rho* (small universes only)."""
    memo: Dict[Tuple[int, int], float] = {}

    def V(st: State) -> float:
        if st.reached_goal():
            return 1.0
        key = (st.done, st.failed)
        hit = memo.get(key)
        if hit is not None:
            return hit
        sigma = st.sigma()
        used = st.used()
        best = 0.0
        for i, t in enumerate(TASKS):
            if used >> i & 1:
                continue
            q = t.p * (t.w(cfg.grade_w) if cfg.env_realizes_w else 1.0)
            for s in t.inputs:
                r = max((rho_star(x, s) for x in sigma), default=0.0)
                q *= r
                if q <= 0.0:
                    break
            if q <= 0.0:
                continue
            v = q * V(st.with_success(i)) + (1.0 - q) * V(st.with_failure(i))
            if v > best:
                best = v
        memo[key] = best
        return best

    return V(State())


# ---------------------------------------------------------------------------
# 8. Training (Thompson sampling, Eq.17)
# ---------------------------------------------------------------------------


def run_episode(belief: RelBelief, cfg: Config, rng: random.Random,
                explore: bool) -> bool:
    rel = belief.snapshot_sample(rng) if explore else belief.snapshot_mean()
    pol = MyopicPolicy(rel, explore_cfg(cfg) if explore else cfg, "train")
    st = State()
    for _ in range(N_TASKS):
        if st.reached_goal():
            return True
        i = pol.act(st)
        if i is None:
            return False
        t = TASKS[i]
        sigma = st.sigma()
        subs = [(pol.est.best_substitution(s, sigma)[0], s) for s in t.inputs]
        # (a) execution error
        p_exec_true = t.p * (t.w(cfg.grade_w) if cfg.env_realizes_w else 1.0)
        if rng.random() >= p_exec_true:
            belief.observe_exec(t.name, False)
            st = st.with_failure(i)
            continue
        belief.observe_exec(t.name, True)
        # (b) schema validation, one independent draw per input
        all_pass = True
        for x, s in subs:
            passed = rng.random() < rho_star(x, s)
            belief.observe(x, s, passed)
            if not passed:
                all_pass = False
        st = st.with_success(i) if all_pass else st.with_failure(i)
    return st.reached_goal()


def clone_belief(b: RelBelief, cfg: Optional[Config] = None) -> RelBelief:
    out = RelBelief(cfg or b.cfg)
    out.a, out.b = dict(b.a), dict(b.b)
    out.ea, out.eb = dict(b.ea), dict(b.eb)
    return out


def train(cfg: Config, episodes: int, seed: int) -> RelBelief:
    rng = random.Random(seed)
    belief = RelBelief(cfg)
    for _ in range(episodes):
        run_episode(belief, cfg, rng, explore=True)
    return belief


def train_checkpoints(cfg: Config, checkpoints: Sequence[int], seed: int
                      ) -> Dict[int, RelBelief]:
    rng = random.Random(seed)
    belief = RelBelief(cfg)
    out: Dict[int, RelBelief] = {}
    done = 0
    for cp in sorted(checkpoints):
        while done < cp:
            run_episode(belief, cfg, rng, explore=True)
            done += 1
        out[cp] = clone_belief(belief)
    return out


# ---------------------------------------------------------------------------
# 9. Reporting helpers
# ---------------------------------------------------------------------------


def ci95(xs: Sequence[float]) -> Tuple[float, float]:
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan"), float("nan")
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def claim_and_real(rel, cfg: Config, dynamic: bool = True) -> Tuple[float, float]:
    r = full_eval(rel, cfg, dynamic)
    return r["claim"], r["real"]


def full_eval(rel, cfg: Config, dynamic: bool = True) -> dict:
    pol: Policy = MyopicPolicy(rel, cfg) if dynamic else FixedPlanPolicy(rel, cfg)
    return evaluate_policy_full(pol, cfg)


def agg(rows: List[dict], keys=("claim", "real", "gap", "ece", "brier")) -> dict:
    out = {}
    for r in rows:
        r["gap"] = r["claim"] - r["real"]
    for k in keys:
        m, h = ci95([r[k] for r in rows])
        out[k], out[k + "_ci"] = m, h
    return out


# ---------------------------------------------------------------------------
# 10. Experiments on the reference universe
# ---------------------------------------------------------------------------

SEEDS = list(range(1, 11))
CHECKPOINTS = [0, 50, 100, 200, 400, 800, 1600]


def experiment1(cfg: Config, seeds=SEEDS, checkpoints=CHECKPOINTS) -> dict:
    """Learning curve (claim, realized, gap, ECE, Brier, rel error)."""
    traj = {sd: train_checkpoints(cfg, checkpoints, sd) for sd in seeds}
    rows = []
    for ep in checkpoints:
        res = [full_eval(traj[sd][ep].snapshot_mean(), cfg) for sd in seeds]
        a = agg(res)
        errs = [traj[sd][ep].rel_errors() for sd in seeds]
        for k in ("all", "nonzero", "observed"):
            m, h = ci95([e[k] for e in errs])
            a["rel_err_" + k], a["rel_err_" + k + "_ci"] = m, h
        a["episodes"] = ep
        rows.append(a)
    return dict(rows=rows, beliefs={sd: traj[sd][checkpoints[-1]] for sd in seeds})


KEY_PAIRS = [("market_raw", "stock_raw"), ("market_raw", "econ_raw"),
             ("econ_raw", "stock_raw"), ("stock_raw", "econ_raw"),
             ("news_digest", "news_summary")]


def learned_rel_summary(beliefs: Dict[int, RelBelief]) -> List[dict]:
    out = []
    for k in KEY_PAIRS:
        pts = [b.point(*k) for b in beliefs.values()]
        means = [b.mean(*k) for b in beliefs.values()]
        ns = [b.n(*k) for b in beliefs.values()]
        m, h = ci95(pts)
        mm, mh = ci95(means)
        out.append(dict(pair=f"{k[0]} -> {k[1]}", est=m, ci=h, post_mean=mm,
                        post_mean_ci=mh, truth=rho_star(*k), n_mean=statistics.fmean(ns)))
    return out


def learned_exec_summary(beliefs: Dict[int, RelBelief], cfg: Config) -> List[dict]:
    out = []
    tp = true_phat(cfg)
    for t in TASKS:
        vals = [b.exec_mean(t) for b in beliefs.values()]
        m, h = ci95(vals)
        out.append(dict(task=t.name, declared=t.p_hat(cfg.grade_w), learned=m, ci=h,
                        truth=tp[t.name],
                        n=statistics.fmean([b.ea.get(t.name, 0) + b.eb.get(t.name, 0)
                                            for b in beliefs.values()])))
    return out


def experiment2(cfg: Config, beliefs: Dict[int, RelBelief],
                beliefs_mean: Optional[Dict[int, RelBelief]] = None) -> List[dict]:
    """Model comparison. cfg is the proposed (LCB) configuration; beliefs are
    re-read under the posterior-mean configuration for the comparison rows."""
    out: List[dict] = []

    def add(label: str, rows: List[dict]) -> None:
        a = agg(rows)
        a["model"] = label
        out.append(a)

    seeds = sorted(beliefs)
    cm = cfg_mean(**{k: v for k, v in cfg.__dict__.items()
                     if k not in ("point_estimate", "theta")})
    cm0 = cfg_replace(cm, theta=0.0)

    add("SH-GD/SM-L (LCB, Eq.19)", [full_eval(beliefs[sd].snapshot_mean(), cfg) for sd in seeds])
    add("SH-GD/SM-L (posterior mean, theta=0.12)",
        [full_eval(clone_belief(beliefs[sd], cm).snapshot_mean(), cm) for sd in seeds])
    add("SH-GD/SM-L (posterior mean, theta=0)",
        [full_eval(clone_belief(beliefs[sd], cm0).snapshot_mean(), cm0) for sd in seeds])
    add("-dynamic (learned, plan fixed at S0)",
        [full_eval(beliefs[sd].snapshot_mean(), cfg, dynamic=False) for sd in seeds])
    # exact-match dynamic model, with learned execution reliability (fair)
    add("-relation (exact match, learned p_hat, dynamic)",
        [full_eval(with_phat(exact_rel_table(), beliefs[sd].snapshot_mean().phat), cfg)
         for sd in seeds])
    add("-learning (prior rel, posterior mean, theta=0.05)",
        [full_eval(with_phat(prior_rel_table(cm), None), cfg_replace(cm, theta=0.05))])
    sp = SinglePathPolicy(cfg)
    r = evaluate_policy_full(sp, cfg)
    r["claim"] = sp.claim(cfg)
    add("PropertyGraph (naive: exact, single path)", [r])
    add("RDF (naive: category match, no validation)",
        [evaluate_policy_full(CategoryMatchPolicy(cfg), cfg)])
    add("(ref) Oracle rel + oracle p_hat, same policy",
        [full_eval(with_phat(oracle_rel_table(), true_phat(cfg)), cfg)])
    if N_TASKS <= 16:
        out.append(dict(model="(ref) Optimal policy under rho* (exact DP)",
                        real=optimal_value(cfg), real_ci=0.0))
    return out


def experiment3(cfg: Config, beliefs: Dict[int, RelBelief], withdraw=("FetchEcon",)) -> List[dict]:
    """Dedicated task withdrawn; beliefs transferred unchanged."""
    saved = get_universe()
    set_universe(saved[0], [t for t in saved[1] if t.name not in withdraw], saved[2],
                 saved[3], saved[4])
    try:
        return experiment2(cfg, beliefs)
    finally:
        set_universe(*saved)


def _evaluate_with_true_p(policy: Policy, cfg: Config, true_p: Dict[str, float]) -> dict:
    return evaluate_policy_full(policy, cfg, true_p)


def _train_mismatched(cfg_agent: Config, ep: int, seed: int) -> RelBelief:
    """Training in the nominal environment with the agent's own w prior."""
    nominal = {t.name: t.p * t.w(GRADE_W_DEFAULT) for t in TASKS}
    rng = random.Random(seed)
    belief = RelBelief(cfg_agent)
    for _ in range(ep):
        rel = belief.snapshot_sample(rng)
        pol = MyopicPolicy(rel, explore_cfg(cfg_agent), "train")
        st = State()
        for _ in range(N_TASKS):
            if st.reached_goal():
                break
            i = pol.act(st)
            if i is None:
                break
            t = TASKS[i]
            sigma = st.sigma()
            subs = [(pol.est.best_substitution(s, sigma)[0], s) for s in t.inputs]
            if rng.random() >= nominal[t.name]:
                belief.observe_exec(t.name, False)
                st = st.with_failure(i)
                continue
            belief.observe_exec(t.name, True)
            ok = True
            for x, s in subs:
                passed = rng.random() < rho_star(x, s)
                belief.observe(x, s, passed)
                ok &= passed
            st = st.with_success(i) if ok else st.with_failure(i)
    return belief


def grade_weight_sweep(cfg: Config, seeds=SEEDS[:6], ep: int = 800) -> List[dict]:
    """Agent's w(g) prior varied; environment fixed at the nominal w."""
    gws = [("w=(1.00,0.80,0.50) [nominal]", {"observed": 1.0, "declared": 0.8, "inferred": 0.5}),
           ("w=(1,1,1)", {"observed": 1.0, "declared": 1.0, "inferred": 1.0}),
           ("w=(1.00,0.90,0.75)", {"observed": 1.0, "declared": 0.9, "inferred": 0.75}),
           ("w=(1.00,0.60,0.25)", {"observed": 1.0, "declared": 0.6, "inferred": 0.25})]
    nominal = {t.name: t.p * t.w(GRADE_W_DEFAULT) for t in TASKS}
    out = []
    for le in (True, False):
        for name, gw in gws:
            ca = cfg_replace(cfg, grade_w=gw, learn_exec=le)
            rows = []
            for sd in seeds:
                b = _train_mismatched(ca, ep, sd)
                pol = MyopicPolicy(b.snapshot_mean(), ca)
                rows.append(_evaluate_with_true_p(pol, ca, nominal))
            a = agg(rows)
            a["setting"] = f"{name}, learn_exec={le}"
            out.append(a)
    return out


def sensitivity_v4(cfg: Config, seeds=SEEDS[:6], ep: int = 800) -> dict:
    res: Dict[str, List[dict]] = {}

    def run(group: str, label: str, c: Config) -> None:
        rows = [full_eval(train(c, ep, sd).snapshot_mean(), c) for sd in seeds]
        a = agg(rows)
        a["setting"] = label
        res.setdefault(group, []).append(a)

    for a_, b_ in [(1.0, 4.0), (1.0, 9.0), (1.0, 19.0), (0.5, 4.5)]:
        run("prior_lcb", f"Beta({a_},{b_})", cfg_replace(cfg, alpha=a_, beta=b_))
        run("prior_mean", f"Beta({a_},{b_})",
            cfg_replace(cfg, alpha=a_, beta=b_, point_estimate="mean", theta=0.12))
    for th in [0.05, 0.12, 0.20, 0.30]:
        run("theta_mean", f"theta={th}", cfg_replace(cfg, point_estimate="mean", theta=th))
    for d in [0.05, 0.10, 0.20, 0.35]:
        run("delta", f"delta={d}", cfg_replace(cfg, lcb_delta=d))
    for flag in (True, False):
        for le in (True, False):
            run("w_semantics", f"env_realizes_w={flag}, learn_exec={le}",
                cfg_replace(cfg, env_realizes_w=flag, learn_exec=le))
    for lam in [0.0, 0.05, 0.10, 0.20]:
        run("lambda", f"lambda={lam}", cfg_replace(cfg, lam=lam))
    res["grade_w"] = grade_weight_sweep(cfg, seeds, ep)
    return res


def trace_table(cfg: Config, belief: RelBelief) -> dict:
    """Post-failure candidate ranking (Tables 12/13)."""
    rel = belief.snapshot_mean()
    pol = MyopicPolicy(rel, cfg)
    est = pol.est
    st = State()
    steps = []
    forced_fail = {"FetchEcon"}
    for _ in range(N_TASKS):
        if st.reached_goal():
            break
        i = pol.act(st)
        if i is None:
            break
        t = TASKS[i]
        v_before = est.p_goal(st)
        if t.name in forced_fail:
            st_f = st.with_failure(i)
            unmet = "econ_raw"
            sc = pol.scores(st_f)
            util = {j: u for j, _, _, _, u in sc}
            # Eq.16  Alt(s,S) = { e in Ready(S) | str(e,s) >= theta }
            alt = []
            for j in sorted(util, key=lambda k: TASKS[k].name):
                tj = TASKS[j]
                if est.str_(tj, unmet) < cfg.theta:
                    continue
                alt.append(dict(task=tj.name, rel=est.rho(tj, unmet), p_hat=est.phat[j],
                                str_=est.str_(tj, unmet), utility=util[j]))
            alt.sort(key=lambda r: -r["str_"])
            rank = [dict(task=TASKS[j].name, p_hat=p_h, p_exec=q,
                         v_succ=est.p_goal(st_f.with_success(j)),
                         v_fail=est.p_goal(st_f.with_failure(j)),
                         utility=u, in_alt=any(a["task"] == TASKS[j].name for a in alt))
                    for j, q, p_h, _, u in sc]
            steps.append(dict(step=len(steps) + 1, task=t.name, outcome="execution error",
                              v_before=v_before, v_after=est.p_goal(st_f), unmet=unmet,
                              eq14_candidates=alt, eq13_ranking=rank))
            st = st_f
            forced_fail = set()
            continue
        steps.append(dict(step=len(steps) + 1, task=t.name, outcome="success",
                          v_before=v_before, v_after=est.p_goal(st.with_success(i))))
        st = st.with_success(i)
    return dict(steps=steps, reached=st.reached_goal())


# ---------------------------------------------------------------------------
# 11. Main
# ---------------------------------------------------------------------------


def _jsonable(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_v4.json")
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    sys.setrecursionlimit(100000)
    seeds = list(range(1, args.seeds + 1))
    cfg = cfg_lcb()

    print("== Experiment 1 (LCB, learned p_hat) ==", flush=True)
    e1 = experiment1(cfg, seeds)
    for r in e1["rows"]:
        print(f"   ep={r['episodes']:>5} claim={pct(r['claim'])} real={pct(r['real'])} "
              f"gap={100*r['gap']:+.2f}pt ECE={r['ece']:.4f} Brier={r['brier']:.4f} "
              f"err(all/nz/obs)={r['rel_err_all']:.4f}/{r['rel_err_nonzero']:.4f}/"
              f"{r['rel_err_observed']:.4f}", flush=True)
    print("== Experiment 1' (posterior mean + theta=0.12, learned p_hat) ==", flush=True)
    e1m = experiment1(cfg_mean(), seeds)
    for r in e1m["rows"]:
        print(f"   ep={r['episodes']:>5} claim={pct(r['claim'])} real={pct(r['real'])} "
              f"gap={100*r['gap']:+.2f}pt ECE={r['ece']:.4f} Brier={r['brier']:.4f}", flush=True)
    lrs = learned_rel_summary(e1["beliefs"])
    les = learned_exec_summary(e1["beliefs"], cfg)
    print("== Experiment 2 ==", flush=True)
    e2 = experiment2(cfg, e1["beliefs"])
    for r in e2:
        print("  ", r["model"], {k: round(v, 4) for k, v in r.items()
                                 if isinstance(v, float) and not k.endswith("_ci")}, flush=True)
    print("== Experiment 3 ==", flush=True)
    e3 = experiment3(cfg, e1["beliefs"])
    for r in e3:
        print("  ", r["model"], {k: round(v, 4) for k, v in r.items()
                                 if isinstance(v, float) and not k.endswith("_ci")}, flush=True)
    tr = trace_table(cfg, e1["beliefs"][seeds[0]])
    sens = {} if args.skip_sensitivity else sensitivity_v4(cfg)
    payload = dict(config=dict(cfg.__dict__), seeds=seeds,
                   cycles=[list(c) for c in RHO_STAR if (c[1], c[0]) in RHO_STAR],
                   experiment1=e1["rows"], experiment1_mean=e1m["rows"],
                   learned_rel=lrs, learned_exec=les,
                   experiment2=e2, experiment3=e3, trace=tr, sensitivity=sens,
                   optimal_value=optimal_value(cfg))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
