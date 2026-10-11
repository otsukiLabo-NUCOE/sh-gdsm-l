#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Writes every number and every results table of the IEEE Access manuscript
from the result files, so that nothing is transcribed by hand.

    python make_tex_numbers.py --out ../IEEE_Access/manuscript

Inputs (missing files are rendered as "[pending]" so the manuscript compiles):
    results_v4.json               Experiments 1-3, 5, trace, sensitivity
    llm_results_armC_v4.json      Experiment 4, arm C (LCB)
    llm_results_real_ABD.json     Experiment 4, arms A/B/D with a real LLM
    results_recovery_v4.json      Experiment 6, decision-rule proxies (LCB)
    results_recovery_llm.json     Experiment 6, real-LLM arms
    results_envgen.json           Experiment 7, random universes
    rho_measured.json             induced rho* of the real-validator stubs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Dict, List, Optional

PENDING = r"\textbf{[pending]}"
NUM: Dict[str, str] = {}


def load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def isnum(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def pc(x, d=2) -> str:            # probability -> percent string
    return f"{100*x:.{d}f}" if isnum(x) else "--"


def pt(x, d=2) -> str:            # probability difference -> signed points
    return f"{100*x:+.{d}f}" if isnum(x) else "--"


def f4(x, d=4) -> str:
    return f"{x:.{d}f}" if isnum(x) else "--"


def ci(x, d=2) -> str:
    return f"{100*x:.{d}f}" if isnum(x) else "--"


def macro(name: str, value: str) -> None:
    assert re.fullmatch(r"N[A-Za-z]+", name), name
    NUM[name] = value


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("->", r"$\rightarrow$").replace("%", r"\%")


def table(path: str, header: List[str], rows: List[List[str]], align: str) -> None:
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        if r == ["MIDRULE"]:
            lines.append(r"\midrule")
        else:
            lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def pending_table(path: str, what: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(r"\begin{tabular}{l}\toprule " + PENDING + " " + what +
                r" \\ \bottomrule\end{tabular}" + "\n")


def find(rows: List[dict], key: str, prefix: str) -> dict:
    for r in rows:
        if str(r.get(key, "")).startswith(prefix):
            return r
    raise KeyError(prefix)


# ---------------------------------------------------------------------------
# Experiments 1-3, 5 (reference universe)
# ---------------------------------------------------------------------------

MODEL_LABELS = [
    ("SH-GD/SM-L (LCB", r"\model{} (LCB, \eqref{eq:lcb})"),
    ("SH-GD/SM-L (posterior mean, theta=0.12)", r"\model{} (posterior mean, $\theta=0.12$)"),
    ("SH-GD/SM-L (posterior mean, theta=0)", r"\model{} (posterior mean, $\theta=0$)"),
    ("-dynamic", r"$-$dynamic (learned, plan fixed at $S_0$)"),
    ("-relation", r"$-$relation (exact match, dynamic)$^{a}$"),
    ("-learning", r"$-$learning (prior $\rel$, mean, $\theta=0.05$)"),
    ("PropertyGraph", r"Property graph (naive: exact, single path)"),
    ("RDF (naive", r"RDF (naive: category match, no validation)"),
    ("(ref) Oracle", r"(ref.) Oracle $\rel$ and $\phat$, same policy"),
    ("(ref) Optimal", r"(ref.) Optimal policy under $\rho^*$ (exact DP)"),
]


def comparison_table(rows: List[dict], path: str) -> None:
    out = []
    for key, label in MODEL_LABELS:
        r = find(rows, "model", key)
        if "claim" in r:
            out.append([label, pc(r["claim"]), pc(r["real"]) + (r" $\pm$ " + ci(r["real_ci"]) if r.get("real_ci") else ""),
                        pt(r["gap"]), f4(r["ece"], 3), f4(r["brier"], 3)])
        else:
            out.append([label, "--", pc(r["real"]), "--", "--", "--"])
    table(path, ["Model", r"Claim [\%]", r"Realized [\%]", "Gap [pt]", "ECE", "Brier"], out, "lrrrrr")


def ref_universe(d: dict, T: str) -> None:
    e1 = d["experiment1"]
    last = e1[-1]
    macro("NRelErrAllStart", f4(e1[0]["rel_err_all"]))
    macro("NRelErrAllEnd", f4(last["rel_err_all"]))
    macro("NRelErrNzStart", f4(e1[0]["rel_err_nonzero"]))
    macro("NRelErrNzEnd", f4(last["rel_err_nonzero"]))
    macro("NRelErrObsEnd", f4(last["rel_err_observed"]))
    macro("NMaxAbsGapCurve", f"{max(abs(100*r['gap']) for r in e1):.2f}")
    macro("NEceCurveMax", f4(max(r["ece"] for r in e1 if r["episodes"] > 0), 3))
    rows = []
    for r in e1:
        rows.append([str(r["episodes"]), pc(r["claim"]) + r" $\pm$ " + ci(r["claim_ci"]),
                     pc(r["real"]) + r" $\pm$ " + ci(r["real_ci"]), pt(r["gap"]),
                     f4(r["ece"], 3), f4(r["brier"], 3),
                     f4(r["rel_err_all"], 3), f4(r["rel_err_nonzero"], 3),
                     f4(r["rel_err_observed"], 3) if isnum(r["rel_err_observed"]) else "--"])
    table(f"{T}/exp1_curve.tex",
          ["Ep.", r"Claim [\%]", r"Realized [\%]", "Gap", "ECE", "Brier",
           r"$\epsilon_{\mathrm{all}}$", r"$\epsilon_{\mathrm{nz}}$", r"$\epsilon_{\mathrm{obs}}$"],
          rows, "rrrrrrrrr")
    em = d["experiment1_mean"]
    rows = [[str(r["episodes"]), pc(r["claim"]), pc(r["real"]), pt(r["gap"]), f4(r["ece"], 3)] for r in em]
    table(f"{T}/exp1_curve_mean.tex", ["Ep.", r"Claim [\%]", r"Realized [\%]", "Gap [pt]", "ECE"],
          rows, "rrrrr")

    rows = []
    for r in d["learned_rel"]:
        rows.append([tex_escape(r["pair"]), f"{r['est']:.3f} $\\pm$ {r['ci']:.3f}",
                     f"{r['post_mean']:.3f} $\\pm$ {r['post_mean_ci']:.3f}", f"{r['truth']:.2f}",
                     f"{r['n_mean']:.1f}"])
    table(f"{T}/learned_rel.tex", [r"Pair $(x\rightarrow s)$", "LCB", "Post. mean", r"$\rho^*$", r"$\bar n$"],
          rows, "lrrrr")
    rows = [[tex_escape(r["task"]), f"{r['declared']:.3f}", f"{r['learned']:.3f}", f"{r['truth']:.3f}",
             f"{r['n']:.0f}"] for r in d["learned_exec"]]
    table(f"{T}/learned_exec.tex", ["Task", r"$\phat^{\,0}_e$", r"$\phat_e$", "True", r"$n$"], rows, "lrrrr")

    e2, e3 = d["experiment2"], d["experiment3"]
    comparison_table(e2, f"{T}/exp2.tex")
    comparison_table(e3, f"{T}/exp3.tex")
    g = lambda rows, k: find(rows, "model", k)
    L, Mn, M0 = g(e2, "SH-GD/SM-L (LCB"), g(e2, "SH-GD/SM-L (posterior mean, theta=0.12)"), \
        g(e2, "SH-GD/SM-L (posterior mean, theta=0)")
    Fx, Ex, Pr = g(e2, "-dynamic"), g(e2, "-relation"), g(e2, "-learning")
    Op, Or = g(e2, "(ref) Optimal"), g(e2, "(ref) Oracle")
    macro("NRealLCB", pc(L["real"])); macro("NClaimLCB", pc(L["claim"])); macro("NGapLCB", pt(L["gap"]))
    macro("NEceLCB", f4(L["ece"], 3)); macro("NBrierLCB", f4(L["brier"], 3))
    macro("NRealMean", pc(Mn["real"])); macro("NGapMean", pt(Mn["gap"])); macro("NEceMean", f4(Mn["ece"], 3))
    macro("NRealMeanNoTheta", pc(M0["real"])); macro("NClaimMeanNoTheta", pc(M0["claim"]))
    macro("NGapMeanNoTheta", pt(M0["gap"])); macro("NEceMeanNoTheta", f4(M0["ece"], 3))
    macro("NRealFixed", pc(Fx["real"])); macro("NClaimFixed", pc(Fx["claim"])); macro("NGapFixed", pt(Fx["gap"]))
    macro("NEceFixed", f4(Fx["ece"], 3))
    macro("NRealExact", pc(Ex["real"])); macro("NGapExact", pt(Ex["gap"])); macro("NEceExact", f4(Ex["ece"], 3))
    macro("NClaimPriorLow", pc(Pr["claim"])); macro("NRealPriorLow", pc(Pr["real"])); macro("NGapPriorLow", pt(Pr["gap"]))
    macro("NRealOpt", pc(Op["real"])); macro("NRealOracle", pc(Or["real"]))
    macro("NClaimRDF", pc(g(e2, "RDF (naive")["claim"]))
    macro("NRatioLCB", f"{100*L['real']/Op['real']:.1f}"); macro("NRatioMean", f"{100*Mn['real']/Op['real']:.1f}")
    macro("NGainLCB", f"{100*(L['real']-Ex['real']):.2f}"); macro("NGainMean", f"{100*(Mn['real']-Ex['real']):.2f}")
    macro("NFixedBelowExact", f"{100*(Ex['real']-Fx['real']):.2f}")
    macro("NDynOverFixed", f"{100*(L['real']-Fx['real']):.2f}")
    L3, Fx3, Op3, M03, Pr3 = g(e3, "SH-GD/SM-L (LCB"), g(e3, "-dynamic"), g(e3, "(ref) Optimal"), \
        g(e3, "SH-GD/SM-L (posterior mean, theta=0)"), g(e3, "-learning")
    Mn3 = g(e3, "SH-GD/SM-L (posterior mean, theta=0.12)")
    macro("NRealLCBWd", pc(L3["real"])); macro("NGapLCBWd", pt(L3["gap"])); macro("NEceLCBWd", f4(L3["ece"], 3))
    macro("NRealMeanWd", pc(Mn3["real"]))
    macro("NRealFixedWd", pc(Fx3["real"])); macro("NRealOptWd", pc(Op3["real"]))
    macro("NRatioLCBWd", f"{100*L3['real']/Op3['real']:.1f}")
    macro("NGapMeanNoThetaWd", pt(M03["gap"])); macro("NClaimPriorLowWd", pc(Pr3["claim"]))
    macro("NGapPriorLowWd", pt(Pr3["gap"]))
    lr = {r["pair"]: r for r in d["learned_rel"]}
    macro("NLcbMarketEcon", f"{lr['market_raw -> econ_raw']['est']:.3f}")
    macro("NLcbMarketStock", f"{lr['market_raw -> stock_raw']['est']:.3f}")
    macro("NMeanMarketStock", f"{lr['market_raw -> stock_raw']['post_mean']:.3f}")
    macro("NLcbDigest", f"{lr['news_digest -> news_summary']['est']:.3f}")
    macro("NMeanDigest", f"{lr['news_digest -> news_summary']['post_mean']:.3f}")
    macro("NLcbEconStock", f"{lr['econ_raw -> stock_raw']['est']:.3f}")
    macro("NnEconStock", f"{lr['econ_raw -> stock_raw']['n_mean']:.0f}")

    # trace
    tr = d["trace"]["steps"]
    rows, alt, rank, fail = [], [], [], None
    for s in tr:
        rows.append([str(s["step"]), tex_escape(s["task"]),
                     "exec. error" if s["outcome"] != "success" else "success",
                     f"{s['v_before']:.4f}", f"{s['v_after']:.4f}"])
        if "eq14_candidates" in s:
            fail = s
    table(f"{T}/trace.tex", ["Step", "Task", "Outcome", r"$\hat V$ before", r"$\hat V$ after"], rows, "rllrr")
    if fail:
        macro("NTraceFailStep", str(fail["step"]))
        for c in fail["eq14_candidates"]:
            alt.append([tex_escape(c["task"]), f"{c['rel']:.3f}", f"{c['p_hat']:.3f}", f"{c['str_']:.4f}",
                        f"{c['utility']:.4f}"])
        table(f"{T}/trace_alt.tex", ["Candidate", r"$\rel(o\rightarrow\text{econ\_raw})$", r"$\phat_e$",
                                     r"$\str$ \eqref{eq:str}", r"$U$ \eqref{eq:utility}"], alt, "lrrrr")
        for c in fail["eq13_ranking"]:
            rank.append([tex_escape(c["task"]), f"{c['p_hat']:.3f}", f"{c['p_exec']:.4f}", f"{c['v_succ']:.4f}",
                         f"{c['v_fail']:.4f}", f"{c['utility']:.4f}", r"$\checkmark$" if c["in_alt"] else ""])
        table(f"{T}/trace_rank.tex", ["Task", r"$\phat_e$", r"$\tilde p$ \eqref{eq:pexec}", r"$\hat V$ succ.",
                                      r"$\hat V$ fail", r"$U$", r"$\in\Alt$"], rank, "lrrrrrc")
        top = fail["eq13_ranking"][0]
        macro("NTraceChosen", tex_escape(top["task"]))
        ce = next(c for c in fail["eq13_ranking"] if c["task"] == "CleanEcon")
        macro("NTraceCleanEconP", f"{ce['p_exec']:.4f}")
        macro("NTraceCleanEconPhat", f"{ce['p_hat']:.3f}")
        macro("NTraceCleanEconU", f"{ce['utility']:.4f}")
        macro("NTraceTopU", f"{top['utility']:.4f}")
        macro("NTraceTopP", f"{top['p_exec']:.4f}")
        macro("NTraceTopVs", f"{top['v_succ']:.4f}")
        macro("NTraceTopVf", f"{top['v_fail']:.4f}")
        a0 = fail["eq14_candidates"][0]
        macro("NTraceAltRel", f"{a0['rel']:.3f}")

    # sensitivity
    S = d["sensitivity"]
    def srows(group):
        return [[tex_escape(r["setting"]), pc(r["claim"]), pc(r["real"]), pt(r["gap"]), f4(r["ece"], 3)]
                for r in S[group]]
    hdr = ["Setting", r"Claim [\%]", r"Realized [\%]", "Gap [pt]", "ECE"]
    for gname in ("theta_mean", "delta", "w_semantics", "lambda", "grade_w"):
        table(f"{T}/sens_{gname}.tex", hdr, srows(gname), "lrrrr")
    rows = []
    for a, b in zip(S["prior_mean"], S["prior_lcb"]):
        rows.append([tex_escape(a["setting"]), pt(a["gap"]), f4(a["ece"], 3), pc(a["real"]),
                     pt(b["gap"]), f4(b["ece"], 3), pc(b["real"])])
    table(f"{T}/sens_prior.tex",
          ["Prior", "Gap", "ECE", r"Real.", "Gap", "ECE", r"Real."], rows, "lrrrrrr")
    macro("NSensMeanPriorWorstGap", pt(max(r["gap"] for r in S["prior_mean"])))
    macro("NSensLcbPriorWorstGap", pt(max(r["gap"] for r in S["prior_lcb"])))
    macro("NSensMeanPriorWorstEce", f4(max(r["ece"] for r in S["prior_mean"]), 3))
    macro("NSensLcbPriorWorstEce", f4(max(r["ece"] for r in S["prior_lcb"]), 3))
    macro("NSensRatioWorst", f"{max(r['gap'] for r in S['prior_mean'])/max(1e-9, max(r['gap'] for r in S['prior_lcb'])):.0f}")
    macro("NSensLcbRealMin", pc(min(r["real"] for r in S["prior_lcb"])))
    macro("NSensLcbRealMax", pc(max(r["real"] for r in S["prior_lcb"])))
    th = {r["setting"]: r for r in S["theta_mean"]}
    macro("NSensThetaLowGap", pt(th["theta=0.05"]["gap"]))
    macro("NSensThetaHighReal", pc(th["theta=0.3"]["real"]))
    de = {r["setting"]: r for r in S["delta"]}
    macro("NSensDeltaHighGap", pt(de["delta=0.35"]["gap"]))
    gw = S["grade_w"]
    le = [r for r in gw if r["setting"].endswith("learn_exec=True")]
    nl = [r for r in gw if r["setting"].endswith("learn_exec=False")]
    span = lambda rs: 100 * (max(r["gap"] for r in rs) - min(r["gap"] for r in rs))
    macro("NGradeSpanLearn", f"{span(le):.2f}")
    macro("NGradeSpanFixed", f"{span(nl):.2f}")
    ws = {r["setting"]: r for r in S["w_semantics"]}
    macro("NWsemEpiLearnGap", pt(ws["env_realizes_w=False, learn_exec=True"]["gap"]))
    macro("NWsemEpiFixedGap", pt(ws["env_realizes_w=False, learn_exec=False"]["gap"]))
    macro("NWsemEpiReal", pc(ws["env_realizes_w=False, learn_exec=True"]["real"]))
    lam = S["lambda"]
    macro("NLambdaRealMin", pc(min(r["real"] for r in lam)))
    macro("NLambdaRealMax", pc(max(r["real"] for r in lam)))


# ---------------------------------------------------------------------------
# Experiment 4
# ---------------------------------------------------------------------------

def exp4(c: Optional[dict], abd: Optional[dict], rho: Optional[dict], T: str) -> None:
    if rho:
        rows = [[tex_escape(k), f"{v:.4f}", f"{dv:.2f}"] for k, v, dv in rho["rows"]]
        table(f"{T}/exp4_rho.tex", [r"Pair $(x\rightarrow s)$", rf"Induced $\rho^*$ ({rho['n']:,} inst.)",
                                    "Design value"], rows, "lrr")
    else:
        pending_table(f"{T}/exp4_rho.tex", "induced rho*")
    names = {"A": r"A\ \ LLM zero-shot", "B": r"B\ \ LLM + reflect/retry", "C": r"C\ \ \model",
             "D": r"D\ \ LLM + learned $\rel$ hint"}
    rows = []
    res = {}
    if c:
        res.update(c["results"])
    if abd:
        res.update(abd["results"])
        # seeds completed so far (a partially finished run is reported as such)
        for arm, seeds in abd.get("per_seed", {}).items():
            if arm not in abd["results"] and seeds:
                comp = [s["completion"] for s in seeds]
                res[arm] = dict(completion=sum(comp) / len(comp), completion_ci=float("nan"),
                                steps=sum(s["steps"] for s in seeds) / len(seeds),
                                malformed=sum(s["malformed"] for s in seeds) / len(seeds),
                                llm_calls_per_episode=sum(s["llm_calls"] for s in seeds) / len(seeds) /
                                max(1, abd["config"]["episodes"]),
                                mean_call_latency_s=sum(s["mean_call_latency_s"] for s in seeds) / len(seeds),
                                partial=len(seeds))
    for arm in "ABCD":
        r = res.get(arm)
        if not r:
            rows.append([names[arm], PENDING, "", "", "", "", ""])
            continue
        rows.append([names[arm], pc(r["completion"]) + (r" $\pm$ " + ci(r["completion_ci"]) if isnum(r.get("completion_ci")) and r.get("completion_ci") else ""),
                     pc(r["claim"]) if ("claim" in r and arm == "C") else "--",
                     pt(r["calibration_gap"]) if ("calibration_gap" in r and arm == "C") else "--",
                     f"{r['steps']:.2f}", f"{r.get('llm_calls_per_episode', 0):.1f}",
                     f"{r.get('mean_call_latency_s', 0):.2f}" if r.get("mean_call_latency_s") else "--"])
    table(f"{T}/exp4.tex", ["Arm", r"Completion [\%]", r"Claim [\%]", "Gap [pt]", "Steps", "Calls/ep.",
                            "s/call"], rows, "lrrrrrr")
    if "C" in res:
        macro("NFourC", pc(res["C"]["completion"])); macro("NFourCClaim", pc(res["C"]["claim"]))
        macro("NFourCGap", pt(res["C"]["calibration_gap"]))
    for arm, key in (("A", "NFourA"), ("B", "NFourB"), ("D", "NFourD")):
        if arm in res:
            macro(key, pc(res[arm]["completion"]))
            macro(key + "Calls", f"{res[arm].get('llm_calls_per_episode', 0):.1f}")
            macro(key + "Malformed", f"{res[arm].get('malformed', 0):.2f}")
        else:
            macro(key, PENDING); macro(key + "Calls", PENDING); macro(key + "Malformed", PENDING)
    lat = [res[a]["mean_call_latency_s"] for a in "ABD" if a in res and res[a].get("mean_call_latency_s")]
    macro("NLatencyMeasured", f"{sum(lat)/len(lat):.2f}" if lat else PENDING)
    if abd:
        macro("NFourEpisodes", str(abd["config"]["episodes"]))
        macro("NFourSeeds", str(abd["config"]["seeds"]))
    else:
        macro("NFourEpisodes", PENDING); macro("NFourSeeds", PENDING)


# ---------------------------------------------------------------------------
# Experiment 6
# ---------------------------------------------------------------------------

ARMS6 = [("C", r"C\ \ \model"), ("A", r"A\ \ delegated"), ("B", r"B\ \ delegated + retry"),
         ("A*", r"A$^*$ delegated ($\rho^*$)")]


def ttr(r, tau_tool=0.5, tau_llm=2.0):
    s, c = r.get("steps_to_recover"), r.get("calls_to_recover")
    if not (isnum(s) and isnum(c)):
        return float("nan")
    return s * tau_tool + c * tau_llm + s * r["decision_ms"] / 1000.0


def breakeven(c, a, tau_tool=0.5):
    # C time < A time  <=>  tau_llm > (dsteps*tau_tool + ddec) / calls_A   (C makes no calls)
    if not all(isnum(x) for x in (c.get("steps_to_recover"), a.get("steps_to_recover"), a.get("calls_to_recover"))):
        return float("nan")
    if a["calls_to_recover"] <= 0:
        return float("inf")
    num = (c["steps_to_recover"] - a["steps_to_recover"]) * tau_tool \
        + c["steps_to_recover"] * c["decision_ms"] / 1000.0
    return max(0.0, num / a["calls_to_recover"])


def exp6(d: Optional[dict], llm: Optional[dict], T: str) -> None:
    if not d:
        for n in ("exp6", "exp6_noise", "exp6_iid"):
            pending_table(f"{T}/{n}.tex", "Experiment 6")
        for k in ("NSixBreakevenA", "NSixBreakevenAstar", "NSixDecMs", "NSixCompDiffMin", "NSixCompDiffMax",
                  "NSixCallsAMin", "NSixCallsAMax", "NSixCallsBMin", "NSixCallsBMax", "NSixTimeCMin",
                  "NSixTimeCMax", "NSixTimeAMin", "NSixTimeAMax", "NSixTimeBMin", "NSixTimeBMax",
                  "NSixSFourC", "NSixSFourA", "NSixSFiveB", "NSixSFiveC", "NSixSFiveCR", "NSixStageSThree",
                  "NSixStageGap", "NSixEpsGap", "NSixIidBMax", "NSixIidDiff"):
            macro(k, PENDING)
    else:
        m = d["main"]
        rows = []
        keys = sorted(m)
        for k in keys:
            first = True
            for arm, label in ARMS6:
                r = m[k][arm]
                rows.append([k if first else "", label, pc(r["completion"]) + r" $\pm$ " + ci(r["completion_ci"]),
                             pc(r["recovery_rate"]),
                             f"{r['steps_to_recover']:.2f}" if isnum(r["steps_to_recover"]) else "--",
                             f"{r['wasted_to_recover']:.2f}" if isnum(r["wasted_to_recover"]) else "--",
                             f"{r['calls_to_recover']:.2f}" if isnum(r["calls_to_recover"]) else "--",
                             f"{r['calls_per_episode']:.1f}",
                             f"{ttr(r):.2f}" if isnum(ttr(r)) else "--"])
                first = False
            if k != keys[-1]:
                rows.append(["MIDRULE"])
        table(f"{T}/exp6.tex", ["Sc.", "Arm", r"Compl. [\%]", r"Rec. [\%]", "Exec.", "Failed", "Calls",
                                "Calls/ep.", r"$T_{\mathrm{rec}}$ [s]"], rows, "llrrrrrrr")
        rec = [k for k in keys if isnum(m[k]["C"]["steps_to_recover"]) and isnum(m[k]["A"]["steps_to_recover"])]
        be_a = [breakeven(m[k]["C"], m[k]["A"]) for k in rec]
        be_s = [breakeven(m[k]["C"], m[k]["A*"]) for k in rec]
        macro("NSixBreakevenA", f"{max(be_a):.2f}")
        macro("NSixBreakevenAstar", f"{max(be_s):.2f}")
        macro("NSixDecMs", f"{sum(m[k]['C']['decision_ms'] for k in keys)/len(keys):.2f}")
        nos4 = [k for k in keys if k != "S4"]
        diffs = [100 * (m[k]["C"]["completion"] - m[k]["A"]["completion"]) for k in nos4]
        macro("NSixCompDiffMin", f"{min(diffs):+.2f}"); macro("NSixCompDiffMax", f"{max(diffs):+.2f}")
        ca = [m[k]["A"]["calls_per_episode"] for k in keys]; cb = [m[k]["B"]["calls_per_episode"] for k in keys]
        macro("NSixCallsAMin", f"{min(ca):.1f}"); macro("NSixCallsAMax", f"{max(ca):.1f}")
        macro("NSixCallsBMin", f"{min(cb):.1f}"); macro("NSixCallsBMax", f"{max(cb):.1f}")
        for arm, nm in (("C", "C"), ("A", "A"), ("B", "B")):
            ts = [ttr(m[k][arm]) for k in rec]
            macro(f"NSixTime{nm}Min", f"{min(ts):.2f}"); macro(f"NSixTime{nm}Max", f"{max(ts):.2f}")
        macro("NSixSFourC", pc(m["S4"]["C"]["completion"])); macro("NSixSFourA", pc(m["S4"]["A"]["completion"]))
        macro("NSixSFiveB", pc(m["S5"]["B"]["completion"])); macro("NSixSFiveC", pc(m["S5"]["C"]["completion"]))
        macro("NSixSFiveCR", pc(m["S5"]["C+R"]["completion"]))
        # sensitivity of the proxy
        nz = d.get("noise", {}); sc = d.get("surface_c", {}); ss = d.get("surface_stage", {})
        rows = []
        for k in keys:
            row = [k, pc(m[k]["C"]["completion"]), pc(m[k]["A"]["completion"])]
            for lab in ("eps=0.1,malformed=0.05", "eps=0.2,malformed=0.1"):
                row.append(pc(nz[lab][k]["A"]["completion"]) if lab in nz and k in nz[lab] else "--")
            row.append(pc(ss[k]["A"]["completion"]) if k in ss else "--")
            row.append(pc(m[k]["A*"]["completion"]))
            rows.append(row)
        table(f"{T}/exp6_noise.tex", ["Sc.", "C", "A", r"A $\epsilon{=}0.1$", r"A $\epsilon{=}0.2$",
                                      "A stage-only", r"A$^*$"], rows, "lrrrrrr")
        if ss:
            macro("NSixStageSThree", pc(ss["S3"]["A"]["completion"]))
            macro("NSixStageGap", f"{100*(m['S3']['C']['completion']-ss['S3']['A']['completion']):.2f}")
        if "eps=0.2,malformed=0.1" in nz:
            g = [100 * (m[k]["C"]["completion"] - nz["eps=0.2,malformed=0.1"][k]["A"]["completion"]) for k in nos4]
            macro("NSixEpsGap", f"{max(g):.2f}")
        iid = d.get("iid", {})
        rows = []
        for k in keys:
            if k in iid:
                r = iid[k]
                rows.append([k, pc(r["A"]["completion"]), pc(r["B"]["completion"]), pc(r["C"]["completion"]),
                             pc(r["C+R"]["completion"]), f"{r['B']['calls_per_episode']:.1f}", "0"])
        table(f"{T}/exp6_iid.tex", ["Sc.", "A", "B", "C", "C+R", "B calls/ep.", "C+R calls/ep."], rows, "lrrrrrr")
        if iid:
            macro("NSixIidBMax", pc(max(iid[k]["B"]["completion"] for k in iid)))
            macro("NSixIidDiff", f"{max(abs(100*(iid[k]['B']['completion']-iid[k]['C+R']['completion'])) for k in iid if k != 'S4'):.2f}")
    # real LLM
    if llm:
        rows = []
        for k in sorted(llm["results"]):
            for arm in ("C", "A-llm", "B-llm"):
                r = llm["results"][k].get(arm)
                if not r:
                    continue
                rows.append([k, {"C": r"C\ \ \model", "A-llm": "A\\ \\ LLM", "B-llm": "B\\ \\ LLM + retry"}[arm],
                             pc(r["completion"]), pc(r["recovery_rate"]),
                             f"{r['steps_to_recover']:.2f}" if isnum(r.get("steps_to_recover")) else "--",
                             f"{r['wasted_to_recover']:.2f}" if isnum(r.get("wasted_to_recover")) else "--",
                             f"{r['calls_to_recover']:.2f}" if isnum(r.get("calls_to_recover")) else "--",
                             f"{r['calls_per_episode']:.1f}"])
        table(f"{T}/exp6_llm.tex", ["Sc.", "Arm", r"Compl. [\%]", r"Rec. [\%]", "Exec.", "Failed", "Calls",
                                    "Calls/ep."], rows, "llrrrrrr")
        lat = [v for v in llm.get("mean_call_latency_s", {}).values() if v]
        macro("NSixLlmLatency", f"{sum(lat)/len(lat):.2f}" if lat else PENDING)
        macro("NSixLlmEpisodes", str(llm.get("episodes")))
        macro("NSixLlmSeeds", str(llm.get("seeds")))
    else:
        pending_table(f"{T}/exp6_llm.tex", "real-LLM recovery")
        for k in ("NSixLlmLatency", "NSixLlmEpisodes", "NSixLlmSeeds"):
            macro(k, PENDING)


# ---------------------------------------------------------------------------
# Experiment 7
# ---------------------------------------------------------------------------

ARMS7 = [("lcb", r"\model{} (LCB)"), ("mean", r"\model{} (mean, $\theta{=}0.12$)"),
         ("exact", "Exact match, dynamic"), ("fixed", r"$-$dynamic (plan fixed)"),
         ("surface", "Delegated proxy (names)"), ("oracle", r"(ref.) Oracle")]


def exp7(d: Optional[dict], T: str) -> None:
    if not d or not d.get("summary"):
        pending_table(f"{T}/exp7.tex", "Experiment 7"); pending_table(f"{T}/exp7_transfer.tex", "Experiment 7")
        for k in ("NEnvMaxTasks", "NEnvMinTasks", "NEnvUniverses", "NEnvSizes"):
            macro(k, PENDING)
        return
    S = d["summary"]
    Ks = sorted(S, key=int)
    hdr = ["Arm"] + [rf"$K{{=}}{k}$ ({S[k]['n_tasks']:.0f})" for k in Ks]
    for part, fn in (("results", "exp7"), ("transfer", "exp7_transfer")):
        rows = []
        for arm, label in ARMS7:
            if arm not in S[Ks[0]][part]:
                continue
            row = [label]
            for k in Ks:
                r = S[k][part][arm]
                row.append(f"{pc(r['real'],1)} / {pt(r['gap'],1)} / {f4(r['ece'],3)}")
            rows.append(row)
        table(f"{T}/{fn}.tex", hdr, rows, "l" + "r" * len(Ks))
    macro("NEnvMaxTasks", f"{max(S[k]['n_tasks'] for k in Ks):.0f}")
    macro("NEnvMinTasks", f"{min(S[k]['n_tasks'] for k in Ks):.0f}")
    macro("NEnvUniverses", str(S[Ks[0]]["universes"]))
    macro("NEnvSizes", ", ".join(Ks))
    macro("NEnvTrain", str(d["config"]["train_episodes"])); macro("NEnvEval", str(d["config"]["eval_episodes"]))
    big = Ks[-1]
    for arm, nm in (("lcb", "Lcb"), ("mean", "Mean"), ("exact", "Exact"), ("fixed", "Fixed"),
                    ("surface", "Surface"), ("oracle", "Oracle")):
        macro(f"NEnvBig{nm}", pc(S[big]["results"][arm]["real"], 1))
        if arm in S[big]["transfer"]:
            macro(f"NEnvBigTr{nm}", pc(S[big]["transfer"][arm]["real"], 1))
        macro(f"NEnvBigGap{nm}", pt(S[big]["results"][arm]["gap"], 1))
        macro(f"NEnvBigEce{nm}", f4(S[big]["results"][arm]["ece"], 3))
    rat = [S[k]["results"]["lcb"]["ratio_to_oracle"] for k in Ks]
    macro("NEnvRatioMin", f"{100*min(rat):.0f}"); macro("NEnvRatioMax", f"{100*max(rat):.0f}")
    gl = [abs(S[k]["results"]["lcb"]["gap"]) for k in Ks]
    macro("NEnvAbsGapLcbMax", f"{100*max(gl):.1f}")
    eces = [S[k]["results"]["lcb"]["ece"] for k in Ks]
    macro("NEnvEceLcbMax", f4(max(eces), 3))
    macro("NEnvBigK", big)


def exp8(d: Optional[dict], T: str) -> None:
    """ToolMaze pilot (external benchmark)."""
    if not d:
        for k in ("NTmTasks", "NTmTsrLlm", "NTmTsrHyb", "NTmHitsLlm", "NTmHitsHyb", "NTmInterv",
                  "NTmToolCallsLlm", "NTmToolCallsHyb", "NTmPairs", "NTmCalls", "NTmLatency", "NTmEarlyStop"):
            macro(k, PENDING)
        return
    runs = d["runs"]
    for arm, nm in (("llm", "Llm"), ("hybrid", "Hyb")):
        rs = [r for r in runs if r["arm"] == arm and "error" not in r]
        macro(f"NTmTsr{nm}", pc(sum(r["passed"] for r in rs) / max(1, len(rs)), 1))
        macro(f"NTmHits{nm}", str(sum(1 for r in rs if r.get("hit"))))
        macro(f"NTmToolCalls{nm}", f"{sum(r['tool_calls'] for r in rs)/max(1,len(rs)):.2f}")
    hy = [r for r in runs if r["arm"] == "hybrid"]
    macro("NTmTasks", str(len({r["task_id"] for r in runs})))
    macro("NTmInterv", str(sum(len(r.get("interventions") or []) for r in hy)))
    macro("NTmPairs", str(d["pretraining"]["pass_observations"]))
    macro("NTmCalls", f"{sum(r['llm_calls'] for r in runs)/max(1,len(runs)):.1f}")
    macro("NTmLatency", f"{sum(r['latency_per_call'] for r in runs)/max(1,len(runs)):.1f}")
    early = [r for r in runs if (r.get("failure_reason") or "").startswith("P2 failed: no perturbed")]
    macro("NTmEarlyStop", str(len(early)))
    macro("NTmRuns", str(len(runs)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../IEEE_Access/manuscript")
    args = ap.parse_args()
    T = os.path.join(args.out, "tables")
    os.makedirs(T, exist_ok=True)
    ref_universe(load("results_v4.json"), T)
    rho = load("rho_measured.json")
    abd = load("llm_results_real_ABD.json")
    dd = load("llm_results_real_D.json")          # arm D re-run separately
    if abd and dd:
        abd["results"].update(dd.get("results", {}))
        abd.setdefault("per_seed", {}).update(dd.get("per_seed", {}))
    exp4(load("llm_results_armC_v4.json"), abd, rho, T)
    exp6(load("results_recovery_v4.json"), load("results_recovery_llm.json"), T)
    exp7(load("results_envgen.json"), T)
    exp8(load("results_toolmaze_pilot.json"), T)
    with open(os.path.join(args.out, "numbers.tex"), "w", encoding="utf-8") as f:
        f.write("% generated by make_tex_numbers.py -- do not edit\n")
        for k in sorted(NUM):
            f.write(f"\\newcommand{{\\{k}}}{{{NUM[k]}}}\n")
    # every \N... macro used in the sections must be defined
    used = set()
    for root, _, files in os.walk(args.out):
        for fn in files:
            if fn.endswith(".tex") and fn != "numbers.tex":
                used |= set(re.findall(r"\\(N[A-Z][A-Za-z]*)", open(os.path.join(root, fn), encoding="utf-8").read()))
    missing = sorted(u for u in used if u not in NUM)
    print(f"wrote {len(NUM)} macros; undefined in text: {missing}")


if __name__ == "__main__":
    main()
