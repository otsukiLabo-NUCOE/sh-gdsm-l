#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures of the IEEE Access manuscript (vector PDF, English labels).

    python make_figures_v4.py --out ../IEEE_Access/manuscript/figures

fig1_linear_vs_graph.pdf   Fig. 1  linear vs non-linear task list
fig3_learning_curve.pdf    learning curve (claim/realized, ECE, rel error)
fig4_recovery.pdf          Experiment 6 (needs results_recovery_v4.json)
fig5_scale.pdf             Experiment 7 (needs results_envgen.json)
"""
from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#1f5fa8", "#d9822b", "#2e8b57", "#c0392b", "#7f7f7f", "#7b4fa0"


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def box(ax, x, y, text, w=0.9, h=0.42):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec=BLUE, lw=1.0))
    ax.text(x, y, text, ha="center", va="center", fontsize=8, fontweight="bold")


def arrow(ax, a, b, color=BLUE):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                                                    shrinkA=9, shrinkB=9))


def fig1(out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 1.9), gridspec_kw=dict(width_ratios=[1, 2.4]))
    for ax in (a1, a2):
        ax.set_axis_off()
    a1.set_xlim(-0.6, 0.6); a1.set_ylim(-0.4, 2.6)
    for i, t in enumerate(["TaskA", "TaskB", "TaskC"]):
        box(a1, 0, 2.2 - i * 1.0, t)
    arrow(a1, (0, 2.2), (0, 1.2)); arrow(a1, (0, 1.2), (0, 0.2))
    a1.set_title("(a) Linear", fontsize=8)
    a2.set_xlim(-0.2, 3.4); a2.set_ylim(-0.4, 2.6)
    pos = {"TaskA": (0.4, 2.2), "TaskB": (2.8, 2.0), "TaskC": (1.4, 1.2), "TaskD": (2.8, 1.0),
           "TaskE": (0.4, 0.1), "TaskF": (2.0, 0.1)}
    for t, (x, y) in pos.items():
        box(a2, x, y, t, w=0.8, h=0.38)
    for a, b in [("TaskA", "TaskC"), ("TaskB", "TaskC"), ("TaskB", "TaskD"), ("TaskC", "TaskE"),
                 ("TaskD", "TaskF")]:
        arrow(a2, pos[a], pos[b])
    for a, b in [("TaskA", "TaskE"), ("TaskC", "TaskF")]:
        arrow(a2, pos[a], pos[b], RED)
    a2.set_title("(b) Non-linear workflow", fontsize=8)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(out, "fig1_linear_vs_graph.pdf"))
    plt.close(fig)


def fig3(out, d):
    e1 = d["experiment1"]
    x = [r["episodes"] for r in e1]
    opt = d["optimal_value"]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.1))
    a = axs[0]
    for key, lab, c in (("claim", "Claimed", ORANGE), ("real", "Realized", BLUE)):
        m = [100 * r[key] for r in e1]; h = [100 * r[key + "_ci"] for r in e1]
        a.plot(x, m, "o-", color=c, ms=2.5, lw=1, label=lab)
        a.fill_between(x, [u - v for u, v in zip(m, h)], [u + v for u, v in zip(m, h)], color=c, alpha=0.18, lw=0)
    a.axhline(100 * opt, ls="--", color=GREY, lw=0.8, label="Optimal (exact DP)")
    a.set_xscale("symlog", linthresh=50); a.set_xlabel("Training episodes"); a.set_ylabel("Completion [%]")
    a.set_title("(a) Claimed vs realized", fontsize=8); a.legend(fontsize=6.5, frameon=False)
    a = axs[1]
    em = d.get("experiment1_mean")
    a.plot(x, [r["ece"] for r in e1], "o-", color=BLUE, ms=2.5, lw=1, label="LCB")
    if em:
        a.plot(x, [r["ece"] for r in em], "s--", color=GREEN, ms=2.5, lw=1, label=r"Mean, $\theta$=0.12")
    a.set_xscale("symlog", linthresh=50); a.set_xlabel("Training episodes"); a.set_ylabel("ECE over decision states")
    a.set_title("(b) Calibration", fontsize=8); a.legend(fontsize=6.5, frameon=False)
    a = axs[2]
    for key, lab, c in (("rel_err_all", "All 110 pairs", GREY), ("rel_err_nonzero", r"Pairs with $\rho^*>0$", PURPLE)):
        m = [r[key] for r in e1]; h = [r[key + "_ci"] for r in e1]
        a.plot(x, m, "o-", color=c, ms=2.5, lw=1, label=lab)
        a.fill_between(x, [u - v for u, v in zip(m, h)], [u + v for u, v in zip(m, h)], color=c, alpha=0.18, lw=0)
    a.set_xscale("symlog", linthresh=50); a.set_xlabel("Training episodes"); a.set_ylabel(r"Mean $|\mathrm{rel}-\rho^*|$ (posterior mean)")
    a.set_title("(c) Estimation error", fontsize=8); a.legend(fontsize=6.5, frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "fig3_learning_curve.pdf"))
    plt.close(fig)


def ttr(r, tau_tool, tau_llm):
    return r["steps_to_recover"] * tau_tool + r["calls_to_recover"] * tau_llm + \
        r["steps_to_recover"] * r["decision_ms"] / 1000.0


def fig4(out, d):
    m = d["main"]
    keys = [k for k in sorted(m) if all(isinstance(m[k][a]["steps_to_recover"], (int, float)) and
                                        not math.isnan(m[k][a]["steps_to_recover"]) for a in ("C", "A", "B", "A*"))]
    arms = [("C", "C  SH-GD/SM-L", BLUE), ("A", "A  delegated", ORANGE), ("B", "B  delegated + retry", RED),
            ("A*", r"A$^*$ delegated ($\rho^*$)", GREEN)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.2))
    w = 0.2
    for j, (arm, lab, c) in enumerate(arms):
        xs = [i + (j - 1.5) * w for i in range(len(keys))]
        a1.bar(xs, [m[k][arm]["wasted_to_recover"] for k in keys], w,
               yerr=[1.96 * 0 + m[k][arm].get("wasted_to_recover_ci", 0) for k in keys],
               color=c, label=lab, error_kw=dict(lw=0.6, capsize=1.5))
    a1.set_xticks(range(len(keys))); a1.set_xticklabels(keys)
    a1.set_ylabel("Failed executions until recovery"); a1.set_title("(a) Wasted executions", fontsize=8)
    a1.legend(fontsize=6.5, frameon=False)
    taus = [0.1 * i for i in range(0, 61)]
    for arm, lab, c in arms:
        ys = [sum(ttr(m[k][arm], 0.5, t) for k in keys) / len(keys) for t in taus]
        a2.plot(taus, ys, color=c, lw=1.1, label=lab)
    a2.set_xlabel(r"Latency of one model call $\tau_{\mathrm{LLM}}$ [s]")
    a2.set_ylabel(r"Time to recover $T_{\mathrm{rec}}$ [s]")
    a2.set_title(r"(b) $T_{\mathrm{rec}}$ vs $\tau_{\mathrm{LLM}}$ ($\tau_{\mathrm{tool}}$ = 0.5 s)", fontsize=8)
    a2.legend(fontsize=6.5, frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "fig4_recovery.pdf"))
    plt.close(fig)


def fig5(out, d):
    S = d["summary"]
    Ks = sorted(S, key=int)
    n = [S[k]["n_tasks"] for k in Ks]
    arms = [("lcb", "SH-GD/SM-L (LCB)", BLUE, "o-"), ("mean", r"SH-GD/SM-L (mean, $\theta$=0.12)", GREEN, "s--"),
            ("exact", "Exact match, dynamic", GREY, "^-"), ("fixed", "Plan fixed at $S_0$", PURPLE, "v:"),
            ("surface", "Delegated proxy (names)", ORANGE, "d-."), ("oracle", "Oracle (true values)", "k", "x--")]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.55))
    for arm, lab, c, st in arms:
        for ax, part, key, scale in ((axs[0], "results", "real", 100), (axs[1], "transfer", "real", 100),
                                     (axs[2], "results", "ece", 1)):
            if arm not in S[Ks[0]][part]:
                continue
            ys = [scale * S[k][part][arm][key] for k in Ks]
            hs = [scale * S[k][part][arm][key + "_ci"] for k in Ks]
            ax.errorbar(n, ys, yerr=hs, fmt=st, color=c, ms=3, lw=1, capsize=1.5, label=lab)
    axs[0].set_title("(a) Completion", fontsize=8); axs[0].set_ylabel("Realized completion [%]")
    axs[1].set_title("(b) After withdrawing a dedicated tool", fontsize=8)
    axs[2].set_title("(c) Calibration", fontsize=8); axs[2].set_ylabel("ECE over decision states")
    for ax in axs:
        ax.set_xlabel("Number of tasks in the universe")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=6.2, frameon=False, handlelength=2.2)
    fig.tight_layout(pad=0.3, rect=(0, 0.09, 1, 1))
    fig.savefig(os.path.join(out, "fig5_scale.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../IEEE_Access/manuscript/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig1(args.out)
    d = load("results_v4.json")
    if d:
        fig3(args.out, d)
    r = load("results_recovery_v4.json")
    if r:
        fig4(args.out, r)
    e = load("results_envgen.json")
    if e and e.get("summary"):
        fig5(args.out, e)
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
