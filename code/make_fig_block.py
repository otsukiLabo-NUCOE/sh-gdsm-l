#!/usr/bin/env python3
"""Revision (2026-11): block diagram of SH-GD/SM-L (Fig. 2 of the revised article)."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out = sys.argv[1] if len(sys.argv) > 1 else "fig2_block.pdf"
plt.rcParams.update({"font.family": "serif", "font.size": 6.4, "mathtext.fontset": "dejavuserif"})
fig, ax = plt.subplots(figsize=(7.16, 3.3))
ax.set_xlim(0, 100)
ax.set_ylim(0, 42)
ax.axis("off")


def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec="#333333", lw=0.8))
    ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=7.0, fontweight="bold")
    ax.text(x + w / 2, y + h - 4.4, body, ha="center", va="top", fontsize=6.1, linespacing=1.3)


def arrow(x0, y0, x1, y1, text="", dx=0.0, dy=0.0, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.8,
                                 color="#333333", connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, text, ha="center", va="center",
                fontsize=6.0, style="italic")


box(1, 23, 20, 17, "Execution logs",
    "per call: tool, input passed\n($x$ for $s$), execution\noutcome, validator\nverdicts, retry outcomes",
    "#eef3fb")
box(25.5, 23, 22, 17, "Learner",
    "rel($x\\to s$): Beta posterior\nper schema pair\n$\\hat p_e$: execution reliability\n$\\hat r_e$: retry success\n(noise model, soft attribution)",
    "#e8f4ea")
box(52, 23, 22, 17, "Monitor + planner",
    "state $S=(\\Sigma,F)$\nLCB point estimate\nAND/OR value $V(S)$\nutility $U$ of each action",
    "#fff6e0")
box(78.5, 23, 21, 17, "Transition rule",
    "substitute: $\\arg\\max U$\nretry: $U_{retry}>U_{alt}$\nhand over: no candidate\nor $\\max U<u_{min}$",
    "#fde9e7")
box(25.5, 2, 22, 15, "Calibration monitor",
    "windowed (claim, outcome)\npairs; over-claim alarm\n$\\rightarrow$ discount counts", "#e6f4f7")
box(52, 2, 22, 15, "Executor + validator",
    "run the tool with the\nchosen substitution;\nvalidate its outputs", "#f1f1f1")
box(78.5, 2, 21, 15, "LLM (optional)",
    "fills natural-language\narguments; decides\nonly after a hand-over", "#efe8f8")

arrow(21.4, 31.5, 25.1, 31.5)
arrow(47.9, 31.5, 51.6, 31.5)
arrow(74.4, 31.5, 78.1, 31.5)
arrow(84, 22.6, 70, 17.4, "action", dx=-3.0, dy=1.4)
arrow(93, 22.6, 93, 17.4, "hand-over", dx=-4.4)
arrow(78.1, 9.5, 74.4, 9.5)
arrow(51.6, 9.5, 47.9, 9.5)
arrow(36.5, 17.4, 36.5, 22.6, "alarm", dx=3.0, dy=-0.4)
# executor -> logs: routed through the gap between the rows
ax.plot([56, 56, 8], [17.4, 20.0, 20.0], color="#333333", lw=0.8)
arrow(8, 20.0, 8, 22.6)
ax.text(17, 21.0, "outcomes logged", ha="center", va="center", fontsize=6.0, style="italic")
fig.savefig(out, bbox_inches="tight", pad_inches=0.02, dpi=220)
print("wrote", out)
