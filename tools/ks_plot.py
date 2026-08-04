#!/usr/bin/env python3
"""ks_plot.py -- the kappa_shear portability figures.

    source conda_env.sh
    tools/ks_plot.py                # -> figures/*.png and *.pdf, light
    tools/ks_plot.py --dark         # slide/dark-background variants
    tools/ks_plot.py --show fig1    # render one figure to the screen

Reads through tools/ks_collect.py, never the raw CSVs -- the collector holds the
filters that separate a result from an artifact (quarantined runs, the blocked
variant masquerading as faithful, grid collisions). Plotting the raw directory
reproduces every one of those bugs.

DESIGN NOTES, so the next edit does not undo them:
  * The palette is validated for colour-vision deficiency (OKLab dE >= 8 on
    adjacent pairs, >= 15 normal-vision) in both light and dark. Do not add a
    fifth series colour without re-validating -- four is the ceiling this
    ordering clears.
  * Series identity is never colour-alone: every line is directly labelled at
    its right-hand end, and a legend is present. Labels are set in ink, not in
    the series colour, with the coloured line end adjacent to carry identity.
  * Where six toolchains appear at once, four are context grey and only the two
    that carry the finding get colour. Six categorical hues would be unreadable
    and would breach the palette's series ceiling.
  * Ratio panels draw an explicit 1.0 baseline; a value above/below it is the
    whole point, so the reader must never have to infer where 1.0 sits.
"""
import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "figures")

# Validated categorical slots (see tools/ notes + dataviz validator).
LIGHT = dict(s=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
             paper="#ffffff", ink="#0d1217", ink2="#55616c", ink3="#7e8a95",
             rule="#e3e8ec", ctx="#b9c2ca")
DARK = dict(s=["#3987e5", "#d95926", "#199e70", "#c98500"],
            paper="#14181c", ink="#eef2f5", ink2="#9fabb5", ink3="#6d7883",
            rule="#28313a", ctx="#48535d")


def style(T):
    plt.rcParams.update({
        "figure.facecolor": T["paper"], "axes.facecolor": T["paper"],
        "savefig.facecolor": T["paper"],
        "text.color": T["ink"], "axes.labelcolor": T["ink2"],
        "xtick.color": T["ink3"], "ytick.color": T["ink3"],
        "axes.edgecolor": T["rule"], "grid.color": T["rule"],
        "font.family": "sans-serif", "font.size": 10.5,
        "axes.titlesize": 12.5, "axes.titleweight": "600",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.linewidth": 0.8, "grid.alpha": 0.9,
        "lines.linewidth": 2.0, "lines.markersize": 5.5,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "figure.dpi": 140,
    })


def data():
    out = subprocess.run([sys.executable, os.path.join(HERE, "ks_collect.py"), "--json"],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def finish(ax, T, title, sub, xlab, ylab, xs, logx=False):
    # Title ABOVE the subtitle with room for both: a small pad puts the title
    # baseline straight through the subtitle text.
    ax.set_title(title, loc="left", pad=30, color=T["ink"])
    if sub:
        ax.text(0, 1.025, sub, transform=ax.transAxes, fontsize=9.5,
                color=T["ink3"], va="bottom")
    ax.set_xlabel(xlab, fontsize=9.5)
    ax.set_ylabel(ylab, fontsize=9.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in xs])
    if logx:
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, p: f"{int(v)}"))
        ax.minorticks_off()
    ax.set_axisbelow(True)
    # nz = 25 and 30 sit close on a linear axis; shrink the tick text rather
    # than dropping ticks, so every measured depth stays marked.
    ax.tick_params(length=0, labelsize=9)


def label_ends(ax, T, xs, series, fs=9.5, only=None):
    """Direct labels in INK at the right-hand end of each line: the coloured line
    end sits immediately beside, so identity is carried without colouring text.

    DE-COLLIDED. Series that converge -- which is exactly what the frame-penalty
    panel shows, three GPUs sitting on 1.0 -- would otherwise stack their labels
    into an unreadable blob. Push them apart to a minimum spacing and draw a
    leader from the line end to the moved label.
    """
    ends = []
    for name, ys, col, ctx in series:
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not pts or ctx or (only is not None and name not in only):
            continue
        ends.append([pts[-1][0], pts[-1][1], name, col])
    if not ends:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.062
    ends.sort(key=lambda e: e[1])
    ypos = [min(max(e[1], lo), hi) for e in ends]
    for i in range(1, len(ypos)):                      # push up from the bottom
        if ypos[i] - ypos[i - 1] < gap:
            ypos[i] = ypos[i - 1] + gap
    over = ypos[-1] - (hi - gap * 0.3)                 # then pull the stack back down
    if over > 0:
        ypos = [y - over for y in ypos]
        for i in range(len(ypos) - 2, -1, -1):
            if ypos[i + 1] - ypos[i] < gap:
                ypos[i] = ypos[i + 1] - gap
    for (x, y, name, col), yl in zip(ends, ypos):
        ax.annotate(name, (x, yl), textcoords="offset points", xytext=(11, 0),
                    va="center", ha="left", fontsize=fs, color=T["ink"],
                    fontweight="600", annotation_clip=False)
        if abs(yl - y) > gap * 0.25:                   # leader only when moved
            ax.annotate("", xy=(x, y), xytext=(x, yl), annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=col,
                                        lw=0.9, alpha=0.65,
                                        shrinkA=1, shrinkB=3))


def plot_lines(ax, T, xs, series, baseline=None, clip=None):
    if baseline is not None:
        ax.axhline(baseline, color=T["ink3"], lw=1.0, ls=(0, (3, 3)), zorder=1)
    for name, ys, col, ctx in series:
        xv = [x for x, y in zip(xs, ys) if y is not None]
        yv = [y for y in ys if y is not None]
        if clip:
            yv = [min(v, clip) for v in yv]
        ax.plot(xv, yv, color=(T["ctx"] if ctx else col), lw=1.4 if ctx else 2.0,
                marker="o", ms=4.0 if ctx else 5.5, zorder=2 if ctx else 3,
                markeredgecolor=T["paper"], markeredgewidth=1.2,
                label=name, alpha=0.9 if ctx else 1.0)
        if clip:  # flag anything pinned to the ceiling, with its true value
            for x, y in zip(xs, ys):
                if y is not None and y > clip:
                    ax.annotate(f"{y:.1f}×", (x, clip), textcoords="offset points",
                                xytext=(0, 9), ha="center", fontsize=9,
                                color=col, fontweight="600")


def build(D, T, dark):
    figs = {}
    nz = D["nz"]
    S = T["s"]

    # 1 — the headline: is the GPU worth it?
    f, ax = plt.subplots(figsize=(7.6, 4.3))
    se = [("MI250X", D["gpu_vs_cpu"]["AMD MI250X / EPYC 7A53"], S[2], False),
          ("Intel Max", D["gpu_vs_cpu"]["Intel Max / Xeon Max"], S[3], False)]
    plot_lines(ax, T, nz, se, baseline=1.0)
    ax.fill_between(nz, 0, 1, color=T["ink3"], alpha=0.05, zorder=0)
    ax.set_ylim(0.4, 1.5)
    finish(ax, T, "Is the GPU worth it?",
           "each GPU against the CPU in the same node, same compiler, same source", "nz",
           "GPU speedup vs same-node CPU", nz)
    ax.annotate("GPU loses", (nz[0], 0.99), xytext=(6, -13), textcoords="offset points",
                ha="left", fontsize=9, color=T["ink3"], style="italic")
    label_ends(ax, T, nz, se)
    ax.legend(["MI250X (1 GCD) / EPYC 7A53, 56 threads",
               "Intel Max (1 tile) / Xeon Max, 104 threads"], loc="lower left", ncols=1)
    figs["fig1_gpu_vs_cpu"] = f

    # 2 — depth curve
    f, ax = plt.subplots(figsize=(7.6, 4.3))
    se = [("V100", D["gpu_ns"]["NVIDIA V100"], S[0], False),
          ("GH200", D["gpu_ns"]["NVIDIA GH200"], S[1], False),
          ("MI250X", D["gpu_ns"]["AMD MI250X"], S[2], False),
          ("Intel Max", D["gpu_ns"]["Intel Max"], S[3], False)]
    plot_lines(ax, T, nz, se)
    ax.set_ylim(0, None)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,}"))
    finish(ax, T, "Cost per column as columns get deeper",
           "do concurrent, 473×297, frame fitted to the column", "nz", "ns per column", nz)
    label_ends(ax, T, nz, se)
    ax.legend(loc="upper left", ncols=2)
    figs["fig2_depth"] = f

    # 3 — the frame penalty
    f, ax = plt.subplots(figsize=(7.6, 4.3))
    se = [("V100", D["frame"]["NVIDIA V100"], S[0], False),
          ("GH200", D["frame"]["NVIDIA GH200"], S[1], False),
          ("MI250X", D["frame"]["AMD MI250X"], S[2], False),
          ("Intel Max", D["frame"]["Intel Max"], S[3], False)]
    plot_lines(ax, T, nz, se, baseline=1.0, clip=3.4)
    ax.set_ylim(0.8, 3.9)
    finish(ax, T, "One compile-time constant costs AMD 3×, and nobody else anything",
           "NZ_STACK_MAX=128 (production) ÷ fitted to nz+1 — identical work, identical answers",
           "nz", "cost ratio, production ÷ fitted", nz)
    label_ends(ax, T, nz, se, only={"MI250X"})
    ax.annotate("V100, GH200 and Intel Max sit on 1.0", (nz[3], 1.0),
                textcoords="offset points", xytext=(0, -22), ha="center",
                fontsize=9, color=T["ink3"], style="italic")
    ax.legend(loc="upper right", ncols=2)
    figs["fig3_frame"] = f

    # 4 — DC costs nothing on CPU; emphasis on the flang family
    f, ax = plt.subplots(figsize=(7.6, 4.3))
    order = ["flang 22.1.5", "amdflang 23.0.0", "nvfortran", "ifx 2026.0.0",
             "ifx 2025.3.2", "gfortran 16.1.0"]
    se = [(k, D["dc_cost"][k], S[1] if i == 0 else S[2] if i == 1 else T["ctx"], i >= 2)
          for i, k in enumerate(order) if k in D["dc_cost"]]
    plot_lines(ax, T, nz, se, baseline=1.0)
    ax.set_ylim(0.94, 1.22)
    finish(ax, T, "do concurrent itself is free — the offload lowering is not",
           "serial do concurrent ÷ plain nested do loops, no offload involved",
           "nz", "dc_serial ÷ serial_do", nz)
    label_ends(ax, T, nz, se)
    ax.legend(loc="upper right", ncols=2, fontsize=8.5)
    figs["fig4_dc_cost"] = f

    # 5 — thread scaling.
    # THREE colours, not five. Five series would cycle the palette (two lines the
    # same hue) and breach the validated ceiling. The three Broadwell compilers
    # sit on top of each other anyway, so one carries the colour and the other two
    # are context grey -- which is also the honest reading: the machines differ,
    # the compilers on a given machine barely do.
    f, ax = plt.subplots(figsize=(7.6, 4.3))
    PICK = {"Xeon Max / ifx": (S[3], False),
            "EPYC 7A53 / amdflang": (S[2], False),
            "Broadwell / ifx": (S[0], False)}
    xs = sorted({t for k in D["threads"] for t, _ in D["threads"][k]})
    se = []
    for k in sorted(D["threads"], key=lambda k: k not in PICK):
        col, ctx = PICK.get(k, (T["ctx"], True))
        m = dict(D["threads"][k])
        se.append((k, [m.get(t) for t in xs], col, ctx))
    plot_lines(ax, T, xs, se)
    ax.set_ylim(0, None)
    ticks = [1, 2, 4, 8, 16, 32, 56, 104]
    finish(ax, T, "The CPU lanes scale",
           "same source, three threading mechanisms; speedup vs each machine's own 1 thread, nz=30",
           "threads", "speedup vs 1 thread", ticks, logx=True)
    label_ends(ax, T, xs, se)
    ax.legend(loc="upper left", ncols=1, fontsize=8.5)
    figs["fig5_threads"] = f

    for f in figs.values():
        f.tight_layout()
    return figs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dark", action="store_true")
    a = ap.parse_args()
    T = DARK if a.dark else LIGHT
    style(T)
    os.makedirs(OUT, exist_ok=True)
    figs = build(data(), T, a.dark)
    suf = "_dark" if a.dark else ""
    for name, f in figs.items():
        for ext in ("png", "pdf"):
            p = os.path.join(OUT, f"{name}{suf}.{ext}")
            f.savefig(p, bbox_inches="tight")
        print("wrote", os.path.join("figures", f"{name}{suf}.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
