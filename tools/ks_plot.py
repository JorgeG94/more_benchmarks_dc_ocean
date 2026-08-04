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
import math
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

    # 1 — every GPU against the whole CPU in its own node.
    # LOG y: the pairs span 0.6x to 5.9x, and on a ratio axis log puts 1.0 in the
    # middle so "twice as fast" and "half as fast" are the same visual distance.
    f, ax = plt.subplots(figsize=(7.6, 4.5))
    order = ["NVIDIA GH200 / Grace, 72 threads", "NVIDIA V100 / Broadwell, 40 threads",
             "AMD MI250X, 1 GCD / EPYC 7A53, 56 thr", "Intel Max, 1 tile / Xeon Max, 104 thr"]
    miss = [k for k in order if k not in D["gpu_vs_cpu"]]
    if miss:
        raise SystemExit(f"fig1: missing pairs {miss}")
    se = [(k, D["gpu_vs_cpu"][k], S[i2], False) for i2, k in enumerate(order)]
    plot_lines(ax, T, nz, se, baseline=1.0)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 7.0)
    ax.set_yticks([0.5, 1, 2, 3, 5, 7])
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}×"))
    ax.minorticks_off()
    ax.fill_between(nz, 0.5, 1, color=T["ink3"], alpha=0.06, zorder=0)
    finish(ax, T, "Single GPU speedup versus whole CPU",
           "same do concurrent source on both sides; only the GH200/Grace pair is measured\n"
           "at the production grid on BOTH sides — the rest run the CPU at 64², which "
           "flatters it by ~10–18%",
           "nz", "GPU ÷ whole-CPU-socket", nz)
    ax.annotate("GPU slower", (nz[0], 0.985), xytext=(6, -14), textcoords="offset points",
                ha="left", fontsize=9, color=T["ink3"], style="italic")
    # Identity via the legend only -- the pair names are long and duplicating them
    # as end labels crowds the right margin for no extra information.
    hs = {l.get_label(): l for l in ax.get_lines()}
    ax.legend([hs[k] for k in order], order, loc="upper right", ncols=1, fontsize=8.5)
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
    finish(ax, T, "Time per column with increasing number of layers",
           "do concurrent, 473×297, frame fitted to the column", "nz", "ns per column", nz)
    ax.legend(loc="upper left", ncols=2)
    figs["fig2_depth"] = f

    # 3 — the frame penalty
    f, ax = plt.subplots(figsize=(7.6, 4.3))
    se = [("V100", D["frame"]["NVIDIA V100"], S[0], False),
          ("GH200", D["frame"]["NVIDIA GH200"], S[1], False),
          ("MI250X", D["frame"]["AMD MI250X"], S[2], False),
          ("Intel Max", D["frame"]["Intel Max"], S[3], False)]
    # LOG y instead of clipping. The MI250X curve runs 14.8x -> 1.0 while the
    # other three sit inside 0.98-1.13; on a linear axis one of those is a flat
    # line at the bottom and the finding for the other three is invisible.
    plot_lines(ax, T, nz, se, baseline=1.0)
    ax.set_yscale("log")
    ax.set_ylim(0.85, 20)
    ax.set_yticks([1, 1.5, 2, 3, 5, 10, 15])
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}×"))
    ax.minorticks_off()
    finish(ax, T, "Cost ratio per GPU (MI250X affected by stack max constant)",
           "NZ_STACK_MAX is a COMPILE-TIME bound, so it sets the per-thread stack\n"
           "frame whatever the runtime nz — identical work, identical answers",
           "nz", "cost ratio, production ÷ fitted", nz)
    # ABOVE the baseline, not below it: at (0, -22) this note landed on the x
    # tick labels and across the three lines it is describing.
    ax.annotate("V100, GH200 and Intel Max sit on 1.0", (nz[-1], 1.0),
                textcoords="offset points", xytext=(-4, 34), ha="right",
                fontsize=9, color=T["ink3"], style="italic")
    lg = ax.legend(loc="upper right", ncols=2, title="cost ratio  =  t(NZ_STACK_MAX=128)  ÷  t(NZ_STACK_MAX=nz+1)")
    lg.get_title().set_fontsize(9)
    lg.get_title().set_color(T["ink2"])
    figs["fig3_frame"] = f

    # 3a — the same ratio from nz=50 on, LINEAR. Once MI250X has converged, the
    # log axis fig3 needs to show its 14.8x compresses everything into a band
    # around 1.0 and the residual differences between the four are unreadable.
    # This is the deep-water half of the same measurement, at a scale that can
    # resolve it: the claim here is that AMD has JOINED the others, and a reader
    # should be able to see by how much rather than take it on trust.
    f, ax = plt.subplots(figsize=(7.6, 4.0))
    j0 = nz.index(50)
    nzd = nz[j0:]
    se = [("V100", D["frame"]["NVIDIA V100"][j0:], S[0], False),
          ("GH200", D["frame"]["NVIDIA GH200"][j0:], S[1], False),
          ("MI250X", D["frame"]["AMD MI250X"][j0:], S[2], False),
          ("Intel Max", D["frame"]["Intel Max"][j0:], S[3], False)]
    plot_lines(ax, T, nzd, se, baseline=1.0)
    ax.set_ylim(0.9, 1.12)
    finish(ax, T, "Cost ratio per GPU, deep columns only (nz \u2265 50)",
           "linear scale: by nz=50 the MI250X frame penalty is gone\n"
           "and all four agree to within a few percent",
           "nz", "cost ratio, production \u00f7 fitted", nzd)
    lg = ax.legend(loc="upper left", ncols=4, fontsize=9, title="cost ratio  =  t(NZ_STACK_MAX=128)  ÷  t(NZ_STACK_MAX=nz+1)")
    lg.get_title().set_fontsize(9)
    lg.get_title().set_color(T["ink2"])
    figs["fig3a_frame_deep"] = f

    # 4 — DC costs nothing on CPU; emphasis on the flang family
    f, ax = plt.subplots(figsize=(7.6, 4.3))
    # The order is hand-set (the two flang-family lines carry the emphasis), so
    # it must be RECONCILED against the data rather than intersected with it.
    # `if k in D[...]` silently drops any series the list has not been told
    # about -- which is how a whole machine can be measured, committed, and then
    # simply not appear on the chart that is supposed to survey every machine.
    order = ["Broadwell / flang 22.1.5", "EPYC 7A53 / amdflang 23.0.0",
             "Broadwell / nvfortran", "Broadwell / ifx 2026.0.0",
             "Xeon Max / ifx 2025.3.2", "Broadwell / gfortran 16.1.0",
             "Mac / gfortran 15.2.0", "Grace / nvfortran 26.3"]
    missing = [k for k in order if k not in D["dc_cost"]]
    unlisted = [k for k in D["dc_cost"] if k not in order]
    if missing or unlisted:
        raise SystemExit(f"fig4 dc_cost series mismatch:\n  in `order`, absent from the data: "
                         f"{missing}\n  measured, absent from `order`: {unlisted}")
    # INVERTED to a speedup. D["dc_cost"] is dc_serial / serial_do in TIME, so a
    # value above 1 there means do concurrent was SLOWER. Plotted as
    # serial_do / dc_serial instead, >1 is genuinely faster and the axis label
    # means what it says.
    se = [(k, [(1.0 / v if v else None) for v in D["dc_cost"][k]],
           S[1] if i == 0 else S[2] if i == 1 else T["ctx"], i >= 2)
          for i, k in enumerate(order)]
    plot_lines(ax, T, nz, se, baseline=1.0)
    ax.set_ylim(0.82, 1.07)
    finish(ax, T, "do concurrent on the CPU (serial / threaded) effects",
           "serial do concurrent against plain nested do loops, no offload involved\n"
           "1.0 = identical; below 1.0 = do concurrent is slower",
           "nz", "speedup over serial do", nz)
    # No end labels: they repeat the legend verbatim and crowd the converged tail.
    ax.legend(loc="upper right", ncols=2, fontsize=8.5)
    figs["fig4_dc_cost"] = f

    # 5 — classic strong scaling: speedup against threads, with the ideal drawn
    # as a reference. LOG-LOG, so ideal is a straight 45-degree line and the gap
    # to it is read as vertical distance at any width -- on linear axes the
    # low-thread end collapses into the corner and the ideal curves away.
    f, ax = plt.subplots(figsize=(7.6, 4.6))
    PICK = {"Grace / nvfortran": (S[1], False),
            "Xeon Max / ifx": (S[3], False),
            "EPYC 7A53 / amdflang": (S[2], False),
            "Broadwell / ifx": (S[0], False)}
    xs = sorted({t for k in D["threads"] for t, _ in D["threads"][k]})
    se = []
    for k in sorted(D["threads"], key=lambda k: k not in PICK):
        col, ctx = PICK.get(k, (T["ctx"], True))
        m = dict(D["threads"][k])
        se.append((k, [m.get(t) for t in xs], col, ctx))
    ax.plot(xs, xs, color=T["ink3"], lw=1.0, ls=(0, (3, 3)), zorder=1, label="_ideal")
    plot_lines(ax, T, xs, se)
    ax.set_xscale("log"); ax.set_yscale("log")
    yt = [1, 2, 4, 8, 16, 32, 64, 128]
    ax.set_yticks(yt)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda v, p: f"{int(v)}"))
    ax.set_ylim(0.9, 150)
    ticks = [1, 2, 4, 8, 16, 32, 72, 104]
    finish(ax, T, "CPU scaling",
           "strong scaling at nz=30, one socket; dashed line is ideal (speedup = threads)",
           "threads", "speedup vs 1 thread", ticks, logx=True)
    ax.minorticks_off()
    ax.annotate("ideal", (xs[-1], xs[-1]), xytext=(-4, 9), textcoords="offset points",
                ha="right", fontsize=9, color=T["ink3"], style="italic")
    ax.legend(loc="upper left", ncols=2, fontsize=8.5)
    figs["fig5_threads"] = f

    # 6 - the only same-grid, whole-device-vs-whole-socket comparison in the set.
    # Separate panel, not a third line on fig1: fig1's unit is one GCD / one tile
    # against an entire CPU, and overlaying a whole-GPU ratio would invite the
    # reader to compare numbers that are not the same measurement.
    f, ax = plt.subplots(figsize=(7.6, 4.3))
    se = [("do concurrent", D["node_gh200"], S[1], False),
          ("CUDA", D["node_gh200_cuda"], S[0], False)]
    # THE BAND IS THE FINDING, not an uncertainty. Its vertical extent at each nz
    # is exactly what hand-CUDA buys over do concurrent on the same kernel, same
    # inputs, same device -- shading it makes that gap a thing you read off the
    # chart instead of subtracting two curves by eye. Widening with depth IS the
    # result: 1.39x at nz=10 growing to 1.74x at nz=100.
    dcv, cuv = D["node_gh200"], D["node_gh200_cuda"]
    ax.fill_between(nz, dcv, cuv, color=S[0], alpha=0.10, zorder=1, lw=0)
    plot_lines(ax, T, nz, se, baseline=1.0)
    for j in (0, len(nz) - 1):
        ax.annotate(f"{cuv[j] / dcv[j]:.2f}×", (nz[j], (dcv[j] + cuv[j]) / 2),
                    xytext=(9 if j == 0 else -9, 0), textcoords="offset points",
                    ha="left" if j == 0 else "right", va="center",
                    fontsize=9.5, fontweight="600", color=S[0])
    ax.annotate("shaded band = what CUDA buys over do concurrent",
                (nz[2], (dcv[2] + cuv[2]) / 2), xytext=(6, 26),
                textcoords="offset points", ha="left", fontsize=9,
                color=T["ink3"], style="italic")
    ax.set_ylim(0, None)
    finish(ax, T, "GH200: Grace versus H200",
           "both sides at production 473\u00d7297, nvfortran, frame at NZ_STACK_MAX=128",
           "nz", "GPU speedup vs 72 Grace cores", nz)
    ax.legend(loc="upper right", ncols=1)
    figs["fig6_node"] = f

    # 7 - performance portability: ONE source, every target it ran on.
    # A bar chart, not lines: this is a magnitude comparison across ~5 named
    # things at one depth, which is exactly what bars are for. Depth is nz=50, a
    # real ocean-model layer count and the middle of the swept range.
    # Colour encodes the one distinction that carries the finding -- accelerator
    # vs CPU socket -- and NOT device identity, which the axis labels already
    # give. Five categorical hues here would be decoration.
    f, ax = plt.subplots(figsize=(7.6, 4.0))
    j = nz.index(50)
    bars = sorted(((v[j], k) for k, v in D["portability"].items() if v[j]))
    ys = range(len(bars))
    cols = [S[1] if D["portability_is_cpu"][k] else S[0] for _, k in bars]
    ax.barh(list(ys), [v for v, _ in bars], height=0.52, color=cols, zorder=3)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([k for _, k in bars], fontsize=10)
    ax.invert_yaxis()                       # fastest at the top
    top = max(v for v, _ in bars)
    ax.set_xlim(0, top * 1.20)
    for y, (v, _) in zip(ys, bars):         # direct value labels, in ink
        ax.annotate(f"{v:,.0f}", (v, y), xytext=(7, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=9.5, color=T["ink"],
                    fontweight="600")
    ax.grid(axis="y", visible=False)
    # Round tick values, not top/4: an axis reading 0, 161, 322, 483 makes the
    # reader do arithmetic to place a bar, which is the axis's job.
    step = 10 ** int(math.floor(math.log10(top / 4)))
    for m in (1, 2, 2.5, 5, 10):
        if top / 4 <= step * m:
            step *= m
            break
    ticks = [int(t) for t in range(0, int(top) + int(step), int(step))]
    finish(ax, T, "Do concurrent performance portability",
           "the same do concurrent kernel at nz=50, production 473\u00d7297, "
           "frame fitted \u2014 lower is better",
           "ns per column", "", ticks)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,}"))
    ax.annotate("a whole CPU socket, second", (bars[1][0], 1), xytext=(52, 0),
                textcoords="offset points", ha="left", va="center", fontsize=9,
                color=T["ink3"], style="italic")
    figs["fig7_portability"] = f

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
