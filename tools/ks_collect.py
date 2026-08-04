#!/usr/bin/env python3
"""ks_collect.py -- fold every paper_data CSV into ONE tidy, trustworthy table.

    tools/ks_collect.py                 > paper_data/tidy.csv
    tools/ks_collect.py --json          > figures.json
    tools/ks_collect.py --audit         # what was dropped, and why

WHY THIS EXISTS. The raw sweep CSVs are append-only records of what was run,
including runs that were later invalidated and experiments that reuse mode names
for a different purpose. Aggregating them naively produces numbers that look
completely plausible and are wrong. Three real examples, all found only because
a plotted curve came out non-monotonic:

  1. VARIANT. The `vlen` experiment emits rows with mode=serial_do / dc_serial
     for VARIANT=block at several lane counts. Averaged in with the faithful
     rows they invented a 1.114 dc/serial-do ratio for nvfortran at nz=75.
  2. GRID. Early smoke tests ran the same (device, mode, nz) at a different
     nxp/nyp. Keyed without the grid they overwrote the real thread sweep and
     produced a "4.4x speedup at 40 threads".
  3. A KNOWN-BAD RUN. Before the ACC_NUM_CORES fix, `-stdpar=multicore` ignored
     OMP_NUM_THREADS, so every point of that thread sweep silently ran on all
     cores. The CSV looks perfect: a flat 1.00x scaling curve.

So the filters below are not tidying-up, they are the difference between a
result and an artifact. Each is stated with its reason; QUARANTINE names whole
files that must never be aggregated, with the commit that fixed the cause.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "paper_data")
OUT_TIDY = os.path.join(DATA, "tidy.csv")

# Whole files that are known-invalid. Keep them on disk -- they are evidence of
# the bug -- but never let them into an aggregate.
QUARANTINE = {
    "ks_sweep_Big-Chungus_20260803_160002.csv":
        "thread sweep predates the ACC_NUM_CORES fix (8b59b1a): nvfortran's "
        "-stdpar=multicore ignores OMP_NUM_THREADS, so every point ran on all "
        "40 cores while the CSV records 1/2/4/... The nz rows in this file are "
        "fine; only the threads rows are wrong, but the file is dropped whole "
        "because nothing in it distinguishes them.",
}

# Rows to drop wherever they appear.
def row_reject(r):
    if r.get("status") != "OK":
        return "status != OK"
    # The vlen experiment reuses serial_do/dc_serial for the BLOCKED kernel.
    # Those are a different measurement and must not mix with the faithful ones.
    # Reject ONLY `block`: older CSVs spell the CUDA rows cuda_faithful/cuda_opt
    # in this column (before impl and variant were split), and those are real.
    if r.get("variant") == "block":
        return "variant=block (the blocked arrangement, not the faithful one)"
    return None


def load(audit=False):
    rows, dropped = [], []
    # EVERY csv, not just ks_*. A run dropped in as `frontier_cpu_prod.csv`
    # matched no glob and vanished without a word -- the same silent-omission
    # failure the filters below exist to prevent. Now the schema decides: a file
    # carrying the sentinel columns is data, anything else is reported as skipped
    # rather than ignored.
    SENTINEL = {"device", "mode", "nz", "ns_per_col", "status"}
    for p in sorted(glob.glob(os.path.join(DATA, "*.csv"))):
        base = os.path.basename(p)
        if os.path.abspath(p) == os.path.abspath(OUT_TIDY):
            continue                      # our own output
        if base in QUARANTINE:
            dropped.append((base, "*", QUARANTINE[base]))
            continue
        with open(p) as fh:
            hdr = (fh.readline() or "").rstrip("\n").split(",")
        if not SENTINEL.issubset(set(hdr)):
            dropped.append((base, "*", "not a measurement CSV: header lacks "
                            + ", ".join(sorted(SENTINEL - set(hdr)))))
            continue
        with open(p) as fh:
            for r in csv.DictReader(fh):
                why = row_reject(r)
                if why:
                    dropped.append((base, r.get("nz", "?"), why))
                    continue
                r["_file"] = base
                rows.append(r)
    if audit:
        seen = {}
        for f, nz, why in dropped:
            seen.setdefault((f, why), 0)
            seen[(f, why)] += 1
        print("DROPPED\n", file=sys.stderr)
        for (f, why), n in sorted(seen.items()):
            print(f"  {f}  x{n}\n    {why}\n", file=sys.stderr)
    return rows


def toolchain(r):
    """compiler + VERSION. `compiler` alone is not a toolchain: two gfortran
    modules both report `gfortran` and do not even offer the same lanes.

    The compiler is basename'd: `FC` is sometimes an absolute path (an env
    script exporting the full nvhpc install prefix), which both makes an
    unreadable legend entry and splits one toolchain into two."""
    c = os.path.basename(r["compiler"])
    m = re.search(r"(\d+\.\d+(?:\.\d+)?)", r.get("compiler_ver") or "")
    return f"{c} {m.group(1)}" if m else c


# `device` is whatever the operator typed at run time -- a nvidia-smi product
# name from the probing harness, a hand-set label from ks_min.conf, or a full
# /proc/cpuinfo model string. ONE V100 in this repo answers to three of them
# ("Tesla V100-DGXS-32GB", "Tesla V100", "V100"), which on a chart is three
# GPUs. Canonicalise on the way in; these are also the names a reader sees.
SHORT_DEV = {"Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz": "Broadwell",
             "Tesla V100-DGXS-32GB": "V100", "Tesla V100": "V100"}


def short_dev(r):
    return SHORT_DEV.get(r["device"], r["device"])


def series(r):
    """DEVICE + toolchain -- the label for a CPU line.

    Toolchain alone is not identity: `gfortran 15.2.0` is the Mac and
    `gfortran 16.1.0` is Broadwell, and a legend carrying only the version
    asks the reader to know that. The earlier form of this bug put three
    entries reading `Broadwell` on one chart."""
    return f"{short_dev(r)} / {toolchain(r)}"


# The identity of a measurement. Anything omitted here is something two
# different runs are allowed to disagree about while silently overwriting each
# other -- which is exactly how the grid bug happened.
KEY = ("device", "mode", "impl", "launcher", "variant", "lanes", "stack_policy",
       "copy_policy", "phys", "nzstack", "bcopy", "threads", "nz", "nxp", "nyp",
       "land_pct")
# `phys` is the physics profile (faithful | prod) and is load-bearing in the
# identity: `prod` is a 4.6-7.3x more expensive PROBLEM, not a faster or slower
# way of doing the same one. Without it here, a prod row and a faithful row at
# the same (device, nz) collapse to one key and tidy() keeps whichever was
# quicker -- i.e. it would silently discard every prod measurement. Rows written
# before the column existed normalise to "faithful", which is what they are.
# `launcher` (solo | cmp) is part of the identity: the same do-concurrent kernel
# is measured BOTH on its own and inside the CUDA head-to-head binary, and
# without this the two collapse to one key and tidy() silently keeps whichever
# ran faster. They agree to 1.000 on GH200 -- which is a RESULT (the cmp harness
# does not perturb the DC side), and one worth being able to check rather than
# having it averaged away.


def tidy(rows):
    out = {}
    for r in rows:
        r["device"] = short_dev(r)     # canonical BEFORE it enters the identity
        r["phys"] = r.get("phys") or "faithful"
        k = tuple(r.get(c, "") for c in KEY) + (toolchain(r),)
        prev = out.get(k)
        # Same configuration measured twice: keep the faster (min-of-runs is the
        # statistic the harness already reports per invocation).
        if prev is None or float(r["ms_min"]) < float(prev["ms_min"]):
            out[k] = r
    return [out[k] for k in sorted(out)]


COLS = list(KEY) + ["toolchain", "gpu_arch", "ms_min", "ms_med", "ns_per_col",
                    "it_outer", "it_inner", "kd_sum", "regs", "local_b",
                    "reps", "nrun", "host", "git_hash", "_file"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="emit figure-ready series")
    ap.add_argument("--audit", action="store_true", help="report what was dropped")
    a = ap.parse_args()

    rows = tidy(load(audit=a.audit))
    if not a.json:
        w = csv.writer(sys.stdout)
        w.writerow(COLS)
        for r in rows:
            r = dict(r, toolchain=toolchain(r))
            w.writerow([r.get(c, "") for c in COLS])
        print(f"# {len(rows)} rows after filtering", file=sys.stderr)
        return 0

    NZ = [10, 25, 30, 50, 75, 100]
    def pick(dev, mode, sp, nz, nxp="473"):
        for r in rows:
            if (r["device"] == dev and r["mode"] == mode and r["stack_policy"] == sp
                    and int(r["nz"]) == nz and r["nxp"] == nxp):
                return float(r["ns_per_col"])
    def cpu(dev, nz, nxp="64", sp="fit"):
        """Best threaded CPU ns/col.

        ⚠ GRID MISMATCH, MEASURED. The default nxp=64 (4,900 columns) is the grid
        the CPU lanes were swept at, while pick() reads the GPU at production
        473x297 (145,137 columns). On Grace, running the SAME threaded lane at
        both grids costs 10-18% more per column at production (1.10x at nz=10,
        1.18x at nz=100, 72 threads) -- the small grid's 3D fields are L3-resident
        and the production one's are not. So any cpu()/gpu ratio built from the
        default is biased IN THE CPU'S FAVOUR by roughly that much. Pass
        nxp="473" wherever production-grid CPU rows exist (currently Grace only,
        stack_policy="prod")."""
        v = [float(r["ns_per_col"]) for r in rows
             if r["device"] == dev and r["mode"] == "dc_multicore"
             and int(r["nz"]) == nz and r["stack_policy"] == sp and r["nxp"] == nxp]
        return min(v) if v else None

    GPU = {"NVIDIA V100": ("V100", "dc_gpu_acc"),
           "NVIDIA GH200": ("GH200", "dc_gpu"),
           "AMD MI250X": ("MI250X", "dc_gpu"),
           "Intel Max": ("Intel GPU", "dc_gpu")}
    o = {"nz": NZ}
    o["gpu_ns"] = {k: [pick(d, m, "fit", nz) for nz in NZ] for k, (d, m) in GPU.items()}
    o["frame"] = {k: [(lambda a, b: a / b if a and b else None)(pick(d, m, "prod", nz), pick(d, m, "fit", nz))
                      for nz in NZ] for k, (d, m) in GPU.items()}
    # Each GPU against the whole CPU in its own node. cpu_nxp is per-pair
    # because only Grace has a PRODUCTION-GRID threaded sweep; the rest are at
    # 64^2, where the 3D fields are cache-resident and the CPU is flattered by
    # 10-18% (measured on Grace at 72 threads). Pairs carrying that caveat are
    # marked, so the figure can say which is which rather than averaging it away.
    PAIRS = [("NVIDIA GH200 / Grace, 72 threads",        "GH200",     "dc_gpu",     "Grace",     "473", "prod"),
             ("AMD MI250X, 1 GCD / EPYC 7A53, 56 thr",   "MI250X",    "dc_gpu",     "EPYC 7A53", "473", "prod"),
             ("NVIDIA V100 / Broadwell, 40 threads",     "V100",      "dc_gpu_acc", "Broadwell", "473", "prod"),
             ("Intel Max, 1 tile / Xeon Max, 104 thr",   "Intel GPU", "dc_gpu",     "Xeon Max",  "473", "prod")]
    o["gpu_vs_cpu"] = {}
    o["gpu_vs_cpu_samegrid"] = {}
    for lab, gdev, gmode, cdev, cnxp, csp in PAIRS:
        o["gpu_vs_cpu"][lab] = [(lambda c, g: c / g if c and g else None)(
            cpu(cdev, nz, nxp=cnxp, sp=csp), pick(gdev, gmode, "fit", nz)) for nz in NZ]
        o["gpu_vs_cpu_samegrid"][lab] = (cnxp == "473")
    # The one pair measured at the SAME grid on both sides -- no cache-residency
    # correction owed, and the only whole-GPU-vs-whole-socket number in the set.
    # Kept separate from gpu_vs_cpu above because the unit differs: those two are
    # ONE GCD / ONE tile against an entire CPU.
    o["node_gh200"] = [cpu("Grace", nz, nxp="473", sp="prod") / pick("GH200", "dc_gpu", "prod", nz)
                       for nz in NZ]
    o["node_gh200_cuda"] = [cpu("Grace", nz, nxp="473", sp="prod")
                            / pick("GH200", "cuda_faithful", "prod", nz) for nz in NZ]

    # PERFORMANCE PORTABILITY: one `do concurrent` source, every target that ran
    # it, all at the SAME production grid. The unit is in each label and is NOT
    # uniform -- an MI250X is measured per GCD and an Intel Max per tile (that is
    # the unit a rank gets), against whole V100/GH200 GPUs and a whole Grace
    # socket. Comparing the bars without reading the labels is the mistake this
    # naming exists to prevent.
    def at(dev, mode, sp, nz, thr=None):
        v = [float(r["ns_per_col"]) for r in rows
             if r["device"] == dev and r["mode"] == mode and r["nxp"] == "473"
             and r["stack_policy"] == sp and int(r["nz"]) == nz
             and (thr is None or r["threads"] == thr)]
        return min(v) if v else None
    # Grace carries stack_policy=prod because that is the only production-grid
    # CPU sweep -- legitimate here, and checked: on Grace prod/fit is 0.997-1.010
    # at every depth, i.e. the frame axis that costs AMD 3x costs this CPU nothing.
    # Three KINDS on one axis: hand-written CUDA, the same do concurrent source,
    # and a whole CPU socket. Only NVIDIA has a CUDA twin -- MI250X and Intel Max
    # are do-concurrent-only by design, so the panel shows what exists rather
    # than implying a missing measurement.
    PORT = [("NVIDIA GH200, CUDA",          "GH200",     "cuda_faithful", "fit", None, "cuda"),
            ("NVIDIA GH200, do concurrent", "GH200",     "dc_gpu",        "fit", None, "gpu"),
            ("Grace CPU, 72 cores",         "Grace",     "dc_multicore",  "prod", "72", "cpu"),
            ("AMD MI250X, 1 GCD",           "MI250X",    "dc_gpu",        "fit", None, "gpu"),
            ("NVIDIA V100, CUDA",           "V100",      "cuda_faithful", "fit", None, "cuda"),
            ("NVIDIA V100, do concurrent",  "V100",      "dc_gpu_acc",    "fit", None, "gpu"),
            ("Intel Max, 1 tile",           "Intel GPU", "dc_gpu",        "fit", None, "gpu")]
    o["portability"] = {lab: [at(d, m, sp, nz, t) for nz in NZ]
                        for lab, d, m, sp, t, k in PORT}
    o["portability_kind"] = {lab: k for lab, d, m, sp, t, k in PORT}

    # GRID SWEEP: ns/col against the number of columns actually solved, at fixed
    # depth. This is the one axis the study asserts but never measured -- CLAUDE.md
    # says "small grids hide the launch-amortisation gap" and every GPU number
    # here is taken at one grid. It also carries the MOM6 consequence: the model
    # decomposes across ranks, so a rank drives its GPU with ITS OWN TILE, not the
    # whole domain. If the curve is still falling at 10^5 columns then a rank
    # holding a small tile never reaches these numbers.
    grid = {}
    for r in rows:
        if r["stack_policy"] != "fit" or r["nzstack"] == "":
            continue
        if r["mode"].startswith("dc_gpu") or r["mode"] == "cuda_faithful":
            kind = "CUDA" if r["mode"] == "cuda_faithful" else "do concurrent"
            key = f"{short_dev(r)}, {kind}"
            grid.setdefault(key, {}).setdefault(int(r["nz"]), []).append(
                (int(r["ncols"]), float(r["ns_per_col"])))
    # keep the depth with the widest grid coverage, so the panel is one clean
    # curve per lane rather than a mix of depths
    o["grid"] = {}
    for k, byz in grid.items():
        best = max(byz, key=lambda z: len({c for c, _ in byz[z]}))
        pts = sorted({c: v for c, v in sorted(byz[best])}.items())
        if len(pts) >= 3:
            o["grid"][k] = {"nz": best, "pts": pts}
    d = {}
    for r in rows:
        if r["mode"] in ("dc_serial", "serial_do") and r["stack_policy"] == "fit" and r["nxp"] == "64":
            d.setdefault(series(r), {}).setdefault(r["mode"], {})[int(r["nz"])] = float(r["ns_per_col"])
    o["dc_cost"] = {k: [(v["dc_serial"][nz] / v["serial_do"][nz])
                        if nz in v.get("dc_serial", {}) and nz in v.get("serial_do", {}) else None
                        for nz in NZ]
                    for k, v in d.items() if "dc_serial" in v and "serial_do" in v}
    # Device from the ROW, never a default. This used to fall back to the string
    # "Broadwell / " for any device not in a hand-kept name table, so a new
    # machine's thread curve would have been drawn under the wrong chip's name.
    # (No version here: each of these devices contributes one build per compiler,
    # so `Broadwell / ifx` is already unambiguous and shorter to read.)
    t = {}
    for r in rows:
        if (r["mode"] == "dc_multicore" and int(r["nz"]) == 30
                and r["stack_policy"] == "prod" and r["nxp"] == "64"):
            t.setdefault(f"{short_dev(r)} / {r['compiler']}", {})[int(r["threads"])] = float(r["ns_per_col"])
    o["threads"] = {k: [[x, v[min(v)] / v[x]] for x in sorted(v)]
                    for k, v in t.items() if len(v) > 2}
    json.dump(o, sys.stdout, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
