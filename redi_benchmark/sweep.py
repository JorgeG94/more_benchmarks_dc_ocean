#!/usr/bin/env python3
"""Robust production-size sweep for the Redi benchmark.

Reports MIN-of-N per variant. The min is the right estimator here: this runs on
a shared analysis node and the V100 drifts under sustained load, so the
distribution has a long RIGHT tail (interference only ever makes a run slower).
Measured spread across repeats: ~44.5 ms best, ~86 ms worst outlier, same
binary and args. Trials that fail to parse are DISCARDED, not folded into the
min as zero -- an earlier version of this sweep silently did that.

Usage: python3 sweep.py [trials]
"""
import re
import subprocess
import sys

TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

# (label, nx_phys, ny_phys, nz, nreps, nwarm) -- nx/ny are POINT COUNTS from
# ~/analysis_gebco/*.nml; nx_total = nx + 2*nghost.
CONFIGS = [
    ("10km     (108x137)", 108, 137, 30, 8, 8),
    ("0.1deg   (473x297)", 473, 297, 30, 8, 8),   # <- the Redi/MEKE config
    ("0.25deg  (240x560)", 240, 560, 30, 8, 8),
    ("3km      (359x458)", 359, 458, 30, 6, 6),
    ("0.05deg  (945x594)", 945, 594, 30, 3, 4),
]

PATS = {
    "dc":   re.compile(r"DC  = PRODUCTION.*?([\d.]+)\s*$", re.M),
    "cuda": re.compile(r"CUDA faithful port\s+([\d.]+)\s*$", re.M),
    "pre":  re.compile(r"DC \+ PRECOMP.*?([\d.]+)\s*$", re.M),
}


def run(cfg):
    _, nx, ny, nz, nr, nw = cfg
    best = {}
    cells = None
    for _ in range(TRIALS):
        try:
            out = subprocess.run(
                ["./redi_bench", str(nx), str(ny), str(nz), str(nr), str(nw), "0"],
                capture_output=True, text=True, timeout=3600).stdout
        except subprocess.TimeoutExpired:
            continue
        m = re.search(r"^ cells  : (\d+)", out, re.M)
        if m:
            cells = int(m.group(1))
        vals = {}
        ok = True
        for k, p in PATS.items():
            mm = p.search(out)
            if not mm:
                ok = False
                break
            vals[k] = float(mm.group(1))
        if not ok:
            continue                       # discard, never fold a bad trial in
        for k, v in vals.items():
            if k not in best or v < best[k]:
                best[k] = v
    return cells, best


print(f"min-of-{TRIALS}, V100, nvfortran 26.5, NZ_STACK_MAX=128, production flags")
print()
print(" config                 cells     dc_ms  cuda_ms   pre_ms  dc/cuda  dc/pre")
print(" " + "-" * 71)
for cfg in CONFIGS:
    cells, b = run(cfg)
    if len(b) < 3:
        print(f" {cfg[0]:20s} NO VALID TRIAL")
        continue
    print(f" {cfg[0]:20s} {cells:8d} {b['dc']:9.2f} {b['cuda']:8.2f} {b['pre']:8.2f}"
          f"  {b['dc']/b['cuda']:7.3f} {b['dc']/b['pre']:7.3f}")
