#!/usr/bin/env python3
"""ks_flags.py -- the exact compile line behind every measurement, for the paper.

    tools/ks_flags.py                 > paper_data/compile_flags.md
    tools/ks_flags.py --check         # only report DISAGREEMENTS

Reviewers will ask what was compiled and how, and "the Makefile decides" is not
an answer they can check. Every row of every CSV carries the resolved compile
line in its `flags` column, recorded by the harness at build time rather than
reconstructed afterwards -- so this is extraction, not archaeology.

WHY IT IS PER (device, mode, compiler) AND NOT ONE LINE PER MACHINE. The flags
are not constant across a sweep: NZ_STACK_MAX is a swept axis and appears in
-DMODEL_NZ_STACK_MAX, so a machine legitimately has one compile line per stack
policy. This groups by the parts that identify a LANE and lists the distinct
lines within it, with the varying -D factored out. If a lane shows more than one
line after that, the two runs really did differ and the table says so instead of
quietly showing the first.
"""
import argparse
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "paper_data")

# Swept axes that legitimately vary WITHIN a lane. Factored out of the compile
# line before comparing, then reported separately as the range that was swept.
SWEPT = [r"-DMODEL_NZ_STACK_MAX=\d+", r"-DKS_OPT_NZMAX=\d+", r"-DKS_VLEN=\d+"]


def canon(flags):
    """The compile line with the swept -D values removed and spacing normalised."""
    f = flags
    for pat in SWEPT:
        f = re.sub(pat, "", f)
    return " ".join(f.split())


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(DATA, "*.csv"))):
        if os.path.basename(p) in ("tidy.csv", "compile_flags.md"):
            continue
        with open(p) as fh:
            r = csv.DictReader(fh)
            if not r.fieldnames or "flags" not in r.fieldnames:
                continue          # pre-ks_min schema; no compile line recorded
            for row in r:
                if row.get("status") == "OK" and row.get("flags"):
                    rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any lane has more than one compile line")
    a = ap.parse_args()

    lanes = {}
    for r in load():
        key = (r["device"], r["mode"], r["compiler"], r.get("compiler_ver", "").strip())
        lanes.setdefault(key, {}).setdefault(canon(r["flags"]), set()).add(
            r.get("nzstack", "?"))

    dis = {k: v for k, v in lanes.items() if len(v) > 1}
    if a.check:
        for (dev, mode, cc, ver), v in sorted(dis.items()):
            print(f"{dev} / {mode} / {cc}: {len(v)} distinct compile lines", file=sys.stderr)
            for f in sorted(v):
                print(f"    {f}", file=sys.stderr)
        print(f"{len(dis)} lane(s) with more than one compile line", file=sys.stderr)
        return 1 if dis else 0

    print("# Compile lines\n")
    print("Extracted from the `flags` column of every measurement CSV, which the")
    print("harness records at build time. `-DMODEL_NZ_STACK_MAX` is a swept axis")
    print("and is factored out; the values swept are listed per lane.\n")
    last_dev = None
    for (dev, mode, cc, ver) in sorted(lanes):
        if dev != last_dev:
            print(f"\n## {dev}\n")
            last_dev = dev
        v = lanes[(dev, mode, cc, ver)]
        m = re.search(r"\d+\.\d+(?:\.\d+)?", ver or "")
        print(f"**{mode}** — `{os.path.basename(cc)}`"
              + (f" {m.group(0)}" if m else ""))
        for f, stacks in sorted(v.items()):
            st = ",".join(sorted(stacks, key=lambda x: (len(x), x)))
            print(f"\n```\n{f}\n```")
            print(f"NZ_STACK_MAX swept: {st}\n")
    if dis:
        print(f"\n> {len(dis)} lane(s) show more than one compile line — see above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
