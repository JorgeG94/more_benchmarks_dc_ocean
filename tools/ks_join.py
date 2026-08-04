#!/usr/bin/env python3
"""ks_join.py -- concatenate measurement CSVs into one, safely.

    tools/ks_join.py paper_data/ks_min_gh1_*.csv -o paper_data/gh200_grid.csv
    tools/ks_join.py paper_data/ks_min_gh1_*.csv          # to stdout
    tools/ks_join.py -n paper_data/*.csv                  # report, write nothing

WHY NOT `cat`. The schema GREW over the study -- `launcher`, then `phys` -- so
files written weeks apart have different headers, and concatenating them shifts
every column right of the insertion point. Silently. This unions the headers and
fills missing fields with the empty string, which is exactly what the collector
already normalises (`phys` absent -> "faithful").

WHAT IT DOES NOT DO. It does not filter, dedupe or quarantine. Joining is not
analysis: tools/ks_collect.py owns the row filters, the QUARANTINE list and the
dedupe-by-identity, and it is the thing to plot from. This is for when you want
one file to move between machines or hand to a reviewer.

QUARANTINE IS BY FILENAME, so joining a quarantined file into a bundle makes it
unquarantinable -- nothing in the rows distinguishes it. Those files are refused
by default; --force overrides, which you should not need.
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:                                  # reuse the one list, never a second copy
    from ks_collect import QUARANTINE
except Exception:                     # standalone (copied to another machine)
    QUARANTINE = {}

SENTINEL = {"device", "mode", "nz", "ns_per_col", "status"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out", help="output path (default: stdout)")
    ap.add_argument("-n", "--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="include quarantined files (don't)")
    a = ap.parse_args()

    cols, rows, used, skipped = [], [], [], []
    for p in a.files:
        base = os.path.basename(p)
        if base in QUARANTINE and not a.force:
            skipped.append((base, "QUARANTINED: " + QUARANTINE[base].split(".")[0]))
            continue
        with open(p, newline="") as fh:
            rd = csv.DictReader(fh)
            if not rd.fieldnames or not SENTINEL.issubset(set(rd.fieldnames)):
                miss = SENTINEL - set(rd.fieldnames or [])
                skipped.append((base, "not a measurement CSV, lacks " + ", ".join(sorted(miss))))
                continue
            for c in rd.fieldnames:               # union, first-seen order
                if c not in cols:
                    cols.append(c)
            n = 0
            for r in rd:
                rows.append(r)
                n += 1
            used.append((base, n, rd.fieldnames))

    if not used:
        print("no measurement CSVs among the inputs", file=sys.stderr)
        return 1

    # Report header disagreements explicitly. They are legitimate (the schema
    # grew) but the reader should be told, not have blanks appear unexplained.
    ref = used[0][2]
    for base, n, fn in used:
        if fn != ref:
            extra, miss = set(fn) - set(ref), set(ref) - set(fn)
            note = []
            if extra:
                note.append("+" + ",".join(sorted(extra)))
            if miss:
                note.append("-" + ",".join(sorted(miss)))
            print(f"  header differs in {base}: {' '.join(note)}", file=sys.stderr)

    for base, n, _ in used:
        print(f"  {base}  {n} rows", file=sys.stderr)
    for base, why in skipped:
        print(f"  SKIPPED {base}: {why}", file=sys.stderr)

    seen, dup = set(), 0
    for r in rows:
        k = tuple(r.get(c, "") for c in cols)
        if k in seen:
            dup += 1
        seen.add(k)
    print(f"  {len(rows)} rows, {len(cols)} columns"
          + (f", {dup} exact duplicate rows (kept -- dedupe is ks_collect's job)"
             if dup else ""), file=sys.stderr)

    if a.dry_run:
        return 0
    fh = open(a.out, "w", newline="") if a.out else sys.stdout
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in cols})
    if a.out:
        fh.close()
        print(f"  -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
