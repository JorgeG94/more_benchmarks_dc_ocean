#!/usr/bin/env python3
"""Parse ONE benchmark's raw stdout logs into a tidy per-benchmark CSV.

Usage:
    parse_one.py <key> <bench_log> <native_log> <out_csv>

Emits long-format rows:  key,variant,family,ms,is_production
  family     : fortran | cuda | native
  is_production: 1 for the as-shipped do-concurrent baseline, else 0

The 8 benches print in two shapes -- a colon form
    "  <label> : 34.7753 ms/call"
and ale's table form
    "  DC (as shipped)   11.180   1.000x".
Both are handled. The native drivers report their best time in assorted ways
(min:/mean:/(min of N)/"(native)"); we collect every plausible timing from the
native log and keep the minimum, which is all collate.py needs from it.
"""
import re
import sys
import csv

# ---- timing line shapes ----------------------------------------------------
# COLON  : "  <label> : 34.7753 ms/call"            (most benches)
# TABLE  : "  DC (as shipped)   11.180   1.000x"    (ale: label, ms, ratio)
# TABLE2 : "  CUDA faithful port      45.6552"      (redi: label, ms, no unit)
COLON      = re.compile(r'^(?P<label>.+?):\s*(?P<ms>\d+\.\d+)\s*ms\b')
TABLE      = re.compile(r'^(?P<label>[A-Za-z][^:]*?)\s{2,}(?P<ms>\d+\.\d+)\s+\d+\.\d+x\s*$')
TABLE2     = re.compile(r'^(?P<label>[A-Za-z][^:]*?)\s{2,}(?P<ms>\d+\.\d+)\s*$')
NATIVEBARE = re.compile(r'^(?P<label>CUDA[^:]*\(native\))\s+(?P<ms>\d+\.\d+)\s*$')

# a label must look like a variant, and must NOT be a derived/ratio/diagnostic
# line. Ratio lines are excluded structurally (no 'ms', or a trailing 'x'), so
# BAD only needs the semantic words -- keep slashes OUT of it ("1 thread / col").
KW  = ('do concurrent', 'dc ', 'dc(', 'dc +', 'dc  ', 'cuda', 'flat', 'signature',
       'faithful', 'fused', 'graph', 'precomp', 'nested', 'openacc', 'kappa',
       'layered', 'epbl', 'continuity', 'remap', 'launch_bounds', 'tuned', 'native')
BAD = ('ratio', ' vs ', 'overhead', 'lookup', 'share', 'variant total', 'sanity',
       'differ', 'agreement', 'present-table', 'probe')

PROD = re.compile(r'production|shipped|verbatim|as shipped', re.I)
# a plain baseline: "do concurrent" with no transform word in the label
XFORM = ('fix', 'fused', 'async', 'plain', 'nested', 'precomp', 'signature',
         'collapse', 'derived', 'dt ', 'flat', 'acc ', 'graph', 'launch', 'tuned',
         'dims', 'vector', 'faithful')


def label_ok(label):
    l = ' ' + label.lower() + ' '
    if not any(k in l for k in KW):
        return False
    if any(b in l for b in BAD):
        return False
    return True


def family_of(label):
    return 'cuda' if 'cuda' in label.lower() else 'fortran'


def is_baseline(label):
    l = label.lower()
    if 'do concurrent' not in l and not l.strip().startswith('dc'):
        return False
    return not any(x in l for x in XFORM)


def parse_bench(path):
    rows = []
    with open(path) as fh:
        for raw in fh:
            s = raw.strip()
            m = COLON.match(s) or TABLE.match(s) or TABLE2.match(s)
            if not m:
                continue
            label = m.group('label').strip()
            if not label_ok(label):
                continue
            fam = family_of(label)
            prod = 1 if (fam == 'fortran' and (PROD.search(label) or is_baseline(label))) else 0
            rows.append((label, fam, float(m.group('ms')), prod))
    # collapse a rare double-production tag to the slowest fortran only
    return rows


def parse_native_best(path):
    """Smallest credible timing in the native log (excludes diff/spread lines)."""
    best = None
    with open(path) as fh:
        for raw in fh:
            s = raw.strip()
            low = s.lower()
            # skip the bit-diff table (sci-notation, '|d|', 'differ') but NOT the
            # timing lines, which carry a harmless "(worst ...)" tail after 'ms'.
            if any(b in low for b in ('differ', '|d', 'e-', 'e+', 'rel ')):
                continue
            for m in (COLON.match(s), NATIVEBARE.match(s)):
                if m:
                    ms = float(m.group('ms'))
                    if 1e-4 < ms < 1e4 and (best is None or ms < best):
                        best = ms
    return best


def main():
    key, bench_log, native_log, out_csv = sys.argv[1:5]
    rows = parse_bench(bench_log)

    # if more than one fortran row got flagged production, keep only the slowest
    prod_rows = [r for r in rows if r[3] == 1]
    if len(prod_rows) > 1:
        keep = max(prod_rows, key=lambda r: r[2])
        rows = [(l, f, m, (1 if (l, f, m, p) == keep else 0)) for (l, f, m, p) in rows]

    nat = parse_native_best(native_log)

    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['key', 'variant', 'family', 'ms', 'is_production'])
        for label, fam, ms, prod in rows:
            w.writerow([key, label, fam, f'{ms:.4f}', prod])
        if nat is not None:
            w.writerow([key, 'native (cudaMalloc, best)', 'native', f'{nat:.4f}', 0])

    n_f = sum(1 for r in rows if r[1] == 'fortran')
    n_c = sum(1 for r in rows if r[1] == 'cuda')
    print(f'  {key:18s} -> {out_csv}  ({n_f} fortran, {n_c} cuda, '
          f'native={"%.4f" % nat if nat else "n/a"})')


if __name__ == '__main__':
    main()
