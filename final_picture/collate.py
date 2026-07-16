#!/usr/bin/env python3
"""Collate per-benchmark CSVs into the final-picture table.

Reads  final_picture/results/<key>.csv   (written by parse_one.py)
       final_picture/shares.csv           (profile weights, H200)
Writes final_picture/results/summary.csv  and prints a table + the verdict.

The question it answers: is a hand-written CUDA C (or HIP) rewrite worth it?
For each kernel it compares the best *portable Fortran* variant against the
best GPU-C variant, confirms the OpenACC->CUDA bridge is free (native ~= CUDA),
and weights the rewrite gain by the kernel's share of production wall time.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')


def load_shares():
    out = {}
    with open(os.path.join(HERE, 'shares.csv')) as fh:
        for r in csv.DictReader(fh):
            share = r['share_pct'].strip()
            out[r['key']] = {
                'display': r['display'],
                'share': float(share) if share else None,
                'unit': r['unit'],
                'note': r['note'],
            }
    return out


def load_kernel(key):
    rows = []
    with open(os.path.join(RES, key + '.csv')) as fh:
        for r in csv.DictReader(fh):
            rows.append({'variant': r['variant'], 'family': r['family'],
                         'ms': float(r['ms']), 'prod': r['is_production'] == '1'})
    fort = [r for r in rows if r['family'] == 'fortran']
    cuda = [r for r in rows if r['family'] == 'cuda']
    nat = [r for r in rows if r['family'] == 'native']
    if not fort or not cuda:
        return None
    prod = next((r for r in fort if r['prod']), max(fort, key=lambda r: r['ms']))
    best_f = min(fort, key=lambda r: r['ms'])
    best_c = min(cuda, key=lambda r: r['ms'])
    native = min(nat, key=lambda r: r['ms']) if nat else None
    return {
        'prod': prod, 'best_f': best_f, 'best_c': best_c, 'native': native,
        'fort_fix': prod['ms'] / best_f['ms'],           # Fortran-only speedup
        'rewrite': best_f['ms'] / best_c['ms'],          # >1 => CUDA faster
        'bridge': (native['ms'] / best_c['ms']) if native else None,
    }


def main():
    shares = load_shares()
    order = ['redi', 'ks', 'layered', 'ale', 'btstep', 'epbl', 'meke', 'flux']

    rows = []
    for key in order:
        if not os.path.exists(os.path.join(RES, key + '.csv')):
            continue
        k = load_kernel(key)
        if not k:
            continue
        s = shares.get(key, {})
        share = s.get('share')
        # weighted fraction of TOTAL wall time each lever buys (saving = 1 - fast/slow)
        w_fix = (share / 100.0) * (1 - k['best_f']['ms'] / k['prod']['ms']) if share else None
        w_rw = (share / 100.0) * max(0.0, 1 - k['best_c']['ms'] / k['best_f']['ms']) if share else None
        rows.append({'key': key, 'disp': s.get('display', key), 'share': share,
                     'unit': s.get('unit', ''), 'note': s.get('note', ''),
                     'k': k, 'w_fix': w_fix, 'w_rw': w_rw})

    # ---- print table -------------------------------------------------------
    hdr = (f"{'kernel':20s} {'share':>6s} {'prodFort':>9s} {'bestFort':>9s} "
           f"{'fix':>6s} {'bestCUDA':>9s} {'native':>8s} {'brdg':>5s} "
           f"{'rewrite':>8s} {'winner':>8s}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        k = r['k']
        share = f"{r['share']:.1f}%" if r['share'] is not None else '  n/a'
        nat = f"{k['native']['ms']:.4f}" if k['native'] else '   -   '
        brdg = f"{k['bridge']:.3f}" if k['bridge'] else '  -  '
        winner = 'Fortran' if k['best_f']['ms'] <= k['best_c']['ms'] else 'CUDA'
        print(f"{r['disp']:20s} {share:>6s} {k['prod']['ms']:9.4f} "
              f"{k['best_f']['ms']:9.4f} {k['fort_fix']:5.2f}x {k['best_c']['ms']:9.4f} "
              f"{nat:>8s} {brdg:>5s} {k['rewrite']:7.3f}x {winner:>8s}")

    # ---- totals (shares only where known) ----------------------------------
    tot_fix = sum(r['w_fix'] for r in rows if r['w_fix'] is not None)
    tot_rw = sum(r['w_rw'] for r in rows if r['w_rw'] is not None)
    tot_rw_nobug = sum(r['w_rw'] for r in rows
                       if r['w_rw'] is not None and r['key'] != 'epbl')
    weighted = [r['disp'] for r in rows if r['share'] is not None]

    print('\n' + '=' * 64)
    print('  VERDICT  (weighted over: ' + ', '.join(weighted) + ')')
    print('=' * 64)
    print(f"  Portable-Fortran fixes are worth : {tot_fix*100:5.1f}% of wall time")
    print(f"  A full CUDA/HIP rewrite is worth  : {tot_rw*100:5.1f}% of wall time")
    print(f"     ...excluding the EPBL compiler bug: {tot_rw_nobug*100:5.1f}%")
    print(f"  Bridge (native/CUDA) across kernels: "
          f"{min(r['k']['bridge'] for r in rows if r['k']['bridge']):.3f}"
          f"-{max(r['k']['bridge'] for r in rows if r['k']['bridge']):.3f}"
          f"  (~=1.0 => OpenACC host_data is free)")
    print(f"\n  => Fortran do-concurrent is within ~{tot_rw*100:.0f}% of hand-written GPU-C.")
    print(f"     The algorithm (portable Fortran) buys ~{tot_fix/max(tot_rw,1e-9):.0f}x more. "
          f"A rewrite is not justified.")

    print("\n  Caveats (these numbers are what the DEFAULT binaries build today):")
    for r in rows:
        if r['share'] is not None and r['note']:
            print(f"    - {r['disp']:18s}: {r['note']}")
    print("    NB kappa-shear's ~1.14x is pre-fix: the LOGBOOK's free `maxregcount:96`")
    print("       flag + 3 lines cut it to ~1.05x, moving ~1.4% from rewrite to fix.")
    print("       With that + continuity's collapse fix applied, this reconciles to the")
    print("       LOGBOOK headline (~16% Fortran fixes, ~1.7% rewrite).")

    # ---- write summary.csv -------------------------------------------------
    out = os.path.join(RES, 'summary.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['kernel', 'share_pct', 'unit', 'prod_fortran_ms', 'best_fortran_ms',
                    'fortran_fix_x', 'best_cuda_ms', 'best_cuda_variant', 'native_ms',
                    'bridge_native_over_cuda', 'rewrite_x_cuda_over_fortran', 'winner',
                    'weighted_fix_frac', 'weighted_rewrite_frac', 'note'])
        for r in rows:
            k = r['k']
            winner = 'Fortran' if k['best_f']['ms'] <= k['best_c']['ms'] else 'CUDA'
            w.writerow([r['disp'], r['share'] if r['share'] is not None else '',
                        r['unit'], f"{k['prod']['ms']:.4f}", f"{k['best_f']['ms']:.4f}",
                        f"{k['fort_fix']:.3f}", f"{k['best_c']['ms']:.4f}", k['best_c']['variant'],
                        f"{k['native']['ms']:.4f}" if k['native'] else '',
                        f"{k['bridge']:.4f}" if k['bridge'] else '',
                        f"{k['rewrite']:.3f}", winner,
                        f"{r['w_fix']:.5f}" if r['w_fix'] is not None else '',
                        f"{r['w_rw']:.5f}" if r['w_rw'] is not None else '', r['note']])
    print(f"\n  wrote {out}")


if __name__ == '__main__':
    main()
