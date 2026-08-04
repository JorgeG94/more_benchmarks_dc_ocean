#!/usr/bin/env python3
"""ks_report.py -- turn tools/ks_sweep.sh CSVs into the tables the blog needs.

    tools/ks_report.py paper_data/ks_sweep_*.csv
    tools/ks_report.py --experiment nz paper_data/*.csv

Reads any number of sweep CSVs (from any number of machines) and prints
markdown. Stdlib only, on purpose: this has to run on a login node with nothing
installed. No plotting here -- the tables ARE the deliverable at this stage, and
a chart would just hide the fingerprint columns that make the numbers checkable.

WHAT IT CHECKS AS WELL AS REPORTS
---------------------------------
Every row carries kd_sum/kd_min/kd_max and the integer iteration counts, so this
script verifies before it summarises:

  * FINGERPRINT: for a given (nz, nxp, nyp, land_pct) every lane must agree on
    kd_sum. A second distinct value means some lane silently changed the
    answers, and its timings are meaningless. Reported loudly, not skipped.
  * WORK: iterations per wet column, so an nz curve can distinguish "more
    layers" from "more iterations". Cost per iteration is the honest
    normalisation when the solver's trip count moves with the problem.
  * PAIRING: DC-vs-CUDA ratios are only formed WITHIN a copy_policy, because
    the Fortran copies its whole-array assignments over the declared extent and
    the CUDA kernels bound theirs -- comparing across that gap times the copy
    rather than the codegen (see kappa_shear/README.md).
"""
import csv
import sys
import argparse
from collections import defaultdict, OrderedDict


def load(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r.get("status") != "OK":
                    continue
                for k in ("ms_min", "ms_med", "ns_per_col", "kd_sum"):
                    try:
                        r[k] = float(r[k])
                    except (TypeError, ValueError):
                        r[k] = None
                for k in ("nz", "nzstack", "ncols", "nwet", "threads", "lanes"):
                    try:
                        r[k] = int(r[k])
                    except (TypeError, ValueError):
                        r[k] = None
                for k in ("it_outer", "it_inner"):
                    try:
                        r[k] = int(r[k])
                    except (TypeError, ValueError):
                        r[k] = None
                # SCHEMA TOLERANCE. CSVs accumulate across machines and across
                # harness versions, and the whole point of aggregating them is to
                # compare a GH200 run against a V100 run that may predate a
                # column. Backfill rather than crash, and derive `impl` from the
                # old per-implementation `mode` where it is missing.
                if not r.get("impl"):
                    m = r.get("mode", "") or ""
                    r["impl"] = m if m.startswith(("cuda_", "hip_")) else "dc"
                for k, dflt in (("launcher", "solo"), ("gpu_arch", "?"),
                                ("copy_policy", "?"), ("stack_policy", "?"),
                                ("variant", "?"), ("proc_bind", "?"),
                                ("experiment", "?"), ("regs", ""),
                                ("local_b", ""), ("device", "?"),
                                ("host", "?"), ("land_pct", "0")):
                    if r.get(k) in (None, ""):
                        r[k] = dflt
                r["_src"] = p
                rows.append(r)
    return rows


def fmt(v, nd=1):
    if v is None:
        return "-"
    return f"{v:,.{nd}f}"


def table(hdr, body):
    out = ["| " + " | ".join(hdr) + " |",
           "|" + "|".join("---" for _ in hdr) + "|"]
    for r in body:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# Above this relative spread in kd_sum, a disagreement is a bug rather than
# arithmetic. This kernel is an ITERATIVE solver with convergence tests, so a
# 1-ulp difference in an input (different compiler, different FMA contraction,
# a different libm) can flip a test, change one column's iteration count and
# show up in the sum. kappa_shear/README.md measures that amplification at
# ~3e-10 for DATA=none vs DATA=acc. Exact equality is the right bar WITHIN one
# toolchain and far too strict ACROSS toolchains.
FINGERPRINT_TOL = 1e-9


def check_fingerprint(rows):
    """Same problem must give the same answer, to within the solver's own
    sensitivity. Reports the spread; only calls it a defect above tolerance."""
    seen = defaultdict(dict)
    for r in rows:
        if r["kd_sum"] is None:
            continue
        key = (r["nz"], r["nxp"], r["nyp"], r["land_pct"])
        seen[key][f"{r['mode']}/{r['impl']}/{r['compiler']}"] = r["kd_sum"]

    exact, within, bad = [], [], []
    for key, vals in seen.items():
        v = list(vals.values())
        lo, hi = min(v), max(v)
        scale = max(abs(lo), abs(hi), 1e-300)
        spread = (hi - lo) / scale
        if spread == 0.0:
            exact.append(key)
        elif spread <= FINGERPRINT_TOL:
            within.append((key, spread, vals))
        else:
            bad.append((key, spread, vals))

    print("## Correctness fingerprint\n")
    print(f"{len(exact)} problem size(s) bit-identical across every lane; "
          f"{len(within)} agreeing within the solver's convergence sensitivity "
          f"(<= {FINGERPRINT_TOL:g} rel); {len(bad)} beyond it.\n")

    if within:
        print("Agreeing, but not bit-identical — expected when different "
              "compilers or FP settings are in the same table:\n")
        body = []
        for (nz, nxp, nyp, land), spread, vals in sorted(within):
            body.append([nz, f"{nxp}x{nyp}", f"{spread:.2e}",
                         ", ".join(sorted(vals))])
        print(table(["nz", "grid", "kd_sum rel spread", "lanes"], body))
        print()

    if bad:
        print("**BEYOND TOLERANCE — a lane changed the answers. Do not quote "
              "its timings until this is explained.**\n")
        body = []
        for (nz, nxp, nyp, land), spread, vals in sorted(bad):
            for who, v in sorted(vals.items()):
                body.append([nz, f"{nxp}x{nyp}", f"{spread:.2e}",
                             f"{v:.10e}", who])
        print(table(["nz", "grid", "rel spread", "kd_sum", "lane"], body))
        print()


def occupancy_inputs(rows):
    """Registers and per-thread frame, DC vs CUDA, per device+arch.

    This is the cross-GPU diagnostic. It does not vary with nz, so one row per
    (device, arch, impl) is the whole story. nvfortran has historically pinned
    this kernel at the 254-register cap while nvcc uses ~80; how much that costs
    depends on how well the architecture hides latency without occupancy, which
    is precisely why the same source shows a different DC-vs-CUDA ratio on
    different cards.
    """
    seen = {}
    for r in rows:
        try:
            regs = int(r.get("regs") or 0)
            loc = int(r.get("local_b") or 0)
        except ValueError:
            continue
        if regs <= 0:
            continue
        key = (r["device"], r.get("gpu_arch", "?"), r["impl"], r["nzstack"])
        seen.setdefault(key, (regs, loc))
    if not seen:
        return
    print("## Occupancy inputs — registers and per-thread frame\n")
    print("Read across implementations on the SAME device. A large register gap "
          "with a similar frame means the two toolchains are asking for very "
          "different amounts of the register file for identical arithmetic — and "
          "how much that costs is architecture-dependent.\n")
    body = []
    for (dev, arch, impl, nzs), (regs, loc) in sorted(seen.items()):
        body.append([dev, arch, impl, nzs, regs, f"{loc:,}"])
    print(table(["device", "arch", "impl", "NZSTACK", "registers",
                 "frame B/thread"], body))
    print()


def toolchain(r):
    """compiler + VERSION. `compiler` alone is not a toolchain: two gfortran
    modules (13.3 and 16.1) both report `gfortran`, and they do not even offer
    the same lanes -- 13.3 cannot compile `do concurrent ... local()` so it has
    only serial_do, while 16.1 has both. Grouping on the bare name silently
    pairs one version's serial_do with another's dc_serial."""
    v = (r.get("compiler_ver") or "").strip()
    import re as _re
    m = _re.search(r"(\d+\.\d+(?:\.\d+)?)", v)
    return f"{r['compiler']} {m.group(1)}" if m else r["compiler"]


def per_device(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["host"], r["device"])].append(r)
    return g


def exp_nz(rows):
    print("## Depth sweep — cost per column vs nz\n")
    print("`ns/col` is ms_min normalised by the column count, so grids of "
          "different width are comparable. `it/col` is Picard iterations per "
          "wet column — the work metric, which is what lets an nz curve "
          "distinguish more layers from more iterations.\n")
    for (host, dev), rs0 in sorted(per_device(rows).items()):
        # Split by COMPILER as well as device and stack policy. serial_do runs
        # under every available compiler, so grouping on mode alone merges e.g.
        # gfortran's serial_do with nvfortran's dc_serial into one row --
        # silently turning a within-compiler comparison into a cross-compiler
        # one. (Every cmp-lane row carries the Fortran compiler that built the
        # binary, so its dc and cuda rows stay together, as they must.)
        for comp in sorted({toolchain(r) for r in rs0}):
            rs = [r for r in rs0 if toolchain(r) == comp]
            for pol in sorted({r["stack_policy"] for r in rs}):
                sub = [r for r in rs if r["stack_policy"] == pol]
                if not sub:
                    continue
                # Key columns on MODE: on the CPU every lane reports impl=dc,
                # so keying on impl would collapse the lanes into one column.
                modes = sorted({r["mode"] for r in sub})
                nzs = sorted({r["nz"] for r in sub})
                polname = ("NZSTACK=128, production" if pol == "prod"
                           else "NZSTACK=nz+1, frame-fitted")
                print(f"\n### {dev} — {comp} — stack policy `{pol}` ({polname})\n")
                hdr = ["nz", "grid", "it/col"] + [f"{m} ns/col" for m in modes]
                ref = "dc_gpu_acc" if "dc_gpu_acc" in modes else None
                cudas = [m for m in modes if m.startswith("cuda_")]
                if ref:
                    hdr += [f"{ref}/{m}" for m in cudas]
                if "serial_do" in modes and "dc_serial" in modes:
                    hdr += ["dc_serial/serial_do"]
                body = []
                for nz in nzs:
                    at = {r["mode"]: r for r in sub if r["nz"] == nz}
                    if not at:
                        continue
                    any_r = next(iter(at.values()))
                    itc = None
                    for r in at.values():
                        if r["it_inner"] and r["nwet"]:
                            itc = r["it_inner"] / r["nwet"]
                            break
                    row = [nz, f"{any_r['nxp']}x{any_r['nyp']}", fmt(itc, 2)]
                    for m in modes:
                        row.append(fmt(at[m]["ns_per_col"], 1) if m in at else "-")
                    if ref and ref in at:
                        for m in cudas:
                            if m not in at:
                                row.append("-")
                            elif at[m]["copy_policy"] != at[ref]["copy_policy"]:
                                # Refuse to divide across copy policies: one side
                                # copies O(NZSTACK) and the other O(nz), so the
                                # quotient measures the copy, not the codegen.
                                row.append(f"unpaired ({at[m]['copy_policy']} vs "
                                           f"{at[ref]['copy_policy']})")
                            else:
                                row.append(fmt(at[ref]["ms_min"] / at[m]["ms_min"], 3) + "x")
                    if "serial_do" in modes and "dc_serial" in modes:
                        if "serial_do" in at and "dc_serial" in at:
                            row.append(fmt(at["dc_serial"]["ms_min"]
                                           / at["serial_do"]["ms_min"], 3) + "x")
                        else:
                            row.append("-")
                    body.append(row)
                print(table(hdr, body))
                print()


def exp_copy(rows):
    print("## Copy policy — the 1:1 pairing\n")
    print("`prod` = both sides copy the DECLARED extent, O(NZSTACK). "
          "`opt` = both bound it to 1..nz+1, O(nz). Ratios are only meaningful "
          "WITHIN a policy; across policies you are timing the copy.\n")
    for (host, dev), rs in sorted(per_device(rows).items()):
        nzs = sorted({r["nz"] for r in rs})
        body = []
        for nz in nzs:
            for pol in ("prod", "opt"):
                for spol in sorted({r["stack_policy"] for r in rs}):
                    at = {r["impl"]: r for r in rs
                          if r["nz"] == nz and r["copy_policy"] == pol
                          and r["stack_policy"] == spol}
                    if "dc" not in at:
                        continue
                    row = [nz, spol, pol, fmt(at["dc"]["ms_min"], 3)]
                    for i in ("cuda_faithful", "cuda_opt"):
                        row.append(fmt(at[i]["ms_min"], 3) if i in at else "-")
                    for i in ("cuda_faithful", "cuda_opt"):
                        row.append(fmt(at["dc"]["ms_min"] / at[i]["ms_min"], 3) + "x"
                                   if i in at else "-")
                    body.append(row)
        if body:
            print(f"\n### {dev}\n")
            print(table(["nz", "stack", "copy", "dc ms", "cuda_f ms", "cuda_opt ms",
                         "dc/cuda_f", "dc/cuda_opt"], body))
            print()


def exp_cols(rows):
    print("## Width sweep — cost per column vs column count\n")
    print("Flat `ns/col` means the normalisation used elsewhere is safe. Where "
          "it rises at the small end, the device is starved — and THAT is the "
          "regime a MOM6 rank actually lives in (an OM4_025 rank owns of order "
          "1e3 columns, not 1e5).\n")
    for (host, dev), rs in sorted(per_device(rows).items()):
        impls = sorted({r["mode"] for r in rs})
        cols = sorted({r["ncols"] for r in rs if r["ncols"]})
        body = []
        for c in cols:
            at = {r["mode"]: r for r in rs if r["ncols"] == c}
            if not at:
                continue
            any_r = next(iter(at.values()))
            row = [f"{any_r['nxp']}x{any_r['nyp']}", f"{c:,}"]
            for i in impls:
                row.append(fmt(at[i]["ns_per_col"], 1) if i in at else "-")
            body.append(row)
        if body:
            print(f"\n### {dev}\n")
            print(table(["grid", "columns"] + [f"{i} ns/col" for i in impls], body))
            print()


def exp_threads(rows):
    print("## Thread scaling (dc_multicore)\n")
    for (host, dev), rs in sorted(per_device(rows).items()):
        for bind in sorted({r["proc_bind"] for r in rs}):
            sub = sorted([r for r in rs if r["proc_bind"] == bind],
                         key=lambda r: r["threads"] or 0)
            if not sub:
                continue
            base = next((r["ms_min"] for r in sub if r["threads"] == 1), None)
            body = [[r["threads"], fmt(r["ms_min"], 3), fmt(r["ns_per_col"], 1),
                     fmt(base / r["ms_min"], 2) + "x" if base else "-",
                     fmt(100.0 * (base / r["ms_min"]) / r["threads"], 1) + "%"
                     if base and r["threads"] else "-"]
                    for r in sub]
            print(f"\n### {dev} — OMP_PROC_BIND={bind}\n")
            print(table(["threads", "ms", "ns/col", "speedup", "efficiency"], body))
            print()


def exp_vlen(rows):
    print("## Arrangement sweep — VLEN (column blocking)\n")
    print("VLEN=1 must be bit-identical to the faithful per-column kernel; it "
          "is the gate on the transformation. Above that, VLEN trades vector "
          "width against the extra work of running masked full-range loops "
          "instead of per-column early exits.\n")
    for (host, dev), rs in sorted(per_device(rows).items()):
        # Group by (mode, compiler): serial_do runs under every available
        # compiler, and merging them produces duplicate rows that look like
        # run-to-run noise but are actually different toolchains.
        for mode, comp in sorted({(r["mode"], toolchain(r)) for r in rs}):
            sub = [r for r in rs if r["mode"] == mode and toolchain(r) == comp]
            nzs = sorted({r["nz"] for r in sub})
            body = []
            for nz in nzs:
                base = next((r["ms_min"] for r in sub
                             if r["nz"] == nz and r["variant"] == "faithful"), None)
                for r in sorted([x for x in sub if x["nz"] == nz],
                                key=lambda x: (x["variant"] != "faithful",
                                               x["lanes"] or 0)):
                    lanes = "-" if r["variant"] == "faithful" else r["lanes"]
                    body.append([nz, r["variant"], lanes, fmt(r["ms_min"], 3),
                                 fmt(r["ns_per_col"], 1),
                                 fmt(base / r["ms_min"], 3) + "x" if base else "-"])
            if body:
                print(f"\n### {dev} — {mode} ({comp})\n")
                print(table(["nz", "variant", "lanes", "ms", "ns/col",
                             "vs faithful"], body))
                print()


HANDLERS = OrderedDict([("nz", exp_nz), ("copy", exp_copy), ("cols", exp_cols),
                        ("threads", exp_threads), ("vlen", exp_vlen),
                        ("cuopt", exp_copy)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--experiment", "-e", action="append",
                    help="limit to these experiments (default: all present)")
    a = ap.parse_args()

    rows = load(a.csv)
    if not rows:
        print("no OK rows found in:", ", ".join(a.csv), file=sys.stderr)
        return 1

    hosts = sorted({(r["host"], r["device"]) for r in rows})
    print("# kappa_shear sweep report\n")
    print(f"{len(rows)} OK rows from {len(a.csv)} file(s), "
          f"{len(hosts)} host/device combination(s):\n")
    for h, d in hosts:
        n = sum(1 for r in rows if (r["host"], r["device"]) == (h, d))
        print(f"- `{h}` / {d} — {n} rows")
    print()

    check_fingerprint(rows)
    occupancy_inputs(rows)

    want = a.experiment or [e for e in HANDLERS if any(r["experiment"] == e for r in rows)]
    for e in want:
        sub = [r for r in rows if r["experiment"] == e]
        if not sub:
            continue
        HANDLERS[e](sub)
    return 0


if __name__ == "__main__":
    sys.exit(main())
