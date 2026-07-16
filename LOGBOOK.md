# LOGBOOK — `do concurrent` vs CUDA C across the ocean model's GPU kernels

_Cold-start handoff, 2026-07-16. Read §0, §1 and §2. Everything else is detail._

> **Companion doc:** `RESUME_GPU_MRE.md` covers the first session (daxpy, the coastal
> HLL flux kernel, and the two NVIDIA bug reports). This file covers the second session:
> seven kernels benchmarked against hand-written CUDA C, and the production profile that
> tells you which of them matter. **Where the two disagree, this file wins** — §1 lists
> what it corrects.

---

## 0. TL;DR — the answer

**`do concurrent` is not costing the ocean model anything meaningful. A full CUDA rewrite of every
kernel measured would buy ~1.7% of wall time. The Fortran-side fixes already measured are
worth ~16%.**

Seven kernels, each against a faithful hand-written CUDA C port, all bit-identical or
FMA-level agreement:

| kernel | share of runtime | best Fortran | best CUDA | CUDA's edge | what the Fortran fix was |
|---|---|---|---|---|---|
| `ocean_redi` | **40.1%** | **33.02 ms** | 45.04 ms* | **Fortran wins** | precompute hoist (8-vs-5 columns) — **1.35×** |
| `ocean_vmix_kshear` | **18.5%** | 36.52 ms | 34.84 ms | 1.048× | `-gpu=maxregcount:96` + 3 lines — **1.08×** |
| `ocean_continuity` | 7.0% | 1.065 ms | 0.964 ms | 1.105× | nothing to fix |
| `ocean_ale_remap` | 5.7% | 7.374 ms | 6.961 ms | 1.059× | sigfix + fusion — **1.455×** |
| `ocean_barotropic_solver` | 4.7% | 2.618 ms | 2.464 ms | 1.062× | sigfix + fusion — **1.415×** |
| `ocean_vmix_epbl` | 1.8% | 6.489 ms | 4.883 ms | **1.329×** | nothing (CUDA-only, via a compiler bug) |
| MEKE (no profiler region) | ? | 0.213 ms | 0.193 ms | 1.103× | async + fusion — **1.67×** |
| coastal HLL flux (session 1) | coastal path | 5.24 ms | 5.52 ms | **Fortran wins** | flat body — **1.4×** |

\* Redi's CUDA port is a faithful transliteration of the **unfixed** algorithm; it never
got the precompute. A CUDA port with the same hoist would presumably tie. **Do not quote
"Fortran beats CUDA by 1.36× on Redi"** as a language result — it is an algorithm result.

**Two conclusions:**

1. **CUDA's edge is a flat 6–10%** across four independent kernels of completely
   different character (launch-bound, spill-bound, compute-bound). That is the residual of
   nvfortran's register allocation, and it is the same number every time. The one outlier
   (EPBL, 1.33×) is a **compiler bug**, not a language property — see §3.
2. **The algorithm is where the money is.** Redi's 1.35× — one hoist, bit-identical,
   portable Fortran — is worth ~4× more than CUDA's best edge anywhere, and it is
   language-neutral (CUDA would get the same win from the same change).

**Total addressable, measured, portable Fortran, no CUDA: ~16% of wall time.**
`ocean_redi` 10.4% + `ale_remap` 1.79% + `kshear` 1.43% + `barotropic` 1.36% +
`continuity` 0.64%.

---

## 1. ⚠ READ THIS BEFORE TRUSTING ANY NUMBER

### 1.1 The build flags are load-bearing. `-gpu=tripcount:host` or everything is ~2× wrong.

NVHPC 26.5 reads loop trip-counts from the device copy, inserting a per-kernel data
refresh in multi-loop `!$acc kernels` regions. `nsys` sees one ~184-byte
`cuMemcpyDtoHAsync` per `do concurrent` loop at ~21 µs (async-to-**pageable** is
synchronous), 63% of API time.

**the ocean model already guards this** — `cmake/compiler_flags.cmake:66-72`, gated on
`CMAKE_Fortran_COMPILER_VERSION >= 26.5`, citing **NVIDIA TPR #38714** and "~2x on the
barotropic solver".

I did not read their flags first and burned hours "explaining" a 2.3× gap between my
harness and production that was **entirely this one flag**. Every intermediate finding
downstream of it was an artifact (see §1.3). **Copy production's flags verbatim:**

```
-Mfree -Mbackslash -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate -gpu=tripcount:host -O3 -fast
```
from `<model>/build/CMakeFiles/core_model_objs.dir/flags.make`. Re-read that file; do not
trust this one.

### 1.2 The profile and the benchmarks ran on DIFFERENT HARDWARE.

- **Profile shares** (Redi 40.1%, kshear 18.5%, …): **H200** (`<h200-node>`,
  `build_hopper`, cc90, `NZ_STACK_MAX=256`).
- **Every benchmark ratio in §0**: **V100** (cc70, `NZ_STACK_MAX=128`), ~900 GB/s vs
  H200's ~4.8 TB/s.

Multiplying a V100 ratio by an H200 share — which §0 does — is an **extrapolation**.
Codegen ratios should travel; anything resting on occupancy or bandwidth may not. Redi at
25.5 ms/call (H200) vs 44.72 ms/call (V100) = 1.75×, which is at least self-consistent.
**Re-running the benchmarks on the H200 is the single cheapest way to firm up §0.**

### 1.3 Mechanism claims that turned out FALSE (this session alone)

Stated as hypotheses, tested, killed. Do not re-derive them.

| claim | status |
|---|---|
| "`!$acc kernels` wrapper costs 1.42× (per-loop scalar D2H copies)" | ❌ artifact of missing `tripcount:host`. With the flag, the wrapper is **0.59×** — it works as designed. |
| "Plain DC beats the ACC wrapper" | ❌ same artifact. |
| "bug 3: the ACC wrapper spills registers" (RESUME §5) | ❌ **does not reproduce** — MEKE found zero spills in every variant, including register-heavy fused loops. Probably the same artifact. |
| "`NZ_STACK_MAX=128` is 4× more than nz=30 needs → shrink it" | ❌ **null twice.** Redi tried 64, ALE tried 256: **no change**. Local-mem traffic scales with what the code *touches*, not the declared frame. |
| "Warp-per-column will rescue the column kernels" | ❌ **inapplicable** to EPBL: the k-sweep is a first-order sequential recursion (tridiagonal forward elimination) — nothing for `__shfl` to reduce. |
| "The signature fix generalises" | ❌ on EPBL it cuts loads 178→74, is bit-identical, and is **3% slower**. Loads were not the binding constraint. |
| "Modules with N loops and no async are the audit target" | ❌ wrong screen. **Loop *cheapness* is the trigger**, not loop count. ALE has 16 loops, no async, and async is worth **1.008×** — its 4 heavy kernels are 91% of the call. **Check `time_per_call / n_loops` first.** |
| "Descriptor loads via `bt_work%` are the btstep cost" | ⚠ half-true: plain dummies cut Pass 1 loads 103→38 (below CUDA's 47) but bought only 1.12× until `tripcount:host` was fixed; then 1.30×. |
| "`host_data use_device` puts the OpenACC runtime (a present-table lookup per array per call) on the CUDA path, so every 'CUDA buys N%' number here is understated and unfair to CUDA" | ❌ **for btstep, tested and killed.** `btstep_benchmark/btstep_native.cu` — `cudaMalloc` + `main()`, no Fortran, no OpenACC, calling the *same* `btstep_kernel.o` — **ties**: faithful 3.009 vs 2.996, fused 2.629 vs 2.602, graph 2.477 vs 2.457 ms (native listed first; 12 clean samples each, alternating order, idle-verified). Native is 0.4–1.0% *slower*, i.e. noise. It was never going to differ: `btstep_bench.F90` calls `fill_args` **once, outside the timing loop**, so the timed loop holds no `host_data` at all. Verified **bit-exact** (`max\|d eta\| = 0.0` after 5040 substeps), so kernels+inputs are identical and OpenACC is the only variable left. ⚠ **Scope: btstep only.** Ports that call `host_data` *inside* their timing loop are NOT cleared by this — check where `host_data` sits relative to the timer before quoting a number. |
| "A native C++ driver that recomputes the Fortran driver's init from the same formulas will reproduce it bit-for-bit" | ❌ **false, and it looks exactly like a port bug.** nvfortran's libm and glibc's libm round `exp()`/`sin()` **1 ulp** apart (29301/145137 of btstep's Gaussian bump; 46/303 of `force_u`). Over 5040 substeps that 1 ulp grows to `3.1e-15` on `\|eta\|~0.265` — indistinguishable from the FMA-level bar this repo uses to *accept* a port. **A cross-language bit-comparison must hand over the transcendental inputs, not recompute them**, or it is measuring libm. btstep's ref dump carries `eta_ini`/`force_u`/`f_corner` for this reason. |

**Session 1's list is in `RESUME_GPU_MRE.md` §1 and still applies** — especially: `ptxas`
"Used N registers" lines must be matched to the right mangled symbol (`*_gpu`); the first
one is `cub::EmptyKernel`.

### 1.4 Contention

Up to 5 agents shared this V100. Redi saw a **2.5× bimodal swing** (161.8 vs 404.5 ms)
purely by position in the run. **Agent-produced timings need a serial re-run** on an idle
GPU. Bit-identity results are unaffected. My own btstep/continuity/HLL numbers predate the
agents and are clean.

**A/B comparisons under contention: alternate the order, or you measure the order.**
Contention here is *intermittent* — idle for a few seconds, then 5 siblings — which
defeats "run A, run B, compare" AND "min across repeats of (A then B)": whichever binary
runs first in each repeat systematically grabs the idle window. btstep's first
native-vs-Fortran script did exactly that and made the native driver look **1.5× slower**
purely from always running second. It is not subtle and it looks like a real result.

What works (`btstep_benchmark/native_vs_fortran.sh`): each sample runs **one** binary,
checks the GPU is idle immediately **before and after**, and is **discarded** unless both
pass; the order **alternates**; take min-of-clean-samples. Keep each run short (~1 s) so a
sample fits inside a typical idle window. Sanity check that it worked: the clean samples
must reproduce known-idle reference numbers — btstep's did (faithful 2.996 vs 3.004 ref,
fused 2.602 vs 2.647).

---

## 2. What to do — ranked by measured value

All bit-identical, all portable Fortran, no CUDA.

1. **`ocean_redi` — 1.35× = 10.4% of runtime.** `redi_apply_flux_impl` rebuilds **8 full
   PPM columns per cell when only 5 distinct exist**. Hoist into a precompute pass.
   Working proof: `redi_benchmark/ocean_redi_pre.F90`. Mechanism is counter-intuitive
   and worth knowing: occupancy **unchanged** (10.8→10.9%), registers **up** (138→194),
   but **local-mem ops fell 17×** (2081→125), DRAM 51%→33%, L1 hit 43%→79%.
2. **`ocean_ale_remap` — 1.455× = 1.79%.** T and S share column geometry that production
   rebuilds twice; fusing cuts local traffic 28%. Plus a signature fix on
   `compute_target_h`.
3. **`ocean_vmix_kshear` — 1.08× = 1.43%.** `-gpu=maxregcount:96` (one build flag) +
   bounded whole-array copies (3 lines).
4. **`ocean_barotropic_solver` — 1.415× = 1.36%.** Signature fix (plain dummies, 8 args →
   34) + fusion 11→5 loops. Cost: an ugly signature. See `btstep_benchmark/`.
5. **`ocean_continuity` — 1.10× = 0.64%.** Marginal; probably leave alone.

**Do NOT write CUDA.** It buys ~1.7% total across everything measured, and on Redi the
algorithmic fix beats the (unfixed) CUDA port outright.

### Cheap, high-value, not yet done

- **`redi_calc_coeffs_x/y` = 9.2% of GPU time** and the Redi agent flagged they have **the
  same 8-vs-5 redundancy**, estimating ~2× — but explicitly labelled it *"a static
  call-graph read, not a result."* **This is the highest-upside unverified claim we have
  (~4.6% of runtime).**
- **Add a `profiler_start("ocean_meke")` region.** MEKE has none, so its cost is invisible
  and its measured 1.67× cannot be valued. Same one-line trick that turned Redi from
  "0.0% in every log" into "40.1%, the biggest win of the session".
- **Re-run every benchmark on the H200**, serially. See §1.2, §1.4.

---

## 3. Compiler bugs

| # | bug | status |
|---|---|---|
| 1 | Auto-collapse silently lost: explicit-shape dummy bounded by a **derived-type component** + a **procedure call** in the body. Either alone is fine. No diagnostic. | **Filed.** `nvbug_dc_collapse/` |
| 2 | CSE lost across an inlined call boundary when the **callee does its own array indexing**. 10 vs 6 LDG for identical semantics. | **Filed.** `nvbug_inline_cse/` |
| 3 | **Trip-count regression** — device-side trip-counts insert a per-kernel data refresh in multi-loop `!$acc kernels` regions. ~2×. | **Filed by the ocean model as TPR #38714.** Worked around with `-gpu=tripcount:host`. I rediscovered it independently from `nsys` before finding their CMake comment. |
| 4 | **`maxregcount` allocator quality.** At an identical 128-register cap: nvfortran `-gpu=maxregcount=128` spills **568 B → 10.95 ms**; nvcc `__launch_bounds__(128)` spills **160 B → 4.87 ms**. Both languages can ask; only nvcc survives. | **NOT filed. Two-line repro available in `epbl_benchmark/`.** ⚠ **Contested**: on kappa_shear, `-gpu=maxregcount:96` *helped* (1.119 → 1.074). So it is not universally broken — characterise before filing. |
| 5 | **Collapse lost with `associate`-names.** `compute_target_h` reports `Loop run sequentially` for k in **production** while every other 3-D loop gets `collapse(3)`. Cause looks like `associate` names aliasing DT components. One-line fix, 1.06×. | **NOT filed.** ⚠ **Extends bug 1**: our filed trigger needs a call *and* a DT array bound; here there is **neither**. **Our filed report may be narrower than the real defect.** Not isolated — flagged as a symptom. |

---

## 4. The production profile — `gabight_sph_meke_v100.nml`

This is the config that exercises everything (MEKE + Redi + EPBL + kappa-shear + PP81 all
live). **The other configs do not** — `gabight_acc_120d` has Redi at **0.0%**, which is
why Redi's 40.1% went unnoticed until today.

```
  ocean_redi                40.1%     <- the biggest thing in the model
  ocean_vmix_compute        21.2%
    ocean_vmix_kshear       18.5%       (87.5% of the vmix region)
    ocean_vmix_epbl          1.8%       (8.3%)
    ocean_vmix_pp81          0.6%       (2.7%)
  ocean_continuity           7.0%
  ocean_ale_remap            5.7%
  ocean_hvisc                4.8%     <- never benchmarked
  ocean_barotropic_solver    4.7%
  ocean_vdiff_apply          3.9%     <- never benchmarked
  ocean_gm                   3.7%     <- never benchmarked
```

**The `ocean_vmix_{pp81,epbl,kshear}` split was added today** —
`ocean_dyn.F90:826-880`, nested inside `ocean_vmix_compute`. They **overlap** the
parent rather than add to it; `profiler_report` computes % against its `root_region`, so
the Total is unaffected. Before the split, "vmix is 39%" was universally read as EPBL. It
is **kappa_shear, 10× EPBL**.

Per-kernel GPU time (`nsys`, `<model>/report2.nsys-rep`):

```
  redi_apply_flux_impl         32.3%   avg 15.06 ms   13.2 -> 17.3   (metronome)
  kappa_shear_column_kernel    19.0%   avg 35.37 ms    6.1 -> 54.6   <- 9x SPREAD
  redi_calc_coeffs_x            4.6%   avg  8.54 ms
  redi_calc_coeffs_y            4.6%   avg  8.50 ms
  epbl_column_kernel            1.6%   avg  2.89 ms
```

**kappa_shear's 9× spread (σ = 39% of mean, vs Redi's 4%) is the iterative solver inside**
— confirmed by the code owner. Cost tracks how hard the flow is to converge, not domain
size. Note **median (38.1) > mean (35.4)**: left-skewed, so the *expensive* case is the
normal one. Any synthetic benchmark state must converge as hard as this or it measures the
wrong regime.

---

## 5. Method — what transferred better than any single result

- `-Minfo=accel,stdpar` → collapse status (`auto-collapsed`, or its absence).
- `cuobjdump -sass` → LDG (CSE), **LDL/STL (spills)**, instruction counts.
- `cuobjdump -res-usage` → REG / STACK / LOCAL per kernel.
- `ncu` → occupancy, roofline, lane efficiency. **Occupancy moves in cliffs, not
  gradients** — 56→58 registers cost 8% (9→8 blocks/SM); 126→114 cost nothing (both 4).
- `nsys` → GPU-busy vs wall (launch-bound?), API mix, per-kernel spread.
- `NVCOMPILER_ACC_NOTIFY=2` → data transfers. This is what cracked the trip-count bug.
- **Verify the verification.** Perturb a term by 1e-7 and confirm the agreement check
  trips. MEKE and ALE both did this; it is how you know `max|diff| = 0.0` means something.

**Diagnose the regime before optimising.** The four regimes seen:
| regime | tell | lever |
|---|---|---|
| launch-bound | wall >> GPU-busy; cheap loops | async wrapper, fusion |
| compute-bound | `ncu` Compute% > Memory%; FP64 pipe saturated | less arithmetic (precision) |
| spill-bound | LDL/STL >> LDG; DRAM high but it is spill traffic | less redundant per-column work |
| divergence-bound | huge per-call variance; lane efficiency low | algorithmic |

**State mechanisms as hypotheses and kill them.** Eight died this session (§1.3), including
several of mine that would have shipped as findings.

---

## 6. Where things are

```
mre_acc_cuda/
  LOGBOOK.md                  this file (session 2)
  RESUME_GPU_MRE.md           session 1: daxpy, HLL flux, the two filed bugs
  nvbug_dc_collapse/          FILED. single-file repro + probe series
  nvbug_inline_cse/           FILED. single-file repro, self-proving via cuobjdump

  redi_benchmark/             ⭐ the big one. 1.35x = 10.4% of runtime
  kappa_shear_benchmark/      18.5% of runtime. 1.048x off CUDA after 1 flag + 3 lines
  ale_remap_benchmark/        1.455x = 1.79%
  btstep_benchmark/           1.415x = 1.36%. sigfix + fusion; also the CUDA graph result
  epbl_benchmark/             1.8% only. The maxregcount bug lives here
  meke_benchmark/             1.67x, but no profiler region so value unknown
  continuity_layered_benchmark/  1.10x. the kernel ocean_continuity actually wraps
  continuity_ppm_benchmark/   the BAROTROPIC continuity — ⚠ TEST-ONLY, not in production
  hll_fluxes_benchmark/       session 1. the coastal path
  daxpy_benchmark/            session 1. proves nothing (bandwidth-bound)
```

Every `*_benchmark/` has a README with its own numbers, caveats and dead ends. **Read the
kernel's README before touching it** — several document hypotheses already killed.

⚠ `continuity_ppm_benchmark/` measures `continuity_compute_fluxes_barotropic`, which has
**no caller in `src`** — only `tests/`. It is dead code in the ocean configs. The layered
`continuity_compute_fluxes` is what `ocean_continuity` wraps.

### Build / run

```bash
source ../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
cd <kernel>_benchmark && make && make run
```
Most take CLI args: `./bench [nx] [ny] [nz] [nreps] [nwarm]`. Production sizes are in
`~/analysis_gebco/*.nml` — **`nx`/`ny` are POINT COUNTS**, extent = `nx*dx`. Most common:
**473 × 297, nghost=3, nz=30**. Max nz across all configs is **50**.

**Do not benchmark only at 4096².** The dc/CUDA gap is a strong function of total cell
count for the launch-bound kernels (1.61× at 145k cells → 1.03× at 17M) and 4096² makes
everything look clean while nothing runs there.

---

## 7. Open questions

1. **`redi_calc_coeffs_x/y` (9.2% of GPU) — same 8-vs-5 redundancy?** Estimated ~2×,
   explicitly unverified. **Highest upside available (~4.6% of runtime).**
2. **Re-run everything on the H200**, serially. §1.2 + §1.4 both gate §0's numbers.
3. **`ocean_hvisc` (4.8%), `ocean_vdiff_apply` (3.9%), `ocean_gm` (3.7%) — never
   benchmarked.** ~12.4% of runtime, completely unexamined.
4. **Is bug 4 (`maxregcount`) real or config-dependent?** It hurt EPBL and helped
   kappa_shear. Characterise before filing.
5. **Bug 5 (`associate` + collapse) may mean our filed bug 1 is too narrow.** Worth
   isolating — it affects what NVIDIA fixes.
6. **MEKE has no profiler region**, so its 1.67× is unvalued.
7. **`dt_h` is read uninitialized in EPBL** on the short-circuit path (EPBL agent, found
   incidentally). Benign in its test state; not proven unreachable in production. **This is
   a correctness question, not a performance one.**
8. **nvfortran inlines none of Redi's 15 `!$acc routine seq` helpers** — not even a 3-line
   `sign()` wrapper. Not established to cost anything. But it **contradicts RESUME §1's**
   finding that `flux_cell` was inlined all along, which is the foundation the whole bug-2
   story rests on.
9. **Mixed precision.** `ncu` on the HLL flux kernel: *"achieved 1% of FP32 peak and 38% of
   FP64 peak … Est. Speedup 25.8%"*. V100/A100/H200 are all 2:1 FP32:FP64, so this is
   structural, not hardware-specific. A science decision, not a code one — but it is the
   only lever that attacks what is actually saturated on the compute-bound kernels.
