# RESUME — nvfortran `do concurrent` vs CUDA C investigation (SESSION 1)

_Cold-start handoff, 2026-07-15. Read §0 and §1; everything else is detail._

> ## ⚠ SUPERSEDED IN PART — READ `LOGBOOK.md` FIRST
> Session 2 (2026-07-16) benchmarked seven kernels against CUDA C and profiled the
> config that actually exercises them. **Where the two disagree, LOGBOOK.md wins.**
> Specifically it corrects:
> - **"bug 3" (the ACC wrapper spilling registers) does NOT reproduce** — almost
>   certainly an artifact of building without `-gpu=tripcount:host` (NVIDIA TPR #38714,
>   which <model> already guards for). **Nothing in this file was built with that flag,
>   so treat every timing here as suspect until re-measured.**
> - `continuity_ppm_benchmark/` measures a **test-only** routine with no caller in `src`.
> - §5b's "pass scalars, not arrays" fix does **not** generalise (3% *slower* on EPBL).
> - §1's "`flux_cell` was inlined all along" is contradicted by Redi, where nvfortran
>   inlines **none** of 15 `!$acc routine seq` helpers. Bug 2's mechanism may need review.
>
> The headline conclusion of this file — that the coastal HLL kernel loses ~1.4x to two
> compiler bugs, recoverable in portable Fortran — **still stands**.

> **⚠ NOTHING HERE IS IN GIT.** `<scratch>/dev/mre_acc_cuda/` is a plain
> directory — no repo, no remote, no backup. A full day of work. **Decide where it lives
> before doing anything else** (§6).

---

## 0. TL;DR

Three benchmarks, built to answer "does `do concurrent` cost us anything vs hand-written
CUDA C on a GPU-native Fortran ocean model?"

| benchmark | verdict |
|---|---|
| `daxpy_*` (root dir) | **No difference.** dc == CUDA C == 810 GB/s. Bandwidth-bound ⇒ everything ties at the memory wall. Proves nothing about compilers; don't generalise from it. |
| `hll_fluxes_benchmark/` | **`do concurrent` was 1.4× slower** than CUDA C on the ocean model's production HLL kernel. Cause found: nvfortran silently stops auto-collapsing. **Fixed** — flat `do concurrent` is now **1.46× the shipped kernel and beats hand-written CUDA C**, bit-identically, with full portability. |
| `continuity_ppm_benchmark/` | **~1.3-1.7x at PRODUCTION sizes** (`<configs>/*.nml`: 108x137 -> 1.71x, 473x297 -> 1.60x, 945x594 -> 1.30x). Only 4096² looks clean (1.06x) and nothing runs there. Bit-identical. **Different defect from the flux kernel**: codegen is fine, both bugs absent; it is ~10 us dead host time between each of the 9 loops. `!$acc kernels async(1)` makes it 1.3-1.6x WORSE (construct spills registers 56->90; `private()` does NOT fix it; mechanism unknown). Possible bug 3. |
| `continuity_layered_benchmark/` | **THE kernel `ocean_continuity` wraps** (11% runtime w/ EPBL, ~20% without). dc/CUDA-C = **1.10x at 0.1deg (4.35M cells), 1.03x at 0.05deg, 1.45x at 10km**. Bit-identical. Already at 90% of hand-written CUDA -> a CUDA rewrite wins ~2% of total. Both bugs absent; all 11 loops collapse(3)/(2). |
| `btstep_benchmark/` | **Portable Fortran gets 1.42x; a CUDA rewrite buys 6% more.** Production 3.704 ms -> signature fix 2.851 -> +fusion **2.618 ms**, all bit-identical (`max|d eta| = 0.0` exactly). CUDA faithful 3.035 / fused 2.646 / graph 2.464 — **fused Fortran BEATS fused CUDA** (0.985x); only CUDA graphs are out of reach, worth **1.062x**. Two free wins: plain dummies vs `bt_work%` components (**1.30x**, Pass 1 goes 103->38 loads, beating CUDA's 47; cost: 8 args -> 34) and fusing 11 loops -> 5 (**1.09x**). Worth ~2.5-4.7% of total runtime. ⚠ Needs `-gpu=tripcount:host` (NVHPC 26.5 regression, NVIDIA TPR #38714, already guarded in <model>'s cmake). |
| `nvbug_dc_collapse/` | Minimal reproducer + bug report for NVIDIA. **Ready to send.** |

**The single actionable finding for the ocean model:** `~1.4×` on the coastal barotropic flux kernel,
recoverable in pure portable Fortran. See §2.

**⚠ COPY PRODUCTION'S BUILD FLAGS FIRST.** An MRE that models production must use
production's flags (`<model>/build/CMakeFiles/core_model_objs.dir/flags.make`):
`-Mfree -Mbackslash -stdpar=gpu -acc=gpu -gpu=<arch>,mem:separate -gpu=tripcount:host
-O3 -fast`. The btstep benchmark burned hours "explaining" a 2.3x gap that was
entirely the missing `-gpu=tripcount:host` -- an NVHPC 26.5 regression <model> already
guards for (`cmake/compiler_flags.cmake:66-72`, NVIDIA TPR #38714). Every finding
downstream of it was an artifact.

**⚠ MEASURE AT PRODUCTION SIZES.** The dc-vs-CUDA gap on the continuity kernels is a pure
function of TOTAL CELL COUNT (launch overhead amortisation), not of the kernel:
~1.6x at ~145k cells, ~1.10x at ~4.3M, ~1.03x at ~17M. Both continuity kernels land on
the same curve — the layered one at nz=1 (145k cells) reads 1.609x, matching the
barotropic's 1.60x at 140k. A 4096²-only benchmark says "clean" about a kernel that is
1.6x at the size it actually runs. Real configs: `<configs>/*.nml` (nx/ny are
POINT COUNTS; extent = nx*dx). Real profile:
`<scratch>/<user>/tassie_runs/gabight_acc_120d.*.log` — vmix 39.1%, ale_remap 11.2%,
continuity 11.1%, barotropic_solver 8.6%. **`ocean_vmix_compute` (EPBL + kappa-shear) is
the real bottleneck at ~40% and is UNMEASURED** — with EPBL off it leaves the budget and
continuity becomes co-dominant (~20%) with ale_remap.

---

## 1. ⚠ READ THIS BEFORE TRUSTING ANY MECHANISM CLAIM

During this session I proposed **four** mechanisms for the 1.4×. **Three were wrong.** They
are listed here so nobody re-derives them:

| claim | status |
|---|---|
| "The un-inlined `call flux_cell` + its 40-byte spill frame is the cost" | ❌ **FALSE** |
| "`-Minline` will fix it" | ❌ **FALSE** — it made it *worse* (7.81 → 8.19) |
| "A procedure call in a DC body blocks auto-collapse" | ❌ **FALSE** — a trivial call collapses fine |
| "Launch geometry explains it" | ❌ **FALSE** — CUDA C wins at *every* geometry |

**Root cause of the errors:** I read `Used 4 registers` out of `ptxas` output and attributed
it to `compute_flux_hll`. It is **`cub::EmptyKernel<void>`**, a CUB artifact printed in the
same dump. Checked properly, the kernel entry uses **120 registers** — `flux_cell` was
**inlined all along, in every variant**. There was never a call to remove. Every "inlining"
conclusion built on that line is void.

**Lesson for the next session:** `-Minfo` collapse reports and wall-clock timings are
reliable. `ptxas` register/spill lines must be matched to the right symbol — grep for the
kernel's actual mangled name (`*_gpu`), never take the first `Used N registers`.

**What IS established** (verified across six probe files by `-Minfo`, which is trustworthy):

> nvfortran silently disables `do concurrent` auto-collapse when **both**:
> 1. an explicit-shape array dummy is bounded by a **derived-type component**
>    (`a(grid%nx_total, grid%ny_total)`), **and**
> 2. the loop body contains a **procedure call**.
>
> Either alone is fine. No diagnostic. Bit-identical results, worse schedule.

**The mechanism is NOT known.** `-Minline=reshape` ("allow inlining even when array shapes do
not match") is the only flag that restores it, which *hints* the shape relationship is what
the collapse analysis chokes on — but that is a hint, not a proof. The bug report says so.

---

## 2. The the ocean model action item (the reason any of this matters)

`src/core/coastal/kernels/structured/barotropic/kernel_flux.F90` — the mature coastal
path — leaves **~1.4×** on the table. Measured, 4096², V100:

```
do concurrent, as shipped                     7.02 - 8.2 ms   NOT collapsed
  + signature fix (plain nx/ny dummies)       6.75 ms         collapses; only ~4% back
  + OpenACC collapse(2) vector_length(512)    6.13 ms
  + body hand-flattened, PLAIN do concurrent  5.24 ms   <-- portable, beats CUDA C
hand-written CUDA C (faithful port)           5.52 ms
```

All **bit-identical** (0 cells differ, verified FLAT vs CUDA directly).

**Two independent costs stack**, and neither fix alone is enough:
- the lost collapse (fixed by plain-integer bounds — a signature change)
- something else worth ~22% that only the flat body recovers — **not isolated**

**The recommended fix is the flat rewrite** (`kernel_flux_flatdc.F90` is a working
proof): plain `do concurrent`, no OpenACC, no CUDA, gfortran/ifx portability intact, and it
beats hand-written CUDA C. Cost: a 530-line flat loop body is materially worse to maintain
than `flux_cell`. That is the "pay the divergence cost only when the profile justifies" call
CLAUDE.md already frames — this is a profile justifying it.

### CLAUDE.md amendment (cheap, applies to every DC kernel taking a descriptor)

The existing rule is **incomplete**:

> **Assumed-shape dummies in `do concurrent` kernels**: … use explicit-shape args
> (`arr(nx,ny,nz)`)

`h(grid%nx_total, grid%ny_total)` **is** explicit-shape and still kills the collapse. The
bounds must come from **plain integer dummies**. One-line change; affects most ocean kernels.

---

## 3. What's where

```
mre_acc_cuda/
  RESUME_GPU_MRE.md          this file
  README.md                  daxpy: 3 variants (dc / OpenACC+CUDA C / pure CUDA C).
  daxpy_*.{F90,cu}           Finding: all identical (810 GB/s); memcpy costs 66-268x if
  Makefile                   you transfer per step -> the argument for device residency.
  aa                         <-- junk, delete

  hll_fluxes_benchmark/      THE MAIN RESULT. 7 variants of the ocean model's production HLL kernel.
    README.md                  Read this. kernel_flux.F90 is byte-identical to
    kernel_flux.F90        production (md5 1ea40efad8567a85914561db5bfc3a55).
    kernel_flux_dims.F90     signature fix (plain nx/ny)
    kernel_flux_acc.F90      explicit OpenACC collapse(2), -DACC_VLEN=N
    kernel_flux_flat.F90     CPP-expanded flat body + OpenACC
    kernel_flux_flatdc.F90   flat body + PLAIN do concurrent  <-- the winner
    kernel_flux_nest.F90     nested dc(j)/dc(i) — identical to fused, no help
    flux_kernel.cu               faithful CUDA C port
    flux_bench.F90               driver: all 7, same data, agreement check

  continuity_ppm_benchmark/  UNFINISHED. continuity.F90 = production
    continuity.F90         continuity_compute_fluxes_barotropic (428-609) + the 4 PPM
    constants.F90          helpers, verbatim, with metrics/bs/scratch/continuity_t
    grid.F90               stubbed to exactly the verified fields. COMPILES.
                               Finding: it AUTO-COLLAPSES (its arrays are derived-type
                               COMPONENTS `bs%h`, not grid%-bounded dummies) -> no bug 1.
                               CORRECTION: the first session said "its body is inline
                               arithmetic" -- FALSE, it makes 5 calls per iteration
                               (ppm_limited_slope x3, ppm_cell_limiter, ppm_limit_pos).
                               It dodges bug 2 for a DIFFERENT and more useful reason:
                               the caller HOISTS every array read into a scalar
                               (`h0 = bs%h(i,j)`) and the helpers take SCALARS. See §5b.

  nvbug_inline_cse/          NVIDIA bug report #2: lost CSE across the inline boundary.
    repro.f90                  single file, self-proving via cuobjdump LDG counts
                               (10 vs 6 for identical semantics). READY TO SEND.

  nvbug_dc_collapse/         NVIDIA bug report. READY TO SEND.
    README.md                  the report — trigger stated as fact, mechanism explicitly
    repro.f90                  marked unknown, our own wrong readings documented.
    repro_mod.f90 repro_main.f90   split for the -Minline=reshape demo
    probe{,2,3,4,5,6}.f90      the elimination series (call / arrays / nested / pure /
                               DT-bounds / routine seq) — each rules something out
```

## 4. Build / run

```bash
source ../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh   # NOT nvhpc_env.sh
cd hll_fluxes_benchmark && make && make run       # ~30 s, 7 variants + agreement
cd ../nvbug_dc_collapse && nvfortran -O2 -stdpar=gpu -gpu=cc70,mem:separate repro.f90 -o repro && ./repro
```

The HPC system analysis node has a V100 (cc70). `-gpu=cc80 NVARCH=sm_80` for A100.

**Harness trap that already bit once:** in `flux_bench.F90`, variants sharing an output array
means only the *last writer* is checked — three variants silently inherited "agreement OK"
from whoever ran last. FLAT has its own `fh_fl`; the others still share `fh_dc`. **Give each
variant its own arrays before trusting a new one.** (This produced a phantom "885197 cells
differ" that I misdiagnosed as a kernel bug for an hour.)

## 5. Open questions

1. ~~**The unexplained ~22%.**~~ **ANSWERED (2026-07-15 follow-up session)** — see §5a.
2. **`-Minline=reshape` on the real kernel** restored collapse but gave only 7.39 — worse
   than the signature fix's 6.75. Consistent with §5a: inliner output defeats CSE.
3. ~~**`continuity_ppm_benchmark` needs a driver**~~ **DONE** — driver + FLAT + ACC-async
   + a faithful CUDA C port (9 kernels, host_data use_device, bit-identical). Sizes and
   reps are now CLI args. See `continuity_ppm_benchmark/README.md`.
   **New loose end: the OpenACC-wrapper register spill (possible bug 3)** — `!$acc kernels`
   around a register-heavy DC loop takes it 56→90 registers with 16 spills, costing 2.3x
   on that kernel. `async` is NOT the cause (bisected: `kernels` alone spills identically;
   `parallel loop` spills 33). Production uses this idiom in
   `<model>/.../barotropic_substep.F90` — its loops are cheap so likely fine, but
   **any register-heavy wrapped loop is worth checking.** No minimal repro yet.
4. ~~`continuity.F90:164` not collapsed~~ **NON-ISSUE** — it is a **1-D** boundary loop
   (`do concurrent(j=1:ny)`); there is nothing to collapse. All five 2-D loops collapse.
5. Wet/dry state for the flux bench (currently all-wet, so the dry branches never fire).

### 5a. The ~22% — SOLVED: nvfortran does not CSE across the inlined call boundary

`ncu --set detailed` on dims (6.75 ms) vs flatdc (5.24 ms), plus static SASS opcode counts
(`cuobjdump -sass`, matched to the right mangled symbols this time). Dynamic and static
agree exactly.

| per thread / per kernel        | dims  | flatdc | CUDA C |
|--------------------------------|-------|--------|--------|
| global LDG per thread          | **66**| **56** | **56** |
| SASS instructions (static)     | 2417  | 2094   | 2509   |
| DFMA (static)                  | 374   | 298    | 380    |
| sqrt/MUFU expansions           | 49    | 37     | 52     |
| registers                      | 126   | 114    | —      |
| theoretical occupancy          | 25%   | 25%    | —      |
| local loads/stores (spills)    | 0     | 0      | 0      |
| warp inst executed (dynamic)   | 1147M | 990M   | —      |
| long-scoreboard stalls (pcsamp)| 56616 | 37026  | —      |

- **flatdc reaches the CUDA-optimal 56 loads/thread exactly.** The inlined-`flux_cell`
  variant does 10 redundant loads per thread (+18%), ~25% more DFMA, 12 extra sqrt
  expansions — recomputation of values the flat compile reuses. (Shipped kernel: 69 LDG.)
- Everything else is a tie: same launch geometry (131072×128), same 25% occupancy
  (register-limited to 4 blocks/SM in BOTH — 126 vs 114 regs does not change the limit),
  zero spills in both, identical global stores (5/thread) and DRAM active cycles.
- Cost mechanism: the extra loads mostly hit L1/L2, but at 25% occupancy their latency
  can't be hidden — long-scoreboard stalls +53%, issued IPC 1.56 vs 1.76.
- Both sources contain the same 8 `sqrt`s and the call is the entire loop body, so the
  dims/flat sources are semantically identical — the ONLY variable is inlined-vs-source.

**Trigger ISOLATED by probe bisection (same session), hypothesis testing done:**

> The redundant loads appear when the inlined callee contains a **branch whose condition
> reads array elements that are also read elsewhere in the body**. The flat form CSEs the
> condition's loads with the arithmetic's; the inlined form reloads them.

Probe evidence (`cuobjdump -sass` LDG counts, call vs flat, nvfortran 26.5-0, -O2 and -O3
identical): straight-line callee 6=6 (no bug); callee with array-condition branch 10 vs 6;
callee with nested minmod-style function (branch inside) 10 vs 6; both 11 vs 6; branch on
scalar `i` only 3=3; branchless `merge()` 3=3. The first-session store-aliasing hypothesis
is REFUTED: `flux_cell` outputs scalars, no array stores in the callee, and a probe with
interleaved array stores but no branches showed no redundancy.

Single-file NVIDIA repro: `nvbug_inline_cse/repro.f90` (verified: 10 vs 6 LDG, bit-identical
results, provable by compile+cuobjdump alone, no GPU run needed). NOTE the toy's wall-clock
ties (bandwidth-bound, full occupancy) — the 22% needs register-pressure/low occupancy to
materialise, which the report says explicitly.

Artifacts: ncu reports were written to the session scratchpad (ephemeral); regenerate with
`ncu -k <mangled_name_without_gpu_suffix> --launch-skip 1 --launch-count 2 --set detailed`.
Note ncu sees kernel names WITHOUT the `_gpu` suffix that `cuobjdump -symbols` shows.

### 5b. Bug 2 has a CHEAP fix: pass scalars, not arrays  ← likely supersedes the flat rewrite

Refined trigger, probe-verified: the redundancy needs the **callee to do its own array
indexing**. Hoist the loads in the caller and hand the helper plain scalars, and the
inlined code is byte-identical to the flat form.

| probe variant (identical semantics)                    | LDG | total inst |
|--------------------------------------------------------|-----|-----------|
| callee takes arrays + indexes internally (`flux_cell`)  | 10  | 184 |
| caller hoists loads, callee takes scalars (PPM shape)   | **6** | **152** |
| flat reference                                          | **6** | **152** |

This explains continuity: it is NOT flat, it makes 5 calls per cell — but it hoists
(`h0 = bs%h(i,j)` … then `call ppm_limited_slope(hm2, hm1, h0, dh_m1)`). The helpers see
scalars, so there is no array for the inliner to reload. The idiom was already correct.

**Implication for the ocean model (UNVERIFIED AT SCALE — do this next):** the recommended fix may
not be the 530-line flat rewrite at all. Changing `flux_cell(h, hu, hv, b, nx, ny, i, j…)`
to take the stencil values as scalars would keep the helper AND (if it holds) the flat
performance. CAVEAT: the probe passes 6 scalars; `flux_cell` needs ~36 (9-point stencil ×
4 arrays). **Whether the effect survives at that argument count is untested** — that is the
one thing to check before recommending it. If it holds, this is a far cheaper fix and the
maintainability argument in §2 dissolves.

Also strengthens the bug report: a workaround that makes the inlined form optimal is
evidence about the mechanism (the callee's dummy-array indexing is what defeats CSE).

## 6. ⚠ Housekeeping — do this first

- **`mre_acc_cuda/` is in no git repo.** Options: a new repo; a `benchmarks/` subtree in
  <model>; or a `gpu-mre` branch. It is one `rm -rf` from gone.
- **`aa`** in the root is junk — delete.
- **`<model>-sea-ice` has 21 uncommitted files** from today: the CLAUDE.md aggregate-`update
  self` gotcha, the RESUME rewrite, and the `environments/toolkits/<machine>/<toolkit>.sh`
  restructure (+ `validate.sh` fail-loud + ~20 swept nml/README refs). All verified, none
  committed.
- **`<model>-plans`**: commit `d8d4c652` (41 plans) is **unpushed**, and the marathon
  reconcile left `PLAN_PR06/12/18` **modified after** that commit — uncommitted.
- The CLAUDE.md amendment in §2 is **not yet applied**.

_Session context: this grew out of an MRE request for OpenACC-allocated / CUDA-C-computed
interop. The daxpy answered that (`host_data use_device`, one directive) and found nothing
interesting perf-wise; the flux kernel is where the real result is._
