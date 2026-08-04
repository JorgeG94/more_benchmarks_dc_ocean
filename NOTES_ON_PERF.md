# Notes on performance

The findings, distilled. Numbers are V100 (cc70), NVHPC 26.5 / CUDA 12.9,
production size 473×297×30 unless noted.

## The answer

**A fully hand-optimized CUDA rewrite of the ocean core is ~8% faster than
optimized `do concurrent` — measured end-to-end, not estimated.** Every kernel
is either a memory-bound stencil or a register-bound column solver, and
nvfortran's codegen ties nvcc's closely on both; what gap remains is single-digit
and concentrated in the cheap kernels.

- **Measured, not weighted.** The whole 8-kernel ocean core, run in one hot-GPU
  RK2 loop (`ideal_benchmark/`), is **all-DC 101.5 vs all-CUDA 94.2 ms/stage →
  CUDA +7.8%** (V100, 473×297×30; stable across 5 clean runs). This *replaces*
  the old profile-share estimate with a direct measurement.
- **Amdahl runs the show.** The two column solvers `ocean_redi` + `ocean_vmix_kshear`
  are ~80% of a stage and near-parity (redi favours DC +7%, kappa favours CUDA
  +16%, net CUDA +4%); the remaining ~20% of light kernels is CUDA +27%. Blended
  → +8%. Speeding up a cheap kernel by 1.7× cannot move a total the giants own.
- **The OpenACC→CUDA bridge is free.** A native `cudaMalloc` driver ties the
  `host_data use_device` path in every kernel (~0.004%–10 µs/call), so the
  single-binary head-to-heads below are apples-to-apples: one device image, one
  clock, opt-CUDA vs opt-DC.
- **The algorithm is where the money is, and it's language-neutral** — the same
  portable fusion/hoist helps CUDA and DC, and (unlike the CUDA-only tricks)
  ports straight to the CPU. Portable Fortran optimizations landed 1.18×–1.34×
  over each kernel's own DC baseline, all bit-identical, all CPU-portable.

> Earlier drafts claimed "DC beats CUDA" on `btstep` and "DC ties on the giants."
> Both were **ratio-of-ratios / contention artifacts** — comparing a standalone
> nvfortran binary's speedup against a standalone nvcc binary's, and (in one case)
> a co-running job inflating the second-timed pass. The single-binary `cmp`
> harness and the exclusive-GPU RK2 runs corrected them. Keep the caveat in mind.

## The single-binary head-to-head (`make cmp` per kernel)

`opt-CUDA` and `opt-DC` launched from ONE binary on the SAME device arrays via
`!$acc host_data use_device` — the allocator/harness variable removed. Agreement
is FMA-contraction level (≤1e-14) everywhere, so these are true codegen deltas.

| kernel | opt-DC ms | opt-CUDA ms | winner |
|---|---:|---:|---|
| ocean_redi | 40.0 | 42.8 | **DC +7%** |
| ocean_hvisc | 1.36 | 1.42 | **DC +4%** |
| ocean_ale_remap | 8.4 | 7.0 | CUDA +20% |
| ocean_barotropic (btstep) | 2.87 | 2.67 | CUDA +7% (graph +13%) |
| MEKE | 0.30 | 0.19 | CUDA +57% |

DC wins the memory-bound stencils where nvfortran's codegen is as good or better;
CUDA wins the launch-count-bound light kernels (its async/graph edge is real but
low-leverage). `ocean_vmix_kshear` (register/divergence-bound, no opt on either
side) is CUDA +16% in the RK2 breakdown — the one heavy kernel that favours CUDA.

## Per-kernel: DC vs faithful CUDA (bit-identical unless noted)

Profile shares are the model's (biggest first).

| kernel | share | dc / cuda (faithful) | note |
|---|---:|---|---|
| ocean_redi | ~40% | Fortran **wins** | precompute hoist (8→5 PPM columns) = 1.35×, portable |
| ocean_vmix_kshear | ~18% | ~1.05× | iterative Picard solver; register-bound |
| ocean_continuity | ~7% | ~1.11× | fuses to 1.5× (CUDA) / 1.36× (DC) — see below |
| ocean_ale_remap | ~6% | ~1.06× | sigfix + fusion = 1.45× portable |
| ocean_hvisc (Smagorinsky) | ~5% | **0.98×** (DC edges) | full closure: strain→A_h + Laplacian apply |
| ocean_barotropic (btstep) | ~5% | fused DC **beats** fused CUDA | only CUDA graphs pull ahead (~6%) |
| ocean_vmix_epbl | ~2% | ~1.05× codegen | the 1.33× that exists is a **compiler bug** (below) |
| MEKE | ~1% | — | async + fusion = 1.67× portable |
| coastal HLL flux | — | flat DC **beats** CUDA | after fixing the lost-collapse bug |

## Case study: optimizing continuity_layered

11 PPM kernels → 3 by fusing each direction's reconstruction + boundary +
transport (recompute the upwind cell in registers, so the `hfl/hfr` scratch
arrays never touch global memory: ~280 MB/call gone) + 32-bit indexing.

- **CUDA: 0.93 → 0.62 ms = 1.50×.** **DC: 1.11 → 0.82 ms = 1.36×.** Bit-identical.
- **Fused DC (0.82) beats the *original* faithful CUDA (0.93).** Optimized CUDA
  then reopens a ~1.32× gap (the fused recon is register-heavy, where nvfortran
  trails nvcc — same story as kappa_shear / the HLL flat body).
- **Everything cleverer LOST** (kept as `OPTVER 3-8`): full recompute-fusion,
  k-loop / k-blocking for ILP, `__launch_bounds__`, div+flux fusion, shared-
  memory tiling. The kernel is memory-bound; the flat one-thread-per-face layout
  lets the GPU's own L2 + occupancy do the reuse better than any hand-rolled
  scheme. **Only removing genuine waste paid off** (the intermediate arrays,
  64-bit addressing). `kO_div` is already ~84% of DRAM peak.

## Compiler findings

Diagnose the regime first: launch-bound (→ fuse/async), compute-bound (→ less
arithmetic/precision), spill-bound (→ less redundant per-column work),
divergence-bound (→ algorithmic).

- **`-gpu=tripcount:host` is load-bearing** (NVHPC 26.5 regression, NVIDIA
  TPR #38714): without it, device-side trip-counts insert a per-kernel data
  refresh in multi-loop regions → timings ~2× wrong. It's in the Makefiles.
- **Bug 1 — lost auto-collapse:** a `do concurrent` stops
  collapsing when a callee's explicit-shape array dummy is bounded by an integer
  passed **by reference** + the body has a call. **Fix: pass the array *bounds*
  by `value`** (`nx,ny` — verified necessary & sufficient; the loop indices
  `i,j` by value do nothing).
- **Bug 2 — lost CSE:** redundant global loads when an
  inlined callee does its own array indexing. **Fix: hoist reads and pass
  scalars.** Inlining the loop body dodges both bug 1 and bug 2.
- **Bug 3 — `maxregcount` quality:** at an identical register cap nvfortran
  spills far more than nvcc's `__launch_bounds__` (EPBL: 568 B → 10.95 ms vs
  160 B → 4.87 ms). Config-dependent — it *helped* kappa_shear. Not universal.

### amdflang: DC→OpenMP device offload can't map derived types that contain derived types

**Status: hard blocker for the AMD GPU column, not a tuning issue.** Observed
2026-07-29 on MI250X/`gfx90a`. Affects `continuity_layered`, `kappa_shear`,
`epbl`, `ale_remap` — i.e. the AMD GPU column is empty, not partial.

**One root cause, two manifestations.** `-fdo-concurrent-to-openmp=device`
cannot build an OpenMP `map` clause for a `do concurrent` live-in whose type has
a derived-type component. Older flang says so; newer flang crashes.

*flang 22.0.0git (AMD AFAR drop #7.0) — clean NYI diagnostic:*

```
error: loc("continuity_layered.F90":111:44): DoConcurrentConversion.cpp:603:
       not yet implemented: Nested record types are not supported yet.
LLVM ERROR: aborting
```

*flang 23.0.0git (ROCm 7.13.0) — hard compiler crash:*

```
terminate called after throwing an instance of 'std::bad_function_call'
#12 DoConcurrentConversion::genMapInfoOpForLiveIn(fir::FirOpBuilder&, mlir::Value)
#13 DoConcurrentConversion::matchAndRewrite(fir::DoConcurrentOp, ...)
flang-23: error: unable to execute command: Aborted (core dumped)
```

The frame that matters is `genMapInfoOpForLiveIn` — the pass is generating the
map-info op for a **live-in** of the loop and cannot do it for that type.

**The trigger is the live-in's TYPE, not the reference syntax.** This is the
non-obvious part and it wrong-foots the obvious diagnosis:

- The pass maps the *whole container* it captures, regardless of which fields
  the body touches. So `kappa_shear` and `ale_remap` fail with **zero** `%a%b`
  references anywhere in the file — grepping for nested accesses finds nothing
  and tells you nothing.
- The reported location is the **declaration**. `111:44` is the `:: this` in
  `type(continuity_t), intent(inout) :: this`. Don't hunt that line for a
  malformed loop; look at the *type definition* instead.

Two nesting shapes are in play, and they are not equally hard:

| shape | example | kernels |
|---|---|---|
| scalar derived component | `type(scratch_3d_buffer_t) :: h_face_left_x` (`continuity_layered.F90:86-91`) | continuity_layered, epbl, kappa_shear |
| allocatable **array** of derived type | `type(tracer_slot_t), allocatable :: tracers(:)` (`ale_remap/remap_state.F90:57`) | ale_remap |

`ale_remap` is the harder case: `ms%tracers(t)%hTr(i,j,k)` is an array-of-derived-type
pointer chase, which is exactly the production layout the kernel was extracted
to measure.

**`=host` is unaffected** — only the device pass has the limitation, so
`make dc DATA=none FC=amdflang` builds and runs. The AMD *CPU* lanes of the
matrix are fine.

This is a compiler limitation, not a portability finding about `do concurrent`:
the construct is standard F2018, and nvfortran offloads it, gfortran compiles
it, and amdflang's own host pass handles it.

**Fix path — do NOT flatten the types.** The nesting IS the shipped data
layout; flattening would benchmark a kernel that is not the one that ships,
which defeats the extraction. Two candidate workarounds keep the layout and
change only how the loop body *names* the arrays, so the live-in becomes a
plain array instead of a record:

- **`associate`-hoist** the component outside the loop, body uses the
  associate-name. Zero structural change.
- **Explicit-shape array dummies** — put the loop in a subroutine taking the
  arrays as dummies. `ale_remap` already uses this idiom for
  `ocean_remap_tracer_pair`.

`tools/bugs/` holds the reproducer + a driver that decides between them:

```bash
cd tools/bugs && ./probe_amdflang_dc.sh          # device pass, gfx90a
```

It compiles 5 semantically-identical variants — the two failing shapes, both
candidate workarounds, and a no-nesting control — and prints a PASS/FAIL table.
If V=3 or V=4 passes, that idiom is the fix to apply to the four kernels
(gated, as always, on the bit-identity check plus a perf-neutrality re-run on
NVIDIA, since the change touches measured source). All 5 variants are verified
equivalent under gfortran, so a PASS also proves correctness.

**For the upstream report** (https://github.com/llvm/llvm-project/issues) —
two standalone MREs, one per crash signature, ~30 lines each, no preprocessor,
each with the compile line / expected / actual in its header:

| file | triggers |
|---|---|
| `tools/bugs/mre_dc_device_nested_scalar_record.f90` | flang 22 `not yet implemented: Nested record types` |
| `tools/bugs/mre_dc_device_alloc_array_of_record.f90` | flang 23 crash, `std::bad_function_call` in `genMapInfoOpForLiveIn` |

Both verified to build and print `36.000000000000000` under
`gfortran -std=f2018 -Wall`. The second is the stronger report (a crash, not a
missing feature) and its header documents the load-bearing detail: the
derived-type component triggers the failure *even when the loop body never
dereferences it*, so the type must not be "simplified" away when triaging.

## kappa_shear: the depth sweep, and a correction to the DC-vs-CUDA ratio

`tools/ks_sweep.sh` (+ `tools/ks_report.py`) sweeps this kernel over depth,
width, arrangement and the two compile-time policies below. Three results.

### 1. The published ratio was not a 1:1 comparison

`ks_solve_column` has three whole-array assignments (`kappa_out = kappa`,
`kq_tmp = k_q` x2). Fortran copies the DECLARED extent — O(NZ_STACK_MAX) — and
`ks.F90:949` notes this is the ONLY reason wall time depends on NZ_STACK_MAX at
all. **Both CUDA kernels bounded that copy to `1..nz+1`, O(nz)**
(`ks_kernel.cu:798`). So part of every published gap was a copy one side simply
did not perform. `opt_kernel.cu`'s header says this was "measured separately by
the `dcfix` variant (ks_fix.F90)" — that file no longer exists.

`-DKS_FULL_COPY` now lets the faithful CUDA kernel reproduce the Fortran copy,
and the two sides are always built as a MATCHED pair. V100, 473x297x30:

| pairing | build | dc / cuda_faithful |
|---|---|---|
| mismatched (as previously measured) | `BCOPY=0`, `FULLCOPY=0` | 1.140x |
| matched `prod` (both O(NZSTACK)) | `BCOPY=0`, `FULLCOPY=1` | **1.106x** |
| matched `opt` (both O(nz)) | `BCOPY=1`, `FULLCOPY=0` | **1.112x** |

The asymmetry inflated CUDA's advantage by ~3 points — about a quarter of the
gap. `cuda_opt` always bounds the copy and so only ever belongs to `opt`.

### 2. The CUDA advantage grows with nz, and it is not the obvious causes

Matched pairs, `fit` stack policy, 473x297, min-of-3 (median-min spread <=0.13%):

| nz | it/col | dc ns/col | cuda ns/col | dc/cuda |
|---|---|---|---|---|
| 10 | 2.07 | 26.9 | 25.9 | 1.038x |
| 25 | 19.14 | 199.8 | 180.7 | 1.105x |
| 30 | 19.75 | 267.8 | 237.2 | 1.129x |
| 50 | 20.23 | 574.9 | 496.7 | 1.157x |
| 75 | 20.23 | 958.6 | 829.2 | 1.156x |
| 100 | 20.23 | 1364.4 | 1148.7 | 1.188x |

Ruled out by the instrumentation, not by argument:
- **not more iterations** — Picard iterations/column SATURATE at 20.23 from
  nz=50 on, while the ratio keeps climbing;
- **not the frame** — `fit` (NZSTACK=nz+1) and `prod` (128) converge at depth;
  NZ_STACK_MAX only bites at the shallow end, where a fixed 129-double frame is
  amortised over little work;
- **not the copy** — every ratio above is a matched pair. (Under `fit` the
  declared-extent copy IS the bounded copy, which is why `cuda_faithful` and
  `cuda_opt` agree to 4 digits there — a free consistency check.)

What is left is the k-sequential recursion dominating the fixed costs as nz
grows. **Jorge measured ~1.67x on a GH200 at nz=100** — far larger than V100's
1.19x, so the gap is strongly architecture-dependent.

### 3. Register pressure — the likely mechanism, and how to test it

`cuobjdump -res-usage` on the column kernel (now captured automatically into
every GPU row's `regs` / `local_b`):

| impl | registers | frame B/thread |
|---|---|---|
| `do concurrent` (nvfortran 26.5) | **254** — the cap | 66,600 |
| `cuda_faithful` (nvcc 12.9) | 80 | 66,528 |

nvfortran pins this kernel at 254 registers in EVERY build: NZSTACK 11 -> 128
scales the frame 6.7 KB -> 66.6 KB and the register count never moves. Same
arithmetic, ~3x the register file. On a V100 it costs little, because
`kappa_shear/OPTIMIZATION.md` established the kernel is latency/divergence-bound
at ~6% achieved occupancy — registers are not the binding constraint. On an
architecture that hides that latency well GIVEN warps to schedule, it should
cost much more. That is a falsifiable prediction: run the sweep on the GH200 and
read the `regs` column.

**`-gpu=tripcount:host` is NOT the cause on V100.** The `tripcount` experiment
A/Bs it; on kappa_shear the effect is <=0.3% at every depth (OFF/ON =
0.9997-1.0033 over nz=10..100), so this file's ~2x warning applies to a
different kernel or configuration. Still run it first on any new GPU or compiler
version: a 2x-wrong DC side is indistinguishable from a large
architecture-dependent CUDA win.

**GPU arch is now DETECTED, not assumed.** `config.mk` hardcoded `cc70`/`sm_70`,
so any run on another card silently compiled for Volta and executed via JIT —
which invalidates exactly the codegen ratio being measured. It now comes from
`nvidia-smi --query-gpu=compute_cap`, is printed in the sweep banner, passed to
every `make`, and recorded per row.

### 3b. NZ_STACK_MAX is catastrophic on MI250X and irrelevant on NVIDIA

**This overturns the "the frame does not matter" conclusion recorded above and
in kappa_shear/OPTIMIZATION.md.** Both were measured on NVIDIA. They do not
transfer.

**The AMD answers are BIT-IDENTICAL to NVIDIA** at all six depths and both
policies -- one `do concurrent` source, OpenACC on NVIDIA and OpenMP-target
device offload on AMD (amdflang 23.0.0git, gfx90a), same kd_sum to the last
digit, same iteration counts. The portability claim holds on the numbers, not
just on "it compiled".

Same source, same binary but for one compile-time constant, and `it_inner` is
IDENTICAL between the two policies at every nz -- so the work is provably the
same and the whole difference is the per-thread frame. `prod` / `fit` cost ratio:

| nz | MI250X (1 GCD, amdflang) | GH200 (nvfortran) | V100 (nvfortran) |
|---|---|---|---|
| 10 | **14.8x** | 1.08x | 1.13x |
| 25 | **2.9x** | 1.03x | 1.03x |
| 30 | **3.0x** | 1.01x | 1.02x |
| 50 | 1.1x | 0.99x | 0.99x |
| 100 | 1.0x | 0.98x | 0.99x |

Normalised to V100 = 1.00 at nz=30: GH200 3.10x (fit) / 3.12x (prod), MI250X
**1.19x (fit) / 0.40x (prod)**. At the setting production actually compiles,
one MI250X GCD is 2.5x SLOWER than a 2017 V100 on this kernel; with the frame
fitted to the column it is 1.19x faster. The constant is worth 3x on that
hardware and ~0% on NVIDIA.

**MOM6 consequence.** Production compiles `NZ_STACK_MAX=128` and runs nz=30-75.
On AMD that is a ~3x penalty at nz=30 for free. Per-GCD is the correct unit
here, not a caveat to scale up: MOM6 runs one rank per GCD (Frontier has 4x
MI250X = 8 GCDs/node) and one rank per GPU on GH200, so these ARE the per-rank
figures.

**The instability is itself a result, and it makes the 3x a LOWER BOUND.**
median-min spread over NRUN=3: `fit` 0.0-0.2% at every depth; `prod` 17.9%,
66.4%, 17.7% at nz = 10, 25, 30, settling to 0.3-4.4% by nz=50. So `ms_min`
flatters the unstable series. On medians the penalty is 4.9x at nz=25 and 3.6x
at nz=30, versus 2.9x / 3.0x on minima. A 66% run-to-run swing is arguably worse
than being slow for an operational model -- you cannot schedule against it --
and the `fit` builds on the same GPU are flat to 0.1%.

⚠ **The MI250X `prod` series is NON-MONOTONIC and must not be quoted point by
point**: 723, 495, 681, 504, 806, 1170 ns/col over nz = 10..100. nz=10 costs
more than nz=25 despite an identical frame and 9x FEWER Picard iterations
(300k vs 2.78M), which is not physically possible. The `fit` series on the same
machine is perfectly monotonic (49, 169, 225, 459, 849, 1144), so the harness is
sound and the instability is specific to the large-frame builds -- consistent
with occupancy collapsing to very few waves at 66 KB/thread of scratch. The
ms_med-vs-ms_min check above confirms it is variance, not physics: the ordering
anomaly is an unlucky draw, not a discontinuity. The QUALITATIVE finding
does not depend on it: `fit` at nz=30 is 225 ns/col and every `prod`
measurement is 495-806.

⚠ **Cross-machine absolute numbers are four confounds deep** (compiler, offload
model, architecture, and 1 GCD vs a whole GPU). The prod-vs-fit RATIO is
within-machine and carries none of them; that is what the finding rests on.

### 4. CPU: thread scaling, and the arrangement question

**Serial `do concurrent` costs NOTHING versus plain nested `do` loops** — on
four independent compilers, which is the cleanest form of the portability claim.
`dc_serial / serial_do`, 64x64 = 4900 columns, `fit` stack:

| nz | flang 22.1.5 | gfortran 16.1 | ifx 2026.0 | nvfortran 26.5 |
|---|---|---|---|---|
| 10 | **1.170** | 1.010 | 1.005 | 1.000 |
| 25 | 1.037 | 1.003 | 1.001 | 0.999 |
| 30 | 1.018 | 1.011 | 1.001 | 0.997 |
| 50 | 1.005 | 0.986 | 0.996 | 1.001 |
| 75 | 1.008 | 0.986 | 0.996 | 1.001 |
| 100 | 1.004 | 0.984 | 0.997 | 1.000 |

Three of the four are flat within noise at every depth. **flang is the one
exception and only at the shallow end** — 17% at nz=10, 3.7% at nz=25, gone by
nz=50. At nz=10 a column runs just 2.37 Picard iterations, so a fixed per-loop
cost in flang's `do concurrent` lowering is visible; once there is real work per
column it disappears. Quote the DC-is-free claim with that caveat, not without.

**Absolute standing, same source** (`serial_do` ns/col) — flang is the fastest
serial compiler here and nvfortran the slowest by ~11%:

| nz | flang | gfortran 16.1 | ifx | nvfortran |
|---|---|---|---|---|
| 30 | **14,489** | 14,599 | 14,812 | 15,682 |
| 100 | **70,668** | 73,434 | 72,906 | 79,384 |

**Thread scaling** (`dc_multicore`, nz=30, 20 physical cores + SMT). Three
different mechanisms for the same source line — nvfortran `-stdpar=multicore`
(driven by `ACC_NUM_CORES`), ifx `-qopenmp` and flang
`-fdo-concurrent-to-openmp=host` (both `OMP_NUM_THREADS`):

| threads | 2 | 4 | 8 | 16 | 32 | 40 | ms @40 |
|---|---|---|---|---|---|---|---|
| ifx 2026.0 | 1.84x | 3.45x | 6.57x | 12.63x | 19.49x | **24.24x** | **3.010** |
| nvfortran 26.5 | 1.84x | 3.37x | 6.43x | 11.35x | 14.46x | 21.22x | 3.680 |
| flang 22.1.5 | 1.85x | 3.39x | 6.45x | 11.32x | 13.95x | 20.63x | 3.484 |

So flang wins serial, ifx wins threaded. gfortran has no multicore lane:
`config.mk` gives it no host-parallel `do concurrent` flag.

⚠ flang's `-fdo-concurrent-to-openmp=host` warns "Mapping `do concurrent` to
OpenMP is still experimental" but works, and handles this kernel's nested
derived types fine. Only the `=device` pass fails on them (see the amdflang
section above) — so "flang cannot do this kernel" is too strong; it is "flang
cannot OFFLOAD this kernel".

**Column blocking (`VARIANT=block`, `ks_block.F90`) never beats the shipped
per-column arrangement on AVX2**, but the vectorisation works. Best-vs-faithful,
Broadwell, same-run baseline: serial lanes reach 0.98x at nz=75 / VLEN=4,
multicore only 0.65x. Three separable facts: the transformation alone costs
~1.7x (VLEN=1 = 0.56-0.66, from masked full-range loops replacing per-column
early exits); vectorisation peaks at VLEN=4, exactly AVX2's four doubles, and
regresses at 8/16; and depth helps (0.78 -> 0.98 from nz=30 to 75) while threads
hurt (blocked frame is VL times larger, so 40 threads thrash cache). Since
VLEN=8 already regresses here, the binding constraint is masking overhead and
footprint rather than vector width — which predicts AVX-512 will help the
single-threaded lane and not the fully-threaded one.

## The four-machine picture (2026-08-03/04)

One `do concurrent` source. Four compilers, three GPU vendors, five devices.

### Correctness first

**Bit-identical everywhere.** 13 distinct problem sizes, every lane, same
`kd_sum` AND same iteration counts: nvfortran (OpenACC and OpenMP-target on
V100/GH200), amdflang (OpenMP-target device, MI250X gfx90a), ifx
(OpenMP-target spir64, Intel Max), plus serial-do and multicore under
nvfortran / ifx / gfortran / flang, plus both hand-written CUDA kernels. For an
ITERATIVE solver with convergence tests that is not a given -- a 1-ulp
difference can flip a test and diverge. It did not, anywhere.

### The result that matters: is the GPU worth it?

ns/column, nz=30, `fit`, 473x297. "host CPU" is the SAME node's CPU with the
SAME compiler, so only the device changes:

| device | GPU | host CPU | GPU/CPU |
|---|---|---|---|
| NVIDIA V100 | 268.5 | (not measured) | - |
| NVIDIA GH200 | 86.5 | (not measured) | - |
| AMD MI250X (1 GCD) | 225.2 | 269.8 (EPYC 7A53, 56t) | **1.20x** |
| Intel Max (1 tile) | 323.7 | 227.7 (Xeon Max, 104t) | **0.70x** |

Over the full depth range: MI250X 1.34x at nz=10 falling to **0.96-0.98x by
nz=75-100 -- i.e. the GPU LOSES to its own host**; Intel Max is 0.60-0.70x
throughout, never winning at any depth. Per-GCD / per-tile is the correct unit
(one MPI rank each), so these are per-rank figures.

⚠ The NVIDIA rows have no host-CPU comparison yet -- that cell needs a
`MODE=dc_multicore` run on the GH200's Grace CPU. Without it we cannot say
whether NVIDIA's offload advantage is large or merely positive.

### Why -- and it is NOT `do concurrent`

`dc_serial / serial_do` on the CPU, i.e. what the abstraction costs with no
offload involved:

| compiler | machine | ratio over nz=10..100 |
|---|---|---|
| nvfortran 26.5 | Broadwell | 0.997 - 1.001 |
| ifx 2026.0 | Broadwell | 0.996 - 1.005 |
| ifx 2025.3.2 | Xeon Max | 0.998 - 1.001 |
| gfortran 16.1 | Broadwell | 0.984 - 1.011 |
| flang 22.1.5 | Broadwell | 1.004 - **1.170** |
| amdflang 23.0.0git | EPYC 7A53 | 1.002 - **1.125** |

`do concurrent` is FREE against plain nested loops on nvfortran, ifx and
gfortran. The flang family carries a fixed per-loop cost visible only at nz=10,
reproduced independently on two LLVM versions, two vendors' packaging and two
ISAs -- and `it/col` explains why it only shows there: 2.37 Picard iterations
per column at nz=10 versus ~20 elsewhere, so a fixed cost has nothing to hide
behind.

So the weak link is the OFFLOAD LOWERING, not the language feature. On Intel
that is stark: the same compiler whose CPU DC lowering is the tightest of all
six (0.998-1.001) produces device code that loses to its own CPU by 1.4x.
`LIBOMPTARGET_INFO` shows the kernel entering with 87 arguments and multi-MB
`tofrom` maps despite the driver having pre-mapped everything with
`DC_ENTER_IN`, which points at the DC->target pass re-establishing the data
environment per launch.

### Thread scaling (dc_multicore, nz=30, fit)

| CPU | compiler | threads | speedup |
|---|---|---|---|
| Xeon Max | ifx 2025.3.2 | 104 | 44.75x |
| EPYC 7A53 | amdflang 23.0.0git | 56 | 37.83x |
| Broadwell | ifx 2026.0 | 40 | 24.24x |
| Broadwell | nvfortran 26.5 | 40 | 21.22x |

### What is still missing

- GH200 `CUDA=on` (the DC-vs-CUDA ratio on Hopper, and the register split)
- GH200 Grace-CPU `dc_multicore` (the NVIDIA GPU/CPU cell above)
- a hand-written `!$omp target teams distribute` control, which is the only
  thing that separates "the DC->target lowering is weak" from "this vendor's
  offload stack is slow" -- the vendor-neutral analogue of the CUDA comparison

## What the domain sweep does and does not vary (write this into the paper)

The production configuration is **473x297 at 0.05 degrees**. The grid sweep in
`fig8` runs the same kernel from 32x32 up to 2048x2048, and it is important to be
precise about what that changes, because it is easy to describe wrongly.

**kappa_shear is horizontally local.** No `dx`, `dy`, cell area or lat/lon
spacing appears anywhere in the column solve. The only `i+-1` reference in the
entire kernel is `u_face_x_layer(i+1,j,kg)` -- the C-grid face-to-centre average
during the gather, not a stencil inside the solve. Every column is solved
independently from its own vertical profile, so the 0.05 degrees enters MOM6's
physics through the HORIZONTAL operators (advection, viscosity, pressure
gradient) and does not reach this kernel at all. What the kernel sees is `nz`,
the layer thicknesses, and each column's own T/S/u/v.

**This is confirmed by measurement, not just by reading the source.**
`it_inner/column = 20.23` at every grid from 1,444 to 4,218,916 columns -- a
2,900x range with identical work per column. If horizontal resolution touched
the solve, that number would move.

**So the sweep varies device occupancy, not resolution.** Larger grids change
only HOW MANY independent columns the device is given. `fig8`'s x axis should be
described as *columns solved* or *device occupancy*, never as resolution or
domain extent.

**The wording trap.** `build_state` maps a FIXED lat/lon extent (-60.6 to -30.9
degrees) onto whatever `nxp x nyp` it is given, so 2048x2048 is *the same ocean
sampled more finely* -- an effective ~0.012 degrees -- not a larger ocean at
0.05. That does not affect any measurement here, because the column population
stays statistically identical (which is exactly why `it/col` is constant), but a
reader who takes fig8 as "0.05 degrees scaled up to 2048^2" has misread it. If a
genuinely larger 0.05-degree domain were ever wanted, `build_state` would need to
extend the lat range with `nyp` rather than rescale it -- a one-line change,
needed for nothing in this study.

**Corollary that matters for the headline.** Because per-column work is
grid-invariant, the DC-vs-CUDA ratio can be quoted without a grid caveat: on
GH200 at nz=75 it is 1.7099 at 145,137 columns and 1.7038 at 4,218,916 -- 0.4%
apart across 29x the domain, with both implementations shedding ~12% of
per-column cost over that range. A launch-overhead explanation for the gap would
have had the ratio decay with size. It does not, so the gap is codegen.

## Methodology / traps

- **Correctness = bit-identity**, bar `max rel diff < 1e-12` (FMA-contraction
  level). Verify the verifier: perturb a term by 1 ulp and confirm it trips.
- **Cross-language libm trap:** don't recompute `sin/exp` inputs in the C++
  driver — nvfortran's libm and glibc's differ ~1 ulp, amplified by cancellation.
  Hand the transcendental inputs over via the ref dump and *adopt* them, or the
  cross-check measures libm, not the kernel. Use field-relative error for fields
  with near-zero cells.
- **Shared V100:** report min-of-reps with warmup; for A/B, idle-gate and
  alternate order, or you measure contention.
- **The whole-model number needs an EXCLUSIVE GPU.** The end-to-end effect is
  ~8% — the *same size as contention noise* from a co-running job (e.g. MOM6).
  A single contended run flipped the RK2 sign (CUDA +8% → DC +8%) because the
  second-timed (CUDA) pass caught the other job. `tmp_local_artifacts/gpu_run.sh`
  idle-gates (<12% util) + serializes with `flock`, but that is NOT enough
  against a *running* job — it can fire in a between-timestep lull. Confirm
  `nvidia-smi --query-compute-apps` is empty, then run 4–5× and check the ratio
  is stable to <0.2% (a clean run is: per-kernel isolated sum ≈ the aggregate).
- **Trust the single-binary harness over ratio-of-ratios.** Comparing an
  nvfortran binary's speedup to a separate nvcc binary's speedup is not a
  language comparison — allocator, warmup, and clock differ. Put both on one
  device image (`make cmp`) before claiming a winner.
- **Measure at production sizes.** The dc/cuda gap is a strong function of total
  cell count (launch amortization) — 4096² makes everything look clean while
  nothing runs there.

## The three measurement layers (what to run)

1. **`make dc_ab`** (per kernel) — faithful DC vs optimized DC in one binary;
   proves the portable optimization is bit-identical and times the speedup. Runs
   on GPU (`DATA=acc`) *and* CPU (`DATA=none`) — same source, so it doubles as
   the CPU-portability check.
2. **`make cmp`** (per kernel) — optimized DC vs optimized CUDA on the SAME
   device arrays via `host_data use_device`. The apples-to-apples per-kernel
   verdict (table above). Mixed nvfortran+nvcc link; `-cuda` is link-only (the
   bridge module must NOT see it → NVFORTRAN-S-0528).
3. **`ideal_benchmark/`** — a "dumb" RK2 (2-stage) driver that runs all 8 ocean
   kernels back-to-back per step on a hot GPU, all-DC then all-CUDA, and reports
   ms/stage + the per-kernel breakdown. This is the model-level answer. Isolation
   trick: each kernel (they redefine `constants`/`grid`/… with different bodies)
   is compiled into its own `-Bsymbolic` `.so` so duplicate module symbols never
   cross-bind; only the `bind(C)` `rk2_<k>_*` entries are global. `hll_fluxes` is
   excluded (coastal SWE, not ocean core); `epbl` stays.
