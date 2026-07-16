# Redi neutral diffusion — `do concurrent` vs hand-written CUDA C

## Result: CUDA buys nothing. Portable Fortran buys 1.35x. Redi is ~45 ms/call.

473x297x30 (the 0.1 deg gabight config), V100, nvfortran 26.5-0, production's
flags, `NZ_STACK_MAX=128`:

```
  DC = PRODUCTION TODAY (verbatim extract)     44.72 ms/call
  CUDA faithful port  (5 kernels)              44.98 ms/call
  DC + PRECOMP (hoist the column rebuild)      33.22 ms/call   <- all Fortran

  dc / cuda      = 0.994 x    Fortran already ties hand-written CUDA
  dc / precomp   = 1.346 x    FREE, in Fortran, bit-identical
  precomp / cuda = 0.738 x    portable Fortran BEATS the CUDA port by 1.35x
```

### ⚠ CORRECTION 2026-07-16: the old "0.0 EXACTLY" claim was an artifact

This README used to claim *"DC vs CUDA is `max|d hTr| = 0.0` EXACTLY, not
1e-15"* and reason from it that "every expression contracted identically".
**That was a harness bug, not a result.** `redi_bench.F90` called `check()`
*before* `state_exit()` — i.e. before the `copyout`. Under `-gpu=mem:separate`
the host `hTr` arrays stay stale until then, so all three variants still held
their identical `build_state` initial condition and **every check compared the
input to itself and reported 0.0 unconditionally.** It could not have failed.

Fixed (the `copyout` now precedes the checks). The true figures, 473x297x30
after 30 accumulating steps:

```
  DC vs CUDA    : tracer 1  max|d| = 4.55e-13  (rel 1.5e-16)   <- NOT 0.0
                  tracer 2  max|d| = 0.0
  DC vs PRECOMP : both tracers exactly 0.0                     <- this one SURVIVES
```

**The conclusions do not change.** 1.5e-16 relative is FMA-contraction noise,
three to four orders inside the harness's own `>1e-12 = PORT BUG` bar, so the
CUDA port is still verified faithful — just not bit-exact, and the "exact
agreement proves identical contraction" flourish is withdrawn. DC vs PRECOMP
*is* still exactly 0.0 post-copyout, so the source-transform claim stands on
real evidence.

**Anyone reusing this harness should check the same thing**: a variant
comparison that runs before the copyout is vacuous, and it fails silently and
in the reassuring direction.

**Redi is the most expensive kernel measured in this investigation by an order
of magnitude**: 44.7 ms/call against `ocean_barotropic_solver`'s 3.7 and
`ocean_continuity`'s 1.07. The user's "when I enabled redi and meke oh god did
they take time" is corroborated.

### Across production configs (min-of-3, `python3 sweep.py`)

```
 config                 cells     dc_ms  cuda_ms   pre_ms  dc/cuda  dc/pre
 -----------------------------------------------------------------------
 10km     (108x137)     489060      6.04     5.36     4.14    1.127   1.458
 0.1deg   (473x297)    4354110     44.60    45.04    33.02    0.990   1.351   <- Redi/MEKE config
 0.25deg  (240x560)    4177080     42.74    42.69    30.10    1.001   1.420
 3km      (359x458)    5080800     52.03    54.53    37.65    0.954   1.382
 0.05deg  (945x594)   17118000   EXCLUDED -- did not reproduce; see below
```

**`dc/cuda` is 0.95-1.13 across every reproducible config — a tie.** Unlike the
continuity kernels, there is no cell-count trend, because Redi has no launch
overhead to amortise (it is 100% GPU-busy at every size). **`dc/precomp` is a
flat 1.35-1.46x**, i.e. the portable win is available everywhere.

**⚠ The 0.05 deg config did not produce reproducible numbers and is excluded.**
Same binary, same args, differing only in which variant is timed first:

```
  order=0 (DC,CUDA,PRE): DC=161.8  CUDA=345.6  PRE=265.3
  order=1 (PRE,CUDA,DC): DC=404.5  CUDA=593.2  PRE=276.4
```

DC reads 161.8 or 404.5 — **2.5x by position in the run alone** — and CUDA is
slower than DC in both orders, though the two are bit-identical and agree within
5% at every smaller size. I could not explain it. **Ruled out by measurement**:
GPU memory (9.0 GB of 32 GB, computed from the shapes), CPU contention (20
cores, load 3.2), another process on the GPU (`nvidia-smi`: only `redi_bench`).
Sustained-load thermal drift is the best remaining guess but the ordering is not
monotonic, so it does not fit either. **Do not quote that row.** It is recorded
in `sweep_results.txt` so nobody re-derives it.

---

## Is the CUDA port handicapped by OpenACC? No — the `host_data` path costs 2.5 µs/call

Every CUDA port in this repo is launched *from Fortran* and gets its device
pointers via `!$acc host_data use_device(...)`, a runtime present-table lookup
per array per call — 24 arrays, every Redi step. If that cost anything, every
"CUDA buys N%" number here would be understated. `redi_native.cu` settles it: a
**native C++/CUDA driver, `cudaMalloc` + `main()`, no Fortran, no OpenACC**,
calling the SAME `redi_cuda_launch_()` from `redi_kernel.cu`.

```
  host_data use_device present-table lookup, 24 arrays  :  0.00246 ms/call
  CUDA variant, whole call (idle, from the table above) : 45.04    ms/call
  -> the OpenACC path is 0.0055% of a Redi call
```

**Measured, and contention-proof** — the lookup is *host-side* work, so unlike a
GPU timing it stays valid on a busy node. Two checks that it is real: it is
stable to ±2% over 5 runs, and it is **flat at 2.46-2.51 µs across 108x137,
473x297 and 945x594** — a 76x cell-count range. A per-array table lookup should
not scale with the grid, and it does not; ~104 ns/array. (`redi_probe.cu` runs
the identical `host_data` region against a no-op instead of the kernels, so the
difference is the lookup alone.)

**This caps the argument.** Whatever a native-vs-Fortran GPU A/B shows, it
cannot be the `host_data` path costing more than 0.006% of the call. So:

1. **`dc/cuda ≈ 1.0` is a fair comparison.** CUDA was not handicapped. The
   headline (`do concurrent` ties a hand-written port; PRECOMP beats it 1.35x)
   is unchanged.
2. **The generalisation is an inference, not a result.** 2.5 µs is a *fixed
   host cost per call*, so it is negligible precisely because a Redi call is
   ~45 ms. For a kernel whose per-call time is microseconds — MEKE is
   launch-bound — the same 2.5 µs could be a real share. **Not measured here.**
   Do not carry "host_data is free" to the launch-bound benchmarks without
   re-measuring it there.

### Verification: bit-identical, and it is a real test

`make native-verify` runs `./redi_bench ... dump=1` (which writes its own
initial and final CUDA state) then `./redi_native`, which diffs both. Result at
473x297x30 after 30 accumulating steps:

```
  hTr(T) : max|d| = 0.0    hTr(S) : max|d| = 0.0    -> BIT-IDENTICAL
```

The native driver reproduces the Fortran-driven CUDA run exactly. Same kernel
object, same IC bits, same rep count, so bit-identity is the only legitimate
outcome and any difference would be a bug in `redi_native.cu` — there is no
second toolchain left to contract an expression differently.

**One subtlety, and it killed two hypotheses.** The natively-computed IC agrees
with `redi_bench`'s to 1-2 ulp (~3e-16), not exactly. Tested, not assumed:
*not* `exp()` (nvfortran's host libm and glibc agree bit-for-bit on every value
in this init — probed directly), and *not* FMA contraction (`gcc
-ffp-contract=off` gives the identical answer). It is nvfortran `-fast` vs gcc
`-O3` differing by one ulp on `2 + 16*e + 3*fx + 3*fy`, in the *harness's own
setup arithmetic* — nothing to do with the kernel or the port. So when the dump
is present `redi_native` **adopts** that IC, which is what makes the final test
bit-exact rather than "agrees to 1e-16"; standalone runs (no dump) compute
their own IC and need no Fortran at all, which is what `nvbug_*/` reports want.

### The GPU A/B: a tie, 1.000x — but read the contention note

```
  CUDA via Fortran driver (host_data)  : 45.043 ms/call   <- the existing number
  CUDA native (cudaMalloc, no OpenACC) : 45.006 ms/call
  ratio                                : 1.001 x          <- a TIE
```

Both are mean-of-20, best-of-6 interleaved A/B rounds (the min is the right
estimator: interference only ever makes a run slower). Native min-of-20 was
44.73. A second, independent low-contention run gave 45.041 vs 45.024 =
**1.0004x**.

**Why this is quotable despite a shared GPU.** Up to 9 sibling jobs used this
V100 and most rounds are junk — `redi_bench`'s CUDA variant read 45.0-112.4
ms/call across the session. But the rounds above are anchored: the
Fortran-driven CUDA variant **reproduced its idle 45.04 to three digits, and
PRECOMP its idle 33.02**, which is the signature of a round that ran clean. The
noise is also diagnosable rather than mysterious — when *both* drivers were hit
equally the ratio stayed ~1.00 (53.8/53.4; 93.5/93.6), and only rounds where
one driver alone was hit produced outliers (63.0/45.6 = 1.38x). That is
contention, not a property of the drivers.

**Still: these are PROVISIONAL and a serial re-run on an idle GPU is the honest
confirmation.** They are also merely consistent with the load-bearing evidence
rather than a substitute for it — the 2.5 µs probe above independently caps any
possible `host_data` effect at 0.006% of a call, and *that* number needs no
contention caveat. A GPU A/B could not have shown a real effect this small even
on an idle node; the two lines of evidence agree, and the probe is the one that
settles it.

---

## The regime: GPU-bound on THREAD-LOCAL memory traffic. Not launch-bound.

Measured, not inferred (`nsys`, 0.1 deg, per call):

| kernel | ms/call | % |
|---|---|---|
| `redi_apply_flux_impl` (x2 tracers) | 28.44 | 61.4% |
| `redi_calc_coeffs_x` | 8.72 | 19.1% |
| `redi_calc_coeffs_y` | 8.61 | 19.1% |
| `redi_snapshot` (x2) | 0.18 | 0.4% |
| `redi_face_copy` (x2) | 0.01 | 0.0% |
| **GPU total** | **45.96** | |
| **wall clock** | **44.97** | |

**GPU-busy is ~100% of wall time.** There are 8 launches per call against
~46 ms of kernel; launch overhead is ~0.2% of runtime. Redi is at the opposite
end of the spectrum from MEKE.

`ncu` on the dominant kernel (`redi_apply_flux_impl`):

| | production | +PRECOMP |
|---|---|---|
| registers/thread | 138 | 194 |
| theoretical occupancy | 18.75% | 12.50% |
| **achieved occupancy** | **10.82%** | **10.94%** |
| **DRAM throughput** | **51.24%** (460 GB/s) | **32.62%** |
| **L1/TEX hit rate** | **43.46%** | **78.56%** |
| Compute (SM) throughput | 11.41% | 12.07% |
| Executed IPC | 0.33 | 0.51 |
| **local-mem ops (SASS LDL+STL)** | **2081** | **125** |

Compute throughput is 11%. This kernel is **not** FP64-limited. It is
**bandwidth-limited on its own thread-local stack**: 460 GB/s of DRAM traffic
that is mostly spilled column arrays, missing L1 more than half the time.

## The cause: 8 redundant PPM column rebuilds per cell

`redi_apply_flux_impl` is a cell-centric double-visit — cell (i,j) recomputes
the flux on each of its 4 bounding faces, and `redi_face_flux` rebuilds **both**
adjacent tracer columns from scratch (`redi_tracer_column` x2). That is **8 full
PPM column reconstructions per cell** (plm_diff + (nz-1) ppm_edge + a limiter
sweep, each) when only 5 distinct columns exist. Every column in the domain is
rebuilt ~8 times over.

Those rebuilds need somewhere to live: `redi_face_flux` holds **12
`NZ_STACK_MAX`-sized locals** (hcL/trcL/hcR/trcR + TlL/TiL/aLL/aRL +
TlR/TiR/aLR/aRR). At NZ_STACK_MAX=128 that is ~12 KB/thread of stack. That
stack is the 460 GB/s.

**This is algorithmic. It is in the Fortran and in the CUDA port equally, and
no compiler can remove it** — the rebuild sits behind an `!$acc routine seq`
call and the redundancy is *across threads*.

## The fix: hoist the rebuild (`ocean_redi_pre.F90`) — 1.35x, bit-identical

```
pass 1 (new kernel)  every cell builds ITS OWN column ONCE -> Tlay/Tint/aLe/aRe
                     in device scratch                          1.21 ms x2
pass 2 (flux)        reads them by scalar index; no rebuild,
                     and no per-thread column arrays at all     6.64 ms x2
                     (was 14.22 ms x2)
```

8 reconstructions per cell -> 1. Same arithmetic, same order, so **exactly
bit-identical**. Phase B: 28.44 -> 15.72 ms (**1.81x**); the flux kernel alone
goes **2.14x**.

**The mechanism is measured, and it is not the obvious one.** Occupancy is
*unchanged* (10.82% -> 10.94%) and registers went *up* (138 -> 194). The win is
entirely traffic and locality: local-memory ops drop **17x** (2081 -> 125), DRAM
throughput drops 51% -> 33%, and L1 hit rate jumps 43% -> 79% — because the
precomputed columns are *shared* between the 8 consumers and cache, whereas
per-thread stack spills are private and do not.

Cost: 4 extra device arrays ~(nx,ny,nz), **~139 MB** at 0.1 deg, reused across
tracers. Phase B stops being one fused pass.

**Dependency checked, not assumed**: pass 1 -> pass 2 is a real barrier (pass 2
reads the columns of *neighbours* that pass 1 wrote), so the two cannot be
fused. Two kernels is that dependency structure, not a limitation.

## The two "known" fixes are both dead here — check before porting them

Both source-level fixes that paid on the barotropic substep are **inapplicable
to Redi**, and I verified rather than assumed:

1. **Signature fix: ALREADY APPLIED.** Redi is written in the outer-shim +
   flat-impl style — `redi_calc_coeffs_x(nx, ny, nz, ns, eos, ..., h_layer,
   t_htr, ...)` already takes plain explicit-shape dummies bounded by plain
   integers, and the caller does the `ms%tracers(it)%hTr` dereference on the
   host. There is nothing to fix. **Consequently `-Minfo=stdpar` reports
   `auto-collapsed` on all six loops** (collapse(2) on the five 2-D loops,
   collapse(3) on `redi_snapshot`): **nvbug_dc_collapse is ABSENT.**
2. **Fusion: worth ~0.** Redi is 100% GPU-busy with 8 launches per call.
   Fusing loops targets host overhead that Redi does not have.

## Hypotheses I tested and KILLED

Stated as hypotheses and tested, per the RESUME §1 rule. Three of my own died:

| hypothesis | verdict |
|---|---|
| "`NZ_STACK_MAX=128` is 4x oversized for nz=30; shrinking it will cut the stack traffic" | ❌ **FALSE.** `-DMODEL_NZ_STACK_MAX=64` (`make STACK=64`): **45.27 vs 44.97 ms — no change.** Local traffic scales with what is *touched* (nz=30 elements), not with the frame size. The frame is address space, not traffic. |
| "`!$acc routine seq` is what forces the helpers out-of-line" | ❌ **FALSE.** Compiling with `-stdpar=gpu` only (directives become comments) still emits all 21 helpers as separate device functions. |
| "`-Minline` will inline them and fix it" | ❌ **FALSE.** `-Minline`, `-Minline=levels:10`, `-Minline=reshape,levels:10` all still emit 21 device functions, and `-Minline` *raises* local traffic in the flux kernel (LDL 1000 -> 1376). Consistent with RESUME §1: `-Minline` made the HLL kernel worse too. |
| "Redi is launch-bound like MEKE" | ❌ **FALSE.** ~100% GPU-busy; 8 launches vs 46 ms of kernel. |
| "Redi is compute-bound like the coastal HLL kernel (FP64-limited)" | ❌ **FALSE.** Compute (SM) throughput 11.4%, IPC 0.33. It is bandwidth-bound on local memory. |
| "The OpenACC `host_data` present-table lookup handicaps the CUDA port, so `dc/cuda ≈ 1.0` understates CUDA" | ❌ **FALSE.** Measured at **2.5 µs/call for all 24 arrays = 0.0055%** of a 45 ms call (`redi_probe.cu`). The comparison was fair all along. |
| "The DC-vs-CUDA agreement is `0.0` because the two toolchains contract every expression identically" | ❌ **FALSE, and it was my predecessor's headline.** `check()` ran before the `copyout`, comparing two stale copies of the *input*. True value 1.5e-16 rel. See the correction at the top. |
| "The native driver's IC will differ from the Fortran one because nvfortran's host `exp()` and glibc's disagree" | ❌ **FALSE.** They agree bit-for-bit on every value in this init — probed directly. The 1-2 ulp gap is nvfortran `-fast` vs gcc `-O3` on `2 + 16*e + 3*fx + 3*fy`. |
| "...then it must be FMA contraction in the init" | ❌ **FALSE.** `gcc -ffp-contract=off` gives the identical answer. |

## An observation I could NOT explain

**nvfortran inlines none of Redi's 15 `!$acc routine seq` helpers.** Every one
is a separately-compiled device function with full ABI call overhead — down to
`redi_signum1`, a 3-line `sign()` wrapper (REG:24) called from the innermost
sublayer loop, and `redi_fv_diff` (REG:68) called from `redi_plm_diff`'s inner
loop. `make regs` shows all 21. No `-Minline` variant changes it.

This is *not* what RESUME §1 found for the HLL kernel, where `flux_cell` was
"inlined all along, in every variant". I do not know what distinguishes the two
cases, and I did not isolate it. **I also did not establish that it costs
anything** — PRECOMP wins by deleting traffic, not calls, and the ~274 call/ret
pairs in the flux kernel are a small share of 4096 instructions. Flagging it as
an unexplained observation, not a defect. It may be worth a probe series (the
`nvbug_inline_cse/` style) if someone wants it; it is not on the critical path
for Redi.

## Build / run

```bash
source ../../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
make && make run                       # 473 297 30 -- the 0.1 deg config
./redi_bench 945 594 30                # [nx_phys] [ny_phys] [nz]
./redi_bench 473 297 30 20 10          # [nreps] [nwarm]
./redi_bench 473 297 30 20 10 1        # [order] 0=DC first, 1=PRECOMP first
./redi_bench 473 297 30 20 10 0 1      # [dump] 1=write redi_ref_{init,final}.bin

make native && ./redi_native           # NATIVE C++/CUDA. No Fortran, no OpenACC.
                                       #   standalone: computes its own IC.
make native-verify                     # dump from redi_bench, then diff the
                                       #   native run against it -> bit-identical
make STACK=64                          # sweep NZ_STACK_MAX (no effect -- see above)
make collapse                          # per-loop -Minfo: all six auto-collapse
make regs                              # registers + LDL/STL + call/ret per function
make nsys                              # GPU-busy vs wall split
python3 sweep.py 3                     # min-of-3 across production configs
```

`nx_phys`/`ny_phys` are **interior** cells; `nx_total = nx_phys + 2*nghost`
(nghost=3). Redi is ghost-aware — it places the no-normal-flow WALL faces at the
*physical* edges (`nghost+1`, `nghost+nx_phys+1`), so that relationship must
hold or the walls land in the wrong place.

## Files

```
ocean_redi.F90       VERBATIM production extract of
                         <model>/src/parameterizations/lateral/structured/
                         ocean_redi.F90 (md5 c941f60101124f003f27453365930d2d).
                         EVERY deviation, reproducible with:
                           diff <(sed -n '/^   implicit none/,/^end module ocean_redi/p' $PROD) \
                                <(sed -n '/^   implicit none/,/^end module ocean_redi/p' ocean_redi.F90)
                         1. dropped `ocean_redi_bytes` + its `bytes` TBP (a host
                            memory accountant; would drag in mem_report)
                         2. added 2 `public ::` lines exporting redi_signum1 /
                            redi_ppm_ave / redi_snapshot / redi_face_copy so the
                            PRECOMP control reuses the IDENTICAL source instead
                            of duplicating it
                         ALL THREE ARE HOST-ONLY. No device code differs from
                         production -- not one line of any `do concurrent` body
                         or any `!$acc routine seq` helper.
ocean_redi_pre.F90   PRECOMP: the hoisted-column Phase B. The 1.35x.
ocean_eos.F90        VERBATIM extract of the two device EOS point routines
                         + all coefficients. ROQUET/WRIGHT kept though dead --
                         deleting a branch perturbs register allocation, and
                         this kernel is register-sensitive.
redi_kernel.cu           faithful CUDA C port. 5 kernels, one per DC loop.
redi_native.cu       NATIVE C++/CUDA driver: cudaMalloc + main(), no Fortran,
                         no OpenACC, no host_data. Calls the SAME
                         redi_cuda_launch_() from redi_kernel.cu and links the
                         SAME redi_kernel.o -- the kernels are NOT duplicated,
                         so the two drivers cannot silently diverge. Verified
                         bit-identical to the Fortran-driven CUDA run.
redi_probe.cu        no-op with redi_cuda_launch_'s signature. Lets
                         redi_cuda_probe time the host_data present-table
                         lookup with the kernels removed. The 2.5 us.
redi_bench.F90           driver. Separate output state per variant.
redi_cuda.F90        host_data use_device bridge (+ redi_cuda_probe, the
                         identical region against redi_probe.cu's no-op).
{constants,grid,ocean_metrics,multilayer_cgrid_state,
     ocean_boundary_types}.F90    MRE stubs -- exactly the fields Redi reads.
sweep.py                 min-of-3 production sweep.
```

## Caveats — read these before quoting a number

- **✅ RESOLVED 2026-07-16 — Redi is 40.1% of runtime, the biggest region in the
  model.** This caveat used to read *"NO PRODUCTION PROFILE HAS REDI ENABLED …
  'Redi is X% of runtime' is UNKNOWN … getting a profile of the MEKE config is
  the single highest-value next step."* That step was taken: profiling
  `gabight_sph_meke_v100.nml` puts `ocean_redi` at **40.1%** (LOGBOOK §4). The
  earlier logs read `0.0%` only because `ocean_redi_nml enable` is `.false.` in
  those configs. At 1.35x, PRECOMP is worth **~10.4% of total runtime**.
  **⚠ But do not multiply carelessly**: the 40.1% share is an **H200** profile
  while every ratio in this README is **V100** (LOGBOOK §1.2). Combining them is
  an extrapolation — codegen ratios should travel, occupancy/bandwidth-resting
  ones may not. Re-running this benchmark on the H200 is the cheap way to firm
  it up.
- **⚠ RUN-TO-RUN VARIANCE IS LARGE.** Same binary, same args: 44.6 / 44.9 /
  44.8 / 48.8 / 53.8 ms across 5 consecutive runs, and one 85.9 ms outlier.
  This is a shared analysis node and the V100 drifts under sustained load. All
  headline numbers are **min-of-3** (`sweep.py`); the min is the right estimator
  because interference only ever makes a run slower. **Differences under ~5%
  here are noise.** `dc/cuda = 0.994` should be read as **"a tie"**, not "Fortran
  wins by 0.6%".
- **⚠ ORDER DEPENDENCE was a real confound and is controlled.** DC was timed
  first and CUDA second, so a warming GPU penalised CUDA. Re-run with `order=1`
  (CUDA first): dc/cuda = 0.994 / 0.990 — **both orders agree**, so the verdict
  is not an artifact. This is exactly the RESUME warm-up trap, and it would have
  inverted a 5% claim.
- **All-wet** (`wet_u = wet_v = 1`), so the Phase-A land branches never fire and
  every face runs the full sweep. A real coast would be *cheaper* and more
  divergent. Untested.
- **Closed basin** (all four edges `OBC_WALL`), which does the most work.
  `gabight_sph_meke_v100.nml` actually has open W/E/S, so production does
  slightly *less* Phase-B work than this harness.
- **EOS = LINEAR**, correct for every Redi-enabled config (no `&eos_nml`). A
  nonlinear EOS (Wright/Roquet) would make Phase A markedly more expensive and
  could shift the Phase A / Phase B balance. Untested.
- The **CUDA port is faithful, not tuned**: no tiling, no `__ldg`, no
  `launch_bounds`, no shared memory. It answers "does nvfortran's codegen keep
  up?" (yes), not "how fast could Redi be in CUDA?". It also deliberately does
  **not** hoist the column rebuild — that would have made it an algorithm
  comparison rather than a codegen one.
- **`ref_pres` is a live dummy of a dead knob**: `redi_calc_coeffs_x/_y` accept
  it and never read it; the interface pressure is always built hydrostatically.
  Left verbatim. Worth a look in production.
- The **initial condition is synthetic** (exponential thermocline + a diagonal
  ~6 degC surface front). The neutral sweep is *data-dependent* — its branch
  behaviour, and therefore its cost, depends on how the isopycnals tilt. A real
  ocean state could differ. This is the caveat I am least able to bound.

## What this means for the ocean model

1. **Do not rewrite Redi in CUDA.** `do concurrent` already ties a faithful
   hand-written port at every production size (0.95-1.14x), and the portable
   PRECOMP variant *beats* that port by 1.35x. This is the fourth kernel in a
   row where CUDA buys nothing. **And the tie is not an artifact of launching
   CUDA from Fortran**: a native `cudaMalloc` + `main()` driver with no OpenACC
   anywhere (`redi_native.cu`) ties the Fortran-driven port at 1.00x, and the
   `host_data` path it removes costs a measured 2.5 µs/call — 0.006%. The
   comparison was fair; CUDA had no hidden handicap to remove.
2. **Take the PRECOMP transform** — 1.35x, bit-identical, pure Fortran, ~139 MB.
   `ocean_redi_pre.F90` is a working proof.
3. **Then do Phase A.** With Phase B fixed, `redi_calc_coeffs_x/_y` become
   **52% of the remaining cost (17.3 of 33.2 ms)** — and they have the *same*
   defect: `redi_face_coeffs` calls `redi_build_column` for **both** adjacent
   columns, so each column is rebuilt by its 2 u-faces and 2 v-faces = **4x
   redundant**, with 10 more NZ_STACK_MAX locals in `redi_face_coeffs`. The same
   hoist applies (precompute Pint/Tint/Sint/dRdT/dRdS per column once).
   **NOT IMPLEMENTED AND NOT MEASURED** — the 4x is a static read of the call
   graph, and a naive scaling to ~2x overall is an *extrapolation*, not a
   result. It is the obvious next experiment.
4. **Profile the MEKE config** before spending any of this. See caveat 1.
