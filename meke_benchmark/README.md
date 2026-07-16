# MEKE step — `do concurrent` vs hand-written CUDA C

## Result: 1.67x is free in Fortran; a CUDA rewrite buys 10% more

473x297 points (+3 ghosts) x nz=30 = **145137 cells**, the config that turns MEKE
on (`~/analysis_gebco/gabight_sph_meke_v100.nml`), V100, nvfortran 26.5-0,
production's flags, min-of-3:

```
  DC = PRODUCTION TODAY (16 blocking loops)   : 0.3554 ms/step
  DC + derived-type dummies (the ANTI-fix)    : 0.4255 ms/step   <- 1.20x WORSE
  DC + FUSED (16 -> 6 loops)                  : 0.2610 ms/step
  DC + ACC async(1) + one drain (16 loops)    : 0.2397 ms/step
  DC + FUSED + ACC async (6 loops)            : 0.2127 ms/step   <-- best Fortran
  CUDA faithful (16 kernels)                  : 0.2318 ms/step
  CUDA fused    ( 6 kernels)                  : 0.1989 ms/step
  CUDA graph    ( 6 kernels, 1 graph)         : 0.1928 ms/step

  dc / dc-fused-acc   -> 1.671 x   FREE, in Fortran, bit-identical
  dc-fused-acc / cuda-fused -> 1.069 x
  dc-fused-acc / cuda-graph -> 1.103 x   <-- ALL a CUDA rewrite buys: 10%
  dc / cuda-graph     -> 1.844 x   (total headroom, most of it free)
```

**Bit-identical throughout, exactly:** DC vs DC-DT, DC vs DC-FUSED, DC vs DC-ACC
and DC vs DC-FUSED-ACC are all `max|d meke| = 0.0`. DC vs CUDA is **also exactly
0.0** — the two toolchains agree bit-for-bit here, not merely to FMA level.

> That exact-zero was verified to be a real agreement and not a dead check:
> perturbing ONE term in ONE CUDA kernel by 1e-7 relative makes it read
> `1.25e-12` and trip the SUSPECT branch. The harness discriminates.

## ⚠ Read this before quoting any of the above: MEKE is ~1% of runtime

> **⚠ RETRACTED — THIS SECTION'S HEADLINE IS NOT ESTABLISHED.** The "~1%" below is
> an *inference* from profiles in which MEKE was switched **off**, not a
> measurement of MEKE. **MEKE has no profiler region in production**, so its true
> share of runtime is **unknown**, and the code owner reports it *did* show up on
> another machine. The 1.67x is therefore **unvalued, not worthless**. The
> arithmetic below (0.3554 ms x 34560 steps) is still correct *as arithmetic*;
> what is unsupported is treating it as MEKE's profile share. **Do not quote the
> ~1% figure.** Settling this needs a profiler region around `meke_step`, which
> nothing in this benchmark can substitute for.

**MEKE does not appear in the production profile at all.** It is default-off
(`&ocean_meke_nml enable = .false.`), only two of the 33 `~/analysis_gebco`
namelists switch it on (`gabight_sph_meke_v100`, `gabight_sph_bkscat_v100`), and
neither is what the 120-day profile ran. It runs **once per outer step at thermo
cadence**, so even when enabled:

> 0.3554 ms x 34560 thermo steps = **12.3 s out of 1340 s = 0.9% of runtime**.
> Making it 1.67x faster is worth **~0.37% of total** — and 0% in every config
> that has actually been profiled.

**The value of this benchmark is not MEKE.** It is that MEKE isolates a cost that
is invisible in the profile and applies to *every* <model> module written as many
small blocking `do concurrent` loops. See "What transfers", below.

## The three findings

### 1. The signature fix does not apply — MEKE already has it (H1 falsified)

The prompt's leading hypothesis (worth 1.30x on btstep) was that arrays reach the
kernel through derived-type components. **Production MEKE already passes plain
explicit-shape dummies bounded by plain integer dummies, in every one of its
kernels.** There is no fix to apply. (`meke_mass(nx, ny, nz, ..., h_layer(nx,ny,nz))`
— the only `grid%`-bounded dummy in the file is `set_f_centre`, a host setup
routine, not a kernel.) Every one of the 16 loops reports `auto-collapsed`,
`collapse(2)` — compiler bug 1 is absent, as that predicts.

So the only way to measure what MEKE's signature is *worth* was to **undo** it:
`meke_dt.F90` rewrites the same kernels in btstep's derived-type idiom,
changing nothing else. Result: **1.20x slower**, bit-identical. The mechanism
reproduces in the SASS —

| kernel | LDG (plain dummies) | LDG (DT components) |
|---|---|---|
| `length_scales` | **16** | **61** |
| `drag` | **4** | **18** |

— and the *magnitude* matches btstep's array-count scaling exactly as predicted:
btstep's Pass 1 touches ~19 arrays and lost 1.30x; MEKE's loops touch 4-11 and
lose 1.20x. **This is a confirmed prediction, not a new finding.** MEKE's existing
signature is already banking it.

### 2. The real win is an async wrapper MEKE simply does not have — 1.48x

**Production MEKE has no `!$acc` compute directive at all.** Its 16 loops are
plain *blocking* `do concurrent`. The barotropic substep next door does have one
(`barotropic_substep.F90:573,589,1346` + a single `wait(1)` at 1369), which is
why btstep never saw this cost.

Under `-stdpar=gpu` a blocking `do concurrent` costs **one `cuStreamSynchronize`
per loop**. Measured, `nsys` on this binary:

```
  cuLaunchKernel        3300 calls   <- EXACTLY the expected Fortran launch count
                                        (16+16+6+16+6 loops x 55 steps)
  cuStreamSynchronize   4348 calls   28.8% of API time, avg 16 us
```

Because MEKE's loops live in 16 separate `pure subroutine`s, one `!$acc kernels`
region cannot span them (a region cannot cross a call). So `meke_acc.F90`
wraps **each kernel's own loop** in `async(1)` and drains once in `meke_step`.
Same stream ⇒ the queue preserves inter-loop ordering ⇒ bit-identical (checked).
That local change alone is **1.48x**.

**It is pure host scheduling, not codegen — measured, not inferred.** The SASS is
byte-identical between the plain and wrapped variants:

| kernel | plain DC | + `!$acc kernels async(1)` |
|---|---|---|
| `meke_mass` | LDG=177, inst=2024 | LDG=177, inst=2024 |
| `meke_length_scales` | LDG=16, inst=864 | LDG=16, inst=864 |
| `meke_drag` | LDG=4, inst=176 | LDG=4, inst=176 |

**Zero spills in every variant.** RESUME_GPU_MRE.md §5's "possible bug 3" (the
`!$acc kernels` wrapper spilling a register-heavy DC loop 56→90 with 16 spills)
**does not reproduce here**, including on the fused mega-loops L3/L6 which are
exactly the register-heavy case that report warns about. That is a negative
result on a real hypothesis, not an absence of testing.

### 3. Fusion is worth 1.36x on its own, and the two stack to 1.67x

16 loops → 6 (`meke_fused.F90`). Walls fold into producers as
single-assignment merges (`uflux = (interior ? computed : 0)`); six pointwise
stages fold into two mega-loops. Bit-identical.

What cannot be fused, **checked rather than assumed**: L3→L4/L5 and L4/L5→L6 are
real barriers (the flux loops read `meke`/`mass_ws` *neighbours*; the divergence
reads `uflux(i+1)`/`vflux(j+1)`). The `sn_u`/`sn_v` zero loops cannot fold into L3
because `length_scales` reads `sn_u(i+1,j)` — a neighbour read. **6 loops is that
dependency structure, not an implementation limit.** Full analysis in the file
header.

Fusion and the async wrapper are largely independent (they remove the same fixed
per-loop cost by different means, so they do not multiply): 1.36x and 1.48x
separately, **1.67x together**.

## The gap is a pure function of TOTAL CELL COUNT

Min-of-3, ms/step. `dc_f+a` = fused + async = best Fortran.

```
 config (cells)             dc    dc_DT   dc_fus   dc_acc   dc_f+a    cu_16     cu_6    cu_gr
 108x137x30   (16302)   0.1899   0.2348   0.0977   0.0744   0.0521   0.0803   0.0499   0.0445
 473x297x30  (145137)   0.3554   0.4255   0.2610   0.2397   0.2127   0.2318   0.1989   0.1928
 240x560x30  (139236)   0.3525   0.4318   0.2586   0.2352   0.2107   0.2284   0.1985   0.1925
 359x458x30  (169360)   0.3904   0.4609   0.2908   0.2714   0.2398   0.2764   0.2219   0.2152
 945x594x30  (570600)   0.9397   1.0492   0.7886   0.8236   0.7388   0.8054   0.7206   0.7117
 1890x1188x30(2263824)  3.1513   3.3118   2.8727   3.0351   2.8198   3.0315   2.7998   2.7959
 --- nz sensitivity, same footprint ---
 473x297x1   (145137)   0.2644   0.3289   0.1628   0.1488   0.1146   0.1477   0.1085   0.1048
 473x297x75  (145137)   0.4951   0.5424   0.3945   0.3750   0.3444   0.3559   0.3273   0.3211
```

`dc / cuda-faithful` (equal kernel count) runs **2.37x → 1.53x → 1.17x → 1.04x**
across 16k → 145k → 570k → 2.26M cells. The *absolute* gap is flat at
**~8 us per loop**, independent of size:

| | 16k | 145k | 570k | 2.26M |
|---|---|---|---|---|
| (dc − cuda_16) / 16 loops | 6.9 us | 7.7 us | 8.4 us | 7.5 us |
| (dc_fus − cuda_6) / 6 loops | 8.0 us | 10.4 us | 11.3 us | 12.2 us |

That is the same ~10 us/loop the `continuity_ppm` benchmark measured, and it is
the *whole* story: a fixed host cost per `do concurrent` that dominates when
there is little work per launch and vanishes when there is a lot. **Consistent
with, and independently confirming, the existing finding — not a new mechanism.**

At 570k+ cells **plain fused Fortran beats hand-written CUDA C's faithful port**
(0.7886 vs 0.8054; 2.8727 vs 3.0315). nvfortran's codegen is not the problem.

## What transfers to the rest of the ocean model (the actual point)

MEKE is 1% of runtime. The ~8 us/loop is not MEKE-specific:

- **Any module written as many small blocking `do concurrent` loops pays it.**
  The barotropic substep is immune only because it explicitly wraps its loops in
  `!$acc kernels async(1)`. That idiom is in the codebase already; it is applied
  in one place.
- The fix is **local and mechanical**: wrap each kernel's loop in `async(1)`,
  drain once per step. It does not require restructuring, does not change codegen,
  and is bit-identical. It does not even require the loops to live in one routine.
- **Worth auditing which other GPU-resident modules lack the wrapper**, weighted
  by their profile share. On the profiled 120-day run that means
  `ocean_vmix_compute` (39.1%), `ocean_ale_remap` (11.2%), `ocean_continuity`
  (11.1%) — none of which this benchmark touched. That is an unmeasured
  hypothesis, and MEKE is very weak evidence for it: **MEKE's loops are tiny, so
  its 8 us/loop is 36% of its runtime; a module with fat loops would hide the
  same 8 us entirely.** The audit is worth doing; the payoff is not predictable
  from here.

## Is the OpenACC runtime taxing the CUDA numbers? No — measured, and it cannot

Every CUDA figure above is launched **from Fortran**, with device pointers from
`!$acc host_data use_device`. If that path cost anything, every "CUDA buys N%"
number here would be understated. `meke_native.cu` settles it: `cudaMalloc` +
`main()`, the same `meke_cuda_launch()` from `meke_kernel.o`, **no Fortran and no
OpenACC in the process at all** (`ldd meke_native` shows no `libacchost`,
`libaccdevice`, `libnvf` or `libnvomp`; `meke_bench` links all four).

**The two drivers are BIT-IDENTICAL, exactly.** Given the same starting bits,
`max|d meke| = 0.0`, **0 of 145137 elements differing at bit level** — see
"How that was verified", below. Contention cannot affect this result.

```
  min over 18 interleaved rounds, 473x297x30    ⚠ HEAVILY CONTENDED — PROVISIONAL
  mode        fortran-driven (host_data)   native (cudaMalloc)   ratio
  faithful              0.2308                   0.2315          0.997 x
  fused                 0.1978                   0.1965          1.007 x
  graph                 0.1917                   0.1907          1.005 x
```

**They tie.** The ratios straddle 1.0 **in both directions** (0.997 / 1.007 /
1.005) at ±0.7%, which is the signature of noise, not of a systematic cost.
**No conclusion in this README shifts. Every CUDA number here is already fair.**

**And the tie is what the code predicts — this is mechanism, not luck.**
`meke_bench.F90:233` calls `fill_args` **ONCE, outside the timing loop**;
`time_cuda` then loops on `meke_cuda_launch` alone. So the host_data
present-table lookup is paid **once per run, not once per call**, and was never
in the timed region to begin with. The general worry ("a present-table lookup
per array per call") is real, but **this benchmark's driver does not have that
shape**, so there was no per-call tax available to remove. What the native driver
actually bounds is the *process-level* cost of the OpenACC runtime being resident
(context setup, default-stream semantics) — measured here as **nil**.

> ⚠ **The timings above were taken with the V100 at 100% utilisation from ~9
> sibling jobs** and are PROVISIONAL. Raw samples swung **15x** (0.19 → 2.83
> ms/step) on *both* binaries; min-of-10 in one window came out *worse* than
> min-of-6 in an earlier one, i.e. the noise floor itself moved. Two things make
> the tie credible anyway: it is a **like-for-like min from interleaved rounds**,
> and the Fortran minima reproduce this README's own uncontended numbers
> (0.2318 / 0.1989 / 0.1928) to within 0.6%. **A serial re-run is still required
> to quote these as absolutes.** The bit-identity result needs no re-run.

### How that was verified (and why the check is not a dead one)

Verifying against the Fortran run's output has a trap: the two drivers compute
their init on the **host**, with **different libm**s. Naively diffing the final
fields gives `max|d| = 2.08e-17` (~1.9e-15 relative) — which proves nothing
either way, because the runs never started from the same bits:

```
  init meke     (host init) : max|d| = 3.47e-18   bit-differing:  8400 / 145137
  init f_centre (host init) : max|d| = 2.71e-20   bit-differing: 14370 / 145137
  init gm_src   (host init) : max|d| = 1.73e-18   bit-differing: 29661 / 145137
```

That is nvfortran's `exp()` vs glibc's `exp()` at ~1 ULP (and host FMA
contraction in `f_centre`) — **not** a port bug, and **not** on the kernel path.
So `MEKE_DUMP=1 ./meke_bench` dumps its own `meke`/`f_centre`/`gm_src` bits, and
`./meke_native --use-ref-init` starts from **those**. Those three are the only
init fields that depend on libm or host FMA; everything else is exact constants
and plain IEEE arithmetic. With that one variable removed:

```
  final meke (end-to-end)   : max|d| = 0.00000e+00  bit-differing: 0 / 145137
```

**The check discriminates — that is shown, not asserted.** The same comparison
reports non-zero when the inputs genuinely differ, and it resolves a difference
as small as **2.71e-20** (`f_centre`). A harness that detects 1 ULP in the init
and then reads exactly 0 for the final field is reporting a real agreement.

The native driver also asserts the **degenerate-config invariants** at runtime
(`le==0`, `kh_diff==0`, `bottom_fac2==1`, `barotr_fac2==1` — all Y), so a silent
config drift surfaces instead of being assumed away. It **inherits the degenerate
config** documented under Caveats: it exercises MEKE's shape, not its arithmetic.

## Verdict on the CUDA question

**No CUDA rewrite is justified, and this is the weakest case for one so far.**
Once the two free Fortran wins are taken, CUDA is 6.9% ahead (fused) / 10.3%
ahead (graph) at the production config, on a routine worth ~1% of runtime — i.e.
a hand-maintained C port of a whole parameterisation to buy **~0.04% of total**.
CUDA graphs remain the only CUDA-exclusive capability, and here they are worth
3% over fused CUDA.

Consistent with the other three kernels: where Fortran lost, it lost to a fixable
Fortran-level defect, never to CUDA.

## ⚠ Build flags are load-bearing: NVHPC 26.5 / NVIDIA TPR #38714

Copied verbatim from `<model>/build/CMakeFiles/core_model_objs.dir/flags.make`:

```
-Mfree -Mbackslash -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate \
-gpu=tripcount:host -O3 -fast
```

`-gpu=tripcount:host` is **mandatory** (NVHPC 26.5 reads loop trip-counts from
the device copy, inserting a per-kernel data refresh in multi-loop `!$acc kernels`
regions — ~2x wrong without it). the ocean model guards it in
`cmake/compiler_flags.cmake:66-72`. This benchmark's flags must stay in sync.

## Build / run

```bash
source ../../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
make && make run                   # 473 297 30 -- the MEKE config
./meke_bench 945 594 30            # [nx] [ny] [nz]  (nx/ny are POINT COUNTS)
./meke_bench 473 297 30 200 20     # [nreps] [nwarm]
N=3 ./sweep.sh                     # min-of-3 across the real configs

make native && ./meke_native       # NATIVE C++/CUDA: cudaMalloc + main(), no
                                   # Fortran, no OpenACC. Same args + [ntrials].
make verify-native                 # dump from meke_bench, then diff native
                                   # against it (init AND final fields)
./meke_native --use-ref-init       # start from the Fortran's own init bits ->
                                   # final field must be EXACTLY bit-identical
make collapse                      # per-loop -Minfo (all report auto-collapsed)
make regs                          # LDG + spills per kernel, mangled names
make verbatim                      # re-slice from production and diff -- MUST pass
```

> `sweep.sh` takes **min**-of-N, not mean. The The HPC system analysis node is shared and a
> co-tenant job produces occasional 20-40x outliers on whichever variant is
> running when it lands (observed: a 7.5 ms `dc` at a config that reproducibly
> measures 0.187 ms). Min-of-N is the robust statistic for "how fast is this
> kernel". Single-shot numbers from this harness are not trustworthy.

## Files

```
meke.F90          KERNEL BODIES VERBATIM from production (md5 of source
                      b29cc175f5203bc994ebd81f04aa7f62; slices 540-577, 583-617,
                      723-962, 1098-1215, kept in kernels_verbatim.inc and
                      re-checked by `make verbatim`). Its `meke_step_ext` driver
                      is a TRANSCRIPTION -- see the header for what was dropped.
meke_dt.F90       same body, derived-type-component dummies. The H1
                      counterfactual. NOT production -- production run backwards.
meke_fused.F90    same body, 16 loops -> 6. Documents what cannot be fused.
meke_acc.F90      auto-generated from meke.F90: every loop wrapped in
                      `!$acc kernels async(1)`, one `wait(1)` drain.
meke_fused_acc.F90 fused + async. Best Fortran.
meke_state.F90    MRE stubs (grid / metrics / multilayer state / gm) +
                      ocean_meke_t copied field-for-field with its defaults.
meke_kernel.cu        faithful CUDA port: 16 kernels / 6 fused / cudaGraph.
meke_args.h           `struct MekeArgs` + the IT/IU/IV/I3 stride macros. ONE
                      definition, shared by meke_kernel.cu, meke_native.cu and
                      (field-for-field, via bind(C)) meke_bench.F90. Previously
                      inside meke_kernel.cu; extracted so the native driver could
                      not hold a COPY that silently drifts.
meke_native.cu        NATIVE driver: cudaMalloc + main(), no Fortran, no OpenACC.
                      Calls the SAME meke_cuda_launch() from meke_kernel.o -- it
                      does not redefine a kernel. Answers "is the OpenACC runtime
                      taxing the CUDA numbers?" (no).
meke_bench.F90        driver. Separate output state per variant. `MEKE_DUMP=1`
                      makes it dump its init/final fields for meke_native.
sweep.sh              min-of-N sweep over the real configs.
```

## Caveats — read before extrapolating

- **MEKE is ~1% of runtime when on, and off in every profiled config.** Restated
  because it is the single most important caveat here.
- **The length-scale machinery is DEGENERATE at this config, and that is
  faithful.** All five `alpha_*` weights default to 0 and no `&wavespeed_nml` is
  present, so `rd_ws = 0` ⇒ `Ldeform = 0` ⇒ `bottom_fac2 = barotr_fac2 = 1`
  exactly, `inv_lmix` returns 0 ⇒ **`le ≡ 0` ⇒ `kh_diff ≡ 0`**. `meke_length_scales`
  and `meke_kh_closure` still run, still load every array and still execute the
  `sqrt`/`pow`, but their branches never fire and their outputs are zero. This is
  exactly what production does with `gabight_sph_meke_v100.nml` — but it means the
  benchmark exercises MEKE's *shape*, not its arithmetic. A config with live
  alphas + a wavespeed slot would be a different kernel. **Untested.**
- **It also explains the exact-zero DC-vs-CUDA agreement** (`pow(1.0, 0.8)` is
  exactly 1 in both toolchains). Do not read that agreement as evidence that
  nvfortran and nvcc contract FMAs identically in general — it is evidence that
  this config's arithmetic is degenerate enough not to expose the difference.
- **Dropped branches** (each statically false at this config, listed in
  `meke.F90`'s header): advection (7 loops), bbl drag (1), `feed_khth` (2),
  the k4 biharmonic block (9). `meke_backscatter_apply` — a separate entry point
  on a different cadence, live only in `gabight_sph_bkscat_v100.nml`, and 3-D
  (nz layers) so NOT launch-bound — is **not benchmarked at all**. The full file
  has 39 DC loops; the live `meke_step` path at this config has 16.
- **Uniform-Cartesian metrics, all-wet, no land.** A coast would add divergence.
- The CUDA port is **faithful, not tuned**: no tiling, no `__ldg`, no
  `launch_bounds`. A tuned port would widen the gap — a different question from
  "does nvfortran's codegen keep up?".
- **UNEXPLAINED:** at >=570k cells the async wrapper is a small net *loss* vs
  plain fusion (0.8236 vs 0.7886 at 945x594; 3.0351 vs 2.8727 at 1890x1188) —
  reproducible across 4 clean runs, ~3-5%. The wrapper's benefit is fixed host
  time, which amortises away, but that explains it going to zero, not negative.
  The SASS is identical, so it is not codegen. No mechanism; it does not affect
  the production config (145k cells), where the wrapper wins.
- The Fortran fused mega-loop L3 issues **133 LDG vs the CUDA port's 76** for the
  same work. That is a real codegen difference and is the most likely source of
  the residual 7-10%, but it was **not isolated** — the loads mostly hit cache
  (the gap nearly vanishes at 2.26M cells, where codegen is all that is left).
```
