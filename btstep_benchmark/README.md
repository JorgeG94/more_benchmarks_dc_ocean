# barotropic substep — `do concurrent` vs hand-written CUDA C

## Result: portable Fortran gets 1.42x; a full CUDA rewrite buys 6% more

473x297, n_inner=24 (production's config), V100, nvfortran 26.5-0, production's flags:

```
  DC = PRODUCTION TODAY (bt_work% components)  : 3.704 ms/call
  DC + signature fix (plain dummies)           : 2.851 ms/call
  DC + signature fix + FUSED (11 -> 5 loops)   : 2.618 ms/call   <-- all Fortran
  CUDA faithful (11 kern/substep)              : 3.035 ms/call
  CUDA fused    ( 5 kern/substep)              : 2.646 ms/call
  CUDA graph    ( 5 kern, 1 graph)             : 2.464 ms/call

  dc(prod) / dc-fused   -> 1.415 x   FREE, in Fortran, bit-identical
  dc-fused / cuda-fused -> 0.985 x   Fortran WINS at equal kernel count
  dc-fused / cuda-graph -> 1.062 x   <-- ALL a CUDA rewrite buys: 6%
```

Bit-identical throughout: DC vs DC-plain and DC vs DC-FUSED are `max|d eta| = 0.0`
**exactly**; DC vs CUDA is 1.3e-15 on `|eta| ~ 0.374` (FMA contraction only).

**No CUDA rewrite is justified.** Everything that matters is reachable in
`do concurrent`, and the fused Fortran *beats* the fused CUDA at equal kernel
count. The only CUDA-exclusive win is **CUDA graphs, worth 6%** -- the price is a
hand-maintained C port of the barotropic substep.

### The two Fortran wins, both bit-identical

**1. Signature fix -- 1.30x.** Plain explicit-shape dummies instead of `bt_work%`
/ `metrics%` / `cor%` components:

| | Pass 1 LDG | Pass 1 inst |
|---|---|---|
| `bt_work%bt_eta(i,j)` etc. | **103** | 504 |
| plain `eta(i,j)` dummies | **38** | 296 |
| (hand-written CUDA, for scale) | 47 | 416 |

The caller keeps its derived types; only the kernel's dummies change. Pass 1
touches ~19 distinct arrays and the cost scales with that count -- which is why
the layered continuity (6 arrays, heavy math) shows no such gap while this kernel
(19 arrays, light arithmetic) loses 1.3x. **Cost: the signature goes from 8
arguments to 34.** That is the real trade, not performance.

**2. Fusion -- 1.09x on top.** 11 loops -> 5 per substep. Walls and accumulators
fold into their producers as single-assignment merges:
`ubt = (wall ? 0 : computed); ubt_sum += ubt`. Same values, same order, so it is
bit-identical rather than an approximation. This is a source transform, NOT a
CUDA feature -- `btstep_fused.F90` does exactly what `btstep_kernel.cu`'s
FUSE_WALL path does.

What cannot be fused, checked rather than assumed: Pass 2b -> Pass 2c is
sequential (2c's `u_at_v` reads the `ubt` 2b just wrote -- Gauss-Seidel, not
Jacobi); Pass 1 -> swap -> Pass 2b are separated by real barriers (Pass 2b reads
eta/zeta *neighbours*). 5 loops is that dependency structure, not a limit of the
implementation.

`ocean_barotropic_solver` is 8.6% of runtime with EPBL on and ~16% with it off,
so 1.42x on it is worth roughly **2.5-4.7% of total runtime, in portable Fortran**.

## Is the CUDA side handicapped by OpenACC? No — a native driver proves it

Every CUDA number above is launched *from Fortran*, and gets its device pointers
from `!$acc host_data use_device(...)`. If that costs anything per call, the
CUDA column is understated and "Fortran wins" is an artifact. `btstep_native.cu`
settles it: `cudaMalloc` + `main()`, no Fortran, no OpenACC, **calling the same
`btstep_cuda_launch()` out of the same `btstep_kernel.o`** that `btstep_bench`
links.

**Result: they tie.** `./native_vs_fortran.sh`, 12 clean samples per binary,
alternating order, min-of-samples, V100 verified idle before *and* after every
sample:

```
  mode                     CUDA via Fortran   CUDA native      ratio
                           (host_data)        (cudaMalloc)
  faithful (11 kern)            2.996 ms        3.009 ms      0.996 x
  fused    ( 5 kern)            2.602 ms        2.629 ms      0.990 x
  graph    ( 5 kern, 1 graph)   2.457 ms        2.477 ms      0.992 x
```

Native is 0.4–1.0% *slower* — i.e. identical to within run-to-run noise, with no
credible direction. **Removing OpenACC from the CUDA path buys nothing here.**

That is what the code already implied, and the point of saying so out loud is
that it is checkable: `fill_args` is called **once, outside the timing loop**,
filling a by-value struct of device pointers, so the timed loop is a bare
`call btstep_cuda_launch(a, n_inner, mode)` with no present-table lookup in it.
There was never any per-call OpenACC overhead to remove. **The CUDA numbers in
this benchmark are fair, and no conclusion above shifts.**

⚠ **Scope.** This clears *this* harness, not the repo. A port that calls
`host_data` **inside** its timing loop pays a per-call lookup this benchmark
never had — those have not been cleared, and the cheap check is to look at where
`host_data` sits relative to the timer.

### Verification: bit-exact, not "close"

`make verify` runs both drivers with identical args; the native one checks
itself against the Fortran-driven result and gets **`max|d eta| = 0.0`,
`max|d ubt| = 0.0`** — bit-exact after 5040 substeps. Same kernels, same
numbers, so the only remaining variable is OpenACC. That is what licenses
comparing the two timings at all.

Getting there turned up a trap worth recording. A native driver that
**recomputes** the init from the same formulas does *not* bit-match — it lands
at `3.1e-15` on `|eta| ~ 0.265`, right at the repo's usual FMA-level bar, which
looks like a port bug and is not one. **nvfortran's libm and glibc's libm round
`exp()` and `sin()` differently by 1 ulp** (measured: 29301/145137 of the
Gaussian bump, 46/303 of `force_u`), and over 5040 substeps that 1 ulp grows to
1e-15. So the ref dump carries the three not-exactly-representable *inputs*
(`eta_ini`, `force_u`, `f_corner`) as well as the outputs, and the native driver
adopts them — reporting the libm gap rather than hiding it. Everything else
(metrics, `H_ref`, `rem_u/v`, the zeroed arrays) is exact in both languages.

**Generalisable:** a cross-language bit-comparison must hand over the
transcendental inputs, not recompute them. Otherwise you are measuring libm.

## ⚠ Build flags are load-bearing: NVHPC 26.5 / NVIDIA TPR #38714

**This benchmark is only valid with `-gpu=tripcount:host`.** Without it:

```
  ACC (!$acc kernels async(1))  7.20 ms   <-- 2.4x SLOWER
  with -gpu=tripcount:host      2.95 ms
```

NVHPC 26.5 reads loop trip-counts from the device copy, inserting an extra
per-kernel data refresh in multi-loop `!$acc kernels` regions. `nsys` sees it as
one ~184-byte `cuMemcpyDtoHAsync` per `do concurrent` loop at ~21 us each (async
to *pageable* memory is synchronous), 63% of API time.

The ocean model already guards this -- `cmake/compiler_flags.cmake:66-72`, gated on
`CMAKE_Fortran_COMPILER_VERSION >= 26.5`, citing TPR #38714 and "~2x on the
barotropic solver". This benchmark's flags are copied from
`build/CMakeFiles/core_model_objs.dir/flags.make` and must stay in sync:

```
-Mfree -Mbackslash -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate \
-gpu=tripcount:host -O3 -fast
```

**Lesson, paid for the hard way:** an MRE that claims to model production must
copy production's *build flags* first. Hours went into "explaining" a 2.3x gap
between this harness and production that was entirely this one flag -- and every
intermediate finding (the wrapper costing 1.42x, plain-DC beating the ACC
wrapper, the per-loop scalar copies) was an artifact of it. With the flag, the
ACC wrapper does exactly what it is supposed to: **ACC/NOACC = 0.59x**.

## Build / run

```bash
source ../../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
make && make run                  # 473 297 24 -- production's config
./btstep_bench 512 512 15         # [nx_phys] [ny_phys] [n_inner]
./btstep_bench 473 297 24 50 10   # [nreps] [nwarm]
make NVTX=1                       # one nsys range per variant (~2.5% overhead)

make native && ./btstep_native    # NATIVE C++/CUDA driver: no Fortran, no
                                  # OpenACC, no nvfortran. Same args as above.
make verify                       # both drivers, identical args -> the native
                                  # one checks itself BIT-EXACT against the
                                  # Fortran-driven CUDA result.
./native_vs_fortran.sh            # paired timing, waits for an idle GPU
```

## Files

```
btstep.F90         TRANSCRIPTION of production's live closed-basin path (NOT a
                       verbatim extract -- see its header: wetdry, BT_cont,
                       upstream-h_face, Flather OBC and periodic wraps are dropped
                       because the gabight configs do not use them). Models the
                       substep's SHAPE; not bit-comparable to a production run.
btstep_plain.F90   same body, plain explicit-shape dummies -> the 1.30x.
btstep_fused.F90   same body again, 11 loops -> 5 -> a further 1.09x. Documents
                       what cannot be fused and why.
btstep_noacc.F90   same body, acc directives deleted -> plain do concurrent.
btstep_kernel.cu       CUDA: faithful (11 kern) / fused (5 kern) / graph.
btstep_args.h          `struct BtArgs` + the launch prototype, shared by the .cu
                       and the native driver so they cannot drift. Mirrored by
                       hand in btstep_bench.F90 as `bt_args_t` -- field order is
                       load-bearing and a mismatch is silent corruption.
btstep_native.cu       NATIVE driver: cudaMalloc + main(), no Fortran, no
                       OpenACC. Does NOT redefine the kernels -- it links the
                       same btstep_kernel.o. Answers "is host_data costing us?"
native_vs_fortran.sh   paired, interleaved timing of the two drivers. Waits for
                       an idle GPU and takes min-across-repeats, because this
                       box is shared and a contended number is worthless.
btstep_bench.F90       driver. NVTX ranges here only, one per variant.
```

## Caveats

- Closed basin, no wetdry, `bebt = 0.2`. Production adds Flather OBC and the
  wetdry branches, so it does strictly more work per substep -- which is why this
  harness (2.85 ms) is *faster* than production's 3.35 ms and consistent with it.
- The CUDA port is faithful, not tuned: no tiling, no `__ldg`, no `launch_bounds`.
- `dc/cuda` numbers assume `cuda_sync=2` in the modes that have it, so CUDA pays
  the same per-kernel host sync `do concurrent` does.
