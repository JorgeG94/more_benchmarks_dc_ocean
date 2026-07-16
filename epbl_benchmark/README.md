# EPBL column kernel — `do concurrent` vs hand-written CUDA C

**The scheme inside `ocean_vmix_compute`, which is 39.1% of production runtime —
the largest region in the profile, and previously unmeasured.**

## Result: `do concurrent` ties hand-written CUDA (1.05x). The 1.33x that exists is occupancy, and it is NOT reachable from Fortran — not because Fortran can't ask, but because nvfortran's register allocator can't deliver.

473x297x30 (the profiled 0.1 deg config), V100, nvfortran 26.5-0 + nvcc 12.9,
production's flags, min of 6 interleaved trials on a quiet GPU:

```
  EPBL do concurrent, PRODUCTION VERBATIM        : 6.489 ms/rep    1.000 x
  DC + signature fix (plain dummies)             : 6.700 ms/rep    0.968 x   <- WORSE
  CUDA C faithful (1 thread/column, block 128)   : 6.155 ms/rep    1.054 x
  CUDA C + __launch_bounds__ (occupancy lever)   : 4.883 ms/rep    1.329 x
```

Three conclusions, in order of confidence:

1. **Codegen is fine.** The faithful CUDA port — one thread per column, same
   loops, same branches — beats `do concurrent` by **5%**. That is the
   codegen-only number, and it ties like every kernel measured before it — but
   for a different reason (see "Evidence"): here both compilers are pinned by the
   same occupancy limit, so nvfortran's genuinely worse codegen (255 regs and 178
   loads vs 184 and 48) cannot express itself.
2. **The lever is occupancy, worth 1.33x**, and it is real: `__launch_bounds__`
   takes the kernel from 12.5% to 25% theoretical occupancy.
3. **Fortran cannot collect it.** `-gpu=maxregcount:128` asks nvfortran for
   exactly what `__launch_bounds__` asks nvcc for. nvfortran obeys the cap and
   then spills **568 bytes** where nvcc spills **160**, and the result is
   **1.69x slower**, not 1.33x faster. See "The one thing CUDA can do" below.

**A CUDA rewrite of EPBL would win ~1.33x on a region worth 39.1% — roughly 10%
of total runtime.** That is the largest headroom this investigation has found.
It is also a hand-maintained C port of a 520-line iterative energetics scheme
whose outputs are chaotically branch-sensitive (see "Bit-identity"). The
arithmetic is genuinely arguable; this README does not make the call.

⚠ **Two caveats gate that 10% before anyone acts on it**, both expanded in
"Caveats". (1) The 39.1% region also wraps kappa-shear; EPBL's share of it is
**unmeasured**. (2) This benchmark ran on a **V100 (cc70)**; the 39.1% profile
ran on an **H200 (cc90)** — `<h200-node>`, `build_hopper`, `-gpu=cc90`.
**The headline 1.33x is the most hardware-sensitive number in this README and it
was measured on the wrong GPU.** Re-run on Hopper before acting on it.

## Why EPBL is different from every kernel measured before it

Everything measured so far (`hll_fluxes`, `continuity_*`, `btstep`) was
one-thread-per-CELL in *both* languages. They tied because CUDA and DC were
doing the identical thing. EPBL is one-thread-per-**COLUMN**:

```
ONE do concurrent (j=1:ny, i=1:nx)     -> auto-collapsed collapse(2), 128 threads
  prep sweep    k = nz..1                                 30 steps
  MLD iteration obl_it = 1..20   (false-position root-find)
      interface sweep ki = nz..2                          29 steps each
                                            -> up to ~610 SEQUENTIAL steps/thread
```

Verified, not assumed — `make collapse` prints exactly one `Loop parallelized
... collapse(2)` and three `Loop run sequentially`. Parallel width is
`nx_total*ny_total` = **145,137 columns** at the profiled config.

### There is no warp-per-column variant, and that is a finding

The brief hypothesised that a CUDA port could do warp-per-column with `__shfl`
reductions — the one thing `do concurrent` structurally cannot express. **It
cannot be done here, and the reason is in the source, not in the compiler.**

The interface sweep is a **first-order sequential recursion in k**: `hp_a`,
`dpe_t_a`, `dpe_s_a`, `dch_t_a`, `dch_s_a`, `th_a`, `sh_a`, `te_lag`, `se_lag`
and `kddt_prev` each read the value the *previous* interface wrote (it is the
forward elimination of a tridiagonal solve). Per interface the kernel touches
two k-levels, `ka` and `kb` — not 32. So there is no per-k parallel work for 32
lanes to share and no reduction for `__shfl` to collapse; a warp-per-column port
would idle 31 of 32 lanes through the dominant loop. The prep loop's `pres` *is*
a prefix scan and *is* warp-parallelisable, but it is 30 of ~610 sequential
steps — Amdahl caps the whole idea at ~5%.

The MLD iteration is likewise a sequential root-find. Parallelising *it* across
lanes (32 simultaneous MLD guesses) is a different **algorithm**, not a port, and
would change the answers.

So the interesting experiment turned out to be occupancy, not warp shape.

## Evidence: what was measured

`ncu`, one launch each, 473x297x30. Registers matched to the **mangled** name
(`ocean_epbl_epbl_column_kernel_791_gpu`) — the first `Used N registers` in a
raw ptxas dump is `cub::EmptyKernel` (RESUME_GPU_MRE §1).

| | regs | spill stack | occ limit | **achieved occ** | issue % | DRAM % | lane eff | time |
|---|---|---|---|---|---|---|---|---|
| DC (nvfortran) | **255** | 128 B | 2 blk | **8.50%** | 18.1% | 20.0% | 43.8% | 7.10 ms |
| CUDA faithful | 184 | **0** | 2 blk | 8.28% | 17.1% | 20.0% | 41.2% | 6.72 ms |
| CUDA tuned | 128 | 160 B | **4 blk** | **17.2%** | 20.5% | 28.0% | 41.1% | 5.42 ms |

Read it in this order:

- **Why DC and faithful tie despite 255 vs 184 registers.** Both land in the
  *same occupancy bucket* — 2 blocks/SM, ~8.3-8.5% achieved. On a V100 (65536
  regs/SM, 128-thread blocks) anything from 129 to 255 registers gives 2 blocks.
  nvfortran's 71 extra registers and 128 bytes of spill are real and cost 5%;
  they are not enough to change the binding constraint.
- **The kernel is LATENCY-bound, not bandwidth- or compute-bound.** DRAM sits at
  20% of peak and the SMs issue on only 18% of cycles. That is the signature of
  a long dependent chain of global loads with too few warps to hide it — which is
  exactly what a 610-step recursion at 8.5% occupancy is. **This is why occupancy
  is the lever**: doubling warps takes issue 17%→20.5% and DRAM 20%→28%.
- **Divergence is real, identical in both languages, and nobody's fault.** Lane
  efficiency (`smsp__thread_inst_executed_per_inst_executed / 32`) is 41-44% in
  *all three* kernels. Measured independently by the harness
  (`./epbl_bench ... 1`, which dumps per-column MLD-iteration counts):

  ```
  mean iterations per COLUMN         : 2.489
  mean iterations per WARP LANE-SLOT : 4.643      <- a warp costs its slowest lane
  worst warp max                     : 20
  DIVERGENCE TAX (warp/column)       : 1.866 x
  ```

  Iteration counts per column: 15461 columns take 0 (land), ~122k take 2-3, and
  **509 columns hit the 20-iteration cap** — each one poisoning an entire warp.
  Neither language fixes this; it is the scheme's root-find. It is the single
  largest inefficiency in the kernel and it is **algorithmic headroom, not
  language headroom**.

## The one thing CUDA can do — and why the Fortran equivalent backfires

The honest framing is *not* "CUDA can express `__launch_bounds__` and Fortran
cannot". nvfortran has `-gpu=maxregcount:<n>`. Both languages can ask for the
same register cap. Only nvcc survives it (`make maxreg`; the `cuda_tuned` column
is a control that must stay flat, proving the machine was stable across rebuilds):

```
  flag              regs stack     dc_ms   [cuda_tuned control]
  (default)         255  128      6.485        4.892
  maxregcount:255   255  136      6.492        4.867
  maxregcount:192   192  312      7.334        4.861
  maxregcount:160   160  440      7.290        4.882
  maxregcount:128   128  568     10.953        4.866    <- 1.69x WORSE
```

**At the identical 128-register cap: nvfortran spills 568 B → 10.95 ms; nvcc
spills 160 B → 4.87 ms. A 2.25x gap from register allocation alone.** nvfortran
reaches the target occupancy and loses anyway, because it pays 3.5x the spill
traffic to get there — on a kernel that is already latency-bound.

That is a much sharper claim than "expressiveness", and it is the one worth
sending to NVIDIA: *nvfortran's register allocator degrades far worse than
ptxas's under an equivalent cap on a register-heavy sequential kernel.*

## The signature fix does NOT work here — and that matters

btstep found plain explicit-shape dummies worth **1.30x** over `bt_work%`
components (103 loads/thread → 38). EPBL has the same signature: `make regs`
shows the DC kernel doing **178 LDG** against the CUDA port's **48**. The obvious
inference is that EPBL has a free 1.3x sitting there.

**It does not.** `ocean_epbl_plain.F90` (mechanically generated; identical
body, every derived-type array read replaced by a plain explicit-shape dummy on
plain integer bounds) does exactly what it says on the tin and buys nothing:

| | LDG | SASS inst | regs | time |
|---|---|---|---|---|
| DC, production (`this%t0%data(i,j,k)`) | **178** | 5072 | 255 | 6.489 ms |
| DC + signature fix (plain dummies) | **74** | 2768 | 255 | **6.700 ms** (0.968x) |

It cuts loads by 2.4x and instructions by 1.8x and is **3% SLOWER**. It is
`max|diff| = 0.0` **exactly** against production, so this is a correct transform,
not a broken one — it simply does not help, because the register count (and hence
the occupancy) does not move, and loads were never the binding constraint.

**Lesson: the btstep result does not generalise.** The same compiler defect costs
1.30x on a load-bound kernel and *nothing* on a latency/occupancy-bound one. Which
resource is binding decides whether a defect is worth anything. Consistent across
every size tested (0.946-0.976x; never a win).

## Numbers at production sizes

`<configs>/*.nml`. `nx_phys`/`ny_phys` are POINT COUNTS; columns =
`(nx_phys+6)*(ny_phys+6)` — the kernel sweeps ghosts too. Min of 6 interleaved
trials, quiet GPU (noise 1.01-1.08):

```
 config              columns      dc    dc_sig   cuda_f  cuda_t   dc/cf  dc/ct
 10km  108x137x30      16302   2.5561  2.6701   2.4940  2.7497   1.025  0.930
 0.1d  473x297x30     145137   6.4767  6.7032   6.1436  4.8827   1.054  1.326
 0.25d 240x560x30     139236   7.1738  7.3979   6.7451  5.1358   1.064  1.397
 0.05d 945x594x30     570600  19.1842 19.6504  18.3564 13.8693   1.045  1.383
 --- nz sweep at 0.1 deg, columns fixed at 145137 ---
 0.1d  nz=10          145137   2.5762  2.7224   2.2934  2.0401   1.123  1.263
 0.1d  nz=30          145137   6.4932  6.7069   6.1661  4.8748   1.053  1.332
 0.1d  nz=75          145137  15.0689 15.4728  14.7150 11.4702   1.024  1.314
```

- **`dc/cuda_faithful` is flat at 1.02-1.12x everywhere.** Codegen does not
  degrade with size or depth. Unlike the continuity kernels, there is no
  launch-overhead curve here — EPBL is ONE kernel launch, so there is nothing to
  amortise.
- **The occupancy win needs enough columns.** At the 10 km config (16302 columns)
  `cuda_tuned` **loses** (0.930x): there is not enough parallelism for extra
  warps to help, so the spills are pure cost. It pays from ~140k columns up.
- **Cost scales sublinearly in columns** (35x the columns → 7.5x the time,
  16302→570600), consistent with latency-bound: extra columns fill idle
  latency slots.

## Bit-identity — the disagreement is FMA, and that is PROVEN, not assumed

At production flags DC and CUDA **do not agree**: `max|d mld| = 4566 m`,
`max rel = 1.0` on `kd_int`. That looks exactly like a port bug. It is not.

**The proof (`make nofma`):** rebuild both sides with FMA contraction off
(`-Mnofma` / `-fmad=false`) and the disagreement collapses:

| | max &#124;d kd_int&#124; | max &#124;d mld&#124; | wet columns disagreeing |
|---|---|---|---|
| production flags (FMA on both) | 2.2e+01 | **4566 m** | 3292 / 129676 (2.54%) |
| **`make nofma` (FMA off both)** | **5.7e-12** | **1.4e-10 m** | **2 / 129676 (0.0015%)** |

Same source, same port, contraction the only variable. So the port is faithful
and **EPBL is genuinely, legitimately chaotic**: a sub-ulp difference flips the
static-stability short-circuit (`tot_tke <= 0 .and. stable`) at the mixed-layer
base, and `mld_output` then accrues one **whole layer** more or less — the output
is layer-quantised, so a 1-ulp input difference produces a 60 m output difference.
97.5% of columns agree to FMA precision; the movers move by ~one layer thickness.
The driver classifies this automatically rather than asserting it.

Other checks:
- **DC vs DC-signature-fix: `max|diff| = 0.0` exactly**, as a same-language
  source transform must be. The harness fails loudly if it is not.
- **CUDA-faithful vs CUDA-tuned agree with each other bitwise** — the disagreement
  with DC is systematic, not random.
- The residual ~1e-10 m under `nofma` is `x**(1.0/3.0)`: nvfortran's `pow` and
  nvcc's `pow` differ sub-ulp on the cube root. Library-level, not port-level.

**Do not "fix" this by loosening a tolerance.** Two real port bugs were found and
fixed this way: `(absf*htot)**3` and `**2`/`**5` in `epbl_find_mstar` are INTEGER
exponents in Fortran (expanding to multiplies) and were originally transcribed as
`pow()`, which is a libm call and does not give the same bits.

## Build / run

```bash
source ../../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
make && make run                      # 473 297 30 -- the profiled config
./epbl_bench 945 594 30               # [nx_phys] [ny_phys] [nz]
./epbl_bench 473 297 30 20 8          # [nreps] [nwarm]
./epbl_bench 473 297 30 20 8 1 1      # [cuda_sync] [dump MLD-iteration histogram]
./epbl_bench 473 297 30 20 8 1 0 1    # [mld_max_its] -- 1 removes the root-find
./epbl_bench 473 297 30 20 8 1 0 20 6 # [n_trials] -- min over interleaved trials

make collapse   # per-loop -Minfo: 1 collapse(2) + 3 sequential
make regs       # registers / spill stack / LDG per kernel (mangled names)
make sweep      # __launch_bounds__ occupancy sweep -- TESTS the hypothesis
make maxreg     # can Fortran reach the same occupancy? (no -- see above)
make nofma      # PROOF that the DC-vs-CUDA delta is FMA, not a port bug
make sizes      # the production configs

make native         # the native C++/CUDA driver: cudaMalloc + main(), no Fortran
make native-verify  # run both drivers; native checks itself against nvfortran's
                    #   own inputs AND outputs, elementwise
make native-tax     # MEASURE the OpenACC launch tax (2.8 us/launch -- see below)
make native-nofma   # native vs `do concurrent`, FMA off both sides -> ~1e-10
```

## Is the CUDA comparison FAIR to CUDA? Yes — the OpenACC launch tax is 0.05%

Every CUDA timing above is launched **from Fortran**: `epbl_bench.F90:fire`
wraps the call in `!$acc host_data use_device(...)` over **24 arrays**, and each
one is a runtime present-table lookup on the host, per launch. So the OpenACC
runtime sits on the CUDA path in every number this benchmark prints. If that
lookup costs anything, `dc / cuda_tuned = 1.329x` is an **understatement**.

`epbl_native.cu` settles it: `cudaMalloc` + `main()`, no Fortran, no OpenACC,
calling the **same** `epbl_cuda_launch()` from the **same** `epbl_kernel.o`.

**Comparing the two drivers at 473x297 cannot answer the question**, and this is
worth being explicit about because it is the trap the measurement was nearly
lost to. The kernel is ~6 ms; a per-launch host cost is microseconds; the signal
is ~0.05% and the GPU's contention noise is ~2x. So that comparison was
abandoned rather than reported as a tie — a tie there would have been a
statement about the noise, not about OpenACC.

Instead `make native-tax` makes the tax *the only thing left*: shrink the kernel
to nothing (8x8, nz=2, `mld_max_its=1`) and set `cu_sync=1` so
`cudaDeviceSynchronize` follows every launch and the host path cannot hide
behind the kernel. The two drivers are then identical except for how they launch:

```
  8x8x2, max_its=1, cu_sync=1, 3000 reps    us/launch
  fortran  (host_data over 24 arrays)         18.5
  native   (cudaMalloc)                       15.7      <- +-0.1 across 4 rounds
  delta                                       +2.8      = ~0.12 us per array
```

It is a **host** cost, which is why it holds still (±0.15 µs over 4 rounds)
while everything GPU-side on this node swings 2x. Against EPBL's ~6 ms/launch:

```
  2.8 us / 6000 us = 0.047%
```

**So every CUDA number in this benchmark is fair, and the 1.329x stands as
measured — it is not understated.** The tax is a *fixed per-launch* cost, so
this conclusion is about EPBL's launch granularity, not about OpenACC in
general: at ~6 ms/launch it is 0.05%, but the same 2.8 µs would be ~3% of a
100 µs kernel. Check it before assuming it transfers to a smaller kernel.

Secondary result: `epbl_native.cu` + `epbl_kernel.cu` + the bathymetry build
**with nvcc alone**. That is what `nvbug_*/` needs — NVIDIA cannot be asked to
install nvfortran to look at an nvcc codegen question, and the
`maxregcount`-vs-`__launch_bounds__` spill asymmetry (LOGBOOK §3 bug 4) needs a
C++ side to be a repro at all.

### The native driver cannot hand-mirror the init, and that is a finding

`epbl_native.cu` re-derives the whole state in C++ (bathymetry resample, zstar
stretch, T profile, 2gyre wind, SST restoring, Coriolis). It lands within
**~1e-14 relative** of nvfortran's — and **that is not close enough**, because
EPBL is chaotic: that last-ulp gap moves MLD by **4.5e3 m**, exactly as
"Bit-identity" below describes. Two causes, both isolated with a micro-test,
both in the **driver's host init** and neither anywhere near the kernel:

| symptom | cause | proof |
|---|---|---|
| `h_layer` differs ~1e-14 rel | nvfortran `-O3` **vectorises and thereby reassociates** the `denom`/`s` reduction loops; g++ won't without `-ffast-math` | `-Mnovect` reproduces g++'s bits **exactly** |
| `f_centre`, `tau_x` differ 1 ulp | `-fast` binds a **relaxed `sin`/`cos` from libnvf**, not glibc's | `-Kieee` reproduces glibc's bits **exactly** |

So the driver **reads nvfortran's own inputs** from `epbl_ref.f64`
(`./epbl_bench ... 1` dumps them) and runs those, making the state identical by
construction. Then the outputs **must** be bit-identical, and they are:

```
  [2] OUTPUTS vs the SAME kernel launched from Fortran, on the SAME inputs:
    mld    (faithful)      BIT-IDENTICAL
    kd_int (faithful)      BIT-IDENTICAL
    mld    (tuned)         BIT-IDENTICAL
    kd_int (tuned)         BIT-IDENTICAL
  VERDICT: PASS
```

The C++ init is still reported, as a **diagnostic, never a pass/fail**. It is
worth keeping because it independently corroborates the chaos claim *from the
opposite direction*: "Bit-identity" perturbs the **arithmetic** (FMA on/off) and
gets ~4.5e3 m of MLD movement; this perturbs the **input** by ~1e-14 relative and
gets the same ~4.5e3 m out. Same amplifier, different stimulus — so the chaos is
a property of the **scheme**, not an artifact of either toolchain.

`make native-nofma` closes the loop: contraction off on both sides collapses
native-vs-`do concurrent` to **1.4e-10 m**, the same figure `make nofma` gets for
`epbl_bench`. No tolerance was loosened anywhere.

## ⚠ The GPU this was developed on was SHARED — read this before re-running

The The HPC system analysis node's single V100 was simultaneously running another job from
this same investigation (`redi_bench`). **The same binary measured 4.9 ms and
23.5 ms in consecutive runs.** Several intermediate readings were pure artifacts
— including an apparent "16k columns and 145k columns cost the same", which
evaporated on a quiet GPU (10 km is 2.56 ms, not 6.42 ms).

The harness therefore:
- runs `n_trials` **interleaved** passes (dc, sig, cuda_f, cuda_t, repeat) so no
  variant can be favoured by a quiet window, and reports the **MIN** (contention
  only ever adds time);
- prints `worst/best` and **shouts `⚠ GPU CONTENDED`** above 1.15x.

Every number quoted in this README was taken at noise <= 1.08, most at <= 1.03.
Check that line before quoting anything.

## Build flags are load-bearing: NVHPC 26.5 / NVIDIA TPR #38714

Copied verbatim from `<model>/build/CMakeFiles/core_model_objs.dir/flags.make`:

```
-Mfree -Mbackslash -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate \
-gpu=tripcount:host -O3 -fast
```

`-gpu=tripcount:host` is mandatory — NVHPC 26.5 otherwise reads loop trip-counts
from the device copy and inserts a per-kernel data refresh. the ocean model guards it in
`cmake/compiler_flags.cmake:66-72`. It cost the btstep benchmark ~2x and
invalidated every finding downstream of it.

## Files

```
ocean_epbl.F90        The extract. `epbl_column_kernel` is spliced in
                          VERBATIM (production lines 818-1340, md5
                          33e307f821c1729434f05fbc8b33550e) and verified
                          byte-identical at assembly time. Its header lists
                          EXHAUSTIVELY what was removed (all host-side
                          scaffolding: the tracer-registry shim `epbl_compute`,
                          the enter/exit_data helpers, `ocean_epbl_bytes`) and
                          which knobs the profiled config leaves live.
                          One scaffolding change: the kernel is made public.
ocean_epbl_plain.F90  The signature-fix control. MECHANICALLY GENERATED --
                          identical body, plain explicit-shape dummies. Proven
                          bit-identical (0.0) to production.
epbl_stubs.F90        Stub types (grid/state/stress/flux/scratch/eos), each
                          cut to exactly the fields the kernel reads.
                          `eos_specvol_derivs` is itself a verbatim splice.
epbl_kernel.cu            CUDA port: `epbl_faithful` (1 thread/column, block
                          128, no launch_bounds -- isolates codegen) and
                          `epbl_tuned` (same body + __launch_bounds__ --
                          isolates occupancy). Documents why there is no
                          warp-per-column variant.
epbl_params.h             The EpblParams ABI + the epbl_cuda_launch prototype,
                          shared by epbl_kernel.cu and epbl_native.cu. It is a
                          header and not a third copy because the struct is
                          passed BY VALUE: a field reordered in one copy and not
                          the other is not a link error, it is silent numerical
                          corruption. epbl_bench.F90's bind(C) `epbl_params_t`
                          is a hand-maintained mirror and must track it.
epbl_bench.F90            Driver. Deep copy in the owning scope; separate output
                          state per variant; min-of-interleaved-trials timing;
                          contention detector; disagreement classifier;
                          MLD-iteration divergence histogram. Arg 10 = 1 dumps
                          epbl_ref.f64 (inputs + outputs) for epbl_native.
epbl_native.cu            The NATIVE driver: cudaMalloc + main(), no Fortran, no
                          OpenACC. Links the SAME epbl_kernel.o -- the kernel is
                          NOT duplicated (cf. daxpy_benchmark/daxpy_pure.cu).
                          Answers "is the CUDA comparison fair to CUDA?" and
                          builds with nvcc ALONE, for nvbug_*/. Re-derives the
                          state in C++, but RUNS nvfortran's dumped inputs when
                          epbl_ref.f64 is present -- see "Is the CUDA comparison
                          FAIR to CUDA?" for why the hand-mirror cannot be
                          bit-exact and why that is itself a finding.
gabight_bathy_0p1_473x297.f64
                          The REAL bathymetry from the profiled run
                          (gabight_bathy_sph_0p1_smooth.nc), 473x297 f64,
                          Fortran order. 10.9% land, mean wet depth 3983 m.
                          Regenerate with:
                            python3 -c "import netCDF4 as nc,numpy as np; \
                              b=np.array(nc.Dataset('<home>/analysis_gebco/gabight_bathy_sph_0p1_smooth.nc').variables['b'][:],dtype=np.float64); \
                              b.T.astype('<f8').tofile('gabight_bathy_0p1_473x297.f64')"
```

## State — what was used and why it is load-bearing

A column scheme's cost is a function of its stratification and forcing. A
trivially-stable or trivially-unstable column takes a fast path production never
takes, and — more importantly — a horizontally-uniform column makes **every
column converge in the same number of iterations**, which sets the MLD-iteration
divergence to artificially zero and measures the wrong kernel.

So the state is built to the profiled config
(`gabight_sph_acc_year_v100.nml`, which produced the 39.1% profile):

| | value | source |
|---|---|---|
| bathymetry | the real gabight field, 10.9% land | the nml's `bathymetry_file` |
| layers | zstar-like, ~5 m at surface stretched to bed | `zstar_h_surf_target = 5.0` |
| T | ~100 m mixed layer over an exponential thermocline | `T_init_surface/bottom = 14/2` |
| S | 35 uniform (so B0 is temperature-driven) | `initial_salinity = 35` |
| wind | MOM6 2gyre, `taux_mag = 0.15` | `&ocean_topo_nml`, transcribed from `ocean_surfstress_set_2gyre` |
| Q_heat | SST restoring to 14 degC, piston 1 m/day, **sign-varying** | `&ocean_restore_nml` |
| EOS | **linear**, rho0 1035, alpha_T 0.2 | run log "EOS variant: linear"; `&ocean_ic_nml` |
| dt | 300 s | `dt_fixed` |

**Two state choices are deliberate and load-bearing:**

1. **Surface T ramps with latitude** (~2 degC at the south wall to ~18 degC at the
   north). The profiled domain is the ACC sector (-60.6 to -30.9). Without this,
   every column has identical stratification and the divergence measurement is
   meaningless.
2. **The far-south band is weakly statically UNSTABLE.** This is not a fudge: a
   stably stratified column is always energy-limited (branch 8b) and can never
   mix deeper than the wind TKE pays for — ~110 m here. Production reports MLD
   up to ~5000 m, which is only reachable through branch **8a**, where mixing
   RELEASES PE and Kd is mixing-length-limited rather than energy-limited.
   Without this band **the 8a branch never fires and the benchmark measures half
   the scheme**.

Resulting state vs the profiled run — the driver prints this every run, so the
claim is checkable rather than asserted:

| | this benchmark | profiled run |
|---|---|---|
| MLD_EPBL mean (wet) | 87.2 m | 120 m |
| MLD_EPBL max | 4503 m | ~5000 m |
| MLD_EPBL min | 3.3 m | 0 (land) |
| land fraction | 10.7% | 10.9% (same file) |

Close enough to be credible; not identical, and not claimed to be. This is a
*plausible* production state, not a dump of one.

## Caveats

- **NOT a production state dump.** The state is constructed to match the
  profiled config's namelist and to reproduce its reported MLD distribution. A
  real 120-day-spun-up state would have different stratification and therefore a
  different iteration-count distribution — and the iteration distribution is
  precisely what sets the divergence tax and hence the cost. **The 6.49 ms and
  the 1.866x divergence tax are conditional on this state.** The ratios
  (dc/cuda) are far more robust than the absolute times: they held at 1.02-1.12x
  across every size, depth, and both state variants tried during development.
- **This measures EPBL, not `ocean_vmix_compute`.** That 39.1% region also wraps
  kappa-shear (a separate benchmark) and the PP81 interior closure. EPBL runs at
  thermo cadence (34560 calls) while the region is entered 69120 times. **The
  "1.33x on 39.1% ≈ 10% of runtime" arithmetic is an EXTRAPOLATION and assumes
  EPBL dominates the region — that assumption is untested here.** Do not quote
  the 10% without measuring EPBL's share of vmix.
- **`epbl_merge_into_kv_kt` is compiled but not benchmarked.** It runs every
  stage while `kd_int` refreshes at thermo cadence; it is a trivial
  `collapse(3)` elementwise loop and nothing suggests it is interesting.
- **The CUDA port is faithful, not tuned** (beyond `__launch_bounds__`): no
  shared memory, no `__ldg`, no restructuring of the k-loop's memory layout. The
  scratch arrays are `(nx,ny,nz)` so each k-step strides by `nx*ny` — a
  column-major-per-column layout (`(nz,nx,ny)`) would coalesce differently and
  is untested in BOTH languages. That is a plausible shared win and nobody has
  looked.
- **`dt_h` is read uninitialized in production** on one path: line 1261
  (`kddt_cur = kd_val*dt_h`) is reached from the static-stability short-circuit,
  which never assigns `dt_h` on that interface. It retains the previous
  interface's value in both languages, so it is benign *unless* the short-circuit
  fires at `ki = nz` on `obl_it = 1`. The CUDA port zero-initialises (the only
  defined choice). The `make nofma` agreement to ~1e-9 m is what proves the
  corner never fires in this state; **it is not proof that it cannot fire in
  production.** Worth a look independently of this benchmark.
- **Langmuir (`use_lt`) and `tke_diags` are compiled but dead** at the profiled
  config, in both languages, so register pressure and control flow are
  production's. The RH18 mstar/vstar branches are likewise compiled and dead.
- **⚠ WRONG GPU: this is a V100 (cc70); production profiles on an H200 (cc90).**
  Verified from the profile log: `host: <h200-node>`, `GPU 0: NVIDIA
  H200`, run from `build_hopper` with `-gpu=cc90`. This is the most important
  caveat in this file, because **the headline 1.33x is precisely the result most
  likely to move**:
    * It is an occupancy result, and occupancy only pays when a kernel is
      latency-starved. The V100 evidence for that is `dram__throughput = 20%` of
      a ~900 GB/s HBM2. An H200 has ~4.8 TB/s HBM3e and a far larger L2, so the
      same dependent load chain is cheaper to serve and the extra warps may buy
      much less — or more, since H200 also has 132 SMs to fill.
    * The 255-vs-128 register threshold is a property of the 65536-register file
      (identical on both), so the *occupancy buckets* should carry over — but
      what they are WORTH will not.
    * The 10 km result already shows the sign of this effect flipping with
      available parallelism (`cuda_tuned` LOSES at 16302 columns). A different
      SM count moves that crossover.
  The codegen tie (dc/cuda_faithful = 1.02-1.12x) and the signature-fix null
  result are structural and should be far more portable — but neither is
  confirmed on cc90. **Nothing here has been measured on the GPU that produced
  the 39.1%.**
- **Single-GPU, no MPI halo exchange.** Production runs the kernel over ghosts
  too (as here), but real runs pay a halo exchange this benchmark does not model.
