# Flux kernel — `do concurrent` vs faithful CUDA C, on the ocean model's real HLL solver

The daxpy MRE next door found `do concurrent` == CUDA C == 810 GB/s and concluded
"the compiler doesn't matter". **That conclusion does not survive a real kernel.**

Same experiment, but the kernel is the ocean model's production HLL Riemann solver —
`kernel_flux.F90`, copied **byte-for-byte** from
`<model>-sea-ice/src/core/coastal/kernels/structured/barotropic/`
(md5 `1ea40efad8567a85914561db5bfc3a55`). Not a paraphrase: the shipped file.

## Result (V100, cc70, nvhpc 26.5 / nvcc 12.9, 4096² interior, 2-D dam break, wet bed)

```
do concurrent (nvfortran -stdpar=gpu) :   7.8133 ms/rep
CUDA C  (faithful port, nvcc)         :   5.5737 ms/rep
ratio                                 :    1.402 x

max relative diff                     :   0.00000E+00     <- BIT-IDENTICAL
```

**Bit-identical output, 40% slower.** Zero difference means the port is faithful
and both toolchains chose the same FP semantics — so the 1.4× is *pure codegen*,
not different arithmetic. Nothing else in the comparison varies: same device
arrays (OpenACC-owned, handed to CUDA via `host_data use_device`), same input,
same process.

## Why — `ptxas` says it outright

```
nvfortran:  Function properties for kernel_flux_flux_cell_
              40 bytes stack frame, 40 bytes spill stores, 40 bytes spill loads
            Function properties for kernel_flux_minmod_          (also a real function)
            compute_flux_hll_45_gpu:  Used 4 registers               (an entry that calls out)

CUDA C:     Used 104 registers, 0 bytes stack frame, 0 spill stores, 0 spill loads
```

`-Minfo` agrees:

```
flux_cell:
    104, Generating implicit acc routine seq
minmod:
    380, Generating implicit acc routine seq
```

**nvfortran did not inline `flux_cell`.** It emitted it as a device-callable
routine with a 40-byte stack frame, so every cell pays a function call plus spill
traffic. The CUDA port marked its helpers `__forceinline__`, so the whole body
folded into the kernel: 104 registers, zero spill. That is the 1.4×.

### This is stronger than the documented gotcha

`CLAUDE.md` says:

> **Cross-TU helper inlining** (`_impl` performance variant): NVHPC's device
> codegen does not inline `pure !$acc routine seq` helpers **across module
> boundaries**.

But `flux_cell` and `compute_flux_hll` are **in the same module**, and it still
did not inline. The rule is broader than "cross-TU": nvfortran declines to inline
a sizable `pure` helper called from `do concurrent` **within a module too**.
Worth correcting in CLAUDE.md.

### Two hypotheses tested. Both WRONG. Do not skip this section.

The obvious reading of the `ptxas` output above is "the un-inlined call + 40-byte
spill is the 1.4×". **That is false**, and so is the runner-up. Both were tested:

**Hypothesis 1 — inlining `flux_cell` recovers the gap. FALSIFIED.**
`-Minline=levels:5` does inline it: the entry kernel goes from **4 registers** (a
stub that calls out) to **124 registers** (the body). And it gets *slower*:

```
baseline, flux_cell as a call     :  7.81 ms/rep     4 registers at the entry
-Minline=levels:5, body inlined   :  8.19 ms/rep   124 registers   <- WORSE
CUDA C                            :  5.63 ms/rep   104 registers, 0 spill
```

At 124 registers occupancy drops to ~16 warps/SM on sm_70, and that costs more
than the call it removed. **The call overhead was never the bottleneck.** The
`ptxas` spill numbers are real but they are not the story — a good reminder that
"I can see a plausible cost" is not "I have found the cost".

**Hypothesis 2 — nvfortran's launch geometry is the gap. FALSIFIED.**
nvfortran picks `threads(128)`, un-collapsed. Forcing the CUDA port through the
same and other geometries:

```
CUDA C block(256,1) : 5.35 ms      <- best
CUDA C block(64,4)  : 5.46 ms
CUDA C block(32,8)  : 5.65 ms      <- the default in this repo
CUDA C block(128,1) : 5.82 ms      <- matches nvfortran's thread count
CUDA C block(32,4)  : 6.43 ms      <- worst
do concurrent       : 7.81 ms      <- best nvfortran
```

Geometry moves CUDA C by ~20%, but **its WORST geometry still beats nvfortran's
best by 1.2×**, and at matched 128 threads it is 5.82 vs 7.81 = 1.34×. Geometry
is not the explanation.

### What it IS — `ncu` settles it: instruction-fetch starvation

**First, the framing that makes the rest make sense: this kernel is NOT
bandwidth-bound.** That is why it can distinguish compilers at all:

```
daxpy         : 810 GB/s = 90% of V100 peak BW   <- bandwidth-bound, everything ties
flux, CUDA C  : 219 GB/s = 24% of peak
flux, do conc : 160 GB/s = 18% of peak           <- nowhere near the memory wall
```

8 `sqrt` and ~12 DP divides per cell, four Riemann solves, long dependency chains.
The bottleneck is the **issue pipeline**, not DRAM — so instruction footprint and
scheduling decide it, and those are exactly where the toolchains differ.

`ncu`, same run, both kernels:

| | `do concurrent` | CUDA C |
|---|---|---|
| instructions executed | 1,107,755,008 | 1,070,123,264 |
| registers / thread | 120 | 104 |
| occupancy (warps active) | 24.72% | 23.44% |
| DRAM throughput | 21.0% | 23.6% |
| **SM throughput** | **51.4%** | **72.7%** |
| launch | `(4096,1,1)x(128,1,1)` = 524k threads | `(128,512,1)x(32,8,1)` = 16.7M threads |
| **cells per thread** | **32** | **1** |

`72.7 / 51.4 = 1.41` = the runtime gap, to two digits. Note what is NOT different:
instruction count (+3.5%), occupancy, DRAM. **nvfortran emits the same work and
issues it 1.41x less efficiently.**

Warp stall reasons (per issue-active):

| stall | `do concurrent` | CUDA C | |
|---|---|---|---|
| **no_instruction** (fetch / I-cache) | **3.34** | **0.65** | **5.1x** |
| long_scoreboard (memory latency) | 1.91 | 0.84 | 2.3x |
| wait (fixed-latency deps) | 2.83 | 2.79 | same |
| math_pipe_throttle | 0.79 | **1.43** | CUDA *more* |
| **issue_active** | **34.9%** | **45.3%** | |

**nvfortran's kernel is starved on instruction fetch.** CUDA C's dominant stall is
`math_pipe_throttle` — the *good* stall, meaning the math units are saturated.

The mechanism follows from the launch: nvfortran runs **32 cells per thread** in a
strided loop over a ~270-line body — a large code footprint hammering the I-cache,
and a stride-128 access pattern that also explains the 2.3x worse memory stalls.
The CUDA port runs one cell per thread: small body, contiguous access, latency
hidden by having 16.7M threads for the scheduler to interleave.

This also explains why `-Minline` HURT: inlining `flux_cell` makes the per-thread
body *bigger*, which is the wrong direction when the problem is fetch starvation.

### The lever to try next — NOT YET TESTED

Get nvfortran to launch **one thread per cell** instead of 32. On the daxpy it
auto-collapsed the DC loop (`-Minfo` said `auto-collapsed ... collapse(3)`); on
this kernel it did **not** collapse `do concurrent(j, i)`, choosing 128 threads
striding over `i` with one block per `j` instead. If the un-collapsed choice is
the root cause, forcing collapse should close most of the gap — and it is a
compiler/pragma question, not a rewrite.

### The implication for the ocean model

~35–40% on the mature coastal barotropic flux path, for a codegen reason, with no
known source-level fix — `-Minline`, the thing CLAUDE.md's `_impl` pattern
prescribes, makes it *worse* here. That matters because the `_impl` pattern is
documented as the remedy for exactly this shape, and on this kernel it is not.
Before applying `_impl` anywhere new on the strength of a `ptxas` reading, measure
it: this kernel says the reading can point the wrong way.

## What makes this kernel a real test (and daxpy not one)

| | daxpy | flux |
|---|---|---|
| bound by | memory bandwidth | codegen / occupancy |
| helper calls in DC | none | `flux_cell`, `minmod`, `extract_velocity`, `hll_flux_x/y` |
| locality clause | none | `local()` with 7 scalars |
| live doubles / cell | ~3 | ~40 |
| branches | none | minmod, dry-tolerance, HLL wave-speed selection |
| stencil | 1-point | 5-point (i-2..i+2) |
| **dc vs CUDA C** | **1.00×** | **1.40×** |

A bandwidth-bound elementwise op cannot distinguish compilers — everything ties at
the memory wall. Do not generalise from a daxpy.

## Build / run

```bash
source ../../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
make && make run
make regs          # CUDA-side register/spill counts

make ARCH=cc80 NVARCH=sm_80    # A100
```

Re-copy the kernel if the production file changes — a drifted copy would make
this benchmark measure a kernel that isn't the one that ships:

```bash
cp ../../<model>-sea-ice/src/core/coastal/kernels/structured/barotropic/kernel_flux.F90 .
```

## The native C++/CUDA driver — `flux_native.cu` (2026-07-16)

`flux_bench.F90` launches the CUDA port from Fortran and gets its device pointers
via `!$acc host_data use_device(...)` — a present-table lookup per array per call,
**9 arrays here, the most of any bench in the repo**. So the OpenACC runtime sat on
the CUDA path, and if that cost anything, every "CUDA buys N%" number here was
understated. `flux_native.cu` is `cudaMalloc` + `main()`, no Fortran, no OpenACC.
It **calls the same `flux_hll_cuda_launch()` from `flux_kernel.cu`** — the kernel is
not duplicated, so the two drivers cannot drift apart.

```bash
make native && ./flux_native            # or: make run-native  (dumps a ref, then verifies)
./flux_native 473 297 2 2000            # nphys_x nphys_y nghost reps
```

### The OpenACC bridge is FREE — measured, at every size

Idle-gated, paired, order-alternated (`./ab_idle.sh`), min over 8 clean samples:

```
                                        4096^2        473x297 (production)
  CUDA via Fortran driver (host_data)   5.4235 ms     0.0546 ms
  CUDA native (cudaMalloc, no OpenACC)  5.4360 ms     0.0541 ms
  ratio                                 0.998 x       1.009 x
```

Both **tie**, and the tie is not an artifact of the kernel being big enough to hide
the host cost: at 16² and 64² — where the kernel is ~10 µs and launches are
issue-bound, so 9 present-table lookups per call should be *maximally* exposed —
the two drivers agree to **four digits** (0.0098 vs 0.0098; 0.0100 vs 0.0100).

**So every CUDA number in this benchmark is already fair.** No repo conclusion
shifts because of the bridge. (A sibling bench reports the bridge costing
~2.8 µs/launch; that is not detectable here and the two need reconciling.)

### Verification: bit-identical, inputs AND outputs

`FLUX_DUMP_REF=1 ./flux_bench` dumps inputs + the CUDA-via-Fortran outputs;
`flux_native` reads them and compares **bit-for-bit**. Same kernel + same input must
give the same bits, so any difference is an *init* mismatch — which is the thing
worth proving, since a comparison against a differently-initialised state is
meaningless. All 9 arrays, 0 differing elements, at both 4096² and 473×297.
Inputs are checked separately from outputs so the two failures cannot be confused.

## ⚠ The harness was lying about the FLAT variants — fixed

`flat`, `flatdc` and `dims` all wrote `fh_fl`, and **`dims` runs last**. So the line
printed as `FLAT vs CUDA: 0 cells differ` was really **dims vs CUDA**, and `flatdc`
— the variant this README calls the winner — was never checked against anything. It
inherited dims' pass. This is exactly the trap `RESUME §4` documents; it had simply
grown a third victim, and `RESUME §4`'s "phantom 885197 cells differ" was **not a
phantom**. Every variant now owns its outputs. Corrected picture:

```
  do concurrent (shipped) : BIT-IDENTICAL to CUDA
  SIGNATURE FIX (dims)    : BIT-IDENTICAL to CUDA
  FLAT + collapse(2)      : max rel 4.1e-12   <- NOT bit-identical
  FLAT + PLAIN do conc.   : max rel 4.1e-12   <- NOT bit-identical
```

The FLAT variants agree to ~4e-12 (FMA-contraction level, not a port bug) but the
Makefile's "all bit-identical" claim is **wrong for them**.

## Why flat DC beats CUDA C: nvfortran CSEs 12 redundant FP64 divides

Not a coin flip. Idle-gated, order-alternated, 14 clean samples across two runs, the
distributions are **disjoint** — every flat-DC sample beats every CUDA sample:

```
  FLAT + PLAIN do concurrent : 5.193 - 5.224 ms
  CUDA C (native or Fortran) : 5.422 - 5.587 ms     -> flat DC ~4.4% faster
```

Static SASS, per kernel, matched to the right mangled symbol:

| | LDG | FP64-divide calls | SASS inst |
|---|---|---|---|
| kernel_flux (shipped, DC) | 69 | 36 | 2360 |
| kernel_flux_dims (sigfix) | 66 | 36 | 2424 |
| **kernel_flux_flatdc** | **56** | **24** | **2096** |
| flux_kernel.cu (CUDA C) | 56 | 36 | 2512 |

(LDG counts reproduce `RESUME §5a` exactly: 69 / 66 / 56 / 56.)

**flatdc issues 12 fewer IEEE FP64 divides than the CUDA port**, on a kernel whose
limiter is the FP64 pipe. `-Kieee` restores all 36 and costs 13% —
**5.19 → 5.99 ms, for bit-identical output** (verified: the dumps `cmp` equal). That
the bits do not move proves the 12 divides were **exactly redundant**: nvfortran
CSEs them bit-exactly, taking no FP latitude, and `-Kieee` merely disables a valid
optimisation. nvcc leaves them on the table and **has no flag that fixes it** —
`-prec-div=false` and `-use_fast_math` are float-only and change nothing here
(48 CALLs regardless); FP64 divide always takes the slow path.

With `-Kieee` the ranking **inverts** (CUDA 5.44 vs flatdc 5.99), which is what
makes the hypothesis testable and is why it is stated here rather than assumed.

**Hypothesis, NOT established:** that these 12 divides *cause* the 4.4%. The
correlation is consistent and the `-Kieee` swing (0.8 ms) is more than large enough
to cover a 0.23 ms gap, but the clean test — hand-hoist the redundant divides in
`flux_kernel.cu` and re-measure — is **untested**. Note the tension: doing so makes
the port no longer a faithful transliteration, the same caveat `LOGBOOK §0` attaches
to Redi. **A hypothesis that died on the way here:** that the win was nvfortran
taking unsafe FP latitude that nvcc refuses. It is not — the bits are identical.

## Inlining: `flux_cell` IS inlined — `RESUME §1` is confirmed

Settled without reading a single register count (the misread that cost `RESUME §1` a
day). All 48 CALLs in the shipped DC kernel entry, resolved via `nvdisasm`:

```
  36 __cuda_sm20_div_rn_f64_full
   8 __cuda_sm20_dsqrt_rn_f64_mediumpath_v1     <- matches "8 sqrt per cell"
   4 __cuda_sm20_dblrcp_rn_slowpath_v3
```

**Zero** calls to `flux_cell`, `minmod`, `hll_flux_x/y` or `extract_velocity`. The
control clinches it: the CUDA port, whose helpers are all `__forceinline__` and whose
object contains exactly **one** device function body, emits the **same 48** calls.
The CALLs are FP64 slow paths in both — they are not evidence of helper calls in
either.

`kernel_flux_flux_cell_` *does* exist as a standalone 3520-instruction body, but its
presence proves nothing: `acc routine seq` requires an emittable callable version.
Nothing calls it. The DC entry is 2360 instructions — a body, not a stub.

**So the Redi agent's "nvfortran inlines NONE of its 15 `!$acc routine seq` helpers"
does not generalise to this kernel** — here it inlines all of them. Both can be true
(Redi's helpers are called from `!$acc parallel loop`, these from `do concurrent`),
but the disagreement is unresolved and the two should be reconciled before either is
quoted as general. Bug 2's mechanism (`RESUME §5a`: "no CSE across the inlined call
boundary") **presupposes inlining happens, and inlining does happen** — so this
evidence *supports* the filed report rather than undermining it.

## Caveats — read before quoting the 1.4×

- **Launch geometry differs, and neither side was tuned.** nvfortran chose
  `threads(128)` on `threadidx%x` with `j` on blocks and did **not** collapse the
  2-D iteration space; the CUDA port uses `block(32,8)` = 256 threads on a 2-D
  grid. That is each toolchain's own choice, which is the honest comparison for
  "what do I get by default" — but part of the 1.4× may be geometry rather than
  inlining. Separating the two means forcing matched geometry, which is a
  follow-up.
- **The CUDA port's 104 registers cap occupancy** at ~6 warps/SM on sm_70. It
  wins anyway. A tuned version (shared-memory tiling for the 5-point stencil,
  `__launch_bounds__`) would likely widen the gap further — and would answer a
  different question than this one.
- **All-wet state**: no dry cells, so the `DRY_TOLERANCE` / `THIN_LAYER_THRESHOLD`
  branches never fire and warps do not diverge on them. This measures the kernel,
  not the wet/dry path. A wet/dry state is a separate experiment and would likely
  favour whichever side handles divergence better.
- Single V100, single run, `N_REPS=20`. The gap is large enough to survive noise,
  but it is not a statistical study.

## Files

```
flux_native.cu    NATIVE C++/CUDA driver: cudaMalloc + main(), no Fortran, no OpenACC.
                  Calls flux_hll_cuda_launch() from flux_kernel.cu -- kernel NOT duplicated.
ab_idle.sh        idle-gated, paired, order-alternated A/B. Up to ~9 agents share this
                  V100; a single unpaired run of each driver is worthless here.
kernel_flux.F90   the ocean model's PRODUCTION kernel, byte-identical (md5 1ea40efa...)
constants.F90     MRE stub: wp, GRAVITY, DRY_TOLERANCE, THIN_LAYER_THRESHOLD
grid.F90          MRE stub: hgrid_t (nx_total, ny_total, nghost, dx, dy)
flux_kernel.cu        faithful CUDA C transliteration of the HLL path
flux_bench.F90        driver: both kernels, same data, timing + agreement check
Makefile              note: -cuda is link-only; no trailing comments on ARCH
```
