# kappa-shear (JHL08) — `do concurrent` vs hand-written CUDA C

**`do concurrent` is 1.12x hand-written CUDA C as shipped, and 1.048x after two
changes that cost three lines of portable Fortran and one build flag.** Both
languages spill the column state to local memory in identical volume, both are
DRAM-bound on that spill traffic, and CUDA's only real edge is occupancy — most
of which nvfortran can be handed with `-gpu=maxregcount:96`.

```
 473x297x30 (production 0.1 deg), V100, min-of-4     dc_ms   cuda_ms  dc/cuda
 as shipped                                          39.57    35.36    1.119
 + -gpu=maxregcount:96          (one build flag)     37.44    34.87    1.074
 + bounded whole-array copies   (3 lines Fortran)    38.88    34.85    1.116
 + BOTH                                              36.52    34.84    1.048
```

**All four produce identical results** (`sum kd_int = 3.299882E+05` to every
digit). A hand-written CUDA C port of a 700-line iterative solver is worth the
remaining **4.8%**.

Measured, V100, nvfortran 26.5-0 / nvcc 12.9, production flags,
`NZ_STACK_MAX=128` (the production V100 build), min-of-N on an idle GPU:

```
 config (nxp x nyp x nz)     columns     dc_ms    cuda_ms  dc/cuda
 10km    (108x137x30)          16074     5.82      5.15     1.130
 0.1deg  (473x297x30)         145137    39.98     35.11     1.139   <- production
 0.25deg (240x560x30)         138636    38.56     34.91     1.104
 0.05deg (945x594x30)         570000   147.89    131.00     1.129
 --- nz sensitivity, 473x297 (nz is the SERIAL dimension) ---
 0.1deg  nz=15                145137    13.89     12.15     1.143
 0.1deg  nz=50                145137    82.85     72.43     1.144
 --- 25% land (columns that early-out on wet_mask) ---
 0.1deg  (473x297x30)         145137    36.94     32.16     1.149
```

**The ratio is flat at ~1.13 everywhere.** That is itself a finding: unlike the
continuity kernels next door, whose dc/cuda gap is a pure function of cell count
(1.61x at 145k cells → 1.03x at 17M, i.e. launch-overhead amortisation), this
kernel's gap does not move with problem size, with nz, or with land fraction. It
is not a launch-overhead story. It is a fat compute kernel and the ratio is a
property of the code generated, not of how often it is launched.

> ### ⚠ The profile and this benchmark ran on DIFFERENT HARDWARE
> The 39-41% `ocean_vmix_compute` figure comes from a run on an **H200**
> (`<h200-node>`, `build_hopper`, cc90, `NZ_STACK_MAX=256`). This
> benchmark runs on a **V100** (cc70, `NZ_STACK_MAX=128`) — ~900 GB/s vs
> H200's ~4.8 TB/s. **For a kernel that turns out to be DRAM-bound on spill
> traffic, that is the single most hardware-sensitive result possible.** The
> dc/cuda *codegen* ratio should port. The absolute times, the occupancy story,
> and anything resting on local-memory bandwidth **should not be assumed to**.
> Every "% of total runtime" below is therefore **extrapolated across
> architectures** and flagged as such. Someone should re-run this on the H200
> before acting on the arithmetic.

## Why this kernel: it is 18.5% of total runtime, on its own

`ocean_vmix_compute` was long quoted as "39-41%, EPBL + kappa-shear". A profiler
change split the region into its three closures for the first time
(`<model-root>/report2.nsys-rep`, gabight MEKE/REDI config):

```
  ocean_vmix_kshear   2.569 s   18.5% of total   <- THIS kernel. 87.5% of the region.
  ocean_vmix_epbl     0.244 s    1.8% of total
  ocean_vmix_pp81     0.080 s    0.6% of total
```

**kappa-shear is ~10x more expensive than EPBL.** The "vmix is 39-41%" figure was
essentially this kernel all along, not EPBL's — and this README's earlier
assumption of a 50/50 split (now removed) was wrong by 10x. It is the #2
addressable region in the model after Redi.

kappa-shear is also structurally unlike everything measured before it:
1443 production lines carry only **four** `do concurrent` loops. The parallelism
is one-thread-per-COLUMN, the work inside is sequential in k, and it **iterates
to convergence**. Every prior benchmark in this investigation was
one-thread-per-cell in both languages — which is why they tied.

### The synthetic state reproduces production's per-call cost

The obvious objection to a synthetic state for an iterative scheme is that it may
converge faster than the real thing and so measure the wrong regime. It does not.
`nsys` on the real run, same config this benchmark defaults to (473x297,
nghost=3, nz=30, dt_fixed=300 s — verified in `gabight_sph_meke_v100.nml`):

```
  production  kappa_shear_column_kernel_348_gpu, 72 instances
              avg 35.37 ms | median 38.13 ms | min 6.12 | max 54.59 | StdDev 13.67
  this bench  as shipped, min-of-4                        39.57 ms
```

**39.57 vs a production median of 38.13 — a 3.8% match.** The state lands in the
"expensive is the normal case" regime (production is left-skewed: median > mean),
which is the regime worth optimising. This is the strongest validation available
short of a restart file, and it was not tuned for — the state was built from the
namelist physics before these production numbers existed.

**What it does NOT reproduce: the variance.** Production spreads 6.12 → 54.59 ms
(9x, StdDev 39% of mean) where every other big kernel is a metronome; the code
owner attributes this to the iterative solver. In this benchmark **every column
takes exactly one outer substep** (`n_out = 1`, 100% of columns), so the adaptive
substep controller never engages, and per-call cost is essentially constant. So:
this benchmark measures production's **median call correctly** and its
**variance not at all**. The 54.59 ms tail is presumably calls where the
controller does engage — `ks_adaptive_dt` is the routine with the second-worst
spill ratio (25.7%) and would be exercised far harder there. **Untested, and the
most valuable thing a follow-up could do with a restart file.**

## The prime suspect died: divergence is a non-issue

The hypothesis was that a warp runs at its slowest lane, so a per-column
iteration count that varies would cost. **Measured, it does not.**

| | do concurrent | CUDA faithful |
|---|---|---|
| `smsp__thread_inst_executed_per_inst_executed` (of 32) | **29.92 → 93.5%** | 29.78 → 93.1% |

Two independent methods agree to three decimals. The driver also computes a
*static* warp work-balance estimate straight from the per-column Picard counts
(`sum(mean per warp) / sum(max per warp)`): **0.936**. Measured lane efficiency:
**0.935**.

The iteration-count distribution across all 145137 columns:

```
  outer adaptive substeps :  n_out = 1  for 100.00% of columns
  total Picard iterations per column:  min = 9   max = 24   mean = 19.75
  columns that mixed at all: 145137 (100.00%)
```

**Why divergence is absent, and why this is not an artifact of the state:** a
warp is 32 consecutive `i` values, i.e. 32 physically adjacent ocean columns.
The ocean state is spatially smooth, so adjacent columns have nearly identical
stratification and shear, and therefore nearly identical iteration counts. The
spread that exists (9→24) is between distant parts of the domain, not between
lanes of a warp. Adding 25% land — which makes a quarter of columns exit
immediately after the gather — moves the ratio from 1.139 to 1.149, i.e. not at
all.

### And it is dead on arrival from the source, independently of the timing

Even if divergence *were* the cost, warp-per-column could not fix it, for the
same reason the EPBL benchmark found for its k-sweep: **every k-sweep in
kappa-shear is a first-order sequential recursion** (a Thomas-algorithm
tridiagonal). Each reads index `k-1` to produce `k`:

```
  ks_projected_state  fwd   :  u_o(k)     = b1*(h_sd(k)*u0(k) + a_a*u_o(k-1))
  ks_projected_state  back  :  u_o(k)     = u_o(k) + c1_o(k+1)*u_o(k+1)
  ks_find_kappa_tke   TKE   :  tke_o(kk)  = bq_l*(hint_s(kk)*tsrc_l + aq_sc(kk-1)*tke_o(kk-1))
  ks_find_kappa_tke   kappa :  kappa_o(kk)= bk_l*(idz_s(kk-1)*kappa_o(kk-1) + ...)
  ks_solve_column     prestep: u_c(k)     = b1_l*(h_sd(k)*u_sd(k) + a1c*u_c(k-1))
  ks_solve_column     e1 tail: eden1_l    = hint_s(kk)*sqrt(...) + ome_l*eden2_l
  ks_precompute       dbot   : dbot(k)    = dbot(k+1) + h_sd(k)
```

There is **no per-k parallel work for `__shfl` to reduce**. The genuinely
parallel k-loops (`ksrc`, `tkedec`, `ild2`, N²/S², the copies and averaging) are
cheap element-wise passes, not the cost. A warp-per-column port would run the
recurrences on one lane with 31 idle, and would need a full parallel-cyclic-
reduction rewrite of the solver to do otherwise — which would change the
arithmetic and the convergence behaviour of a scheme whose whole point is the
iteration.

**Consequence: the warp-per-column variant was not built.** Two independent
reasons, one measured (93.5% lane efficiency: there is no divergence to
recover — at most ~6.5% even if perfect) and one structural (the work is
recurrences: `__shfl` has nothing to reduce). Building it on this evidence would
be motivated reasoning. The `Makefile` keeps a `WARP=1` hook and `ks_bench.F90`
a `-DHAVE_WARP` path. **This is an argued decision, not a measurement — if you
disagree, the hook is there.**

## What the cost actually is: DRAM-bound on SPILL traffic

`ncu`, 473x297x30, one launch each:

| metric | do concurrent | CUDA faithful |
|---|---|---|
| time | 40.59 ms | 36.61 ms |
| **registers / thread** | **254** (the sm_70 cap) | **80** |
| occupancy limit | 2 blocks/SM | 6 blocks/SM |
| **achieved warps active** | **11.95%** | **34.47%** |
| local ld sectors | 586.6 M | 531.8 M |
| local st sectors | 309.3 M | 286.4 M |
| **local ld+st (bytes)** | **28.7 GB** | **26.2 GB** |
| global ld sectors | 10.0 M | 8.1 M |
| **global ld (bytes)** | **0.32 GB** | 0.26 GB |
| DRAM bytes | 22.98 GB | 24.37 GB |
| achieved DRAM bandwidth | 566 GB/s (63% of peak) | 666 GB/s (74% of peak) |
| L1 hit rate | 30.6% | 18.0% |

Read the two bold rows in the middle together: **the kernel moves 28.7 GB of
local memory per launch against 0.32 GB of actual data — 90x more spill traffic
than data traffic.** kappa-shear is not computing; it is paging a 66 KB
per-thread column workspace in and out of DRAM.

**The decisive control — DC and CUDA spill the SAME amount** (895.9 M vs 818.3 M
local sectors, a 9.5% difference that the bounded-copy fix below accounts for).
That is what settles the language question: the *algorithm* demands this
workspace, not the compiler. Neither language can put 66 KB of runtime-indexed
per-column state into 255 registers. (The ALE-remap benchmark reached the same
conclusion by the same control: 33.4 M vs 31.5 M sectors, CUDA worth 1.03x.)

**And the headline: CUDA has 2.9x the occupancy and 3.2x fewer registers, and is
1.11x faster.** That is the whole answer. Occupancy is not the binding
constraint — DRAM bandwidth on the spill traffic is, and *both languages pay it
almost identically* (28.7 vs 26.2 GB). The extra occupancy only buys better
latency hiding, taking CUDA from 63% to 74% of peak bandwidth. There is no
codegen defect here and no Fortran expressiveness gap. The column state genuinely
does not fit in 255 registers **in either language**, and neither compiler has a
better option.

### Handing nvfortran the register cap: worth 5.4%, free

nvcc's kernel carries `__launch_bounds__(128)` and lands at 80 registers.
nvfortran, given nothing, takes all 254. Given the same instruction, it obeys —
**and gets faster**:

```
 nvfortran flag              REG   STACK    LDL+STL   dc_ms
 (none, as shipped)          254   66600     16743    39.64
 -gpu=maxregcount:128        128   66688     16858    38.97
 -gpu=maxregcount:96          96   66816     17030    37.49   <- optimum
 -gpu=maxregcount:64          64   66904     17637    39.47
 nvcc __launch_bounds__(128)  80   66512     (n/a)     34.8
```

Capping to 96 costs +287 local ops (+1.7%) and +216 B of frame, and buys enough
occupancy to be **5.4% faster**. Below 96 the spill starts to outrun the
occupancy gain — a textbook tradeoff, correctly navigated. **Note nvfortran will
not find this itself: left alone it picks the worst point on this curve.**

> **The EPBL "nvfortran register-allocator bug" does NOT reproduce here.** EPBL
> reported that at a matched 128-register cap nvfortran spilled 568 B / 10.95 ms
> where nvcc's `__launch_bounds__(128)` spilled 160 B / 4.87 ms. On kappa-shear
> nvfortran's allocator handles every cap gracefully (128/96/64 all build, spill
> grows smoothly and proportionately, and 96 is a net win). Whatever EPBL hit, it
> is not general to nvfortran under a register cap — **this is one negative data
> point against that hypothesis, on a different kernel; it does not refute EPBL's
> own measurement on its own kernel.**

### The spill is structural, and it is not the `routine seq` directive

nvfortran emits all six column-solve helpers as out-of-line ABI device
functions and **never inlines them**. From the real production object
(`<model>/build/.../ocean_kappa_shear.F90.o`), name-matched:

```
  ks_solve_column_        REG:238   LDL+STL=10719 / 42136 inst = 25.4%
  ks_solve_column_$8      REG:254   LDL+STL=11350 / 42096 inst
  ks_adaptive_dt_         REG:254   LDL+STL= 1379 /  5360 inst = 25.7%
  ks_find_kappa_tke_      REG:246
  ks_projected_state_     REG:211
  kappa_shear_column_kernel_348_gpu  REG:192  LDL+STL=859 / 9272 inst = 9.3%
                                     ... and 235 CALL.ABS.NOINC
```

**A quarter of every instruction in `ks_solve_column` is a local-memory access.**
For scale, every other kernel in this investigation has ~zero spills (HLL flux
~2400 inst / 0; layered continuity ~400 inst / 0; the whole barotropic substep,
42 kernels / 4). kappa-shear is the only one that spills, and it does so by
~200x.

> ⚠ The figure "859 spills" is the **entry kernel only** — 8% of the total. The
> real traffic is in `ks_solve_column`: 10,719 local ops on its own. Quoting 859
> understates it by ~12x.

Three hypotheses about the cause, **all tested, all negative**:

| hypothesis | result |
|---|---|
| "`!$acc routine seq` forces the out-of-lining" | ❌ **FALSE** — stripping every `routine seq` marker and rebuilding leaves the helpers as separate functions at the identical REG:238/254/246/211. |
| "`-Minline` will fix it" | ❌ **FALSE** — `-Minline`, `-Minline=reshape`, and `-Minline=levels:5` all leave `ks_solve_column_`/`ks_adaptive_dt_`/`ks_find_kappa_tke_` out of line at REG:246/254/246. Only the entry kernel's own count moves (120→142→164). |
| "the per-column arrays live in registers" (the production source says so) | ❌ **FALSE** — see below. |

### A source-comment defect

`ocean_kappa_shear.F90:299-301` states:

> Per-thread work is fixed-size `local()` arrays (L1 layout — **all live in GPU
> registers, no shared memory**)

`-Minfo=stdpar` on the shipped code says otherwise, unprompted:

```
116, Local memory used for tke_avg_sd,v_sd,u_sd,s_sd,kappa_avg_sd,
     idz_int_s,hint_s,h_sd,idz_s,il2_s,t_sd
```

They cannot live in registers: the arrays are indexed by a runtime loop variable
and there are ~111 of them across the call tree. The comment is wrong and is
worth fixing — it is the kind of claim that stops someone looking in the one
place that matters.

## NZ_STACK_MAX: a real but small effect (~2-3%), not a cliff

`NZ_STACK_MAX` sizes every per-column array at COMPILE time. Production V100
builds with 128; `CMakeLists.txt:88` **defaults to 256**; the deepest config in
`~/analysis_gebco/*.nml` is nz=50 and the common case is nz=30. So the frames are
2.5-5x larger than any config needs. Measured at nz=30, 473x297, min-of-4:

```
 NZ_STACK_MAX   dc_ms    cuda_ms   dc/cuda   local frame (CUDA)
     32         38.69    34.78     1.113      17360 B/thread
    128         39.60    34.75     1.139      66512 B/thread   <- production V100
    256         40.76    35.17     1.159     132048 B/thread   <- CMake default
```

**Results are bit-identical across all three** (`sum kd_int = 3.299882E+05`,
`max rel = 1.98657E-09` — identical to the last digit). Nothing reads past nz+1.

This **confirms the Redi agent's null result**: 32 → 256 is a 5.4% effect on
`do concurrent` and 1.1% on CUDA, not the 2x the frame ratio suggests. Local
traffic scales with what the code *touches* (bounded by the runtime `nz`), not
with the declared extent.

> ⚠ **Two phantom findings died here, and they nearly shipped.** An early
> unrepeated run read "NZ_STACK_MAX=32 is 2x slower" (80 ms vs 40 ms) and a later
> one read "a 2.2x cliff at 256" (88 ms vs 40 ms). Both were **GPU contention** —
> this V100 is shared with three sibling benchmarks driven by other agents, and a
> bare `./ks_bench` gave 120.7 / 77.5 / 135.5 / 40.8 ms for the *same binary*.
> Every number in this README comes through `measure.sh`, which gates on an idle
> GPU and reports min-of-N. **Do not time this kernel with a bare `./ks_bench`.**

### The one portable-Fortran lever the data points to

`do concurrent` is 4x more sensitive to `NZ_STACK_MAX` than CUDA is (5.4% vs
1.1%), and there is exactly one reason. `ks_solve_column` contains three
**whole-array assignments** — `kappa_out = kappa` and `kq_tmp = k_q` (x2). These
are declared `(NZLI)`, so Fortran copies all 129 elements at NZ_STACK_MAX=128
while only 1..nz+1 = 31 are ever read. It is the only place in the routine that
touches the full declared extent — which is precisely why wall time depends on
NZ_STACK_MAX at all. The CUDA port copies 1..nz+1 and shows no sensitivity.

`make BOUNDEDCOPY=1` replaces them with bounded loops (results unaffected —
elements above nz+1 are never read):

```
 473x297x30, NZ_STACK_MAX=128, min-of-3        dc_ms    cuda_ms   dc/cuda
 as shipped                                    39.63    34.70     1.142
 + bounded copies (3 lines)                    38.75    34.69     1.117
 as shipped, NZ_STACK_MAX=32 (for reference)   38.70    34.81     1.112
```

**The prediction and the result match exactly.** Bounding the copies at
NZ_STACK_MAX=128 (38.75 ms) reproduces NZ_STACK_MAX=32's time (38.70 ms) to
within 0.1%. CUDA is unmoved (34.70 vs 34.81) because it never had the problem.
So the three whole-array assignments are **the entire** NZ_STACK_MAX
sensitivity — mechanism confirmed, not inferred. Worth **2.2%** of the kernel,
and it makes the build flag irrelevant.

Results are unaffected, verified rather than assumed: `sum kd_int = 3.299882E+05`
and `max|d| = 6.48842E-12` are **identical to the last digit** with and without
the fix.

**Recommendation, in priority order:**
1. **Add `-gpu=maxregcount:96`** for this translation unit. One flag, 5.4%,
   results identical. ⚠ It is V100-tuned: the optimum is a
   registers-vs-spill-vs-occupancy tradeoff and **must be re-swept on the H200**
   (different register file, different bandwidth) before being applied there.
2. **Bound the three whole-array assignments** (`make BOUNDEDCOPY=1`). Three
   lines, 2.2%, results identical, portable, and it makes NZ_STACK_MAX
   irrelevant. Together with (1): **1.119x → 1.048x**.
3. **Do NOT rewrite kappa-shear in CUDA C.** After (1) and (2) it is worth 4.8%
   of a kernel that is part of a 39-41% region — see the arithmetic below — in
   exchange for a permanently hand-maintained C port of an iterative solver with
   an adaptive substep controller.
4. **`-DMODEL_NZ_STACK_MAX=128` at configure** is worth ~3% *if* (2) is not
   done; (2) makes it moot. The CMake default is 256 for a deepest-config nz=50.
   **Cap caveat: 128 covers nz ≤ 128 (Redi needs `2*nz+2`, so nz ≤ 63), and
   `config.F90:2484` only WARNS on overflow, it does not error.**
5. **The real lever, unmeasured: reduce the column workspace.** See below.

## The arithmetic that decides it

⚠ **Extrapolated across architectures**: measured on a V100, the profile ran on
an H200. For a DRAM-bound kernel that is the most hardware-sensitive translation
possible. Treat as an order-of-magnitude argument.

kappa-shear is **18.5% of total runtime** (measured, not assumed — see above).

| action | wins on the kernel | of total runtime |
|---|---|---|
| `-gpu=maxregcount:96` + bounded copies | **7.7%** | **~1.4%** |
| ...then a full CUDA C rewrite on top | 4.8% | ~0.9% |

For context, the siblings found 1.35-1.46x from *algorithmic* fixes on their
kernels. Nothing of that size was found here: the 1.12x total gap to CUDA simply
is not that big, and two thirds of it is recoverable in portable Fortran. **If a
1.35x exists in kappa-shear it is in the algorithm — the redundant-workspace
leads below — not in the language.** The arithmetic does not support a C
rewrite, the same conclusion the layered-continuity benchmark reached for a
completely different reason (there nvfortran was already at 90% of CUDA; here
both are equally stuck against DRAM).

## Where the headroom actually is (identified, NOT measured)

The kernel moves **28.7 GB of spill traffic against 0.32 GB of data**. Neither
language can fix that — but the *algorithm* can, and that is where the remaining
performance lives. This mirrors what the Redi benchmark found: its win was 1.35x
from **removing redundant per-column work** (local-mem ops fell 17x), and it beat
the CUDA port; occupancy was unchanged and registers went *up*.

`ks_solve_column` declares ~40 NZLI-sized work arrays and `ks_adaptive_dt`
another 10. Candidates visible by inspection — **none of these are measured, they
are leads**:
- `n2p/s2p/n2c/s2c`, `tke_pred/kappa_pred/kappa_mid/tke_fin/kappa_pred2` — the
  predictor and corrector each carry their own full-column buffers; several look
  aliasable.
- `ks_adaptive_dt`'s `tol_max/tol_min_a/tol_chg_a` are recomputed from
  `kappa_src_s`/`local_src_s` on every outer substep, and its `u_pr/v_pr/t_pr/
  s_pr/c1_pr/n2_pr/s2_pr` duplicate `ks_projected_state`'s outputs.
- `ks_projected_state` is called up to 11 times per outer substep purely to
  *choose* `dt` (a halving pass then a 5-step refinement). In the measured state
  the halving pass accepts on the first try, so this costs ~1 call — **but that
  is a property of this state, and a shear-heavier state would pay all 11.**
  Untested.

**The contrast worth stating:** EPBL — the other half of the same profiler
region — took the opposite design decision *deliberately*.
`ocean_epbl.F90:282`: "no per-column work arrays, no NZ_STACK_MAX locals",
carrying everything as scalars plus `(nx,ny,nz)` scratch buffers. Same region,
opposite strategy, and EPBL does not spill. Whether kappa-shear can be
restructured the same way is the open question, and it is worth more than any
language choice.

## Agreement: NOT bit-exact, and here is why

```
  DC vs CUDA-faithful  max|d| = 6.48842E-12   max rel = 1.98657E-09
                       68909 cells > 1e-12 rel
```

This is **not** bit-exact, unlike every sibling benchmark. That is expected for
this kernel and the driver says so loudly rather than loosening the threshold.
The diagnosis:

- **`max|d| = 6.5e-12` against a field max of 1.65** — i.e. ~4e-12 relative to
  the field scale. The 2e-9 figure is small-value amplification in the ratio, not
  a large absolute error.
- **The iteration counts are NOT flipping.** This is the load-bearing argument.
  The Picard convergence test is `|dk| <= tol_err*(...)` with **`tol_err = 0.1`**
  — a 10% tolerance. If a 1-ulp FMA difference had flipped a convergence
  decision, that column would stop one iteration earlier or later and differ by
  **O(10%)**, not by 2e-9. No cell shows anything remotely like that. So every
  column takes the same number of iterations in both, and the residual is pure
  FMA-contraction drift compounded through ~20 Picard iterations of a contraction
  map — which is exactly what ~1e-9 after 20 iterations of a 1e-16 seed looks
  like.
- The `sum`/`max`/`min` diagnostics and the full iteration histogram are printed
  every run; a genuine port bug would move them.

### The FMA test — run, and only partly confirming

The decisive check is to remove FMA contraction and see whether the difference
goes away. It **does not**:

```
 473x297x30                                        max|d|       max rel     cells>1e-12
 production flags (baseline)                      6.48842E-12  1.98657E-09    68909
 + FMA off both (-Mnofma / -fmad=false)           6.55211E-12  4.55491E-10    18590
 + nvfortran -Kieee (drops -fast)                 5.70391E-12  2.36657E-10    19078
 + nvcc -prec-div=true -prec-sqrt=true explicit   5.70391E-12  2.36657E-10    19078
```

So: **FMA contraction accounts for ~4.4x of the drift, and nvfortran's `-fast`
(relaxed div/sqrt) for another ~1.9x — but a 2.4e-10 residual survives both, and
I did not identify it.** `sum kd_int` is unchanged to 7 digits throughout, and
nvcc's div/sqrt were already IEEE (the last row changes nothing), so the residual
is on the nvfortran side or in operation ordering. It is consistent with last-ulp
differences amplified through ~20 Picard iterations of a contraction map, but
**that is a plausibility argument, not a demonstration.**

**Honest statement: this is an argument from the magnitude of the discrepancy,
plus the FMA test above. It is not a proof.** Instrumenting the Fortran with the
same per-column iteration counters the CUDA port carries would settle it
directly, and was not done.

## Is the CUDA number unfair to CUDA? No — `host_data` costs 0.004%

Every CUDA figure above is launched **from Fortran**, and gets its device
pointers from `!$acc host_data use_device(...)` — a runtime present-table lookup
per array per call, **11 arrays, every launch**. So the OpenACC runtime sits on
the CUDA path inside the very number used to claim `dc/cuda = 1.119x`. If that
lookup costs anything, CUDA is understated and the comparison is rigged against
it. `ks_native.cu` settles it: `cudaMalloc` + `main()`, no Fortran, no OpenACC,
linking **the same `ks_kernel.o`**.

```
  CUDA via Fortran driver (host_data)  : 34.76 ms     min-of-5, idle-gated
  CUDA native (cudaMalloc, no OpenACC) : 34.81 ms
  ratio                                :  0.999 x     <- a TIE
```

**They tie.** Two independent measurements agree:

1. **Paired full-size runs** (6 pairs, both halves verified uncontended): mean
   delta **+0.08 ms (+0.23%)**, range **−0.14 .. +0.40 ms** — *straddling zero*.
   Native is not reliably faster or slower.
2. **A direct probe of the launch cost.** The full-size run *cannot* resolve this
   effect — so it was measured where it is visible. At `1 1 2` (49 columns, nz=2)
   the kernel is trivial and per-rep time is essentially launch cost:

   ```
     cuda_fortran (host_data)  : 17.1 us/launch     min of 29 paired samples
     native       (cudaMalloc) : 15.8 us/launch
     delta                     :  1.3 us/launch     ~0.12 us per array lookup
   ```

`host_data` costs **~1.3 µs/launch**, which against the 35.36 ms production rep
is **0.0037%** — and the observed noise floor (0.54 ms) is **417x larger** than
the effect. That is why the full-size ratio is a tie and always would be: the
instrument cannot see 1.3 µs, so a full-size A/B was never the right instrument.

**No conclusion in this repo shifts.** Every "CUDA buys N%" number here is
already fair; the OpenACC runtime is not on the critical path in any measurable
sense. The kernel is DRAM-bound on spill traffic (28.7 GB local vs 0.32 GB
global) — a per-launch host cost 4 orders of magnitude below the rep time was
never going to matter, and now that is measured rather than assumed.

The native driver also **confirms the port is exact**, which the Fortran driver
could not: run on the Fortran's own input bits it returns `kd_int`
**bit-identical** to the `host_data` path (0 of 4.5M cells differ). So the two
paths are the same computation, and the tie is a statement about cost.

### What it does NOT change

- Native reproduces `regs=80, local=66512 B/thread` exactly — same kernel object,
  so **the spill story is untouched**. DC and CUDA still spill equally; the
  algorithm demands the workspace.
- `n_out = 1` for 100% of columns here too: the native driver mirrors
  `ks_bench`'s state, so it **inherits** the caveat that this reproduces
  production's median (39.6 vs 38.1 ms) but not its 9x variance. A native driver
  cannot fix that — it is a property of the state, not of who owns the memory.
- The `dc`-vs-CUDA agreement is unchanged at **max rel 1.99e-9**, reproduced
  exactly by the native driver at the same 1e-12 tolerance (not loosened).

### Why the C++ state mirror is not bit-identical, and why that is fine

`ks_native.cu` re-implements `build_state` in C++. It agrees with the Fortran to
**~1 ulp of each array's scale** but *not* bit-for-bit, and it cannot: nvfortran
`-fast` and glibc round `sin/cos/tanh/exp` differently. This was measured, not
assumed — `f_centre` is just `|2*Omega*sin(lat)|`, one `sin()` with no arithmetic
to contract, and it still differs by 1 ulp.

Two traps, both hit:

- **A per-element relative bound is the wrong instrument.** `u = us*0.5*(1 -
  tanh((zm-mld)/25))` cancels *by construction* deep in the column, where the jet
  has decayed. One ulp of `tanh` there becomes a **2.7e-10 relative** error on a
  velocity of ~1e-7 m/s — while `max|d|` is 1.7e-15, one ulp of the jet's actual
  scale. The gate is therefore `max|d|/max|ref|`, not per-element relative.
- **Same ulp-level inputs do not automatically mean the same work.** This scheme
  iterates to convergence, so 1 ulp could in principle flip a convergence test
  and change the timing. Checked, not assumed: **2866522 vs 2866522** total
  Picard iterations, **0 of 145137 columns** with a different count. The libm
  noise flips nothing, so the native timing times the same problem.

`make native-verify` runs all of this and exits non-zero on failure.

## Build / run

```bash
source ../../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
make && make run                       # 473 297 30 -- the 0.1 deg config

./measure.sh "0.1deg" 4 473 297 30 20 10 1   # ⚠ USE THIS for any timing
./ks_bench 945 594 30                  # [nxp] [nyp] [nz]
./ks_bench 473 297 30 20 10 1 25       # [reps] [warm] [cuda_sync] [land_pct]

make native                            # cudaMalloc + main(), no Fortran/OpenACC
make native-verify                     # ⚠ the correctness gate for `native`:
                                       #   inputs + kd_int vs the Fortran run,
                                       #   bit-for-bit. Exits non-zero on fail.

make NZSTACK=32                        # per-column stack bound (prod V100 = 128)
make BOUNDEDCOPY=1                     # bound the 3 whole-array assignments
make collapse                          # -Minfo: collapse + "Local memory used for"
make regs                              # LDG / LDL / STL per kernel, name-matched
make ncu                               # occupancy + LANE EFFICIENCY + local traffic
make sweep                             # all production configs
```

`nxp`/`nyp` are **interior** cells; `nx_total = nxp + 2*nghost`, nghost=3. The
parallel width is `nx_total*ny_total` columns (145137 at 0.1 deg) and nz is the
**serial** dimension.

## Files

```
ks.F90        the kernel. Six column-solve helpers VERBATIM from production
                  (:474-1430). The column kernel around them is a TRANSCRIPTION
                  -- see its header for the three dropped pieces and why each is
                  dead in the gabight configs.
ks_state.F90  MRE stubs: hgrid_t, multilayer_cgrid_state_t, ocean_eos_t +
                  eos_specvol_derivs. Field names/defaults verbatim.
constants.F90 wp/GRAVITY/H_VANISHED/NZ_STACK_MAX, verbatim values.
ks_kernel.cu      faithful CUDA C port, one thread per column -- the same thing
                  `do concurrent` does, to isolate codegen. Keeps Fortran's
                  1-based indexing throughout. Carries the per-column iteration
                  counters the divergence analysis needs.
ks_bench.F90      driver. mem:separate + manual deep copy; both toolchains read
                  the SAME device allocation via host_data use_device.
                  `KS_DUMP=<path>` dumps inputs + kd_int(dc) + kd_cu for
                  ks_native.cu to verify against. Off by default (~250 MB).
ks_native.cu      NATIVE driver: cudaMalloc + main(), no Fortran, no OpenACC.
                  Answers "does host_data cost anything?" -- it does not (0.004%).
                  Links the SAME ks_kernel.o; the kernel is NEVER duplicated,
                  so the two binaries cannot silently drift apart. Mirrors
                  ks_bench's build_state in C++ and CHECKS that mirror against
                  the Fortran rather than trusting it -- see `make native-verify`.
measure.sh        ⚠ the only trustworthy way to time this. Idle-gates the GPU,
                  min-of-N. Two phantom findings died to its absence. Times
                  ks_native too when it is built, with its own idle-gate (a
                  sibling starting BETWEEN the two binaries hits only the
                  second one and fakes a host_data cost -- seen, twice).
results.txt       raw output behind the tables above.
```

## Caveats

- **The merge path is dropped**, so this benchmark's per-thread frame is
  ~9 KB SMALLER than production's. Every frame/spill number here is a **lower
  bound** on production's. The ratio should be unaffected (both languages drop
  it equally), but that is an assumption.
- **The state is synthetic**, not a restart file. It is built to straddle
  Ri_c = 0.25 (see `ks_bench.F90`'s header) because a too-stable state converges
  in one iteration and would fake a tie — and the run confirms it earns its keep
  (100% of columns mix, 9-24 Picard iterations). But the iteration-count
  distribution, and therefore the divergence conclusion, **is a property of this
  state**. A restart from a real gabight run would settle it. `n_out = 1` for
  100% of columns means the adaptive substep controller never engaged here;
  a state that forces multiple substeps would exercise `ks_adaptive_dt` far
  harder, and that is the routine with the second-worst spill ratio.
- **All-wet by default.** `land_pct=25` is offered and measured (1.149 vs 1.139
  — no effect), but a real coastline is neither 0% nor a rectangle.
- **The CUDA port is faithful, not tuned**: no `__ldg`, no shared-memory
  staging, no tiling. Tuning it further is a different question from "does
  nvfortran's codegen keep up?" — though the ncu data says there is little to
  tune toward, since both are against the DRAM wall.
- **EOS is linear** (the gabight default), so `eos_specvol_derivs` returns a
  constant and its branch is free. A Wright-97 or Roquet run would add real
  per-interface arithmetic. The `WRIGHT_97` branch is retained but untested here.
- `dc/cuda` at 0.25deg (1.104) is the low outlier and I have no explanation for
  why that config specifically sits 3% below the others. It reproduces.
