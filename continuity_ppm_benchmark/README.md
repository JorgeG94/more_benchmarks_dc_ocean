# continuity-PPM benchmark — does the ocean path pay the flux kernel's 1.4×?

**Yes — ~1.3-1.7x at the sizes production actually runs.** An earlier revision of this
file said "clean"; that was measured at 4096², a size no config uses. Benchmarked at
the real grids in `~/analysis_gebco/*.nml` (V100, nvfortran 26.5-0, `cuda_sync=2`):

```
 production config        cells     dc_ms    acc_ms   cuda_ms  dc/cuda   dc/acc
 10km    (108x137)        14796    0.1273    0.1706    0.0745    1.709    0.746
 0.1deg  (473x297)       140481    0.1597    0.2280    0.0998    1.600    0.700
 3km     (359x458)       164422    0.1613    0.2396    0.1049    1.538    0.673
 0.25deg (240x560)       134400    0.1527    0.2279    0.1026    1.488    0.670
 0.05deg (945x594)       561330    0.2629    0.4079    0.2022    1.300    0.645
 MRE-only (4096x4096)  16777216    3.8272    6.8876    3.6144    1.059    0.556
```

All bit-identical (0 cells differ; DC vs CUDA max |diff| = 0.0 exactly).

**The ratio is a function of domain size, and every production config sits at the
expensive end.** 4096² is the only size that looks clean, and nothing runs there.
Always report the size with the ratio.

### But it is a DIFFERENT defect from the flux kernel's 1.4x

The coastal HLL kernel loses 1.4x to **codegen** (lost collapse + lost CSE). This
kernel's codegen is fine — both compiler bugs are absent (see below), and at 256² its
x-PPM *kernel* is 5.87 us against the CUDA port's 5.31 us, ~10%. The 1.3-1.7x is
almost entirely **~10 us of dead host time between each of the 9 loops**: plain
`do concurrent` under `-stdpar=gpu` blocks the host after every loop; the CUDA port
queues all 9 and returns. As the domain shrinks, that fixed cost dominates — hence
1.06x at 4096² and 1.71x at 108x137.

So the fix is not flattening or a signature change. It is **launching fewer times, or
not synchronising between launches**. Which is exactly what `!$acc kernels async(1)`
is for — and it does not work here:

## ⚠ The OpenACC wrapper spills registers (bug 3?)

Production already wraps DC loops in `!$acc kernels async(1)` to avoid per-loop syncs
(`<model>/.../barotropic_substep.F90:1369`, "single drain"). Applied to *this*
kernel it backfires — same source, same `-Minfo` report (`auto-collapsed ...
collapse(2) vector(128)`), very different register allocation:

| x-PPM loop | registers | spills (LDL+STL) | theor. occ | kernel time |
|---|---|---|---|---|
| plain `do concurrent`          | **56** | **0**  | 56.25% | **5.87 us** |
| + `!$acc kernels async(1)`     | 90     | 16     | 31.25% | 13.41 us |

Bisected — **`async` is not the culprit, the compute construct is**:

| construct | spills |
|---|---|
| plain `do concurrent`                            | **0** |
| `!$acc kernels async(1)`                         | 16 |
| `!$acc kernels` (no async)                       | **16** |
| `!$acc parallel loop gang vector collapse(2) async(1)` | **33** |

So you cannot buy async here: getting it requires an OpenACC construct, and on a
register-heavy loop that construct costs more than the syncs it saves. That is why
`dc/acc` is 0.64-0.75 at every production size — the async variant is 1.3-1.6x
*slower* than the plain DC it was meant to rescue.

**Mechanism: UNKNOWN.** One hypothesis is dead already: that `do concurrent`'s
`local()` clause states privatisation in terms the OpenACC path re-derives worse.
Adding an explicit `!$acc loop independent collapse(2) private(dh_m1, dh_0, ...)`
to the x-PPM loop changes **nothing** — 16 spills, 400 instructions, identical to
without it. Whatever the construct does to register allocation, it is not about
privatisation. (A single-loop probe is not a valid test bed for this: extracted on
its own, even plain DC spills 16. The 56-reg/0-spill baseline only exists in the
full 9-loop subroutine, so the bisect must be done on the real file.)

On the *cheap* loops production wraps (`eta_sum(i,j) = 0.0`) there is little to
spill, so this does not indict `barotropic_substep.F90`'s use of the idiom —
**but any register-heavy wrapped loop is worth checking.** Not yet reported to
NVIDIA; no minimal repro yet.

### Where that leaves the fix

Neither lever available today works: plain DC pays 9 syncs, and the only way to
drop them (an OpenACC construct) costs more than it saves. Untried options, roughly
in order of promise:
- **Fuse loops.** 9 launches -> fewer. The 4 one-dimensional boundary loops are
  0.4% of the work but a fixed ~10 us each; folding them into their neighbours is
  pure win and needs no compiler cooperation.
- **CUDA graphs** to amortise launch cost, if nvfortran can be made to emit them.
- **Get NVIDIA to fix the spill**, which would make `async(1)` the one-line answer.

## Why it dodges both compiler defects

| defect (see ../RESUME_GPU_MRE.md) | trigger | present here? |
|---|---|---|
| **bug 1** — silent loss of auto-collapse | `grid%`-bounded explicit-shape **dummy** + a call in the body | **No.** Arrays are derived-type *components* (`bs%h`), not dummies. All five 2-D loops report `auto-collapsed ... collapse(2)`. |
| **bug 2** — lost CSE across the inline boundary | the **callee** does its own array indexing | **No.** The caller hoists every read into a scalar (`h0 = bs%h(i,j)`) and hands the PPM helpers *scalars*. There is no array inside the callee to reload. |

The kernel makes **five calls per cell** (`ppm_limited_slope` ×3, `ppm_cell_limiter`,
`ppm_limit_pos`) — it is not "inline arithmetic", as an earlier session recorded. It is
protected by the *scalar-hoisting idiom*, not by an absence of calls. That idiom is
exactly the cheap fix §5b proposes for the coastal HLL kernel.

## Why FLAT loses — an occupancy cliff, not instruction count

The surprise: FLAT emits **fewer** instructions and **fewer** loads, and is still slower.

| x-direction PPM loop | LDG | SASS inst | registers | blocks/SM | theor. occ | achieved |
|---|---|---|---|---|---|---|
| production (calls)   | 28  | 416 | **56** | **9** | 56.25% | 52.95% |
| FLAT (inlined)       | 21  | 368 | **58** | **8** | 50.00% | 44.58% |

Two registers. 56 → 58 crosses the sm_70 threshold where blocks-per-SM drops 9 → 8, and
the 8% occupancy loss outweighs the 12% instruction saving. FLAT's extra registers come
from hoisting `wet_T(i±1,j)`/`wet_T(i±2,j)` into named scalars — which is also why it
issues 7 fewer loads. **Do not read the 28-vs-21 gap as bug 2**: it is a source-level
difference this control introduces, not an inline artifact.

## Why the flux kernel bled and this one doesn't — and how far that generalises

Redundant loads appear to cost wall-clock only when there are too few warps to hide their
latency. Two data points:

| | occupancy | redundant loads | wall-clock cost |
|---|---|---|---|
| coastal HLL flux (`../hll_fluxes_benchmark`) | **25%** (126 regs) | 66 vs 56 (+18%) | **22%** |
| ocean continuity PPM (here) | **53%** (56 regs) | — | none |

At 25% the flux kernel has 4 warps/scheduler and stalls on every extra load
(long-scoreboard +53%); at 53% there are enough warps to cover them.

**⚠ Two kernels is not a law.** These two are *not* representative of the ocean model's kernel zoo,
and the differences that matter here (register pressure, occupancy, call idiom, stencil
width) vary wildly across it. The HLL solver in particular is a simple, arithmetic-dense,
register-hungry kernel; continuity is branchier, cheaper per cell, and touches nested
derived-type state. The barotropic subset, ePBL, and tracer advection are different again —
ePBL especially (column-wise, sequential-in-k, tiny parallel width) shares almost nothing
structurally with either. **Nothing here predicts them.** What transfers is the *method*
(`-Minfo` for collapse, LDG counts for CSE, `ncu` for occupancy), not the verdict.

The one durable conclusion: **"flatten the body" is not a general recipe.** It is the right
call for a register-starved, low-occupancy kernel and the wrong one here — it *lost* 15%.
Profile before flattening.

## Build / run

```bash
source ../../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
make && make run                       # 4096^2, 1000 reps, 4 variants
./continuity_bench                     # same as `make run`
./continuity_bench 1024 1024           # [nx] [ny]
./continuity_bench 1024 1024 5000 20   # [nreps] [nwarm]
./continuity_bench 1024 1024 1000 10 0 # [cuda_sync] 0=async 1=per-rep 2=per-kernel
make collapse                          # per-loop -Minfo
make regs                              # LDG + instruction counts per kernel
```

`cuda_sync=2` is the default and the honest one: it makes the CUDA port pay the same
per-kernel host sync `do concurrent` pays, so the ratio reflects codegen. Use `0` to
see what a fully pipelined hand-written CUDA implementation could do — a real
speedup available to CUDA, but not a compiler comparison.

Warm-up matters: a single warm-up call left a cold-run bias big enough to invert the
verdict (one 4.15 ms outlier against a 3.83 ms steady state). Default is 10 untimed.

`-gpu=mem:separate` + manual deep copy, same as the flux bench. The state is nested
allocatable derived types, so each payload needs its own `enter data` *after* its
parent's (`copyin(cont%h_face_left_x)` then `create(cont%h_face_left_x%data)`);
without those, `-Minfo`'s "Generating implicit copyin" fires on every call and the
benchmark times PCIe rather than the kernel. This is also what lets the CUDA port
read the *same* device allocation via `host_data use_device` — verified working for
3-level components. (An earlier revision used `mem:managed` to dodge the deep copy;
it works, but managed memory has no distinct device pointer to hand to CUDA.)

Trap, already paid for once here: **`-cuda` must be link-only.** It also switches
nvfortran into CUDA Fortran mode, which type-checks device attributes and rejects the
`host_data` bridge with `NVFORTRAN-S-0528-... device attribute mismatch`.

## Files

```
continuity.F90        production extract, VERBATIM (see its header for line ranges)
                          + MRE stubs for metrics/bs/scratch/continuity_t.
                          Only scaffolding change: ppm_limit_pos made public.
continuity_flat.F90   FLAT control — helpers hand-inlined. Tests "do the CALLS
                          cost?" An EXPERIMENT, not a proposal (and it loses).
continuity_acc.F90    ACC control — production body + `!$acc kernels async(1)`.
                          Tests "is the gap the LAUNCH MODEL?" (it is, but the fix
                          costs more than it saves — see above). Not `pure`, the one
                          deviation from the verbatim body; production's async
                          substep is not pure either.
continuity_kernel.cu      faithful CUDA C port, 9 kernels, one per Fortran loop.
continuity_bench.F90      driver. Separate state per variant (see RESUME §4 harness
                          trap); read-only inputs deliberately shared.
```

## Caveats

- **All-wet** (`wet_T = 1`), so `ppm_mirror_h` and the slope-flatten multiply by 1 and the
  land branches never fire. Matches `flux_bench.F90`'s choice; a coast would exercise
  divergence. Untested.
- **`use_ppm_limit_pos = .false.`** (production default) — `ppm_limit_pos` never executes.
- Timings are single-run on a shared analysis node. DC is stable to ~0.3% run-to-run
  (3.820-3.849 over 6 runs at 4096²); CUDA to ~1.5%. Treat sub-2% differences as noise.
- The CUDA port is faithful, not tuned: no shared-memory tiling, no `__ldg`, no
  `launch_bounds`. A tuned port would likely widen the gap; that is a different
  question from "does nvfortran's codegen keep up?".
```
