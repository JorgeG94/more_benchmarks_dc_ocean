# nvfortran: `do concurrent` silently loses auto-collapse when an array dummy is bounded by a derived-type component

**Component:** nvfortran `-stdpar=gpu` (`do concurrent` → GPU lowering)
**Version:** nvfortran **26.5-0** (also reproduced on 25.5), CUDA 12.9, Tesla V100-PCIE-32GB (cc70), Linux x86-64
**Severity:** silent performance regression — correct answers, no diagnostic, 1.2× on a minimal case and measurable on a production kernel
**Repro:** `repro.f90`, self-contained, 176 lines, no dependencies

## Summary

A `do concurrent(j, i)` nest is normally auto-collapsed to one thread per iteration.
nvfortran **silently stops collapsing** — mapping `j` to thread blocks and `i` to 128
threads, so each thread serially walks `n_i/128` iterations — when **both** of these hold:

1. an **explicit-shape array dummy is bounded by a derived-type component**, e.g.
   `real(wp), intent(in) :: a(grid%nx_total, grid%ny_total)`, **and**
2. the loop body contains a **procedure call**.

Either condition **alone** is fine. Only the conjunction triggers it. Results are
bit-identical; only the schedule changes. There is no `-Minfo` note, no warning, nothing.

This matters because `type(grid_t) :: grid` + `a(grid%nx_total, grid%ny_total)` is the
standard Fortran idiom for passing a grid descriptor alongside its arrays, and calling a
`pure` per-cell routine from the loop is the standard way to keep a kernel readable.

## Reproducer

```
$ nvfortran -O2 -stdpar=gpu -gpu=cc70,mem:separate -Minfo=stdpar,accel -c repro.f90
```

| | array dummy bounds | DC body | `-Minfo` |
|---|---|---|---|
| **A** | plain `nx, ny` dummies | `call cell(...)` | ✅ `auto-collapsed … collapse(2)` |
| **B** | `a(grid%nx_total, …)` | inline | ✅ `auto-collapsed … collapse(2)` |
| **C** | `a(grid%nx_total, …)` | `call cell(...)` | ❌ **no collapse** |
| **D** | plain dummies, `grid` also passed | `call cell(...)` | ✅ `auto-collapsed … collapse(2)` |

```
a_plain_call:
     45,   ! blockidx%x threadidx%x auto-collapsed
         Loop parallelized across CUDA thread blocks, CUDA threads(128) collapse(2)
b_dt_inline:
     61,   ! blockidx%x threadidx%x auto-collapsed
         Loop parallelized across CUDA thread blocks, CUDA threads(128) collapse(2)
c_dt_call:                                    <-- no collapse, no diagnostic
     78, Loop parallelized across CUDA threads(128) ! threadidx%x
         Loop parallelized across CUDA thread blocks ! blockidx%x
d_workaround:
     96,   ! blockidx%x threadidx%x auto-collapsed
         Loop parallelized across CUDA thread blocks, CUDA threads(128) collapse(2)
```

```
$ nvfortran -O2 -stdpar=gpu -gpu=cc70,mem:separate repro.f90 -o repro && ./repro
grid 4096 x 4096, 50 reps
  A  plain bounds  + call    :    0.3733 ms   [collapses]
  B  grid% bounds  + inline  :    0.3667 ms   [collapses]
  C  grid% bounds  + call    :    0.4476 ms   *** NO COLLAPSE ***
  D  workaround (plain dims) :    0.3815 ms   [collapses]
  C is   1.20x slower than A (same maths)
  max |difference| between all four:  0.000E+00  (0 = identical)
```

`D` is the point of interest: it passes `grid` **and** plain `nx, ny`, uses `grid%nghost`
for the loop bounds, and calls `cell` — and collapses. Only the array *dummy declaration*
matters.

## What we have NOT established

**We do not know the mechanism.** We report the trigger, not the cause. Specifically:

- **It is not "calls can't be lowered".** Variant A has a call and collapses. On the
  production kernel below, the signature fix collapses while the callee is still a call.
- **It is not inlining.** We initially believed the callee went un-inlined in the bad case.
  That was a misreading of `ptxas` output — the `Used 4 registers` line we attributed to the
  kernel is `cub::EmptyKernel<void>`. Checked properly, the kernel entry uses **120
  registers in the non-collapsed baseline** and **116–128 in the collapsed variants**: the
  callee is inlined in *all* of them. Inlining does not distinguish the cases.
- **`!$acc routine seq` on the callee is irrelevant** — explicit or implicit, collapsing or
  not (`probe6.f90`).

## `-Minline=reshape` also restores the collapse

```
$ nvfortran -O2 -stdpar=gpu -gpu=cc70,mem:separate -Minline=reshape -c repro_mod.f90
c_dt_call:
     78,   ! blockidx%x threadidx%x auto-collapsed          <-- restored
         Loop parallelized across CUDA thread blocks, CUDA threads(128) collapse(2)

  C  grid% bounds  + call    :    0.3664 ms   (was 0.4431)   -> 1.00x vs A (was 1.18x)
```

`reshape` = "allow inlining in Fortran even when array shapes do not match". That the
*shape-mismatch* switch is the one that fixes a *collapse* decision is the strongest hint we
have at the cause, and is why we suspect the `a(grid%nx_total, …)` → `a(nx, ny)` shape
relationship is what the collapse analysis is choking on.

Plain `-Minline`, `-Minline=name:cell`, `-Minline=levels:5`, `-Minline=smallsize:1000` and
`-Mautoinline` do **not** help. Only `reshape` does.

**Possible second bug:** applying `-Minline=reshape` to a file containing both the module
and a `program` that calls it fails with

```
NVFORTRAN-S-1074-Procedure call in Do Concurrent is not supported yet   (repro.f90:126)
```

at call sites the compiler accepted without the flag. Compiling the module separately works
(`repro_mod.f90` / `repro_main.f90`).

## Impact on a production kernel

An HLL Riemann solver (shallow-water, 2-D structured, `pure` per-cell routine called from
`do concurrent`, dummies declared `h(grid%nx_total, grid%ny_total)`) — variant C's shape —
at 4096², V100:

```
as shipped (grid% bounds + call)          7.02 ms    NOT collapsed
signature fix (plain nx/ny dummies)       6.75 ms    collapses
body hand-flattened, no call at all       5.24 ms    collapses
hand-written CUDA C (faithful port)       5.52 ms
```

All bit-identical. Two observations we cannot yet separate:

- The signature fix restores the collapse but recovers only ~4% here, so **collapse alone is
  not the whole cost** on this kernel.
- The flat version is 1.34× the shipped one and beats hand-written CUDA C, but changes two
  things at once (collapse + no call), so it does not isolate either.

`ncu` on the collapsed-vs-not pair of the flat kernel shows the schedule difference is real
and is an issue-pipeline effect, not memory: same instruction count (+3.5%), same occupancy,
same DRAM throughput (~21% of peak — the kernel is arithmetic-bound: 8 `sqrt` + ~12 DP
divides per cell), but

| | not collapsed | collapsed |
|---|---|---|
| stall `no_instruction` (per issue-active) | **3.34** | **0.12** |
| SM throughput | 51.4% | 62.3% |
| launch | 524k threads, **32 cells each** | 16.7M threads, 1 cell each |

i.e. 32 cells/thread over a large body starves the warps on instruction fetch.

## What we'd like

1. **Diagnose it.** A `-Minfo` note — "loop not collapsed: array dummy `a` bounded by a
   derived-type component" — would turn this from a multi-day investigation into a one-line
   fix. Today the failure is invisible: right answers, no warning.
2. **Collapse anyway.** `a(grid%nx_total, grid%ny_total)` where `nx == grid%nx_total` is the
   ordinary grid-descriptor idiom. If some analysis is being defeated by the bounds
   expression, the runtime extents are identical and the collapse should still be legal.
3. **Explain the `reshape` connection**, if it is not simply a bug — it is surprising that an
   inlining/shape switch changes a loop-schedule decision.

## Workarounds

- **`-Minline=reshape`** — one flag, no source change, restores the collapse. (See the
  `S-1074` caveat above.)
- **Pass the extents as plain integer dummies** and declare the arrays from those
  (variant **D**). Fixes the collapse; on our real kernel that recovered only part of the
  loss.
- **Write the body inline, no call** (variant **B**). Recovered the full loss for us, but it
  is a large and unpleasant source change.

## Files

```
repro.f90       the reproducer: 4 variants + timing + bit-identity check
repro_mod.f90   module alone, repro_main.f90  program alone  (for the reshape demo)
probe.f90       trivial call                          -> collapses
probe2.f90      array-passing call; conditional write -> collapses
probe3.f90      nested calls; 40-local body           -> collapses
probe4.f90      pure driver; derived-type arg         -> DT-bounded dummy is the trigger
probe5.f90      the isolation: bounds x call          -> the conjunction
probe6.f90      explicit !$acc routine seq            -> irrelevant either way
```

Build everything with:
`nvfortran -O2 -stdpar=gpu -gpu=cc70,mem:separate -Minfo=stdpar,accel`
