# Optimizing the barotropic-substep CUDA kernel

A best-CUDA-practices pass over `btstep_kernel.cu` (the faithful 11-kernel-per-
substep transliteration), freed from the faithfulness rule but held
**bit-identical** to the faithful port (same per-cell arithmetic order).
`ab_main.cu` runs both launchers from an identical seed for the same substep
count and checks `eta`. V100, nvcc 12.9, `n_inner = 24` substeps/call.

The check is **field-relative**: `bt_eta` is divergence-driven and its interior
cells are ~0, so `max|d eta|` is judged against the field magnitude `max|eta|`
(the same bar `btstep_bench.F90` / the repo LOGBOOK use), not pointwise.

## Result — bit-identical, faster

Over **5040 substeps** (210 calls) from a fresh seed, `max|d eta| == 0.0`
exactly — the optimized path is not "close", it is byte-for-byte the faithful
result. Both the fused and the graph variant.

| grid (interior) | faithful | opt-fused (5 kern) | opt-graph (5 kern, 1 graph) | max\|d eta\| |
|---|---|---|---|---|
| 473×297 | 3.148 ms | 2.601 ms — **1.21×** | 2.458 ms — **1.28×** | 0 |
| 236×148 | 1.263 ms | 0.833 ms — **1.52×** | 0.725 ms — **1.74×** | 0 |

`|eta|` field magnitude ≈ 0.26 (473×297) / 0.32 (236×148), so field-rel error is
0.0. The speedup grows on smaller grids, where the per-substep launch count
dominates — exactly the continuity_layered pattern.

## What worked

Two changes, both *removing genuine waste* rather than being clever:

1. **Fuse 11 kernels/substep → 5.** Walls and the eta/u/v/-sum accumulators fold
   into their producers as single-assignment merges
   (`ubt = wall ? 0 : computed; ubt_sum += ubt`; `eta = eta_new; eta_sum += eta`).
   Same values, same order — so bit-identical — but 6 separate launches and their
   global round-trips per substep vanish (264 launches/call → 120). **This is the
   dominant win.** The merges are legal because each is single-assignment and
   `eta`/`ubt`/`vbt` are already final at the fold point.
2. **Capture the whole `n_steps` loop into ONE `cudaGraph`** (opt_mode 1). The
   120 per-call launches are issued once at capture and replayed with ~0 host
   cost. Same kernels, same work, same values — only the launch overhead leaves
   the timed loop. Buys the extra 1.21×→1.28× (and 1.52×→1.74× small-grid). This
   is the CUDA-only lever `do concurrent` cannot express.

### The 5-kernel floor is the dependency structure, not laziness
Checked, not assumed (same as the faithful port documents):
* **Pass 2b → Pass 2c is sequential** — 2c's `u_at_v` reads the `ubt` 2b just
  wrote (Gauss-Seidel, not Jacobi). Fusing them changes the answer.
* **Pass 1 → eta-swap → Pass 2b are separated by real barriers** — Pass 2b reads
  `eta`/`zeta` *neighbours*, so every producer must have landed first.

## What was tried and LOST / was neutral

| idea | result | why |
|---|---|---|
| **32-bit indexing** (`-DIDX32=0` isolates it) | **neutral**: 2.60 vs 2.60 ms | These arrays are 2-D and small (145k); the kernels are memory-/launch-bound, not address-bound, so cutting `size_t`→`int` addressing frees nothing on the critical path. (It won 1.06× for continuity's *3-D* layered arrays, where the address math was deeper.) Kept as the default — harmless, and correct-by-construction for any realistic grid — but it is **not** where the time goes here. |
| Fusing 2b+2c into one kernel | rejected (not built) | Gauss-Seidel dependency above — would change `eta`. |
| Fusing pass1+swap+2b | rejected (not built) | neighbour reads across a real barrier. |

## Reproduce

```bash
nvcc ab_main.cu opt_kernel.cu btstep_kernel.cu -O3 -arch=sm_70 -I../common -I. -o ab
# ALL GPU runs go through the shared-V100 lock:
bash ../tmp_local_artifacts/gpu_run.sh btstep ./ab 473 297 24 200 10
bash ../tmp_local_artifacts/gpu_run.sh btstep ./ab 236 148 24 200 10   # small grid
# isolate idx32:  nvcc -DIDX32=0 ...  (opt-fused time is unchanged -> idx32 neutral)
```

`opt_kernel.cu` is a standalone best-CUDA artifact exporting
`btstep_opt_launch(BtArgs*, n_steps, opt_mode)` — `opt_mode 0` = fused (5 kern),
`opt_mode 1` = fused captured in a cudaGraph. The faithful `btstep_kernel.cu`
stays untouched as the codegen-comparison baseline.

## DC (do concurrent) optimization

The **portable subset** of the CUDA win — loop fusion only — ported back to
Fortran `do concurrent`. The cudaGraph is not portable and is omitted; the
fusion is, and is the dominant lever. `btstep_nonlinear_closed_fused`
(alongside the untouched faithful `btstep_nonlinear_closed`) collapses the
**11 loops/substep → 5**, each fusion a single-assignment merge that keeps the
per-cell arithmetic order, so it is **bit-identical** to the faithful routine.
Driver `drivers/dc_ab.F90` (target `make dcab`) runs both launchers from the
identical seed, resetting the evolving device state between them, and checks
`bt_eta` field-relative.

### What fused (all bit-identical, same per-cell order)
1. **prev-save u + prev-save v → one loop.** The two BEBT prev-save copies
   ((1:ny,1:nx+1) and (1:ny+1,1:nx)) ride one guard-merged sweep over the
   (1:ny+1,1:nx+1) superset — disjoint independent copies.
2. **eta-swap + zeta walls + eta_sum.** `eta_sum += eta` folds into the swap:
   it reads the `bt_eta` just written in the same cell, and `bt_eta` is untouched
   for the rest of the substep, so the accumulated value is identical.
3. **Pass 2b + u-wall + ubt_sum.** One sweep over all east faces (1:nx+1):
   `ubt = wall ? 0 : computed; ubt_sum += ubt`. Stays its own loop — Gauss-Seidel:
   Pass 2c reads this `bt_ubt` (a real barrier), exactly as the faithful routine.
4. **Pass 2c + v-wall + vbt_sum.** Symmetric.

Pass 1 (with its ride-along KE + interior zeta) is left untouched.

### Result — bit-identical, faster (V100, nvfortran, n_inner=24, 200 reps)

| grid (interior) | faithful DC | fused DC | speedup | max rel diff |
|---|---|---|---|---|
| 473×297 | 3.674 ms/rep | 2.899 ms/rep | **1.268×** | 0.0 |
| 236×148 | 1.588 ms/rep | 0.928 ms/rep | **1.710×** | 0.0 |

`max rel diff == 0.0` (byte-for-byte). The DC fused **beats the CUDA fused**
(1.27× vs 1.21× large grid; 1.71× vs 1.52× small grid) — no cudaGraph needed —
consistent with `NOTES_ON_PERF.md`. The win grows on smaller grids where the
per-substep launch count dominates, the same launch-bound signature as CUDA.

**CPU-portable: yes.** `make dcab DATA=none` builds and runs under
`-stdpar=multicore` (any F2018 `do concurrent … local(…)` compiler); bit-identical
there too (max rel 0.0, and ~1.6× faster on the CPU at 60×40). Only portable
transforms used: loop fusion, wall-as-branch, register locals — no shared memory,
no launch_bounds, no cudaGraph.

## Head-to-head: opt-CUDA vs opt-DC (shared host_data driver)

The DC and CUDA sections above were timed in **two separate binaries** — fused-DC
in the nvfortran `dc_ab` harness, fused-CUDA in the nvcc `ab` harness — two seeds,
two processes, the shared V100 at different moments. Their absolute ms/rep are
therefore *not* directly comparable, which is why the verdict there leaned on the
**ratio-of-ratios** (each fused vs its own faithful baseline: 1.27× DC beat 1.21×
CUDA). This section removes that caveat.

`drivers/cmp_main.F90` (target `make cmp`, run `make run-cmp`) puts **both**
optimized endpoints in **one binary on one device truth**:
* opt-DC   = `btstep_nonlinear_closed_fused` (do concurrent, 5 loops/substep);
* opt-CUDA = `btstep_opt_launch_flat(opt_mode=0)` (5 kernels/substep, fused),
handed the **same device arrays** the DC routine uses via `!$acc host_data
use_device` (bridge `drivers/btstep_bridge.F90` → flat wrapper
`btstep_opt_launch_flat` in `opt_kernel.cu`, which assembles `BtArgs` C-side). No
copies between the two; the state is reset to the identical seed between the DC
timing and the CUDA timing (both evolve `eta`/velocity). Agreement is
field-relative (`max|d eta| / max|eta|`), the documented `bt_eta` bar.

### Result — fused vs fused, apples-to-apples (V100, n_inner=24, 200 reps, 473×297)

| endpoint (same device arrays) | ms/rep | vs opt-DC |
|---|---|---|
| opt-DC   (fused do concurrent)     | 2.86–2.88 | 1.00× |
| opt-CUDA (fused, 5 kern)           | 2.67      | **1.07–1.08× faster** |
| opt-CUDA (fused-in-cudaGraph)      | 2.52–2.54 | 1.13–1.14× faster *(CUDA-only)* |

`max|d eta| = 3.09e-15`, field-rel **1.17e-14** (< 1e-12) — the two optimized
paths are byte-for-byte the same result up to FMA contraction (nvfortran vs nvcc
codegen). The headline is **fused-vs-fused**; the cudaGraph line is a CUDA-only
lever `do concurrent` cannot express and is reported separately.

**Reading:** on one truth, hand-CUDA fused is a genuine but **single-digit %**
ahead of `do concurrent` fused (~7–8%), and the graph adds ~6% more — consistent
with `NOTES_ON_PERF.md` ("a full rewrite buys single-digit % of wall time"). The
earlier ratio-of-ratios verdict ("DC beats CUDA") was an artifact of comparing two
harnesses; the honest same-binary comparison puts CUDA modestly ahead, but well
inside the margin the portable-Fortran fusion already captured.

## Caveats

* Shared V100 (7 sibling agents) → ms/call is noisy at the ~1% level; the
  speedup ratios are stable across repeats (1.207–1.213× fused, 1.280–1.281×
  graph at 473×297) because both sides are timed back-to-back under the GPU lock.
* Bit-identity is a property of *these inputs run on one device* — the fusion and
  idx32 are numerically inert by construction (single-assignment merges; address
  math only), so `max|d eta| == 0` is the expected, not lucky, result.
* The graph caches on `(n_steps, nx, ny)`; a size change reinstantiates it. The
  first call after a change pays capture+instantiate once (outside the timed
  window here, as production would keep the grid fixed).
