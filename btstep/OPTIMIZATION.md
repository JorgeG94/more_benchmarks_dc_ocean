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
