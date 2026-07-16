# Optimizing the MEKE-step CUDA kernel

A best-CUDA-practices pass over `meke_kernel.cu` (the faithful 16-kernel
transliteration of the `do concurrent` MEKE step), freed from the faithfulness
rule but held **bit-identical** to the faithful launcher. `ab_main.cu` checks
`max|diff| == 0` on the primary output field `meke`. V100, gabight config
(473×297×30, degenerate nml as in `meke_bench.F90`).

## Result

**0.239 → 0.188 ms/step, 1.28× (bit-identical).** 1.21–1.90× across sizes — much
more on smaller grids, where cutting 16 launches to 3 dominates.

| size | faithful ms | opt ms | speedup | max\|diff\| |
|---|---|---|---|---|
| 108×137×30 | 0.0818 | 0.0430 | **1.90×** | 0 |
| 473×297×30 | 0.2393 | 0.1876 | **1.28×** | 0 |
| 473×297×50 | 0.2950 | 0.2427 | **1.22×** | 0 |
| 945×594×30 | 0.8125 | 0.6720 | **1.21×** | 0 |

## What worked (the default build, `OPTVER=1`)

Three changes, all *removing genuine waste* rather than being clever:

1. **Launch fusion, 16 kernels → 3.** The step has exactly three barriers:
   a per-cell pass A (`mass / length / source / drag`), a per-cell pass B
   (`flux-divergence / drag / kh / ku`), and the `sn_u`/`sn_v` zeroing that
   pass A reads. `meke_kernel.cu`'s own `fused` path already collapses to 6;
   this goes to **3** by merging the two sn-zero launches into one strided
   kernel and folding the two flux launches + the divergence launch into
   pass B (see #2). Fewer launches is the whole game on small grids (1.90×).
2. **Eliminate the `uflux`/`vflux` global arrays.** Pass B recomputes the 4
   surrounding face fluxes **in registers** from the post-drag energy
   (`meke_A`) + `mass_ws` stencils, instead of a producer kernel writing
   `nu+nv` doubles that the divergence kernel reads back. The MEKE flux is
   cheap (a harmonic mean + a few mults), so the 2× redundant flux math costs
   far less than the round-trip it removes — the opposite of the continuity
   kernel, where recompute of the expensive PPM *lost*.
3. **32-bit indexing.** `nx*ny*nz < 2³¹` for every realistic MEKE grid
   (479×303×30 ≈ 4.35e6 ≪ 2.1e9), so `meke_args.h`'s `size_t` address math is
   pure overhead. Overridden locally in `opt_kernel.cu`.

**Double-buffer.** Pass A reads `meke`(old) and writes a scratch `meke_A`
(post-drag); pass B reads the `meke_A` 5-point stencil and writes the final
`meke`. One extra `nt`-sized scratch, but it removes the `nu+nv` flux arrays
(net device memory *down*) and gives the flux a race-free neighbourhood
(`meke_A` is finished at the kernel boundary).

## What was tried and LOST / not taken

| idea | where | result | why |
|---|---|---|---|
| staged flux (keep `uflux`/`vflux`, 4 kernels) | `OPTVER=2` | 1.23× (vs 1.28×) | the flux round-trip it keeps costs more than the recompute it avoids; kept as the general fallback (below) |
| fewer than 3 kernels | — | impossible | pass A → flux → pass B are three genuine data-dependency barriers; nothing collapses them without a device-wide sync |

## Caveats

- **Config assumption in the recompute path.** Pass B's flux recompute reads
  `kh_diff` while pass B also *writes* `kh_diff` — a read/write hazard in
  general. It is benign here because the nml sets `khmeke_fac = 0`, so the
  `kh_diff` term is `0.0 * (finite) == 0.0` regardless of which copy is read,
  and `kh_u` collapses to `fmax(0, kh_bg)` exactly. **With `khmeke_fac ≠ 0`,
  use `OPTVER=2`** (staged flux in its own kernel), which is general and only
  ~4% slower. Every other input pass B reads (`meke_A`, `mass_ws`, the per-cell
  factors) is produced by pass A, so it is hazard-free by construction.
- **Conservative timing.** Both launchers `cudaDeviceSynchronize()` once per
  step (matching `meke_cuda_launch`), so the speedup isolates launch-count /
  memory-traffic, *not* a synchronization difference. Removing the per-step
  sync (letting the optimized launcher pipeline across steps) is an additional
  win not counted here.
- Degenerate gabight config (`a_* = 0`, `cd_scale = 0`) exercises MEKE's
  *shape*, not every arithmetic branch — same limitation as `meke_bench.F90`.
  The fusion is structural, so it holds regardless, but the exact-zero
  agreement should not be read as general nvcc FMA-grouping proof.

## Reproduce

```bash
nvcc ab_main.cu opt_kernel.cu meke_kernel.cu -O3 -arch=sm_70 -I../common -I. -o ab
# through the shared-GPU lock only:
bash ../tmp_local_artifacts/gpu_run.sh meke ./ab 473 297 30 300   # max|diff| + timing
nvcc -DOPTVER=2 ... -o ab            # the general staged-flux variant
```

`opt_kernel.cu` is a standalone best-CUDA artifact; the faithful
`meke_kernel.cu` stays untouched as the codegen-comparison baseline. Wiring
`meke_opt_launch(const MekeArgs*, double *meke_scratch)` into a driver is a
small follow-up (it needs one extra `nt`-sized scratch and drops the
`uflux`/`vflux` allocations).
