# Optimizing the Redi apply-flux CUDA kernel

A best-CUDA-practices pass over the faithful port (`redi_kernel.cu`), freed from
the "faithful transliteration" rule but held **bit-identical** to it
(`ab_main.cu` checks `max|diff| == 0` on both tracers, not just <1e-12). Redi is
the biggest kernel in the model (~40%), so this is the highest-value target.
V100, 473×297×30, `NZ_STACK_MAX=128` (production).

## The optimization — precompute hoist in `redi_apply_flux`

The faithful `kRediApplyFlux` visits every T-cell `(i,j)` and evaluates **four**
face fluxes (W/E/S/N). Each face flux reconstructs its **left and right** tracer
column from scratch (`redi_tracer_column`: PLM slopes + PPM edges + per-layer
limiter over the whole column). That is **8 column rebuilds per cell** — but
only **5 distinct columns** are involved:

```
        (i,j)   <- W-right, E-left, S-right, N-left   (used by ALL FOUR faces)
        (i-1,j) <- W-left        (i+1,j) <- E-right
        (i,j-1) <- S-left        (i,j+1) <- N-right
```

The **centre column `(i,j)` is rebuilt 4×**. `kOptApplyFlux` reconstructs it
**once** into registers/local, reuses it in all four faces, and reconstructs
only the distinct neighbour per face → **8 rebuilds → 5**. This is the exact
analogue of the continuity fusion win and the lever the README calls out.

`redi_tracer_column` is a deterministic pure function of the column data, so
computing it once vs four times yields **the same bits**; the W,E,S,N order and
the `ks=1..ns-1` accumulation order into `dTr` are preserved exactly. Hence the
result is **bit-exact**, not merely FMA-level.

Secondary: 32-bit indexing (`IDX32`, default on) — the domain fits int32
((nx+1)·ny·nsurf ≈ 9.0e6 ≪ 2³¹). FP-neutral, so bit-identicality holds.

## Result — bit-identical, across sizes

| grid | faithful ms/call | optimized ms/call | speedup | max\|diff\| |
|---|---|---|---|---|
| 236×149×30 | 10.87 | 9.36 | **1.16×** | 0 |
| 473×297×30 | 44.98 | 42.11 | **1.07×** | 0 |
| 473×297×50 | 75.44 | 70.53 | **1.07×** | 0 |

Reproduce:
```bash
nvcc ab_main.cu opt_kernel.cu redi_kernel.cu -O3 -arch=sm_70 -I../common \
     -DMODEL_NZ_STACK_MAX=128 -o ab
# GPU runs go THROUGH THE LOCK on the shared V100:
bash ../tmp_local_artifacts/gpu_run.sh redi ./ab 473 297 30 50 10
```

## Why the whole-launch number is "only" ~1.07×, not the ~1.35× the lever quotes

Per-kernel profile (nsys, one launch = CalcCoeffsX + CalcCoeffsY + 2×(Snapshot +
ApplyFlux)), 473×297×30:

| kernel | faithful ms | optimized ms | share of launch |
|---|---|---|---|
| ApplyFlux (×2) | 13.78 each | **12.17 each** | 61% |
| CalcCoeffsX | 8.70 | 8.80 | 19% |
| CalcCoeffsY | 8.68 | 8.75 | 19% |
| Snapshot/FaceCopy | ~0.09 | ~0.09 | <1% |

The hoist targets ApplyFlux and speeds **that kernel 1.13×** (13.78→12.17 ms).
Two facts dilute it to 1.07× at the launch level:

1. **ApplyFlux is 61% of the launch, not 100%.** CalcCoeffs (the neutral-position
   search) is the other 38% and the hoist does not touch it.
2. **Recon is only ~31% of ApplyFlux; the per-face sublayer sweep is the rest.**
   Eliminating 3 of 8 rebuilds removes ~1.6 ms of ~4.3 ms of recon. The
   `ks`-loop over `redi_sublayer_dT` / `redi_ppm_ave` (4 faces × ns−1 surfaces)
   is **genuine, non-redundant** work — nothing to hoist there bit-identically.

The ~1.35× the README quotes is the kernel/algorithm-level figure (and the
Fortran `do concurrent`, where apply-flux dominates more). On the GPU the flux
sweep, not the recon, is the apply-flux hot path, so the same lever buys less.

## Bonus the hoist bought for free (ptxas `-v`, sm_70)

| | registers | stack frame | spills | occupancy (reg-limited) |
|---|---|---|---|---|
| faithful ApplyFlux | 93 | 16432 B | 0 | ~34% (704 thr/SM) |
| **opt ApplyFlux** | **64** | **14384 B** | 0 | **~50% (1024 thr/SM)** |

Reconstructing the centre once collapses the redundant recon live-ranges: 29
fewer registers and 2 KB less local memory, lifting reg-limited occupancy from
~34% to ~50% with no new spills. Part of the 1.13× is this occupancy gain, not
just the removed FLOPs.

## Tried and LOST / rejected (kept off the shared GPU after analysis)

| idea | verdict | why |
|---|---|---|
| 32-bit indexing (`IDX32=0` vs `1`) | **neutral** (within run noise, 41–42 ms) | ApplyFlux is local-memory/compute-bound, not address-bound; the register drop above came from the recon hoist, not int32. Kept on (free, harmless). |
| Global precompute-once of tracer columns to scratch (1 recon/cell in a separate kernel, ApplyFlux just reads) | rejected by cost model | Removes the last 4 redundant recons (5→1) but adds ~140 MB write + ~5× read of 4 per-cell recon arrays. Est. net ≈ break-even (recon is only ~22% of the hoisted ApplyFlux); the added global traffic eats the saved FLOPs. Not worth the complexity/risk on a shared V100. |
| Carry `tlb→tlt`, `trb→trt` between consecutive `ks` in the sublayer loop | rejected | The bottom interface of surface `ks` equals the top of `ks+1`, so 2 of 4 interface interpolations repeat — but they must then be computed even for `hEff==0` surfaces the loop currently skips, and they are a small part of `redi_sublayer_dT` (the two `redi_ppm_ave` calls dominate). Marginal, and fragile for bit-identicality. |
| Fuse CalcCoeffsX+Y, or Snapshot into ApplyFlux | rejected | Only saves launch overhead (µs) against 8–14 ms kernels; no work removed. |
| Hoist the redundant `build_column` in CalcCoeffs (each cell's column is built 4× across adjacent u/v-faces) | not attempted | The redundancy is **cross-thread** (adjacent faces in different threads), unreachable in the one-thread-per-face layout without a global precompute pass — a much larger rewrite, and the neutral-position search (the actual CalcCoeffs hot path) is per-face and non-shareable. |

## Files

- `opt_kernel.cu` — optimized launcher `redi_opt_launch` + `kOptApplyFlux` (the
  hoist). All device helpers and Phase-A kernels are transcribed verbatim from
  `redi_kernel.cu` (renamed / anonymous-namespace so both objects link);
  **only apply-flux carries the optimization**.
- `ab_main.cu` — A/B harness (init mirrors `drivers/cpp_main.cu`): faithful vs
  optimized, diffs both tracers (bit-exact check), times both with CUDA events.
- `redi_kernel.cu` — untouched faithful baseline.
