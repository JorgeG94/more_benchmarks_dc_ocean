# ocean horizontal viscosity (Smagorinsky) — `do concurrent` vs hand-written CUDA C

The `ocean_hvisc` profile region (~4.8% of production runtime), never benchmarked
before. The **full Smagorinsky closure** (anonymised, verbatim from `<model>`'s
structured lateral-mixing + horizontal-viscosity modules), two stages:

1. `hvisc_compute_smag` — strain → per-face viscosity
   `A_h = clamp((Cs·√(dxT·dyT))²·|D|)`, `|D| = √(D_T² + D_S²)`: cell-centred
   tension `D_T = ∂u/∂x − ∂v/∂y` + corner shear `D_S = ∂v/∂x + ∂u/∂y`
   (slip-masked by `wet_q`), each averaged onto the face.
2. `hvisc_compute_face` — `ah_face` × the curvilinear FV Laplacian of the
   C-grid face velocities.

Plain-integer explicit-shape dummies, no procedure calls in the loop bodies →
the loops auto-collapse and dodge both filed compiler bugs.

## Result (V100, 473×297×30) — DC ≈ (slightly better than) faithful CUDA

Full closure (smag + apply):

| sync mode | DC ms/rep | CUDA ms/rep | dc/cuda |
|---|---|---|---|
| 2 (per-kernel, faithful codegen) | 1.831 | 1.869 | **0.980×** |

Agreement (both `ah_face` and the tendency): `max|d ah_face| ≈ 1.7e-14`,
`max rel ≈ 1.3e-14` → **OK (<1e-12, FMA-contraction level)**. Not bit-exact 0
because the `sqrt` + strain reassociation differ 1 ulp between nvfortran and
nvcc; the apply-only stage is bit-exact.

Cost split: the apply alone is ~0.58 ms, so the **Smagorinsky coefficient
computation (~1.25 ms) dominates** — and DC matches CUDA on it.

**Verdict: another kernel where `do concurrent` ties (here edges out) a faithful
CUDA port.** The strain math is compute-heavier than the pure stencils, and
nvfortran holds its own — same story as continuity / ale / the HLL flux.

## Notes

- Metrics are uniform-square (`dx=dy` → ratios 1, `iareaC = 1/(dx·dy)`,
  `id* = 1/dx`, all-wet corners), the common production case; the kernels still
  evaluate the full curvilinear FV form. Free-slip (`ns=0`).
- The optional VarMix resolution scaling (`res_fn`) is omitted.
- Optimisation headroom (not yet done): fuse the two stages + fold the tiny
  boundary/reuse loops — the same play as the continuity 11→3.

## Build / run

```bash
make && make run                       # needs nvfortran + nvcc on PATH
./hvisc_bench 473 297 30 200 10 0       # nx_phys ny_phys nz reps warm cuda_sync
```

## Files

```
hvisc.F90         do-concurrent kernel (hvisc_compute_face) — the production extract
hvisc_kernel.cu   faithful CUDA C port (6 kernels, host_data use_device)
hvisc_bench.F90   driver: DC vs CUDA on the same device arrays, bit-check + timing
constants.F90     wp = real64 stub
```
