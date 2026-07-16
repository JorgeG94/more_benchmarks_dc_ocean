# Optimizing the hvisc Smagorinsky-closure CUDA kernel

A best-CUDA-practices pass over `hvisc_kernel.cu` (the faithful 12-kernel port
of the ocean horizontal-viscosity closure), freed from the transliteration rule
but held to **exact agreement**: the strain carries two `sqrt` calls, so the bar
is field-relative `max|diff|/max|field| < 1e-12`, not bitwise. In practice, with
the per-cell arithmetic order kept identical, the fused path comes out
**bit-identical** (`max|diff| == 0`). V100, 473x297x30.

## Result

**1.79 → 1.42 ms/rep, 1.26x (bit-identical, du_visc and dv_visc).**
1.25–1.83x across sizes — most on small grids, where cutting 12 launches to 1
matters most.

| size | speedup | max\|diff\| |
|---|---|---|
| 108×137×30 | 1.83× | 0 |
| 473×297×30 | 1.26× | 0 |
| 473×297×50 | 1.25× | 0 |
| 945×594×30 | 1.31× | 0 |

## What the faithful path does (12 kernels)

Two stages joined by two full face-sized global intermediate arrays:

1. **smag** (6 kernels): from the strain rate compute `ah_face_x` (U-shaped) and
   `ah_face_y` (V-shaped) — interior `kS_u`/`kS_v` (each with a `sqrt` on the
   metric and a `sqrt` on the strain), plus 4 tiny halo kernels
   (`kS_u_reuse/_bg`, `kS_v_reuse/_bg`) that fill only `ah_face`'s boundary.
2. **apply** (6 kernels): `du_visc = ah_face_x · Laplacian(u)`,
   `dv_visc = ah_face_y · Laplacian(v)` — interior `kH_u`/`kH_v` plus 4 tiny
   wall kernels that zero `du/dv` on the domain edges.

## What worked (the default build, `OPTVER=2`)

1. **Fuse the entire closure into ONE kernel (12 → 1).** Each interior face
   thread computes its `ah_face` value (the full `kS_*` body, both `sqrt`s) in
   registers and feeds it straight into the Laplacian (the `kH_*` body); the
   `ah_face_x`/`ah_face_y` global arrays **never touch memory**. That removes
   two face-sized write+read round-trips per call. The one kernel walks
   U-then-V faces via a single flattened thread index. **This is the win.**
   - **Dead-code drop:** the 4 `ah_face` halo kernels (`kS_*_reuse/_bg`) fill
     only `ah_face`'s boundary, and `kH_u`/`kH_v` read `ah_face` *solely* at the
     interior cells they themselves produced — so those 4 kernels never affect
     `du/dv` and are simply dropped.
   - **Walls for free:** boundary threads write `0.0`, folding the 4 `kH_*_b*`
     wall-zeroing kernels into the same kernel — no per-rep memset.
2. **32-bit indexing** (`IDX32=1`): the domain fits `int32`
   (`nx·ny·nz < 2³¹`), so `size_t` address math was pure overhead. A
   `-DIDX32=0` size_t-safe path is kept for pathologically large grids.

Agreement is **bit-identical** because every per-cell expression is reproduced
in the same order (same operands, same grouping), so the compiler emits the same
FMA-contracted code; storing `ah_face` to global and reloading it in the faithful
path is a value-preserving no-op.

## What was tried and LOST / tied (kept as `OPTVER=1`)

| idea | `OPTVER` | result | why |
|---|---|---|---|
| separate U and V fused kernels (2 launches) | 1 | 1.73× @108³-ish, ties on big | one extra launch; only visible when launch-bound (small grids) |

`OPTVER=1` (a kernel each for U and V) is the same fusion minus the u/v merge.
It ties `OPTVER=2` on the large memory-bound grids but loses on the small grid
(1.73× vs 1.83×) — the merged single launch is strictly ≥, so it is the default.

The closure is memory-bound at these sizes (the fused kernel is a wide,
low-arithmetic-intensity stencil over the U and V face arrays), so the payoff is
exactly "delete intermediate DRAM traffic + delete launches"; there is no
compute slack for cleverer restructuring to buy back.

## Reproduce

```bash
nvcc ab_main.cu opt_kernel.cu hvisc_kernel.cu -O3 -arch=sm_70 -I../common -o ab
# ALWAYS through the shared GPU lock (never run ./ab directly):
bash ../tmp_local_artifacts/gpu_run.sh hvisc ./ab 473 297 30 200
# variant: nvcc -DOPTVER=1 ...   (separate U/V kernels)
```

`opt_kernel.cu` is a standalone best-CUDA artifact (launcher
`hvisc_opt_launch`); the faithful `hvisc_kernel.cu` stays untouched as the
codegen-comparison baseline. `ab_main.cu` reuses `drivers/cpp_main.cu`'s init,
runs faithful → `du0/dv0` and optimized → `du1/dv1`, diffs both fields
field-relative, and times both with CUDA events (min over windows).
```
