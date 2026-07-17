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

## DC (do concurrent) optimization

The same win, ported to portable Fortran `do concurrent` — no CUDA, no
shared-memory / launch-bounds tricks, builds and runs `DATA=none` on the CPU
unchanged.

**Result (V100, 473×297×30, `-stdpar=gpu`):**

| variant | ms/rep |
|---|---|
| faithful (two-pass: `smag` → ah_face → `face`) | 1.823 |
| **fused (single pass, `hvisc_compute_fused`)** | **1.362** |
| **speedup** | **1.34×** |

**max rel diff = 0 (bit-identical, both `du_visc` and `dv_visc`).**
CPU `DATA=none` (60×40×8): identical numbers, 1.85× — smaller grids gain more,
exactly as the CUDA side (launch/traffic overhead is a larger fraction).

### The transform — and why it is bit-identically fusable

The DC side ships as two routines (`hvisc_compute_smag` writes `ahx/ahy`, then
`hvisc_compute_face` reads them into `du/dv`), so the CUDA "12 kernels" here is
**12 `do concurrent` loops** (6 in smag: interior + 2 halo per face; 6 in face:
interior + 2 wall per face). `hvisc_compute_fused` collapses them to **2 loops**
(one per face direction), computing `ah` in a register and feeding it straight
into the Laplacian — the `ahx/ahy` face arrays **never touch memory**.

The dependency the fusion had to clear: `hvisc_compute_face` reads `ah_face_x`
at *neighbouring* faces? **No.** It reads `ah_face_x(i,j,k)` at exactly the
`(i,j,k)` it writes `du_visc`, and the du-interior range (`j=2:ny-1, i=2:nx`) is
**identical** to the smag u-face interior write range. So every consumed `ah` is
the one produced at the same cell — **no inline neighbour recompute is needed**,
and the ah halo/reuse writes (`ahx(i,1)`, `ahx(1,j)`, …) are **dead code** for
the du/dv output and are dropped. (Same for the v-face.) Because each per-cell
expression is reproduced in the identical operand order and grouping, the
compiler contracts FMAs the same way → `max|diff| == 0`, not merely `< 1e-12`.
Boundary faces write `0.0` inline, folding the 4 wall-zeroing loops in for free.

Because no thread recomputes a neighbour's `ah`, the arithmetic work is *also*
unchanged (unlike a fusion that duplicates the strain at shared faces) — the
payoff is purely deleted DRAM traffic + deleted launches, matching the CUDA
finding that the closure is memory-bound at these sizes.

### Reproduce

```bash
cd hvisc
make dcab DATA=acc
# ALWAYS through the shared GPU lock (never run ./dc_ab_acc directly):
bash ../tmp_local_artifacts/gpu_run.sh hvisc-dcab ./dc_ab_acc 473 297 30 200 10
# portability (no CUDA, CPU cores):
make dcab DATA=none && ./dc_ab_none 60 40 8 20 3
```

`drivers/dc_ab.F90` shares this binary's libm across both paths, so `du/dv` are
compared directly in memory — no reference dump, no cross-language libm trap.
The faithful `hvisc_compute_smag`/`hvisc_compute_face` stay untouched as the
baseline; `hvisc_compute_fused` is the new public routine.

## Head-to-head: opt-CUDA vs opt-DC (shared host_data driver)

The optimized DC (`hvisc_compute_fused`) and optimized CUDA (`hvisc_opt_launch`)
each had their own harness — DC timed in the `dc_ab` Fortran binary, CUDA in the
`ab` nvcc binary — so the earlier "opt-DC 1.362 vs opt-CUDA 1.438" comparison
carried a two-harness caveat (different clocks, different init, different
processes). `drivers/cmp_main.F90` removes it: **both endpoints run in ONE
binary, on ONE set of device allocations**, timed over identical reps with the
same `system_clock`. The DC routine writes the device `du/dv`; the CUDA launcher
is handed those *same* device pointers via `!$acc host_data use_device` (the
`drivers/hvisc_bridge.F90` bridge — a `bind(C)` interface, built **without**
`-cuda`, `-cuda` link-only), so there is one truth and the two outputs are
diffed in place.

**Result (V100, 473×297×30, 200 reps + 10 warm, `-stdpar=gpu`), 3 runs:**

| endpoint | ms/rep |
|---|---|
| **opt-DC** (`hvisc_compute_fused`) | **1.363 – 1.371** |
| opt-CUDA (`hvisc_opt_launch`, OPTVER=2) | 1.416 – 1.429 |
| **ratio** | **opt-DC faster, ~1.04×** |

**agreement: max rel diff = 1.31e-14** (both `du`/`dv`; max\|diff\| 8.6e-23) —
well under the 1e-12 bar. The tiny nonzero (vs the DC-vs-DC `max|diff| == 0`) is
FMA-contraction: nvcc and nvfortran group the two-`sqrt` strain the same way but
the compilers' codegen contracts a hair differently. Both sum to the identical
`du = -5.784248e-05`, `dv = -1.182050e-04`.

**Verdict: the head-to-head CONFIRMS the earlier split-harness reading — opt-DC
beats faithful hand-optimized CUDA on this kernel, by ~4%.** `do concurrent`
loses nothing to CUDA here; the fused single-launch closure is memory-bound, and
nvfortran's `-stdpar=gpu` codegen for the two fused loops is marginally tighter
than the OPTVER=2 merged kernel. There is no CUDA rewrite dividend to collect.

### Reproduce

```bash
cd hvisc
make cmp                       # Fortran (no -cuda) + nvcc opt_kernel.cu, link -cuda
# ALWAYS through the shared GPU lock (never run ./cmp_acc directly):
bash ../tmp_local_artifacts/gpu_run.sh hvisc-cmp ./cmp_acc 473 297 30 200 10
# or: make run-cmp   (wraps the lock)
```

The kernel is stateless per rep (reads `u/v` + geometry, writes `du/dv`), so
between the two timed runs the driver only snapshots opt-DC's result to host;
opt-CUDA overwrites every device cell (interior computed, walls → 0), so no
device reset is needed.
