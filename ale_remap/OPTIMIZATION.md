# Optimizing the ALE remap CUDA kernel

An optimized launcher (`opt_kernel.cu`, `ale_remap_opt`) over the **faithful**
port (`ale_kernel.cu`, `ale_remap_cuda(..., fused=0)` — one CUDA kernel per
production do-concurrent loop). Held **bit-identical** to the faithful path:
`ab_main.cu` checks `max|diff| == 0` on every output array the two paths share.
V100-PCIE-32GB, nvcc 12.9, `-O3 -arch=sm_70`.

## Result

**7.82 → 6.92 ms/rep, 1.13× (bit-identical).** Every output array —
`h_layer`, `hTr_T`, `hTr_S`, `u`, `v`, `bt_eta`, and the heat/salt/mass budgets —
matches the faithful path to `max|diff| = 0.0`.

| grid (phys) | faithful ms/rep | optimized ms/rep | speedup | max rel |
|---|---|---|---|---|
| 473×297×30 | 7.82 | 6.92 | 1.13× | 0 |
| 240×180×30 | 2.50 | 2.17 | 1.16× | 0 |

## The one real lever: T and S share the column geometry

The faithful path remaps the two tracers with **two separate `k_tracer`
launches**. Each one, per column, rebuilds the *entire* PPM machinery from
scratch: the cumulative interface depths `z_old`/`z_new`, the PPM edge
reconstruction (`q_L`/`q_R`/`q6` with the parabola limiters), and then walks the
old↔new overlap sweep. T and S ride on **exactly the same `h_old`/`h_new`
column**, so all of that geometry is identical between them — the faithful path
just pays for it twice.

`k_opt_tracer2` builds the geometry **once** and reuses it for both tracers in a
single pass:

* `z_old`/`z_new` accumulated once,
* one overlap sweep whose per-`ko` interval (`z_lo`, `z_hi`, `overlap`, `xi_lo`,
  `xi_hi`) is computed once and consumed by both tracers' integrals,
* the only per-tracer work kept is the arithmetic that genuinely differs — the
  two edge reconstructions and the two integral accumulations, **in the same
  order as the faithful kernel**, which is why the result is bit-identical.

The cheap surrounding do-concurrent loops are folded too:
`k_opt_pre` does `total_h + h_ref + h_old-snapshot + target_h` in one per-column
kernel, and `k_opt_post` does `mass-budget + h_layer-commit + bt_eta` in one.
10 launches → 5.

## No intermediate global scratch

`total_h` and `h_ref` are consumed only inside the pre-pass. The faithful path
round-trips both through DRAM (`k_total_h` writes `total_h`, `k_h_ref` reads it
and writes `h_ref`, `k_target_h` reads that). `k_opt_pre` keeps both in
registers — `h_ref = Σh − eta` and the column total `= h_ref + eta` never leave
the thread. `ale_remap_opt` does not even allocate them.

## What was tried and did NOT help: 32-bit indexing

Worth **1.06× on the sibling continuity kernel**, it was a **no-op here**
(1.134× with `size_t` indexing vs 1.129× with `int32`, i.e. inside the noise;
`-DIDX32=0` reproduces). Why the difference: continuity is memory/address-bound
with a flat one-thread-per-face layout, so address-register pressure mattered.
This kernel is **compute- and register-bound on the per-column PPM** — each
thread carries ~10 `NZ_STACK_MAX`-sized stack arrays, and the bottleneck is that
register/occupancy pressure and the serial overlap sweep, not global address
math. Cutting the index width frees nothing the schedule was waiting on. It is
left as the default anyway (correct, free, and future-proof if the balance
shifts), but the honest attribution is: **the 1.13× is entirely the T+S geometry
fusion.**

## Why not more

The two face-remap kernels (`k_opt_xface`, `k_opt_yface`) are the momentum
remap on u/v faces; they have no T/S-style twin to fuse with and are already a
single launch each, so they are carried through unchanged (bit-identical). They
are a large, irreducible share of the runtime, which caps the whole-remap
speedup: the fusion halves the *tracer* geometry cost, but the tracer step is
only one of the three heavy PPM passes (T+S, u, v).

## Reproduce

```bash
nvcc ab_main.cu opt_kernel.cu ale_kernel.cu -O3 -arch=sm_70 -I../common -o ab
# ALL GPU runs go through the shared-GPU lock:
bash ../tmp_local_artifacts/gpu_run.sh ale ./ab 473 297 30 100 20 25
# isolate the 32-bit-index contribution:  nvcc -DIDX32=0 ... -o ab_i64
```

Args: `nx ny nz reps warmup hdrift%`. State is restored from a pristine copy
before every timed rep (the remap has a fixed point; un-restored reps measure a
kernel doing strictly less work than production). Timing is CUDA events around a
single call, restore untimed, min over reps.

## DC (do concurrent) optimization

The portable analogue of the CUDA win, added as `ale_remap_step_opt`
(`ale_remap.F90`, alongside the untouched faithful `ale_remap_step`) and driven
A/B by `drivers/dc_ab.F90` (`make dcab`, binary `dc_ab_$(DATA)`). Same three
levers as the CUDA path, all portable Fortran (fusion + register hoist, no
CUDA-specific tricks); runs unchanged under `DATA=acc/omp/none`.

**Result (V100 cc70, NVHPC 26.5, 473×297×30, h-drift 25%):**

| | ms/rep | speedup | max rel diff |
|---|---:|---:|---:|
| faithful `ale_remap_step` | 11.19 | — | — |
| optimized `ale_remap_step_opt` | 8.43 | **1.33×** | **0.0** |

Bit-identical: `max rel diff = 0.0` on **every** shared output (`h_layer`,
`hTr_T`, `hTr_S`, mass/heat/salt budgets, `u`, `v`, `bt_eta`).

**What is fused (10 do-concurrent launches → 5):**

* **T+S share the column geometry** (the one real lever). `ocean_remap_tracer_pair`
  → `remap_column_ppm2` builds `z_old`/`z_new` and walks the old↔new overlap
  sweep **once**, reusing the per-`ko` interval for both tracers' integrals; only
  the two edge reconstructions (`ppm_edges`) and the two integral accumulations
  stay per-tracer, **in `remap_column_ppm`'s exact association order** (the `d3`
  raw-cube-difference `* q6 / 3` is kept un-refactored — pre-dividing re-rounds
  ~1 ulp). Non-PPM / `nz<3` columns fall back to the faithful per-tracer path.
* **PRE (L1–L4 in one 2-D pass):** column total + `h_ref` + `h_old` snapshot +
  `target_h`. `total_h`/`h_ref` stay in registers (the faithful path round-trips
  both through global memory); `column_total = (Σh − eta) + eta` is kept, *not*
  simplified to `Σh` (FP: `(a−b)+b ≠ a`). Folds in the sig/collapse fix — the
  `target_h` loop bounds are plain locals, not derived-type components.
* **POST (L9+L10 in one 2-D pass):** mass-budget delta + `h_layer = target_h` +
  `bt_eta` re-derive (`Σ target_h`, same k order).

The two face-velocity remaps (`u`,`v`) have no T/S twin and read neighbour
columns (a real barrier), so they are carried through unchanged.

**The structural trap that decides win vs loss.** A first cut that flattened
*all* ~14 `NZ_STACK_MAX` per-column arrays into the fused loop's `local()` clause
ran **0.22× (4.6× slower)** — nvfortran spills the whole private set to local
memory and occupancy collapses (this kernel is register/occupancy-bound; see the
CUDA §"32-bit indexing" note). The fix is purely structural: the outer
`do concurrent` privatizes only the **six small column vectors**
(`dz_old/dz_new/cT/cS/nT/nS`); the heavy geometry+edge workspace lives in the
`remap_column_ppm2` callee's frame, off the `local()` list. Same arithmetic,
1.33× instead of 0.22×.

**CPU portability (`DATA=none`).** Bit-identical `max rel diff = 0.0` on a
sequential host build (`-Kieee`, or `-O0`) — proving the source, not just the
GPU codegen, is bit-identical. Two host caveats, **both pre-existing and
independent of this optimization**:
* `-O3 -fast` reassociates the register reductions (e.g. `s = s + h_layer(k)`)
  that the faithful path can't reorder (it accumulates into a *global* array
  element), giving a benign ~4e-11 rel diff. `-Kieee -Mnofma` → exactly 0.0.
* The **default `-stdpar=multicore` host backend has an nvfortran `local()`
  privatization bug** on this kernel's large per-column stack arrays: it makes
  even *faithful-vs-faithful* non-deterministic (reproduced with a standalone
  that uses only the shipped `kernel_remap`; single-thread too, so not a race).
  So `make dcab DATA=none` (which is `-stdpar=multicore`) reports a spurious
  DIFF from the faithful baseline, not from the optimization. Verify CPU
  bit-identity with `DC_HOST_FLAGS="-Kieee"` or on the GPU.

**Reproduce:**

```bash
make dcab DATA=acc
bash ../tmp_local_artifacts/gpu_run.sh ale-dcab ./dc_ab_acc 473 297 30 200 10 25
# CPU bit-identity (avoid the multicore local() compiler bug + FMA reassoc):
make dcab DATA=none DC_HOST_FLAGS="-Kieee -Mnofma"
./dc_ab_none 60 40 30 5 2 25
```

## Head-to-head: opt-CUDA vs opt-DC (shared host_data driver)

The optimized DC and optimized CUDA numbers above and in the DC section were
gathered in **two separate binaries** — a real caveat, since the two harnesses
own separate device state, so a like-for-like claim rests on both harnesses
seeding and mapping identically. `drivers/cmp_main.F90` + `drivers/ale_bridge.F90`
(`make cmp`) removes that caveat: **one binary, one truth**. The optimized DC
routine (`ale_remap_step_opt`) and the optimized CUDA launcher
(`opt_kernel.cu::ale_remap_opt`) run over the **same** device arrays — the bridge
hands the CUDA kernels the DC path's own allocations inside
`!$acc host_data use_device`. Timed over identical reps at production size, with
the pristine-state restore between every rep (fixed-point kernel + accumulating
budgets), and cross-checked field-by-field.

**Result (V100-PCIE-32GB, 473×297×30, 200 reps, opt-DC timed first):**

| endpoint | ms/rep |
|---|---|
| opt-DC (`ale_remap_step_opt`) | 8.44 |
| opt-CUDA (`ale_remap_opt`) | 7.02 |
| **ratio** | **opt-CUDA 1.20× faster** |

Agreement (opt-CUDA vs opt-DC, same device arrays):

| field | h_layer | hTr_T | hTr_S | u | v | bt_eta | budgets |
|---|---|---|---|---|---|---|---|
| max rel diff | 0.0 | 5e-16 | 6e-16 | 6e-16 | 7e-16 | 0.0 | 3e-15 |

Every prognostic field agrees at **FMA-contraction level** (< 1e-12); `h_layer`
and `bt_eta` are exactly bit-identical. This is cross-compiler (nvfortran libm/
codegen vs nvcc), so ~1e-16 is the floor, not 0.0.

**The metric matters — normalize by the field scale, not per element.** A first
cut used a per-element relative diff (`|x−y| / max(|x|,|y|)` per cell) and
reported a spurious **9e-9** on the budgets and **1.6e-10** on `v`, while the
absolute differences were all ~1e-16. Cause: the budgets are `new − old` (a
difference of two ~equal large numbers) and `v` crosses zero at flow nodes, so
both have cells whose own magnitude is ~1e-6 — dividing a 1e-16 absolute FMA
difference by that tiny local denominator manufactures a large ratio. Normalizing
each field by its **global** max magnitude (redi_bench's metric) reports the true
FMA-level agreement. Not a layout/geometry bug — the geometry is provably right
(`h_layer` exactly identical, since `target_h` is deterministic).

**Why opt-CUDA wins here but the DC section shows DC competitive elsewhere.**
opt-CUDA is timed *second*, so under sustained-load V100 drift it is if anything
*penalised* — the 1.20× is a floor. The DC-vs-faithful-DC A/B (`dc_ab`) and this
opt-vs-opt head-to-head answer different questions: `dc_ab` shows the portable
Fortran fusion recovers most of the win; this driver shows that once *both* sides
are optimized, hand-CUDA still keeps a ~20% edge on this register/occupancy-bound
kernel — but on one shared truth, not two harnesses.

**Reproduce:**

```bash
make cmp
bash ../tmp_local_artifacts/gpu_run.sh ale-cmp ./cmp_acc 473 297 30 200 10
```
