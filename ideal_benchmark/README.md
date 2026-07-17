# ideal_benchmark — the whole-model RK2 harness (all-DC vs all-CUDA)

A **dumb RK2 (2-stage) time-stepping driver that runs ALL 9 build-mode kernels
back-to-back per stage**, keeping the GPU hot, to measure the *aggregate*
whole-model cost. Physical correctness does **not** matter — each kernel runs on
its own independent state, driven in lockstep; this is a timing harness, not a
coupled model.

The binary times **two passes on the same resident device state**:

* **Phase 1 — all-DC**: each stage calls the kernel's optimized `do concurrent`
  routine.
* **Phase 2 — all-CUDA**: each stage calls the kernel's optimized CUDA launcher
  (`opt_kernel.cu`) via `!$acc host_data use_device`, on the *same* device
  arrays `rk2_<k>_init` already allocated + `enter data`'d. The CUDA pass runs on
  the state the DC pass left behind — fine for timing; per-kernel bit-identity
  DC==CUDA is already proven by each kernel's `cmp` build, so this harness only
  times and finite/non-zero-checks.

Every kernel sits behind a `bind(C)` seam: `rk2_<k>_init` / `rk2_<k>_stage`
(DC) / `rk2_<k>_stage_cuda` (CUDA) / `rk2_<k>_probe`.

## Run

```bash
make                 # build ./rk2  (nvfortran + nvcc, OpenACC GPU, cc70/sm_70)
make run             # build + run THROUGH THE GPU LOCK  (473 297 30 100 10)
# or explicitly (never run the GPU binary directly — shared card):
../tmp_local_artifacts/gpu_run.sh rk2-cuda ./rk2 473 297 30 100 10
make clean
```

Args: `nx_phys ny_phys nz nsteps nwarm`. Each kernel adds its own ghost halo
(most `nghost=3`, hll `nghost=2`) exactly as its standalone driver does.

The harness prints the headline **all-DC vs all-CUDA ms/RK2-step + ratio**, a
**per-kernel ms/stage breakdown for BOTH sides side by side** (one isolated loop
per kernel), and a finite+non-zero sanity probe of each kernel's output field on
both passes.

## The CUDA side (Phase 2)

Each `wrap/rk2_<k>.F90` gains `rk2_<k>_stage_cuda`, which wraps the kernel's
`extern "C"` optimized launcher in `!$acc host_data use_device(<that kernel's
device arrays>)`. Five kernels lift the interface + host_data call verbatim from
the kernel's existing `drivers/*_bridge.F90`; four are new bridges read straight
from the launcher's signature in `opt_kernel.cu`:

| kernel     | opt-CUDA launcher            | bridge source                        |
|------------|------------------------------|--------------------------------------|
| redi       | `redi_opt_launch`            | reuse `redi/drivers/redi_bridge.F90` |
| hvisc      | `hvisc_opt_launch`           | reuse `hvisc/.../hvisc_bridge.F90`   |
| ale        | `ale_remap_opt`              | reuse `ale_remap/.../ale_bridge.F90` |
| btstep     | `btstep_opt_launch_flat`     | reuse `btstep/.../btstep_bridge.F90` |
| meke       | `meke_opt_launch_flat`       | reuse `meke/.../meke_bridge.F90`     |
| continuity | `continuity_opt_launch`      | new (flat pointer list)              |
| hll        | `flux_hll_opt_launch`        | new (flat pointer list)              |
| kappa      | `ks_opt_launch`              | new, via `shim_ks.cu`                |
| epbl       | `epbl_opt_launch`            | new, via `shim_epbl.cu`              |

`ks_opt_launch` / `epbl_opt_launch` take a struct-of-knobs (`KsPar` /
`EpblParams`) plus device scratch counters, awkward to build from Fortran, so
`shim_ks.cu` / `shim_epbl.cu` (in this dir) build the struct C-side (values
copied verbatim from each kernel's `drivers/cpp_main.cu`), `cudaMalloc` the
counters once, and expose a uniform flat `*_opt_flat` entry — exactly the
pattern btstep/meke already ship.

**Build rule:** the Fortran wrappers (with `host_data`) are compiled **without**
`-cuda` (that flag flips nvfortran into CUDA-Fortran mode and trips
NVFORTRAN-S-0528 on the device-attributed host_data arrays). Each kernel's
`opt_kernel.cu` (+ shim) is compiled by `nvcc` with that kernel's flags and
linked into its `build/lib<k>.so`; `-cuda` is **link-only**, on both the `.so`
and the final exe, to pull in `libcudart`.

All 9 CUDA launchers wired up cleanly — **none excluded**.

## The 9 kernels and their opt-DC compute entry points

| # | kernel dir           | module                        | compute routine (per stage)                         |
|---|----------------------|-------------------------------|-----------------------------------------------------|
| 1 | `continuity_layered` | `continuity_layered`          | `continuity_compute_fluxes_fused`                   |
| 2 | `redi`               | `ocean_redi`                  | `redi_calc_coeffs` **+** `redi_apply_flux_hoist`    |
| 3 | `ale_remap`          | `ale_remap`                   | `ale_remap_step_opt`                                |
| 4 | `hvisc`              | `ocean_horizontal_viscosity`  | `hvisc_compute_fused`                               |
| 5 | `btstep`             | `btstep`                      | `btstep_nonlinear_closed_fused` (24 inner substeps) |
| 6 | `kappa_shear`        | `ks`                          | `kappa_shear_column_kernel` (faithful, no opt)      |
| 7 | `epbl`               | `ocean_epbl`                  | `epbl_column_kernel` (faithful, no opt)             |
| 8 | `meke`               | `meke`                        | `meke_step_ext_fused`                               |
| 9 | `hll_fluxes`         | `kernel_flux`                 | `compute_flux_hll` (faithful)                       |

Each `wrap/rk2_<k>.F90` holds that kernel's state as module `save` allocatables
and reuses its standalone driver's allocate + init + `!$acc enter data` verbatim
(from `<kernel>/drivers/dc_ab.F90` or `dc_main.F90`). It exposes three `bind(C)`
entries with unique C names: `rk2_<k>_init(nx,ny,nz)`, `rk2_<k>_stage()`,
`rk2_<k>_probe(vmin,vmax)`.

## The hard part: module-name collisions, and how they're isolated

The 9 kernels redefine **same-named modules with DIFFERENT bodies**:

| module name              | defined by (different bodies)                     |
|--------------------------|---------------------------------------------------|
| `constants`              | all 9 (parameters only — emits no link symbols)   |
| `grid`                   | continuity, redi, kappa, epbl, hll                |
| `ocean_metrics`          | continuity, redi                                  |
| `multilayer_cgrid_state` | continuity, redi, kappa                           |
| `ocean_eos`              | redi, kappa                                        |

(ale spells its own via `remap_state`; btstep via `bt_state`; meke via
`meke_state` — different module *names*, no clash.)

nvfortran mangles module symbols as `<module>_<name>_`, so the bodies above
collide at link. Empirically (via `nm`) the real duplicated defined symbols are
15: the module static-init routines (`constants_`, `grid_`, `ocean_eos_`, …),
the derived-type descriptors (`..._td_`, `…$$…$td$ld`), and the one shared EOS
device routine `ocean_eos_eos_specvol_derivs_`.

**Isolation strategy — per-kernel self-contained shared library:**

1. Each kernel's sources + its wrapper are compiled `-fPIC` in an **isolated
   build dir with its own `-module` dir** (so `.mod` files never coexist), with
   the same DC GPU flags every kernel uses
   (`-stdpar=gpu -acc=gpu -gpu=cc70,mem:separate -gpu=tripcount:host
   -DDC_DATA_ACC`) plus that kernel's `-D` (redi/kappa/ale/epbl carry
   `-DMODEL_NZ_STACK_MAX=128`, epbl also the EPBL launch-bound defs).
2. The objects are linked into a **self-contained `build/lib<k>.so`** with
   `nvfortran -shared -Wl,-Bsymbolic`. Two properties make the 9 co-link where a
   flat merge does not:
   * **The OpenACC device code is device-linked *inside* each `.so`.** Intra-
     kernel `!$acc routine` calls (e.g. `kernel_remap_remap_column_` called from
     the ale remap device kernels) resolve locally. A plain `ld -r` host merge
     does **not** preserve NVHPC relocatable device code, and `nvlink` then fails
     with *"Undefined reference to kernel_remap_remap_column_"* — that dead end
     is why this uses `.so`s, not `ld -r` + `objcopy --localize-symbol`.
   * **`-Bsymbolic` (DF_SYMBOLIC)** binds every reference *inside* a library to
     that library's **own** definition, at both link and load time. So each
     library's duplicated `constants_` / `ocean_eos_*` / type-descriptor symbols
     never cross-bind to another library's. The only symbols crossing the
     boundary are the `rk2_<k>_*` `bind(C)` entries, which are unique.
3. `rk2_main` (a plain program — no kernel modules, just `iso_c_binding`
   interfaces to the 27 entries) links against the 9 libraries with an rpath to
   `build/`.

No kernel needed the module-rename fallback: `-Bsymbolic` on self-contained
`.so`s isolated all 15 duplicated symbols and device registration stayed intact
(all 9 outputs come back finite + non-zero, which a cross-bind or a broken
device-kernel registration would not).

## What `rk2_main` does

* parse `nx ny nz nsteps nwarm`; call all 9 `rk2_<k>_init`;
* **Phase 1 (all-DC)** — warm-up + timed `nsteps` RK2 steps (2 stages each),
  every stage calls all 9 `rk2_<k>_stage` in order **continuity, redi, ale,
  hvisc, btstep, kappa, epbl, meke, hll**; `!$acc wait` + `system_clock` bound
  the loop → aggregate DC **ms/RK2-step**; then one isolated DC loop per kernel
  (`nsteps*2` calls) → the DC ms/stage column; probe all 9;
* **Phase 2 (all-CUDA)** — the identical structure on `rk2_<k>_stage_cuda`,
  bounded by `cudaDeviceSynchronize` (the raw CUDA launches are on the default
  stream, not the OpenACC queue) → CUDA aggregate + per-kernel column; probe all 9;
* report the DC/CUDA headline ratio + the side-by-side per-kernel table.

## Caveats (timing-harness honesty)

* **redi** apply-flux *accumulates* into the tracer (not idempotent), so its
  output field grows step to step. It stays finite; the timing is unaffected.
* **ale_remap** has a **fixed point** — after the first call `h_old ==
  target_h`, so a repeated call does strictly less work than the first. Calling
  it once per stage (as the task specifies) means every step after the first
  measures the fixed-point cost. Its per-kernel number is therefore a *lower*
  bound on a fresh-state remap.
* **btstep / meke / kappa / epbl** evolve or relax their own state across steps;
  all are dissipative and stay bounded over the run.
* **CUDA timing**: the launchers that expose a `sync` flag are called with
  `sync=0` and the loop is bounded by a single `cudaDeviceSynchronize`, so the
  GPU stays hot (no per-call barrier). Three launchers (redi, btstep, meke) sync
  internally every call regardless — a fixed per-call overhead, negligible next
  to their kernel cost.
* The two passes run on shared-GPU wall time; the DC-aggregate and the
  sum-of-isolated columns differ by whatever contention the card saw between
  them. The **DC/CUDA ratio** is the robust number — both passes see the same
  contention envelope.
