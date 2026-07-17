# ideal_benchmark — the whole-model RK2 harness (naive-DC | opt-DC | hand-CUDA | CPU)

A **dumb RK2 (2-stage) time-stepping driver that runs the 8-kernel ocean core
back-to-back per stage**, keeping the GPU (or CPU) hot, to measure the
*aggregate* whole-model cost. Physical correctness does **not** matter — each
kernel runs on its own independent state, driven in lockstep; this is a timing
harness, not a coupled model. `hll` is excluded, so the core is the 8 kernels
continuity, redi, ale, hvisc, btstep, kappa, epbl, meke.

It gives a **3-way model-level comparison** — naive `do concurrent`, optimized
`do concurrent`, and hand-written CUDA — plus an **all-CPU whole-model number**:

* **MODE** `dc | cuda | both` — which side(s) to TIME.
* **VERSION** `opt | unopt` — the optimized routines vs the **faithful** ports
  (naive DC / faithful CUDA). kappa & epbl have no opt DC variant, so their
  opt-DC == unopt-DC (expected).
* **`rk2_cpu`** — a second binary built `DATA=none` (`-stdpar=multicore`, or
  serial) with all CUDA compiled out (`-DRK2_NO_CUDA`): the whole model in pure
  `do concurrent` on the host.

Every kernel sits behind a `bind(C)` seam, up to four stage entries:
`rk2_<k>_stage` (opt-DC), `rk2_<k>_stage_unopt` (faithful-DC),
`rk2_<k>_stage_cuda` (opt-CUDA), `rk2_<k>_stage_cuda_unopt` (faithful-CUDA),
plus `rk2_<k>_init` / `rk2_<k>_probe`. `rk2_main` dispatches on MODE+VERSION.

## Run — the exact CLI

```bash
make                 # build ./rk2 (GPU) and ./rk2_cpu (CPU); nvfortran + nvcc
make rk2_cpu DC_HOST_FLAGS=      # CPU serial build (empty host flags)

# GPU: MODE in {dc,cuda,both} (default both), VERSION in {opt,unopt} (default opt)
./rk2      NXP NYP NZ NSTEPS NWARM [MODE] [VERSION]
# CPU: DC-only, mode implicit; arg 6 is VERSION
./rk2_cpu  NXP NYP NZ NSTEPS NWARM [VERSION]

make run       # rk2 through the GPU LOCK (473 297 30 100 10 both opt)
make run-cpu   # rk2_cpu directly (CPU-only, no lock)
# GPU must go through the lock (shared card):
../tmp_local_artifacts/gpu_run.sh rk2-cuda ./rk2 473 297 30 100 10 both opt
```

Each kernel adds its own ghost halo (`nghost=3`, hll would be 2 but is excluded).

**Machine-readable output** — one line per timed side (so `MODE=both VERSION=opt`
emits two, `mode=dc` and `mode=cuda`):

```
RESULT target=<gpu|cpu> mode=<dc|cuda> version=<opt|unopt> ms_per_stage=<f> ms_per_step=<f>
```

plus a human per-kernel ms/stage table + finite/non-zero sanity per side.

## The CUDA side

Each `wrap/rk2_<k>.F90` wraps the kernel's `extern "C"` launcher in `!$acc
host_data use_device(<that kernel's device arrays>)`. Both an **opt** launcher
(`opt_kernel.cu`) and a **faithful** launcher (`<k>_kernel.cu`) are linked into
each kernel's `.so`:

| kernel     | opt-CUDA launcher        | faithful-CUDA launcher              | bridge / shim |
|------------|--------------------------|-------------------------------------|---------------|
| redi       | `redi_opt_launch`        | `redi_cuda_launch_`                 | reuse `redi_bridge.F90` |
| hvisc      | `hvisc_opt_launch`       | `hvisc_compute_smag/face_cuda_launch` (2-pass) | reuse `hvisc_bridge.F90` |
| ale        | `ale_remap_opt`          | `ale_remap_cuda` (fused=0)          | reuse `ale_bridge.F90` |
| btstep     | `btstep_opt_launch_flat` | `btstep_cuda_launch` (mode 0)       | reuse + `shim_btstep.cu` |
| meke       | `meke_opt_launch_flat`   | `meke_cuda_launch` (mode 0)         | reuse + `shim_meke.cu` |
| continuity | `continuity_opt_launch`  | `continuity_layered_cuda_launch`    | flat pointer list |
| kappa      | `ks_opt_launch`          | `ks_cuda_launch`                    | `shim_ks.cu` |
| epbl       | `epbl_opt_launch`        | `epbl_cuda_launch` (variant 0)      | `shim_epbl.cu` |

All 8 faithful CUDA launchers link and run (validated finite+non-zero) — **none
excluded**. The struct-taking faithful launchers (btstep `BtArgs`, meke
`MekeArgs`, kappa `KsPar`, epbl `EpblParams`) are fed by the same flat shims that
serve the opt side, extended with a second `*_cuda_flat` entry.

`ks_opt_launch` / `epbl_opt_launch` take a struct-of-knobs (`KsPar` /
`EpblParams`) plus device scratch counters, awkward to build from Fortran, so

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

## The 8 kernels — opt-DC and faithful-DC compute routines

| # | kernel dir           | module                        | opt-DC routine (per stage)                          | faithful-DC routine |
|---|----------------------|-------------------------------|-----------------------------------------------------|---------------------|
| 1 | `continuity_layered` | `continuity_layered`          | `continuity_compute_fluxes_fused`                   | `continuity_compute_fluxes` |
| 2 | `redi`               | `ocean_redi`                  | `redi_calc_coeffs` + `redi_apply_flux_hoist`        | `redi_calc_coeffs` + `redi_apply_flux` |
| 3 | `ale_remap`          | `ale_remap`                   | `ale_remap_step_opt`                                | `ale_remap_step` |
| 4 | `hvisc`              | `ocean_horizontal_viscosity`  | `hvisc_compute_fused`                               | `hvisc_compute_smag` + `hvisc_compute_face` |
| 5 | `btstep`             | `btstep`                      | `btstep_nonlinear_closed_fused` (24 substeps)       | `btstep_nonlinear_closed` |
| 6 | `kappa_shear`        | `ks`                          | `kappa_shear_column_kernel` (no opt variant)        | = opt |
| 7 | `epbl`               | `ocean_epbl`                  | `epbl_column_kernel` (no opt variant)               | = opt |
| 8 | `meke`               | `meke`                        | `meke_step_ext_fused`                               | `meke_step_ext` |

(`hll_fluxes` is excluded from the ocean core.) Each `wrap/rk2_<k>.F90` holds
that kernel's state as module `save` allocatables and reuses its standalone
driver's allocate + init + `!$acc enter data` verbatim.

## The hard part: module-name collisions, and how they're isolated

The kernels redefine **same-named modules with DIFFERENT bodies**:

| module name              | defined by (different bodies)                     |
|--------------------------|---------------------------------------------------|
| `constants`              | all 8 (parameters only — emits no link symbols)   |
| `grid`                   | continuity, redi, kappa, epbl                     |
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
(all 8 outputs come back finite + non-zero, which a cross-bind or a broken
device-kernel registration would not). The CPU `.so`s (build_cpu/) are the same
wrappers compiled `DATA=none` + `-DRK2_NO_CUDA`, so all `host_data`/CUDA code is
`#ifdef`'d out and the whole model is bare `do concurrent` on the host.

## What `rk2_main` does

* parse `NXP NYP NZ NSTEPS NWARM [MODE] [VERSION]` (CPU build: arg 6 is VERSION,
  mode forced `dc`); call all 8 `rk2_<k>_init`;
* for each selected side (DC if `mode∈{dc,both}`, CUDA if `mode∈{cuda,both}`):
  warm-up + timed `NSTEPS` RK2 steps (2 stages each), every stage calls the 8
  kernels in order **continuity, redi, ale, hvisc, btstep, kappa, epbl, meke**,
  dispatching to `rk2_<k>_stage[_unopt]` (DC) or `rk2_<k>_stage_cuda[_unopt]`
  (CUDA) per VERSION; DC bounded by `!$acc wait`, CUDA by `cudaDeviceSynchronize`
  (raw launches are on the default stream, not the OpenACC queue);
* one isolated timed loop per kernel → the ms/stage column; probe all 8;
* emit one `RESULT` line per timed side + the human table (+ DC/CUDA ratio when
  `mode=both`).

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
