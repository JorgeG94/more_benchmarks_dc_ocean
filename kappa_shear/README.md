# kappa_shear — build-mode layout (kappa-shear JHL08 column kernel)

Port of `../kappa_shear_benchmark/` to the new layout: **one kernel dir,
single-sourced compute, strategy chosen as a build MODE** (not a directory),
with the plumbing macro'd. Mirrors the `../continuity_layered/` pilot. The old
`../kappa_shear_benchmark/` stays as the results-of-record.

The compute is a per-column JHL08 solve: one `do concurrent (j, i)` over owned
cells, one thread per COLUMN, with a fully sequential-in-k, **iterative**
(Picard × adaptive-substep) solve inside. `ks.F90`'s six column-solve helpers
are VERBATIM production; the six `!$acc routine seq` markers became
`DC_ROUTINE_SEQ`. The bare `do concurrent(...) local(...)` compute is untouched.

## Two knobs, two macro headers

| axis | knob | header | values |
|---|---|---|---|
| Fortran data layer | `DATA` | `../common/directives.h` | `acc` (OpenACC) · `omp` (OpenMP target) · `none` (CPU) |
| C++ GPU runtime | `BACKEND` (from `../config.mk`) | `../common/gpu_rt.h` | `cuda` (nvcc) · `hip` (hipcc) |

`-DMODEL_NZ_STACK_MAX=128` (the production per-column stack bound) is appended
to BOTH the Fortran and CUDA compile lines so the two kernels size their
per-thread column arrays identically.

## Build / run

```bash
make dc                 # do concurrent, OpenACC on the NVIDIA GPU   [default]
make dc DATA=omp        # do concurrent, OpenMP target (NVIDIA now; AMD/Intel with their compilers)
make dc DATA=none       # do concurrent on the CPU (-stdpar=multicore) -- NO CUDA, NO OpenACC
make cmp                # THE HEAD-TO-HEAD: one Fortran binary running
                        #   do concurrent + cuda_faithful + cuda_opt on the SAME
                        #   device allocation, one RESULT line each
make serialdo           # plain nested `do` baseline (binary `sdo`), any F2008 FC
make verify             # OpenACC-GPU dumps a ref; the CPU build recomputes + cross-checks it
make verify-variant     # faithful vs VARIANT=block bit-identity gate (strict FP)
make run-dc / run-cmp / clean

make cpp                # historical control only -- see "Why cmp, not cpp" below
```

## Why `cmp`, not `cpp`

`drivers/cpp_main.cu` is a standalone C++ `main()`, so it has to HAND-MIRROR
`build_state` in C++. A mirror that drifts even slightly measures a different
problem, silently, and a wall-clock number cannot tell you that -- which is why
that file carries ~150 lines of `[RISK A]` / `[RISK B]` machinery whose whole
job is to prove its inputs match the Fortran's. It also walks into the
cross-language libm trap (`CLAUDE.md`): nvfortran's and glibc's `sin`/`exp`
disagree in the last ulp.

`make cmp` deletes both problems. It compiles the SAME `drivers/dc_main.F90`
with `-DKS_WITH_CUDA`, links both hand-written CUDA kernels, and launches them
through `!$acc host_data use_device` on the device pointers `do concurrent` just
used. One state builder, one allocation, one process, three RESULT lines. The
sweep uses only this; `cpp` is kept as the historical no-Fortran control that
established OpenACC adds no launch overhead.

## The 1:1 pairing (read before quoting any DC-vs-CUDA ratio)

`ks_solve_column`'s three whole-array assignments copy the DECLARED extent in
Fortran -- O(NZ_STACK_MAX) -- but both CUDA kernels bounded them to `1..nz+1`,
O(nz) (`ks_kernel.cu:798`). Comparing across that gap times the copy, not the
codegen. `-DKS_FULL_COPY` (`make cmp FULLCOPY=1`) now lets the faithful CUDA
kernel reproduce the Fortran copy, and the two sides are built as MATCHED pairs:

| pairing | build | dc / cuda_faithful, 473x297x30, V100 |
|---|---|---|
| mismatched (historical) | `BCOPY=0`, `FULLCOPY=0` | 1.140x |
| matched `prod` (both O(NZSTACK)) | `BCOPY=0`, `FULLCOPY=1` | **1.106x** |
| matched `opt` (both O(nz)) | `BCOPY=1`, `FULLCOPY=0` | **1.112x** |

So the copy asymmetry was inflating CUDA's advantage by ~3 points -- about a
quarter of the claimed gap. `cuda_opt` always bounds the copy and therefore only
ever belongs to the `opt` pairing.

Args: `nxp nyp nz nreps nwarm [land_pct]` (default `473 297 30 8 2`). The DC-only
driver drops the old bench's trailing `cuda_sync` flag; the native `cpp` driver
is a verbatim copy of `ks_native.cu` and still accepts it.

## Measured on this node (V100, nvfortran / nvcc, 473×297×30, 8 timed / 2 warm)

The V100 was shared with sibling agents; ms/rep is noisy — read it as a band.

| mode | ms/rep | cross-check vs OpenACC ref |
|---|---|---|
| `dc DATA=acc`  (OpenACC, GPU) | ~40 | reference |
| `dc DATA=omp`  (OpenMP target, GPU) | ~40 | **max\|diff\| = 0.0 (bit-identical)** |
| `dc DATA=none` (CPU, -stdpar=multicore) | ~187 | max\|diff\| = 5.2e-12, max rel = 3.1e-10 (see below) |
| `cpp BACKEND=cuda` (native cudaMalloc) | ~35 | sum/max match acc (3.299882e+05 / 1.647386) |

**What this proves:** the macro'd data layer (`directives.h`) is numerically
inert on the SAME GPU codegen — `DATA=omp` reproduces `DATA=acc` **bit-for-bit
(0.0)**, which is the clean cross-check the port requires.

**The `DATA=none` relative gap is expected for THIS kernel, not a port bug.**
`ks.F90`'s own `verdict()` header warns of it: an iterative scheme is
legitimately sensitive — a 1-ulp FMA-contraction difference can flip a
convergence test, change an iteration count, then diverge visibly. `none`
switches the COMPUTE backend from GPU to CPU multicore (different FMA
contraction), independent of the data-layer macros. The **absolute** agreement
is ~5e-12 (kd_int ranges to 1.65, so ~12 significant figures); the relative
metric amplifies because the convergence tests key off small differences. This
is why the continuity pilot's `none` was bit-identical (straight-line PPM, no
convergence test) and this one is not. It is a property of the physics kernel,
reproducible run-to-run, not an artifact of the layout swap.

## Deviations from the pilot

- Driver rewritten from `ks_bench.F90` (not `continuity_layered.F90`): kept the
  full `build_state()` (z* grid + mixed layer straddling Ri_crit + surface jet)
  and the `nz+1 > NZ_STACK_MAX` guard. Dropped the CUDA variant, the `ks_par_t`
  bind(C) bridge, `host_data use_device`, the `n_out`/`n_in` iteration counters,
  and the DC-vs-CUDA `verdict()` (the old "*** PORT BUG ***" line — a documented
  false alarm from a sign-blind reduction metric, irrelevant here).
- Primary output for dump/cross-check is `kd_int` (shape `nx,ny,nz+1`).
- `DEFS := -DMODEL_NZ_STACK_MAX=128` appended to BASE_FFLAGS and CUFLAGS.
- No `-cuda` link flag (the pilot has none either): this DC driver has no
  `host_data`, so nvfortran's CUDA-Fortran mode is never needed.

## Layout

```
kappa_shear/
  constants.F90  ks_state.F90   MRE stubs (grid, multilayer_cgrid_state, ocean_eos)
  ks.F90                        the do-concurrent kernel (routine-seq -> DC_ROUTINE_SEQ)
  ks_kernel.cu                  the hand-written C++/CUDA kernel (now #include "gpu_rt.h")
  drivers/
    dc_main.F90                 DC-only driver (data via macros; dump/cross-check)
    cpp_main.cu                 native cudaMalloc driver (copy of ks_native.cu)
  Makefile                      build modes: dc DATA=... / cpp BACKEND=...
```
