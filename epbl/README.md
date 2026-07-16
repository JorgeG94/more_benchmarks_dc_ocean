# epbl — build-mode layout

The ocean model's **PRODUCTION EPBL column kernel** ported to the new layout
(one kernel dir, single-sourced compute, strategy chosen as a build MODE, the
plumbing macro'd). Follows the `../continuity_layered` pilot. The old
`../epbl_benchmark/` stays as the results-of-record.

EPBL is the largest region in the profile (`ocean_vmix_compute`, 39.1% of
runtime). Unlike the per-cell kernels, it is one-thread-per-COLUMN with ~630
sequential steps inside (a prep sweep + an MLD root-find), so it is the first
kernel where the two languages could plausibly diverge — and where the outputs
are chaotically branch-sensitive (see "MLD branch sensitivity" below).

## Two knobs, two macro headers

| axis | knob | header | values |
|---|---|---|---|
| Fortran data layer | `DATA` | `../common/directives.h` | `acc` (OpenACC) · `omp` (OpenMP target) · `none` (CPU) |
| C++ GPU runtime | `BACKEND` (from `../config.mk`) | `../common/gpu_rt.h` | `cuda` (nvcc) · `hip` (hipcc) |

The compute is single-sourced: `ocean_epbl.F90` (the `do concurrent` kernel,
`!$acc routine seq` → `DC_ROUTINE_SEQ`) and `epbl_kernel.cu` (the C++ kernel)
never change between modes — only the macro lines do. `epbl_stubs.F90` carries
the two device-routine helpers (`eos_specvol_derivs`, `roquet_spv_point`).

## Build / run

```bash
make dc                 # do concurrent, OpenACC on the NVIDIA GPU   [default]
make dc DATA=omp        # do concurrent, OpenMP target (NVIDIA now; AMD/Intel with their compilers)
make dc DATA=none       # do concurrent on the CPU (-stdpar=multicore) -- NO CUDA, NO OpenACC
make cpp                # native C++/CUDA (cudaMalloc), no Fortran
make cpp BACKEND=hip    # native C++/HIP -- structural (needs ROCm/hipcc)
make verify             # OpenACC-GPU dumps a ref; the CPU build recomputes + cross-checks it
make run-dc / run-cpp / clean
```

`DEFS = -DMODEL_NZ_STACK_MAX=128 -DEPBL_LB_THREADS=128 -DEPBL_LB_BLOCKS_PER_SM=4`
is appended to BOTH the Fortran and the nvcc flag sets. The real gabight
bathymetry (`gabight_bathy_0p1_473x297.f64`) is read at runtime from this dir —
the binaries run from here, so the relative path resolves.

## Measured on this node (V100, shared with 6 sibling agents), 473×297×30, short reps (8/2, 5 trials)

| mode | ms/rep (min) | cross-check vs OpenACC ref |
|---|---|---|
| `dc DATA=acc`  (OpenACC, GPU) | 7.1 | reference |
| `dc DATA=omp`  (OpenMP target, GPU) | 7.1 | **bit-identical** (kd_int & mld max\|diff\| = 0.0) |
| `dc DATA=none` (CPU, -stdpar=multicore) | ~118 (noisy) | kd_int **1.1e-13 vs field** ✓ · mld 3.9e-13 rel ✓ (no branch flip) |
| `cpp BACKEND=cuda` (native cudaMalloc) | 6.7 faithful / 5.3 tuned | own-init C++ (matches `../epbl_benchmark`) |

Timing is noisy on this shared V100 (`dc none` swung 4× across trials); the MIN
over interleaved trials is reported and the driver flags contention. Treat the
CPU number as an order of magnitude, not a measurement.

**What this proves:** the macro'd data layer (`directives.h`) is a numerically
inert swap — OpenMP target reproduces OpenACC bit-for-bit, and the CPU build
(zero CUDA, zero OpenACC) reproduces kd_int to FMA level. The C++ runtime
redirect (`gpu_rt.h`) keeps `epbl_kernel.cu` + the native driver building under
nvcc, with hipcc wired behind `-DUSE_HIP`.

## kd_int is the cross-check bar — judged against the field, not pointwise

`kd_int` is diapycnal diffusivity **at interfaces** and is structurally zero at
the bed (`k=1`) and surface (`k=nz+1`), and near-zero at statically stable
interior interfaces. A *pointwise* relative error therefore divides an FMA-level
roundoff (~3e-12 abs) by a ~1e-21 near-zero and reports a meaningless ~1e-9
"relative" error at cells that carry no signal — exactly the argument the
pilot's native driver makes for the divergence field `flux_h`. So the driver
gates on `max|diff| / max|kd_field|` (**1.1e-13** here, well under 1e-12) and
prints the inflated pointwise rel alongside, with the near-zero cell count, for
transparency.

## MLD branch sensitivity — a divergence is EXPECTED, not a bug

EPBL is genuinely, legitimately chaotic. A sub-ulp FMA difference can flip the
static-stability short-circuit (`tot_tke <= 0 .and. stable`) at the mixed-layer
base, and `mld_output` then accrues one **whole layer** more or less — the
output is layer-quantised, so a 1-ulp input difference can produce a ~60 m MLD
difference in a small % of columns. Across acc/omp/none **at these args no
column flipped** (mld agreed to 3.9e-13 rel), but a different size/rep count can
surface a handful. When it does, the driver reports it as a percentage of wet
columns and how many moved by ≤ ~1 layer thickness, with this note — it does
**not** treat it as a failure. `../epbl_benchmark`'s `make nofma` proves the
mechanism: with FMA contraction off on both sides the disagreement collapses to
~1e-10 m. See that README's "Bit-identity" section.

## Layout

```
epbl/
  constants.F90 grid.F90 epbl_stubs.F90   MRE stubs (+ 2 device routines)
  ocean_epbl.F90                          the do-concurrent EPBL kernel (production)
  epbl_kernel.cu                          the hand-written C++/CUDA kernel
  epbl_params.h                           shared EpblParams ABI + epbl_cuda_launch
  gabight_bathy_0p1_473x297.f64           REAL bathymetry, read at runtime
  drivers/
    dc_main.F90                           DC-only driver (data via macros; dump/cross-check)
    cpp_main.cu                           native cudaMalloc/hipMalloc driver
  Makefile                                build modes: dc DATA=... / cpp BACKEND=...
```
