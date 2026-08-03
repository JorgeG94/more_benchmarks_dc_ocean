# mre_acc_cuda/config.mk — the ONE file you edit per machine / per backend.
#
# Every benchmark Makefile does `-include ../config.mk` at its top, so these
# defaults flow into all of them without touching their internals. Override on
# the CLI to sweep a single benchmark, or at the root to sweep them all:
#
#   make -C redi_benchmark ARCH=cc90 NVARCH=sm_90 run     # one benchmark, H200
#   make all-dc                                           # all, BACKEND=cuda (default)
#   make all-dc BACKEND=hip                               # all, AMD path (see notes)
#
# Concept mirrors ../dc_patterns/config.mk: the COMPUTE is always Fortran
# `do concurrent` (nvfortran -stdpar=gpu); BACKEND selects only which compiler
# builds the hand-written GPU-C *comparison* kernels (the `.cu` files).

# ---- Fortran (do concurrent) toolchain — identical for every backend --------
# NOTE: keep these as bare `VAR ?= value` with NO trailing inline comment — the
# spaces before a `#` land INSIDE the value and break `-gpu=$(ARCH),mem:separate`.
FC       ?= nvfortran

# ---- GPU ARCHITECTURE: DETECTED, NOT ASSUMED -------------------------------
# This used to be a hardcoded `cc70`, which was a foot-gun the moment these
# benchmarks left the V100 box: on any other card the entire sweep compiles for
# Volta and runs through JIT/compat, and every DC-vs-CUDA ratio in the resulting
# table is measuring the wrong target. Detect from the driver instead.
#
#   nvidia-smi --query-gpu=compute_cap  ->  "9.0"  ->  cc90 / sm_90
#
# Assigned with `=` (lazy) on purpose: it is only expanded if nothing overrides
# it, so `make ARCH=cc90 NVARCH=sm_90` and tools/ks_sweep.sh (which detects once
# and passes both) never pay for an nvidia-smi call per make invocation.
GPU_CC    = $(shell command -v nvidia-smi >/dev/null 2>&1 && \
              nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
              | head -1 | tr -d ' .')
GPU_ARCH  = $(if $(strip $(GPU_CC)),cc$(strip $(GPU_CC)),cc70)
# the benchmarks spell it ARCH; keep the two in sync
ARCH     ?= $(GPU_ARCH)
# -DMODEL_NZ_STACK_MAX; production build uses 128
NZSTACK  ?= 128
# redi_benchmark spells the same knob STACK
STACK    ?= $(NZSTACK)

# --- base Fortran flags for the do-concurrent COMPUTE, per compiler ----------
# The compute is standard F2018 `do concurrent`; only these base flags change by
# compiler. The GPU data layer (DATA=acc|omp, added by each kernel's Makefile)
# is nvfortran-only; DATA=none is the portable CPU build and works with any
# compiler that supports F2018 `do concurrent ... local(...)` -- nvfortran,
# gfortran >= 12, or ifx >= 2023. Keyed on FC; override on the CLI too.
# MODFLAG: how the compiler is told where to WRITE .mod files. Spelled as a
# space-separated flag so the kernel Makefiles use `$(MODFLAG) <dir>` uniformly:
#   nvfortran / ifx -> -module <dir>;  gfortran -> -J <dir>;  flang -> -module-dir.
ifeq ($(notdir $(FC)),nvfortran)
  FFLAGS_BASE   ?= -Mfree -Mbackslash -O3 -fast
  # DATA=none: run do concurrent across CPU cores
  DC_HOST_FLAGS ?= -stdpar=multicore
  MODFLAG       ?= -module
else ifeq ($(notdir $(FC)),gfortran)
  FFLAGS_BASE   ?= -O3
  DC_HOST_FLAGS ?=
  MODFLAG       ?= -J
else ifeq ($(notdir $(FC)),ifx)
  FFLAGS_BASE   ?= -O3
  # DATA=none default: do concurrent across CPU cores (Intel maps DC under -qopenmp)
  DC_HOST_FLAGS ?= -qopenmp
  MODFLAG       ?= -module
else ifeq ($(notdir $(FC)),amdflang)
  FFLAGS_BASE   ?= -O3
  DC_HOST_FLAGS ?= -fopenmp -fdo-concurrent-to-openmp=host
  MODFLAG       ?= -module-dir
else ifeq ($(notdir $(FC)),flang)
  FFLAGS_BASE   ?= -O3
  DC_HOST_FLAGS ?= -fopenmp -fdo-concurrent-to-openmp=host
  MODFLAG       ?= -module-dir
else ifeq ($(notdir $(FC)),flang-new)
  FFLAGS_BASE   ?= -O3
  DC_HOST_FLAGS ?= -fopenmp -fdo-concurrent-to-openmp=host
  MODFLAG       ?= -module-dir
else
  # unknown compiler: a plain optimised build is the safe default
  FFLAGS_BASE   ?= -O3
  DC_HOST_FLAGS ?=
  MODFLAG       ?= -J
endif

# ---- vendor GPU-offload flags for `do concurrent` (Intel / AMD) -------------
# nvfortran offloads via the kernel Makefile's DATA=acc|omp path. For the OTHER
# vendors, run_all.sh injects DC_GPU_FLAGS as a DC_MODE_FLAGS override on a
# DATA=omp build -- the OpenMP-target data layer (DC_DATA_OMP) is vendor-portable,
# only the compile flags differ. STRUCTURAL for Intel/AMD GPUs: validated where
# such a device + toolchain exists (none on this dev node), mirroring the
# BACKEND=hip seam below. Override the AMD arch as needed.
# AMD arch: DETECTED, not assumed -- same foot-gun as the NVIDIA one above.
# gfx90a is MI210/MI250; an MI300 is gfx942 and would silently get the wrong
# target. Lazy `=` so an override (or a non-AMD box) never pays for rocminfo.
AMD_GPU_CC   = $(shell command -v rocminfo >/dev/null 2>&1 && \
                 rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1)
AMD_GPU_ARCH = $(if $(strip $(AMD_GPU_CC)),$(strip $(AMD_GPU_CC)),gfx90a)

# LLVM flang and AMD's amdflang are the same driver for this purpose -- both
# take `-fdo-concurrent-to-openmp=device`. Listing only `amdflang` meant a site
# using the plain `flang` module got NO GPU lane at all and the AMD column came
# back empty for the wrong reason.
ifeq ($(notdir $(FC)),ifx)
  DC_GPU_FLAGS ?= -qopenmp -fopenmp-targets=spir64 -fopenmp-target-do-concurrent -DDC_DATA_OMP
else ifneq ($(filter $(notdir $(FC)),amdflang flang flang-new),)
  DC_GPU_FLAGS ?= -fopenmp --offload-arch=$(AMD_GPU_ARCH) -fdo-concurrent-to-openmp=device -DDC_DATA_OMP
else
  DC_GPU_FLAGS ?=
endif

# ---- GPU-C comparison-kernel backend: cuda (default) | hip ------------------
BACKEND  ?= cuda

ifeq ($(BACKEND),cuda)
  GPUCC        ?= nvcc
  # nvcc -arch, detected the same way as GPU_ARCH above (V100=sm_70, A100=sm_80,
  # H100/GH200=sm_90, B200=sm_100). Falls back to sm_70 only if no driver is
  # queryable -- which on a GPU box means something is wrong, not that it is a V100.
  NVARCH       ?= $(if $(strip $(GPU_CC)),sm_$(strip $(GPU_CC)),sm_70)
  GPU_ARCHFLAG ?= -arch=$(NVARCH)

else ifeq ($(BACKEND),hip)
  # ---------------------------------------------------------------------------
  # AMD / ROCm path. The .cu kernels are plain CUDA C and hipcc compiles most
  # of them unmodified, BUT this switch is structural only — it is not yet
  # validated (no ROCm on this node). Two things remain before `BACKEND=hip`
  # actually links, both mechanical:
  #   1. `hipify-perl *_kernel.cu *_native.cu` — the cudaMalloc/cudaMemcpy/
  #      stream/event calls in the native drivers need the hip* spellings.
  #   2. The benchmark CUFLAGS still carry nvcc-isms (`--compiler-options`);
  #      hipcc wants `-Xcompiler`. GPU_ARCHFLAG below already handles -arch.
  # Treat this block as the documented seam, the way dc_patterns documents its
  # amdflang/ifx variants.
  # ---------------------------------------------------------------------------
  GPUCC        ?= hipcc
  # MI210/MI250
  HIPARCH      ?= gfx90a
  # so any stray $(NVARCH) stays meaningful
  NVARCH       ?= $(HIPARCH)
  GPU_ARCHFLAG ?= --offload-arch=$(HIPARCH)

else
  $(error Unknown BACKEND '$(BACKEND)' -- use 'cuda' or 'hip')
endif

# The benchmarks spell the GPU-C compiler NVCC; point that name at the backend.
NVCC ?= $(GPUCC)
