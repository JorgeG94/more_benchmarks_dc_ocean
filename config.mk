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
# nvfortran -gpu=<arch>: V100=cc70, A100=cc80, H200=cc90
GPU_ARCH ?= cc70
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
  DC_HOST_FLAGS ?=
  MODFLAG       ?= -module
else ifeq ($(notdir $(FC)),amdflang)
  FFLAGS_BASE   ?= -O3
  DC_HOST_FLAGS ?=
  MODFLAG       ?= -module-dir
else
  # unknown compiler: a plain optimised build is the safe default
  FFLAGS_BASE   ?= -O3
  DC_HOST_FLAGS ?=
  MODFLAG       ?= -J
endif

# ---- GPU-C comparison-kernel backend: cuda (default) | hip ------------------
BACKEND  ?= cuda

ifeq ($(BACKEND),cuda)
  GPUCC        ?= nvcc
  # nvcc -arch: V100=sm_70, A100=sm_80, H200=sm_90
  NVARCH       ?= sm_70
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
