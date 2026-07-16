# redi — new-style build-mode layout (Redi neutral diffusion).
# The KERNEL is the unit; the strategy is a build MODE, not a directory:
#
#   make dc                 # do concurrent, DATA=acc  (OpenACC, NVIDIA GPU) [default]
#   make dc DATA=omp        # do concurrent, OpenMP target (NVIDIA/AMD/Intel GPU)
#   make dc DATA=none       # do concurrent on the CPU (-stdpar=multicore) -- NO CUDA
#   make cpp                # hand-written C++, BACKEND=cuda (nvcc)
#   make cpp BACKEND=hip    # hand-written C++, HIP (hipcc) -- structural, needs ROCm
#   make verify             # acc dumps a ref; CPU re-runs and cross-checks it
#   make run-dc / run-cpp / clean
#
# Toolchain + arch + BACKEND come from ../config.mk (the one file to edit).
# Compute is single-sourced; only directives.h (Fortran) / gpu_rt.h (C++) change.
-include ../config.mk
FC      ?= nvfortran
NVCC    ?= nvcc
MODFLAG ?= -module
ARCH    ?= cc70
GPU_ARCHFLAG ?= -arch=sm_70
BACKEND ?= cuda

# -DMODEL_NZ_STACK_MAX=128 is production's (NOT the 256 header default); WITH_CUDA
# matches the benchmark's Fortran/CUDA flags. Appended to BOTH toolchains below.
DEFS        := -DMODEL_NZ_STACK_MAX=128 -DWITH_CUDA
BASE_FFLAGS := $(FFLAGS_BASE) -I../common $(DEFS)
DC_SRCS     := constants.F90 grid.F90 ocean_metrics.F90 multilayer_cgrid_state.F90 \
               ocean_boundary_types.F90 ocean_eos.F90 ocean_redi.F90 drivers/dc_main.F90

# ---- do-concurrent build mode: DATA selects the device data layer ----------
# tripcount:host IS MANDATORY for Redi (NVHPC 26.5 regression, NVIDIA TPR #38714).
DATA ?= acc
ifeq ($(DATA),acc)
  DC_MODE_FLAGS := -stdpar=gpu -acc=gpu -gpu=$(ARCH),mem:separate -gpu=tripcount:host -DDC_DATA_ACC
else ifeq ($(DATA),omp)
  DC_MODE_FLAGS := -stdpar=gpu -mp=gpu -gpu=$(ARCH),mem:separate -gpu=tripcount:host -DDC_DATA_OMP
else ifeq ($(DATA),none)
  DC_MODE_FLAGS := $(DC_HOST_FLAGS)
else
  $(error DATA must be acc | omp | none)
endif
DCBIN  := dc_$(DATA)
DCBLD  := build/dc_$(DATA)
DCOBJS := $(addprefix $(DCBLD)/,$(notdir $(DC_SRCS:.F90=.o)))

# ---- native C++ build mode: BACKEND selects the GPU runtime ----------------
CU_SRCS := redi_kernel.cu drivers/cpp_main.cu
CUFLAGS := -O3 $(GPU_ARCHFLAG) -I../common --compiler-options -fPIC $(DEFS)
ifeq ($(BACKEND),hip)
  CUFLAGS += -DUSE_HIP
endif
CPPBIN  := cpp_$(BACKEND)
CPPBLD  := build/cpp_$(BACKEND)
CUOBJS  := $(addprefix $(CPPBLD)/,$(notdir $(CU_SRCS:.cu=.o)))

vpath %.F90 . drivers
vpath %.cu  . drivers

.PHONY: dc cpp run-dc run-cpp verify clean
.NOTPARALLEL:            # honour the DC_SRCS dependency order

dc: $(DCBIN)
cpp: $(CPPBIN)

$(DCBLD) $(CPPBLD):
	@mkdir -p $@

$(DCBLD)/%.o: %.F90 ../common/directives.h | $(DCBLD)
	$(FC) $(BASE_FFLAGS) $(DC_MODE_FLAGS) $(MODFLAG) $(DCBLD) -c $< -o $@

$(DCBIN): $(DCOBJS)
	$(FC) $(BASE_FFLAGS) $(DC_MODE_FLAGS) $(DCOBJS) -o $@

$(CPPBLD)/%.o: %.cu ../common/gpu_rt.h | $(CPPBLD)
	$(NVCC) $(CUFLAGS) -c $< -o $@

$(CPPBIN): $(CUOBJS)
	$(NVCC) $(CUOBJS) -o $@ $(GPU_ARCHFLAG)

ARGS ?= 473 297 30 8 2
run-dc: dc
	@./$(DCBIN) $(ARGS)
run-cpp: cpp
	@./$(CPPBIN) $(ARGS)

# Prove the macro'd data layer is numerically inert: OpenACC (GPU) writes a
# reference; the CPU build recomputes and cross-checks it to FMA level.
verify:
	@$(MAKE) --no-print-directory dc DATA=acc
	@DC_DUMP=redi_ref.bin ./dc_acc $(ARGS)
	@$(MAKE) --no-print-directory dc DATA=none
	@DC_REF=redi_ref.bin  ./dc_none $(ARGS)

clean:
	@rm -rf build dc_acc dc_omp dc_none cpp_cuda cpp_hip redi_ref.bin *.mod
