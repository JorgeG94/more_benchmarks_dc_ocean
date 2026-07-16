# ale_remap — new-style build-mode layout (ports ../ale_remap_benchmark).
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
ARCH    ?= cc70
GPU_ARCHFLAG ?= -arch=sm_70
BACKEND ?= cuda

# ⚠ NZ_STACK_MAX=128 matches production's ACTUAL build (-DMODEL_NZ_STACK_MAX=128),
# NOT the 256 default in constants.F90's #define. Appended to BOTH flag sets.
DEFS := -DMODEL_NZ_STACK_MAX=128

BASE_FFLAGS := $(FFLAGS_BASE) -I../common $(DEFS)
# module compile-order (from ../ale_remap_benchmark OBJS): the PRODUCTION DC
# variant only -- ale_remap.F90 built with default MODNAME (no ASYNC/PPM_DIRECT/
# COLLAPSE_FIX/FLATSIG). The _acc/_fix/_fused variants are dropped.
DC_SRCS     := constants.F90 remap_state.F90 kernel_remap.F90 ale_remap.F90 drivers/dc_main.F90

# ---- do-concurrent build mode: DATA selects the device data layer ----------
# tripcount:host is MANDATORY for this kernel (NVIDIA TPR #38714) -- carried by
# the acc/omp GPU modes below, same as the old benchmark.
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
CU_SRCS := ale_kernel.cu drivers/cpp_main.cu
CUFLAGS := -O3 $(GPU_ARCHFLAG) -I../common --compiler-options -fPIC $(DEFS) -Wno-deprecated-gpu-targets
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
	$(FC) $(BASE_FFLAGS) $(DC_MODE_FLAGS) -module $(DCBLD) -c $< -o $@

$(DCBIN): $(DCOBJS)
	$(FC) $(BASE_FFLAGS) $(DC_MODE_FLAGS) $(DCOBJS) -o $@

$(CPPBLD)/%.o: %.cu ../common/gpu_rt.h | $(CPPBLD)
	$(NVCC) $(CUFLAGS) -c $< -o $@

$(CPPBIN): $(CUOBJS)
	$(NVCC) $(CUOBJS) -o $@ $(GPU_ARCHFLAG) -Wno-deprecated-gpu-targets

# args: nx_pts ny_pts nz reps warm hdrift%
ARGS  ?= 473 297 30 200 10 25
VARGS ?= 473 297 30 8 2 25
run-dc: dc
	@./$(DCBIN) $(ARGS)
run-cpp: cpp
	@./$(CPPBIN) $(ARGS)

# Prove the macro'd data layer is numerically inert: OpenACC (GPU) writes a
# reference; the CPU build recomputes and cross-checks it to FMA level.
verify:
	@$(MAKE) --no-print-directory dc DATA=acc
	@DC_DUMP=h_ref.bin ./dc_acc $(VARGS)
	@$(MAKE) --no-print-directory dc DATA=none
	@DC_REF=h_ref.bin  ./dc_none $(VARGS)

clean:
	@rm -rf build dc_acc dc_omp dc_none cpp_cuda cpp_hip h_ref.bin *.mod
