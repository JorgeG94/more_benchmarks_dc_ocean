# epbl — new-style build-mode layout (ported from ../epbl_benchmark).
# The KERNEL is the unit; the strategy is a build MODE, not a directory:
#
#   make dc                 # do concurrent, DATA=acc  (OpenACC, NVIDIA GPU) [default]
#   make dc DATA=omp        # do concurrent, OpenMP target (NVIDIA/AMD/Intel GPU)
#   make dc DATA=none       # do concurrent on the CPU (-stdpar=multicore) -- NO CUDA
#   make cpp                # hand-written C++/CUDA (cudaMalloc), no Fortran
#   make cpp BACKEND=hip    # hand-written C++, HIP (hipcc) -- structural, needs ROCm
#   make verify             # acc dumps a ref; CPU re-runs and cross-checks kd_int
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

# EPBL DEFS -- appended to BOTH flag sets so the Fortran preprocessor and nvcc
# see the same knobs. EPBL_LB_* set the CUDA `tuned` __launch_bounds__;
# MODEL_NZ_STACK_MAX mirrors the production stack cap.
DEFS := -DMODEL_NZ_STACK_MAX=128 -DEPBL_LB_THREADS=128 -DEPBL_LB_BLOCKS_PER_SM=4

BASE_FFLAGS := -Mfree -Mbackslash -O3 -fast -I../common $(DEFS)
# Module order matches ../epbl_benchmark's OBJS (minus ocean_epbl_plain), with
# drivers/dc_main.F90 replacing epbl_bench.F90 as the DC-only driver.
DC_SRCS     := constants.F90 grid.F90 epbl_stubs.F90 ocean_epbl.F90 drivers/dc_main.F90

# ---- do-concurrent build mode: DATA selects the device data layer ----------
# NOTE: -gpu=tripcount:host is LOAD-BEARING (NVIDIA TPR #38714); see the old
# Makefile. Kept on the acc path exactly as the pilot does.
DATA ?= acc
ifeq ($(DATA),acc)
  DC_MODE_FLAGS := -stdpar=gpu -acc=gpu -gpu=$(ARCH),mem:separate -gpu=tripcount:host -DDC_DATA_ACC
else ifeq ($(DATA),omp)
  DC_MODE_FLAGS := -stdpar=gpu -mp=gpu -gpu=$(ARCH),mem:separate -DDC_DATA_OMP
else ifeq ($(DATA),none)
  DC_MODE_FLAGS := -stdpar=multicore
else
  $(error DATA must be acc | omp | none)
endif
DCBIN  := dc_$(DATA)
DCBLD  := build/dc_$(DATA)
DCOBJS := $(addprefix $(DCBLD)/,$(notdir $(DC_SRCS:.F90=.o)))

# ---- native C++ build mode: BACKEND selects the GPU runtime ----------------
CU_SRCS := epbl_kernel.cu drivers/cpp_main.cu
# -I. so nvcc finds epbl_params.h (in this dir) when compiling drivers/cpp_main.cu.
CUFLAGS := -O3 $(GPU_ARCHFLAG) -I. -I../common --compiler-options -fPIC $(DEFS)
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

$(CPPBLD)/%.o: %.cu ../common/gpu_rt.h epbl_params.h | $(CPPBLD)
	$(NVCC) $(CUFLAGS) -c $< -o $@

$(CPPBIN): $(CUOBJS)
	$(NVCC) $(CUOBJS) -o $@ $(GPU_ARCHFLAG)

# Short verify-sized run by default (small reps -- this V100 is shared).
ARGS ?= 473 297 30 8 2 1 0 20 5
run-dc: dc
	@./$(DCBIN) $(ARGS)
run-cpp: cpp
	@./$(CPPBIN) $(ARGS)

# Prove the macro'd data layer is numerically inert: OpenACC (GPU) writes a
# reference; the CPU build recomputes and cross-checks it. The bar is kd_int
# (<1e-12 rel). MLD is reported too but EPBL branch sensitivity means a small
# % of columns may jump a whole layer -- expected, not a failure (see README).
verify:
	@$(MAKE) --no-print-directory dc DATA=acc
	@DC_DUMP=kd_ref.bin ./dc_acc $(ARGS)
	@$(MAKE) --no-print-directory dc DATA=none
	@DC_REF=kd_ref.bin  ./dc_none $(ARGS)

clean:
	@rm -rf build dc_acc dc_omp dc_none cpp_cuda cpp_hip kd_ref.bin *.mod
