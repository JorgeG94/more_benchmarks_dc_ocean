# btstep — new-style build-mode layout (barotropic substep, inner substep loop).
# The KERNEL is the unit; the strategy is a build MODE, not a directory:
#
#   make dc                 # do concurrent, DATA=acc  (OpenACC, NVIDIA GPU) [default]
#   make dc DATA=omp        # do concurrent, OpenMP target (NVIDIA/AMD/Intel GPU)
#   make dc DATA=none       # do concurrent on the CPU (-stdpar=multicore) -- NO CUDA
#   make cpp                # hand-written C++, BACKEND=cuda (nvcc, uses CUDA graphs)
#   make cpp BACKEND=hip    # hand-written C++, HIP (hipcc) -- structural, needs ROCm
#   make verify             # acc dumps a ref; CPU + OpenMP re-run and cross-check it
#   make run-dc / run-cpp / clean
#
# Toolchain + arch + BACKEND come from ../config.mk (the one file to edit).
# Compute is single-sourced; only directives.h (Fortran) / gpu_rt.h (C++) change.
# The production `!$acc kernels async(1)` idiom in btstep.F90 is honoured only
# under DATA=acc; under omp/none those lines are inert (bare do concurrent runs).
-include ../config.mk
FC      ?= nvfortran
NVCC    ?= nvcc
MODFLAG ?= -module
ARCH    ?= cc70
GPU_ARCHFLAG ?= -arch=sm_70
BACKEND ?= cuda

BASE_FFLAGS := $(FFLAGS_BASE) -I../common
# module order: constants -> bt_state -> btstep -> driver (OLD OBJS minus
# NVTX/plain/fused), driver last.
DC_SRCS     := constants.F90 bt_state.F90 btstep.F90 drivers/dc_main.F90

# ---- do-concurrent build mode: DATA selects the device data layer ----------
DATA ?= acc
ifeq ($(DATA),acc)
  DC_MODE_FLAGS := -stdpar=gpu -acc=gpu -gpu=$(ARCH),mem:separate -gpu=tripcount:host -DDC_DATA_ACC
else ifeq ($(DATA),omp)
  DC_MODE_FLAGS := -stdpar=gpu -mp=gpu -gpu=$(ARCH),mem:separate -DDC_DATA_OMP
else ifeq ($(DATA),none)
  DC_MODE_FLAGS := $(DC_HOST_FLAGS)
else
  $(error DATA must be acc | omp | none)
endif
DCBIN  := dc_$(DATA)
DCBLD  := build/dc_$(DATA)
DCOBJS := $(addprefix $(DCBLD)/,$(notdir $(DC_SRCS:.F90=.o)))

# ---- native C++ build mode: BACKEND selects the GPU runtime ----------------
CU_SRCS := btstep_kernel.cu drivers/cpp_main.cu
CUFLAGS := -O3 $(GPU_ARCHFLAG) -I. -I../common --compiler-options -fPIC
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

$(CPPBLD)/%.o: %.cu ../common/gpu_rt.h btstep_args.h | $(CPPBLD)
	$(NVCC) $(CUFLAGS) -c $< -o $@

$(CPPBIN): $(CUOBJS)
	$(NVCC) $(CUOBJS) -o $@ $(GPU_ARCHFLAG)

ARGS ?= 473 297 24 200 10
run-dc: dc
	@./$(DCBIN) $(ARGS)
run-cpp: cpp
	@./$(CPPBIN) $(ARGS)

# Prove the macro'd data layer is numerically inert: OpenACC (GPU) writes a
# reference; the CPU and OpenMP-target builds recompute and cross-check it.
# Small reps here -- the V100 is shared and timing is noisy; this checks numbers.
VARGS ?= 473 297 24 20 5
verify:
	@$(MAKE) --no-print-directory dc DATA=acc
	@DC_DUMP=eta_ref.bin ./dc_acc $(VARGS)
	@$(MAKE) --no-print-directory dc DATA=none
	@DC_REF=eta_ref.bin  ./dc_none $(VARGS)
	@$(MAKE) --no-print-directory dc DATA=omp
	@DC_REF=eta_ref.bin  ./dc_omp  $(VARGS)

clean:
	@rm -rf build dc_acc dc_omp dc_none cpp_cuda cpp_hip eta_ref.bin *.mod
