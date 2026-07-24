# meke — new-style build-mode layout (ported from meke_benchmark, pilot pattern).
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

BASE_FFLAGS := $(FFLAGS_BASE) -I../common
DC_SRCS     := constants.F90 meke_state.F90 meke.F90 drivers/dc_main.F90
DCAB_SRCS   := constants.F90 meke_state.F90 meke.F90 drivers/dc_ab.F90

# ---- do-concurrent build mode: DATA selects the device data layer ----------
# ⚠ `-gpu=tripcount:host` is MANDATORY on the acc path (NVIDIA TPR #38714): the
# many-loop MEKE region otherwise reads trip counts from the device copy and
# inserts a per-kernel data refresh -- ~2x slower. Copied from production flags.
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

# ---- A/B build mode: faithful vs fused do concurrent, same DATA layers -------
DCABBIN  := dc_ab_$(DATA)
DCABBLD  := build/dc_ab_$(DATA)
DCABOBJS := $(addprefix $(DCABBLD)/,$(notdir $(DCAB_SRCS:.F90=.o)))

# ---- native C++ build mode: BACKEND selects the GPU runtime ----------------
# cpp_main.cu lives in drivers/ but includes meke_args.h from THIS dir, so -I.
# is needed in addition to -I../common (gpu_rt.h).
CU_SRCS := meke_kernel.cu drivers/cpp_main.cu
CUFLAGS := -O3 $(GPU_ARCHFLAG) -I../common -I. --compiler-options -fPIC
ifeq ($(BACKEND),hip)
  CUFLAGS += -DUSE_HIP
endif
CPPBIN  := cpp_$(BACKEND)
CPPBLD  := build/cpp_$(BACKEND)
CUOBJS  := $(addprefix $(CPPBLD)/,$(notdir $(CU_SRCS:.cu=.o)))

# ---- cmp build mode: SHARED single-binary head-to-head ----------------------
# opt-DC (meke_step_ext_fused) vs opt-CUDA (meke_opt_launch) on the SAME device
# arrays via `!$acc host_data use_device`. Mixed toolchain: the Fortran (driver +
# bridge + compute) is nvfortran with the acc data layer but NO -cuda (the bridge
# declares plain host arrays; -cuda would type-check them as device -> S-0528);
# opt_kernel.cu is nvcc. `-cuda` appears ONLY at the link line. acc-only by
# nature (the CUDA leg needs device pointers), so not DATA-parameterized.
CMP_FF_FLAGS   := -stdpar=gpu -acc=gpu -gpu=$(ARCH),mem:separate -gpu=tripcount:host -DDC_DATA_ACC
CMP_LINK_FLAGS := -stdpar=gpu -acc=gpu -gpu=$(ARCH),mem:separate -cuda
CMP_FSRCS := constants.F90 meke_state.F90 meke.F90 drivers/meke_bridge.F90 drivers/cmp_main.F90
CMPBLD    := build/cmp
CMPBIN    := cmp_acc
CMP_FOBJS := $(addprefix $(CMPBLD)/,$(notdir $(CMP_FSRCS:.F90=.o)))
CMP_CUOBJ := $(CMPBLD)/opt_kernel.o

vpath %.F90 . drivers
vpath %.cu  . drivers

.PHONY: dc dcab cpp cmp run-dc run-dcab run-cpp run-cmp verify clean
.NOTPARALLEL:            # honour the DC_SRCS dependency order

dc: $(DCBIN)
dcab: $(DCABBIN)
cpp: $(CPPBIN)
cmp: $(CMPBIN)

$(DCBLD) $(DCABBLD) $(CPPBLD) $(CMPBLD):
	@mkdir -p $@

$(DCBLD)/%.o: %.F90 ../common/directives.h | $(DCBLD)
	$(FC) $(BASE_FFLAGS) $(DC_MODE_FLAGS) $(MODFLAG) $(DCBLD) -c $< -o $@

$(DCBIN): $(DCOBJS)
	$(FC) $(BASE_FFLAGS) $(DC_MODE_FLAGS) $(DCOBJS) -o $@

$(DCABBLD)/%.o: %.F90 ../common/directives.h | $(DCABBLD)
	$(FC) $(BASE_FFLAGS) $(DC_MODE_FLAGS) $(MODFLAG) $(DCABBLD) -c $< -o $@

$(DCABBIN): $(DCABOBJS)
	$(FC) $(BASE_FFLAGS) $(DC_MODE_FLAGS) $(DCABOBJS) -o $@

$(CPPBLD)/%.o: %.cu ../common/gpu_rt.h meke_args.h | $(CPPBLD)
	$(NVCC) $(CUFLAGS) -c $< -o $@

$(CPPBIN): $(CUOBJS)
	$(NVCC) $(CUOBJS) -o $@ $(GPU_ARCHFLAG)

# ---- cmp: Fortran (NO -cuda) + nvcc opt_kernel, linked with -cuda -----------
$(CMPBLD)/%.o: %.F90 ../common/directives.h | $(CMPBLD)
	$(FC) $(BASE_FFLAGS) $(CMP_FF_FLAGS) -I. $(MODFLAG) $(CMPBLD) -c $< -o $@

$(CMP_CUOBJ): opt_kernel.cu ../common/gpu_rt.h meke_args.h | $(CMPBLD)
	$(NVCC) -O3 $(GPU_ARCHFLAG) -I../common -I. --compiler-options -fPIC -c $< -o $@

$(CMPBIN): $(CMP_FOBJS) $(CMP_CUOBJ)
	$(FC) $(BASE_FFLAGS) $(CMP_LINK_FLAGS) $(CMP_FOBJS) $(CMP_CUOBJ) -o $@

ARGS ?= 473 297 30 200 20
run-dc: dc
	@./$(DCBIN) $(ARGS)
run-dcab: dcab
	@./$(DCABBIN) $(ARGS)
run-cpp: cpp
	@./$(CPPBIN) $(ARGS)
# GPU RUN through the shared-card lock (never run the binary directly):
run-cmp: cmp
	@bash ../tmp_local_artifacts/gpu_run.sh meke-cmp ./$(CMPBIN) $(ARGS)

# Prove the macro'd data layer is numerically inert: OpenACC (GPU) writes a
# reference; the CPU build recomputes and cross-checks it to FMA level. Small
# reps -- the field is a deterministic function of the step count, so both runs
# MUST use identical args for the dump/ref to be comparable.
VERIFY_ARGS := 473 297 30 20 5
verify:
	@$(MAKE) --no-print-directory dc DATA=acc
	@DC_DUMP=meke_ref.bin ./dc_acc $(VERIFY_ARGS)
	@$(MAKE) --no-print-directory dc DATA=none
	@DC_REF=meke_ref.bin  ./dc_none $(VERIFY_ARGS)

# ---- serial do-loop build mode (committed serialdo/ sources) ---------------
-include ../common/serialdo.mk

clean: clean-serialdo
	@rm -rf build dc_acc dc_omp dc_none dc_ab_acc dc_ab_omp dc_ab_none cpp_cuda cpp_hip cmp_acc meke_ref.bin *.mod
