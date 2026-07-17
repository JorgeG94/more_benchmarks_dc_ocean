# hvisc — new-style build-mode layout (pilot).
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
DC_SRCS     := constants.F90 hvisc_kernel.F90 drivers/dc_main.F90

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

# ---- A/B build mode: faithful vs fused do concurrent, same binary ----------
AB_SRCS := constants.F90 hvisc_kernel.F90 drivers/dc_ab.F90
ABBIN   := dc_ab_$(DATA)
ABBLD   := build/dc_ab_$(DATA)
ABOBJS  := $(addprefix $(ABBLD)/,$(notdir $(AB_SRCS:.F90=.o)))

# ---- cmp build mode: SHARED head-to-head, opt-DC vs opt-CUDA, one binary ----
# Both endpoints run on ONE set of device arrays via `!$acc host_data
# use_device`. The Fortran (compute + bridge) is built WITHOUT -cuda: that flag
# switches nvfortran into CUDA Fortran mode and type-checks device attributes,
# tripping NVFORTRAN-S-0528 inside host_data. -cuda is LINK-ONLY here.
CMP_F_SRCS := constants.F90 hvisc_kernel.F90 drivers/hvisc_bridge.F90 drivers/cmp_main.F90
CMP_BLD    := build/cmp
CMP_FOBJS  := $(addprefix $(CMP_BLD)/,$(notdir $(CMP_F_SRCS:.F90=.o)))
CMP_CUOBJ  := $(CMP_BLD)/opt_kernel.o
CMP_FFLAGS := -stdpar=gpu -acc=gpu -gpu=$(ARCH),mem:separate -gpu=tripcount:host -DDC_DATA_ACC
CMP_LDFLAGS:= -stdpar=gpu -acc=gpu -gpu=$(ARCH),mem:separate -cuda
CMPBIN     := cmp_acc

# ---- native C++ build mode: BACKEND selects the GPU runtime ----------------
CU_SRCS := hvisc_kernel.cu drivers/cpp_main.cu
CUFLAGS := -O3 $(GPU_ARCHFLAG) -I../common --compiler-options -fPIC
ifeq ($(BACKEND),hip)
  CUFLAGS += -DUSE_HIP
endif
CPPBIN  := cpp_$(BACKEND)
CPPBLD  := build/cpp_$(BACKEND)
CUOBJS  := $(addprefix $(CPPBLD)/,$(notdir $(CU_SRCS:.cu=.o)))

vpath %.F90 . drivers
vpath %.cu  . drivers

.PHONY: dc dcab cpp cmp run-dc run-dcab run-cpp run-cmp verify clean
.NOTPARALLEL:            # honour the DC_SRCS dependency order

dc: $(DCBIN)
dcab: $(ABBIN)
cpp: $(CPPBIN)
cmp: $(CMPBIN)

$(DCBLD) $(ABBLD) $(CPPBLD) $(CMP_BLD):
	@mkdir -p $@

$(DCBLD)/%.o: %.F90 ../common/directives.h | $(DCBLD)
	$(FC) $(BASE_FFLAGS) $(DC_MODE_FLAGS) $(MODFLAG) $(DCBLD) -c $< -o $@

$(DCBIN): $(DCOBJS)
	$(FC) $(BASE_FFLAGS) $(DC_MODE_FLAGS) $(DCOBJS) -o $@

$(ABBLD)/%.o: %.F90 ../common/directives.h | $(ABBLD)
	$(FC) $(BASE_FFLAGS) $(DC_MODE_FLAGS) $(MODFLAG) $(ABBLD) -c $< -o $@

$(ABBIN): $(ABOBJS)
	$(FC) $(BASE_FFLAGS) $(DC_MODE_FLAGS) $(ABOBJS) -o $@

$(CPPBLD)/%.o: %.cu ../common/gpu_rt.h | $(CPPBLD)
	$(NVCC) $(CUFLAGS) -c $< -o $@

$(CPPBIN): $(CUOBJS)
	$(NVCC) $(CUOBJS) -o $@ $(GPU_ARCHFLAG)

# cmp: Fortran WITHOUT -cuda, CUDA via nvcc, link with FC + -cuda (LINK-ONLY).
$(CMP_BLD)/%.o: %.F90 ../common/directives.h | $(CMP_BLD)
	$(FC) $(BASE_FFLAGS) $(CMP_FFLAGS) $(MODFLAG) $(CMP_BLD) -c $< -o $@

$(CMP_CUOBJ): opt_kernel.cu ../common/gpu_rt.h | $(CMP_BLD)
	$(NVCC) -O3 $(GPU_ARCHFLAG) -I../common -I. --compiler-options -fPIC -c $< -o $@

$(CMPBIN): $(CMP_FOBJS) $(CMP_CUOBJ)
	$(FC) $(BASE_FFLAGS) $(CMP_LDFLAGS) $(CMP_FOBJS) $(CMP_CUOBJ) -o $@

ARGS ?= 473 297 30 200 10
run-dc: dc
	@./$(DCBIN) $(ARGS)
run-dcab: dcab
	@./$(ABBIN) $(ARGS)
run-cpp: cpp
	@./$(CPPBIN) $(ARGS)
run-cmp: cmp
	@../tmp_local_artifacts/gpu_run.sh hvisc-cmp ./$(CMPBIN) $(ARGS)

# Prove the macro'd data layer is numerically inert: OpenACC (GPU) writes a
# reference; the CPU build recomputes and cross-checks it to FMA level.
verify:
	@$(MAKE) --no-print-directory dc DATA=acc
	@DC_DUMP=fh_ref.bin ./dc_acc $(ARGS)
	@$(MAKE) --no-print-directory dc DATA=none
	@DC_REF=fh_ref.bin  ./dc_none $(ARGS)

clean:
	@rm -rf build dc_acc dc_omp dc_none dc_ab_acc dc_ab_omp dc_ab_none cpp_cuda cpp_hip cmp_acc fh_ref.bin *.mod
