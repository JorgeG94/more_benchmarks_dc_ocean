# kappa_shear — new-style build-mode layout (kappa-shear JHL08 column kernel).
# The KERNEL is the unit; the strategy is a build MODE, not a directory:
#
#   make dc                 # do concurrent, DATA=acc  (OpenACC, NVIDIA GPU) [default]
#   make dc DATA=omp        # do concurrent, OpenMP target (NVIDIA/AMD/Intel GPU)
#   make dc DATA=none       # do concurrent on the CPU (-stdpar=multicore) -- NO CUDA
#   make cmp                # ONE FORTRAN BINARY: do concurrent + cuda_faithful
#                           #   + cuda_opt, same state, same device allocation.
#                           #   This is the head-to-head, and what the sweep uses.
#   make serialdo           # plain nested `do` baseline (binary `sdo`), any F2008 FC
#   make verify             # acc dumps a ref; CPU re-runs and cross-checks it
#   make verify-variant     # faithful vs VARIANT=block bit-identity gate
#   make run-dc / run-cmp / clean
#
#   make cpp                # historical control ONLY: the no-Fortran C++ main().
#   make cpp BACKEND=hip    #   It hand-mirrors build_state in C++ and so needs
#                           #   ~150 lines of verification to prove its inputs
#                           #   match. `cmp` supersedes it; the sweep ignores it.
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

# ===========================================================================
# SWEEP AXES -- every one of these is COMPILE-TIME, so each combination gets
# its own build dir and its own binary name. That is what lets tools/ks_sweep.sh
# walk a size/variant matrix without rebuilding what it already built.
#
# The tag suffix is EMPTY at the default settings, so the historical names
# (`dc_acc`, `dc_none`, `sdo`, `build/dc_acc`, ...) are unchanged and
# ../run_all.sh keeps working untouched.
# ===========================================================================

# Per-column stack bound. PRODUCTION IS 128 -- see constants.F90. Appended to
# BOTH the Fortran and CUDA compile lines so the two kernels size their
# per-thread column arrays identically. `NZSTACK=fit` is not a thing here: the
# sweep passes an explicit number (typically 128 = production, or nz+1 =
# frame-fitted) so the two policies are separable in the results.
NZSTACK ?= 128

# Which column-solve arrangement to build:
#   faithful  ks.F90        one thread per COLUMN, k sequential inside  (production)
#   block     ks_block.F90  one thread per BLOCK of VLEN columns; k sequential
#                           OUTER, lanes vectorised INNER.
VARIANT ?= faithful

# Lanes per column block (VARIANT=block only). This is the portability knob:
# 1 = one column per lane-group (must be bit-identical to `faithful`),
# 8/16 = AVX2/AVX-512 width, 32/64 = a warp/wavefront.
VLEN ?= 1

# 1 -> -DKS_BOUNDED_COPY: bound the `kappa_out = kappa` whole-array copy to
# 1..nz+1 instead of the DECLARED extent. See ks.F90:949 -- that copy is the
# ONLY reason wall time depends on NZ_STACK_MAX at all, so this knob and
# NZSTACK must be swept together or neither is interpretable.
BCOPY ?= 0

# 1 -> -DKS_COUNTERS: per-column outer-substep / inner-Picard iteration counts,
# written to ks%it_outer / ks%it_inner and summed by the driver. This is the
# WORK metric -- without it you cannot tell "more layers" from "more
# iterations" in an nz sweep. NEVER time a CNT=1 build: it adds registers to a
# kernel that is already register-bound. The sweep runs it separately at reps=1.
CNT ?= 0

# `-gpu=tripcount:host` on/off. DECLARED HERE, not next to the flag it controls,
# because the TAG block below tests it: make evaluates `ifneq` at parse time in
# file order, so a `?=` default placed after that test leaves the variable EMPTY
# when the test runs and every build silently gets the "-notc" suffix.
TRIPCOUNT ?= host

# ---- build tag: empty at defaults, else a unique suffix per config ----------
TAG :=
ifneq ($(NZSTACK),128)
  TAG := $(TAG)-z$(NZSTACK)
endif
ifneq ($(VARIANT),faithful)
  TAG := $(TAG)-$(VARIANT)$(VLEN)
endif
ifeq ($(BCOPY),1)
  TAG := $(TAG)-bc
endif
ifeq ($(CNT),1)
  TAG := $(TAG)-cnt
endif
ifneq ($(TRIPCOUNT),host)
  TAG := $(TAG)-notc
endif

# ---- preprocessor defines --------------------------------------------------
# DEFS_SIZE is the ONLY part that must match between the Fortran and the CUDA
# builds (it sizes the per-thread column arrays on both sides).
DEFS_SIZE := -DMODEL_NZ_STACK_MAX=$(NZSTACK)
DEFS      := $(DEFS_SIZE) -DKS_VLEN=$(VLEN)
ifeq ($(BCOPY),1)
  DEFS += -DKS_BOUNDED_COPY
endif
ifeq ($(CNT),1)
  DEFS += -DKS_COUNTERS
endif

# ---- variant source selection ----------------------------------------------
ifeq ($(VARIANT),faithful)
  KS_SRC := ks.F90
else ifeq ($(VARIANT),block)
  KS_SRC := ks_block.F90
else
  $(error VARIANT must be faithful | block)
endif

BASE_FFLAGS := $(FFLAGS_BASE) -I../common $(DEFS)
DC_SRCS     := constants.F90 ks_state.F90 $(KS_SRC) drivers/dc_main.F90

# ---- do-concurrent build mode: DATA selects the device data layer ----------
# `-gpu=tripcount:host` is MANDATORY for the acc path (NVIDIA TPR #38714 -- the
# multi-loop column region is ~2x wrong without it). Carried by the acc mode.
# `-gpu=tripcount:host` is MANDATORY on NVHPC 26.5 (NVIDIA TPR #38714): without
# it the multi-loop column region is timed ~2x wrong. It is exposed as a knob
# ONLY so the sweep can DEMONSTRATE that on each new machine + compiler version
# instead of assuming it -- a regression that is fixed, or that behaves
# differently on a newer arch, would otherwise be invisible and would show up as
# a mysteriously large DC-vs-CUDA gap. Default stays on; never turn it off for
# a headline number.
ifeq ($(TRIPCOUNT),host)
  TC_FLAG := -gpu=tripcount:host
else ifeq ($(TRIPCOUNT),off)
  TC_FLAG :=
else
  $(error TRIPCOUNT must be host | off)
endif

DATA ?= acc
ifeq ($(DATA),acc)
  DC_MODE_FLAGS := -stdpar=gpu -acc=gpu -gpu=$(ARCH),mem:separate $(TC_FLAG) -DDC_DATA_ACC
else ifeq ($(DATA),omp)
  DC_MODE_FLAGS := -stdpar=gpu -mp=gpu -gpu=$(ARCH),mem:separate -DDC_DATA_OMP
else ifeq ($(DATA),none)
  DC_MODE_FLAGS := $(DC_HOST_FLAGS)
else
  $(error DATA must be acc | omp | none)
endif
DCBIN  := dc_$(DATA)$(TAG)
DCBLD  := build/dc_$(DATA)$(TAG)
DCOBJS := $(addprefix $(DCBLD)/,$(notdir $(DC_SRCS:.F90=.o)))

# ---- native C++ build mode: BACKEND selects the GPU runtime ----------------
# CUVARIANT picks WHICH hand-written kernel:
#   faithful  ks_kernel.cu   the codegen baseline, frame sized to NZSTACK
#   opt       opt_kernel.cu  frame sized to OPT_NZMAX + 32-bit indexing, held
#                            bit-identical to faithful (see OPTIMIZATION.md)
# The opt kernel REFUSES to launch when nz+1 > OPT_NZMAX rather than corrupt
# results, so an nz sweep must raise OPT_NZMAX with nz (the sweep's `fit` stack
# policy does exactly that).
CUVARIANT ?= faithful
OPT_NZMAX ?= 48
MINBLOCKS ?= 0
IDX32     ?= 1

# 1 -> -DKS_FULL_COPY: the faithful CUDA kernel copies the DECLARED extent in
# its three whole-array assignments, exactly as Fortran does. THIS IS THE 1:1
# PAIRING KNOB -- it must be the mirror image of the Fortran's BCOPY:
#   baseline  : make cpp FULLCOPY=1  <->  make dc BCOPY=0   (both O(NZSTACK))
#   optimised : make cpp FULLCOPY=0  <->  make dc BCOPY=1   (both O(nz))
# Any other combination times the copy rather than the codegen. See the block
# at the top of ks_kernel.cu.
FULLCOPY ?= 0

ifeq ($(CUVARIANT),faithful)
  CU_KERNEL := ks_kernel.cu
  ifeq ($(FULLCOPY),1)
    CUVFLAGS := -DKS_FULL_COPY
    CUVTAG   := -fc
  else
    CUVFLAGS :=
    CUVTAG   :=
  endif
else ifeq ($(CUVARIANT),opt)
  CU_KERNEL := opt_kernel.cu
  CUVFLAGS  := -DKS_USE_OPT -DKS_OPT_NZMAX=$(OPT_NZMAX) \
               -DKS_MINBLOCKS=$(MINBLOCKS) -DIDX32=$(IDX32)
  CUVTAG    := -opt$(OPT_NZMAX)
  ifneq ($(MINBLOCKS),0)
    CUVTAG := $(CUVTAG)-mb$(MINBLOCKS)
  endif
  ifneq ($(IDX32),1)
    CUVTAG := $(CUVTAG)-idx64
  endif
else
  $(error CUVARIANT must be faithful | opt)
endif

CU_SRCS := $(CU_KERNEL) drivers/cpp_main.cu
CUFLAGS := -O3 $(GPU_ARCHFLAG) -I../common --compiler-options -fPIC \
           $(DEFS_SIZE) $(CUVFLAGS)
ifeq ($(BACKEND),hip)
  CUFLAGS += -DUSE_HIP
endif
CUTAG   := $(CUVTAG)
ifneq ($(NZSTACK),128)
  CUTAG := -z$(NZSTACK)$(CUVTAG)
endif
CPPBIN  := cpp_$(BACKEND)$(CUTAG)
CPPBLD  := build/cpp_$(BACKEND)$(CUTAG)
CUOBJS  := $(addprefix $(CPPBLD)/,$(notdir $(CU_SRCS:.cu=.o)))

# ===========================================================================
# cmp build mode -- ONE FORTRAN BINARY, BOTH TOOLCHAINS. This is the mode the
# sweep uses; `cpp` is kept only as the historical no-Fortran control.
#
# It builds the SAME drivers/dc_main.F90 with -DKS_WITH_CUDA and links BOTH
# hand-written CUDA kernels, so `do concurrent`, cuda_faithful and cuda_opt all
# run on the SAME device allocation (via `!$acc host_data use_device`) against
# the SAME state, in one process, and each emits its own RESULT line.
#
# Why this replaces the C++ main(): drivers/cpp_main.cu had to HAND-MIRROR
# build_state in C++, and a mirror that drifts silently measures a different
# problem. It also hits the cross-language libm trap (CLAUDE.md) -- nvfortran's
# and glibc's sin/exp disagree in the last ulp -- which is why that file carries
# ~150 lines of [RISK A]/[RISK B] machinery just to prove its inputs match.
# Launching from Fortran deletes both problems: there is one state builder.
CMP_FFLAGS  := $(FFLAGS_BASE) -I../common $(DEFS) -DKS_WITH_CUDA \
               -DKS_OPT_NZMAX=$(OPT_NZMAX) \
               -stdpar=gpu -acc=gpu -gpu=$(ARCH),mem:separate \
               $(TC_FLAG) -DDC_DATA_ACC
CMP_CUFLAGS := -O3 $(GPU_ARCHFLAG) -I../common --compiler-options -fPIC \
               $(DEFS_SIZE) -DKS_OPT_NZMAX=$(OPT_NZMAX) \
               -DKS_MINBLOCKS=$(MINBLOCKS) -DIDX32=$(IDX32)
ifeq ($(FULLCOPY),1)
  CMP_CUFLAGS += -DKS_FULL_COPY
endif
# -cuda is LINK-ONLY. It also switches nvfortran into CUDA-Fortran mode, which
# type-checks device attributes and rejects the host_data bridge's plain host
# array declarations (NVFORTRAN-S-0528). It must appear ONLY here.
# -lstdc++ is REQUIRED: nvfortran drives this link, and the nvcc-compiled
# objects reference the C++ runtime (`__cxa_guard_acquire` -- the guard emitted
# for a function-local static). nvfortran does not pull libstdc++ in on its own.
# It happened to resolve on one x86 box and failed on aarch64/GH200 with
#   undefined reference to symbol '__cxa_guard_acquire@@CXXABI_1.3'
#   libstdc++.so.6: error adding symbols: DSO missing from command line
CMP_LDFLAGS := -stdpar=gpu -acc=gpu -gpu=$(ARCH),mem:separate -cuda -lstdc++

CMPTAG := $(TAG)-o$(OPT_NZMAX)
ifeq ($(FULLCOPY),1)
  CMPTAG := $(CMPTAG)-fc
endif
ifneq ($(MINBLOCKS),0)
  CMPTAG := $(CMPTAG)-mb$(MINBLOCKS)
endif
CMPBIN   := cmp_acc$(CMPTAG)
CMPBLD   := build/cmp$(CMPTAG)
CMP_FSRC := constants.F90 ks_state.F90 $(KS_SRC) \
            drivers/ks_bridge.F90 drivers/dc_main.F90
CMP_FOBJ := $(addprefix $(CMPBLD)/,$(notdir $(CMP_FSRC:.F90=.o)))
CMP_CUOBJ := $(CMPBLD)/ks_kernel.o $(CMPBLD)/opt_kernel.o

# ---- REBUILD WHEN THE FLAGS CHANGE ----------------------------------------
# make keys on timestamps only, so `make dc FFLAGS_BASE=...` silently reuses
# objects built with the PREVIOUS flags and reports a number for a build that
# never happened. That is not hypothetical: it once turned a -O0 / -Kieee /
# -O3-fast floating-point comparison into the same stale binary three times,
# and made a correct kernel look like a compiler bug.
#
# Fix: stamp the resolved command line into the build dir and make every object
# depend on the stamp. The stamp is only touched when the contents change, so
# repeated builds with the same flags still do nothing.
define FLAGSTAMP
$(1)/.flags: ; @true
$(shell mkdir -p $(1) 2>/dev/null; printf '%s\n' '$(2)' > $(1)/.flags.new; \
        cmp -s $(1)/.flags.new $(1)/.flags 2>/dev/null || mv -f $(1)/.flags.new $(1)/.flags; \
        rm -f $(1)/.flags.new)
endef
$(eval $(call FLAGSTAMP,$(DCBLD),$(FC) $(BASE_FFLAGS) $(DC_MODE_FLAGS)))
$(eval $(call FLAGSTAMP,$(CPPBLD),$(NVCC) $(CUFLAGS)))
$(eval $(call FLAGSTAMP,$(CMPBLD),$(FC) $(CMP_FFLAGS) | $(NVCC) $(CMP_CUFLAGS)))

vpath %.F90 . drivers
vpath %.cu  . drivers

.PHONY: dc cpp cmp run-dc run-cpp run-cmp verify verify-variant verify-variant-fast clean
.NOTPARALLEL:            # honour the DC_SRCS dependency order

dc: $(DCBIN)
cpp: $(CPPBIN)
cmp: $(CMPBIN)

$(DCBLD) $(CPPBLD) $(CMPBLD):
	@mkdir -p $@

$(DCBLD)/%.o: %.F90 ../common/directives.h $(DCBLD)/.flags | $(DCBLD)
	$(FC) $(BASE_FFLAGS) $(DC_MODE_FLAGS) $(MODFLAG) $(DCBLD) -c $< -o $@

$(DCBIN): $(DCOBJS)
	$(FC) $(BASE_FFLAGS) $(DC_MODE_FLAGS) $(DCOBJS) -o $@

$(CPPBLD)/%.o: %.cu ../common/gpu_rt.h $(CPPBLD)/.flags | $(CPPBLD)
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
	@DC_DUMP=kd_ref.bin ./$(DCBIN) $(ARGS)
	@$(MAKE) --no-print-directory dc DATA=none
	@DC_REF=kd_ref.bin  ./dc_none$(TAG) $(ARGS)

# Bit-identity gate for the ARRANGEMENT rewrite: the faithful per-column kernel
# dumps a reference, the blocked kernel re-runs on the SAME device/host and
# cross-checks it. Run this at VLEN=1 FIRST (the transformation alone, which
# must be 0.0), then at the real VLEN (the vectorisation on top).
#   make verify-variant DATA=none VLEN=8
#
# STRICT_FF is deliberate. Under -O3 -fast the two arrangements are NOT
# bit-identical and that is not a defect: the lane-array form gives the compiler
# different FMA-contraction opportunities than the scalar form, and this kernel
# amplifies a 1-ulp input change through its convergence tests (the same
# mechanism README.md documents for DATA=none vs DATA=acc). Measured on V100 /
# nvfortran 26.5, 473x297x30:
#   -O0                    max rel = 0.0  at VLEN = 1, 2, 4, 8   <- exact
#   -O1 -Kieee -Mnofma     max rel = 0.0  at VLEN = 1, 2, 4, 8   <- exact
#   -O2/-O3                max rel ~ 1e-10, iteration counts still IDENTICAL
# At -O2 and above the residual is codegen, not arrangement: it moves with the
# optimiser (it even appears at VLEN=1 when the FAITHFUL reference is rebuilt at
# `-O2 -Mnovect`), and the integer iteration counts -- immune to rounding -- are
# identical at every width. ~1e-10 is also the amplification this kernel already
# shows for DATA=none vs DATA=acc (README.md), i.e. a property of an iterative
# solver with convergence tests, not of the blocking.
# So the gate that proves the TRANSFORMATION is right runs with contraction off;
# the production-flag agreement is reported separately by `verify-variant-fast`.
STRICT_FF ?= -Mfree -Mbackslash -O1 -Kieee -Mnofma
verify-variant:
	@echo "== bit-identity gate (strict FP: $(STRICT_FF)) =="
	@$(MAKE) --no-print-directory dc VARIANT=faithful FFLAGS_BASE='$(STRICT_FF)'
	@DC_DUMP=ksvar_ref.bin ./dc_$(DATA) $(ARGS)
	@$(MAKE) --no-print-directory dc VARIANT=block VLEN=$(VLEN) FFLAGS_BASE='$(STRICT_FF)'
	@DC_REF=ksvar_ref.bin ./dc_$(DATA)-block$(VLEN) $(ARGS)

# Same comparison at PRODUCTION flags -- expected to differ at ~1e-10 rel for
# the reason above. Reported, not gated.
verify-variant-fast:
	@echo "== agreement at production flags (contraction ON) =="
	@$(MAKE) --no-print-directory dc VARIANT=faithful
	@DC_DUMP=ksvar_ref.bin ./dc_$(DATA) $(ARGS)
	@$(MAKE) --no-print-directory dc VARIANT=block VLEN=$(VLEN)
	@DC_REF=ksvar_ref.bin ./dc_$(DATA)-block$(VLEN) $(ARGS)

# ---- cmp build rules ------------------------------------------------------
# .NOTPARALLEL above keeps the module dependency order (constants -> ks_state ->
# ks -> ks_bridge -> dc_main).
$(CMPBLD)/%.o: %.F90 ../common/directives.h $(CMPBLD)/.flags | $(CMPBLD)
	$(FC) $(CMP_FFLAGS) $(MODFLAG) $(CMPBLD) -c $< -o $@

$(CMPBLD)/ks_kernel.o: ks_kernel.cu ../common/gpu_rt.h $(CMPBLD)/.flags | $(CMPBLD)
	$(NVCC) $(CMP_CUFLAGS) -c $< -o $@

$(CMPBLD)/opt_kernel.o: opt_kernel.cu ../common/gpu_rt.h $(CMPBLD)/.flags | $(CMPBLD)
	$(NVCC) $(CMP_CUFLAGS) -c $< -o $@

$(CMPBIN): $(CMP_FOBJ) $(CMP_CUOBJ)
	$(FC) $(CMP_FFLAGS) $(CMP_FOBJ) $(CMP_CUOBJ) -o $@ $(CMP_LDFLAGS)

run-cmp: cmp
	@../tmp_local_artifacts/gpu_run.sh ks-cmp ./$(CMPBIN) $(ARGS)

# ---- serial do-loop build mode (committed serialdo/ sources) ---------------
SD_TAG := $(TAG)
-include ../common/serialdo.mk

clean: clean-serialdo
	@rm -rf build dc_acc* dc_omp* dc_none* cpp_cuda* cpp_hip* cmp_acc* sdo* \
	        kd_ref.bin ksvar_ref.bin serialdo_ref.bin *.mod
