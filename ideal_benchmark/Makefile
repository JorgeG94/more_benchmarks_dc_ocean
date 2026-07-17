# ideal_benchmark — top-level whole-model RK2 harness over the 8-kernel ocean
# core (hll excluded). Two binaries from one set of wrappers:
#
#   rk2      GPU: naive-DC | opt-DC | hand-CUDA, opt or unopt, one hot loop
#            ./rk2 NXP NYP NZ NSTEPS NWARM [MODE=dc|cuda|both] [VERSION=opt|unopt]
#   rk2_cpu  CPU (pure do concurrent, DATA=none, no CUDA), DC-only
#            ./rk2_cpu NXP NYP NZ NSTEPS NWARM [VERSION=opt|unopt]
#
#   make            # build both
#   make rk2        # GPU binary
#   make rk2_cpu    # CPU binary  (serial: `make rk2_cpu DC_HOST_FLAGS=`)
#   make run        # rk2 through the GPU lock (473 297 30 100 10 both opt)
#   make run-cpu    # rk2_cpu directly (CPU-only, no lock)
#   make clean
#
# Isolation (see README): each kernel + its rk2_<k> wrapper -> a self-contained
# build/lib<k>.so with -Wl,-Bsymbolic (device-links the OpenACC code internally
# and binds each library's duplicated module symbols to itself). The GPU .so
# also links the kernel's opt_kernel.cu (opt CUDA), its <k>_kernel.cu (faithful
# CUDA) and, for kappa/epbl/btstep/meke, a flat struct-building shim; -cuda is
# link-only. The CPU .so (build_cpu/) is built with DATA=none host flags and
# -DRK2_NO_CUDA (all CUDA compiled out), linked without -cuda.
-include ../config.mk
FC       ?= nvfortran
NVCC     ?= nvcc
ARCH     ?= cc70
GPU_ARCHFLAG ?= -arch=sm_70
FFLAGS_BASE   ?= -Mfree -Mbackslash -O3 -fast
DC_HOST_FLAGS ?= -stdpar=multicore

COMMON   := ../common
DC_FLAGS := -stdpar=gpu -acc=gpu -gpu=$(ARCH),mem:separate -gpu=tripcount:host -DDC_DATA_ACC
FF_GPU   := $(FFLAGS_BASE) -fPIC $(DC_FLAGS) -I$(COMMON)
FF_CPU   := $(FFLAGS_BASE) -fPIC $(DC_HOST_FLAGS) -I$(COMMON) -DRK2_NO_CUDA
CUBASE   := -O3 $(GPU_ARCHFLAG) --compiler-options -fPIC -I$(COMMON)

# per-kernel Fortran DEFS (mirror each kernel's own Makefile)
D_REDI   := -DMODEL_NZ_STACK_MAX=128 -DWITH_CUDA
D_KAPPA  := -DMODEL_NZ_STACK_MAX=128
D_ALE    := -DMODEL_NZ_STACK_MAX=128
D_EPBL   := -DMODEL_NZ_STACK_MAX=128 -DEPBL_LB_THREADS=128 -DEPBL_LB_BLOCKS_PER_SM=4
# per-kernel nvcc DEFS for opt_kernel.cu / <k>_kernel.cu
CUD_REDI  := -DMODEL_NZ_STACK_MAX=128
CUD_KAPPA := -DMODEL_NZ_STACK_MAX=128
CUD_ALE   := -DMODEL_NZ_STACK_MAX=128 -Wno-deprecated-gpu-targets
CUD_EPBL  := -DMODEL_NZ_STACK_MAX=128 -DEPBL_LB_THREADS=128 -DEPBL_LB_BLOCKS_PER_SM=4

KNAMES := continuity redi ale hvisc btstep kappa epbl meke
KLIBS     := $(addprefix build/lib,$(addsuffix .so,$(KNAMES)))
KLIBS_CPU := $(addprefix build_cpu/lib,$(addsuffix .so,$(KNAMES)))

BATHY := gabight_bathy_0p1_473x297.f64

.PHONY: all run run-cpu clean
.NOTPARALLEL:

all: rk2 rk2_cpu $(BATHY)

# ---- GPU executable (both DC + CUDA), -cuda link-only ----------------------
rk2: build/rk2_main.o $(KLIBS)
	$(FC) $(FF_GPU) -cuda build/rk2_main.o $(KLIBS) -Wl,-rpath,$(CURDIR)/build -o $@
	@echo "built ./rk2"

build/rk2_main.o: rk2_main.F90 | build
	$(FC) $(FF_GPU) -c $< -o $@

# ---- CPU executable (DC-only, pure do concurrent, no CUDA) -----------------
rk2_cpu: build_cpu/rk2_main.o $(KLIBS_CPU)
	$(FC) $(FF_CPU) build_cpu/rk2_main.o $(KLIBS_CPU) -Wl,-rpath,$(CURDIR)/build_cpu -o $@
	@echo "built ./rk2_cpu"

build_cpu/rk2_main.o: rk2_main.F90 | build_cpu
	$(FC) $(FF_CPU) -c $< -o $@

build build_cpu:
	@mkdir -p $@

$(BATHY):
	@ln -sf ../epbl/$(BATHY) $(BATHY)

# ---- GPU kernel .so: F srcs + wrapper + opt_kernel.cu + <k>_kernel.cu + shim -
# $(1)=name $(2)=dir $(3)=F srcs $(4)=F defs $(5)=wrapper $(6)=nvcc defs
# $(7)=faithful <k>_kernel.cu  $(8)=optional flat shim (.cu in this dir)
define KBUILD_GPU
build/lib$(1).so: wrap/$(5) $(8) | build
	@mkdir -p build/$(1)
	@for f in $(3); do \
	  echo "  FC[$(1)] $$$$f"; \
	  $(FC) $(FF_GPU) $(4) -I../$(2) -module build/$(1) -c ../$(2)/$$$$f -o build/$(1)/`basename $$$$f .F90`.o || exit 1; \
	done
	@echo "  FC[$(1)] $(5)"
	@$(FC) $(FF_GPU) $(4) -I../$(2) -module build/$(1) -c wrap/$(5) -o build/$(1)/$(5:.F90=.o)
	@echo "  NVCC[$(1)] opt_kernel.cu + $(7)$(if $(8), + $(8))"
	@$(NVCC) $(CUBASE) $(6) -I../$(2) -c ../$(2)/opt_kernel.cu -o build/$(1)/cu_opt.o
	@$(NVCC) $(CUBASE) $(6) -I../$(2) -c ../$(2)/$(7) -o build/$(1)/cu_faith.o
	@if [ -n "$(8)" ]; then $(NVCC) $(CUBASE) $(6) -I../$(2) -c $(8) -o build/$(1)/cu_shim.o; fi
	$(FC) $(FF_GPU) -cuda -shared -Wl,-Bsymbolic build/$(1)/*.o -o $$@
endef

# ---- CPU kernel .so: F srcs + wrapper only, host flags, no CUDA ------------
define KBUILD_CPU
build_cpu/lib$(1).so: wrap/$(5) | build_cpu
	@mkdir -p build_cpu/$(1)
	@for f in $(3); do \
	  echo "  FC-cpu[$(1)] $$$$f"; \
	  $(FC) $(FF_CPU) $(4) -I../$(2) -module build_cpu/$(1) -c ../$(2)/$$$$f -o build_cpu/$(1)/`basename $$$$f .F90`.o || exit 1; \
	done
	@echo "  FC-cpu[$(1)] $(5)"
	@$(FC) $(FF_CPU) $(4) -I../$(2) -module build_cpu/$(1) -c wrap/$(5) -o build_cpu/$(1)/$(5:.F90=.o)
	$(FC) $(FF_CPU) -shared -Wl,-Bsymbolic build_cpu/$(1)/*.o -o $$@
endef

# name, dir, F srcs, F defs, wrapper, nvcc defs, faithful cu, shim
$(eval $(call KBUILD_GPU,continuity,continuity_layered,constants.F90 grid.F90 continuity_layered.F90,,rk2_continuity.F90,,layered_kernel.cu,))
$(eval $(call KBUILD_GPU,redi,redi,constants.F90 grid.F90 ocean_metrics.F90 multilayer_cgrid_state.F90 ocean_boundary_types.F90 ocean_eos.F90 ocean_redi.F90,$(D_REDI),rk2_redi.F90,$(CUD_REDI),redi_kernel.cu,))
$(eval $(call KBUILD_GPU,ale,ale_remap,constants.F90 remap_state.F90 kernel_remap.F90 ale_remap.F90,$(D_ALE),rk2_ale.F90,$(CUD_ALE),ale_kernel.cu,))
$(eval $(call KBUILD_GPU,hvisc,hvisc,constants.F90 hvisc_kernel.F90,,rk2_hvisc.F90,,hvisc_kernel.cu,))
$(eval $(call KBUILD_GPU,btstep,btstep,constants.F90 bt_state.F90 btstep.F90,,rk2_btstep.F90,,btstep_kernel.cu,shim_btstep.cu))
$(eval $(call KBUILD_GPU,kappa,kappa_shear,constants.F90 ks_state.F90 ks.F90,$(D_KAPPA),rk2_kappa.F90,$(CUD_KAPPA),ks_kernel.cu,shim_ks.cu))
$(eval $(call KBUILD_GPU,epbl,epbl,constants.F90 grid.F90 epbl_stubs.F90 ocean_epbl.F90,$(D_EPBL),rk2_epbl.F90,$(CUD_EPBL),epbl_kernel.cu,shim_epbl.cu))
$(eval $(call KBUILD_GPU,meke,meke,constants.F90 meke_state.F90 meke.F90,,rk2_meke.F90,,meke_kernel.cu,shim_meke.cu))

$(eval $(call KBUILD_CPU,continuity,continuity_layered,constants.F90 grid.F90 continuity_layered.F90,,rk2_continuity.F90))
$(eval $(call KBUILD_CPU,redi,redi,constants.F90 grid.F90 ocean_metrics.F90 multilayer_cgrid_state.F90 ocean_boundary_types.F90 ocean_eos.F90 ocean_redi.F90,$(D_REDI),rk2_redi.F90))
$(eval $(call KBUILD_CPU,ale,ale_remap,constants.F90 remap_state.F90 kernel_remap.F90 ale_remap.F90,$(D_ALE),rk2_ale.F90))
$(eval $(call KBUILD_CPU,hvisc,hvisc,constants.F90 hvisc_kernel.F90,,rk2_hvisc.F90))
$(eval $(call KBUILD_CPU,btstep,btstep,constants.F90 bt_state.F90 btstep.F90,,rk2_btstep.F90))
$(eval $(call KBUILD_CPU,kappa,kappa_shear,constants.F90 ks_state.F90 ks.F90,$(D_KAPPA),rk2_kappa.F90))
$(eval $(call KBUILD_CPU,epbl,epbl,constants.F90 grid.F90 epbl_stubs.F90 ocean_epbl.F90,$(D_EPBL),rk2_epbl.F90))
$(eval $(call KBUILD_CPU,meke,meke,constants.F90 meke_state.F90 meke.F90,,rk2_meke.F90))

ARGS ?= 473 297 30 100 10 both opt
run: rk2 $(BATHY)
	@bash ../tmp_local_artifacts/gpu_run.sh rk2-cuda ./rk2 $(ARGS)

CPU_ARGS ?= 120 80 20 20 5 opt
run-cpu: rk2_cpu $(BATHY)
	@./rk2_cpu $(CPU_ARGS)

clean:
	@rm -rf build build_cpu rk2 rk2_cpu $(BATHY) *.mod
