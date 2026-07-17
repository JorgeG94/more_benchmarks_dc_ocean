# ideal_benchmark — top-level whole-model RK2 harness over ALL 9 build-mode
# kernels: all-DC (do concurrent) vs all-CUDA (opt launchers via host_data), one
# binary, one hot-GPU loop.
#
#   make            # build ./rk2
#   make run        # build + run THROUGH THE GPU LOCK (473 297 30 100 10)
#   make clean
#
# Design (see README.md): each kernel + its thin rk2_<k> wrapper is compiled in
# an ISOLATED build dir (its own -module dir) and linked into a self-contained
# SHARED LIBRARY build/lib<k>.so with `-Wl,-Bsymbolic` -- that both device-links
# the OpenACC code INSIDE each .so and binds each library's duplicated module
# symbols to its own definitions, so the 9 same-named-module kernels co-link.
#
# Phase 2 adds the CUDA side: each kernel's opt_kernel.cu (the optimized CUDA
# launcher) is compiled by nvcc and linked INTO that kernel's .so; the wrapper's
# rk2_<k>_stage_cuda calls it via `!$acc host_data use_device`. The Fortran
# wrappers are compiled WITHOUT -cuda (host_data would trip NVFORTRAN-S-0528);
# -cuda is LINK-ONLY (on each .so and the final exe) to pull in libcudart.
# kappa/epbl take a struct-of-knobs launcher, so a tiny flat C shim (shim_ks.cu,
# shim_epbl.cu) builds the struct C-side and exposes a uniform pointer-list.
-include ../config.mk
FC       ?= nvfortran
NVCC     ?= nvcc
ARCH     ?= cc70
GPU_ARCHFLAG ?= -arch=sm_70
FFLAGS_BASE ?= -Mfree -Mbackslash -O3 -fast

COMMON   := ../common
DC_FLAGS := -stdpar=gpu -acc=gpu -gpu=$(ARCH),mem:separate -gpu=tripcount:host -DDC_DATA_ACC
FF       := $(FFLAGS_BASE) -fPIC $(DC_FLAGS) -I$(COMMON)
CUBASE   := -O3 $(GPU_ARCHFLAG) --compiler-options -fPIC -I$(COMMON)

# per-kernel Fortran DEFS (mirror each kernel's own Makefile)
D_REDI   := -DMODEL_NZ_STACK_MAX=128 -DWITH_CUDA
D_KAPPA  := -DMODEL_NZ_STACK_MAX=128
D_ALE    := -DMODEL_NZ_STACK_MAX=128
D_EPBL   := -DMODEL_NZ_STACK_MAX=128 -DEPBL_LB_THREADS=128 -DEPBL_LB_BLOCKS_PER_SM=4
# per-kernel nvcc DEFS for opt_kernel.cu (from each kernel's cmp/cpp build)
CUD_REDI  := -DMODEL_NZ_STACK_MAX=128
CUD_KAPPA := -DMODEL_NZ_STACK_MAX=128
CUD_ALE   := -DMODEL_NZ_STACK_MAX=128 -Wno-deprecated-gpu-targets
CUD_EPBL  := -DMODEL_NZ_STACK_MAX=128 -DEPBL_LB_THREADS=128 -DEPBL_LB_BLOCKS_PER_SM=4

KLIBS := build/libcontinuity.so build/libredi.so build/libkappa.so build/libale.so \
         build/libbtstep.so build/libepbl.so build/libmeke.so build/libhll.so build/libhvisc.so

BATHY := gabight_bathy_0p1_473x297.f64

.PHONY: all run clean
.NOTPARALLEL:

all: rk2 $(BATHY)

# ---- final link: executable against the 9 self-contained kernel libraries ---
# -cuda is LINK-ONLY here (pulls libcudart for the opt launchers in the .so's).
rk2: build/rk2_main.o $(KLIBS)
	$(FC) $(FF) -cuda build/rk2_main.o $(KLIBS) -Wl,-rpath,$(CURDIR)/build -o $@
	@echo "built ./rk2"

build/rk2_main.o: rk2_main.F90 | build
	$(FC) $(FF) -c $< -o $@

build:
	@mkdir -p build

# stage the EPBL bathymetry next to the binary (epbl reads it from CWD)
$(BATHY):
	@ln -sf ../epbl/$(BATHY) $(BATHY)

# ---- per-kernel isolated build -> self-contained shared library ------------
# $(1)=name $(2)=kernel dir $(3)=ordered F srcs $(4)=F defs $(5)=wrapper
# $(6)=nvcc defs for opt_kernel.cu  $(7)=optional flat C shim (.cu in this dir)
define KBUILD
build/lib$(1).so: wrap/$(5) $(7) | build
	@mkdir -p build/$(1)
	@for f in $(3); do \
	  echo "  FC[$(1)] $$$$f"; \
	  $(FC) $(FF) $(4) -I../$(2) -module build/$(1) -c ../$(2)/$$$$f -o build/$(1)/`basename $$$$f .F90`.o || exit 1; \
	done
	@echo "  FC[$(1)] $(5)"
	@$(FC) $(FF) $(4) -I../$(2) -module build/$(1) -c wrap/$(5) -o build/$(1)/$(5:.F90=.o)
	@echo "  NVCC[$(1)] opt_kernel.cu"
	@$(NVCC) $(CUBASE) $(6) -I../$(2) -c ../$(2)/opt_kernel.cu -o build/$(1)/opt_kernel.o
	@if [ -n "$(7)" ]; then echo "  NVCC[$(1)] $(7)"; \
	  $(NVCC) $(CUBASE) $(6) -I../$(2) -c $(7) -o build/$(1)/`basename $(7) .cu`.o; fi
	$(FC) $(FF) -cuda -shared -Wl,-Bsymbolic build/$(1)/*.o -o $$@
endef

$(eval $(call KBUILD,continuity,continuity_layered,constants.F90 grid.F90 continuity_layered.F90,,rk2_continuity.F90,,))
$(eval $(call KBUILD,redi,redi,constants.F90 grid.F90 ocean_metrics.F90 multilayer_cgrid_state.F90 ocean_boundary_types.F90 ocean_eos.F90 ocean_redi.F90,$(D_REDI),rk2_redi.F90,$(CUD_REDI),))
$(eval $(call KBUILD,kappa,kappa_shear,constants.F90 ks_state.F90 ks.F90,$(D_KAPPA),rk2_kappa.F90,$(CUD_KAPPA),shim_ks.cu))
$(eval $(call KBUILD,ale,ale_remap,constants.F90 remap_state.F90 kernel_remap.F90 ale_remap.F90,$(D_ALE),rk2_ale.F90,$(CUD_ALE),))
$(eval $(call KBUILD,btstep,btstep,constants.F90 bt_state.F90 btstep.F90,,rk2_btstep.F90,,))
$(eval $(call KBUILD,epbl,epbl,constants.F90 grid.F90 epbl_stubs.F90 ocean_epbl.F90,$(D_EPBL),rk2_epbl.F90,$(CUD_EPBL),shim_epbl.cu))
$(eval $(call KBUILD,meke,meke,constants.F90 meke_state.F90 meke.F90,,rk2_meke.F90,,))
$(eval $(call KBUILD,hll,hll_fluxes,constants.F90 grid.F90 kernel_flux.F90,,rk2_hll.F90,,))
$(eval $(call KBUILD,hvisc,hvisc,constants.F90 hvisc_kernel.F90,,rk2_hvisc.F90,,))

ARGS ?= 473 297 30 100 10
run: all
	@bash ../tmp_local_artifacts/gpu_run.sh rk2-cuda ./rk2 $(ARGS)

clean:
	@rm -rf build rk2 rk2_dc $(BATHY) *.mod
