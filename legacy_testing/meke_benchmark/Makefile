-include ../../config.mk
# MEKE step: `do concurrent` vs CUDA C (faithful / fused / graph).
#
#   source ../../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
#   make && make run
#
# meke.F90's KERNEL BODIES are verbatim slices of production; its
# `meke_step_ext` driver is a transcription specialised to the one config that
# turns MEKE on. See that file's header. `make verbatim` re-checks the slices
# against the production file -- if it fails, this MRE has drifted.

FC   ?= nvfortran
NVCC ?= nvcc
ARCH   ?= cc70
NVARCH ?= sm_70

PROD := <model-root>/src/parameterizations/lateral/structured/ocean_meke.F90

# ⚠ `-gpu=tripcount:host` IS MANDATORY and these flags are copied verbatim from
# production's actual flags:
#   <model>/build/CMakeFiles/core_model_objs.dir/flags.make
#   -Mfree -Mbackslash -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate \
#   -gpu=tripcount:host -O3 -fast -module modules -fPIC
# Without tripcount:host, NVHPC 26.5 reads loop trip-counts from the DEVICE copy
# and inserts a per-kernel data refresh in multi-loop regions -- ~2x wrong
# (NVIDIA TPR #38714; <model> guards it in cmake/compiler_flags.cmake:66-72).
GPUFLAGS := -gpu=$(ARCH),mem:separate -gpu=tripcount:host
FFLAGS   := -Mfree -Mbackslash -O3 -fast -acc=gpu -stdpar=gpu $(GPUFLAGS)
CUFLAGS  := -O3 $(GPU_ARCHFLAG) --compiler-options -fPIC

# -cuda is LINK-ONLY: it also switches nvfortran into CUDA Fortran mode, which
# type-checks device attributes and rejects the host_data bridge with
# NVFORTRAN-S-0528 device attribute mismatch.
LDFLAGS  := -acc=gpu -stdpar=gpu $(GPUFLAGS) -cuda

OBJS := constants.o meke_state.o meke.o meke_dt.o \
        meke_fused.o meke_acc.o meke_fused_acc.o \
        meke_kernel.o meke_bench.o

.PHONY: all run native run-native verify-native sweep collapse regs verbatim clean
all: meke_bench

constants.o: constants.F90
	$(FC) $(FFLAGS) -c $< -o $@
meke_state.o: meke_state.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@
meke.o: meke.F90 meke_state.o
	$(FC) $(FFLAGS) -c $< -o $@
meke_dt.o: meke_dt.F90 meke_state.o
	$(FC) $(FFLAGS) -c $< -o $@
meke_fused.o: meke_fused.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@
meke_acc.o: meke_acc.F90 meke_state.o
	$(FC) $(FFLAGS) -c $< -o $@
meke_fused_acc.o: meke_fused_acc.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@
meke_kernel.o: meke_kernel.cu meke_args.h
	$(NVCC) $(CUFLAGS) -c $< -o $@
meke_bench.o: meke_bench.F90 meke.o meke_dt.o meke_fused.o meke_acc.o meke_fused_acc.o
	$(FC) $(FFLAGS) -c $< -o $@

meke_bench: $(OBJS)
	$(FC) $^ -o $@ $(LDFLAGS)

run: meke_bench
	@./meke_bench

# ---- native C++/CUDA driver: cudaMalloc + main(), no Fortran, no OpenACC ----
# Links ONLY meke_kernel.o -- the same object meke_bench links, so both drivers
# run the byte-identical kernels. No nvfortran, no -stdpar, no -acc, no libacc
# in the process at all: that absence is what the binary exists to measure.
# Built with nvcc, so the Fortran flags above (incl. -gpu=tripcount:host, which
# is an nvfortran loop-lowering flag) do not apply and are not needed here.
native: meke_native

meke_native.o: meke_native.cu meke_args.h
	$(NVCC) $(CUFLAGS) -c $< -o $@

meke_native: meke_native.o meke_kernel.o
	$(NVCC) -O3 $(GPU_ARCHFLAG) $^ -o $@

run-native: meke_native
	@./meke_native

# End-to-end correctness: run the Fortran driver with dumps on, then diff the
# native driver's own fields against them (init AND final -- see meke_native.cu).
verify-native: meke_bench meke_native
	@MEKE_DUMP=1 ./meke_bench > /dev/null && ./meke_native --verify

# The gap is a strong function of TOTAL CELL COUNT, so sweep the real configs.
# nx/ny are POINT COUNTS from ~/analysis_gebco/*.nml (extent = nx*dx).
# min-of-N: this node is shared and co-tenant jobs produce 20-40x outliers.
sweep: meke_bench
	@N=$(or $(N),3) ./sweep.sh

collapse: meke.F90 meke_fused.F90 meke_dt.F90 meke_acc.F90 meke_fused_acc.F90
	@for f in $^; do echo "=== $$f ==="; \
	  $(FC) $(FFLAGS) -Minfo=accel,stdpar -c $$f -o /dev/null 2>&1 | \
	    grep -E "auto-collapsed|collapse|Loop parallelized|Loop run sequentially"; done

# ⚠ Match the MANGLED kernel name (`*_gpu`). The first `Used N registers` in a
# ptxas dump is cub::EmptyKernel, not your kernel -- that misread cost a day.
regs: meke_bench
	@cuobjdump -sass meke_bench | awk '/Function : /{fn=$$3} /LDG/{ldg[fn]++} \
	   /LDL|STL/{loc[fn]++} /\/\*[0-9a-f]+\*\//{tot[fn]++} \
	   END{for (f in tot) if (f ~ /meke|k_fused|k_length|k_mass|k_drag|k_uflux|k_vflux|k_div/) \
	         printf "  %-58s LDG=%3d spills=%3d inst=%5d\n", f, ldg[f]+0, loc[f]+0, tot[f]}' | sort

# Re-slice the verbatim kernel ranges from production and diff. A failure here
# means the MRE has drifted from the shipped kernel and every number is void.
verbatim:
	@{ sed -n '540,577p' $(PROD); echo; sed -n '583,617p' $(PROD); echo; \
	   sed -n '723,962p' $(PROD); echo; sed -n '1098,1215p' $(PROD); } > .verbatim_check.inc
	@if diff -q .verbatim_check.inc kernels_verbatim.inc >/dev/null; then \
	   echo "  VERBATIM OK: kernel slices match production"; rm -f .verbatim_check.inc; \
	 else \
	   echo "  *** MRE HAS DRIFTED FROM PRODUCTION -- every number is void ***"; \
	   diff .verbatim_check.inc kernels_verbatim.inc | head -20; \
	   rm -f .verbatim_check.inc; exit 1; fi

clean:
	rm -f meke_bench meke_native *.o *.mod .verbatim_check.inc meke_ref_*.bin
