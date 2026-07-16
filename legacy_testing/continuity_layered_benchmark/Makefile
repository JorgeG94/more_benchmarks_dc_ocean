-include ../../config.mk
# Layered continuity-PPM kernel: `do concurrent` vs a faithful CUDA C port.
#
#   source ../../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
#   make && make run
#
# This is the kernel `ocean_continuity` actually wraps (11% of runtime with
# EPBL on, ~20% with it off) -- see layered_bench.F90's header. The barotropic
# benchmark next door measures a DIFFERENT, much smaller kernel.
#
# continuity_layered.F90 is a verbatim extract -- see its header for source
# line ranges. Re-extract if the production file moves.

FC   ?= nvfortran
NVCC ?= nvcc

# V100. ARCH and NVARCH must agree.
# NOTE: no trailing comments on these two -- make folds the spaces before '#'
# into the value, producing `-gpu=cc70   ,mem:separate` and a link error.
ARCH   ?= cc70
NVARCH ?= sm_70

GPUFLAGS := -gpu=$(ARCH),mem:separate
CUFLAGS  := -O3 $(GPU_ARCHFLAG) --compiler-options -fPIC

# COMPILE flags -- deliberately NO -cuda here. See LDFLAGS.
FFLAGS := -O3 -fast -acc -stdpar=gpu $(GPUFLAGS)

# LINK flags. -cuda pulls in the CUDA runtime so the nvcc object resolves.
# WHY -cuda IS LINK-ONLY: it also switches nvfortran into CUDA Fortran mode,
# which type-checks DEVICE ATTRIBUTES. Inside `!$acc host_data use_device(...)`
# the arrays carry the device attribute while the bind(C) interface declares
# plain host arrays, so compiling with -cuda fails:
#     NVFORTRAN-S-0528-Argument number 1 to ...: device attribute mismatch
LDFLAGS := -acc -stdpar=gpu $(GPUFLAGS) -cuda

OBJS := constants.o grid.o continuity_layered.o layered_kernel.o layered_bench.o

.PHONY: all run native run-native collapse regs clean
all: layered_bench layered_native

constants.o: constants.F90
	$(FC) $(FFLAGS) -c $< -o $@

grid.o: grid.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@

continuity_layered.o: continuity_layered.F90 constants.o grid.o
	$(FC) $(FFLAGS) -c $< -o $@

layered_bench.o: layered_bench.F90 continuity_layered.o
	$(FC) $(FFLAGS) -c $< -o $@

layered_kernel.o: layered_kernel.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

layered_bench: $(OBJS)
	$(FC) $^ -o $@ $(LDFLAGS)

run: layered_bench
	@./layered_bench

# NATIVE driver -- no Fortran, no OpenACC, no nvfortran runtime. nvcc drives the
# link. It shares layered_kernel.o with layered_bench, so the only thing that
# differs between the two CUDA timings is who owns the memory and whether the
# OpenACC runtime is on the launch path.
layered_native.o: layered_native.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

layered_native: layered_native.o layered_kernel.o
	$(NVCC) $(CUFLAGS) $^ -o $@

native: layered_native

# The dump/compare pair: layered_bench writes fh_ref.bin (its inputs and both
# its outputs), layered_native checks itself against it. Same args both sides or
# the reference is for a different problem and gets skipped.
run-native: layered_bench layered_native
	@./layered_bench 473 297 30 200 10 2 1
	@./layered_native 473 297 30 200 10 2

# All eleven loops should auto-collapse: collapse(3) on the five 3-D loops,
# collapse(2) on the boundary loops.
collapse:
	@$(FC) $(FFLAGS) -Minfo=stdpar -c continuity_layered.F90 -o /dev/null 2>&1 \
	   | grep -E 'auto-collapsed|Loop parallelized' | head -30
	@rm -f *.mod

# Loads/spills per kernel. Match the mangled name (*_gpu) -- the first
# `Used N registers` in a ptxas dump is cub::EmptyKernel (RESUME §1).
regs: layered_bench
	@cuobjdump -sass layered_bench | awk '/Function : /{fn=$$3} \
	   /LDG/{ldg[fn]++} /LDL|STL/{loc[fn]++} /\/\*[0-9a-f]+\*\//{tot[fn]++} \
	   END{for (f in ldg) if (f ~ /continuity|kL_/) \
	         printf "  %-58s LDG=%3d spills=%3d inst=%5d\n", f, ldg[f], loc[f]+0, tot[f]}' | sort

clean:
	rm -f layered_bench layered_native fh_ref.bin *.o *.mod
