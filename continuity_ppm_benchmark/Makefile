# Continuity-PPM kernel: `do concurrent` vs hand-inlined vs a faithful CUDA C port.
#
#   source ../../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
#   make && make run
#
# Variants (all agree; numbers in README.md):
#   continuity.F90        as shipped, verbatim from production
#   continuity_flat.F90   PPM helpers hand-inlined -> tests "do CALLS cost?"
#   continuity_acc.F90    same body + !$acc kernels async(1) -> tests
#                             "is the gap the LAUNCH MODEL?" (production idiom;
#                             see barotropic_substep.F90)
#   continuity_kernel.cu      faithful CUDA C port     -> tests "does CODEGEN cost?"
#
# continuity.F90 is a verbatim extract -- see its header for source line
# ranges. Re-extract if the production file moves; a drifted copy would
# benchmark a kernel that is not the one that ships.

FC   := nvfortran
NVCC := nvcc

# V100. ARCH and NVARCH must agree.
# NOTE: no trailing comments on these two -- make folds the spaces before '#'
# into the value, producing `-gpu=cc70   ,mem:separate` and a link error about
# a file named ",mem:separate".
ARCH   ?= cc70
NVARCH ?= sm_70

# mem:separate + manual deep copy in the driver, matching hll_fluxes_benchmark.
# (An earlier revision used mem:managed to dodge the deep copy; that works but
# cannot hand device pointers to CUDA via host_data, so it can't host the port.)
GPUFLAGS := -gpu=$(ARCH),mem:separate
CUFLAGS  := -O3 -arch=$(NVARCH) --compiler-options -fPIC

# COMPILE flags -- deliberately NO -cuda here. See LDFLAGS.
FFLAGS := -O3 -fast -acc -stdpar=gpu $(GPUFLAGS)

# LINK flags. -cuda pulls in the CUDA runtime so the nvcc object resolves.
#
# WHY -cuda IS LINK-ONLY (load-bearing):
# -cuda also switches nvfortran into CUDA Fortran mode, which type-checks DEVICE
# ATTRIBUTES. Inside `!$acc host_data use_device(...)` the arrays carry the
# device attribute while a bind(C) interface declares plain host arrays, so
# compiling with -cuda fails:
#     NVFORTRAN-S-0528-Argument number 1 to ...: device attribute mismatch
# Without it the OpenACC host_data bridge is unaffected. (This bit once here.)
LDFLAGS := -acc -stdpar=gpu $(GPUFLAGS) -cuda

OBJS := constants.o grid.o continuity.o continuity_flat.o \
        continuity_acc.o continuity_kernel.o continuity_bench.o

.PHONY: all run collapse regs clean
all: continuity_bench

# ---- module order matters: constants <- grid <- continuity <- flat <- bench ----
constants.o: constants.F90
	$(FC) $(FFLAGS) -c $< -o $@

grid.o: grid.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@

continuity.o: continuity.F90 constants.o grid.o
	$(FC) $(FFLAGS) -c $< -o $@

continuity_flat.o: continuity_flat.F90 continuity.o
	$(FC) $(FFLAGS) -c $< -o $@

continuity_acc.o: continuity_acc.F90 continuity.o
	$(FC) $(FFLAGS) -c $< -o $@

continuity_bench.o: continuity_bench.F90 continuity.o continuity_flat.o \
                    continuity_acc.o
	$(FC) $(FFLAGS) -c $< -o $@

continuity_kernel.o: continuity_kernel.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

continuity_bench: $(OBJS)
	$(FC) $^ -o $@ $(LDFLAGS)

run: continuity_bench
	@./continuity_bench

# Which loops auto-collapse? Every 2-D loop should; the 1-D boundary loops
# (j=1:ny alone) have nothing to collapse and are expected to be absent.
collapse:
	@$(FC) $(FFLAGS) -Minfo=stdpar,accel -c continuity.F90 -o /dev/null 2>&1 \
	   | grep -E 'auto-collapsed|Loop parallelized' | head -40
	@rm -f *.mod

# Loads per thread. Match the mangled name (*_gpu) -- the first `Used N
# registers` in a ptxas dump is cub::EmptyKernel, NOT your kernel (RESUME §1).
regs: continuity_bench
	@cuobjdump -sass continuity_bench | awk '/Function : /{fn=$$3} \
	   /LDG/{ldg[fn]++} /\/\*[0-9a-f]+\*\//{tot[fn]++} \
	   END{for (f in ldg) if (f ~ /continuity|k_ppm|k_trans|k_div/) \
	         printf "  %-58s LDG=%3d  inst=%5d\n", f, ldg[f], tot[f]}' | sort

clean:
	rm -f continuity_bench *.o *.mod
