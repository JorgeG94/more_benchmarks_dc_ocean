-include ../../config.mk
# Redi neutral diffusion: `do concurrent` vs a faithful CUDA C port.
#
#   source ../../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
#   make && make run
#
# ocean_redi.F90 is a VERBATIM extract (md5 of the production source is in
# its header). Re-extract if the production file moves.

FC   ?= nvfortran
NVCC ?= nvcc

# V100. ARCH and NVARCH must agree.
# NOTE: no trailing comments on these two -- make folds the spaces before '#'
# into the value, producing `-gpu=cc70   ,mem:separate` and a link error.
ARCH   ?= cc70
NVARCH ?= sm_70

# ⚠ NZ_STACK_MAX is production's, from
#   <model>/build/CMakeFiles/core_model_objs.dir/flags.make
#     Fortran_DEFINES = ... -DMODEL_NZ_STACK_MAX=128
# NOT the 256 the production header defaults to. `make STACK=64` sweeps it
# (nz=30 needs >= 2*nz+2 = 62). Measured: no effect -- see README.
STACK ?= 128

# ⚠ tripcount:host IS MANDATORY (NVHPC 26.5 regression, NVIDIA TPR #38714).
# Flags copied verbatim from production's flags.make:
#   -Mfree -Mbackslash -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate \
#   -gpu=tripcount:host -O3 -fast -module modules -fPIC
GPUFLAGS := -gpu=$(ARCH),mem:separate -gpu=tripcount:host
DEFS     := -DMODEL_NZ_STACK_MAX=$(STACK) -DWITH_CUDA
FFLAGS   := -Mfree -Mbackslash -stdpar=gpu -acc=gpu $(GPUFLAGS) -O3 -fast $(DEFS)
CUFLAGS  := -O3 $(GPU_ARCHFLAG) -DMODEL_NZ_STACK_MAX=$(STACK) --compiler-options -fPIC

# -cuda is LINK-ONLY: it also switches nvfortran into CUDA Fortran mode, which
# type-checks device attributes and rejects the host_data bridge with
# NVFORTRAN-S-0528 device attribute mismatch.
LDFLAGS := -stdpar=gpu -acc=gpu $(GPUFLAGS) -cuda

OBJS := constants.o grid.o ocean_metrics.o multilayer_cgrid_state.o \
        ocean_boundary_types.o ocean_eos.o ocean_redi.o ocean_redi_pre.o \
        redi_cuda.o redi_kernel.o redi_probe.o redi_bench.o

.PHONY: all run native native-run native-verify collapse regs sweep nsys clean
all: redi_bench

constants.o: constants.F90
	$(FC) $(FFLAGS) -c $< -o $@
grid.o: grid.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@
ocean_metrics.o: ocean_metrics.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@
multilayer_cgrid_state.o: multilayer_cgrid_state.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@
ocean_boundary_types.o: ocean_boundary_types.F90
	$(FC) $(FFLAGS) -c $< -o $@
ocean_eos.o: ocean_eos.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@
ocean_redi.o: ocean_redi.F90 ocean_eos.o grid.o ocean_metrics.o \
                  multilayer_cgrid_state.o ocean_boundary_types.o
	$(FC) $(FFLAGS) -c $< -o $@
ocean_redi_pre.o: ocean_redi_pre.F90 ocean_redi.o
	$(FC) $(FFLAGS) -c $< -o $@
redi_cuda.o: redi_cuda.F90 ocean_redi.o
	$(FC) $(FFLAGS) -c $< -o $@
redi_bench.o: redi_bench.F90 ocean_redi.o ocean_redi_pre.o redi_cuda.o
	$(FC) $(FFLAGS) -c $< -o $@
redi_kernel.o: redi_kernel.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

# No-op stand-in for the kernels: isolates the host_data lookup. See redi_probe.cu.
redi_probe.o: redi_probe.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

redi_bench: $(OBJS)
	$(FC) $^ -o $@ $(LDFLAGS)

run: redi_bench
	@./redi_bench

# ---- NATIVE C++/CUDA driver: no Fortran, no OpenACC, no host_data ----------
# Links with nvcc alone -- no nvfortran, no -stdpar, no -acc, no libacchost.
# It reuses redi_kernel.o, the SAME object redi_bench links, so the kernels
# cannot diverge between the two drivers.
native: redi_native

redi_native.o: redi_native.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

redi_native: redi_native.o redi_kernel.o
	$(NVCC) $^ -o $@ $(GPU_ARCHFLAG)

native-run: redi_native
	@./redi_native

# The verification pass: run the Fortran driver with dump=1 to write its own
# initial + final CUDA state, then run the native driver, which diffs both.
# Same kernel + same input + same rep count => must be bit-identical.
native-verify: redi_bench redi_native
	@./redi_bench 473 297 30 20 10 0 1
	@echo ""
	@./redi_native 473 297 30 20 10

# Every loop should report `auto-collapsed`: collapse(2) on the five 2-D loops,
# collapse(3) on redi_snapshot. Measured: all six do. Bug 1 is absent.
collapse:
	@$(FC) $(FFLAGS) -Minfo=stdpar -c ocean_redi.F90 -o /dev/null 2>&1 \
	   | grep -E 'auto-collapsed|Loop parallelized|^[a-z_]+:'
	@rm -f *.mod

# Registers + local-memory traffic per device function. Match the mangled name
# (*_gpu) -- the first `Used N registers` in a ptxas dump is cub::EmptyKernel
# (RESUME §1). NOTE the `!$acc routine seq` helpers appear as their OWN
# functions: nvfortran does not inline any of them. That is the finding.
regs: ocean_redi.o
	@cuobjdump -res-usage ocean_redi.o 2>/dev/null | grep -A1 "Function ocean_redi" \
	   | grep -E "Function|REG" | paste - - | sed 's/Function //;s/:\t/ /'
	@echo "--- local-memory traffic (LDL/STL) + call/ret per function ---"
	@cuobjdump -sass ocean_redi.o 2>/dev/null | awk '/Function : /{fn=$$3} \
	   /LDL/{l[fn]++} /STL/{s[fn]++} /LDG/{g[fn]++} /CALL|RET/{c[fn]++} \
	   END{for (f in l) if (f ~ /redi/) \
	         printf "  %-50s LDL=%4d STL=%4d LDG=%4d call/ret=%4d\n", f, l[f]+0, s[f]+0, g[f]+0, c[f]+0}' | sort

# Production configs from ~/analysis_gebco/*.nml. nx/ny are POINT COUNTS.
sweep: redi_bench
	@./redi_bench 108 137 30 10 10   # 10 km
	@./redi_bench 473 297 30 10 10   # 0.1 deg  <- the Redi/MEKE config
	@./redi_bench 240 560 30 10 10   # 0.25 deg
	@./redi_bench 945 594 30  5 10   # 0.05 deg

nsys: redi_bench
	@nsys profile -f true -o redi_nsys --trace=cuda ./redi_bench 473 297 30 5 3 >/dev/null 2>&1
	@nsys stats --report cuda_gpu_kern_sum redi_nsys.nsys-rep 2>/dev/null | tail -12

clean:
	rm -f redi_bench redi_native *.o *.mod *.nsys-rep *.sqlite *.bin
	rm -rf s64
