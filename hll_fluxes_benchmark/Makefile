-include ../config.mk
# HLL flux kernel: `do concurrent` variants vs a faithful CUDA C port.
#
#   source ../../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
#   make && make run
#
# Size and reps are env knobs (4096^2 is a stress size; nothing runs there):
#   FLUX_NPHYS_X=473 FLUX_NPHYS_Y=297 FLUX_REPS=2000 ./flux_bench
#   FLUX_DUMP_REF=1 ./flux_bench      # reference for ./flux_native (~1.2 GB at 4096^2)
#
# Variants. NOT all bit-identical -- the FLAT pair agrees with CUDA only to ~4e-12
# (FMA contraction). That used to read "all bit-identical" because flat/flatdc/dims
# shared an output array and only the last writer (dims) was ever checked; each
# variant now owns its outputs and the run prints one agreement line per variant.
# Numbers in README.md:
#   kernel_flux         as shipped (grid%-bounded dummies + call)  -> NOT collapsed
#   kernel_flux_nest    nested dc(j)/dc(i)                          -> same as fused
#   kernel_flux_dims    signature fix: plain nx/ny dummies          -> collapses
#   kernel_flux_acc     explicit OpenACC collapse(2)                -> ACC_VLEN
#   kernel_flux_flat    CPP-flattened body + OpenACC                -> INL_VLEN
#   kernel_flux_flatdc  flattened body + PLAIN do concurrent        -> the winner
#   flux_kernel.cu          faithful CUDA C port of the same kernel
#
# kernel_flux.F90 is byte-identical to the ocean model's production file
# (md5 1ea40efad8567a85914561db5bfc3a55). Re-copy if the original changes:
#   cp ../../<model>-sea-ice/src/core/coastal/kernels/structured/barotropic/kernel_flux.F90 .

FC   ?= nvfortran
NVCC ?= nvcc

# V100. ARCH and NVARCH must agree.
# NOTE: no trailing comments on these two -- make folds the spaces before '#'
# into the value, producing `-gpu=cc70   ,mem:separate` and a link error about
# a file named ",mem:separate".
ARCH   ?= cc70
NVARCH ?= sm_70

# vector_length for the two variants that take one. 512 measured best on V100.
ACC_VLEN ?= 512
INL_VLEN ?= 512

GPUFLAGS := -gpu=$(ARCH),mem:separate
CUFLAGS  := -O3 $(GPU_ARCHFLAG) --compiler-options -fPIC

# COMPILE flags -- deliberately NO -cuda here. See LDFLAGS.
# -Minfo is off by default: it is very noisy across 7 variants. To see the
# finding (`auto-collapsed ... collapse(2)`, and its ABSENCE on the shipped
# kernel) use the `collapse` target below, or:
#   make FFLAGS='-O2 -acc -stdpar=gpu -gpu=cc70,mem:separate -Minfo=accel,stdpar'
FFLAGS := -O3 -fast  -acc -stdpar=gpu $(GPUFLAGS)

# LINK flags. -cuda pulls in the CUDA runtime so the nvcc object resolves.
#
# WHY -cuda IS LINK-ONLY (load-bearing):
# -cuda also switches nvfortran into CUDA Fortran mode, which type-checks DEVICE
# ATTRIBUTES. Inside `!$acc host_data use_device(...)` the arrays carry the
# device attribute while a bind(C) interface declares plain host arrays, so
# compiling with -cuda fails:
#     NVFORTRAN-S-0528-Argument number 1 to ...: device attribute mismatch
# Without it the OpenACC host_data bridge is unaffected.
LDFLAGS := -acc -stdpar=gpu $(GPUFLAGS) -cuda

OBJS := flux_bench.o \
        kernel_flux.o kernel_flux_nest.o kernel_flux_dims.o \
        kernel_flux_acc.o kernel_flux_flat.o kernel_flux_flatdc.o \
        grid.o constants.o flux_kernel.o

.PHONY: all run native run-native regs collapse clean clean-ref
all: flux_bench

# ---- module order matters: constants <- grid <- kernels <- bench ----
constants.o: constants.F90
	$(FC) $(FFLAGS) -c $< -o $@

grid.o: grid.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@

kernel_flux.o: kernel_flux.F90 constants.o grid.o
	$(FC) $(FFLAGS) -c $< -o $@

kernel_flux_nest.o: kernel_flux_nest.F90 constants.o grid.o
	$(FC) $(FFLAGS) -c $< -o $@

# takes plain nx/ny dummies -> no grid dependency
kernel_flux_dims.o: kernel_flux_dims.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@

# these two take a compile-time vector length
kernel_flux_acc.o: kernel_flux_acc.F90 constants.o grid.o
	$(FC) $(FFLAGS) -DACC_VLEN=$(ACC_VLEN) -c $< -o $@

kernel_flux_flat.o: kernel_flux_flat.F90 constants.o grid.o
	$(FC) $(FFLAGS) -DINL_VLEN=$(INL_VLEN) -c $< -o $@

kernel_flux_flatdc.o: kernel_flux_flatdc.F90 constants.o grid.o
	$(FC) $(FFLAGS) -c $< -o $@

# flux_bench `use`s EVERY kernel module. A missing prerequisite here is exactly
# the `NVFORTRAN-F-0004-Unable to open MODULE file <name>.mod` error.
flux_bench.o: flux_bench.F90 kernel_flux.o kernel_flux_nest.o \
              kernel_flux_dims.o kernel_flux_acc.o \
              kernel_flux_flat.o kernel_flux_flatdc.o
	$(FC) $(FFLAGS) -c $< -o $@

flux_kernel.o: flux_kernel.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

flux_bench: $(OBJS)
	$(FC) $^ -o $@ $(LDFLAGS)

run: flux_bench
	@./flux_bench

# ---- NATIVE C++/CUDA driver: cudaMalloc + main(), no Fortran, no OpenACC ----
# Isolates the kernel from the OpenACC runtime, which the Fortran driver puts on
# the CUDA path via `host_data use_device` (a present-table lookup per array per
# call -- 9 arrays here). Links flux_kernel.o, the SAME object flux_bench links:
# the kernel is not duplicated, so the two drivers cannot drift apart.
#
# nvcc links this one. No nvfortran, no -acc, no -stdpar, no -cuda -- the whole
# point is that none of them are present.
flux_native.o: flux_native.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

flux_native: flux_native.o flux_kernel.o
	$(NVCC) $(CUFLAGS) $^ -o $@

native: flux_native

# The reference is only produced at flux_bench.F90's compiled-in size, so the
# native run below uses its defaults (4096^2, nghost=2) to match. Other sizes:
#   ./flux_native 473 297 2       (production-like; timing only, no reference)
run-native: flux_native flux_bench
	@FLUX_DUMP_REF=1 ./flux_bench
	@./flux_native

# Which variants auto-collapse? The ABSENCE of `auto-collapsed` on the shipped
# kernel -- while every other variant has it -- is the whole finding.
collapse:
	@for f in kernel_flux kernel_flux_nest kernel_flux_dims kernel_flux_flatdc; do \
	   printf '  %-24s : ' $$f; \
	   if $(FC) $(FFLAGS) -Minfo=accel,stdpar -c $$f.F90 -o /dev/null 2>&1 | grep -q 'auto-collapsed'; then \
	     echo 'collapse(2)'; else echo '*** NOT COLLAPSED ***'; fi; \
	 done
	@rm -f *.mod

# CUDA-side registers/occupancy. For the nvfortran side use -Minfo and match the
# mangled kernel name (`*_gpu`): the first `Used N registers` in a ptxas dump is
# cub::EmptyKernel, NOT your kernel. That misread cost a day -- see RESUME §1.
regs: flux_kernel.cu
	$(NVCC) $(CUFLAGS) -Xptxas -v -c $< -o /dev/null

clean:
	rm -f flux_bench flux_bench_inl flux_native *.o *.mod

# The reference dumps are ~1.2 GB each; `clean` leaves them alone deliberately.
clean-ref:
	rm -f flux_ref_*.bin
