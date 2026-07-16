-include ../../config.mk
# Barotropic substep: `do concurrent` vs CUDA C (faithful / fused / graph).
#
#   source ../../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
#   make && make run
#
# btstep.F90 is a TRANSCRIPTION of production's live closed-basin path,
# not a verbatim extract -- see its header for what was dropped and why.

FC   ?= nvfortran
NVCC ?= nvcc
ARCH   ?= cc70
NVARCH ?= sm_70

# `tripcount:host` and `-acc=gpu` copied from production's actual flags:
#   <model>/build/CMakeFiles/core_model_objs.dir/flags.make
#   -Mfree -Mbackslash -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate \
#   -gpu=tripcount:host -O3 -fast -module modules -fPIC
GPUFLAGS := -gpu=$(ARCH),mem:separate -gpu=tripcount:host
CUFLAGS  := -O3 $(GPU_ARCHFLAG) --compiler-options -fPIC
# NVTX ranges: `make NVTX=1`. Off by default -- an NVTX range around an ASYNC
# region marks HOST time, so the build we quote numbers from must not carry it
# (measured cost: ~2.5%). -cudalib=nvtx is how production links it
# (<model>/CMakeLists.txt:338-339).
#
# NOTE: this block MUST come before FFLAGS/LDFLAGS. They use `:=` (immediate
# expansion), so an ifdef placed after them expands to nothing and NVTX=1
# silently does nothing -- which is exactly what happened first time.
ifdef NVTX
NVTXFLAGS := -DUSE_NVTX -cudalib=nvtx
NVTXLIBS  := -cudalib=nvtx
endif

FFLAGS   := -O3 -fast -acc=gpu -stdpar=gpu $(GPUFLAGS) $(NVTXFLAGS)
# -cuda is LINK-ONLY: it also switches nvfortran into CUDA Fortran mode, which
# type-checks device attributes and rejects the host_data bridge with
# NVFORTRAN-S-0528 device attribute mismatch.
LDFLAGS  := -acc=gpu -stdpar=gpu $(GPUFLAGS) -cuda $(NVTXLIBS)

# PROFILING BUILD: signature-fix DC only. The CUDA port and the
# derived-type DC variant are commented out -- see btstep_bench.F90.full
# and git history to restore them.
OBJS := constants.o bt_state.o btnvtx.o btstep.o \
        btstep_plain.o btstep_fused.o btstep_kernel.o btstep_bench.o

.PHONY: all run native run-native verify regs clean
all: btstep_bench

constants.o: constants.F90
	$(FC) $(FFLAGS) -c $< -o $@
bt_state.o: bt_state.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@
btstep.o: btstep.F90 bt_state.o
	$(FC) $(FFLAGS) -c $< -o $@
btnvtx.o: btnvtx.F90
	$(FC) $(FFLAGS) -c $< -o $@

btstep_plain.o: btstep_plain.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@
btstep_noacc.o: btstep_noacc.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@

btstep_kernel.o: btstep_kernel.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

btstep_fused.o: btstep_fused.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@

btstep_bench.o: btstep_bench.F90 btstep.o btstep_plain.o btstep_fused.o
	$(FC) $(FFLAGS) -c $< -o $@
btstep_bench: $(OBJS)
	$(FC) $^ -o $@ $(LDFLAGS)

run: btstep_bench
	@./btstep_bench

# ---- NATIVE C++/CUDA driver: no Fortran, no OpenACC, no nvfortran ----------
# Links with nvcc alone. Reuses btstep_kernel.o -- the SAME object btstep_bench
# links -- so the two drivers cannot possibly be running different kernels.
btstep_kernel.o btstep_native.o: btstep_args.h
btstep_native.o: btstep_native.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@
btstep_native: btstep_native.o btstep_kernel.o
	$(NVCC) $(CUFLAGS) $^ -o $@

native: btstep_native

run-native: btstep_native
	@./btstep_native

# The whole point: same args to both, then the native driver checks its own
# mode-2 result against the Fortran-driven one bit-for-bit.
REF  := btstep_cuda_ref.bin
ARGS ?= 473 297 24 200 10
verify: btstep_bench btstep_native
	BTSTEP_REF_OUT=$(REF) ./btstep_bench  $(ARGS)
	BTSTEP_REF_IN=$(REF)  ./btstep_native $(ARGS)

regs: btstep_bench
	@cuobjdump -sass btstep_bench | awk '/Function : /{fn=$$3} /LDG/{ldg[fn]++} \
	   /LDL|STL/{loc[fn]++} /\/\*[0-9a-f]+\*\//{tot[fn]++} \
	   END{for (f in tot) if (f ~ /btstep|k_pass|k_swap/) \
	         printf "  %-52s LDG=%3d spills=%3d inst=%5d\n", f, ldg[f]+0, loc[f]+0, tot[f]}' | sort

clean:
	rm -f btstep_bench btstep_native $(REF) *.o *.mod
