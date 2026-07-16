-include ../config.mk
# kappa-shear (JHL08) column kernel: `do concurrent` vs hand-written CUDA C.
#
#   source ../../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
#   make && make run
#
# ks.F90's six column-solve helpers are VERBATIM production; the column
# kernel around them is a TRANSCRIPTION with the dead massless-merge path
# dropped -- see its header for exactly what went and why.

FC   ?= nvfortran
NVCC ?= nvcc

# V100. ARCH and NVARCH must agree.
# NOTE: no trailing comments on these two -- make folds the spaces before '#'
# into the value, producing `-gpu=cc70   ,mem:separate` and a link error.
ARCH   ?= cc70
NVARCH ?= sm_70

# Per-column stack bound. PRODUCTION IS 128 -- do not change it to make the
# numbers look better; `make NZSTACK=32` exists to MEASURE the sensitivity,
# which is a finding, not a fix.
NZSTACK ?= 128

# Copied verbatim from production's actual flags:
#   <model>/build/CMakeFiles/core_model_objs.dir/flags.make
#   -Mfree -Mbackslash -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate \
#   -gpu=tripcount:host -O3 -fast -module modules -fPIC
#   Fortran_DEFINES = ... -DMODEL_DOUBLE_PRECISION ... -DMODEL_NZ_STACK_MAX=128
#
# `-gpu=tripcount:host` IS MANDATORY. Without it NVHPC 26.5 reads loop
# trip-counts from the DEVICE copy and inserts a per-kernel data refresh in
# multi-loop regions -- ~2x wrong (NVIDIA TPR #38714). <model> already guards
# for it (cmake/compiler_flags.cmake:66-72).
# `make MAXREG=96` caps nvfortran's register allocation, the way nvcc's
# __launch_bounds__(128) does in ks_kernel.cu. Left alone nvfortran takes all
# 254 (the sm_70 cap) and lands on the WORST point of the occupancy/spill curve;
# 96 is 5.4% faster with results identical. NOT default here because the optimum
# is V100-tuned and must be re-swept per architecture -- see README.
MAXREG ?=
ifneq ($(MAXREG),)
REGFLAG := -gpu=maxregcount:$(MAXREG)
endif

# ⚠ maxregcount must be on the LINK line too, not just the compiles -- device
# linking re-runs ptxas and silently reverts the kernel to 254 without it.
GPUFLAGS := -gpu=$(ARCH),mem:separate -gpu=tripcount:host $(REGFLAG)
DEFS     := -DMODEL_NZ_STACK_MAX=$(NZSTACK)

FFLAGS := -Mfree -Mbackslash -stdpar=gpu -acc=gpu $(GPUFLAGS) -O3 -fast $(DEFS)

# -cuda is LINK-ONLY. It also switches nvfortran into CUDA Fortran mode, which
# type-checks device attributes: inside `!$acc host_data use_device(...)` the
# arrays carry the device attribute while the bind(C) interface declares plain
# host arrays, so COMPILING with -cuda fails with
#   NVFORTRAN-S-0528-Argument number 1 to ...: device attribute mismatch
LDFLAGS := -acc=gpu -stdpar=gpu $(GPUFLAGS) -cuda

CUFLAGS := -O3 $(GPU_ARCHFLAG) --compiler-options -fPIC $(DEFS)

# `make WARP=1` adds the warp-per-column variant. OFF by default: it answers a
# question that only exists if the Picard iteration count actually diverges
# across a warp. Measure the `warp work-balance ratio` the default build prints
# BEFORE building it. See README.
# `make BOUNDEDCOPY=1` bounds ks_solve_column's three whole-array assignments
# (`kappa_out = kappa`, `kq_tmp = k_q` x2) to 1..nz+1 instead of the declared
# 1..NZ_STACK_MAX+1. Results are unaffected (elements above nz+1 are never
# read); this exists to test whether those copies are the entire reason wall
# time depends on NZ_STACK_MAX.
ifdef BOUNDEDCOPY
FFLAGS += -DKS_BOUNDED_COPY
endif

ifdef WARP
FFLAGS  += -DHAVE_WARP
WARPOBJ := ks_warp.o
endif

OBJS := constants.o ks_state.o ks.o ks_kernel.o $(WARPOBJ) ks_bench.o

.PHONY: all run best native native-verify collapse regs occ ncu sweep clean

all: ks_bench

# The recommended configuration: one build flag + three lines of Fortran.
# 1.119x -> 1.048x vs hand-written CUDA, results bit-identical.
best:
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory MAXREG=96 BOUNDEDCOPY=1
	@./measure.sh "recommended (maxregcount:96 + bounded copies)" 4 473 297 30 20 10 1

constants.o: constants.F90
	$(FC) $(FFLAGS) -c $< -o $@
ks_state.o: ks_state.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@
ks.o: ks.F90 constants.o ks_state.o
	$(FC) $(FFLAGS) -c $< -o $@
ks_bench.o: ks_bench.F90 ks.o
	$(FC) $(FFLAGS) -c $< -o $@

ks_kernel.o: ks_kernel.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@
ks_warp.o: ks_warp.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

ks_bench: $(OBJS)
	$(FC) $^ -o $@ $(LDFLAGS)

run: ks_bench
	@./ks_bench

# ---------------------------------------------------------------------------
# NATIVE driver: cudaMalloc + main(), no Fortran, no OpenACC. Links the SAME
# ks_kernel.o that ks_bench does -- the kernel is never duplicated, so the two
# binaries cannot drift apart (daxpy_pure.cu's rule). nvcc drives the link; no
# nvfortran runtime is involved at all.
#
# Answers: does `!$acc host_data use_device` (11 present-table lookups per
# launch) cost anything? If native ties, every CUDA number here is already fair.
ks_native.o: ks_native.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

ks_native: ks_native.o ks_kernel.o
	$(NVCC) $(GPU_ARCHFLAG) $^ -o $@

native: ks_native
	@./ks_native

# The correctness gate. ks_native re-implements ks_bench's build_state in C++;
# if that mirror is wrong the native timing silently measures a different
# problem. So: run the Fortran with KS_DUMP, then have the native driver check
# its own INPUTS against it bit-for-bit, and its kd_int against Fortran-driven
# CUDA bit-for-bit (same kernel + same bits in => same bits out). Exits non-zero
# on failure. ~250 MB of dump, removed afterwards.
native-verify: ks_bench ks_native
	@KS_DUMP=$(CURDIR)/ks_ref.bin ./ks_bench 473 297 30 4 2 1 \
	   | grep -E 'KS_DUMP|max\|d\||agreement|sum kd_int'
	@KS_REF=$(CURDIR)/ks_ref.bin ./ks_native 473 297 30 4 2 1; st=$$?; \
	   rm -f $(CURDIR)/ks_ref.bin; exit $$st

# The ONE `do concurrent` under test. Expect collapse(2) over (j,i) and, more
# importantly, the `Local memory used for ...` line -- the per-column arrays do
# NOT fit in registers, contrary to the production source comment.
collapse:
	@$(FC) $(FFLAGS) -Minfo=accel,stdpar -c ks.F90 -o /dev/null 2>&1 \
	   | grep -E 'auto-collapsed|Loop parallelized|Local memory|routine seq' | head -20
	@rm -f *.mod

# Loads / SPILLS per kernel. LDL/STL here are the headline: a per-column solver
# whose state cannot live in registers pays local-memory traffic for it.
# ⚠ Match the mangled name (*_gpu for nvfortran). The first `Used N registers`
# in a ptxas dump is cub::EmptyKernel, NOT your kernel (RESUME §1 -- that
# misread cost a day).
regs: ks_bench
	@cuobjdump -sass ks_bench | awk '/Function : /{fn=$$3} \
	   /LDG/{ldg[fn]++} /LDL/{ldl[fn]++} /STL/{stl[fn]++} \
	   /\/\*[0-9a-f]+\*\//{tot[fn]++} \
	   END{for (f in tot) if (f ~ /kappa_shear|k_kappa|k_warp/) \
	         printf "  %-46s LDG=%5d LDL=%5d STL=%5d inst=%6d\n", \
	                f, ldg[f]+0, ldl[f]+0, stl[f]+0, tot[f]}' | sort

# Registers + stack frame straight from ptxas, per kernel, name-matched.
occ:
	@$(FC) $(FFLAGS) -Minfo=accel -Mcuda=ptxinfo -c ks.F90 -o /dev/null 2>&1 \
	   | grep -iE 'kappa_shear.*(regs|registers|stack)|spill' | head
	@$(NVCC) $(CUFLAGS) -Xptxas -v -c ks_kernel.cu -o /dev/null 2>&1 \
	   | grep -A2 -iE 'k_kappa_shear' | head -20
	@rm -f *.mod

# Occupancy + LANE EFFICIENCY (the divergence measurement) + local-memory
# traffic, for both kernels. ncu sees kernel names WITHOUT nvfortran's `_gpu`
# suffix that `cuobjdump -symbols` shows.
ncu: ks_bench
	ncu --kernel-name regex:'kappa_shear|k_kappa_shear' --launch-skip 2 --launch-count 2 \
	    --metrics smsp__thread_inst_executed_per_inst_executed.ratio,\
	sm__warps_active.avg.pct_of_peak_sustained_active,\
	launch__occupancy_limit_registers,\
	l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,\
	l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,\
	launch__registers_per_thread \
	    ./ks_bench 473 297 30 4 2 1 2>&1 | grep -vE '^\s*$$'

# The production configs (~/analysis_gebco/*.nml). nx/ny are POINT COUNTS.
sweep: ks_bench
	@echo "  --- production configs (nghost=3, nz=30 unless noted) ---"
	@./ks_bench 108 137 30 20 10 1 | grep -E 'COLUMNS|do concurrent|faithful|ratio  dc / cuda_f'
	@./ks_bench 473 297 30 20 10 1 | grep -E 'COLUMNS|do concurrent|faithful|ratio  dc / cuda_f'
	@./ks_bench 240 560 30 20 10 1 | grep -E 'COLUMNS|do concurrent|faithful|ratio  dc / cuda_f'
	@./ks_bench 945 594 30 20 10 1 | grep -E 'COLUMNS|do concurrent|faithful|ratio  dc / cuda_f'

clean:
	rm -f ks_bench *.o *.mod
