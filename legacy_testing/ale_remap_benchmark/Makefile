-include ../../config.mk
# ALE remap: production `do concurrent` vs improved Fortran vs hand-written CUDA C.
#
#   source ../../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
#   make && make run
#
# `kernel_remap.F90` is a BYTE-IDENTICAL copy of production (`make verbatim`
# re-checks it). `ale_remap_dc.F90` is a transcription of
# `ocean_apply_ale_remap_step` specialised to the zstar config every production
# namelist uses -- see that file's header for exactly what was dropped and why.

FC   ?= nvfortran
NVCC ?= nvcc
ARCH   ?= cc70
NVARCH ?= sm_70

MODEL_ROOT := <model-root>
PROD   := $(MODEL_ROOT)/src/ALE/kernel_remap.F90

# ⚠ `-gpu=tripcount:host` IS MANDATORY. Flags copied verbatim from production:
#   <model>/build/CMakeFiles/core_model_objs.dir/flags.make
#   -Mfree -Mbackslash -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate \
#   -gpu=tripcount:host -O3 -fast -module modules -fPIC
# Without tripcount:host, NVHPC 26.5 reads loop trip-counts from the DEVICE copy
# and inserts a per-kernel data refresh in multi-loop `!$acc kernels` regions --
# ~2x wrong (NVIDIA TPR #38714; <model> guards it in cmake/compiler_flags.cmake:66-72).
# This benchmark measures an `!$acc kernels` region, so it is doubly exposed.
#
# NZ_STACK_MAX=128 matches production's ACTUAL build (-DMODEL_NZ_STACK_MAX=128 in
# flags.make), NOT the 256 default in constants.F90's #define.
NZSTACK  ?= 128
GPUFLAGS := -gpu=$(ARCH),mem:separate -gpu=tripcount:host
FFLAGS   := -Mfree -Mbackslash -O3 -fast -acc=gpu -stdpar=gpu $(GPUFLAGS) \
            -DMODEL_NZ_STACK_MAX=$(NZSTACK)
CUFLAGS  := -O3 $(GPU_ARCHFLAG) --compiler-options -fPIC -DMODEL_NZ_STACK_MAX=$(NZSTACK) \
            -Wno-deprecated-gpu-targets

# -cuda is LINK-ONLY: it also switches nvfortran into CUDA Fortran mode, which
# type-checks device attributes and rejects the `host_data use_device` bridge
# with NVFORTRAN-S-0528 device attribute mismatch.
LDFLAGS  := -acc=gpu -stdpar=gpu $(GPUFLAGS) -cuda

OBJS := constants.o remap_state.o kernel_remap.o \
        ale_remap.o ale_remap_acc.o ale_remap_fix.o \
        ale_remap_fused.o ale_remap_fused_acc.o \
        ale_kernel.o ale_bench.o

.PHONY: all run native native-verify sweep collapse regs verbatim clean
all: ale_bench

constants.o: constants.F90
	$(FC) $(FFLAGS) -c $< -o $@
remap_state.o: remap_state.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@
kernel_remap.o: kernel_remap.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@

# ---- The three DC variants are ONE source compiled three ways. That is the
# point: it makes the async test airtight (the ONLY difference is the
# directives) and the fixes test honest (the only difference is the fix).
ale_remap.o: ale_remap_dc.F90 remap_state.o kernel_remap.o
	$(FC) $(FFLAGS) -c $< -o $@
ale_remap_acc.o: ale_remap_dc.F90 remap_state.o kernel_remap.o
	$(FC) $(FFLAGS) -DMODNAME=ale_remap_acc -DASYNC -c $< -o $@
ale_remap_fix.o: ale_remap_dc.F90 remap_state.o kernel_remap.o
	$(FC) $(FFLAGS) -DMODNAME=ale_remap_fix -DPPM_DIRECT \
	      -DCOLLAPSE_FIX -DFLATSIG -c $< -o $@

ale_remap_fused.o: ale_remap_fused.F90 remap_state.o kernel_remap.o
	$(FC) $(FFLAGS) -c $< -o $@
ale_remap_fused_acc.o: ale_remap_fused.F90 remap_state.o kernel_remap.o
	$(FC) $(FFLAGS) -DMODNAME=ale_remap_fused_acc -DASYNC -c $< -o $@

ale_kernel.o: ale_kernel.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@
ale_bench.o: ale_bench.F90 ale_remap.o ale_remap_acc.o ale_remap_fix.o \
             ale_remap_fused.o ale_remap_fused_acc.o
	$(FC) $(FFLAGS) -c $< -o $@

ale_bench: $(OBJS)
	$(FC) $^ -o $@ $(LDFLAGS)

run: ale_bench
	@./ale_bench

# ---- NATIVE C++/CUDA driver: cudaMalloc + main(), no Fortran, no OpenACC.
# Answers "is the `host_data use_device` present-table lookup on the CUDA path
# costing anything?" -- if it is, every CUDA number here is understated.
# It links ale_kernel.o, the SAME object ale_bench links: the kernels are not
# duplicated, so the two drivers cannot drift apart. nvcc links libcudart
# itself; nothing Fortran, nothing OpenACC, no -cuda.
ale_native.o: ale_native.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@
ale_native: ale_native.o ale_kernel.o
	$(NVCC) $(GPU_ARCHFLAG) -Wno-deprecated-gpu-targets $^ -o $@

native: ale_native
	@./ale_native

# The verified run: ale_bench dumps the pristine state + variants 1/6/7, then
# the native driver diffs its own init and its own output against them.
# ale_ref.bin is ~571 MB and transient -- `make clean` removes it.
#
# BOTH modes are run, and the second is the load-bearing one:
#   pass 1 (computed init)  -- checks the native init reproduces the Fortran
#                              one. It agrees to 1-2 ulp, not exactly: the
#                              synthetic init calls sin/cos/exp and glibc's
#                              differ from nvfortran's. So pass 1 CANNOT show a
#                              bit-identical output -- that ulp noise propagates.
#   pass 2 (--use-ref-state) -- runs on the DUMPED state, so the input is
#                              guaranteed identical and the output check
#                              isolates the CUDA path. THIS is where
#                              "PASS (bit-identical)" must appear.
native-verify: ale_bench ale_native
	@./ale_bench 473 297 30 20 10 25 1
	@echo ""
	@echo "#### pass 1: native's own init (checks the init; expect ~1e-11 downstream)"
	@./ale_native 473 297 30 20 10 25 --ref ale_ref.bin
	@echo ""
	@echo "#### pass 2: --use-ref-state (isolates the CUDA path; expect exactly 0.0)"
	@./ale_native 473 297 30 20 10 25 --ref ale_ref.bin --use-ref-state

# The gap is a function of TOTAL CELL COUNT, so sweep the real configs.
# nx/ny are POINT COUNTS from ~/analysis_gebco/*.nml (extent = nx*dx).
sweep: ale_bench
	@N=$(or $(N),3) ./sweep.sh

collapse: ale_remap_dc.F90 ale_remap_fused.F90
	@for f in $^; do echo "=== $$f ==="; \
	  $(FC) $(FFLAGS) -Minfo=accel,stdpar -c $$f -o /dev/null 2>&1 | \
	    grep -E "auto-collapsed|collapse\(|Loop run sequentially"; done
	@echo "=== ale_remap_dc.F90 WITH -DCOLLAPSE_FIX ==="
	@mkdir -p .modtmp
	@$(FC) $(FFLAGS) -DCOLLAPSE_FIX -DMODNAME=x -Minfo=accel,stdpar -module .modtmp \
	   -c ale_remap_dc.F90 -o /dev/null 2>&1 | grep -E "auto-collapsed|collapse\(|Loop run sequentially"
	@rm -rf .modtmp

# ⚠ Match the MANGLED kernel name (`*_gpu`). The first `Used N registers` in a
# ptxas dump is cub::EmptyKernel, not your kernel -- that misread cost a day.
regs: ale_bench
	@cuobjdump -res-usage ale_bench 2>/dev/null | grep -A1 -E "Function (ale_remap|k_)" \
	  | grep -E "Function|REG:" | paste - - | sed 's/Function //' \
	  | awk '{printf "  %-62s %s %s\n", $$1, $$2, $$3}'

# Re-check the verbatim kernel against production. A failure here means the MRE
# has drifted from the shipped kernel and every number is void.
verbatim:
	@if diff -q kernel_remap.F90 $(PROD) >/dev/null; then \
	   echo "  VERBATIM OK: kernel_remap.F90 is byte-identical to production"; \
	 else \
	   echo "  *** DRIFTED FROM PRODUCTION -- every number is void ***"; \
	   diff kernel_remap.F90 $(PROD) | head -20; exit 1; fi

clean:
	rm -f ale_bench ale_native ale_ref.bin quick_probe* *.o *.mod *.nsys-rep *.sqlite
	rm -rf .modtmp
