-include ../config.mk
# the ocean model's EPBL column kernel: `do concurrent` vs hand-written CUDA C.
#
#   source ../../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
#   make && make run
#
# EPBL is the scheme inside `ocean_vmix_compute` -- 39.1% of production runtime,
# the largest region in the profile. ocean_epbl.F90 splices the production
# kernel in VERBATIM; see its header for the md5 and the exact line ranges.
# Re-extract if the production file moves.

FC   ?= nvfortran
NVCC ?= nvcc

# V100. ARCH and NVARCH must agree.
# NOTE: no trailing comments on these two -- make folds the spaces before '#'
# into the value, producing `-gpu=cc70   ,mem:separate` and a link error.
ARCH   ?= cc70
NVARCH ?= sm_70

# __launch_bounds__ for the `tuned` CUDA variant. Overridable so the occupancy
# hypothesis can be SWEPT rather than asserted:
#   make LB_THREADS=128 LB_BLOCKS=4      -> cap 128 regs, 25% occupancy
LB_THREADS ?= 128
LB_BLOCKS  ?= 4

# COMPILE flags -- copied VERBATIM from production
# (<model>/build/CMakeFiles/core_model_objs.dir/flags.make), minus -module/-fPIC.
#
# ⚠ -gpu=tripcount:host IS LOAD-BEARING. NVHPC 26.5 otherwise reads loop
# trip-counts from the DEVICE copy and inserts a per-kernel data refresh
# (NVIDIA TPR #38714); <model> guards this in cmake/compiler_flags.cmake:66-72.
# Dropping it cost the btstep benchmark ~2x and invalidated every finding
# downstream. Keep these in sync with production's flags.make.
FFLAGS := -Mfree -Mbackslash -stdpar=gpu -acc=gpu -gpu=$(ARCH),mem:separate \
          -gpu=tripcount:host -O3 -fast

CUFLAGS := -O3 $(GPU_ARCHFLAG) --compiler-options -fPIC \
           -DEPBL_LB_THREADS=$(LB_THREADS) -DEPBL_LB_BLOCKS_PER_SM=$(LB_BLOCKS)

# LINK flags. -cuda pulls in the CUDA runtime so the nvcc object resolves.
# WHY -cuda IS LINK-ONLY: it also switches nvfortran into CUDA Fortran mode,
# which type-checks DEVICE ATTRIBUTES. Inside `!$acc host_data use_device(...)`
# the arrays carry the device attribute while the bind(C) interface declares
# plain host arrays, so compiling with -cuda fails:
#     NVFORTRAN-S-0528-Argument number 1 to ...: device attribute mismatch
LDFLAGS := -acc=gpu -stdpar=gpu -gpu=$(ARCH),mem:separate -gpu=tripcount:host -cuda

OBJS := constants.o grid.o epbl_stubs.o ocean_epbl.o \
        ocean_epbl_plain.o epbl_kernel.o epbl_bench.o

.PHONY: all run collapse regs sweep nofma maxreg sizes clean \
        native native-run native-verify native-nofma native-tax
all: epbl_bench

constants.o: constants.F90
	$(FC) $(FFLAGS) -c $< -o $@

grid.o: grid.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@

epbl_stubs.o: epbl_stubs.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@

ocean_epbl.o: ocean_epbl.F90 constants.o grid.o epbl_stubs.o
	$(FC) $(FFLAGS) -c $< -o $@

ocean_epbl_plain.o: ocean_epbl_plain.F90 ocean_epbl.o
	$(FC) $(FFLAGS) -c $< -o $@

epbl_bench.o: epbl_bench.F90 ocean_epbl.o ocean_epbl_plain.o
	$(FC) $(FFLAGS) -c $< -o $@

epbl_kernel.o: epbl_kernel.cu epbl_params.h
	$(NVCC) $(CUFLAGS) -c $< -o $@

epbl_bench: $(OBJS)
	$(FC) $^ -o $@ $(LDFLAGS)

run: epbl_bench
	@./epbl_bench

# ---------------------------------------------------------------------------
# NATIVE C++/CUDA driver: cudaMalloc + main(), no Fortran, no OpenACC.
#
# WHY: every CUDA number in this benchmark is launched from Fortran through
# `!$acc host_data use_device(...)` over 24 arrays -- a present-table lookup per
# array per launch. That puts the OpenACC runtime on the CUDA path, so if it
# costs anything, `dc / cuda_tuned = 1.329x` is UNDERSTATED. This target removes
# OpenACC and settles it. It also builds with nvcc ALONE, which is what makes it
# usable as a standalone reproducer in nvbug_*/.
#
# It links epbl_kernel.o -- the SAME object epbl_bench links. The kernel is not
# duplicated; see epbl_native.cu's header.
native: epbl_native

epbl_native: epbl_native.cu epbl_kernel.o epbl_params.h
	$(NVCC) $(CUFLAGS) epbl_native.cu epbl_kernel.o -o $@

native-run: epbl_native
	@./epbl_native

# The honest comparison: ONE binary each, SAME args, and the native driver
# checks itself against the Fortran driver's dumped inputs AND outputs. Both
# must exist, so both are built. epbl_bench's arg 10 = 1 writes epbl_ref.f64.
#
# ⚠ Run these back-to-back on a QUIET GPU. epbl_bench.F90:369 documents the same
# binary measuring 4.9 ms and 23.5 ms on a contended device -- a 4.8x swing,
# which is larger than every effect this benchmark studies.
native-verify: epbl_bench epbl_native
	@echo "=== 1/2  Fortran driver (host_data) -- also dumps epbl_ref.f64 ==="
	@./epbl_bench 473 297 30 20 8 1 0 20 5 1
	@echo ""
	@echo "=== 2/2  native driver (cudaMalloc) -- verifies against that dump ==="
	@./epbl_native 473 297 30 20 8 1 0 20 5

# MEASURE THE OPENACC LAUNCH TAX -- the question epbl_native.cu exists for.
#
# Comparing the two drivers at 473x297 CANNOT answer it: the kernel is ~6 ms, a
# per-launch host cost is microseconds, and this GPU is shared badly enough that
# the noise is ~2x. Signal is ~0.05%. Hopeless.
#
# So don't measure it there. Shrink the kernel to nothing (8x8, nz=2,
# mld_max_its=1) and force the host path into the critical path (cu_sync=1 =
# cudaDeviceSynchronize after every launch, so nothing hides behind the kernel).
# Everything is then identical between the two drivers EXCEPT the launch path:
#   fortran = !$acc host_data use_device over 24 arrays + bind(C) + launch
#   native  = launch
# The DIFFERENCE is the tax, in us/launch. It is a HOST cost, so it barely moves
# under GPU contention -- the native column should sit rock-steady.
#
# Measured 2026-07-16, V100, NVHPC 26.5 / nvcc 12.9: fortran 18.5, native 15.7,
# delta +2.8 us/launch (+-0.15 over 4 rounds) = ~0.12 us per array lookup.
# Against EPBL's ~6 ms/launch at the profiled config that is 0.05%.
native-tax: epbl_bench epbl_native
	@echo "  8x8x2, max_its=1, cu_sync=1, 3000 reps -- host launch path only"
	@echo "  round   fortran(host_data)   native(cudaMalloc)      delta"
	@for r in 1 2 3 4; do \
	   f=$$(./epbl_bench 8 8 2 3000 50 1 0 1 3 2>/dev/null \
	        | grep 'faithful (1 thr' | awk '{print $$9}'); \
	   n=$$(./epbl_native 8 8 2 3000 50 1 0 1 3 2>/dev/null \
	        | grep 'faithful (1 thr' | awk '{print $$9}'); \
	   printf "  %5s   %14.2f us %17.2f us %10.2f us\n" $$r \
	     $$(echo "$$f*1000" | bc -l) $$(echo "$$n*1000" | bc -l) \
	     $$(echo "($$f-$$n)*1000" | bc -l); \
	 done

# PROOF the native driver is faithful to `do concurrent` and not merely to the
# CUDA kernel. At production flags nvcc and nvfortran contract FMAs differently
# and EPBL amplifies a sub-ulp difference into a whole quantised layer of MLD
# (README "Bit-identity"). Contraction off on BOTH sides collapses it to ~1e-10.
# This is the same argument `nofma` makes for epbl_bench, extended to the native
# driver -- and it is why the [3] block in epbl_native.cu does NOT get a
# loosened tolerance instead.
native-nofma:
	@echo "=== rebuilding both drivers with FMA contraction DISABLED ==="
	@mkdir -p .nofma && cp *.F90 *.cu *.h Makefile .nofma/
	@ln -sf ../gabight_bathy_0p1_473x297.f64 .nofma/ 2>/dev/null || true
	@$(MAKE) -sC .nofma FFLAGS="$(FFLAGS) -Mnofma" CUFLAGS="$(CUFLAGS) -fmad=false" \
	   epbl_bench epbl_native >/dev/null 2>&1
	@cd .nofma && ./epbl_bench 473 297 30 3 2 1 0 20 1 1 >/dev/null 2>&1 \
	   && ./epbl_native 473 297 30 3 2 1 0 20 1 \
	   | grep -E 'BIT-IDENTICAL|DIFFERS|max\|d\||VERDICT|vs dc|\[1\]|\[2\]|\[3\]'

# The scheme is ONE do concurrent. Expect exactly one "Loop parallelized ...
# collapse(2)" and three "Loop run sequentially" (prep k, dry-zero k, sweep ki).
collapse:
	@$(FC) $(FFLAGS) -Minfo=stdpar,accel -c ocean_epbl.F90 -o /dev/null 2>&1 \
	   | grep -E 'auto-collapsed|Loop parallelized|run sequentially|Loop not' | head -20
	@rm -f *.mod

# Registers + spills. ⚠ Match the mangled name -- the nvfortran kernel is
# `ocean_epbl_epbl_column_kernel_791_gpu`, and the first `Used N registers`
# in a raw ptxas dump is cub::EmptyKernel, NOT your kernel (RESUME_GPU_MRE §1).
# Expected: DC 255 regs / 128 B stack; CUDA faithful 184 / 0; CUDA tuned 128 / 160.
regs: epbl_bench
	@echo "--- registers + spill stack per kernel (matched by mangled name) ---"
	@cuobjdump -res-usage epbl_bench 2>/dev/null \
	   | grep -A1 -E 'epbl_column_kernel_791_gpu|_Z13epbl_faithful|_Z10epbl_tuned' \
	   | grep -E 'Function|REG:'
	@echo "--- spill instructions (LDL/STL) + LDG per kernel ---"
	@cuobjdump -sass epbl_bench | awk '/Function : /{fn=$$3} \
	   /LDG/{ldg[fn]++} /LDL|STL/{loc[fn]++} /\/\*[0-9a-f]+\*\//{tot[fn]++} \
	   END{for (f in tot) if (f ~ /epbl/) \
	         printf "  %-52s LDG=%4d spills=%4d inst=%6d\n", f, ldg[f]+0, loc[f]+0, tot[f]}' | sort

# TEST the occupancy hypothesis rather than assert it: sweep __launch_bounds__
# over blocks/SM. 2 blocks = the default 12.5% occupancy; 4 = 25%.
sweep:
	@echo " blocks/SM  target_occ   cuda_tuned_ms   (V100: 4 warps/block, 64 warps/SM)"
	@for b in 2 3 4 5 6; do \
	   $(MAKE) -s LB_BLOCKS=$$b relink >/dev/null 2>&1; \
	   printf " %9s  %8.1f%%   " $$b $$(echo $$b | awk '{print $$1*4/64*100}'); \
	   ./epbl_bench 473 297 30 20 8 1 0 20 5 | grep 'launch_bounds' | awk '{print $$6}'; \
	done
	@$(MAKE) -s LB_BLOCKS=$(LB_BLOCKS) relink >/dev/null 2>&1

relink:
	@$(NVCC) $(CUFLAGS) -c epbl_kernel.cu -o epbl_kernel.o
	@$(FC) $(OBJS) -o epbl_bench $(LDFLAGS)

# PROOF that the DC-vs-CUDA disagreement is FMA contraction through a marginal
# branch and NOT a port bug. Rebuilds BOTH sides with contraction off; the MLD
# disagreement must collapse from ~60 m to ~1e-9 m. See README "Bit-identity".
nofma:
	@echo "=== rebuilding both toolchains with FMA contraction DISABLED ==="
	@mkdir -p .nofma && cp *.F90 *.cu *.h Makefile .nofma/
	@ln -sf ../gabight_bathy_0p1_473x297.f64 .nofma/ 2>/dev/null || true
	@$(MAKE) -sC .nofma FFLAGS="$(FFLAGS) -Mnofma" CUFLAGS="$(CUFLAGS) -fmad=false" \
	   epbl_bench >/dev/null 2>&1
	@cd .nofma && ./epbl_bench 473 297 30 3 2 1 0 20 1 \
	   | grep -E 'max\|d\||agreement|VERDICT|columns whose'

# Can portable Fortran reach the occupancy the CUDA `tuned` variant gets?
# nvfortran DOES expose a register cap (-gpu=maxregcount:N) -- so the honest
# question is not "can Fortran express it" but "does it survive it". It does
# not: at the same 128-register cap nvfortran spills 568 B vs nvcc's 160 B.
# The cuda_tuned column is a CONTROL -- it must stay flat, proving the machine
# was stable across the rebuilds.
maxreg:
	@echo "  flag              regs stack     dc_ms   [cuda_tuned = control, must stay flat]"
	@for m in 0 128 160 192 255; do \
	   if [ $$m -eq 0 ]; then f=""; t="(default)"; else f="-gpu=maxregcount:$$m"; t="maxregcount:$$m"; fi; \
	   $(MAKE) -s clean >/dev/null 2>&1; \
	   $(MAKE) -s FFLAGS="$(FFLAGS) $$f" LDFLAGS="$(LDFLAGS) $$f" epbl_bench >/dev/null 2>&1; \
	   rs=$$(cuobjdump -res-usage epbl_bench 2>/dev/null | grep -A1 epbl_column_kernel_791_gpu \
	          | grep REG | sed 's/.*REG:\([0-9]*\) STACK:\([0-9]*\).*/\1 \2/'); \
	   o=$$(./epbl_bench 473 297 30 20 8 1 0 20 5); \
	   printf "  %-17s %-9s %8s   %8s\n" "$$t" "$$rs" \
	     "$$(echo "$$o" | grep 'do concurrent (production)' | awk '{print $$6}')" \
	     "$$(echo "$$o" | grep launch_bounds | awk '{print $$6}')"; \
	done

# The production configs from ~/analysis_gebco/*.nml, plus an nz sweep.
sizes: epbl_bench
	@echo " config                columns    dc_ms   cf_ms   ct_ms   dc/cf  dc/ct"
	@for c in "108 137 30" "473 297 30" "240 560 30" "945 594 30"; do \
	   o=$$(./epbl_bench $$c 20 8 1 0 20 5); \
	   printf " %-18s %9s %8s %7s %7s  %6s %6s\n" "$$c" \
	     "$$(echo "$$o"|grep COLUMNS|awk '{print $$6}')" \
	     "$$(echo "$$o"|grep 'do concurrent (production)'|awk '{print $$6}')" \
	     "$$(echo "$$o"|grep 'faithful (1 thr'|awk '{print $$9}')" \
	     "$$(echo "$$o"|grep launch_bounds|awk '{print $$6}')" \
	     "$$(echo "$$o"|grep 'dc / cuda_faithful'|awk '{print $$6}')" \
	     "$$(echo "$$o"|grep 'dc / cuda_tuned'|awk '{print $$6}')"; \
	done

clean:
	rm -f epbl_bench epbl_native epbl_ref.f64 *.o *.mod
	rm -rf .nofma .mrc
