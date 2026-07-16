# mre_acc_cuda top-level — build and run every benchmark.
# Per-machine / per-backend config lives in ./config.mk (the one file to edit).
#
#   make all-dc                 # build every benchmark (BACKEND=cuda default)
#   make all-dc BACKEND=hip     # ...against the AMD/HIP backend (see config.mk)
#   make all-dc ARCH=cc90 NVARCH=sm_90     # ...for the H200
#   make run-all                # build, then run bench+native for all, on an idle GPU
#   make run-dc                 # run ONLY the do-concurrent side (timing + agreement check)
#   make picture                # collate the run logs into the final-picture table
#   make clean                  # remove ALL build + run artifacts across the whole tree
#
# New-style build-mode matrix (over the ported KERNELS, see below):
#   make all-cuda               # do concurrent (OpenACC) + native CUDA, per kernel
#   make all-omp                # do concurrent via OpenMP target (AMD/Intel-portable)
#   make all-cpu                # do concurrent on the CPU (no CUDA/OpenACC/OpenMP-target)
#   make run-cpu                # ...then RUN all 8 small on the CPU (make FC=gfortran run-cpu)
#   make all-hip                # native HIP (structural -- needs hipcc/ROCm)
#   make verify-all             # every kernel: OpenACC dumps a ref, CPU cross-checks it
#
# `all-dc` = "build all the OLD-style benchmarks" (each also builds its
# hand-written GPU-C comparison driver, so the Fortran-vs-C numbers are there).

ROOT := $(CURDIR)
include $(ROOT)/config.mk

# Original benchmarks (profile-share order) -- retired to legacy_testing/,
# kept building until deleted. The active kernels are the build-mode KERNELS.
BENCH := $(addprefix legacy_testing/, \
         redi_benchmark kappa_shear_benchmark continuity_layered_benchmark \
         ale_remap_benchmark btstep_benchmark epbl_benchmark \
         meke_benchmark hll_fluxes_benchmark)

# Vars forwarded to every sub-make so config.mk / the CLI stay authoritative.
FWD := BACKEND=$(BACKEND) ARCH=$(ARCH) NVARCH=$(NVARCH) \
       NZSTACK=$(NZSTACK) STACK=$(STACK) GPU_ARCHFLAG=$(GPU_ARCHFLAG) NVCC=$(NVCC)

.PHONY: all-dc run-all run-dc picture clean clean-all $(BENCH)

all-dc: $(BENCH)

$(BENCH):
	@echo "==> $@  (BACKEND=$(BACKEND) ARCH=$(ARCH) NVARCH=$(NVARCH))"
	@$(MAKE) --no-print-directory -C $@ $(FWD)          # the do-concurrent + CUDA-via-Fortran bench
	@$(MAKE) --no-print-directory -C $@ $(FWD) native   # the native cudaMalloc driver

run-all: all-dc
	@ROOT=$(ROOT) ./final_picture/build_and_run_all.sh

# Just the do-concurrent implementation: runs each bench, shows the DC timing and
# its bit-agreement vs the CUDA reference. No native driver, no collation.
run-dc:
	@ROOT=$(ROOT) DC_ONLY=1 ./final_picture/build_and_run_all.sh

picture:
	@python3 final_picture/collate.py

# ============================================================================
# New-style build-mode kernels: the do-concurrent / cuda / hip / cpu matrix.
# Strategy is a build MODE (make dc DATA=... / make cpp BACKEND=...), not a
# directory. See continuity_layered/README.md and common/{directives,gpu_rt}.h.
# ============================================================================
KERNELS := continuity_layered redi kappa_shear ale_remap btstep epbl meke hll_fluxes hvisc

.PHONY: all-cuda all-omp all-cpu all-hip verify-all run-all-cuda run-cpu

# NVIDIA: do concurrent (OpenACC) + the native CUDA driver, per kernel.
all-cuda:
	@for k in $(KERNELS); do echo "==> $$k  dc(acc) + cpp(cuda)"; \
	  $(MAKE) --no-print-directory -C $$k dc DATA=acc && \
	  $(MAKE) --no-print-directory -C $$k cpp BACKEND=cuda || exit 1; done

# Portable do concurrent via OpenMP target (NVIDIA now; AMD/Intel with amdflang/ifx).
all-omp:
	@for k in $(KERNELS); do echo "==> $$k  dc(omp)"; \
	  $(MAKE) --no-print-directory -C $$k dc DATA=omp || exit 1; done

# Do concurrent on the CPU -- no CUDA, no OpenACC/OpenMP-target.
all-cpu:
	@for k in $(KERNELS); do echo "==> $$k  dc(none)"; \
	  $(MAKE) --no-print-directory -C $$k dc DATA=none || exit 1; done

# Build (if needed) + RUN every kernel's CPU do-concurrent binary, small & fast.
# The GPU-sized defaults (473x297, up to 200 reps) are slow on a CPU, so run a
# small grid/few reps. Only the leading `nx ny nz nreps nwarm` are overridden --
# any kernel-specific trailing args keep their own defaults. Tune via CPU_ARGS.
#   make FC=gfortran run-cpu                     # all 8, small
#   make FC=gfortran run-cpu CPU_ARGS="256 256 30 20 5"
CPU_ARGS ?= 96 96 10 8 2
run-cpu: all-cpu
	@for k in $(KERNELS); do echo "== $$k =="; \
	  $(MAKE) --no-print-directory -C $$k run-dc DATA=none ARGS="$(CPU_ARGS)" || exit 1; done

# AMD: native HIP driver (structural -- needs hipcc/ROCm; see config.mk).
all-hip:
	@for k in $(KERNELS); do echo "==> $$k  cpp(hip)"; \
	  $(MAKE) --no-print-directory -C $$k cpp BACKEND=hip || exit 1; done

# The portability proof, every kernel: OpenACC-GPU dumps a ref, the CPU build
# recomputes and cross-checks it bit-identical (to FMA level).
verify-all:
	@for k in $(KERNELS); do echo "==> verify $$k"; \
	  $(MAKE) --no-print-directory -C $$k verify || exit 1; done

run-all-cuda: all-cuda
	@for k in $(KERNELS); do echo "== $$k =="; \
	  $(MAKE) --no-print-directory -C $$k run-dc; \
	  $(MAKE) --no-print-directory -C $$k run-cpp; done

# Every subdir with its own Makefile (top-level + retired legacy_testing/).
MAKEDIRS := $(patsubst %/,%,$(dir $(wildcard */Makefile) $(wildcard legacy_testing/*/Makefile)))

# Full clean: every benchmark's own clean, plus the nvbug repros, the scratch
# cubin dir, and the generated final_picture logs/results. Sources are untouched.
clean:
	@for d in $(MAKEDIRS); do echo "  clean $$d"; $(MAKE) --no-print-directory -C $$d clean || true; done
	@rm -f  nvbug_dc_collapse/*.o nvbug_dc_collapse/*.mod nvbug_dc_collapse/repro
	@rm -f  nvbug_inline_cse/*.o  nvbug_inline_cse/*.mod  nvbug_inline_cse/repro
	@rm -rf tmp_local_artifacts
	@rm -f  final_picture/logs/* final_picture/results/*
	@echo "cleaned: build artifacts, nvbug repros, tmp_local_artifacts, final_picture outputs."

# Backwards-compatible alias.
clean-all: clean
