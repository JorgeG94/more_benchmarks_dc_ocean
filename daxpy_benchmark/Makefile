# MRE: OpenACC-allocated device memory + pure CUDA C compute
#
#   source ../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
#   make && make run
#
# Override the arch for a non-V100 box:
#   make ARCH=cc80 NVARCH=sm_80      # A100
#   make ARCH=cc90 NVARCH=sm_90      # H100

FC    := nvfortran
NVCC  := nvcc

# V100. ARCH and NVARCH must agree.
# NOTE: no trailing comments on these two lines — make would fold the spaces
# before the '#' into the value, producing `-gpu=cc70   ,mem:separate` and a
# link error about a file named ",mem:separate".
ARCH   ?= cc70
NVARCH ?= sm_70

# mem:separate  = NO managed/unified memory. Explicit data movement only.
#                 This is the interesting case: it is what forces the
#                 host_data/use_device bridge to be real rather than papered
#                 over by the runtime migrating pages behind your back.
# -stdpar=gpu   = lower `do concurrent` to the GPU (variant 1).
# -acc          = honour the OpenACC directives (both variants).
GPUFLAGS := -gpu=$(ARCH),mem:separate
CUFLAGS  := -O2 -arch=$(NVARCH) --compiler-options -fPIC

# COMPILE flags — note: NO -cuda here, deliberately. See LDFLAGS.
FFLAGS   := -O2 -acc -stdpar=gpu $(GPUFLAGS) -Minfo=accel,stdpar

# LINK flags — -cuda pulls in the CUDA runtime so the nvcc-compiled object
# resolves (without it: undefined reference to cudaLaunchKernel).
#
# WHY -cuda IS LINK-ONLY (this bit is load-bearing):
# -cuda also switches nvfortran into CUDA Fortran mode, which type-checks
# DEVICE ATTRIBUTES. Inside `!$acc host_data use_device(a,b,c)` the arrays
# carry the device attribute, but our bind(C) interface declares plain host
# arrays — so compiling daxpy_cuda.F90 WITH -cuda fails:
#     NVFORTRAN-S-0528-Argument number 1 to daxpy3d_cuda_launch:
#         device attribute mismatch
# Compile without it and the OpenACC host_data bridge is unaffected (host_data
# is OpenACC, not CUDA Fortran). Hence the split below.
# The alternative — keep -cuda everywhere and pass `type(c_ptr), value` +
# c_loc() instead of arrays-by-reference — also works; this way keeps the
# Fortran call site idiomatic.
LDFLAGS  := -acc -stdpar=gpu $(GPUFLAGS) -cuda

BIN := daxpy_dc daxpy_cuda daxpy_pure

.PHONY: all run clean
all: $(BIN)

%.o: %.F90
	$(FC) $(FFLAGS) -c $< -o $@

daxpy_dc.o daxpy_cuda.o: daxpy_common.o

daxpy_kernel.o: daxpy_kernel.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

# Variant 1 — do concurrent. Needs no CUDA object at all.
daxpy_dc: daxpy_dc.o daxpy_common.o
	$(FC) $^ -o $@ $(LDFLAGS)

# Variant 2 — pure CUDA C kernel linked against the Fortran driver.
daxpy_cuda: daxpy_cuda.o daxpy_common.o daxpy_kernel.o
	$(FC) $^ -o $@ $(LDFLAGS)

# Variant 3 — no Fortran at all. Links the SAME kernel object as variant 2, so
# the only difference vs v2 is who owns the memory. nvcc drives the link here;
# no nvfortran runtime is involved.
daxpy_pure.o: daxpy_pure.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

daxpy_pure: daxpy_pure.o daxpy_kernel.o
	$(NVCC) -arch=$(NVARCH) $^ -o $@

run: $(BIN)
	@echo "=============== variant 1: do concurrent ==============="
	@./daxpy_dc
	@echo ""
	@echo "=============== variant 2: OpenACC data + CUDA C ======="
	@./daxpy_cuda
	@echo ""
	@echo "=============== variant 3: pure CUDA C ================="
	@./daxpy_pure

clean:
	rm -f $(BIN) *.o *.mod
