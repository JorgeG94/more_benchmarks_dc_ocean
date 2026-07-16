# ocean horizontal viscosity (Smagorinsky/Laplacian application):
# do concurrent vs a faithful CUDA C port, compared in one process via
# OpenACC host_data use_device. Toolchain from ../config.mk.
#
#   make && make run
#   ./hvisc_bench [nx_phys] [ny_phys] [nz] [nreps] [nwarm] [cuda_sync]
-include ../../config.mk
FC   ?= nvfortran
NVCC ?= nvcc
ARCH   ?= cc70
NVARCH ?= sm_70
GPU_ARCHFLAG ?= -arch=$(NVARCH)

GPUFLAGS := -gpu=$(ARCH),mem:separate -gpu=tripcount:host
FFLAGS   := -Mfree -Mbackslash -O3 -fast -acc=gpu -stdpar=gpu $(GPUFLAGS)
# -cuda is LINK-ONLY (it switches nvfortran into CUDA-Fortran mode, which
# rejects the host_data device attribute on the plain bind(C) interface).
LDFLAGS  := -acc=gpu -stdpar=gpu $(GPUFLAGS) -cuda
CUFLAGS  := -O3 $(GPU_ARCHFLAG) -I../../common --compiler-options -fPIC -Wno-deprecated-gpu-targets

OBJS := constants.o hvisc.o hvisc_kernel.o hvisc_bench.o

.PHONY: all run clean
all: hvisc_bench

constants.o: constants.F90
	$(FC) $(FFLAGS) -c $< -o $@
hvisc.o: hvisc.F90 constants.o
	$(FC) $(FFLAGS) -c $< -o $@
hvisc_bench.o: hvisc_bench.F90 hvisc.o constants.o
	$(FC) $(FFLAGS) -c $< -o $@
hvisc_kernel.o: hvisc_kernel.cu ../../common/gpu_rt.h
	$(NVCC) $(CUFLAGS) -c $< -o $@

hvisc_bench: $(OBJS)
	$(FC) $(LDFLAGS) $(OBJS) -o $@

ARGS ?= 473 297 30 200 10
run: hvisc_bench
	@./hvisc_bench $(ARGS)

clean:
	@rm -f hvisc_bench *.o *.mod
