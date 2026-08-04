# Compile lines

Extracted from the `flags` column of every measurement CSV, which the
harness records at build time. `-DMODEL_NZ_STACK_MAX` is a swept axis
and is factored out; the values swept are listed per lane.


## EPYC 7A53

**dc_multicore** — `amdflang` 23.0.0

```
-O3 -I../common -DKS_NO_NEWTON -fopenmp -fdo-concurrent-to-openmp=host
```
NZ_STACK_MAX swept: 128


```
-O3 -I../common -fopenmp -fdo-coP0+r4B31\P0+r4B33\P0+r4B34\P0+r4B35\P0+r6B42\P0+r5053\P0+r5045\ncurrent-to-openmp=host
```
NZ_STACK_MAX swept: 128


```
-O3 -I../common -fopenmp -fdo-concurrent-to-openmp=host
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**dc_serial** — `amdflang` 23.0.0

```
-O3 -I../common
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**serial_do** — `amdflang` 23.0.0

```
-O3 -I../common -stdpar=gpu -acc=gpu -gpu=cc90,mem:separate -gpu=tripcount:host -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128


## GH200

**cuda_faithful** — `nvfortran` 26.3

```
-Mfree -Mbackslash -O3 -fast -I../common -stdpar=gpu -acc=gpu -gpu=cc90,mem:separate -gpu=tripcount:host -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**cuda_opt** — `nvfortran` 26.3

```
-Mfree -Mbackslash -O3 -fast -I../common -stdpar=gpu -acc=gpu -gpu=cc90,mem:separate -gpu=tripcount:host -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**dc_gpu** — `nvfortran` 26.3

```
-Mfree -Mbackslash -O3 -fast -I../common -stdpar=gpu -acc=gpu -gpu=cc90,mem:separate -gpu=tripcount:host -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128


## Grace

**dc_multicore** — `nvfortran` 26.3

```
-Mfree -Mbackslash -O3 -fast -I../common -stdpar=multiP1+r4632=1B5B32347E\P0+r2531\P0+r2638\P1+r6B62=7F\P0+r6B49\P1+r6B44=1B5B337E\P1+r6B68=1B4F48\P1+r4037=1B4F46\P1+r6B50=1B5B357E\P1+r6B4E=1B5B367E\core
```
NZ_STACK_MAX swept: 128


```
-Mfree -Mbackslash -O3 -fast -I../common -stdpar=multicore
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**dc_serial** — `nvfortran` 26.3

```
-Mfree -Mbackslash -O3 -fast -I../common
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**serial_do** — `nvfortran` 26.3

```
-Mfree -Mbackslash -O3 -fast -I../common -stdpar=gpu -acc=gpu -gpu=cc90,mem:separate -gpu=tripcount:host -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128


## Intel GPU

**dc_gpu** — `ifx` 2025.3.2

```
-O3 -I../common -qopenmp -fopenmp-targets=spir64 -fopenmp-target-do-concurrent -DDC_DATA_OMP
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128


## Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz

**dc_multicore** — `flang` 22.1.5

```
-O3 -I../common -fopenmp -fdo-concurrent-to-openmp=host
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**dc_multicore** — `ifx` 2026.0.0

```
-O3 -I../common -DKS_NO_NEWTON -qopenmp
```
NZ_STACK_MAX swept: 128


```
-O3 -I../common -qopenmp
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**dc_multicore** — `nvfortran`

```
-Mfree -Mbackslash -O3 -fast -I../common -DKS_BOUNDED_COPY -stdpar=multicore
```
NZ_STACK_MAX swept: 31,76


```
-Mfree -Mbackslash -O3 -fast -I../common -stdpar=multicore
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**dc_serial** — `flang` 22.1.5

```
-O3 -I../common
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**dc_serial** — `gfortran` 16.1.0

```
-O3 -I../common
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**dc_serial** — `ifx` 2026.0.0

```
-O3 -I../common
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**dc_serial** — `nvfortran`

```
-Mfree -Mbackslash -O3 -fast -I../common
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128


```
-Mfree -Mbackslash -O3 -fast -I../common -DKS_BOUNDED_COPY
```
NZ_STACK_MAX swept: 31,76

**serial_do** — `flang` 22.1.5

```
-O3 -I../common -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate -gpu=tripcount:host -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**serial_do** — `gfortran` 16.1.0

```
-O3 -I../common -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate -gpu=tripcount:host -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**serial_do** — `gfortran` 13.3.0

```
-O3 -I../common -DKS_BOUNDED_COPY -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate -gpu=tripcount:host -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 31,76


```
-O3 -I../common -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate -gpu=tripcount:host -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**serial_do** — `ifx` 2026.0.0

```
-O3 -I../common -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate -gpu=tripcount:host -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**serial_do** — `nvfortran`

```
-Mfree -Mbackslash -O3 -fast -I../common -DKS_BOUNDED_COPY -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate -gpu=tripcount:host -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 31,76


```
-Mfree -Mbackslash -O3 -fast -I../common -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate -gpu=tripcount:host -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128


## MI250X

**dc_gpu** — `amdflang` 23.0.0

```
-O3 -I../common -fopenmp --offload-arch=gfx90a -fdo-concurrent-to-openmp=device -DDC_DATA_OMP
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128


## Mac

**dc_serial** — `gfortran` 15.2.0

```
-O3 -I../common
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**serial_do** — `gfortran` 15.2.0

```
-O3 -I../common -stdpar=gpu -acc=gpu -gpu=cc90,mem:separate -gpu=tripcount:host -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128


## Tesla V100

**dc_gpu** — `nvfortran` 26.5

```
-Mfree -Mbackslash -O3 -fast -I../common -stdpar=gpu -mp=gpu -gpu=cc70,mem:separate -DDC_DATA_OMP
```
NZ_STACK_MAX swept: 31


## Tesla V100-DGXS-32GB

**cuda_faithful** — `nvcc` 12.9

```
-O3 -arch=sm_70 -I../common --compiler-options -fPIC -DKS_FULL_COPY
```
NZ_STACK_MAX swept: 11,31,128

**cuda_faithful** — `nvfortran` 12.9

```
-O3 -arch=sm_70 -I../common --compiler-options -fPIC -DKS_MINBLOCKS=0 -DIDX32=1 -DKS_FULL_COPY
```
NZ_STACK_MAX swept: 11,16,26,31,51,76,101,128

**cuda_opt** — `nvcc` 12.9

```
-O3 -arch=sm_70 -I../common --compiler-options -fPIC -DKS_USE_OPT -DKS_MINBLOCKS=0 -DIDX32=1
```
NZ_STACK_MAX swept: 11,31,128

**cuda_opt** — `nvfortran` 12.9

```
-O3 -arch=sm_70 -I../common --compiler-options -fPIC -DKS_MINBLOCKS=0 -DIDX32=1 -DKS_FULL_COPY
```
NZ_STACK_MAX swept: 11,16,26,31,51,76,101,128

**dc_gpu_acc** — `nvfortran`

```
-Mfree -Mbackslash -O3 -fast -I../common -DKS_WITH_CUDA -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 128


```
-Mfree -Mbackslash -O3 -fast -I../common -DKS_WITH_CUDA -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate -gpu=tripcount:host -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 11,16,26,31,51,76,101,128


```
-Mfree -Mbackslash -O3 -fast -I../common -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 128


```
-Mfree -Mbackslash -O3 -fast -I../common -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate -gpu=tripcount:host -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 11,16,26,31,51,76,101,128

**dc_gpu_omp** — `nvfortran`

```
-Mfree -Mbackslash -O3 -fast -I../common -stdpar=gpu -mp=gpu -gpu=cc70,mem:separate -DDC_DATA_OMP
```
NZ_STACK_MAX swept: 11,31,128


## V100

**dc_gpu** — `nvfortran` 26.5

```
-Mfree -Mbackslash -O3 -fast -I../common -stdpar=gpu -acc=gpu -gpu=cc70,mem:separate -gpu=tripcount:host -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 31


## Xeon Max

**dc_multicore** — `ifx` 2025.3.2

```
-O3 -I../common -DKS_NO_NEWTON -qopenmp
```
NZ_STACK_MAX swept: 128


```
-O3 -I../common -qopenmp
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**dc_serial** — `ifx` 2025.3.2

```
-O3 -I../common
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128

**serial_do** — `ifx` 2025.3.2

```
-O3 -I../common -stdpar=gpu -acc=gpu -gpu=cc90,mem:separate -gpu=tripcount:host -DDC_DATA_ACC
```
NZ_STACK_MAX swept: 11,26,31,51,76,101,128


> 9 lane(s) show more than one compile line — see above.
