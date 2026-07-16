# MRE — OpenACC allocates the GPU memory, pure CUDA C computes on it

3-D daxpy, `c(i,j,k) = alpha*a(i,j,k) + b(i,j,k)`, 512³ doubles (1.07 GB/array),
three variants:

| variant | allocation | compute |
|---|---|---|
| `daxpy_dc` | OpenACC `!$acc enter data` | Fortran `do concurrent` (`-stdpar=gpu`) |
| `daxpy_cuda` | OpenACC `!$acc enter data` — *identical* | **pure CUDA C** via `host_data use_device` |
| `daxpy_pure` | **`cudaMalloc` + `cudaMemcpy`** | pure CUDA C — no Fortran at all |

Variants 2 and 3 link the **same kernel object** (`daxpy_kernel.o`), so the only
difference between them is who owns the memory.

## Results (Tesla V100-PCIE-32GB, cc70, nvhpc 26.5 / nvcc 12.9)

```
v1  do concurrent, OpenACC data          3.963 ms/rep   812.8 GB/s   PASS
v2  CUDA C via host_data use_device      3.975 ms/rep   810.4 GB/s   PASS
v3  CUDA C via cudaMalloc/cudaMemcpy     3.975 ms/rep   810.4 GB/s   PASS
```

**All three are the same speed.** v2 and v3 agree to four significant figures —
same kernel, same device memory, different allocator. So:

- **`do concurrent` costs nothing** vs hand-written CUDA C for a bandwidth-bound
  kernel. Both saturate HBM (V100 STREAM ceiling ≈ 900 GB/s). Reach for CUDA C
  when the kernel does something `do concurrent` cannot express — not for speed
  on an elementwise op.
- **OpenACC's `enter data` costs nothing** vs `cudaMalloc`. It is the same device
  allocation; the runtime is not adding overhead to the kernel.

## Does the memcpy matter? — yes, catastrophically (v3)

`daxpy_pure` measures what v1/v2 structurally cannot see, because they keep the
data resident and time only the kernel:

```
(a) KERNEL ONLY, device-resident
      3.97 ms/rep       810.4 GB/s   (HBM2)

(b) TRANSFERS over PCIe
      H2D pageable :  427.31 ms  ->   5.0 GB/s
      D2H pageable :  634.33 ms  ->   1.7 GB/s
      H2D pinned   :   87.17 ms  ->  12.3 GB/s
      D2H pinned   :   81.84 ms  ->  13.1 GB/s

(c) DOES THE MEMCPY MATTER?
      kernel alone         :    3.97 ms/rep
      + transfer every rep : 1065.61 ms/rep  (pageable)  = 268x slower
      + transfer every rep :  260.15 ms/rep  (pinned)    =  66x slower
      HBM : PCIe bandwidth : 66x
```

**The transfer is 66–268× the kernel.** A daxpy that ships its operands over PCIe
every step is not a GPU workload — it is a PCIe benchmark with a GPU attached.
Pinning host memory (`cudaMallocHost`) buys ~2.5× on H2D and ~8× on D2H by letting
the DMA engine read host pages directly instead of staging through a driver
bounce buffer — but 66× is still 66×.

**This is the whole argument for device residency.** It is why the sensible design
is to allocate once, keep state on the device, and transfer only for
diagnostics/IO — and why `-gpu=mem:separate` (no managed memory, no implicit
copies) is the honest way to build: it makes every transfer something you wrote
on purpose, instead of something the runtime does behind your back at 12 GB/s.

*Caveat on the pageable D2H (1.7 GB/s):* that destination buffer is `malloc`'d and
never touched before the copy, so the number includes first-touch page faults on
1 GB of fresh pages. Real, but pessimistic versus a warm buffer — do not quote it
as "pageable D2H bandwidth" in isolation.

## Build / run

```bash
source ../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
make && make run

make ARCH=cc80 NVARCH=sm_80    # A100
make ARCH=cc90 NVARCH=sm_90    # H100
```

## The one directive that does the work

```fortran
!$acc enter data copyin(a, b) create(c)      ! OpenACC owns the allocation
...
!$acc host_data use_device(a, b, c)          ! <- the bridge
call daxpy3d_cuda_launch(c, a, b, alpha, nx, ny, nz, 0)
!$acc end host_data
```

Inside `host_data use_device`, the names `a`/`b`/`c` evaluate to their **device**
addresses instead of their host addresses. A `bind(C)` call passes Fortran arrays
by reference — i.e. passes an address — so CUDA C receives a device pointer, and
a plain `double*` on the C side is all that is needed:

```c
extern "C" void daxpy3d_cuda_launch(double *c, const double *a, const double *b,
                                    double alpha, int nx, int ny, int nz, int sync)
```

`daxpy_cuda.F90` asserts this rather than trusting it — `daxpy3d_is_device_ptr()`
calls `cudaPointerGetAttributes` and checks for `cudaMemoryTypeDevice`. It prints
`a is device ptr = 1`. That check is the first thing to reach for if an interop
MRE misbehaves.

## Four things that will bite you

**1. `-cuda` must be LINK-ONLY.** It pulls in the CUDA runtime (without it:
undefined reference to `cudaLaunchKernel`) — but it *also* switches nvfortran
into CUDA Fortran mode, which type-checks device attributes. Inside `host_data`
the arrays carry the device attribute while the `bind(C)` interface declares
plain host arrays, so compiling with `-cuda` fails:

```
NVFORTRAN-S-0528-Argument number 1 to daxpy3d_cuda_launch: device attribute mismatch
```

Hence the Makefile's compile/link split. (Alternative: keep `-cuda` everywhere
and pass `type(c_ptr), value` + `c_loc()` instead of arrays-by-reference. Also
correct; this way keeps the call site idiomatic Fortran.)

**2. Fortran is column-major; your CUDA indexing must agree.**
`a(nx,ny,nz)` puts element `(i,j,k)` at `i + nx*(j + ny*k)` (0-based). So CUDA's
`x` dimension must map to Fortran's **first** index:

```c
const size_t idx = (size_t)i + (size_t)nx*((size_t)j + (size_t)ny*(size_t)k);
```

Map `x -> k` instead and it still computes the *right answer* while every warp
strides `nx*ny` doubles — an order of magnitude of bandwidth, lost silently.
Use `size_t`: `nx*ny*nz` overflows `int` at ~1290³, and 512³ is not far off.

**3. `mem:separate` means no safety net.** These flags use
`-gpu=cc70,mem:separate` — **no** managed/unified memory, so there are no
implicit host↔device copies. Every array a kernel touches must be explicitly
mapped or it silently reads stale host memory: no crash, just wrong numbers.
This is the interesting case, and the one that makes the `host_data` bridge real
rather than papered over by the runtime migrating pages behind your back. With
managed memory a *host* pointer would often appear to work, which is worse.

**4. The verification data must depend on all three indices.**
`daxpy_common.F90` fills `a(i,j,k) = i + 1000*j + 1000000*k`. Fill it with
`a(i,j,k) = i` and any index permutation still "verifies" — so the MRE would
pass with the linearisation transposed, which is the exact bug it exists to
catch. The check is for **bit-exact equality**, not a tolerance, for the same
reason: the expected values are exact in binary64, so a tolerance could only
hide a real error.

## Files

```
daxpy_kernel.cu    pure CUDA C: the kernel + an extern "C" launcher + the ptr assert
                   -- linked by BOTH v2 and v3, so the kernel cannot diverge
daxpy_common.F90   shared setup/verify/timing — identical for v1 and v2
daxpy_dc.F90       variant 1: OpenACC data + do concurrent
daxpy_cuda.F90     variant 2: OpenACC data + host_data use_device -> CUDA C
daxpy_pure.cu      variant 3: cudaMalloc + cudaMemcpy, no Fortran; measures transfers
Makefile           note the compile/link split, and no trailing comments on ARCH
```

All three verify **bit-exactly** against a host recomputation (not a tolerance —
the expected values are exact in binary64, so a tolerance could only hide a real
indexing error).
