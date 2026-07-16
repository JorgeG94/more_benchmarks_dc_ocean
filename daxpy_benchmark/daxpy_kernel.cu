// daxpy_kernel.cu — pure CUDA C, no CUDA Fortran, no OpenACC.
//
// This file knows NOTHING about how the memory it is handed was allocated.
// It receives three raw device pointers and treats them as device memory.
// That is the whole point of the MRE: OpenACC owns the allocation, CUDA C
// owns the compute, and the only contract between them is "this is a valid
// device address".
//
// INDEXING — the one thing that is easy to get wrong.
// The caller's arrays are Fortran `a(nx,ny,nz)`, which is COLUMN-MAJOR:
// element (i,j,k) sits at linear offset  i + nx*(j + ny*k)  with 0-based
// i,j,k. So the CUDA x-dimension must map to Fortran's FIRST index for
// threads in a warp to read consecutive addresses (coalesced). Map x->k
// instead and it still computes the right answer, but every warp strides
// nx*ny doubles and you lose ~an order of magnitude of bandwidth. Silent.

#include <cstdio>
#include <cuda_runtime.h>

__global__ void daxpy3d_kernel(double *__restrict__ c,
                               const double *__restrict__ a,
                               const double *__restrict__ b,
                               const double alpha,
                               const int nx, const int ny, const int nz)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // Fortran 1st index
    const int j = blockIdx.y * blockDim.y + threadIdx.y;   // Fortran 2nd index
    const int k = blockIdx.z * blockDim.z + threadIdx.z;   // Fortran 3rd index

    if (i < nx && j < ny && k < nz) {
        // Fortran column-major linearisation. size_t: nx*ny*nz overflows
        // int at ~1290^3, and this MRE's default 512^3 is close enough to
        // that to be worth doing right.
        const size_t idx = (size_t)i
                         + (size_t)nx * ((size_t)j + (size_t)ny * (size_t)k);
        c[idx] = alpha * a[idx] + b[idx];
    }
}

// Called from Fortran via bind(C). Arrays arrive by reference (Fortran
// default), so these are already `double*`; alpha/nx/ny/nz arrive by VALUE
// because the Fortran interface declares them `value`.
//
// `sync != 0` makes the launch blocking, which is what the timing loop wants.
// Left to the caller because a real code would rather batch launches and let
// OpenACC's `!$acc wait` do the synchronising.
extern "C" void daxpy3d_cuda_launch(double *c, const double *a, const double *b,
                                    double alpha, int nx, int ny, int nz,
                                    int sync)
{
    // 128 threads = 4 warps/block, all in x, so consecutive threads touch
    // consecutive addresses along Fortran's fast index.
    const dim3 block(128, 1, 1);
    const dim3 grid((nx + block.x - 1) / block.x,
                    (ny + block.y - 1) / block.y,
                    (nz + block.z - 1) / block.z);

    daxpy3d_kernel<<<grid, block>>>(c, a, b, alpha, nx, ny, nz);

    // A launch failure here is asynchronous and would otherwise surface as a
    // wrong answer several calls later, so check both the launch and (if
    // asked) the execution.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    if (sync) {
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            std::fprintf(stderr, "CUDA sync failed: %s\n", cudaGetErrorString(err));
    }
}

// Proves the pointer Fortran handed us is really device memory. If OpenACC's
// use_device_addr were mis-wired and we got a HOST pointer, this reports
// cudaMemoryTypeUnregistered (or fails) instead of segfaulting somewhere
// less informative. Diagnostic only.
extern "C" int daxpy3d_is_device_ptr(const void *p)
{
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, p);
    if (err != cudaSuccess) { cudaGetLastError(); return 0; }
    return (attr.type == cudaMemoryTypeDevice) ? 1 : 0;
}
