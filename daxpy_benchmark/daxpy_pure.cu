// VARIANT 3 of 3 — entirely CUDA C. No Fortran, no OpenACC, no host_data.
// cudaMalloc + cudaMemcpy + the same kernel.
//
// PURPOSE: answer "does the memcpy matter for performance?"
//
// Variants 1 and 2 both keep the data device-resident and time ONLY the kernel,
// so neither of them can see the transfer cost. This one measures it explicitly
// and reports three numbers that are easy to conflate:
//
//   (a) kernel-only        — device-resident, directly comparable to v1/v2
//   (b) H2D / D2H          — the PCIe transfer, pageable AND pinned
//   (c) end-to-end per rep — what you pay if you must transfer every step
//
// The kernel itself is NOT redefined here: this calls daxpy3d_cuda_launch() from
// daxpy_kernel.cu, the identical kernel v2 uses. If the kernel were duplicated,
// a divergence between the two copies would silently corrupt the comparison —
// and the comparison is the entire point of the file.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

extern "C" void daxpy3d_cuda_launch(double *c, const double *a, const double *b,
                                    double alpha, int nx, int ny, int nz, int sync);

// Must match daxpy_common.F90 exactly, or the comparison is meaningless.
static const int    NX = 512, NY = 512, NZ = 512;
static const double ALPHA = 2.5;
static const int    N_REPS = 20;

#define CUDA_OK(call)                                                          \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(_e), __FILE__, __LINE__);          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Same values as daxpy_common.F90:fill_inputs, and same reason: every element
// depends on all three indices, so a transposed linearisation cannot pass.
static inline double a_val(int i, int j, int k) {
    return (double)(i + 1) + 1000.0 * (double)(j + 1) + 1000000.0 * (double)(k + 1);
}
static inline double b_val(int i, int j, int k) {
    return (double)(i + 1) - 0.5 * (double)(j + 1) + 0.25 * (double)(k + 1);
}

struct Timer {
    cudaEvent_t beg, end;
    Timer()  { cudaEventCreate(&beg); cudaEventCreate(&end); }
    ~Timer() { cudaEventDestroy(beg); cudaEventDestroy(end); }
    void start() { CUDA_OK(cudaEventRecord(beg)); }
    float stop_ms() {
        CUDA_OK(cudaEventRecord(end));
        CUDA_OK(cudaEventSynchronize(end));
        float ms = 0.f;
        CUDA_OK(cudaEventElapsedTime(&ms, beg, end));
        return ms;
    }
};

int main() {
    const size_t n     = (size_t)NX * NY * NZ;
    const size_t bytes = n * sizeof(double);
    const double gb    = (double)bytes / 1.0e9;

    std::printf("  variant        : pure CUDA C (cudaMalloc + cudaMemcpy)\n");
    std::printf("  grid           : %d x %d x %d  (%zu elements, %.3f GB/array)\n",
                NX, NY, NZ, n, gb);

    // ---- host buffers: PAGEABLE (the default, what most code does) --------
    double *ha = (double *)std::malloc(bytes);
    double *hb = (double *)std::malloc(bytes);
    double *hc = (double *)std::malloc(bytes);
    if (!ha || !hb || !hc) { std::fprintf(stderr, "host alloc failed\n"); return 1; }

    for (int k = 0; k < NZ; ++k)
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                const size_t idx = (size_t)i + (size_t)NX * ((size_t)j + (size_t)NY * k);
                ha[idx] = a_val(i, j, k);
                hb[idx] = b_val(i, j, k);
            }

    // ---- device buffers ---------------------------------------------------
    double *da, *db, *dc;
    CUDA_OK(cudaMalloc(&da, bytes));
    CUDA_OK(cudaMalloc(&db, bytes));
    CUDA_OK(cudaMalloc(&dc, bytes));

    Timer t;

    // ---- (b1) H2D, PAGEABLE ----------------------------------------------
    // Pageable memory cannot be DMA'd directly: the driver stages it through an
    // internal pinned buffer, so this is really memcpy-then-DMA.
    t.start();
    CUDA_OK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));
    const float h2d_pageable_ms = t.stop_ms();

    // ---- (a) kernel-only, device-resident --------------------------------
    // Identical measurement to variants 1 and 2: data already on device,
    // nothing transferred inside the loop.
    daxpy3d_cuda_launch(dc, da, db, ALPHA, NX, NY, NZ, 1);   // warm-up
    t.start();
    for (int r = 0; r < N_REPS; ++r)
        daxpy3d_cuda_launch(dc, da, db, ALPHA, NX, NY, NZ, 0);
    const float kernel_total_ms = t.stop_ms();
    const float kernel_ms = kernel_total_ms / (float)N_REPS;

    // ---- (b2) D2H, PAGEABLE ----------------------------------------------
    t.start();
    CUDA_OK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
    const float d2h_pageable_ms = t.stop_ms();

    // ---- verify (bit-exact, same bar as the Fortran variants) ------------
    size_t n_bad = 0;
    double max_err = 0.0;
    for (int k = 0; k < NZ; ++k)
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                const size_t idx = (size_t)i + (size_t)NX * ((size_t)j + (size_t)NY * k);
                const double expect = ALPHA * a_val(i, j, k) + b_val(i, j, k);
                const double err = std::fabs(hc[idx] - expect);
                if (err > 0.0) { ++n_bad; if (err > max_err) max_err = err; }
            }

    // ---- (b3) H2D/D2H, PINNED -------------------------------------------
    // cudaMallocHost gives page-locked memory the DMA engine can read directly:
    // no staging copy. This is the first lever anyone reaches for when
    // transfers hurt, so measuring it is part of answering the question.
    double *pa, *pc;
    CUDA_OK(cudaMallocHost(&pa, bytes));
    CUDA_OK(cudaMallocHost(&pc, bytes));
    std::memcpy(pa, ha, bytes);

    t.start();
    CUDA_OK(cudaMemcpy(da, pa, bytes, cudaMemcpyHostToDevice));
    const float h2d_pinned_ms = t.stop_ms();

    t.start();
    CUDA_OK(cudaMemcpy(pc, dc, bytes, cudaMemcpyDeviceToHost));
    const float d2h_pinned_ms = t.stop_ms();

    // ---- report -----------------------------------------------------------
    // Bandwidth conventions:
    //   kernel moves 3 arrays (read a, read b, write c) -> 3*gb
    //   H2D above moved 2 arrays (a and b)              -> 2*gb
    //   D2H moved 1 array (c)                           -> 1*gb
    const double kernel_gbps = (3.0 * gb) / (kernel_ms / 1000.0);
    const double h2d_pg_gbps = (2.0 * gb) / (h2d_pageable_ms / 1000.0);
    const double d2h_pg_gbps = (1.0 * gb) / (d2h_pageable_ms / 1000.0);
    const double h2d_pn_gbps = (1.0 * gb) / (h2d_pinned_ms / 1000.0);
    const double d2h_pn_gbps = (1.0 * gb) / (d2h_pinned_ms / 1000.0);

    // The punchline: one rep if you had to ship a+b over and c back each time.
    const double per_rep_pageable = kernel_ms + (h2d_pageable_ms + d2h_pageable_ms);
    const double per_rep_pinned   = kernel_ms + (2.0 * h2d_pinned_ms + d2h_pinned_ms);

    std::printf("  reps           : %d\n", N_REPS);
    std::printf("%s\n", "  ------------------------------------------------------------");
    std::printf("  (a) KERNEL ONLY, device-resident  -- compare to v1 / v2\n");
    std::printf("        time / rep : %10.5f ms\n", kernel_ms);
    std::printf("        bandwidth  : %8.1f GB/s   (HBM2)\n", kernel_gbps);
    std::printf("%s\n", "  ------------------------------------------------------------");
    std::printf("  (b) TRANSFERS over PCIe\n");
    std::printf("        H2D pageable : %8.2f ms  -> %6.1f GB/s\n", h2d_pageable_ms, h2d_pg_gbps);
    std::printf("        D2H pageable : %8.2f ms  -> %6.1f GB/s\n", d2h_pageable_ms, d2h_pg_gbps);
    std::printf("        H2D pinned   : %8.2f ms  -> %6.1f GB/s\n", h2d_pinned_ms, h2d_pn_gbps);
    std::printf("        D2H pinned   : %8.2f ms  -> %6.1f GB/s\n", d2h_pinned_ms, d2h_pn_gbps);
    std::printf("%s\n", "  ------------------------------------------------------------");
    std::printf("  (c) DOES THE MEMCPY MATTER?\n");
    std::printf("        kernel alone            : %8.2f ms/rep\n", kernel_ms);
    std::printf("        + transfer every rep    : %8.2f ms/rep  (pageable)  = %.1fx slower\n",
                per_rep_pageable, per_rep_pageable / kernel_ms);
    std::printf("        + transfer every rep    : %8.2f ms/rep  (pinned)    = %.1fx slower\n",
                per_rep_pinned, per_rep_pinned / kernel_ms);
    std::printf("        HBM : PCIe bandwidth    : %.0fx\n", kernel_gbps / h2d_pn_gbps);
    std::printf("%s\n", "  ------------------------------------------------------------");
    if (n_bad == 0) {
        std::printf("  verify         : PASS (bit-exact vs host)\n");
    } else {
        std::printf("  verify         : *** FAIL *** %zu wrong, max err %.5e\n", n_bad, max_err);
    }

    cudaFreeHost(pa); cudaFreeHost(pc);
    cudaFree(da); cudaFree(db); cudaFree(dc);
    std::free(ha); std::free(hb); std::free(hc);
    return n_bad == 0 ? 0 : 1;
}
