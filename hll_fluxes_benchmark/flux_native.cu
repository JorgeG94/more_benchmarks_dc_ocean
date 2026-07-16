// NATIVE C++/CUDA driver for the coastal HLL flux kernel.
// cudaMalloc + main(). No Fortran, no OpenACC, no host_data, no runtime present table.
//
// THE QUESTION: every "CUDA C" number in this repo was produced by launching the
// kernel FROM flux_bench.F90, getting device pointers via `!$acc host_data
// use_device(...)`. That is a present-table lookup per array per call — 9 arrays
// here, the most of any bench in the repo — sitting on the CUDA path. If it costs
// anything, every "CUDA buys N%" number is understated and unfair to CUDA.
// This driver removes OpenACC entirely and settles it.
//
// THE KERNEL IS NOT REDEFINED HERE. This calls flux_hll_cuda_launch() from
// flux_kernel.cu — the identical launch flux_bench.F90 uses, same object file,
// same 32x8 block, same __restrict__ signature. Duplicating the kernel would let
// the two copies drift and silently corrupt the comparison, which IS the file.
// (Precedent and rationale: daxpy_benchmark/daxpy_pure.cu.)
//
// STATE: mirrors flux_bench.F90's init exactly — 2-D radial dam break on a wet
// bed, flat bed, hu = 0.05*i/nx, hv = -0.03*j/ny. Both depths (10 m inside,
// 2 m outside) are far above THIN_LAYER_THRESHOLD=1e-3, so the dry/thin branches
// NEVER fire and warps do not diverge on them. ALL-WET BY CONSTRUCTION: this
// measures the kernel, not the coast. A wet/dry state is a different experiment.
//
// OWN ARRAYS — DELIBERATE. flux_bench.F90 has a documented trap (RESUME §4):
// variants sharing an output array means only the LAST WRITER is ever checked,
// and three variants silently inherited "agreement OK" from whoever ran last.
// Every array below is cudaMalloc'd for this driver alone and read back by this
// driver alone. Nothing else can write them, so nothing can be inherited.
//
// VERIFICATION: run `FLUX_DUMP_REF=1 ./flux_bench` first. That dumps the inputs
// AND the CUDA-via-Fortran outputs (fh_cu/fhu_cu/fhv_cu/mx_cu/my_cu — arrays no
// other variant touches) to flux_ref_<nx>x<ny>x<ng>.bin. This driver reads it and
// compares BIT-FOR-BIT. Same kernel + same input must give the same bits, so any
// mismatch is an init mismatch, not rounding — which is exactly what needs
// proving, since a comparison against a differently-initialised state is
// meaningless. Inputs are checked separately from outputs so the two failures
// cannot be confused.
//
//   make native && ./flux_native [nphys_x [nphys_y [nghost [nreps]]]]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <string>
#include <cuda_runtime.h>

extern "C" void flux_hll_cuda_launch(const double *h, const double *hu, const double *hv,
                                     const double *b,
                                     double *flux_h, double *flux_hu, double *flux_hv,
                                     double *mass_flux_x, double *mass_flux_y,
                                     int nx, int ny, int nghost,
                                     double dx, double dy, int sync);

// Must match flux_bench.F90 exactly, or the comparison is meaningless.
static const int    DEF_NPHYS = 4096;   // interior cells per side
static const int    DEF_NGH   = 2;      // stencil reads i-2..i+2
static const int    DEF_REPS  = 20;
static const double DX = 10.0, DY = 10.0;
static const int    N_WARMUP = 10;      // brief: >= 10 untimed reps

#define CUDA_OK(call)                                                          \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(_e), __FILE__, __LINE__);          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Fortran 1-based (i,j) -> 0-based linear, column-major. Same rule flux_kernel.cu
// uses, because these are the same arrays it would have been handed.
static inline size_t idx(int i, int j, int nx) {
    return (size_t)(i - 1) + (size_t)nx * (size_t)(j - 1);
}

static double wall() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

// ---- init: byte-for-byte the same state as flux_bench.F90 -------------------
//
// Fortran:
//   b(i,j)  = 0.0_wp
//   if ((real(i - nx/2, wp)**2 + real(j - ny/2, wp)**2) < real(nx/4, wp)**2) then
//      h(i,j) = 10.0_wp   else   h(i,j) = 2.0_wp
//   hu(i,j) =  0.05_wp*real(i, wp)/real(nx, wp)
//   hv(i,j) = -0.03_wp*real(j, wp)/real(ny, wp)
//
// Two things that silently break bit-identity if you get them wrong, both
// preserved here:
//   * nx/2 and nx/4 are INTEGER division (nx=4100 -> 2050 and 1025), evaluated
//     before the conversion to real. Doing the divide in double moves the dam
//     radius and changes which cells are deep.
//   * i and j are 1-BASED, as in Fortran. The loops below run 1..nx to keep the
//     expressions readable against the original rather than rebasing them.
// The arithmetic order (0.05*i)/nx matches Fortran's left-to-right evaluation of
// `0.05_wp*real(i,wp)/real(nx,wp)`; a*(i/nx) would round differently.
static void init_state(double *h, double *hu, double *hv, double *b, int nx, int ny) {
    const double r = (double)(nx / 4);
    for (int j = 1; j <= ny; ++j) {
        const double dj = (double)(j - ny / 2);
        for (int i = 1; i <= nx; ++i) {
            const size_t k = idx(i, j, nx);
            const double di = (double)(i - nx / 2);
            b[k] = 0.0;
            h[k] = (di * di + dj * dj < r * r) ? 10.0 : 2.0;
            hu[k] = 0.05 * (double)i / (double)nx;
            hv[k] = -0.03 * (double)j / (double)ny;
        }
    }
}

// Bit-exact compare over the region the kernel actually defines: the interior
// nghost+1 .. n-nghost, which is what flux_bench.F90's own agreement check uses.
// Halo cells are never written (the device outputs are cudaMalloc'd here and
// `create`d — not zeroed — on the Fortran side), so comparing them would compare
// two piles of garbage and is not evidence of anything.
struct Cmp { size_t nbad; double max_abs; double max_rel; };

static Cmp compare_interior(const double *a, const double *r,
                            int nx, int ny, int ng) {
    Cmp c = {0, 0.0, 0.0};
    for (int j = ng + 1; j <= ny - ng; ++j)
        for (int i = ng + 1; i <= nx - ng; ++i) {
            const size_t k = idx(i, j, nx);
            const double d = std::fabs(a[k] - r[k]);
            if (d > 0.0) {
                ++c.nbad;
                if (d > c.max_abs) c.max_abs = d;
                const double s = std::fmax(std::fabs(a[k]), std::fabs(r[k]));
                if (s > 1.0e-30) { const double rel = d / s; if (rel > c.max_rel) c.max_rel = rel; }
            }
        }
    return c;
}

// A cheap fingerprint of the interior, printed whenever no reference file exists
// so a run at a non-default size still leaves something checkable behind.
static double checksum_interior(const double *a, int nx, int ny, int ng) {
    double s = 0.0;
    for (int j = ng + 1; j <= ny - ng; ++j)
        for (int i = ng + 1; i <= nx - ng; ++i) s += a[idx(i, j, nx)];
    return s;
}

int main(int argc, char **argv) {
    const int nphys_x = (argc > 1) ? std::atoi(argv[1]) : DEF_NPHYS;
    const int nphys_y = (argc > 2) ? std::atoi(argv[2]) : DEF_NPHYS;
    const int ngh     = (argc > 3) ? std::atoi(argv[3]) : DEF_NGH;
    const int nreps   = (argc > 4) ? std::atoi(argv[4]) : DEF_REPS;

    if (ngh < 2) { std::fprintf(stderr, "nghost must be >= 2 (stencil reads i-2..i+2)\n"); return 1; }

    const int nx = nphys_x + 2 * ngh;
    const int ny = nphys_y + 2 * ngh;
    const size_t n = (size_t)nx * (size_t)ny;
    const size_t bytes = n * sizeof(double);

    std::printf("%s\n", std::string(66, '=').c_str());
    std::printf("  driver : NATIVE C++/CUDA (cudaMalloc + main, no Fortran, no OpenACC)\n");
    std::printf("  kernel : flux_hll_cuda_launch() from flux_kernel.cu (NOT redefined here)\n");
    std::printf("  grid   : %d x %d total (%d x %d interior), nghost=%d\n",
                nx, ny, nphys_x, nphys_y, ngh);
    std::printf("  reps   : %d timed, %d warm-up (untimed)\n", nreps, N_WARMUP);
    std::printf("  state  : 2-D dam break, wet bed -- ALL-WET, dry branches never fire\n");
    std::printf("%s\n", std::string(66, '=').c_str());

    // ---- host: our own buffers, nobody else's ----------------------------
    double *h  = (double *)std::malloc(bytes);
    double *hu = (double *)std::malloc(bytes);
    double *hv = (double *)std::malloc(bytes);
    double *b  = (double *)std::malloc(bytes);
    double *fh = (double *)std::malloc(bytes);
    double *fhu = (double *)std::malloc(bytes);
    double *fhv = (double *)std::malloc(bytes);
    double *mx = (double *)std::malloc(bytes);
    double *my = (double *)std::malloc(bytes);
    if (!h || !hu || !hv || !b || !fh || !fhu || !fhv || !mx || !my) {
        std::fprintf(stderr, "host alloc failed (%.2f GB needed)\n", 9.0 * bytes / 1e9);
        return 1;
    }
    init_state(h, hu, hv, b, nx, ny);

    // ---- device: cudaMalloc, no present table, no host_data ---------------
    double *dh, *dhu, *dhv, *db, *dfh, *dfhu, *dfhv, *dmx, *dmy;
    CUDA_OK(cudaMalloc(&dh, bytes));   CUDA_OK(cudaMalloc(&dhu, bytes));
    CUDA_OK(cudaMalloc(&dhv, bytes));  CUDA_OK(cudaMalloc(&db, bytes));
    CUDA_OK(cudaMalloc(&dfh, bytes));  CUDA_OK(cudaMalloc(&dfhu, bytes));
    CUDA_OK(cudaMalloc(&dfhv, bytes)); CUDA_OK(cudaMalloc(&dmx, bytes));
    CUDA_OK(cudaMalloc(&dmy, bytes));

    CUDA_OK(cudaMemcpy(dh,  h,  bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dhu, hu, bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dhv, hv, bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(db,  b,  bytes, cudaMemcpyHostToDevice));
    // Zeroed so a cell the kernel fails to write shows up as 0 rather than as
    // whatever the allocator handed back. The kernel writes every interior cell,
    // so this is a safety net, not part of the measurement.
    CUDA_OK(cudaMemset(dfh, 0, bytes));  CUDA_OK(cudaMemset(dfhu, 0, bytes));
    CUDA_OK(cudaMemset(dfhv, 0, bytes)); CUDA_OK(cudaMemset(dmx, 0, bytes));
    CUDA_OK(cudaMemset(dmy, 0, bytes));

    // ---- warm-up: untimed ------------------------------------------------
    for (int r = 0; r < N_WARMUP; ++r)
        flux_hll_cuda_launch(dh, dhu, dhv, db, dfh, dfhu, dfhv, dmx, dmy,
                             nx, ny, ngh, DX, DY, 0);
    CUDA_OK(cudaDeviceSynchronize());

    // ---- timed: TWO SEPARATE PASSES, deliberately --------------------------
    //
    // Pass 1 (mean) must contain NOTHING the Fortran driver does not also do.
    // flux_bench.F90 times `system_clock` around a bare loop of launches, so this
    // loop is a bare loop of launches -- no events, no queries, no host work.
    //
    // This is not fussiness. The first version of this file recorded two
    // cudaEvents per rep INSIDE the timed loop. At 4096^2 that is invisible under
    // a 5 ms kernel, but at small sizes the launches are issue-bound and the two
    // extra host-side API calls per rep inflated the native mean by ~3x -- making
    // native look slower than Fortran at exactly the sizes where host-side cost is
    // the thing being measured. The artifact pointed the same direction as the
    // hypothesis, which is how it nearly survived. Keep the passes separate.
    //
    // Pass 2 (min/max) uses per-rep events and is timed on the DEVICE, so its
    // host-side cost does not enter the numbers it reports. Under contention min
    // is the more honest estimate of the kernel's own cost: it is the rep that
    // got the least interference.
    const double t0 = wall();
    for (int r = 0; r < nreps; ++r)
        flux_hll_cuda_launch(dh, dhu, dhv, db, dfh, dfhu, dfhv, dmx, dmy,
                             nx, ny, ngh, DX, DY, 0);
    CUDA_OK(cudaDeviceSynchronize());
    const double t1 = wall();
    const double mean_ms = (t1 - t0) * 1000.0 / (double)nreps;

    cudaEvent_t *beg = new cudaEvent_t[nreps];
    cudaEvent_t *end = new cudaEvent_t[nreps];
    for (int r = 0; r < nreps; ++r) {
        CUDA_OK(cudaEventCreate(&beg[r]));
        CUDA_OK(cudaEventCreate(&end[r]));
    }

    for (int r = 0; r < nreps; ++r) {
        CUDA_OK(cudaEventRecord(beg[r]));
        flux_hll_cuda_launch(dh, dhu, dhv, db, dfh, dfhu, dfhv, dmx, dmy,
                             nx, ny, ngh, DX, DY, 0);
        CUDA_OK(cudaEventRecord(end[r]));
    }
    CUDA_OK(cudaDeviceSynchronize());
    double min_ms = 1.0e30, max_ms = 0.0, sum_ms = 0.0;
    for (int r = 0; r < nreps; ++r) {
        float ms = 0.f;
        CUDA_OK(cudaEventElapsedTime(&ms, beg[r], end[r]));
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
        sum_ms += ms;
    }
    const double ev_mean_ms = sum_ms / (double)nreps;

    // ---- results back ----------------------------------------------------
    CUDA_OK(cudaMemcpy(fh,  dfh,  bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(fhu, dfhu, bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(fhv, dfhv, bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(mx,  dmx,  bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(my,  dmy,  bytes, cudaMemcpyDeviceToHost));

    std::printf("  kernel time (device-resident, no transfers in the loop)\n");
    std::printf("        min of %-4d : %10.4f ms/rep   <- device events, least-disturbed rep\n",
                nreps, min_ms);
    std::printf("        mean        : %10.4f ms/rep   <- bare launch loop == flux_bench.F90\n",
                mean_ms);
    std::printf("        event mean  : %10.4f ms/rep\n", ev_mean_ms);
    std::printf("        max         : %10.4f ms/rep   (spread %.2fx over min)\n",
                max_ms, max_ms / min_ms);
    if (max_ms / min_ms > 1.30)
        std::printf("        ^^ SPREAD > 1.3x: another process is very likely sharing this\n"
                    "           GPU. Treat the mean as contended. Check:\n"
                    "           nvidia-smi --query-compute-apps=pid --format=csv\n");
    std::printf("%s\n", std::string(66, '-').c_str());

    // ---- verify against the Fortran run's output -------------------------
    char ref_name[256];
    std::snprintf(ref_name, sizeof(ref_name), "flux_ref_%dx%dx%d.bin", nx, ny, ngh);
    std::FILE *fp = std::fopen(ref_name, "rb");
    int rc = 0;

    if (!fp) {
        std::printf("  verify : NO REFERENCE. %s not found.\n", ref_name);
        std::printf("           Produce it with:  FLUX_DUMP_REF=1 ./flux_bench\n");
        std::printf("           (only exists for flux_bench.F90's compiled-in size)\n");
        std::printf("           interior checksums, for the record:\n");
        std::printf("             sum(flux_h)  = %+.17e\n", checksum_interior(fh, nx, ny, ngh));
        std::printf("             sum(flux_hu) = %+.17e\n", checksum_interior(fhu, nx, ny, ngh));
        std::printf("             sum(flux_hv) = %+.17e\n", checksum_interior(fhv, nx, ny, ngh));
    } else {
        // Layout written by flux_bench.F90: h, hu, hv, b, fh_cu, fhu_cu, fhv_cu,
        // mx_cu, my_cu -- 9 arrays, unformatted stream, column-major, nx*ny each.
        double *ref = (double *)std::malloc(bytes);
        if (!ref) { std::fprintf(stderr, "ref buffer alloc failed\n"); return 1; }

        const char *names[9] = {"h", "hu", "hv", "b", "flux_h", "flux_hu", "flux_hv",
                                "mass_flux_x", "mass_flux_y"};
        const double *ours[9] = {h, hu, hv, b, fh, fhu, fhv, mx, my};
        size_t total_bad = 0;

        // Inputs first, and reported separately. If the INIT diverged, every
        // output difference is a consequence of it and says nothing about the
        // port -- so the two must not be pooled into one pass/fail.
        std::printf("  verify : bit-exact vs %s (same kernel, so any diff = init mismatch)\n",
                    ref_name);
        for (int a = 0; a < 9; ++a) {
            if (std::fread(ref, sizeof(double), n, fp) != n) {
                std::fprintf(stderr, "  short read on array %s -- stale/truncated reference?\n",
                             names[a]);
                rc = 1; break;
            }
            // Inputs: compare everything, halo included -- we wrote all of it.
            // Outputs: interior only; the halo is never written by the kernel.
            Cmp c;
            if (a < 4) {
                Cmp t = {0, 0.0, 0.0};
                for (size_t k = 0; k < n; ++k) {
                    const double d = std::fabs(ours[a][k] - ref[k]);
                    if (d > 0.0) {
                        ++t.nbad;
                        if (d > t.max_abs) t.max_abs = d;
                        const double s = std::fmax(std::fabs(ours[a][k]), std::fabs(ref[k]));
                        if (s > 1.0e-30) { const double rl = d / s; if (rl > t.max_rel) t.max_rel = rl; }
                    }
                }
                c = t;
            } else {
                c = compare_interior(ours[a], ref, nx, ny, ngh);
            }
            total_bad += c.nbad;
            std::printf("        %-12s %-8s : %8zu differ", names[a],
                        a < 4 ? "(input)" : "(output)", c.nbad);
            if (c.nbad) std::printf("   max abs %.5e   max rel %.5e", c.max_abs, c.max_rel);
            std::printf("\n");
        }
        std::free(ref);
        std::fclose(fp);

        std::printf("%s\n", std::string(66, '-').c_str());
        if (total_bad == 0)
            std::printf("  verify : PASS -- BIT-IDENTICAL to the CUDA-via-Fortran run,\n"
                        "           inputs and outputs, all 9 arrays. The native driver\n"
                        "           runs the same kernel on the same state.\n");
        else {
            std::printf("  verify : *** FAIL *** %zu elements differ.\n", total_bad);
            rc = 1;
        }
    }
    std::printf("%s\n", std::string(66, '=').c_str());

    for (int r = 0; r < nreps; ++r) { cudaEventDestroy(beg[r]); cudaEventDestroy(end[r]); }
    delete[] beg; delete[] end;
    cudaFree(dh); cudaFree(dhu); cudaFree(dhv); cudaFree(db);
    cudaFree(dfh); cudaFree(dfhu); cudaFree(dfhv); cudaFree(dmx); cudaFree(dmy);
    std::free(h); std::free(hu); std::free(hv); std::free(b);
    std::free(fh); std::free(fhu); std::free(fhv); std::free(mx); std::free(my);
    return rc;
}
