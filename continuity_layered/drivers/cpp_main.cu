// NATIVE driver for the LAYERED continuity-PPM kernel: cudaMalloc + main().
// No Fortran, no OpenACC, no host_data, no nvfortran runtime.
//
// THE QUESTION IT ANSWERS
// -----------------------
// Every CUDA number in this repo was produced by launching the kernel FROM
// Fortran, handing it device pointers via `!$acc host_data use_device(...)`.
// That directive is not free by construction: it is a present-table lookup, per
// array, per call -- fourteen of them here, every rep. So the OpenACC runtime
// sits on the CUDA path in every measurement, and if it costs anything then
// every "CUDA buys N%" figure in this repo is understated and the comparison is
// unfair to CUDA. This driver owns its memory outright and removes OpenACC from
// the picture, which settles it: whatever the delta is between
// ./layered_bench's CUDA column and this binary IS the OpenACC path's price.
//
// THE KERNEL IS NOT REDEFINED HERE. This calls continuity_layered_cuda_launch()
// from layered_kernel.cu -- the identical object layered_bench links. Following
// daxpy_pure.cu's rule, and for its reason: two copies of an eleven-kernel PPM
// scheme would drift, and the drift would show up as a timing result rather
// than as a compile error. If you are editing a __global__ body to make this
// file work, you have taken a wrong turn.
//
// SYNC MODEL: `cuda_sync` (0=async, 1=per-rep, 2=per-kernel) has the same
// meaning here as in layered_bench.F90 and the same DEFAULT, 2. It exists
// because plain `do concurrent` blocks the host at every loop; unless CUDA is
// made to pay that same tax the ratio flatters it. Reported numbers below are
// cuda_sync=2 unless the banner says otherwise.
//
// INIT: mirrors layered_bench.F90 line for line -- all-wet (wet_T=1), Gaussian
// bump + k-dependent stratification, u and v of BOTH signs so the upwind branch
// in the transport loops goes both ways, use_ppm_limit_pos=.false. A different
// init would leave the two binaries computing different problems and make the
// comparison meaningless, so `verify` below checks the init itself against the
// Fortran dump rather than taking this comment's word for it.
//
// nx_phys/ny_phys are INTERIOR cells; nx_total = nx_phys + 2*nghost, nghost=3.
// The kernel is ghost-aware -- it zeroes mass flux at the PHYSICAL edges
// (nghost+1, nghost+nx_phys+1) -- so that relationship must hold or the walls
// land in the wrong place and the answer is quietly wrong.
//
// THREE STRIDE CLASSES. Mixing them is silent corruption, not a compile error:
//     h, flux_h                        (nx,   ny,   nz)
//     u, mass_flux_x, h_face_*_x       (nx+1, ny,   nz)
//     v, mass_flux_y, h_face_*_y       (nx,   ny+1, nz)
// Metrics stay 2-D: wet_T/iareaT (nx,ny), dy_cu (nx+1,ny), dx_cv (nx,ny+1).
//
// Usage: ./layered_native [nx_phys] [ny_phys] [nz] [nreps] [nwarm] [cuda_sync]
//   ./layered_native                     -> 473 297 30 (the 0.1 deg config)
//   ./layered_native 945 594 30          -> the 0.05 deg hi-res config
//   ./layered_native 473 297 1           -> the nz=1 control (145k cells)
// Verification is automatic when ./fh_ref.bin is present:
//   ./layered_bench 473 297 30 200 10 2 1 && ./layered_native

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include "gpu_rt.h"

extern "C" void continuity_layered_cuda_launch(
    const double *h, const double *u, const double *v, const double *wet_T,
    const double *dy_cu, const double *dx_cv, const double *iareaT,
    double *hfl_x, double *hfr_x, double *hfl_y, double *hfr_y,
    double *mfx, double *mfy, double *fh,
    int nx, int ny, int nz, int nghost, int nx_phys, int ny_phys,
    int do_pos, double h_min_pos, int sync);

// Must match layered_bench.F90's parameters exactly.
static const int    NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30;
static const int    REPS_DEF = 200, WARM_DEF = 10, NGHOST = 3;
static const double DX = 10.0, DY = 10.0;
static const double H_MIN = 1.0e-6;
static const int    DO_POS = 0;            // use_ppm_limit_pos = .false.
static const char  *REF_FILE = "fh_ref.bin";

#define CUDA_OK(call)                                                          \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(_e), __FILE__, __LINE__);          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// system_clock() in the Fortran driver; same clock, same units.
static double wall() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

static int iarg(int argc, char **argv, int k, int dflt) {
    if (argc <= k) return dflt;
    char *end = nullptr;
    long v = std::strtol(argv[k], &end, 10);
    if (end == argv[k] || *end != '\0') return dflt;
    return (int)v;
}

// Comparison of one array against the Fortran dump. Bit-identity is the bar the
// Fortran driver already meets between its own two variants (max|diff| = 0.0
// exactly), so anything else here is this driver's fault, not the kernel's.
struct Diff {
    double dmax = 0.0, rmax = 0.0;
    size_t n_bad = 0;
    void feed(double a, double b) {
        const double d = std::fabs(a - b);
        if (d > dmax) dmax = d;
        const double scale = std::max(std::fabs(a), std::fabs(b));
        if (scale > 1.0e-30) {
            const double r = d / scale;
            if (r > rmax) rmax = r;
            if (r > 1.0e-12) ++n_bad;
        }
    }
};

// Pointwise relative error is the right bar for the INPUTS (h, u, v are O(1)
// smooth fields). It is the WRONG bar for flux_h_layer: that field is a
// divergence -- (mfx(i+1)-mfx(i)) + (mfy(j+1)-mfy(j)) -- so cells where the
// fluxes nearly cancel have fh ~ 0 and any perturbation at all is 100%
// "relative" error there. Judging fh by pointwise rel would flag a 1-ulp input
// wobble as a port bug. fh is therefore scored against the FIELD's magnitude,
// and the bit-identity claim is made where it is actually meaningful: the
// ref-input pass below, which removes the input wobble entirely.
static void report_rel(const char *what, const Diff &d, double bar) {
    const char *verdict = (d.dmax == 0.0) ? "BIT-IDENTICAL"
                          : (d.rmax < bar) ? "ok (ULP-level)"
                                           : "*** SUSPECT ***";
    std::printf("    %-22s max|diff| %11.4e   max rel %11.4e   %s\n",
                what, d.dmax, d.rmax, verdict);
}

static void report_abs(const char *what, const Diff &d, double scale) {
    const double rel_field = (scale > 0.0) ? d.dmax / scale : 0.0;
    const char *verdict = (d.dmax == 0.0) ? "BIT-IDENTICAL"
                          : (rel_field < 1.0e-12) ? "ok (ULP-level vs field)"
                                                  : "*** SUSPECT ***";
    std::printf("    %-22s max|diff| %11.4e   /max|fh| %11.4e   %s\n",
                what, d.dmax, rel_field, verdict);
}

int main(int argc, char **argv) {
    const int nxp    = iarg(argc, argv, 1, NXP_DEF);
    const int nyp    = iarg(argc, argv, 2, NYP_DEF);
    const int nz     = iarg(argc, argv, 3, NZ_DEF);
    const int n_reps = iarg(argc, argv, 4, REPS_DEF);
    const int n_warm = iarg(argc, argv, 5, WARM_DEF);
    const int cu_sync = iarg(argc, argv, 6, 2);

    const int nx = nxp + 2 * NGHOST;
    const int ny = nyp + 2 * NGHOST;
    if (nx < 6 || ny < 6 || nz < 1) {
        std::fprintf(stderr, "ERROR: need nx_phys,ny_phys >= 1 and nz >= 1\n");
        return 1;
    }

    // Three stride classes -- see the header.
    const size_t nT = (size_t)nx * ny * nz;
    const size_t nU = (size_t)(nx + 1) * ny * nz;
    const size_t nV = (size_t)nx * (ny + 1) * nz;
    const size_t mT = (size_t)nx * ny;
    const size_t mU = (size_t)(nx + 1) * ny;
    const size_t mV = (size_t)nx * (ny + 1);
    const double gib = 17.0 * (double)nx * ny * nz * 8.0 / (1024.0 * 1024.0 * 1024.0);

    std::printf("%s\n", std::string(70, '=').c_str());
    std::printf("  driver: NATIVE C++/CUDA (cudaMalloc + main), no Fortran, no OpenACC\n");
    std::printf("  domain: %d x %d x %d interior (+%d ghosts/side)\n", nxp, nyp, nz, NGHOST);
    std::printf("  arrays: %d x %d x %d = %zu cells\n", nx, ny, nz, nT);
    std::printf("  reps:   %d timed, %d warm-up\n", n_reps, n_warm);
    std::printf("  device memory ~%.2f GiB;  cuda_sync = %d%s\n", gib, cu_sync,
                cu_sync == 2 ? "  (per-kernel -- matches do concurrent)" : "");
    std::printf("%s\n", std::string(70, '=').c_str());

    // ---- host init: a line-for-line mirror of layered_bench.F90 -----------
    // Fortran is 1-based, so every loop below runs the Fortran index directly
    // and subtracts 1 only in the linearisation. nx/2 and nx/8 are INTEGER
    // divisions there; C agrees for positive operands, which is the only case
    // that occurs. Getting this wrong is not a crash -- it is a slightly
    // different Gaussian, and a comparison that silently means nothing.
    std::vector<double> h(nT), u(nU), v(nV), fh(nT, 0.0);
    std::vector<double> wet_T(mT), iareaT(mT), dy_cu(mU), dx_cv(mV);

    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                const double gi = (double)(i - nx / 2) / (double)std::max(nx / 8, 1);
                const double gj = (double)(j - ny / 2) / (double)std::max(ny / 8, 1);
                h[(size_t)(i - 1) + (size_t)nx * ((size_t)(j - 1) + (size_t)ny * (k - 1))] =
                    20.0 + (double)k * 0.5
                    + 5.0 * std::exp(-(gi * gi + gj * gj))
                    + 0.4 * std::sin(0.01 * (double)i) * std::cos(0.013 * (double)j);
            }
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i) {
            wet_T[(size_t)(i - 1) + (size_t)nx * (j - 1)]  = 1.0;      // all-wet
            iareaT[(size_t)(i - 1) + (size_t)nx * (j - 1)] = 1.0 / (DX * DY);
        }
    // Both signs of u/v, so the upwind branch goes both ways.
    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx + 1; ++i)
                u[(size_t)(i - 1) + (size_t)(nx + 1) * ((size_t)(j - 1) + (size_t)ny * (k - 1))] =
                    0.5 * std::sin(0.02 * (double)i) * std::cos(0.03 * (double)k);
    for (int k = 1; k <= nz; ++k)
        for (int j = 1; j <= ny + 1; ++j)
            for (int i = 1; i <= nx; ++i)
                v[(size_t)(i - 1) + (size_t)nx * ((size_t)(j - 1) + (size_t)(ny + 1) * (k - 1))] =
                    0.3 * std::cos(0.017 * (double)j);
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx + 1; ++i) dy_cu[(size_t)(i - 1) + (size_t)(nx + 1) * (j - 1)] = DY;
    for (int j = 1; j <= ny + 1; ++j)
        for (int i = 1; i <= nx; ++i) dx_cv[(size_t)(i - 1) + (size_t)nx * (j - 1)] = DX;

    // ---- device buffers ---------------------------------------------------
    double *d_h, *d_u, *d_v, *d_wet, *d_dycu, *d_dxcv, *d_iarea;
    double *d_hflx, *d_hfrx, *d_hfly, *d_hfry, *d_mfx, *d_mfy, *d_fh;
    CUDA_OK(cudaMalloc(&d_h,     nT * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_u,     nU * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_v,     nV * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_wet,   mT * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_iarea, mT * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_dycu,  mU * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_dxcv,  mV * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_hflx,  nU * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_hfrx,  nU * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_hfly,  nV * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_hfry,  nV * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_mfx,   nU * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_mfy,   nV * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_fh,    nT * sizeof(double)));

    CUDA_OK(cudaMemcpy(d_h,     h.data(),      nT * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_u,     u.data(),      nU * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_v,     v.data(),      nV * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_wet,   wet_T.data(),  mT * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_iarea, iareaT.data(), mT * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_dycu,  dy_cu.data(),  mU * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_dxcv,  dx_cv.data(),  mV * sizeof(double), cudaMemcpyHostToDevice));
    // The Fortran driver zeroes these before `enter data`; cudaMalloc does not,
    // and the PPM/boundary loops do not write every element of the face arrays.
    CUDA_OK(cudaMemset(d_hflx, 0, nU * sizeof(double)));
    CUDA_OK(cudaMemset(d_hfrx, 0, nU * sizeof(double)));
    CUDA_OK(cudaMemset(d_hfly, 0, nV * sizeof(double)));
    CUDA_OK(cudaMemset(d_hfry, 0, nV * sizeof(double)));
    CUDA_OK(cudaMemset(d_mfx,  0, nU * sizeof(double)));
    CUDA_OK(cudaMemset(d_mfy,  0, nV * sizeof(double)));
    CUDA_OK(cudaMemset(d_fh,   0, nT * sizeof(double)));

#define LAUNCH(SYNC)                                                            \
    continuity_layered_cuda_launch(d_h, d_u, d_v, d_wet, d_dycu, d_dxcv, d_iarea, \
                                   d_hflx, d_hfrx, d_hfly, d_hfry,              \
                                   d_mfx, d_mfy, d_fh, nx, ny, nz, NGHOST,      \
                                   nxp, nyp, DO_POS, H_MIN, (SYNC))

    // ---- warm-up (untimed) ------------------------------------------------
    for (int rep = 0; rep < n_warm; ++rep) LAUNCH(1);
    CUDA_OK(cudaDeviceSynchronize());

    // ---- timed ------------------------------------------------------------
    // Same shape as layered_bench.F90's CUDA loop, deliberately: n_reps calls at
    // cu_sync, then ONE extra call at sync=1 to force a device sync (needed when
    // cu_sync=0), divided by n_reps+1. Mirroring the arithmetic means a delta
    // between the two binaries cannot be blamed on how the clock was read.
    std::vector<double> per_rep(n_reps);
    const double t0 = wall();
    for (int rep = 0; rep < n_reps; ++rep) {
        const double a = wall();
        LAUNCH(cu_sync);
        per_rep[rep] = (wall() - a) * 1000.0;
    }
    LAUNCH(1);
    const double t1 = wall();
    const double ms_mean = (t1 - t0) * 1000.0 / (double)(n_reps + 1);
    const double ms_min  = *std::min_element(per_rep.begin(), per_rep.end());

    CUDA_OK(cudaMemcpy(fh.data(), d_fh, nT * sizeof(double), cudaMemcpyDeviceToHost));

    // ---- sanity -----------------------------------------------------------
    // Independent of the reference file: an all-zero or NaN field would sail
    // through a timing loop and produce a perfectly respectable ms/rep.
    double fmin = fh[0], fmax = fh[0], fsum = 0.0;
    for (size_t i = 0; i < nT; ++i) {
        fmin = std::min(fmin, fh[i]);
        fmax = std::max(fmax, fh[i]);
        fsum += fh[i];
    }
    const double fscale = std::max(std::fabs(fmin), std::fabs(fmax));

    // ---- verify against the Fortran run -----------------------------------
    // fh_ref.bin is written by `./layered_bench <same args> 1` and holds the
    // Fortran driver's INPUTS (h, u, v) as well as both its outputs (do
    // concurrent, and CUDA-via-host_data). Two separate questions get asked,
    // and conflating them is how a clean port gets convicted of a bug:
    //
    //  (1) OWN-INIT pass. Does this driver, initialising itself, reproduce the
    //      Fortran's fluxes? It does NOT come out bit-identical, and it cannot:
    //      the init calls sin/cos/exp, nvfortran resolves those to libpgmath
    //      and nvcc's host pass to glibc's libm, and the two disagree by ~1 ulp.
    //      That is a property of two libm implementations, not of the kernel.
    //  (2) REF-INIT pass. Upload the Fortran's OWN h/u/v and run the same
    //      kernel. This removes the libm difference and leaves exactly one
    //      variable: whether the memory came from cudaMalloc or from OpenACC.
    //      Here bit-identity is both achievable and demanded -- anything else
    //      is a real bug in this driver (shapes, strides, ghost walls, init).
    //
    // (2) is the load-bearing check. (1) is reported because a large diff there
    // would mean the init mirror is wrong, which (2) alone could not detect.
    bool verified = false, verify_ok = false;
    if (FILE *f = std::fopen(REF_FILE, "rb")) {
        int rnx = 0, rny = 0, rnz = 0;
        bool okhdr = std::fread(&rnx, sizeof(int), 1, f) == 1 &&
                     std::fread(&rny, sizeof(int), 1, f) == 1 &&
                     std::fread(&rnz, sizeof(int), 1, f) == 1;
        if (!okhdr || rnx != nx || rny != ny || rnz != nz) {
            std::printf("  verify  : SKIPPED -- %s is %dx%dx%d, this run is %dx%dx%d.\n",
                        REF_FILE, rnx, rny, rnz, nx, ny, nz);
            std::printf("            re-run: ./layered_bench %d %d %d 200 10 2 1\n", nxp, nyp, nz);
        } else {
            std::vector<double> rh(nT), ru(nU), rv(nV), rdc(nT), rcu(nT);
            bool okrd = std::fread(rh.data(),  sizeof(double), nT, f) == nT &&
                        std::fread(ru.data(),  sizeof(double), nU, f) == nU &&
                        std::fread(rv.data(),  sizeof(double), nV, f) == nV &&
                        std::fread(rdc.data(), sizeof(double), nT, f) == nT &&
                        std::fread(rcu.data(), sizeof(double), nT, f) == nT;
            if (!okrd) {
                std::printf("  verify  : SKIPPED -- %s truncated\n", REF_FILE);
            } else {
                std::printf("  VERIFY vs %s (written by ./layered_bench)\n", REF_FILE);

                // (1) own init vs the Fortran's init, and the fluxes it produced.
                Diff dh, du, dv, ddc_own;
                for (size_t i = 0; i < nT; ++i) dh.feed(h[i], rh[i]);
                for (size_t i = 0; i < nU; ++i) du.feed(u[i], ru[i]);
                for (size_t i = 0; i < nV; ++i) dv.feed(v[i], rv[i]);
                for (size_t i = 0; i < nT; ++i) ddc_own.feed(fh[i], rdc[i]);
                std::printf("    (1) OWN INIT -- same problem? (libm differs ~1 ulp: expected)\n");
                report_rel("h_layer",        dh, 1.0e-13);
                report_rel("u_face_x_layer", du, 1.0e-13);
                report_rel("v_face_y_layer", dv, 1.0e-13);
                report_abs("flux_h vs dc",   ddc_own, fscale);

                // (2) the decisive one: identical inputs, so identical outputs.
                CUDA_OK(cudaMemcpy(d_h, rh.data(), nT * sizeof(double), cudaMemcpyHostToDevice));
                CUDA_OK(cudaMemcpy(d_u, ru.data(), nU * sizeof(double), cudaMemcpyHostToDevice));
                CUDA_OK(cudaMemcpy(d_v, rv.data(), nV * sizeof(double), cudaMemcpyHostToDevice));
                LAUNCH(1);
                CUDA_OK(cudaDeviceSynchronize());
                std::vector<double> fh_ref_in(nT);
                CUDA_OK(cudaMemcpy(fh_ref_in.data(), d_fh, nT * sizeof(double),
                                   cudaMemcpyDeviceToHost));
                Diff ddc, dcu;
                for (size_t i = 0; i < nT; ++i) ddc.feed(fh_ref_in[i], rdc[i]);
                for (size_t i = 0; i < nT; ++i) dcu.feed(fh_ref_in[i], rcu[i]);
                std::printf("    (2) FORTRAN'S INIT uploaded -- only the allocator differs\n");
                report_abs("flux_h vs dc",        ddc, fscale);
                report_abs("flux_h vs host_data", dcu, fscale);

                verified = true;
                // The bar for (2) is exact equality; for (1) it is ULP-level,
                // both on the inputs and on fh measured against the field.
                verify_ok = ddc.dmax == 0.0 && dcu.dmax == 0.0 &&
                            dh.rmax < 1.0e-13 && du.rmax < 1.0e-13 && dv.rmax < 1.0e-13 &&
                            (fscale > 0.0 && ddc_own.dmax / fscale < 1.0e-12);
            }
        }
        std::fclose(f);
    } else {
        std::printf("  verify  : SKIPPED -- no %s. Produce one with:\n", REF_FILE);
        std::printf("            ./layered_bench %d %d %d 200 10 2 1\n", nxp, nyp, nz);
    }

    std::printf("%s\n", std::string(70, '-').c_str());
    std::printf("  CUDA native (cudaMalloc, no OpenACC) : %10.4f ms/rep  (mean of %d)\n",
                ms_mean, n_reps + 1);
    std::printf("                                       : %10.4f ms/rep  (min of %d)\n",
                ms_min, n_reps);
    if (cu_sync == 0)
        std::printf("    NOTE: cuda_sync=0 -- per-rep times are enqueue cost, not kernel time.\n"
                    "          Only the mean is meaningful in async mode.\n");
    std::printf("    Compare against ./layered_bench's 'CUDA C (faithful port, nvcc)'\n"
                "    row at the same args: that one is the same kernel reached through\n"
                "    !$acc host_data use_device. The difference is the OpenACC path.\n");
    std::printf("%s\n", std::string(70, '-').c_str());
    std::printf("  min flux_h_layer : %14.6e\n", fmin);
    std::printf("  max flux_h_layer : %14.6e\n", fmax);
    std::printf("  sum flux_h_layer : %14.6e\n", fsum);
    if (fmin != fmin || fmax != fmax)
        std::printf("  sanity           : *** NaN -- results are garbage ***\n");
    else if (fmin == 0.0 && fmax == 0.0)
        std::printf("  sanity           : *** all-zero -- kernel did nothing ***\n");
    else
        std::printf("  sanity           : OK (finite, non-zero)\n");
    if (verified)
        std::printf("  verify           : %s\n", verify_ok ? "PASS" : "*** FAIL ***");
    std::printf("%s\n", std::string(70, '=').c_str());

    cudaFree(d_h); cudaFree(d_u); cudaFree(d_v); cudaFree(d_wet);
    cudaFree(d_iarea); cudaFree(d_dycu); cudaFree(d_dxcv);
    cudaFree(d_hflx); cudaFree(d_hfrx); cudaFree(d_hfly); cudaFree(d_hfry);
    cudaFree(d_mfx); cudaFree(d_mfy); cudaFree(d_fh);
    return (verified && !verify_ok) ? 1 : 0;
}
