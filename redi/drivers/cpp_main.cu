// ===========================================================================
// NATIVE C++/CUDA driver for the Redi benchmark. No Fortran, no OpenACC.
// cudaMalloc + cudaMemcpy + main().
//
// PURPOSE: every CUDA port in this repo is launched FROM FORTRAN and receives
// its device pointers through `!$acc host_data use_device(...)`, which does a
// runtime present-table lookup per array per call -- 24 arrays, 8 launches, on
// every Redi step. So the OpenACC runtime sits on the CUDA path in every
// "CUDA vs do concurrent" number the repo quotes. If that lookup costs
// anything, CUDA has been handicapped and `dc/cuda = 0.99` is not a tie.
//
// This driver removes OpenACC entirely: raw cudaMalloc'd pointers go straight
// into the same launch bridge. The delta against redi_bench's CUDA variant IS
// the cost of the host_data path.
//
// ⚠ THE KERNELS ARE NOT REDEFINED HERE (daxpy_pure.cu's rule). This calls
// redi_cuda_launch_() from redi_kernel.cu -- the IDENTICAL entry point
// redi_bench.F90's CUDA variant calls through redi_cuda.F90. Nothing below is
// a __global__. If the kernels were duplicated, a divergence between the two
// copies would silently corrupt the comparison, and the comparison is the
// entire point of the file.
//
// The init below mirrors redi_bench.F90:build_state + the metrics/VarMix
// allocations EXACTLY -- verified, not asserted: `make native-verify` runs
// ./redi_bench ... dump=1 and this binary diffs both the initial condition and
// the final stepped state against the Fortran run's own arrays. See --verify.
//
// Usage: ./redi_native [nx_phys] [ny_phys] [nz] [nreps] [nwarm]
//   defaults 473 297 30 20 10 -- the 0.1 deg gabight config, matching
//   redi_bench's defaults so the two numbers are directly comparable.
// ===========================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include "gpu_rt.h"

// The SAME bridge redi_cuda.F90 calls. Fortran bind(C) convention: every
// argument by reference, hence the pointers on the scalars.
extern "C" void redi_cuda_launch_(
    const int *nx_, const int *ny_, const int *nz_, const int *ns_, const double *dt_,
    const int *nghost_, const int *nxp_, const int *nyp_,
    const int *ww_, const int *we_, const int *wsz_, const int *wn_,
    const double *rho0_, const double *T_ref_, const double *S_ref_,
    const double *alpha_T_, const double *beta_S_,
    const double *h_layer, double *t_htr, double *s_htr,
    const double *wet_u, const double *wet_v,
    double *uPoL, double *uPoR, int *uKoL, int *uKoR, double *uhEff,
    double *vPoL, double *vPoR, int *vKoL, int *vKoR, double *vhEff,
    const double *khtr_u_ext, const double *khtr_v_ext, double *khtr_u, double *khtr_v,
    const double *dy_cu, const double *dx_cv, const double *idxCu, const double *idyCv,
    const double *areaT, double *tr_snap);

#define CUDA_OK(call)                                                          \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(_e), __FILE__, __LINE__);          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ---- EOS defaults, verbatim from ocean_eos.F90's ocean_eos_t ---------------
// The driver never overrides them, so these ARE the values redi_bench uses.
// variant stays EOS_VARIANT_LINEAR (no &eos_nml in any Redi-enabled config).
static const double EOS_RHO0    = 1035.0;
static const double EOS_T_REF   = 10.0;
static const double EOS_S_REF   = 35.0;
static const double EOS_ALPHA_T = 1.7e-4;
static const double EOS_BETA_S  = 7.6e-4;

// ---- metrics, verbatim from redi_bench.F90's allocate(...source=) ----------
static const double MET_DY_CU  = 11100.0;
static const double MET_DX_CV  = 8000.0;
static const double MET_IDXCU  = 1.0 / 8000.0;
static const double MET_IDYCV  = 1.0 / 11100.0;
static const double MET_AREAT  = 8000.0 * 11100.0;
static const double KHTR_EXT   = 100.0;   // VarMix per-face KhTr (MEKE khcoeff=1.0)
static const double DT         = 900.0;

static double *dalloc(size_t n) {
    double *p = nullptr;
    CUDA_OK(cudaMalloc(&p, n * sizeof(double)));
    return p;
}
static int *ialloc(size_t n) {
    int *p = nullptr;
    CUDA_OK(cudaMalloc(&p, n * sizeof(int)));
    return p;
}
// cudaMemset writes BYTES; 0x00 bytes are a valid 0.0 double and a valid int 0,
// but `source=1` on the Ko arrays is NOT memset-able -- those go up from host.
static void up(double *d, const std::vector<double> &h) {
    CUDA_OK(cudaMemcpy(d, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice));
}
static void upi(int *d, const std::vector<int> &h) {
    CUDA_OK(cudaMemcpy(d, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));
}

// redi_bench.F90:build_state, transcribed. Fortran is 1-based column-major;
// the loop bounds and the index arithmetic below keep that convention so the
// two can be read against each other. Native k indexing is BOTTOM-UP:
// k=1 bed, k=nz surface.
static void build_state(int nx, int ny, int nz, std::vector<double> &h_layer,
                        std::vector<double> &hTr_T, std::vector<double> &hTr_S) {
    const double depth = 4000.0;
    const double hh = depth / (double)nz;
    for (int kk = 1; kk <= nz; ++kk) {
        // zz = depth below surface at layer centre; k=nz is the surface layer.
        const double zz = ((double)(nz - kk) + 0.5) * hh;
        for (int jj = 1; jj <= ny; ++jj) {
            const double fy = (double)(jj - 1) / (double)(ny - 1);
            for (int ii = 1; ii <= nx; ++ii) {
                const double fx = (double)(ii - 1) / (double)(nx - 1);
                // Exponential thermocline + a diagonal surface front (~6 degC
                // across the basin) -- genuinely tilted neutral surfaces.
                const double tt = 2.0 + 16.0 * std::exp(-zz / 800.0) + 3.0 * fx + 3.0 * fy;
                const double ss = 34.5 + 0.7 * (1.0 - std::exp(-zz / 1500.0)) - 0.2 * fx;
                const size_t idx = (size_t)(ii - 1) + (size_t)nx * ((size_t)(jj - 1) + (size_t)ny * (size_t)(kk - 1));
                h_layer[idx] = hh;
                hTr_T[idx] = tt * hh;
                hTr_S[idx] = ss * hh;
            }
        }
    }
}

struct Cmp { double max_abs; double rel; };
static Cmp compare(const std::vector<double> &a, const std::vector<double> &b) {
    double mx = 0.0, scl = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(a[i] - b[i]);
        if (d > mx) mx = d;
        if (std::fabs(a[i]) > scl) scl = std::fabs(a[i]);
    }
    return Cmp{mx, mx / std::max(scl, 1.0e-30)};
}

// Reads a redi_bench.F90:dump_state stream dump. Returns false if absent.
static bool read_ref(const char *fname, int nx, int ny, int nz,
                     std::vector<double> &h_layer, std::vector<double> &hTr_T,
                     std::vector<double> &hTr_S) {
    std::FILE *f = std::fopen(fname, "rb");
    if (!f) return false;
    int hdr[3];
    if (std::fread(hdr, sizeof(int), 3, f) != 3) { std::fclose(f); return false; }
    if (hdr[0] != nx || hdr[1] != ny || hdr[2] != nz) {
        std::fprintf(stderr, "%s: shape %dx%dx%d != this run's %dx%dx%d -- "
                             "re-dump at the same size\n",
                     fname, hdr[0], hdr[1], hdr[2], nx, ny, nz);
        std::fclose(f);
        std::exit(1);
    }
    const size_t n = (size_t)nx * ny * nz;
    h_layer.resize(n); hTr_T.resize(n); hTr_S.resize(n);
    bool ok = std::fread(h_layer.data(), sizeof(double), n, f) == n &&
              std::fread(hTr_T.data(), sizeof(double), n, f) == n &&
              std::fread(hTr_S.data(), sizeof(double), n, f) == n;
    std::fclose(f);
    return ok;
}

int main(int argc, char **argv) {
    int nxp = 473, nyp = 297, nz = 30, nreps = 20, nwarm = 10;
    const int nghost = 3;
    if (argc >= 2) nxp = std::atoi(argv[1]);
    if (argc >= 3) nyp = std::atoi(argv[2]);
    if (argc >= 4) nz = std::atoi(argv[3]);
    if (argc >= 5) nreps = std::atoi(argv[4]);
    if (argc >= 6) nwarm = std::atoi(argv[5]);

    const int nx = nxp + 2 * nghost;
    const int ny = nyp + 2 * nghost;
    const int ns = 2 * nz + 2;          // ocean_redi_init: nsurf = 2*nz+2

    std::printf("================================================================\n");
    std::printf(" Redi neutral diffusion: NATIVE C++/CUDA driver (no Fortran, no OpenACC)\n");
    std::printf("================================================================\n");
    std::printf(" grid   : %d x %d phys (+%d ghost) x %d layers\n", nxp, nyp, nghost, nz);
    std::printf(" arrays : nx_total=%d ny_total=%d nsurf=%d\n", nx, ny, ns);
    std::printf(" cells  : %lld\n", (long long)nx * ny * nz);
    std::printf(" reps   : %d   warmup: %d\n", nreps, nwarm);
    std::printf(" launch : redi_cuda_launch_() from redi_kernel.cu (the SAME entry\n");
    std::printf("          point redi_bench's CUDA variant calls -- not a copy)\n\n");

    const size_t nT  = (size_t)nx * ny * nz;              // T-cell 3D
    const size_t nU  = (size_t)(nx + 1) * ny;             // u-face 2D
    const size_t nV  = (size_t)nx * (ny + 1);             // v-face 2D
    const size_t nUs = (size_t)(nx + 1) * ny * ns;        // u-face x nsurf
    const size_t nVs = (size_t)nx * (ny + 1) * ns;        // v-face x nsurf
    const size_t nUe = (size_t)(nx + 1) * ny * (ns - 1);  // u-face x nsurf-1
    const size_t nVe = (size_t)nx * (ny + 1) * (ns - 1);  // v-face x nsurf-1
    const size_t nA  = (size_t)nx * ny;                   // T-cell 2D

    // ---- host init: mirrors redi_bench.F90 exactly -------------------------
    std::vector<double> h_layer(nT), hTr_T(nT), hTr_S(nT);
    build_state(nx, ny, nz, h_layer, hTr_T, hTr_S);

    // ---- INIT cross-check, then ADOPT the Fortran IC if it is available ----
    //
    // MEASURED, and it killed two of my own hypotheses. The natively-computed
    // IC agrees with redi_bench's to 1-2 ulp (~3e-16 relative), NOT bit-for-bit.
    // Cause, tested rather than assumed:
    //   * NOT exp(). nvfortran's host libm and glibc agree bit-for-bit on every
    //     exp(-zz/800) / exp(-zz/1500) in this init -- probed directly.
    //   * NOT FMA contraction. gcc -ffp-contract=off gives the identical answer.
    //   * It is nvfortran -fast vs gcc -O3 differing by one ulp on
    //     `2 + 16*e + 3*fx + 3*fy`, and only where fx and fy are both
    //     non-trivial -- i.e. a host-compiler evaluation difference in the
    //     HARNESS's own setup arithmetic. It is not a property of the kernel,
    //     the port, or the host_data path, none of which are involved yet.
    //
    // A 1-ulp IC would make the final states differ at ~1e-16 too, and then
    // "does the native driver reproduce the Fortran-driven CUDA run?" could not
    // be answered with a bit-exact test -- the interesting signal (a shape or
    // constant bug in THIS file) and the uninteresting one (host libm noise)
    // would be the same size. So when the reference IC is present it is ADOPTED
    // verbatim: both drivers then start from literally the same bits and any
    // final difference is attributable to this driver alone. Standalone runs
    // (no dump present) use the natively-computed IC and are self-sufficient,
    // which is what the nvbug_*/ reports need.
    std::vector<double> rh, rT, rS;
    const bool have_init_ref = read_ref("redi_ref_init.bin", nx, ny, nz, rh, rT, rS);
    if (have_init_ref) {
        const Cmp ch = compare(rh, h_layer), ct = compare(rT, hTr_T), cs = compare(rS, hTr_S);
        std::printf(" INIT: natively-computed vs redi_bench's (redi_ref_init.bin)\n");
        std::printf("   h_layer : max|d| = %11.4e   rel = %11.4e\n", ch.max_abs, ch.rel);
        std::printf("   hTr(T)  : max|d| = %11.4e   rel = %11.4e\n", ct.max_abs, ct.rel);
        std::printf("   hTr(S)  : max|d| = %11.4e   rel = %11.4e\n", cs.max_abs, cs.rel);
        if (ch.max_abs == 0.0 && ct.max_abs == 0.0 && cs.max_abs == 0.0) {
            std::printf("   -> transcription is BIT-IDENTICAL.\n");
        } else {
            std::printf("   -> agrees to %.1e relative (1-2 ulp: nvfortran -fast vs gcc -O3\n",
                        std::max(ct.rel, cs.rel));
            std::printf("      on the harness's own setup arithmetic; not exp, not FMA).\n");
        }
        std::printf("   ADOPTING the Fortran IC so the final check below isolates\n");
        std::printf("   this driver from host-compiler FP noise.\n\n");
        h_layer = rh; hTr_T = rT; hTr_S = rS;
    } else {
        std::printf(" INIT check SKIPPED: redi_ref_init.bin not found -- using the\n");
        std::printf(" natively-computed IC (standalone mode; no Fortran required).\n");
        std::printf(" For the bit-exact cross-check run `make native-verify`.\n\n");
    }

    // ---- device allocation: every array redi_cuda_launch_ touches ----------
    double *d_h_layer = dalloc(nT), *d_t_htr = dalloc(nT), *d_s_htr = dalloc(nT);
    double *d_tr_snap = dalloc(nT);
    double *d_wet_u = dalloc(nU), *d_wet_v = dalloc(nV);
    double *d_uPoL = dalloc(nUs), *d_uPoR = dalloc(nUs), *d_uhEff = dalloc(nUe);
    int    *d_uKoL = ialloc(nUs), *d_uKoR = ialloc(nUs);
    double *d_vPoL = dalloc(nVs), *d_vPoR = dalloc(nVs), *d_vhEff = dalloc(nVe);
    int    *d_vKoL = ialloc(nVs), *d_vKoR = ialloc(nVs);
    double *d_khtr_u_ext = dalloc(nU), *d_khtr_v_ext = dalloc(nV);
    double *d_khtr_u = dalloc(nU), *d_khtr_v = dalloc(nV);
    double *d_dy_cu = dalloc(nU), *d_idxCu = dalloc(nU);
    double *d_dx_cv = dalloc(nV), *d_idyCv = dalloc(nV);
    double *d_areaT = dalloc(nA);

    // ---- H2D: the same values redi_bench's enter data copyin ships ---------
    up(d_h_layer, h_layer); up(d_t_htr, hTr_T); up(d_s_htr, hTr_S);
    up(d_wet_u, std::vector<double>(nU, 1.0));           // ALL WET
    up(d_wet_v, std::vector<double>(nV, 1.0));
    up(d_khtr_u_ext, std::vector<double>(nU, KHTR_EXT));
    up(d_khtr_v_ext, std::vector<double>(nV, KHTR_EXT));
    up(d_dy_cu, std::vector<double>(nU, MET_DY_CU));
    up(d_idxCu, std::vector<double>(nU, MET_IDXCU));
    up(d_dx_cv, std::vector<double>(nV, MET_DX_CV));
    up(d_idyCv, std::vector<double>(nV, MET_IDYCV));
    up(d_areaT, std::vector<double>(nA, MET_AREAT));
    // ocean_redi_init's `source=` values. Phase A overwrites all of these on
    // every call, so they only matter for a first-call read of uninitialised
    // memory -- but redi_bench sets them, so this sets them.
    up(d_uPoL, std::vector<double>(nUs, 0.0)); up(d_uPoR, std::vector<double>(nUs, 0.0));
    up(d_vPoL, std::vector<double>(nVs, 0.0)); up(d_vPoR, std::vector<double>(nVs, 0.0));
    up(d_uhEff, std::vector<double>(nUe, 0.0)); up(d_vhEff, std::vector<double>(nVe, 0.0));
    upi(d_uKoL, std::vector<int>(nUs, 1)); upi(d_uKoR, std::vector<int>(nUs, 1));
    upi(d_vKoL, std::vector<int>(nVs, 1)); upi(d_vKoR, std::vector<int>(nVs, 1));
    up(d_khtr_u, std::vector<double>(nU, 0.0)); up(d_khtr_v, std::vector<double>(nV, 0.0));
    up(d_tr_snap, std::vector<double>(nT, 0.0));

    // ---- scalars: Fortran passes by reference, so these need addresses -----
    const int i_nx = nx, i_ny = ny, i_nz = nz, i_ns = ns;
    const int i_ng = nghost, i_nxp = nxp, i_nyp = nyp;
    const int i_ww = 1, i_we = 1, i_wsz = 1, i_wn = 1;   // all four edges OBC_WALL
    const double d_dt = DT, d_rho0 = EOS_RHO0, d_Tref = EOS_T_REF, d_Sref = EOS_S_REF;
    const double d_alpha = EOS_ALPHA_T, d_beta = EOS_BETA_S;

    auto step = [&]() {
        // redi_cuda_launch_ ends in cudaDeviceSynchronize(), exactly as it does
        // when Fortran calls it -- so a rep boundary is a full device sync in
        // both drivers and the two timings measure the same thing.
        redi_cuda_launch_(&i_nx, &i_ny, &i_nz, &i_ns, &d_dt, &i_ng, &i_nxp, &i_nyp,
                          &i_ww, &i_we, &i_wsz, &i_wn,
                          &d_rho0, &d_Tref, &d_Sref, &d_alpha, &d_beta,
                          d_h_layer, d_t_htr, d_s_htr, d_wet_u, d_wet_v,
                          d_uPoL, d_uPoR, d_uKoL, d_uKoR, d_uhEff,
                          d_vPoL, d_vPoR, d_vKoL, d_vKoR, d_vhEff,
                          d_khtr_u_ext, d_khtr_v_ext, d_khtr_u, d_khtr_v,
                          d_dy_cu, d_dx_cv, d_idxCu, d_idyCv, d_areaT, d_tr_snap);
        CUDA_OK(cudaGetLastError());
    };

    // ---- warm-up (untimed) -------------------------------------------------
    // hTr ACCUMULATES: redi_bench does not reset state between warm-up and
    // timed reps, so the reference final state is after nwarm+nreps steps and
    // this driver must take exactly the same number.
    for (int r = 0; r < nwarm; ++r) step();

    // ---- timed -------------------------------------------------------------
    cudaEvent_t beg, end;
    CUDA_OK(cudaEventCreate(&beg)); CUDA_OK(cudaEventCreate(&end));
    std::vector<float> per_rep(nreps);
    for (int r = 0; r < nreps; ++r) {
        CUDA_OK(cudaEventRecord(beg));
        step();
        CUDA_OK(cudaEventRecord(end));
        CUDA_OK(cudaEventSynchronize(end));
        CUDA_OK(cudaEventElapsedTime(&per_rep[r], beg, end));
    }
    double total = 0.0;
    for (float v : per_rep) total += v;
    const double mean_ms = total / nreps;
    const double min_ms = *std::min_element(per_rep.begin(), per_rep.end());
    const double max_ms = *std::max_element(per_rep.begin(), per_rep.end());

    // ---- D2H + verify final state against the Fortran run ------------------
    std::vector<double> out_T(nT), out_S(nT);
    CUDA_OK(cudaMemcpy(out_T.data(), d_t_htr, nT * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(out_S.data(), d_s_htr, nT * sizeof(double), cudaMemcpyDeviceToHost));

    std::printf(" NATIVE timing (min-of-%d is the headline; the V100 only ever\n", nreps);
    std::printf(" drifts SLOWER, so min is the right estimator)\n");
    std::printf("   min  : %10.4f ms/call\n", min_ms);
    std::printf("   mean : %10.4f ms/call   <- compare to redi_bench, which reports a mean\n", mean_ms);
    std::printf("   max  : %10.4f ms/call\n", max_ms);
    std::printf("   spread (max/min) : %6.3f x %s\n\n", max_ms / min_ms,
                (max_ms / min_ms > 1.10) ? "  <- >10%: contended or drifting" : "");

    int rc = 0;
    std::vector<double> fh, fT, fS;
    if (read_ref("redi_ref_final.bin", nx, ny, nz, fh, fT, fS)) {
        const Cmp ct = compare(fT, out_T), cs = compare(fS, out_S);
        std::printf(" FINAL vs Fortran-driven CUDA (redi_ref_final.bin), after %d steps:\n",
                    nwarm + nreps);
        std::printf("   hTr(T) : max|d| = %11.4e   rel = %11.4e\n", ct.max_abs, ct.rel);
        std::printf("   hTr(S) : max|d| = %11.4e   rel = %11.4e\n", cs.max_abs, cs.rel);
        const bool exact = ct.max_abs == 0.0 && cs.max_abs == 0.0;
        if (have_init_ref) {
            // IC adopted => same kernel, same bits in, same rep count, same
            // device. The ONLY legitimate outcome is bit-identical: there is no
            // second toolchain left to contract an expression differently.
            // Anything else is a bug in THIS file (shape, constant, rep count).
            std::printf("   -> %s\n", exact
                ? "PASS: BIT-IDENTICAL to the Fortran-driven CUDA run.\n"
                  "         Same kernel + same IC bits + same rep count. The native\n"
                  "         driver is a faithful substitute for the host_data path,\n"
                  "         so the timings below are comparable like-for-like."
                : "*** FAIL: differs despite an identical IC. Suspect a shape,\n"
                  "         constant or rep-count bug in redi_native.cu.");
            if (!exact) rc = 1;
        } else {
            // IC not adopted: a ~1e-16 difference here is the IC's 1-2 ulp
            // carried through, not a port bug. Bar is the harness's own: 1e-12.
            std::printf("   -> %s\n", (ct.rel < 1.0e-12 && cs.rel < 1.0e-12)
                ? "PASS (loose bar): agrees to <1e-12 relative. Exact test needs\n"
                  "         the shared IC -- run `make native-verify`."
                : "*** FAIL: exceeds the 1e-12 port-bug bar.");
            if (!(ct.rel < 1.0e-12 && cs.rel < 1.0e-12)) rc = 1;
        }
    } else {
        std::printf(" FINAL check SKIPPED: redi_ref_final.bin not found.\n");
        std::printf(" Run `make native-verify` to generate it and compare.\n");
    }

    cudaFree(d_h_layer); cudaFree(d_t_htr); cudaFree(d_s_htr); cudaFree(d_tr_snap);
    cudaFree(d_wet_u); cudaFree(d_wet_v);
    cudaFree(d_uPoL); cudaFree(d_uPoR); cudaFree(d_uKoL); cudaFree(d_uKoR); cudaFree(d_uhEff);
    cudaFree(d_vPoL); cudaFree(d_vPoR); cudaFree(d_vKoL); cudaFree(d_vKoR); cudaFree(d_vhEff);
    cudaFree(d_khtr_u_ext); cudaFree(d_khtr_v_ext); cudaFree(d_khtr_u); cudaFree(d_khtr_v);
    cudaFree(d_dy_cu); cudaFree(d_idxCu); cudaFree(d_dx_cv); cudaFree(d_idyCv); cudaFree(d_areaT);
    return rc;
}
