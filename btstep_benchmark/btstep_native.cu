// NATIVE C++/CUDA driver for the barotropic substep. No Fortran, no OpenACC.
// cudaMalloc + cudaMemcpy + main().
//
// THE QUESTION: every CUDA port in this repo is launched FROM Fortran and gets
// its device pointers through `!$acc host_data use_device(...)`, so the OpenACC
// runtime sits on the CUDA path. If that costs anything, every "CUDA buys N%"
// number in this repo understates CUDA. This driver removes OpenACC entirely.
//
// PREDICTION FOR THIS BENCHMARK, stated up front so it can be falsified:
// btstep should TIE. Unlike the other benchmarks, btstep_bench.F90 already
// hoists `host_data` OUT of the timing loop -- `fill_args` is called ONCE and
// fills a by-value struct of device pointers, after which the timed loop is a
// bare `call btstep_cuda_launch(a, n_inner, mode)` with no present-table lookup
// in sight. So there should be no per-call OpenACC overhead left to remove. A
// tie confirms that reasoning; a gap falsifies it and is worth chasing.
//
// THE CARDINAL RULE -- the kernels are NOT redefined here. This calls the same
// btstep_cuda_launch() out of btstep_kernel.cu that btstep_bench.F90 calls, and
// shares `struct BtArgs` with it through btstep_args.h. If the kernels were
// duplicated, a divergence between the two copies would silently corrupt the
// comparison -- and the comparison is the entire point of the file.
//
// Usage: ./btstep_native [nx_phys] [ny_phys] [n_inner] [nreps] [nwarm]
//   Defaults 473 297 24 200 10 (the 0.1 deg config).
//   To reproduce the Fortran driver's state bit-for-bit, pass IDENTICAL args to
//   both binaries: the final eta depends on (nwarm + nreps) * n_inner substeps.
//
// Verification against the Fortran run:
//   BTSTEP_REF_OUT=cuda_ref.bin ./btstep_bench   473 297 24 200 10   # writes
//   BTSTEP_REF_IN=cuda_ref.bin  ./btstep_native  473 297 24 200 10   # checks
// Both sides run mode 2 (graph) last, so the dumped state is the graph result.
//
// WHY THE REF FILE CARRIES INPUTS AS WELL AS OUTPUTS -- this cost an hour, so
// it is written down. Recomputing the init here and comparing final eta gives
// 3.1e-15 on |eta| ~ 0.265, NOT bit-exact. The cause is not the port: nvfortran's
// libm and glibc's libm round exp() and sin() differently by 1 ulp (measured:
// 29301/145137 of the Gaussian bump, 46/303 of force_u), and over
// (nwarm+nreps)*n_inner = 4920 substeps a 1-ulp seed grows to ~1e-15. That is a
// libm difference, not a CUDA difference, and it masks the thing under test.
// So when BTSTEP_REF_IN is set the driver ADOPTS the Fortran's eta_ini/force_u/
// f_corner (reporting how far its own recomputation was) and then demands
// BIT-EXACT agreement on the final state. Both facts get reported:
//   * own init      -> 3.1e-15, i.e. the repo's usual FMA-level bar
//   * adopted init  -> 0.0, proving the two drivers run identical kernels on
//                      identical numbers, so the ONLY variable left is OpenACC.
// Without BTSTEP_REF_IN the driver is fully standalone and computes its own init.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>

#include "btstep_args.h"

// ---- must match btstep_bench.F90's parameters exactly --------------------
static const int    NXP_DEF = 473, NYP_DEF = 297, NINNER_DEF = 24;
static const int    REPS_DEF = 200, WARM_DEF = 10, NGH = 3;
static const double DXM = 10000.0, DYM = 10000.0, DT_INNER = 12.0;
static const double G_BT = 9.81;   // bt_work_t%g_bt default (bt_state.F90:42)
static const double BEBT = 0.2;    // init_state: the gabight configs' value

#define CUDA_OK(call)                                                          \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(_e), __FILE__, __LINE__);          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Column-major, 1-based, as the Fortran allocations are.
#define AT(i, j, ld) ((size_t)((i)-1) + (size_t)(ld) * (size_t)((j)-1))

static double maxdiff(const std::vector<double> &x, const std::vector<double> &y) {
    double m = 0.0;
    for (size_t k = 0; k < x.size(); ++k) m = std::max(m, std::fabs(x[k] - y[k]));
    return m;
}

static double wall_s() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int iarg(int argc, char **argv, int k, int dflt) {
    if (argc <= k) return dflt;
    char *end = nullptr;
    long v = std::strtol(argv[k], &end, 10);
    if (end == argv[k] || v <= 0) return dflt;
    return (int)v;
}

// One device array + its host mirror, so re-seeding is one call.
struct Arr {
    std::vector<double> h;
    double *d = nullptr;
    void alloc(size_t n) {
        h.assign(n, 0.0);
        CUDA_OK(cudaMalloc(&d, n * sizeof(double)));
    }
    void up()   { CUDA_OK(cudaMemcpy(d, h.data(), h.size()*sizeof(double), cudaMemcpyHostToDevice)); }
    void down() { CUDA_OK(cudaMemcpy(h.data(), d, h.size()*sizeof(double), cudaMemcpyDeviceToHost)); }
    void fill(double v) { std::fill(h.begin(), h.end(), v); }
    ~Arr() { if (d) cudaFree(d); }
};

int main(int argc, char **argv) {
    const int nxp     = iarg(argc, argv, 1, NXP_DEF);
    const int nyp     = iarg(argc, argv, 2, NYP_DEF);
    const int n_inner = iarg(argc, argv, 3, NINNER_DEF);
    const int n_reps  = iarg(argc, argv, 4, REPS_DEF);
    const int n_warm  = iarg(argc, argv, 5, WARM_DEF);
    const int nx = nxp + 2*NGH, ny = nyp + 2*NGH;

    // ---- allocate: same four stride classes as the Fortran driver ---------
    const size_t nT = (size_t)nx*ny, nU = (size_t)(nx+1)*ny;
    const size_t nV = (size_t)nx*(ny+1), nB = (size_t)(nx+1)*(ny+1);

    // ---- the Fortran reference, read BEFORE the run so its inputs can be
    // adopted (see the header note on libm). Absent -> fully standalone.
    struct Ref {
        bool have = false;
        std::vector<double> eta0, force_u, f_corner, eta_f, ubt_f;
    } ref;
    if (const char *rp = std::getenv("BTSTEP_REF_IN")) {
        std::FILE *f = std::fopen(rp, "rb");
        if (!f) { std::fprintf(stderr, "cannot open BTSTEP_REF_IN '%s'\n", rp); return 1; }
        int hdr[2] = {0, 0};
        if (std::fread(hdr, sizeof(int), 2, f) != 2) {
            std::fprintf(stderr, "short ref header\n"); return 1;
        }
        if (hdr[0] != nx || hdr[1] != ny) {
            std::fprintf(stderr, "ref is %dx%d, this run is %dx%d -- pass identical args\n",
                         hdr[0], hdr[1], nx, ny);
            return 1;
        }
        auto rd = [&](std::vector<double> &v, size_t n) {
            v.resize(n);
            if (std::fread(v.data(), sizeof(double), n, f) != n) {
                std::fprintf(stderr, "short ref payload\n"); std::exit(1);
            }
        };
        rd(ref.eta0, nT); rd(ref.force_u, nU); rd(ref.f_corner, nB);   // inputs
        rd(ref.eta_f, nT); rd(ref.ubt_f, nU);                          // outputs
        std::fclose(f);
        ref.have = true;
    }

    Arr eta, eta_new, H_ref, ke, eta_sum, iareaT;                 // (nx,   ny)
    Arr ubt, ubt_prev, rem_u, ubt_sum, uhbt_sum;                  // (nx+1, ny)
    Arr dy_cu, areaCu, dxCu, idxCu, force_u;                      //   "
    Arr vbt, vbt_prev, rem_v, vbt_sum, vhbt_sum;                  // (nx,   ny+1)
    Arr dx_cv, areaCv, dyCv, idyCv, force_v;                      //   "
    Arr zeta, f_corner, iareaBu;                                  // (nx+1, ny+1)

    for (Arr *p : {&eta, &eta_new, &H_ref, &ke, &eta_sum, &iareaT}) p->alloc(nT);
    for (Arr *p : {&ubt, &ubt_prev, &rem_u, &ubt_sum, &uhbt_sum,
                   &dy_cu, &areaCu, &dxCu, &idxCu, &force_u})      p->alloc(nU);
    for (Arr *p : {&vbt, &vbt_prev, &rem_v, &vbt_sum, &vhbt_sum,
                   &dx_cv, &areaCv, &dyCv, &idyCv, &force_v})      p->alloc(nV);
    for (Arr *p : {&zeta, &f_corner, &iareaBu})                    p->alloc(nB);

    // ---- metrics: uniform-Cartesian, btstep_bench.F90:90-95 ---------------
    dy_cu.fill(DYM);            dx_cv.fill(DXM);
    iareaT.fill(1.0/(DXM*DYM));
    areaCu.fill(DXM*DYM);       areaCv.fill(DXM*DYM);
    dxCu.fill(DXM);             dyCv.fill(DYM);
    idxCu.fill(1.0/DXM);        idyCv.fill(1.0/DYM);
    iareaBu.fill(1.0/(DXM*DYM));

    // beta-plane f (btstep_bench.F90:97-101). `ny/2` is INTEGER division in the
    // Fortran, so it must be integer division here too.
    for (int j = 1; j <= ny+1; ++j)
        for (int i = 1; i <= nx+1; ++i)
            f_corner.h[AT(i,j,nx+1)] = -1.0e-4 + 2.0e-11*(double)(j - ny/2)*DYM;

    // wind-like forcing (btstep_bench.F90:102-107). force_v stays 0.
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx+1; ++i)
            force_u.h[AT(i,j,nx+1)] = 1.0e-6*std::sin(3.14159*(double)j/(double)ny);

    // ---- init_state's eta (btstep_bench.F90:279-299) ---------------------
    // Mirrored exactly, including the integer divisions in the Gaussian.
    std::vector<double> eta_ini(nT);
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i) {
            // a Gaussian SSH bump -> a geostrophic adjustment problem
            const double a = (double)(i - nx/2) / (double)std::max(nx/8, 1);
            const double b = (double)(j - ny/2) / (double)std::max(ny/8, 1);
            eta_ini[AT(i,j,nx)] = 0.5*std::exp(-(a*a + b*b));
        }

    // ---- adopt the Fortran's transcendental inputs, if given -------------
    // Reported, not hidden: this is exactly the libm gap, and it is the reason
    // a recomputed init cannot bit-match. Everything else is exact in both.
    double d_eta0 = 0.0, d_fu = 0.0, d_fc = 0.0;
    if (ref.have) {
        d_eta0 = maxdiff(eta_ini, ref.eta0);
        d_fu   = maxdiff(force_u.h, ref.force_u);
        d_fc   = maxdiff(f_corner.h, ref.f_corner);
        eta_ini    = ref.eta0;      // exp()
        force_u.h  = ref.force_u;   // sin()
        f_corner.h = ref.f_corner;  // possible FMA contraction
    }

    auto init_state = [&]() {
        H_ref.fill(4000.0);
        eta.h = eta_ini;
        eta_new.fill(0.0);
        ubt.fill(0.0);      vbt.fill(0.0);
        ubt_prev.fill(0.0); vbt_prev.fill(0.0);
        rem_u.fill(1.0);    rem_v.fill(1.0);   // drag off -> 1.0, as production init
        zeta.fill(0.0);     ke.fill(0.0);
        ubt_sum.fill(0.0);  vbt_sum.fill(0.0); eta_sum.fill(0.0);
        uhbt_sum.fill(0.0); vhbt_sum.fill(0.0);
    };

    // ---- upload -----------------------------------------------------------
    // The metrics never change; the state is re-uploaded before each mode, the
    // same way btstep_bench.F90 re-seeds wc between its three CUDA variants.
    for (Arr *p : {&dy_cu, &dx_cv, &iareaT, &areaCu, &areaCv, &dxCu, &dyCv,
                   &idxCu, &idyCv, &iareaBu, &f_corner, &force_u, &force_v})
        p->up();

    auto seed_device = [&]() {
        init_state();
        for (Arr *p : {&eta, &eta_new, &H_ref, &ke, &eta_sum,
                       &ubt, &ubt_prev, &rem_u, &ubt_sum, &uhbt_sum,
                       &vbt, &vbt_prev, &rem_v, &vbt_sum, &vhbt_sum, &zeta})
            p->up();
    };

    // ---- the struct: cudaMalloc'ed pointers, no host_data anywhere --------
    BtArgs args{};
    args.eta = eta.d; args.eta_new = eta_new.d; args.H_ref = H_ref.d;
    args.ubt = ubt.d; args.vbt = vbt.d;
    args.ubt_prev = ubt_prev.d; args.vbt_prev = vbt_prev.d;
    args.rem_u = rem_u.d; args.rem_v = rem_v.d;
    args.zeta = zeta.d; args.ke = ke.d;
    args.ubt_sum = ubt_sum.d; args.vbt_sum = vbt_sum.d; args.eta_sum = eta_sum.d;
    args.uhbt_sum = uhbt_sum.d; args.vhbt_sum = vhbt_sum.d;
    args.dy_cu = dy_cu.d; args.dx_cv = dx_cv.d; args.iareaT = iareaT.d;
    args.areaCu = areaCu.d; args.areaCv = areaCv.d;
    args.dxCu = dxCu.d; args.dyCv = dyCv.d;
    args.idxCu = idxCu.d; args.idyCv = idyCv.d; args.iareaBu = iareaBu.d;
    args.f_corner = f_corner.d; args.force_u = force_u.d; args.force_v = force_v.d;
    args.nx = nx; args.ny = ny; args.nghost = NGH;
    args.nx_phys = nxp; args.ny_phys = nyp;
    args.G = G_BT; args.bebt = BEBT; args.dt = DT_INNER;

    std::printf("%s\n", std::string(72, '=').c_str());
    std::printf("  driver : NATIVE C++/CUDA (cudaMalloc + main, no Fortran, no OpenACC)\n");
    std::printf("  domain : %d x %d interior (+%d ghosts)\n", nxp, nyp, NGH);
    std::printf("  arrays : %d x %d = %d\n", nx, ny, nx*ny);
    std::printf("  n_inner: %d substeps;  %d timed reps, %d warm-up\n", n_inner, n_reps, n_warm);
    std::printf("  bebt = %.2f;  dt_inner = %.1f s\n", BEBT, DT_INNER);
    std::printf("%s\n", std::string(72, '=').c_str());

    // ---- time the three modes --------------------------------------------
    // Wall-clock around each call, exactly like btstep_bench.F90's time_cuda:
    // btstep_cuda_launch() ends in cudaDeviceSynchronize(), so a host timer
    // captures launch overhead + device time, which is what we want to compare.
    const char *mode_name[3] = {"CUDA faithful (11 kern/substep)",
                                "CUDA fused    ( 5 kern/substep)",
                                "CUDA graph    ( 5 kern, 1 graph)"};
    double mean_ms[3], min_ms[3];
    std::vector<double> eta_out[3], ubt_out[3];

    for (int mode = 0; mode < 3; ++mode) {
        seed_device();
        for (int r = 0; r < n_warm; ++r) btstep_cuda_launch(&args, n_inner, mode);

        std::vector<double> t(n_reps);
        const double ta = wall_s();
        for (int r = 0; r < n_reps; ++r) {
            const double s = wall_s();
            btstep_cuda_launch(&args, n_inner, mode);
            t[r] = (wall_s() - s)*1000.0;
        }
        const double tb = wall_s();

        mean_ms[mode] = (tb - ta)*1000.0/(double)n_reps;
        min_ms[mode]  = *std::min_element(t.begin(), t.end());

        eta.down(); ubt.down();
        eta_out[mode] = eta.h;
        ubt_out[mode] = ubt.h;
    }

    for (int m = 0; m < 3; ++m)
        std::printf("  %-38s: %8.4f ms/call   (min %8.4f)\n",
                    mode_name[m], mean_ms[m], min_ms[m]);
    std::printf("\n");
    std::printf("  faithful / fused    (FUSION win)      : %8.3f x\n", mean_ms[0]/mean_ms[1]);
    std::printf("  fused / graph       (LAUNCH residual) : %8.3f x\n", mean_ms[1]/mean_ms[2]);
    std::printf("\n");

    // ---- internal consistency: fusion must be BIT-identical ---------------
    // Same claim btstep_bench.F90 makes about its Fortran fusion: the merges are
    // single-assignment and eta_sum moves to a point where eta is already final,
    // so anything non-zero here means a merge changed the answer.
    double r_eta = 0.0;
    for (double v : eta_out[0]) r_eta = std::max(r_eta, std::fabs(v));

    std::printf("  faithful vs fused  max|d eta| : %12.5e  (expect 0 -- merges are exact)\n",
                maxdiff(eta_out[0], eta_out[1]));
    std::printf("  fused    vs graph  max|d eta| : %12.5e  (expect 0 -- same kernels)\n",
                maxdiff(eta_out[1], eta_out[2]));
    std::printf("                       |eta| ~  : %12.5e\n", r_eta);

    // ---- verify against the Fortran-driven run ---------------------------
    // Both drivers ran mode 2 (graph) last from the same seed for the same
    // (nwarm+nreps)*n_inner substeps, on inputs this driver adopted above. Same
    // kernels, same numbers -> BIT-EXACT is the bar, not "close".
    int rc = 0;
    if (ref.have) {
        const double de = maxdiff(eta_out[2], ref.eta_f);
        const double du = maxdiff(ubt_out[2], ref.ubt_f);
        std::printf("\n  vs FORTRAN-DRIVEN CUDA (host_data), adopting its exp/sin inputs\n");
        std::printf("        max|d eta| : %12.5e   on |eta| ~ %.5e\n", de, r_eta);
        std::printf("        max|d ubt| : %12.5e\n", du);
        if (de == 0.0 && du == 0.0) {
            std::printf("        verify     : PASS -- BIT-EXACT. Both drivers run the same\n");
            std::printf("                     kernels on the same numbers; OpenACC is the\n");
            std::printf("                     only difference left, so the timings compare.\n");
        } else {
            std::printf("        verify     : *** FAIL -- the two drivers diverge ***\n");
            rc = 1;
        }
        // How far off a from-scratch native init would have been: this is the
        // nvfortran-libm vs glibc-libm gap, NOT anything about the port.
        std::printf("\n  libm gap this run stepped around (native recompute vs nvfortran):\n");
        std::printf("        max|d eta_ini(exp)| : %12.5e\n", d_eta0);
        std::printf("        max|d force_u(sin)| : %12.5e\n", d_fu);
        std::printf("        max|d f_corner|     : %12.5e\n", d_fc);
        std::printf("        (recomputing these instead gives max|d eta| ~ 3.1e-15 after\n");
        std::printf("         %d substeps -- the repo's usual FMA-level agreement)\n",
                    (n_warm + n_reps)*n_inner);
    } else {
        std::printf("\n  (set BTSTEP_REF_IN=<dump> to check against the Fortran-driven run)\n");
    }
    std::printf("%s\n", std::string(72, '=').c_str());
    return rc;
}
