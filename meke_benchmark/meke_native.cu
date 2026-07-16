// MEKE step, NATIVE C++/CUDA driver: cudaMalloc + main(). No Fortran, no OpenACC.
//
// THE QUESTION THIS ANSWERS
// Every CUDA number in this repo is produced by launching the kernels FROM
// Fortran, with device pointers obtained through `!$acc host_data use_device`.
// The OpenACC runtime therefore sits somewhere on the CUDA path. If that costs
// anything, every "CUDA buys N%" figure here is understated and the comparison
// is unfair to CUDA. This driver removes OpenACC from the process entirely --
// no libacc, no present table, no stdpar -- and re-measures the same kernels on
// cudaMalloc'd memory.
//
// ⚠ WHAT THE FORTRAN DRIVER ACTUALLY DOES (read before interpreting the delta):
// meke_bench.F90 calls `fill_args` ONCE, at line 233, OUTSIDE the timing loop;
// `time_cuda` then loops on `meke_cuda_launch` alone. So the host_data
// present-table lookup is paid once per RUN, not once per CALL. The premise
// "a lookup per array per call" does not hold for this benchmark. What remains
// for this driver to detect is therefore only the PROCESS-level cost of having
// the OpenACC/stdpar runtime resident at all (context setup, default-stream
// semantics, driver state) -- a real but much smaller question. A tie here is
// the expected result, and is itself the useful finding.
//
// THE KERNELS ARE NOT REDEFINED HERE. This calls meke_cuda_launch() from
// meke_kernel.cu -- the identical function meke_bench.F90 calls -- and shares
// `struct MekeArgs` with it through meke_args.h rather than copying it. If the
// kernels or the struct were duplicated, a divergence between the two copies
// would silently corrupt the comparison, and the comparison is the entire point
// of the file. (daxpy_benchmark/daxpy_pure.cu is the precedent.)
//
// INIT MUST MATCH meke_bench.F90 OR THE COMPARISON IS MEANINGLESS. Every
// initialiser below is a transcription of a specific line of that file, cited
// inline. Verified, not assumed: `--verify` diffs this driver's fields against
// binary dumps of the Fortran driver's own arrays (see `make verify-native`).
//
// ⚠ THE CONFIG IS DEGENERATE, AND THAT IS FAITHFUL. Inherited from
// meke_bench.F90's init: all five alpha_* are 0 and there is no wavespeed slot,
// so rd_ws == 0 => Ldeform == 0 => le == 0 => kh_diff == 0, and cd_scale == 0
// with min_gamma2 == 1e-4 leaves bottom_fac2 == barotr_fac2 == 1 EXACTLY. The
// length-scale and Kh-closure kernels still run, still load every array and
// still execute their sqrt/pow, but their branches never fire. This exercises
// MEKE's SHAPE, not its arithmetic -- and it is why DC-vs-CUDA agrees at
// exactly 0 (`pow(1.0, 0.8)` is exactly 1 in both toolchains). Do not read the
// exact-zero agreement as general FMA agreement between nvfortran and nvcc.
//
// Usage: ./meke_native [nx] [ny] [nz] [nreps] [nwarm] [ntrials]
//   nx/ny are POINT COUNTS (arrays are nx+2*nghost), as in meke_bench.F90.
//   ./meke_native                -> 473 297 30, the gabight_sph_meke_v100 config
//   ./meke_native --verify       -> also diff against the Fortran dumps
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "meke_args.h"   // struct MekeArgs + IT/IU/IV/I3 + meke_cuda_launch

// ---- constants: meke_bench.F90:76-78 -------------------------------------
static const int    NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30;
static const int    REPS_DEF = 200, WARM_DEF = 20, NGH = 3;
static const int    TRIALS_DEF = 3;
static const double DXM = 8000.0, DYM = 8000.0, DT_THERMO = 1800.0;

#define CUDA_OK(call)                                                          \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(_e), __FILE__, __LINE__);          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// mirrors meke_bench.F90's `wall()` (system_clock, monotonic, seconds)
static double wall() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

// Fortran `max(k, 1)` on the integer divisions in the Gaussian bumps.
static inline int imax(int a, int b) { return a > b ? a : b; }

struct HostField {
    std::vector<double> h;
    double *d = nullptr;
    size_t  n = 0;
    void alloc(size_t nn) { n = nn; h.assign(nn, 0.0); CUDA_OK(cudaMalloc(&d, nn * sizeof(double))); }
    void up()   { CUDA_OK(cudaMemcpy(d, h.data(), n * sizeof(double), cudaMemcpyHostToDevice)); }
    void down() { CUDA_OK(cudaMemcpy(h.data(), d, n * sizeof(double), cudaMemcpyDeviceToHost)); }
};

// ---- the benchmark state, laid out exactly as meke_bench.F90 allocates it --
struct Bench {
    int nx, ny, nz;
    // --use-ref-init: replace the three libm-dependent init fields with the
    // Fortran driver's own bits. See main() for why this exists -- it is what
    // turns "agrees to 1.9e-15" into a claim about the KERNEL PATH rather than
    // about two libm implementations.
    bool use_ref_init = false;
    std::vector<double> ref_meke, ref_f, ref_gm;
    // ocean_meke_t payload (meke_bench.F90:321-332 `alloc_meke`). del2 / baro_hu /
    // baro_hv are allocated there but never reach the CUDA port, so they are
    // omitted here -- MekeArgs has no slot for them.
    HostField meke, kh_diff, le, ku, i_mass, depth_tot, bottom_fac2, barotr_fac2;
    HostField src, uflux, vflux, mass_ws, u_bbl2, ke_diss_ws, rd_ws, f_centre;
    HostField sn_u_ws, sn_v_ws;
    // metrics / state / forcing
    HostField areaT, iareaT, idxT, idyT, dy_cu, dx_cv, idxCu, idyCv;
    HostField h_layer, rho_layer, gm_src, ke_diss_ext;

    void allocate_all() {
        const size_t nt = (size_t)nx * ny;
        const size_t nu = (size_t)(nx + 1) * ny;
        const size_t nv = (size_t)nx * (ny + 1);
        const size_t n3 = (size_t)nx * ny * nz;
        meke.alloc(nt); kh_diff.alloc(nt); le.alloc(nt); ku.alloc(nt);
        i_mass.alloc(nt); depth_tot.alloc(nt); bottom_fac2.alloc(nt);
        barotr_fac2.alloc(nt); src.alloc(nt); mass_ws.alloc(nt);
        u_bbl2.alloc(nt); ke_diss_ws.alloc(nt); rd_ws.alloc(nt); f_centre.alloc(nt);
        uflux.alloc(nu); sn_u_ws.alloc(nu);
        vflux.alloc(nv); sn_v_ws.alloc(nv);
        areaT.alloc(nt); iareaT.alloc(nt); idxT.alloc(nt); idyT.alloc(nt);
        dy_cu.alloc(nu); idxCu.alloc(nu);
        dx_cv.alloc(nv); idyCv.alloc(nv);
        h_layer.alloc(n3); rho_layer.alloc(n3);
        gm_src.alloc(nt); ke_diss_ext.alloc(nt);
    }

    // meke_bench.F90:106-126 -- metrics, stratification, GM source, KE dissipation.
    // These are set once and never re-seeded (the kernels only read them).
    void init_static() {
        for (size_t t = 0; t < areaT.n; ++t) {
            areaT.h[t]  = DXM * DYM;            // :106
            iareaT.h[t] = 1.0 / (DXM * DYM);
            idxT.h[t]   = 1.0 / DXM;            // :107
            idyT.h[t]   = 1.0 / DYM;
        }
        for (size_t t = 0; t < dy_cu.n; ++t) { dy_cu.h[t] = DYM; idxCu.h[t] = 1.0 / DXM; }   // :108-109
        for (size_t t = 0; t < dx_cv.n; ++t) { dx_cv.h[t] = DXM; idyCv.h[t] = 1.0 / DYM; }
        // :112-119 -- a stratified column
        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i) {
                    h_layer.h[I3(i, j, k)]   = 4000.0 / (double)nz;
                    rho_layer.h[I3(i, j, k)] = 1025.0 + 5.0 * (double)(nz - k) / (double)nz;
                }
        // :120-126 -- a GM PE-release source that varies in space, so the
        // drag/closure branches see a real spread of E rather than one value
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                const double a = (double)(i - nx / 2) / (double)imax(nx / 6, 1);
                const double b = (double)(j - ny / 2) / (double)imax(ny / 6, 1);
                gm_src.h[IT(i, j)] = 1.0e-2 * std::exp(-(a * a + b * b));
                ke_diss_ext.h[IT(i, j)] = -1.0e-3;   // hvisc KE dissipation (<=0); frcoeff<0 ignores it
            }
        if (use_ref_init && ref_gm.size() == gm_src.n) gm_src.h = ref_gm;
    }

    // meke_bench.F90:336-363 `init_meke` -- the per-variant re-seedable state.
    // Called before each timed mode, exactly as `reseed_cu` (:365-372) does.
    void init_meke() {
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i) {
                // a seeded eddy-energy field so sqrt/drag see a real spread
                const double a = (double)(i - nx / 3) / (double)imax(nx / 8, 1);
                const double b = (double)(j - ny / 3) / (double)imax(ny / 8, 1);
                meke.h[IT(i, j)] = 1.0e-3 + 1.0e-2 * std::exp(-(a * a + b * b));
                // |f| at cell centres, beta-plane -- as set_f_centre would fill it
                f_centre.h[IT(i, j)] = std::fabs(-1.0e-4 + 2.0e-11 * (double)(j - ny / 2) * DYM);
            }
        if (use_ref_init && ref_meke.size() == meke.n) { meke.h = ref_meke; f_centre.h = ref_f; }
        std::fill(kh_diff.h.begin(), kh_diff.h.end(), 0.0);
        std::fill(le.h.begin(), le.h.end(), 0.0);
        std::fill(ku.h.begin(), ku.h.end(), 0.0);
        std::fill(i_mass.h.begin(), i_mass.h.end(), 0.0);
        std::fill(depth_tot.h.begin(), depth_tot.h.end(), 0.0);
        std::fill(bottom_fac2.h.begin(), bottom_fac2.h.end(), 0.0);
        std::fill(barotr_fac2.h.begin(), barotr_fac2.h.end(), 0.0);
        std::fill(src.h.begin(), src.h.end(), 0.0);
        std::fill(uflux.h.begin(), uflux.h.end(), 0.0);
        std::fill(vflux.h.begin(), vflux.h.end(), 0.0);
        std::fill(mass_ws.h.begin(), mass_ws.h.end(), 0.0);
        std::fill(u_bbl2.h.begin(), u_bbl2.h.end(), 0.0);   // use_bbl_drag off -> stays 0, as production
        std::fill(ke_diss_ws.h.begin(), ke_diss_ws.h.end(), 0.0);
        std::fill(rd_ws.h.begin(), rd_ws.h.end(), 0.0);
        std::fill(sn_u_ws.h.begin(), sn_u_ws.h.end(), 0.0);
        std::fill(sn_v_ws.h.begin(), sn_v_ws.h.end(), 0.0);
    }

    void push_static() {
        areaT.up(); iareaT.up(); idxT.up(); idyT.up();
        dy_cu.up(); idxCu.up(); dx_cv.up(); idyCv.up();
        h_layer.up(); rho_layer.up(); gm_src.up(); ke_diss_ext.up();
    }
    // mirrors `reseed_cu` (meke_bench.F90:365-372) field-for-field
    void push_reseed() {
        meke.up(); kh_diff.up(); le.up(); ku.up(); i_mass.up();
        depth_tot.up(); bottom_fac2.up(); barotr_fac2.up(); src.up();
        uflux.up(); vflux.up(); mass_ws.up(); u_bbl2.up();
        ke_diss_ws.up(); rd_ws.up(); f_centre.up();
        sn_u_ws.up(); sn_v_ws.up();
    }

    // meke_bench.F90:436-471 `fill_args`, minus host_data: these pointers come
    // straight from cudaMalloc, which is the entire point of this driver.
    void fill_args(MekeArgs &a) {
        a.meke = meke.d; a.kh_diff = kh_diff.d; a.le = le.d; a.ku = ku.d;
        a.i_mass = i_mass.d; a.depth_tot = depth_tot.d;
        a.bottom_fac2 = bottom_fac2.d; a.barotr_fac2 = barotr_fac2.d;
        a.src = src.d; a.uflux = uflux.d; a.vflux = vflux.d;
        a.mass_ws = mass_ws.d; a.rd_ws = rd_ws.d;
        a.sn_u_ws = sn_u_ws.d; a.sn_v_ws = sn_v_ws.d;
        a.ke_diss_ws = ke_diss_ws.d; a.u_bbl2 = u_bbl2.d;
        a.f_centre = f_centre.d; a.gm_src = gm_src.d;
        a.ke_diss_ext = ke_diss_ext.d;
        a.h_layer = h_layer.d; a.rho_layer = rho_layer.d;
        a.areaT = areaT.d; a.iareaT = iareaT.d;
        a.idxT = idxT.d; a.idyT = idyT.d;
        a.dy_cu = dy_cu.d; a.dx_cv = dx_cv.d;
        a.idxCu = idxCu.d; a.idyCv = idyCv.d;
        a.nx = nx; a.ny = ny; a.nz = nz;
        // scalars: ocean_meke_t's production defaults (meke_state.F90:52-105)
        // overridden by the gabight_sph_meke_v100.nml knobs init_meke sets.
        a.dt = DT_THERMO;
        a.dtscale   = 1.0;        // default
        a.cd_scale  = 0.0;        // default
        a.cb        = 25.0;       // default
        a.ct        = 50.0;       // default
        a.min_gamma2 = 1.0e-4;    // default
        a.cdrag     = 2.5e-3;     // nml (also the default)
        a.uscale    = 0.0;        // default
        a.a_deform = 0.0; a.a_rhines = 0.0; a.a_eady = 0.0;   // defaults -> le == 0
        a.a_frict  = 0.0; a.a_grid  = 0.0;
        a.bgsrc    = 0.0;         // default
        a.gmcoeff  = 0.9;         // nml
        a.frcoeff  = -1.0;        // default -> ke_diss_ws term off
        a.damping  = 0.0;         // default
        a.kh_bg    = 100.0;       // nml
        a.k4       = -1.0;        // default -> biharmonic off
        a.khmeke_fac = 0.0;       // default
        a.khcoeff  = 1.0;         // nml
        a.visc_coeff_ku = 0.0;    // default
        a.rho0     = 1035.0;      // ocean_gm_t default
        a.backscatter = 0;        // default .false.
    }
};

// ---- reference dumps from the Fortran driver (see `make verify-native`) ----
static bool read_dump(const char *path, std::vector<double> &out) {
    std::FILE *f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize((size_t)bytes / sizeof(double));
    size_t got = std::fread(out.data(), sizeof(double), out.size(), f);
    std::fclose(f);
    return got == out.size();
}

// max|a-b| and the count of bit-level mismatches -- both matter: a 0.0 max with
// a non-zero mismatch count is impossible, and a tiny max with a huge count is
// last-bit noise rather than a port bug.
static void diff_report(const char *what, const std::vector<double> &a,
                        const std::vector<double> &b) {
    if (a.size() != b.size()) {
        std::printf("  %-26s : SIZE MISMATCH (%zu vs %zu)\n", what, a.size(), b.size());
        return;
    }
    double md = 0.0, mag = 0.0;
    size_t nbit = 0;
    for (size_t t = 0; t < a.size(); ++t) {
        double d = std::fabs(a[t] - b[t]);
        if (d > md) md = d;
        if (std::fabs(a[t]) > mag) mag = std::fabs(a[t]);
        if (std::memcmp(&a[t], &b[t], sizeof(double)) != 0) ++nbit;
    }
    std::printf("  %-26s : max|d| = %11.5e   |ref|~%10.4e   bit-differing: %zu / %zu%s\n",
                what, md, mag, nbit, a.size(), (nbit == 0 ? "   <- BIT-IDENTICAL" : ""));
}

int main(int argc, char **argv) {
    bool verify = false, use_ref_init = false;
    std::vector<char *> pos;
    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "--verify") == 0)       verify = true;
        else if (std::strcmp(argv[i], "--use-ref-init") == 0) { use_ref_init = true; verify = true; }
        else pos.push_back(argv[i]);
    }
    auto arg = [&](size_t k, int dflt) { return k < pos.size() ? std::atoi(pos[k]) : dflt; };

    const int nxp = arg(0, NXP_DEF), nyp = arg(1, NYP_DEF);
    const int n_reps = arg(3, REPS_DEF), n_warm = arg(4, WARM_DEF);
    const int n_trials = arg(5, TRIALS_DEF);

    Bench B;
    B.nx = nxp + 2 * NGH; B.ny = nyp + 2 * NGH; B.nz = arg(2, NZ_DEF);
    const int nx = B.nx, ny = B.ny;

    std::printf("%s\n", std::string(74, '=').c_str());
    std::printf("  NATIVE C++/CUDA driver: cudaMalloc + main(), no Fortran, no OpenACC\n");
    std::printf("  config: %d x %d points (+%d ghosts), gabight_sph_meke_v100.nml\n", nxp, nyp, NGH);
    std::printf("  arrays: %d x %d = %d cells;  nz = %d\n", nx, ny, nx * ny, B.nz);
    std::printf("  %d timed reps, %d warm-up, min of %d trials\n", n_reps, n_warm, n_trials);
    std::printf("%s\n", std::string(74, '=').c_str());

    B.allocate_all();

    // --use-ref-init: take meke / f_centre / gm_src from the Fortran driver's
    // own dumps instead of recomputing them here.
    //
    // WHY: the three fields above are the ONLY init values that depend on libm
    // (exp) or on host FMA contraction. Everything else in the init is exact
    // constants and plain IEEE arithmetic, identical in both compilers. Without
    // this flag, a final-field difference is ambiguous: nvfortran's exp() and
    // glibc's exp() differ by ~1 ULP, so the two runs do not start from the same
    // bits and CANNOT be expected to end on them. Feeding in the Fortran's bits
    // removes that variable, leaving the kernel path as the only thing that can
    // differ -- which is the actual question.
    if (use_ref_init) {
        const bool ok = read_dump("meke_ref_init.bin", B.ref_meke) &&
                        read_dump("meke_ref_fcentre.bin", B.ref_f) &&
                        read_dump("meke_ref_gmsrc.bin", B.ref_gm);
        if (!ok) {
            std::fprintf(stderr, "  --use-ref-init: dumps missing. Run: MEKE_DUMP=1 ./meke_bench\n");
            return 1;
        }
        B.use_ref_init = true;
        std::printf("  init source: FORTRAN DUMPS (meke/f_centre/gm_src) -- libm difference removed\n");
    }

    B.init_static();
    B.push_static();

    MekeArgs args;
    B.init_meke();
    B.fill_args(args);   // no host_data: these ARE the cudaMalloc pointers

    // Same order and rep counts as meke_bench.F90:233-236, so the final `meke`
    // is a deterministic function of the same number of steps and can be
    // compared bit-for-bit against the Fortran run's own output.
    const char *names[3] = {"CUDA faithful (16 kernels)", "CUDA fused    ( 6 kernels)",
                            "CUDA graph    ( 6 kern, 1 graph)"};
    double best[3] = {1e30, 1e30, 1e30};

    for (int trial = 0; trial < n_trials; ++trial) {
        for (int mode = 0; mode < 3; ++mode) {
            B.init_meke();
            B.push_reseed();
            for (int r = 0; r < n_warm; ++r) meke_cuda_launch(&args, mode);
            const double t0 = wall();
            for (int r = 0; r < n_reps; ++r) meke_cuda_launch(&args, mode);
            const double t1 = wall();
            const double ms = (t1 - t0) * 1000.0 / (double)n_reps;
            if (ms < best[mode]) best[mode] = ms;
        }
    }

    std::printf("  %-36s : %10.4f ms/step\n", names[0], best[0]);
    std::printf("  %-36s : %10.4f ms/step\n", names[1], best[1]);
    std::printf("  %-36s : %10.4f ms/step\n", names[2], best[2]);
    std::printf("\n");
    std::printf("  cuda-faithful / cuda-fused (FUSION in C)             : %10.3f x\n", best[0] / best[1]);
    std::printf("  cuda-fused / cuda-graph    (launch residual)         : %10.3f x\n", best[1] / best[2]);
    std::printf("\n");

    // ---- results back, and verification --------------------------------
    // The last mode to run was the graph, from a fresh reseed -- exactly as
    // meke_bench.F90 leaves m_cu. So this array is directly comparable.
    B.meke.down();
    double mn = 1e300, mx = -1e300;
    for (size_t t = 0; t < B.meke.n; ++t) { if (B.meke.h[t] < mn) mn = B.meke.h[t]; if (B.meke.h[t] > mx) mx = B.meke.h[t]; }
    std::printf("  final meke range           : [%.17g, %.17g]\n", mn, mx);

    // Also report the degenerate-config invariants, so a silent config drift
    // (a live alpha_*, a wavespeed slot) shows up instead of being assumed away.
    B.le.down(); B.kh_diff.down(); B.bottom_fac2.down(); B.barotr_fac2.down();
    bool le0 = true, kh0 = true, bf1 = true, tf1 = true;
    for (size_t t = 0; t < B.le.n; ++t) {
        if (B.le.h[t] != 0.0) le0 = false;
        if (B.kh_diff.h[t] != 0.0) kh0 = false;
        if (B.bottom_fac2.h[t] != 1.0) bf1 = false;
        if (B.barotr_fac2.h[t] != 1.0) tf1 = false;
    }
    std::printf("  degenerate-config check    : le==0 %s, kh_diff==0 %s, bottom_fac2==1 %s, barotr_fac2==1 %s\n",
                le0 ? "Y" : "N", kh0 ? "Y" : "N", bf1 ? "Y" : "N", tf1 ? "Y" : "N");

    if (verify) {
        std::printf("\n  --- vs Fortran driver's own arrays (MEKE_DUMP=1 ./meke_bench) ---\n");
        std::vector<double> ref;
        if (!B.use_ref_init) {
            // (a) INIT fields: quantifies the host-side libm/FMA difference in the
            //     initialiser, which is NOT on the kernel path. If these differ,
            //     a difference in the final field is not evidence of a port bug --
            //     the two runs simply did not start from the same bits. Rerun with
            //     --use-ref-init to remove this variable entirely.
            std::vector<double> h_init_meke, h_init_f;
            B.init_meke(); h_init_meke = B.meke.h; h_init_f = B.f_centre.h;
            if (read_dump("meke_ref_init.bin", ref))    diff_report("init meke   (host init)", h_init_meke, ref);
            else std::printf("  (meke_ref_init.bin not found -- run: MEKE_DUMP=1 ./meke_bench)\n");
            if (read_dump("meke_ref_fcentre.bin", ref)) diff_report("init f_centre (host init)", h_init_f, ref);
            if (read_dump("meke_ref_gmsrc.bin", ref))   diff_report("init gm_src (host init)", B.gm_src.h, ref);
        } else {
            std::printf("  init fields taken from the Fortran dumps -- identical by construction.\n");
        }
        // (b) FINAL field: the end-to-end check. Under --use-ref-init both runs
        //     start from identical bits and run identical kernels, so the ONLY
        //     remaining difference would be the device pointers' provenance
        //     (cudaMalloc vs host_data). Expect exactly 0.
        B.meke.down();
        if (read_dump("meke_ref_final.bin", ref))   diff_report("final meke  (end-to-end)", B.meke.h, ref);
    }
    std::printf("%s\n", std::string(74, '=').c_str());

    meke_cuda_graph_reset();
    return 0;
}
