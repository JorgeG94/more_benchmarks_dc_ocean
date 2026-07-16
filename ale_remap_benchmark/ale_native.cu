// NATIVE C++/CUDA driver for the ALE remap benchmark.
// cudaMalloc + main(). No Fortran, no OpenACC, no `host_data`, no CUDA Fortran.
//
// PURPOSE: every CUDA port in this repo is launched FROM Fortran and gets its
// device pointers through `!$acc host_data use_device(...)`, which performs a
// runtime present-table lookup per array per call -- 15 of them here, on the
// CUDA path, inside the timed region. If that lookup costs anything, every
// "CUDA buys N%" number in this repo is UNDERSTATED and the comparison is
// unfair to CUDA. This driver removes the OpenACC runtime from the picture
// entirely and settles it.
//
// THE KERNELS ARE NOT REDEFINED HERE. This calls ale_remap_cuda() from
// ale_kernel.cu -- byte-for-byte the same launcher, and the same 12 kernels,
// that ale_bench.F90 calls through host_data. Duplicating a kernel body would
// let the two copies diverge and silently corrupt the comparison, and the
// comparison is the entire point of the file. (Precedent and rationale:
// ../daxpy_benchmark/daxpy_pure.cu.)
//
//   make native && ./ale_native                    # timings, self-contained
//   ./ale_bench 473 297 30 20 10 25 1              # writes ale_ref.bin
//   ./ale_native --ref ale_ref.bin                 # + verify vs the Fortran run
//   ./ale_native --ref ale_ref.bin --use-ref-state # + isolate init from path
//
// VERIFICATION, three independent legs (see --ref):
//   1. INIT.   The init below is written from ale_bench.F90's init and diffed
//              against the PRISTINE state that run actually used. If this leg
//              does not pass, nothing downstream means anything -- a driver
//              that feeds the kernel different numbers is not measuring the
//              same kernel.
//   2. PATH.   Native output vs the SAME variant's output from the Fortran
//              run (v6 <-> fused=0, v7 <-> fused=1). Same kernels, same input
//              => must be bit-identical. This is the leg that proves the
//              native driver is a like-for-like replacement and not a
//              different computation that merely runs at a similar speed.
//   3. PHYSICS. Native output vs variant 1, DC-as-shipped Fortran. Expected to
//              agree at FMA-contraction level (~1e-12 on hTr), exactly as the
//              Fortran-driven CUDA does -- i.e. the native driver inherits the
//              existing port's agreement with production, it does not launder it.
// --use-ref-state replaces the computed init with the dumped one, which
// separates leg 1 from leg 2: if leg 2 passes only with --use-ref-state, the
// difference is in the init (host libm / FP contraction), not in the CUDA path.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// The ONE thing this file shares with ale_bench.F90: the launcher in
// ale_kernel.cu. `real_t` there is `typedef double`, so plain `double *` here
// is the same type through C linkage.
// ---------------------------------------------------------------------------
extern "C" void ale_remap_cuda(int nx, int ny, int nz, int method, int fused,
                               double *h_layer, double *h_old, double *target_h,
                               double *total_h, double *h_ref,
                               double *hTr_t, double *hTr_s, double *u, double *v,
                               double *mass_b, double *heat_b, double *salt_b,
                               double *eta, double *H_ref, double *dsig);

// Mirrors constants.F90 / ale_bench.F90.
static const int NGHOST = 3;
static const int REMAP_PPM = 3;

#define CUDA_OK(call)                                                          \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(_e), __FILE__, __LINE__);          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Fortran storage order, exactly as ale_kernel.cu's IDX2/IDX3 assume.
static inline size_t id2(int i, int j, int nx) { return (size_t)i + (size_t)nx * j; }
static inline size_t id3(int i, int j, int k, int nx, int ny) {
    return (size_t)i + (size_t)nx * ((size_t)j + (size_t)ny * k);
}

// ---------------------------------------------------------------------------
// Host state. One struct so "pristine" and "working" are obviously the same
// shape, the way ale_bench.F90's p_* arrays are.
// ---------------------------------------------------------------------------
struct HostState {
    int nxt, nyt, nz;
    std::vector<double> dsig;      // (nz)
    std::vector<double> H_ref;     // (nxt,nyt)      bt_H_ref
    std::vector<double> eta;       // (nxt,nyt)      bt_eta
    std::vector<double> h_layer;   // (nxt,nyt,nz)
    std::vector<double> hTr_t;     // (nxt,nyt,nz)
    std::vector<double> hTr_s;     // (nxt,nyt,nz)
    std::vector<double> u;         // (nxt+1,nyt,nz)
    std::vector<double> v;         // (nxt,nyt+1,nz)

    void alloc(int nxt_, int nyt_, int nz_) {
        nxt = nxt_; nyt = nyt_; nz = nz_;
        dsig.assign((size_t)nz, 0.0);
        H_ref.assign((size_t)nxt * nyt, 0.0);
        eta.assign((size_t)nxt * nyt, 0.0);
        h_layer.assign((size_t)nxt * nyt * nz, 0.0);
        hTr_t.assign((size_t)nxt * nyt * nz, 0.0);
        hTr_s.assign((size_t)nxt * nyt * nz, 0.0);
        u.assign((size_t)(nxt + 1) * nyt * nz, 0.0);
        v.assign((size_t)nxt * (nyt + 1) * nz, 0.0);
    }
};

// ---------------------------------------------------------------------------
// Init -- transcribed from ale_bench.F90:108-157, expression by expression.
// Fortran i/j/k are 1-based and the arithmetic below keeps them that way, so
// this reads against the Fortran line-for-line. Association order is preserved
// where it is observable (README §4 records a 1-ulp bug from re-associating
// exactly this kind of expression).
// ---------------------------------------------------------------------------
static void init_state(HostState &s, int pert_pct) {
    const int nxt = s.nxt, nyt = s.nyt, nz = s.nz;

    // dsig(k) = 1 + 3*(k-1)/(nz-1), then normalised to sum 1.
    for (int k = 1; k <= nz; ++k)
        s.dsig[k - 1] = 1.0 + 3.0 * (double)(k - 1) / (double)std::max(1, nz - 1);
    double dsum = 0.0;
    for (int k = 0; k < nz; ++k) dsum += s.dsig[k];   // Fortran sum(): 30 elements
    for (int k = 0; k < nz; ++k) s.dsig[k] = s.dsig[k] / dsum;

    for (int j = 1; j <= nyt; ++j) {
        for (int i = 1; i <= nxt; ++i) {
            double hbed = 15.0 + 4485.0 * std::fabs(std::sin(0.013 * (double)i) *
                                                    std::cos(0.017 * (double)j));
            if ((i + j) % 97 == 0) hbed = 3.0;        // very shallow => vanished layers
            s.H_ref[id2(i - 1, j - 1, nxt)] = hbed;
            const double eta = 0.4 * std::sin(0.05 * (double)i + 0.03 * (double)j);
            s.eta[id2(i - 1, j - 1, nxt)] = eta;

            for (int k = 1; k <= nz; ++k) {
                const double sig = s.dsig[k - 1];
                const double pert = (0.01 * (double)pert_pct) *
                                    std::sin(3.0 * (double)k + 0.02 * (double)i +
                                             0.03 * (double)j);
                s.h_layer[id3(i - 1, j - 1, k - 1, nxt, nyt)] =
                    (hbed + eta) * sig * (1.0 + pert);
            }
            // renormalise so sum_k h_layer == hbed + eta exactly
            double csum = 0.0;
            for (int k = 1; k <= nz; ++k) csum += s.h_layer[id3(i - 1, j - 1, k - 1, nxt, nyt)];
            for (int k = 1; k <= nz; ++k) {
                double &h = s.h_layer[id3(i - 1, j - 1, k - 1, nxt, nyt)];
                h = h * (hbed + eta) / csum;
            }
            for (int k = 1; k <= nz; ++k) {
                const double cT = 2.0 + 12.0 * std::exp(-(double)(nz - k) / 6.0) +
                                  0.5 * std::sin(0.02 * (double)i);
                const double cS = 34.5 + 0.8 * (double)k / (double)nz;
                const double h = s.h_layer[id3(i - 1, j - 1, k - 1, nxt, nyt)];
                s.hTr_t[id3(i - 1, j - 1, k - 1, nxt, nyt)] = h * cT;
                s.hTr_s[id3(i - 1, j - 1, k - 1, nxt, nyt)] = h * cS;
            }
        }
    }

    for (int k = 1; k <= nz; ++k) {
        for (int j = 1; j <= nyt; ++j)
            for (int i = 1; i <= nxt + 1; ++i)
                s.u[id3(i - 1, j - 1, k - 1, nxt + 1, nyt)] =
                    0.3 * std::sin(0.01 * (double)i) * std::cos(0.02 * (double)j) *
                    (1.0 + 0.1 * (double)k);
        for (int j = 1; j <= nyt + 1; ++j)
            for (int i = 1; i <= nxt; ++i)
                s.v[id3(i - 1, j - 1, k - 1, nxt, nyt + 1)] =
                    0.2 * std::cos(0.015 * (double)i) * std::sin(0.018 * (double)j) *
                    (1.0 - 0.05 * (double)k);
    }
}

// ---------------------------------------------------------------------------
// ale_ref.bin reader (written by ale_bench.F90 with arg 7 = 1)
// ---------------------------------------------------------------------------
struct RefVariant {
    int id;
    std::vector<double> h, t, u, eta, b;
};
struct Ref {
    int nxt, nyt, nz, pert_pct;
    HostState pristine;
    std::vector<RefVariant> vars;
};

static void rd(std::FILE *f, void *p, size_t n, const char *what) {
    if (std::fread(p, 1, n, f) != n) {
        std::fprintf(stderr, "ale_ref.bin: short read on %s\n", what);
        std::exit(1);
    }
}
static int32_t rd_i32(std::FILE *f, const char *what) {
    int32_t x; rd(f, &x, 4, what); return x;
}
static void rd_vec(std::FILE *f, std::vector<double> &v, size_t n, const char *what) {
    v.resize(n); rd(f, v.data(), n * sizeof(double), what);
}

static bool load_ref(const std::string &path, Ref &r) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "  cannot open %s\n", path.c_str()); return false; }
    if (rd_i32(f, "magic") != 1094992177) {
        std::fprintf(stderr, "  %s: bad magic\n", path.c_str()); std::fclose(f); return false;
    }
    if (rd_i32(f, "version") != 1) {
        std::fprintf(stderr, "  %s: bad version\n", path.c_str()); std::fclose(f); return false;
    }
    r.nxt = rd_i32(f, "nxt"); r.nyt = rd_i32(f, "nyt");
    r.nz = rd_i32(f, "nz");   r.pert_pct = rd_i32(f, "pert_pct");
    r.pristine.alloc(r.nxt, r.nyt, r.nz);
    const size_t n2 = (size_t)r.nxt * r.nyt, n3 = n2 * r.nz;
    const size_t nu = (size_t)(r.nxt + 1) * r.nyt * r.nz;
    const size_t nv = (size_t)r.nxt * (r.nyt + 1) * r.nz;
    rd_vec(f, r.pristine.dsig, (size_t)r.nz, "dsig");
    rd_vec(f, r.pristine.H_ref, n2, "H_ref");
    rd_vec(f, r.pristine.eta, n2, "eta");
    rd_vec(f, r.pristine.h_layer, n3, "h_layer");
    rd_vec(f, r.pristine.hTr_t, n3, "hTr_t");
    rd_vec(f, r.pristine.hTr_s, n3, "hTr_s");
    rd_vec(f, r.pristine.u, nu, "u");
    rd_vec(f, r.pristine.v, nv, "v");
    const int nvar = rd_i32(f, "nvar");
    for (int i = 0; i < nvar; ++i) {
        RefVariant rv;
        rv.id = rd_i32(f, "variant id");
        rd_vec(f, rv.h, n3, "res_h");
        rd_vec(f, rv.t, n3, "res_t");
        rd_vec(f, rv.u, nu, "res_u");
        rd_vec(f, rv.eta, n2, "res_eta");
        rd_vec(f, rv.b, n3, "res_b");
        r.vars.push_back(std::move(rv));
    }
    std::fclose(f);
    return true;
}

static const RefVariant *find_var(const Ref &r, int id) {
    for (const auto &v : r.vars) if (v.id == id) return &v;
    return nullptr;
}

static double maxdiff(const std::vector<double> &a, const std::vector<double> &b) {
    if (a.size() != b.size()) return HUGE_VAL;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    int nx = 473, ny = 297, nz = 30, reps = 20, warmup = 10, pert_pct = 25;
    std::string ref_path;
    bool use_ref_state = false;

    // positional: nx ny nz reps warmup pert_pct  (same order as ale_bench)
    int pos = 0;
    int *posv[6] = {&nx, &ny, &nz, &reps, &warmup, &pert_pct};
    for (int a = 1; a < argc; ++a) {
        const std::string s = argv[a];
        if (s == "--ref" && a + 1 < argc)      ref_path = argv[++a];
        else if (s == "--use-ref-state")       use_ref_state = true;
        else if (s.rfind("--", 0) == 0) {
            std::fprintf(stderr, "usage: %s [nx ny nz reps warmup pert_pct] "
                                 "[--ref ale_ref.bin] [--use-ref-state]\n", argv[0]);
            return 2;
        } else if (pos < 6) *posv[pos++] = std::atoi(s.c_str());
    }

    const int nxt = nx + 2 * NGHOST, nyt = ny + 2 * NGHOST;
    const size_t n2 = (size_t)nxt * nyt, n3 = n2 * nz;
    const size_t nu = (size_t)(nxt + 1) * nyt * nz;
    const size_t nv = (size_t)nxt * (nyt + 1) * nz;

    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));

    std::printf("================================================================\n");
    std::printf(" ALE remap -- NATIVE C++/CUDA driver (cudaMalloc + main, no Fortran,\n");
    std::printf("              no OpenACC, no host_data). Kernels: ale_kernel.cu.\n");
    std::printf(" %dx%d pts (+%d ghost each side) x nz=%d\n", nx, ny, NGHOST, nz);
    std::printf("   nx_total=%d ny_total=%d => %zu cells\n", nxt, nyt, n3);
    std::printf("   reps=%d  warmup=%d  h-drift=%d%%\n", reps, warmup, pert_pct);
    std::printf("   device: %s (cc%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("================================================================\n");

    // ---- host init -------------------------------------------------------
    HostState hs;
    hs.alloc(nxt, nyt, nz);
    init_state(hs, pert_pct);

    // ---- leg 1: does the init match the Fortran driver's? ----------------
    Ref ref;
    bool have_ref = false;
    if (!ref_path.empty()) {
        have_ref = load_ref(ref_path, ref);
        if (have_ref && (ref.nxt != nxt || ref.nyt != nyt || ref.nz != nz ||
                         ref.pert_pct != pert_pct)) {
            std::fprintf(stderr, "  *** %s is %dx%dx%d pert=%d, this run is "
                                 "%dx%dx%d pert=%d -- not comparable\n",
                         ref_path.c_str(), ref.nxt, ref.nyt, ref.nz, ref.pert_pct,
                         nxt, nyt, nz, pert_pct);
            return 1;
        }
    }
    if (have_ref) {
        std::printf("\n INIT CHECK -- native init vs the Fortran run's PRISTINE state\n");
        std::printf("   (if this is not 0.0, the two drivers do not feed the kernel the\n");
        std::printf("    same numbers and no downstream comparison means anything)\n");
        const struct { const char *n; const std::vector<double> *a, *b; } fields[] = {
            {"dsig",     &hs.dsig,    &ref.pristine.dsig},
            {"bt_H_ref", &hs.H_ref,   &ref.pristine.H_ref},
            {"bt_eta",   &hs.eta,     &ref.pristine.eta},
            {"h_layer",  &hs.h_layer, &ref.pristine.h_layer},
            {"hTr_T",    &hs.hTr_t,   &ref.pristine.hTr_t},
            {"hTr_S",    &hs.hTr_s,   &ref.pristine.hTr_s},
            {"u_face_x", &hs.u,       &ref.pristine.u},
            {"v_face_y", &hs.v,       &ref.pristine.v},
        };
        int n_exact = 0;
        for (const auto &f : fields) {
            const double d = maxdiff(*f.a, *f.b);
            std::printf("   %-10s max|diff| = %10.3e%s\n", f.n, d, d == 0.0 ? "   (exact)" : "");
            if (d == 0.0) ++n_exact;
        }
        std::printf("   => %d/8 fields bit-identical\n", n_exact);
        if (use_ref_state) {
            std::printf("   --use-ref-state: running on the DUMPED state instead, so the\n");
            std::printf("   output check below isolates the CUDA path from the init.\n");
            hs.dsig = ref.pristine.dsig;   hs.H_ref = ref.pristine.H_ref;
            hs.eta = ref.pristine.eta;     hs.h_layer = ref.pristine.h_layer;
            hs.hTr_t = ref.pristine.hTr_t; hs.hTr_s = ref.pristine.hTr_s;
            hs.u = ref.pristine.u;         hs.v = ref.pristine.v;
        }
    }

    // ---- device buffers --------------------------------------------------
    // Working set, exactly the arrays ale_bench.F90 hands to ale_remap_cuda.
    double *d_h_layer, *d_h_old, *d_target_h, *d_total_h, *d_h_ref;
    double *d_hTr_t, *d_hTr_s, *d_u, *d_v;
    double *d_mass_b, *d_heat_b, *d_salt_b, *d_eta, *d_H_ref, *d_dsig;
    CUDA_OK(cudaMalloc(&d_h_layer,  n3 * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_h_old,    n3 * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_target_h, n3 * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_total_h,  n2 * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_h_ref,    n2 * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_hTr_t,    n3 * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_hTr_s,    n3 * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_u,        nu * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_v,        nv * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_mass_b,   n3 * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_heat_b,   n3 * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_salt_b,   n3 * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_eta,      n2 * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_H_ref,    n2 * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_dsig,     (size_t)nz * sizeof(double)));

    // Pristine device copies -- the equivalent of ale_bench.F90's p_* arrays.
    // The restore they feed is MANDATORY for a meaningful measurement, not just
    // for correctness: the remap has a FIXED POINT (after one call
    // h_old == target_h, so every new layer aligns exactly with an old one and
    // the PPM overlap sweep collapses to one iteration per layer). Timing reps
    // without restoring measures a kernel doing strictly less work than
    // production's. Same reason, same placement (outside the timed section) as
    // the Fortran driver -- otherwise this is not the same measurement.
    double *p_h_layer, *p_hTr_t, *p_hTr_s, *p_u, *p_v, *p_eta;
    CUDA_OK(cudaMalloc(&p_h_layer, n3 * sizeof(double)));
    CUDA_OK(cudaMalloc(&p_hTr_t,   n3 * sizeof(double)));
    CUDA_OK(cudaMalloc(&p_hTr_s,   n3 * sizeof(double)));
    CUDA_OK(cudaMalloc(&p_u,       nu * sizeof(double)));
    CUDA_OK(cudaMalloc(&p_v,       nv * sizeof(double)));
    CUDA_OK(cudaMalloc(&p_eta,     n2 * sizeof(double)));

    CUDA_OK(cudaMemcpy(p_h_layer, hs.h_layer.data(), n3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(p_hTr_t,   hs.hTr_t.data(),   n3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(p_hTr_s,   hs.hTr_s.data(),   n3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(p_u,       hs.u.data(),       nu * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(p_v,       hs.v.data(),       nv * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(p_eta,     hs.eta.data(),     n2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_H_ref,   hs.H_ref.data(),   n2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_dsig,    hs.dsig.data(), (size_t)nz * sizeof(double), cudaMemcpyHostToDevice));
    // scratch: written before read by the kernels, zeroed only for determinism
    CUDA_OK(cudaMemset(d_h_old, 0, n3 * sizeof(double)));
    CUDA_OK(cudaMemset(d_target_h, 0, n3 * sizeof(double)));
    CUDA_OK(cudaMemset(d_total_h, 0, n2 * sizeof(double)));
    CUDA_OK(cudaMemset(d_h_ref, 0, n2 * sizeof(double)));

    // ale_bench.F90's restore(): h_layer, both tracers, the three budgets
    // (pristine value 0), u, v, bt_eta. target_h/h_old/total_h/h_ref are
    // outputs, recomputed every call, and are NOT restored there either.
    auto restore = [&]() {
        CUDA_OK(cudaMemcpy(d_h_layer, p_h_layer, n3 * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpy(d_hTr_t,   p_hTr_t,   n3 * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpy(d_hTr_s,   p_hTr_s,   n3 * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpy(d_u,       p_u,       nu * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpy(d_v,       p_v,       nv * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpy(d_eta,     p_eta,     n2 * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemset(d_mass_b, 0, n3 * sizeof(double)));
        CUDA_OK(cudaMemset(d_heat_b, 0, n3 * sizeof(double)));
        CUDA_OK(cudaMemset(d_salt_b, 0, n3 * sizeof(double)));
    };

    auto call_remap = [&](int fused) {
        ale_remap_cuda(nxt, nyt, nz, REMAP_PPM, fused,
                       d_h_layer, d_h_old, d_target_h, d_total_h, d_h_ref,
                       d_hTr_t, d_hTr_s, d_u, d_v,
                       d_mass_b, d_heat_b, d_salt_b, d_eta, d_H_ref, d_dsig);
    };

    // ---- time ------------------------------------------------------------
    // Same protocol as ale_bench.F90: restore (untimed), sync, wall clock
    // around ONE call + sync, min over reps. `system_clock` there, steady_clock
    // here -- both wall clock.
    struct Out { std::vector<double> h, t, u, eta, b; };
    double best[2];
    Out out[2];
    for (int fused = 0; fused <= 1; ++fused) {
        for (int r = 0; r < warmup; ++r) { restore(); call_remap(fused); }
        CUDA_OK(cudaDeviceSynchronize());

        double bst = HUGE_VAL;
        for (int r = 0; r < reps; ++r) {
            restore();
            CUDA_OK(cudaDeviceSynchronize());
            const auto t0 = std::chrono::steady_clock::now();
            call_remap(fused);
            CUDA_OK(cudaDeviceSynchronize());
            const auto t1 = std::chrono::steady_clock::now();
            const double dt = std::chrono::duration<double>(t1 - t0).count();
            if (dt < bst) bst = dt;
        }
        best[fused] = bst;

        Out &o = out[fused];
        o.h.resize(n3); o.t.resize(n3); o.u.resize(nu); o.eta.resize(n2); o.b.resize(n3);
        CUDA_OK(cudaMemcpy(o.h.data(),   d_h_layer, n3 * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(o.t.data(),   d_hTr_t,   n3 * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(o.u.data(),   d_u,       nu * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(o.eta.data(), d_eta,     n2 * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(o.b.data(),   d_heat_b,  n3 * sizeof(double), cudaMemcpyDeviceToHost));
    }

    std::printf("\n variant                          ms\n");
    std::printf(" ------------------------------------\n");
    std::printf("  %-26s%9.3f\n", "CUDA faithful (native)", best[0] * 1.0e3);
    std::printf("  %-26s%9.3f\n", "CUDA fused (native)", best[1] * 1.0e3);

    int rc = 0;
    if (have_ref) {
        const char *lbl[2] = {"CUDA faithful", "CUDA fused"};
        const int match[2] = {6, 7};    // ale_bench variant ids for fused=0/1

        // ---- leg 2: same kernels, same input => must be bit-identical -----
        std::printf("\n PATH CHECK -- native vs the SAME kernel launched from Fortran via\n");
        std::printf("   host_data (v6/v7). Same kernels, same state => expect exactly 0.0.\n");
        std::printf(" %-24s %10s %10s %10s %10s %10s\n", "", "h_layer", "hTr_T",
                    "u_face_x", "bt_eta", "heat_budget");
        for (int f = 0; f <= 1; ++f) {
            const RefVariant *rv = find_var(ref, match[f]);
            if (!rv) continue;
            const double d[5] = {maxdiff(out[f].h, rv->h), maxdiff(out[f].t, rv->t),
                                 maxdiff(out[f].u, rv->u), maxdiff(out[f].eta, rv->eta),
                                 maxdiff(out[f].b, rv->b)};
            const bool exact = (d[0] == 0.0 && d[1] == 0.0 && d[2] == 0.0 &&
                                d[3] == 0.0 && d[4] == 0.0);
            std::printf("  %-23s %10.3e %10.3e %10.3e %10.3e %10.3e%s\n", lbl[f],
                        d[0], d[1], d[2], d[3], d[4],
                        exact ? "   PASS (bit-identical)" : "");
            // Only a real failure when the state is guaranteed identical.
            // Under the computed init these are expected to be ~1e-11: that is
            // the init's libm noise propagating, not a path difference.
            if (use_ref_state && !exact) {
                std::printf("   *** FAIL: same kernels + same state must be bit-identical ***\n");
                rc = 1;
            }
        }
        if (!use_ref_state)
            std::printf("   (computed init: non-zero here is the init's libm noise, not the\n"
                        "    CUDA path -- re-run with --use-ref-state to isolate it.)\n");

        // ---- leg 3: vs DC-as-shipped Fortran (FMA-level expected) ---------
        const RefVariant *v1 = find_var(ref, 1);
        if (v1) {
            std::printf("\n PHYSICS CHECK -- native vs variant 1, DC-as-shipped Fortran.\n");
            std::printf("   FMA contraction between nvcc and nvfortran is expected here;\n");
            std::printf("   README reports 1.8e-12 on hTr_T for the Fortran-driven CUDA.\n");
            for (int f = 0; f <= 1; ++f)
                std::printf("  %-23s %10.3e %10.3e %10.3e %10.3e %10.3e\n", lbl[f],
                            maxdiff(out[f].h, v1->h), maxdiff(out[f].t, v1->t),
                            maxdiff(out[f].u, v1->u), maxdiff(out[f].eta, v1->eta),
                            maxdiff(out[f].b, v1->b));
        }

        // ---- does the check DISCRIMINATE? A check that cannot fail proves
        // nothing. Perturb ONE cell of the native result by 1 ulp and diff it
        // against the UNPERTURBED native result: the one thing that differs is
        // the perturbation, so a non-zero answer proves maxdiff() can see a
        // single-cell 1-ulp change anywhere in a 4.35M-element array.
        //   Deliberately NOT diffed against the Fortran result, the way
        // ale_bench.F90's probe is. That form is only meaningful when the
        // baseline is exactly 0.0 -- which holds there, and holds here under
        // --use-ref-state, but NOT under the computed init, whose libm-level
        // noise (~1e-13) swamps a 1-ulp probe and would make the probe report
        // a pass it did not earn.
        {
            std::vector<double> probe = out[1].h;
            const size_t c = id3(nxt / 2, nyt / 2, std::max(1, nz / 2), nxt, nyt);
            probe[c] = std::nextafter(probe[c], HUGE_VAL);
            const double trip = maxdiff(probe, out[1].h);
            std::printf("\n check-discriminates probe: perturb 1 cell of the native result by\n");
            std::printf("   1 ulp -> max|diff| vs the unperturbed native result = %.3e"
                        "   (MUST be > 0)\n", trip);
            if (!(trip > 0.0)) { std::printf("   *** PROBE FAILED: the check cannot fail ***\n"); rc = 1; }
        }
    } else {
        std::printf("\n (no --ref given: timings only, nothing verified. Run\n");
        std::printf("  `./ale_bench 473 297 30 20 10 25 1` then re-run with\n");
        std::printf("  `--ref ale_ref.bin` to verify against the Fortran driver.)\n");
    }

    cudaFree(d_h_layer); cudaFree(d_h_old); cudaFree(d_target_h); cudaFree(d_total_h);
    cudaFree(d_h_ref); cudaFree(d_hTr_t); cudaFree(d_hTr_s); cudaFree(d_u); cudaFree(d_v);
    cudaFree(d_mass_b); cudaFree(d_heat_b); cudaFree(d_salt_b); cudaFree(d_eta);
    cudaFree(d_H_ref); cudaFree(d_dsig);
    cudaFree(p_h_layer); cudaFree(p_hTr_t); cudaFree(p_hTr_s); cudaFree(p_u);
    cudaFree(p_v); cudaFree(p_eta);
    return rc;
}
