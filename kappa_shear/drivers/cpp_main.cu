// NATIVE driver for the kappa-shear (JHL08) column kernel: cudaMalloc + main().
// No Fortran, no OpenACC, no host_data. nvcc links it on its own.
//
// THE QUESTION
// ------------
// Every CUDA port in this repo is launched FROM Fortran and gets its device
// pointers from `!$acc host_data use_device(...)`, which is a runtime
// present-table lookup per array per call -- ELEVEN of them here, every single
// launch. So the OpenACC runtime sits on the CUDA path inside the very number
// used to say "CUDA buys 1.119x". If that lookup costs anything, the CUDA side
// is understated and the comparison is unfair to CUDA. This driver removes
// OpenACC entirely: pointers come from cudaMalloc and go straight to the launch
// function. If native ties with the Fortran-driven number, the repo's CUDA
// figures are already fair -- which is a real result, not a null one.
//
// THE CARDINAL RULE -- the kernel is NOT redefined here (daxpy_pure.cu's
// precedent). This calls ks_cuda_launch() from ks_kernel.cu, the IDENTICAL
// object ks_bench links. Duplicating the kernel would let the two copies drift
// and silently corrupt the comparison -- and the comparison is the entire point
// of the file. There is no __global__ and no line of solver physics below.
//
// WHAT IS TIMED
// -------------
// Launch + cudaDeviceSynchronize on device-resident data: exactly the window
// ks_bench.F90 times for its CUDA variant (H2D happens once, in setup, as it
// does there). So native vs Fortran-driven isolates ONE variable -- who owns
// the memory and how the pointer reaches the kernel.
//
// HOW CORRECTNESS IS ESTABLISHED  (verify_against_reference)
// ----------------------------------------------------------
// build_state() below is a hand mirror of ks_bench.F90:build_state. A mirror
// that is subtly wrong makes the timing meaningless -- it would be a different
// problem, quietly, and a wall-clock number cannot tell you that. So the mirror
// is not trusted: `make native-verify` runs ks_bench with KS_DUMP set and this
// binary checks itself against the result.
//
// That check has to separate TWO risks that are easy to conflate:
//
//   RISK A -- is the DRIVER right? (shapes, column-major layout, KsPar layout,
//   launch, D2H). Confounded by nothing, and testable exactly: with KS_REF set
//   this driver OVERWRITES its mirror with the Fortran's actual input bits and
//   runs the kernel on those. Same kernel + same bits in => same bits out,
//   necessarily. So kd_int MUST come back BIT-IDENTICAL to the Fortran-driven
//   CUDA run. No tolerance is appropriate; any difference is a bug here.
//
//   RISK B -- is the MIRROR right? Checked separately, by comparing the mirror
//   against those same Fortran inputs. It CANNOT be bit-identical and that is
//   not a defect: nvfortran -fast evaluates sin/cos/tanh/exp with its own libm,
//   glibc rounds them differently, and the two disagree in the last ulp. This
//   was measured, not assumed -- f_centre is just |2*Omega*sin(lat)|, one sin()
//   and no arithmetic to contract, and it still differs by ~1 ulp. So the
//   mirror is held to a few-ulp RELATIVE bound (MIRROR_ULP_TOL below) and the
//   physics is checked where it counts: identical Picard iteration counts, i.e.
//   the kernel does the same work either way.
//
// Then, as a cross-check against the existing Fortran numbers:
//   kd_int vs `do concurrent` reproduces the PRE-EXISTING ~2e-9 rel gap, judged
//   at the same 1e-12 tolerance ks_bench uses. That tolerance is NOT loosened
//   here, and the gap is a property of the DC-vs-CUDA comparison that predates
//   this file (see README) -- not something the native driver introduced.
//
// INHERITED CAVEAT: this mirrors ks_bench's state, so it inherits its known
// limitation -- n_out = 1 for 100% of columns, i.e. the adaptive substepper
// never engages, which is why ks_bench reproduces production's MEDIAN runtime
// but not its 9x variance. A native driver cannot change that: it is a property
// of the state, not of who owns the memory.
//
// Usage: ./ks_native [nxp] [nyp] [nz] [nreps] [nwarm] [cuda_sync] [land_pct]
//   ./ks_native                  -> 473 297 30, the 0.1 deg gabight config
//   KS_REF=ks_ref.bin ./ks_native   -> also verify against the Fortran dump

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include "gpu_rt.h"
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Mirrors `struct KsPar` in ks_kernel.cu and `type, bind(C) :: ks_par_t` in
// ks_bench.F90. All three must agree; a layout drift would feed the kernel
// garbage knobs and it would still happily run.
// ---------------------------------------------------------------------------
struct KsPar {
   double dt, ri_crit, shearmix_rate, fri_curvature;
   double c_n, c_s, lambda, lz_rescale;
   double kappa_0, kappa_seed, kappa_trunc, tke_bg;
   double tol_err, src_max_chg, vel_underflow, rho0;
   int max_inner_it, max_substep_it;
   int eos_variant;
   double eos_rho0, eos_alpha_T, eos_beta_S;
};

// KERNEL UNDER TEST. Default is the faithful port in ks_kernel.cu; building
// with -DKS_USE_OPT (make cpp CUVARIANT=opt) links opt_kernel.cu instead --
// same signature, same 1-based contract, so nothing below changes. The two
// launchers deliberately have distinct names so a mislink is a link error
// rather than a silently-wrong measurement.
//
// ⚠ BOTH CUDA kernels bound the `kappa_out = kappa` copy to 1..nz+1
// (ks_kernel.cu:798). The Fortran copies the DECLARED extent unless built with
// BCOPY=1. Compare cuda_* against DC BCOPY=1 for like-for-like; the BCOPY=0
// rows measure that copy, they do not measure codegen.
// bcopy=N in the RESULT line means "this build does the SAME copy work as a
// Fortran build with BCOPY=N", so the CSV shows at a glance whether a
// DC-vs-CUDA row pair is matched. cuda_opt always bounds the copy.
#if defined(KS_USE_OPT) || !defined(KS_FULL_COPY)
#define KS_BCOPY_EQUIV 1
#else
#define KS_BCOPY_EQUIV 0
#endif

#ifdef KS_USE_OPT
#define KS_CUVARIANT "cuda_opt"
#define ks_kernel_launch ks_opt_launch
#define ks_kernel_attrs  ks_opt_attrs
#else
#define KS_CUVARIANT "cuda_faithful"
#define ks_kernel_launch ks_cuda_launch
#define ks_kernel_attrs  ks_cuda_attrs
#endif

extern "C" void ks_kernel_launch(const double *h_layer, const double *u_face,
                                 const double *v_face, const double *hT,
                                 const double *hS, const double *wet_mask,
                                 const double *f_centre, double *kd_int,
                                 double *tke_int, int *n_out_a, int *n_in_a,
                                 int nx, int ny, int nz, const KsPar *pin,
                                 int sync);

extern "C" void ks_kernel_attrs(int *nregs, int *local_bytes, int *shared_bytes,
                                int *max_tpb);

// From ks_bench.F90. DT_THERM is not a knob: dt sets how many adaptive
// substeps the outer loop needs, so changing it changes the problem.
static const int NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30;
static const int REPS_DEF = 200, WARM_DEF = 10, NGHOST = 3;
static const double DT_THERM = 300.0;  // &time_nml dt_fixed, therm ratio 1
static const int EOS_VARIANT_LINEAR = 1;

#ifndef MODEL_NZ_STACK_MAX
#define MODEL_NZ_STACK_MAX 128
#endif

#define CUDA_OK(call)                                                          \
   do {                                                                        \
      cudaError_t _e = (call);                                                 \
      if (_e != cudaSuccess) {                                                 \
         std::fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                      cudaGetErrorString(_e), __FILE__, __LINE__);             \
         std::exit(1);                                                         \
      }                                                                        \
   } while (0)

// Same clock discipline as ks_bench.F90's system_clock wall(): host-side,
// around a synchronising launch. Deliberately NOT cudaEvent -- the point is to
// measure what the Fortran driver measures, including any host-side per-launch
// cost, since host-side per-launch cost is exactly what is under test.
static double wall_s() {
   struct timespec ts;
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

// ===========================================================================
// Host state
// ===========================================================================
struct State {
   int nx = 0, ny = 0, nz = 0, land_pct = 0;
   size_t nT = 0, nU = 0, nV = 0;  // per-level element counts, Fortran shapes
   std::vector<double> h_layer, u_face, v_face, hT, hS, wet_mask, f_centre;
};

// A LINE-BY-LINE mirror of ks_bench.F90:build_state (its :368-455). Expression
// structure is kept identical -- same groupings, same order of operations --
// because the two must agree to the last bit and a re-associated `0.5*(a+b)`
// does not. Fortran's 1-based (i,j,kg) indexing is preserved for the same
// reason ks_kernel.cu preserves it: it kills a class of transcription bug that
// no comparison can catch once both sides are wrong the same way.
//
// The physics rationale (z* stretch, mixed layer, jet at the ML base straddling
// Ri_c) lives in ks_bench.F90's header. Not repeated, so there is one place to
// change it.
static void build_state(State &s, int nxp, int nyp) {
   const int nx = s.nx, ny = s.ny, nz = s.nz;
   const double PI = 3.141592653589793;

   std::vector<double> w(nz + 1, 0.0);
   const double r = 1.18;  // geometric stretch -> ~5 m surface layer at H=4000
   double wsum = 0.0;
   for (int m = 1; m <= nz; ++m) {
      w[m] = std::pow(r, (double)(m - 1));
      wsum = wsum + w[m];
   }
   for (int m = 1; m <= nz; ++m) w[m] = w[m] / wsum;

   for (int j = 1; j <= ny; ++j) {
      const double yr = (double)(j - NGHOST) / (double)std::max(nyp, 1);
      const double lat = -60.61 + 0.1 * (double)(j - NGHOST);
      for (int i = 1; i <= nx; ++i) {
         const double xr = (double)(i - NGHOST) / (double)std::max(nxp, 1);

         // |f| from the planetary metric (coriolis_scheme = "planetary").
         s.f_centre[(i - 1) + (size_t)(j - 1) * nx] =
             std::fabs(2.0 * 7.2921e-5 * std::sin(lat * PI / 180.0));

         // Bathymetry: shelf -> abyssal, smooth, 200..4500 m.
         const double depth =
             200.0 + 4300.0 * (0.5 * (1.0 + std::tanh(3.0 * (yr - 0.25)))) *
                         (0.85 + 0.15 * std::sin(6.0 * xr));

         // Land: a contiguous NW block, so the mask is spatially coherent
         // (scattered land would understate warp divergence).
         double wet = 1.0;
         if (s.land_pct > 0) {
            if (xr < 0.01 * (double)s.land_pct && yr > 0.55) wet = 0.0;
         }
         s.wet_mask[(i - 1) + (size_t)(j - 1) * nx] = wet;

         // Mixed-layer depth 40..100 m, and the jet strength that competes with
         // the stratification at its base.
         const double mld =
             40.0 + 60.0 * (0.5 * (1.0 + std::sin(5.0 * xr) * std::cos(4.0 * yr)));
         const double us =
             0.25 + 0.60 * (0.5 * (1.0 + std::sin(7.0 * xr + 2.0 * yr) *
                                             std::cos(3.0 * yr)));
         const double vs = 0.10 * std::sin(4.0 * xr) * std::cos(5.0 * yr);

         double zt = 0.0;
         for (int m = 1; m <= nz; ++m) {  // m = local, surface-down
            const double hh = std::fmax(depth * w[m], 1.0e-3);
            const double zb = zt + hh;
            const double zm = 0.5 * (zt + zb);

            // T uniform in the ML, thermocline below (T_surf 14 -> T_bot 2).
            double tt;
            if (zm <= mld) {
               tt = 14.0;
            } else {
               tt = 2.0 + 12.0 * std::exp(-(zm - mld) / 600.0);
            }
            const double ss = 35.0 - 0.5 * std::exp(-zm / 300.0);

            // u: surface jet, shear concentrated at the ML base.
            const double uu = us * 0.5 * (1.0 - std::tanh((zm - mld) / 25.0));
            const double vv = vs * 0.5 * (1.0 - std::tanh((zm - mld) / 25.0));

            const int kg = nz + 1 - m;  // global bottom-up: kg = nz is surface
            const size_t t =
                (i - 1) + (size_t)(j - 1) * nx + (size_t)(kg - 1) * s.nT;
            s.h_layer[t] = hh;
            s.hT[t] = tt * hh;
            s.hS[t] = ss * hh;
            s.u_face[(i - 1) + (size_t)(j - 1) * (nx + 1) +
                     (size_t)(kg - 1) * s.nU] = uu;
            s.v_face[(i - 1) + (size_t)(j - 1) * nx + (size_t)(kg - 1) * s.nV] = vv;
            zt = zb;
         }
      }
   }

   // Face arrays carry one extra row/column; fill the overhang by copy so the
   // face average at i = nx / j = ny is not reading uninitialised memory.
   for (int kg = 1; kg <= nz; ++kg) {
      for (int j = 1; j <= ny; ++j)
         s.u_face[nx + (size_t)(j - 1) * (nx + 1) + (size_t)(kg - 1) * s.nU] =
             s.u_face[(nx - 1) + (size_t)(j - 1) * (nx + 1) +
                      (size_t)(kg - 1) * s.nU];
      for (int i = 1; i <= nx; ++i)
         s.v_face[(i - 1) + (size_t)ny * nx + (size_t)(kg - 1) * s.nV] =
             s.v_face[(i - 1) + (size_t)(ny - 1) * nx + (size_t)(kg - 1) * s.nV];
   }
}

// ===========================================================================
// Comparison -- ks_bench.F90:compare, same 1e-12 threshold, plus an exact-bit
// count (the Fortran does not need one; here bit-identity is the actual claim).
// ===========================================================================
struct Cmp {
   double dmax = 0.0, rmax = 0.0;
   double bscale = 0.0;  // max|b| -- the array's own natural scale
   long nbad = 0;        // cells > 1e-12 rel  -- ks_bench's threshold, unchanged
   long ndiff = 0;       // cells differing in ANY bit
   // Error measured against the array's scale rather than each element's own
   // magnitude. See check_mirror for why that is the right instrument there.
   double scaled() const { return bscale > 0.0 ? dmax / bscale : 0.0; }
};

static Cmp compare(const double *a, const double *b, size_t n) {
   Cmp c;
   for (size_t m = 0; m < n; ++m) {
      if (a[m] != b[m]) c.ndiff++;
      const double d = std::fabs(a[m] - b[m]);
      if (d > c.dmax) c.dmax = d;
      const double ab = std::fabs(b[m]);
      if (ab > c.bscale) c.bscale = ab;
      const double sc = std::fmax(std::fabs(a[m]), ab);
      if (sc > 1.0e-30) {
         const double rel = d / sc;
         if (rel > c.rmax) c.rmax = rel;
         if (rel > 1.0e-12) c.nbad++;
      }
   }
   return c;
}

// The mirror is held to max|d| / max|ref| -- error against the ARRAY'S OWN
// SCALE, not against each element's magnitude. Why not per-element relative:
//
//   u = us * 0.5 * (1 - tanh((zm-mld)/25))
//
// is the surface jet, and deep in the column tanh -> 1, so (1 - tanh) is
// catastrophic cancellation BY CONSTRUCTION. A 1-ulp libm disagreement in
// tanh() there lands as a ~5e-10 RELATIVE error on a velocity that is
// physically nil (~1e-7 m/s). Measured, not assumed: u_face_x showed max rel
// 2.7e-10 at max|d| = 1.7e-15 -- i.e. one ulp of the jet's actual scale, while
// h/hT/hS/f_centre (no cancellation) sat at ~1e-15 relative. So per-element
// relative flags the tail of a decaying jet and calls the mirror broken; it is
// the wrong instrument, not a real finding.
//
// 1e-12 of scale is ~1e4 ulp: loose enough to be robust to libm, and still
// orders of magnitude tighter than any genuine transcription error (a swapped
// index or dropped term misses by O(1) relative, never by 1e-12).
static const double MIRROR_TOL = 1.0e-12;

struct Ref {
   bool loaded = false;
   State st;                    // the Fortran's INPUT bits
   std::vector<double> kd_dc;   // do concurrent output
   std::vector<double> kd_cu;   // CUDA-via-Fortran output
};

static bool read_block(std::FILE *f, std::vector<double> &dst, size_t n,
                       const char *what) {
   dst.resize(n);
   if (std::fread(dst.data(), sizeof(double), n, f) != n) {
      std::fprintf(stderr, "  KS_REF: short read on %s\n", what);
      return false;
   }
   return true;
}

// Reads ks_bench.F90:maybe_dump's stream file. Must run BEFORE the device
// allocation: on success the caller swaps the mirror out for these exact bits,
// which is what makes the bit-identity check downstream a real test of the
// driver rather than a test of libm.
static bool load_reference(const char *path, const State &s, Ref &r) {
   std::FILE *f = std::fopen(path, "rb");
   if (!f) {
      std::printf("\n  KS_REF: cannot open %s -- NOT VERIFIED\n", path);
      std::printf("          run:  make native-verify\n");
      return false;
   }
   int hdr[4];
   if (std::fread(hdr, sizeof(int), 4, f) != 4) {
      std::printf("\n  KS_REF: short header -- NOT VERIFIED\n");
      std::fclose(f);
      return false;
   }
   if (hdr[0] != s.nx || hdr[1] != s.ny || hdr[2] != s.nz || hdr[3] != s.land_pct) {
      std::printf("\n  KS_REF: dump is %dx%dx%d land=%d, this run is %dx%dx%d land=%d\n",
                  hdr[0], hdr[1], hdr[2], hdr[3], s.nx, s.ny, s.nz, s.land_pct);
      std::printf("          -> re-run ks_bench with matching args. NOT VERIFIED.\n");
      std::fclose(f);
      return false;
   }
   const size_t nz = (size_t)s.nz, nI = s.nT * (nz + 1);
   r.st = s;  // shapes
   bool ok = read_block(f, r.st.h_layer, s.nT * nz, "h_layer") &&
             read_block(f, r.st.u_face, s.nU * nz, "u_face_x") &&
             read_block(f, r.st.v_face, s.nV * nz, "v_face_y") &&
             read_block(f, r.st.hT, s.nT * nz, "hT") &&
             read_block(f, r.st.hS, s.nT * nz, "hS") &&
             read_block(f, r.st.wet_mask, s.nT, "wet_mask") &&
             read_block(f, r.st.f_centre, s.nT, "f_centre") &&
             read_block(f, r.kd_dc, nI, "kd_int(dc)") &&
             read_block(f, r.kd_cu, nI, "kd_cu");
   std::fclose(f);
   r.loaded = ok;
   return ok;
}

// RISK B: is build_state() a faithful mirror? Reports agreement, does not gate
// on bit-identity (it cannot -- see the header).
static bool check_mirror(const State &mine, const Ref &r) {
   const size_t nz = (size_t)mine.nz;
   struct InArr {
      const char *name;
      const std::vector<double> *a, *b;
      size_t n;
   };
   const InArr in[] = {
       {"h_layer ", &mine.h_layer, &r.st.h_layer, mine.nT * nz},
       {"u_face_x", &mine.u_face, &r.st.u_face, mine.nU * nz},
       {"v_face_y", &mine.v_face, &r.st.v_face, mine.nV * nz},
       {"hT      ", &mine.hT, &r.st.hT, mine.nT * nz},
       {"hS      ", &mine.hS, &r.st.hS, mine.nT * nz},
       {"wet_mask", &mine.wet_mask, &r.st.wet_mask, mine.nT},
       {"f_centre", &mine.f_centre, &r.st.f_centre, mine.nT},
   };
   std::printf("\n  [RISK B] build_state mirror vs the Fortran's inputs\n");
   std::printf("           Bit-identity is NOT attainable: nvfortran -fast and glibc\n");
   std::printf("           round sin/cos/tanh/exp differently. Evidence: f_centre is\n");
   std::printf("           one sin() with nothing to contract, and still differs.\n");
   std::printf("           Gate = max|d|/max|ref| <= %.0e (see MIRROR_TOL). The\n", MIRROR_TOL);
   std::printf("           per-element `max rel` column is shown but NOT gated on --\n");
   std::printf("           0.5*(1-tanh) cancels by construction in the jet's tail.\n");
   bool ok = true;
   for (const InArr &a : in) {
      const Cmp c = compare(a.a->data(), a.b->data(), a.n);
      const bool pass = (c.scaled() <= MIRROR_TOL);
      if (!pass) ok = false;
      std::printf("    %s : %-8s  max|d|/scale = %.3e   max|d| = %.3e   (per-elem max rel = %.3e)\n",
                  a.name, pass ? "OK" : "*** FAIL", c.scaled(), c.dmax, c.rmax);
   }
   std::printf("    -> %s\n",
               ok ? "mirror agrees to ~1 ulp of each array's scale; same problem."
                  : "*** mirror is WRONG -- a real transcription error, not libm.");
   return ok;
}

// RISK A: is the driver right? Run on the Fortran's own input bits, so this is
// a clean test with libm factored out entirely.
static bool check_driver(const Ref &r, const std::vector<double> &kd_native,
                         size_t nI) {
   std::printf("\n  [RISK A] native driver, run on the Fortran's OWN input bits\n");

   const Cmp cdc = compare(kd_native.data(), r.kd_dc.data(), nI);
   std::printf("    vs `do concurrent`   : max|d| = %.5e  max rel = %.5e\n",
               cdc.dmax, cdc.rmax);
   std::printf("        %ld cells > 1e-12 rel (ks_bench's threshold, unchanged).\n",
               cdc.nbad);
   std::printf("        Expected: the PRE-EXISTING ~2e-9 DC-vs-CUDA gap the Fortran\n");
   std::printf("        driver already reports. Predates this file; see README.\n");

   const Cmp ccu = compare(kd_native.data(), r.kd_cu.data(), nI);
   const bool bit = (ccu.ndiff == 0);
   std::printf("    vs CUDA-via-Fortran  : %s\n",
               bit ? "BIT-IDENTICAL (0 of the cells differ)" : "*** MISMATCH ***");
   if (bit) {
      std::printf("        Same kernel, same input bits, same output bits. The\n");
      std::printf("        cudaMalloc path reproduces the host_data path EXACTLY,\n");
      std::printf("        so any timing difference is cost, not a different problem.\n");
   } else {
      std::printf("        %ld cells differ, max|d| = %.5e, max rel = %.5e\n",
                  ccu.ndiff, ccu.dmax, ccu.rmax);
      std::printf("        Same kernel + same input bits MUST give the same bits.\n");
      std::printf("        This is a bug in ks_native.cu, not a tolerance question.\n");
   }
   return bit;
}

// ===========================================================================
int main(int argc, char **argv) {
   const int nxp = iarg(argc, argv, 1, NXP_DEF);
   const int nyp = iarg(argc, argv, 2, NYP_DEF);
   const int nz = iarg(argc, argv, 3, NZ_DEF);
   int n_reps = iarg(argc, argv, 4, REPS_DEF);   // <= 0 -> auto (see below)
   const int n_warm = iarg(argc, argv, 5, WARM_DEF);
   const int cu_sync = iarg(argc, argv, 6, 1);
   const int land_pct = iarg(argc, argv, 7, 0);

   const int nx = nxp + 2 * NGHOST;
   const int ny = nyp + 2 * NGHOST;
   if (nxp < 1 || nyp < 1 || nz < 2) {
      std::fprintf(stderr, "ERROR: need nxp,nyp >= 1 and nz >= 2\n");
      return 1;
   }
   if (nz + 1 > MODEL_NZ_STACK_MAX) {
      std::fprintf(stderr, "ERROR: nz+1 = %d exceeds NZ_STACK_MAX = %d\n", nz + 1,
                   MODEL_NZ_STACK_MAX);
      return 1;
   }

   State s;
   s.nx = nx; s.ny = ny; s.nz = nz; s.land_pct = land_pct;
   s.nT = (size_t)nx * ny;
   s.nU = (size_t)(nx + 1) * ny;
   s.nV = (size_t)nx * (ny + 1);
   const size_t nI = s.nT * (size_t)(nz + 1);
   const size_t ncol = s.nT;

   s.h_layer.assign(s.nT * nz, 0.0);
   s.u_face.assign(s.nU * nz, 0.0);
   s.v_face.assign(s.nV * nz, 0.0);
   s.hT.assign(s.nT * nz, 0.0);
   s.hS.assign(s.nT * nz, 0.0);
   s.wet_mask.assign(s.nT, 0.0);
   s.f_centre.assign(s.nT, 0.0);

   build_state(s, nxp, nyp);

   // The reference must be loaded BEFORE the H2D copies: on success the mirror
   // is swapped out for the Fortran's actual input bits, which is what turns
   // the kd_cu comparison into a clean test of the DRIVER (Risk A) instead of a
   // test of whose libm rounds sin() better (Risk B). Both are then reported
   // separately rather than being allowed to alibi each other.
   Ref ref;
   State mirror;          // kept so the mirror's WORK can be checked below
   bool mirror_ok = true;
   const char *refpath = std::getenv("KS_REF");
   if (refpath && refpath[0]) {
      if (load_reference(refpath, s, ref)) {
         mirror_ok = check_mirror(s, ref);
         mirror = s;
         s.h_layer = ref.st.h_layer;
         s.u_face = ref.st.u_face;
         s.v_face = ref.st.v_face;
         s.hT = ref.st.hT;
         s.hS = ref.st.hS;
         s.wet_mask = ref.st.wet_mask;
         s.f_centre = ref.st.f_centre;
         std::printf("\n  KS_REF: running the kernel on the FORTRAN'S input bits\n");
         std::printf("          (mirror set aside for this run -- see [RISK A]).\n");
      }
   }

   // Knob values: ocean_kappa_shear_t's declared defaults (ks.F90:66-90) plus
   // the three ks_bench.F90 overrides sourced from gabight's namelists
   // (alpha_T = 0.2 and rho_0 = 1035 from &ocean_ic_nml; beta_S is the EOS
   // default). Kept in the same order as the struct so a drift is visible.
   KsPar p;
   p.dt = DT_THERM;
   p.ri_crit = 0.25;
   p.shearmix_rate = 0.089;
   p.fri_curvature = -0.97;
   p.c_n = 0.24;
   p.c_s = 0.14;
   p.lambda = 0.82;
   p.lz_rescale = 1.0;
   p.kappa_0 = 1.0e-7;
   p.kappa_seed = 1.0;
   p.kappa_trunc = 1.0e-9;
   p.tke_bg = 0.0;
   p.tol_err = 0.1;
   p.src_max_chg = 10.0;
   p.vel_underflow = 0.0;
   p.rho0 = 1035.0;
   p.max_inner_it = 50;
   p.max_substep_it = 13;
   p.eos_variant = EOS_VARIANT_LINEAR;
   p.eos_rho0 = 1035.0;
   p.eos_alpha_T = 0.2;
   p.eos_beta_S = 7.6e-4;

   long nwet = 0;
   for (size_t m = 0; m < s.nT; ++m) if (s.wet_mask[m] > 0.0) nwet++;
   const double gib = 11.0 * (double)nx * (double)ny * (double)nz * 8.0 /
                      (1024.0 * 1024.0 * 1024.0);

   std::printf("%s\n", std::string(74, '=').c_str());
   std::printf("  driver: NATIVE C++/CUDA -- cudaMalloc, no Fortran, no OpenACC\n");
   std::printf("  domain: %d x %d x %d interior (+%d ghosts/side)\n", nxp, nyp, nz, NGHOST);
   std::printf("  arrays: %d x %d -> %zu COLUMNS (the parallel width)\n", nx, ny, ncol);
   std::printf("  wet columns: %ld  (%5.1f %%)\n", nwet,
               100.0 * (double)nwet / (double)ncol);
   std::printf("  NZ_STACK_MAX = %d (production); nz = %d\n", MODEL_NZ_STACK_MAX, nz);
   std::printf("  dt (therm) = %6.1f s\n", DT_THERM);
   std::printf("  reps:   %d timed, %d warm-up\n", n_reps, n_warm);
   std::printf("  global mem ~%.2f GiB;  cuda_sync = %d\n", gib, cu_sync);
   std::printf("%s\n", std::string(74, '=').c_str());

   // ---- device allocation: the whole point. No present table, no host_data.
   double *d_h, *d_u, *d_v, *d_hT, *d_hS, *d_wet, *d_fc, *d_kd, *d_tke;
   int *d_nout, *d_nin;
   CUDA_OK(cudaMalloc(&d_h, s.h_layer.size() * sizeof(double)));
   CUDA_OK(cudaMalloc(&d_u, s.u_face.size() * sizeof(double)));
   CUDA_OK(cudaMalloc(&d_v, s.v_face.size() * sizeof(double)));
   CUDA_OK(cudaMalloc(&d_hT, s.hT.size() * sizeof(double)));
   CUDA_OK(cudaMalloc(&d_hS, s.hS.size() * sizeof(double)));
   CUDA_OK(cudaMalloc(&d_wet, s.wet_mask.size() * sizeof(double)));
   CUDA_OK(cudaMalloc(&d_fc, s.f_centre.size() * sizeof(double)));
   CUDA_OK(cudaMalloc(&d_kd, nI * sizeof(double)));
   CUDA_OK(cudaMalloc(&d_tke, nI * sizeof(double)));
   CUDA_OK(cudaMalloc(&d_nout, ncol * sizeof(int)));
   CUDA_OK(cudaMalloc(&d_nin, ncol * sizeof(int)));

   CUDA_OK(cudaMemcpy(d_h, s.h_layer.data(), s.h_layer.size() * sizeof(double), cudaMemcpyHostToDevice));
   CUDA_OK(cudaMemcpy(d_u, s.u_face.data(), s.u_face.size() * sizeof(double), cudaMemcpyHostToDevice));
   CUDA_OK(cudaMemcpy(d_v, s.v_face.data(), s.v_face.size() * sizeof(double), cudaMemcpyHostToDevice));
   CUDA_OK(cudaMemcpy(d_hT, s.hT.data(), s.hT.size() * sizeof(double), cudaMemcpyHostToDevice));
   CUDA_OK(cudaMemcpy(d_hS, s.hS.data(), s.hS.size() * sizeof(double), cudaMemcpyHostToDevice));
   CUDA_OK(cudaMemcpy(d_wet, s.wet_mask.data(), s.wet_mask.size() * sizeof(double), cudaMemcpyHostToDevice));
   CUDA_OK(cudaMemcpy(d_fc, s.f_centre.data(), s.f_centre.size() * sizeof(double), cudaMemcpyHostToDevice));
   // ks_bench uses `acc enter data create` for these -- the kernel writes every
   // element of 1..nz+1 for every column, wet or dry, so their prior contents
   // are never read. Zeroed here only so a future kernel change fails loudly
   // and reproducibly rather than reading whatever cudaMalloc handed back.
   CUDA_OK(cudaMemset(d_kd, 0, nI * sizeof(double)));
   CUDA_OK(cudaMemset(d_tke, 0, nI * sizeof(double)));
   CUDA_OK(cudaMemset(d_nout, 0, ncol * sizeof(int)));
   CUDA_OK(cudaMemset(d_nin, 0, ncol * sizeof(int)));

   auto launch = [&](int sy) {
      ks_kernel_launch(d_h, d_u, d_v, d_hT, d_hS, d_wet, d_fc, d_kd, d_tke,
                       d_nout, d_nin, nx, ny, nz, &p, sy);
   };

   for (int rep = 0; rep < n_warm; ++rep) launch(1);
   CUDA_OK(cudaDeviceSynchronize());

   // Auto rep count (nreps <= 0), matching dc_main.F90: one timed probe launch
   // sizes the timed loop to ~KS_TARGET_MS. An nz sweep spans ~2 orders of
   // magnitude in cost/rep, so a fixed count either burns an hour at the deep
   // end or measures clock noise at the shallow end.
   double target_ms = 1000.0;
   if (const char *e = std::getenv("KS_TARGET_MS")) {
      const double v = std::atof(e);
      if (v > 0.0) target_ms = v;
   }
   // Ceiling on the auto rep count -- rep-to-rep spread is small here, so past
   // ~50-100 reps you buy decimal places nobody reads, once per sweep config.
   int max_reps = 100;
   if (const char *e = std::getenv("KS_MAX_REPS")) {
      const int v = std::atoi(e);
      if (v > 0) max_reps = v;
   }
   if (n_reps <= 0) {
      const double a = wall_s();
      launch(1);
      const double ms_probe = (wall_s() - a) * 1000.0;
      n_reps = (ms_probe <= 0.0) ? max_reps
                                 : std::max(3, std::min(max_reps, (int)(target_ms / ms_probe) + 1));
      std::printf("  auto-reps: probe %10.4f ms/rep -> reps = %d (target %.1f ms, cap %d)\n",
                  ms_probe, n_reps, target_ms, max_reps);
   }

   // ks_bench times (n_reps launches at cuda_sync) + (1 forced-sync launch) and
   // divides by n_reps+1. Reproduced exactly, so `mean` is comparable to its
   // printed number without a footnote. Per-rep times are also kept: the brief
   // asks for min-of-N, and under GPU contention min is the only statistic with
   // a defensible meaning.
   std::vector<double> per(n_reps + 1, 0.0);
   const double t0 = wall_s();
   for (int rep = 0; rep < n_reps; ++rep) {
      const double a = wall_s();
      launch(cu_sync);
      per[rep] = (wall_s() - a) * 1000.0;
   }
   {
      const double a = wall_s();
      launch(1);
      per[n_reps] = (wall_s() - a) * 1000.0;
   }
   const double t1 = wall_s();
   const double ms_mean = (t1 - t0) * 1000.0 / (double)(n_reps + 1);

   std::vector<double> srt = per;
   std::sort(srt.begin(), srt.end());
   const double ms_min = srt.front();
   const double ms_med = srt[srt.size() / 2];
   const double ms_max = srt.back();

   std::vector<double> kd(nI), tke(nI);
   std::vector<int> nout(ncol), nin(ncol);
   CUDA_OK(cudaMemcpy(kd.data(), d_kd, nI * sizeof(double), cudaMemcpyDeviceToHost));
   CUDA_OK(cudaMemcpy(tke.data(), d_tke, nI * sizeof(double), cudaMemcpyDeviceToHost));
   CUDA_OK(cudaMemcpy(nout.data(), d_nout, ncol * sizeof(int), cudaMemcpyDeviceToHost));
   CUDA_OK(cudaMemcpy(nin.data(), d_nin, ncol * sizeof(int), cudaMemcpyDeviceToHost));

   std::printf("\n  CUDA native (cudaMalloc, no OpenACC)  : %10.4f ms/rep  (mean, ks_bench's statistic)\n", ms_mean);
   std::printf("    per-rep min / med / max             : %10.4f %9.4f %9.4f ms\n",
               ms_min, ms_med, ms_max);
   std::printf("    -> compare against ks_bench's `CUDA C faithful` line.\n");
   std::printf("       ⚠ TIMING IS ONLY MEANINGFUL ON AN IDLE GPU. Check\n");
   std::printf("         nvidia-smi --query-compute-apps=pid --format=csv first;\n");
   std::printf("         sibling jobs have been seen to move this by 4x.\n");

   int nregs, lmem, smem, maxtpb;
   ks_kernel_attrs(&nregs, &lmem, &smem, &maxtpb);
   std::printf("\n  --- kernel resources (cudaFuncGetAttributes, authoritative) ---\n");
   std::printf("  %-8s : regs=%5d  local=%8d B/thread  shared=%7d B  maxTPB=%5d\n",
               KS_CUVARIANT, nregs, lmem, smem, maxtpb);
   std::printf("    (identical to ks_bench's line -- same kernel object. If these\n");
   std::printf("     ever differ, the two binaries are not linking the same code.)\n");

   // n_out is the caveat in the header, printed rather than asserted: if a
   // future state change makes the substepper engage, this line moves and the
   // "reproduces the median, not the variance" caveat needs revisiting.
   long h_out[16] = {0};
   long nactive = 0, in_tot = 0, out_tot = 0;
   int in_mx = 0, in_mn = 1 << 30;
   for (size_t m = 0; m < ncol; ++m) {
      h_out[std::min(nout[m], 15)]++;
      if (nin[m] > 0) nactive++;
      out_tot += nout[m];
      in_tot += nin[m];
      in_mx = std::max(in_mx, nin[m]);
      in_mn = std::min(in_mn, nin[m]);
   }
   std::printf("\n  --- iteration counts (the inherited-state caveat) ---\n");
   for (int b = 0; b <= 15; ++b)
      if (h_out[b] > 0)
         std::printf("    n_out = %2d : %9ld  (%6.2f %%)\n", b, h_out[b],
                     100.0 * (double)h_out[b] / (double)ncol);
   std::printf("    Picard/column  min = %d  max = %d  mean = %8.2f\n", in_mn, in_mx,
               (double)in_tot / (double)ncol);
   std::printf("    columns that mixed at all: %ld  (%6.2f %%)\n", nactive,
               100.0 * (double)nactive / (double)ncol);
   if (h_out[1] == (long)ncol)
      std::printf("    NOTE: n_out = 1 for 100%% of columns -- the adaptive substepper\n"
                  "          never engages. Inherited from ks_bench's state: it matches\n"
                  "          production's MEDIAN runtime but not its 9x variance.\n");

   double kd_min = kd[0], kd_max = kd[0], kd_sum = 0.0;
   for (size_t m = 0; m < nI; ++m) {
      kd_min = std::fmin(kd_min, kd[m]);
      kd_max = std::fmax(kd_max, kd[m]);
      kd_sum += kd[m];
   }
   std::printf("\n  min kd_int : %14.6e\n", kd_min);
   std::printf("  max kd_int : %14.6e\n", kd_max);
   std::printf("  sum kd_int : %14.6e   (ks_bench prints 3.299882E+05 for DC)\n", kd_sum);
   if (kd_min != kd_min || kd_max != kd_max)
      std::printf("  sanity     : *** NaN -- results are garbage ***\n");
   else if (kd_min == 0.0 && kd_max == 0.0)
      std::printf("  sanity     : *** all-zero -- NO COLUMN MIXED ***\n");
   else
      std::printf("  sanity     : OK (finite, non-zero)\n");

   bool verified = true;
   if (ref.loaded) {
      const bool driver_ok = check_driver(ref, kd, nI);

      // The ulp bound above says the mirror's NUMBERS are right. It does not
      // say the kernel does the same WORK on them -- and work is what the
      // timing measures. This scheme iterates to convergence, so a 1-ulp input
      // difference could in principle flip a convergence test and change a
      // column's iteration count. Whether it does is a question, not a given,
      // so: re-run on the mirror's inputs and compare the Picard totals. Equal
      // totals => the timed native run does exactly the work the Fortran timed.
      CUDA_OK(cudaMemcpy(d_h, mirror.h_layer.data(), mirror.h_layer.size() * sizeof(double), cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(d_u, mirror.u_face.data(), mirror.u_face.size() * sizeof(double), cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(d_v, mirror.v_face.data(), mirror.v_face.size() * sizeof(double), cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(d_hT, mirror.hT.data(), mirror.hT.size() * sizeof(double), cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(d_hS, mirror.hS.data(), mirror.hS.size() * sizeof(double), cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(d_wet, mirror.wet_mask.data(), mirror.wet_mask.size() * sizeof(double), cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(d_fc, mirror.f_centre.data(), mirror.f_centre.size() * sizeof(double), cudaMemcpyHostToDevice));
      launch(1);
      std::vector<int> nin_m(ncol);
      std::vector<double> kd_m(nI);
      CUDA_OK(cudaMemcpy(nin_m.data(), d_nin, ncol * sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(kd_m.data(), d_kd, nI * sizeof(double), cudaMemcpyDeviceToHost));

      long tot_m = 0, ncols_diff = 0;
      for (size_t m = 0; m < ncol; ++m) {
         tot_m += nin_m[m];
         if (nin_m[m] != nin[m]) ncols_diff++;
      }
      const Cmp ckm = compare(kd_m.data(), ref.kd_cu.data(), nI);
      std::printf("\n  [RISK B, cont.] does the mirror make the kernel do the SAME WORK?\n");
      std::printf("    Picard iterations, total : %ld (mirror) vs %ld (Fortran bits)\n",
                  tot_m, in_tot);
      std::printf("    columns with a different iteration count : %ld / %zu\n",
                  ncols_diff, ncol);
      std::printf("    mirror-state kd vs CUDA-via-Fortran      : max rel = %.3e\n",
                  ckm.rmax);
      if (tot_m == in_tot && ncols_diff == 0)
         std::printf("    -> identical work. The libm ulp noise does NOT flip any\n"
                     "       convergence test, so the native timing below is timing\n"
                     "       the same problem the Fortran driver times.\n");
      else
         std::printf("    -> ⚠ WORK DIFFERS. The ulp noise flips convergence tests, so\n"
                     "       native vs Fortran timings are NOT strictly comparable.\n"
                     "       Quantify (%.4f %% of iterations) before trusting a ratio.\n",
                     100.0 * (double)(tot_m - in_tot) / (double)in_tot);

      verified = driver_ok && mirror_ok;
      std::printf("\n  verdict: %s\n",
                  verified ? "PASS -- driver reproduces the Fortran run bit-for-bit,"
                             " and the mirror is faithful to a few ulp."
                           : "*** FAIL -- see above ***");
   } else {
      std::printf("\n  KS_REF unset -- results NOT checked against the Fortran run.\n");
      std::printf("  Run `make native-verify` to do that. The sum above alone is a\n");
      std::printf("  weak check: it is a reduction, and reductions hide sign-paired\n");
      std::printf("  and permuted errors.\n");
   }
   std::printf("%s\n", std::string(74, '=').c_str());

   // ---- the ONE machine-readable line tools/ks_sweep.sh parses -------------
   // Same key=value contract as dc_main.F90's RESULT line, so DC and CUDA rows
   // land in one table. `ms` is the SAME statistic on both sides (total/reps).
   // The counters are always on in this driver (the CUDA kernel writes n_out /
   // n_in unconditionally), hence counters=1.
   std::printf("RESULT kernel=kappa_shear variant=%s lanes=1 data=cuda"
               " nzstack=%d bcopy=%d counters=1"
               " nxp=%d nyp=%d nx=%d ny=%d ncols=%zu nwet=%ld nz=%d"
               " reps=%d warm=%d land_pct=%d"
               " ms=%.10E ns_per_col=%.10E"
               " ms_min=%.10E ms_med=%.10E ms_max=%.10E"
               " kd_sum=%.10E kd_min=%.10E kd_max=%.10E"
               " it_outer=%ld it_inner=%ld regs=%d local_b=%d\n",
               KS_CUVARIANT, MODEL_NZ_STACK_MAX, KS_BCOPY_EQUIV, nxp, nyp, nx, ny, ncol, nwet, nz,
               n_reps, n_warm, land_pct,
               ms_mean, ms_mean * 1.0e6 / (double)ncol,
               ms_min, ms_med, ms_max,
               kd_sum, kd_min, kd_max,
               out_tot, in_tot, nregs, lmem);

   cudaFree(d_h); cudaFree(d_u); cudaFree(d_v); cudaFree(d_hT); cudaFree(d_hS);
   cudaFree(d_wet); cudaFree(d_fc); cudaFree(d_kd); cudaFree(d_tke);
   cudaFree(d_nout); cudaFree(d_nin);

   return verified ? 0 : 1;
}
