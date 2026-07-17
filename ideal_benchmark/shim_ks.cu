// shim_ks.cu -- flat extern "C" wrapper around kappa_shear/opt_kernel.cu's
// ks_opt_launch, so the Fortran rk2_kappa wrapper can call one uniform
// pointer-list bind(C) (like btstep/meke's *_launch_flat) instead of building
// the KsPar struct + the n_out/n_in device counters itself.
//
// KsPar is a local mirror of the struct in kappa_shear/opt_kernel.cu (field
// order + types identical -- it is passed by pointer and read back as that
// type, so the byte layout must match). The knob VALUES are the fixed
// ocean_kappa_shear_t defaults + the gabight overrides, copied verbatim from
// kappa_shear/drivers/cpp_main.cu:504. dt is the only run-time argument.
#include <cuda_runtime.h>

#define EOS_VARIANT_LINEAR 1

struct KsPar {
   double dt, ri_crit, shearmix_rate, fri_curvature;
   double c_n, c_s, lambda, lz_rescale;
   double kappa_0, kappa_seed, kappa_trunc, tke_bg;
   double tol_err, src_max_chg, vel_underflow, rho0;
   int max_inner_it, max_substep_it;
   int eos_variant;
   double eos_rho0, eos_alpha_T, eos_beta_S;
};

extern "C" void ks_opt_launch(const double *h_layer, const double *u_face,
                              const double *v_face, const double *hT,
                              const double *hS, const double *wet_mask,
                              const double *f_centre, double *kd_int,
                              double *tke_int, int *n_out_a, int *n_in_a,
                              int nx, int ny, int nz, const KsPar *pin, int sync);
extern "C" void ks_cuda_launch(const double *h_layer, const double *u_face,
                               const double *v_face, const double *hT,
                               const double *hS, const double *wet_mask,
                               const double *f_centre, double *kd_int,
                               double *tke_int, int *n_out_a, int *n_in_a,
                               int nx, int ny, int nz, const KsPar *pin, int sync);

static void fill_kspar(KsPar &p, double dt) {
   p.dt = dt;            p.ri_crit = 0.25;   p.shearmix_rate = 0.089; p.fri_curvature = -0.97;
   p.c_n = 0.24;         p.c_s = 0.14;       p.lambda = 0.82;         p.lz_rescale = 1.0;
   p.kappa_0 = 1.0e-7;   p.kappa_seed = 1.0; p.kappa_trunc = 1.0e-9;  p.tke_bg = 0.0;
   p.tol_err = 0.1;      p.src_max_chg = 10.0; p.vel_underflow = 0.0; p.rho0 = 1035.0;
   p.max_inner_it = 50;  p.max_substep_it = 13;
   p.eos_variant = EOS_VARIANT_LINEAR;
   p.eos_rho0 = 1035.0;  p.eos_alpha_T = 0.2; p.eos_beta_S = 7.6e-4;
}

static int *g_n_out = nullptr, *g_n_in = nullptr;
static long g_cap = 0;
static void ensure_counters(long ncol) {
   if (ncol > g_cap) {
      if (g_n_out) cudaFree(g_n_out);
      if (g_n_in)  cudaFree(g_n_in);
      cudaMalloc(&g_n_out, ncol * sizeof(int));
      cudaMalloc(&g_n_in,  ncol * sizeof(int));
      g_cap = ncol;
   }
}

extern "C" void ks_opt_flat(const double *h, const double *u, const double *v,
                            const double *hT, const double *hS, const double *wet,
                            const double *fc, double *kd, double *tke,
                            int nx, int ny, int nz, double dt, int sync)
{
   ensure_counters((long)nx * (long)ny);
   KsPar p; fill_kspar(p, dt);
   ks_opt_launch(h, u, v, hT, hS, wet, fc, kd, tke, g_n_out, g_n_in, nx, ny, nz, &p, sync);
}

extern "C" void ks_cuda_flat(const double *h, const double *u, const double *v,
                             const double *hT, const double *hS, const double *wet,
                             const double *fc, double *kd, double *tke,
                             int nx, int ny, int nz, double dt, int sync)
{
   ensure_counters((long)nx * (long)ny);
   KsPar p; fill_kspar(p, dt);
   ks_cuda_launch(h, u, v, hT, hS, wet, fc, kd, tke, g_n_out, g_n_in, nx, ny, nz, &p, sync);
}
