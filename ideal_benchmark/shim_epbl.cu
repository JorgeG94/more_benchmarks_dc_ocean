// shim_epbl.cu -- flat extern "C" wrapper around epbl/opt_kernel.cu's
// epbl_opt_launch, so the Fortran rk2_epbl wrapper calls one uniform
// pointer-list bind(C) instead of building the (large) EpblParams struct + the
// its_used device counter itself.
//
// EpblParams comes from the SHARED header epbl/epbl_params.h (the same struct
// the kernel is compiled against -- no hand-mirrored copy). The knob VALUES are
// the EPBL defaults copied verbatim from epbl/drivers/cpp_main.cu:fill_params;
// only inv_rho0_cp / inv_rho0 / dt / nx,ny,nz / mld_max_its vary at run time.
#include <cuda_runtime.h>
#include "epbl_params.h"   // struct EpblParams (+ epbl_cuda_launch decl, unused)

// scheme enum values (epbl/opt_kernel.cu + ocean_epbl.F90)
#define EPBL_MSTAR_OM4       2
#define EPBL_VSTAR_CUBE_ROOT 1
#define EPBL_COMBINE_ADD     1
#define EPBL_LT_RESCALE      1
#define EOS_VARIANT_LINEAR   1

#define EPBL_LB_THREADS_DEFAULT 128

// opt launcher (opt_kernel.cu) + faithful launcher (epbl_kernel.cu, +variant/threads).
extern "C" void epbl_opt_launch(
   const double *h_layer, const double *wet_mask, const double *hT, const double *hS,
   const double *tau_x, const double *tau_y, const double *Q_heat, const double *Q_salt,
   const double *f_centre, double *mld, double *kd_int, double *la, double *t0, double *s0,
   double *dpe_t, double *dpe_s, double *dcolht_t, double *dcolht_s,
   double *tke_wind, double *tke_conv, double *tke_forcing, double *tke_mixing,
   double *tke_mech_decay, double *tke_conv_decay,
   int *its_used, const EpblParams *Pin, int sync);
extern "C" void epbl_cuda_launch(
   const double *h_layer, const double *wet_mask, const double *hT, const double *hS,
   const double *tau_x, const double *tau_y, const double *Q_heat, const double *Q_salt,
   const double *f_centre, double *mld, double *kd_int, double *la, double *t0, double *s0,
   double *dpe_t, double *dpe_s, double *dcolht_t, double *dcolht_s,
   double *tke_wind, double *tke_conv, double *tke_forcing, double *tke_mixing,
   double *tke_mech_decay, double *tke_conv_decay,
   int *its_used, const EpblParams *Pin, int variant, int threads, int sync);

static void fill_epbl(EpblParams &P, double inv_rho0_cp, double inv_rho0, double dt,
                      int nx, int ny, int nz, int mld_max_its) {
   P.mstar_scheme = EPBL_MSTAR_OM4; P.vstar_scheme = EPBL_VSTAR_CUBE_ROOT; P.combine_mode = EPBL_COMBINE_ADD;
   P.mstar_const = 1.2; P.mstar_cap = -1.0; P.mstar_coef1 = 0.3;
   P.c_ek = 0.085; P.mstar_conv_adj = 0.0;
   P.rh18_cn1 = 0.275; P.rh18_cn2 = 8.0; P.rh18_cn3 = -5.0; P.rh18_cs1 = 0.2; P.rh18_cs2 = 0.4;
   P.nstar = 0.2; P.tke_decay = 2.5; P.wstar_ustar_coef = 1.0;
   P.vstar_scale_fac = 1.0; P.vstar_surf_fac = 1.2;
   P.von_karman = 0.41; P.ekman_scale_coef = 1.0; P.min_mix_len = 0.0;
   P.mixlen_exponent = 2.0; P.translay_scale = 0.1;
   P.mld_iteration = 1; P.mld_max_its = mld_max_its; P.mld_bisection = 0; P.mld_use_prev_guess = 0;
   P.mld_tol = 1.0;
   P.rho0 = 1035.0; P.omega = 7.2921e-5; P.omega_frac = 0.0;
   P.ustar_min = 1.0e-8; P.prandtl = 1.0;
   P.tke_diags = 0; P.use_lt = 0; P.lt_scheme = EPBL_LT_RESCALE;
   P.lt_enhance_coef = 0.447; P.lt_enhance_exp = -1.33;
   P.lt_max_enhance = 5.0; P.la_frac_hbl = 0.04;
   P.lt_lac1 = -0.87; P.lt_lac2 = 0.0; P.lt_lac3 = 0.0; P.lt_lac4 = 0.95; P.lt_lac5 = 0.95;
   P.eos_variant = EOS_VARIANT_LINEAR;
   P.eos_rho0 = 1035.0; P.eos_T_ref = 10.0; P.eos_S_ref = 35.0;
   P.eos_alpha_T = 0.2; P.eos_beta_S = 7.6e-4; P.eos_p_ref = 0.0;
   P.inv_rho0_cp = inv_rho0_cp; P.inv_rho0 = inv_rho0; P.dt = dt;
   P.nx = nx; P.ny = ny; P.nz = nz;
}

static int *g_its = nullptr;
static long g_cap = 0;
static void ensure_its(long ncol) {
   if (ncol > g_cap) { if (g_its) cudaFree(g_its); cudaMalloc(&g_its, ncol * sizeof(int)); g_cap = ncol; }
}

extern "C" void epbl_opt_flat(
   const double *h_layer, const double *wet_mask, const double *hT, const double *hS,
   const double *tau_x, const double *tau_y, const double *Q_heat, const double *Q_salt,
   const double *f_centre, double *mld, double *kd_int, double *la, double *t0, double *s0,
   double *dpe_t, double *dpe_s, double *dcolht_t, double *dcolht_s,
   double *tke_wind, double *tke_conv, double *tke_forcing, double *tke_mixing,
   double *tke_mech_decay, double *tke_conv_decay,
   double inv_rho0_cp, double inv_rho0, double dt,
   int nx, int ny, int nz, int mld_max_its, int sync)
{
   ensure_its((long)nx * (long)ny);
   EpblParams P; fill_epbl(P, inv_rho0_cp, inv_rho0, dt, nx, ny, nz, mld_max_its);
   epbl_opt_launch(h_layer, wet_mask, hT, hS, tau_x, tau_y, Q_heat, Q_salt, f_centre,
                   mld, kd_int, la, t0, s0, dpe_t, dpe_s, dcolht_t, dcolht_s,
                   tke_wind, tke_conv, tke_forcing, tke_mixing, tke_mech_decay, tke_conv_decay,
                   g_its, &P, sync);
}

extern "C" void epbl_cuda_flat(
   const double *h_layer, const double *wet_mask, const double *hT, const double *hS,
   const double *tau_x, const double *tau_y, const double *Q_heat, const double *Q_salt,
   const double *f_centre, double *mld, double *kd_int, double *la, double *t0, double *s0,
   double *dpe_t, double *dpe_s, double *dcolht_t, double *dcolht_s,
   double *tke_wind, double *tke_conv, double *tke_forcing, double *tke_mixing,
   double *tke_mech_decay, double *tke_conv_decay,
   double inv_rho0_cp, double inv_rho0, double dt,
   int nx, int ny, int nz, int mld_max_its, int sync)
{
   ensure_its((long)nx * (long)ny);
   EpblParams P; fill_epbl(P, inv_rho0_cp, inv_rho0, dt, nx, ny, nz, mld_max_its);
   // variant 0 = epbl_faithful; threads = block size for the faithful kernel.
   epbl_cuda_launch(h_layer, wet_mask, hT, hS, tau_x, tau_y, Q_heat, Q_salt, f_centre,
                    mld, kd_int, la, t0, s0, dpe_t, dpe_s, dcolht_t, dcolht_s,
                    tke_wind, tke_conv, tke_forcing, tke_mixing, tke_mech_decay, tke_conv_decay,
                    g_its, &P, 0, EPBL_LB_THREADS_DEFAULT, sync);
}
