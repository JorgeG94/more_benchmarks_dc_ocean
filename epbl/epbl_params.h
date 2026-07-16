// Shared ABI between epbl_kernel.cu, epbl_bench.F90's `epbl_params_t`, and
// epbl_native.cu.
//
// WHY THIS IS A HEADER AND NOT A THIRD COPY: `EpblParams` is passed BY VALUE
// into the kernel, so it is a raw struct ABI. A field reordered in one copy and
// not the other is not a link error -- it is silent numerical corruption that
// looks like a port bug. epbl_bench.F90 already carries a bind(C) mirror it
// must hand-maintain (see the comment on `epbl_params_t`); the native driver
// does NOT get to add a third, so the struct lives here and both C++ TUs
// include it.
//
// If you change this struct you MUST update epbl_bench.F90:epbl_params_t to
// match, field for field, in order.
#ifndef EPBL_PARAMS_H
#define EPBL_PARAMS_H

// Mirror of `ocean_epbl_t`'s knobs + the `ocean_eos_t` handle. Passed BY VALUE,
// exactly as production passes `this` (whose knobs are all scalars) and `eos`.
struct EpblParams {
   int mstar_scheme, vstar_scheme, combine_mode;
   double mstar_const, mstar_cap, mstar_coef1, c_ek, mstar_conv_adj;
   double rh18_cn1, rh18_cn2, rh18_cn3, rh18_cs1, rh18_cs2;
   double nstar, tke_decay, wstar_ustar_coef, vstar_scale_fac, vstar_surf_fac;
   double von_karman, ekman_scale_coef, min_mix_len, mixlen_exponent, translay_scale;
   int mld_iteration, mld_max_its, mld_bisection, mld_use_prev_guess;
   double mld_tol;
   double rho0, omega, omega_frac, ustar_min, prandtl;
   int tke_diags;
   int use_lt, lt_scheme;
   double lt_enhance_coef, lt_enhance_exp, lt_max_enhance, la_frac_hbl;
   double lt_lac1, lt_lac2, lt_lac3, lt_lac4, lt_lac5;
   // eos handle (flat POD, by value -- ocean_eos.F90:807-810)
   int eos_variant;
   double eos_rho0, eos_T_ref, eos_S_ref, eos_alpha_T, eos_beta_S, eos_p_ref;
   // shim-precomputed multipliers (ocean_epbl.F90:809-815)
   double inv_rho0_cp, inv_rho0, dt;
   int nx, ny, nz;
};

// The ONE launch entry point. Both the Fortran driver (via bind(C) +
// `!$acc host_data use_device`) and epbl_native.cu (via cudaMalloc) call THIS
// -- there is exactly one copy of the kernel in the repo and both drivers run
// it. variant 0 = epbl_faithful, 1 = epbl_tuned (__launch_bounds__).
extern "C" void epbl_cuda_launch(const double *h_layer, const double *wet_mask, const double *hT,
                                 const double *hS, const double *tau_x, const double *tau_y,
                                 const double *Q_heat, const double *Q_salt, const double *f_centre,
                                 double *mld, double *kd_int, double *la, double *t0, double *s0,
                                 double *dpe_t, double *dpe_s, double *dcolht_t, double *dcolht_s,
                                 double *tke_wind, double *tke_conv, double *tke_forcing,
                                 double *tke_mixing, double *tke_mech_decay, double *tke_conv_decay,
                                 int *its_used, const EpblParams *Pin, int variant, int threads,
                                 int sync);

#endif // EPBL_PARAMS_H
