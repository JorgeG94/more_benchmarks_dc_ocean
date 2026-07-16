// ============================================================================
// CUDA C port of the ocean model's `epbl_column_kernel`
// (<model>/src/parameterizations/vertical/structured/ocean_epbl.F90:818-1340,
//  md5 33e307f821c1729434f05fbc8b33550e).
//
// Two variants, deliberately answering two different questions:
//
//   epbl_faithful   ONE THREAD PER COLUMN, block 128. Structurally identical to
//                   what nvfortran generates for the production `do concurrent
//                   (j=1:ny, i=1:nx)` (confirmed by -Minfo: "Loop parallelized
//                   across CUDA thread blocks, CUDA threads(128) collapse(2)").
//                   Same loops, same order, same branches, same arithmetic
//                   association. It isolates CODEGEN and nothing else.
//
//   epbl_tuned      Identical body, but with __launch_bounds__ and a caller-
//                   chosen block size. This is the one thing `do concurrent`
//                   structurally CANNOT express: occupancy/register control.
//                   If the faithful port ties and this one wins, the lever is
//                   the launch configuration, not the language.
//
// WHY THERE IS NO WARP-PER-COLUMN VARIANT: see README.md. Short version -- the
// sweep is a first-order sequential recursion in k (hp_a, dpe_*_a, th_a, te_lag
// each depend on the previous interface). There is no per-k parallel work and
// no reduction, so there is nothing for 32 lanes to do and nothing for __shfl
// to reduce. The prep loop's `pres` IS a prefix scan, but it is nz=30 of ~630
// sequential steps. This was checked against the source, not assumed.
//
// FMA: nvcc contracts by default, as does nvfortran -O3 -fast. Both are left at
// their defaults -- forcing them apart would measure the flag, not the compiler.
// Expect ~1e-15 relative agreement, not bitwise.
//
// Fortran is column-major and 1-based; the index macros keep the 1-based
// algebra so the body reads against the Fortran line-for-line.
// ============================================================================
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define IDX2(i, j, n1) ((size_t)((i) - 1) + (size_t)(n1) * (size_t)((j) - 1))
#define IDX3(i, j, k, n1, n2) \
   ((size_t)((i) - 1) + (size_t)(n1) * ((size_t)((j) - 1) + (size_t)(n2) * (size_t)((k) - 1)))

// ---- scheme tags: verbatim from ocean_epbl.F90:62-88 -------------------
#define EPBL_MSTAR_CONSTANT 1
#define EPBL_MSTAR_OM4      2
#define EPBL_MSTAR_RH18     3
#define EPBL_VSTAR_CUBE_ROOT 1
#define EPBL_VSTAR_RH18      2
#define EPBL_LT_NONE     0
#define EPBL_LT_RESCALE  1
#define EPBL_LT_ADDITIVE 2
// ---- ocean_eos.F90:33-46 ----
#define EOS_VARIANT_LINEAR     1
#define EOS_VARIANT_WRIGHT_97  2
#define EOS_VARIANT_ROQUET_SPV 4

// ---- constants.F90 ----
__constant__ double c_dummy_unused;
#define GRAVITY   9.80665
#define H_NEGLECT 1.0e-20    // = H_DIV_EPS (ocean_epbl.F90:96)

// ---- LF17 / COARE 3.5 constants: verbatim ocean_epbl.F90:103-125 -------
#define LT_VONKAR_WAVES   0.40
#define LT_NU_AIR         1.0e-6
#define LT_RHO_AIR        1.225
#define LT_CHARNOCK_MIN   0.028
#define LT_CHARNOCK_SLOPE 0.0017
#define LT_CHARNOCK_ICPT  (-0.005)
#define LT_US_TO_U10      0.0162
#define LT_SWH_FROM_U10SQ 0.0246
#define LT_U19P5_TO_U10   1.075
#define LT_FM_INTO_FP     1.296
#define LT_R_LOSS         0.667
#define LT_PI             3.14159265358979323846

// ---- Wright (1997) Table A1: verbatim ocean_eos.F90:77-91 --------------
#define WRIGHT_A1 3.480336e-7
#define WRIGHT_A2 (-1.112733e-7)
#define WRIGHT_B0 5.790749e8
#define WRIGHT_B1 3.516535e6
#define WRIGHT_B2 (-4.002714e4)
#define WRIGHT_B3 2.084372e2
#define WRIGHT_B4 5.944068e5
#define WRIGHT_B5 (-9.643486e3)
#define WRIGHT_C0 1.704853e5
#define WRIGHT_C1 7.904722e2
#define WRIGHT_C2 (-7.984422e0)
#define WRIGHT_C3 5.140652e-2
#define WRIGHT_C4 (-2.302158e2)
#define WRIGHT_C5 (-3.079464e0)

// `EpblParams` (the by-value knob/EOS mirror) and the epbl_cuda_launch
// prototype live in epbl_params.h, because epbl_native.cu needs the identical
// struct ABI to build P. It was moved OUT of this file rather than copied INTO
// that one: the struct is passed by value, so two copies that drift is silent
// corruption, not a link error.
#include "epbl_params.h"

// Bare device pointers to the SAME allocations the OpenACC runtime made.
struct EpblPtrs {
   const double * __restrict__ h_layer, * __restrict__ wet_mask, * __restrict__ hT, * __restrict__ hS, * __restrict__ tau_x, * __restrict__ tau_y, * __restrict__ Q_heat, * __restrict__ Q_salt, * __restrict__ f_centre;
   double * __restrict__ mld, * __restrict__ kd_int, * __restrict__ la;
   double * __restrict__ t0, * __restrict__ s0, * __restrict__ dpe_t, * __restrict__ dpe_s, * __restrict__ dcolht_t, * __restrict__ dcolht_s;
   double * __restrict__ tke_wind, * __restrict__ tke_conv, * __restrict__ tke_forcing, * __restrict__ tke_mixing, * __restrict__ tke_mech_decay, * __restrict__ tke_conv_decay;
   int *its_used;   // diagnostic only; NULL disables (never written in timed runs)
};

// ===========================================================================
// Device helpers -- transcribed 1:1 from the `!$acc routine seq` originals.
// ===========================================================================

// ocean_eos.F90:800-864
__device__ __forceinline__ void eos_specvol_derivs(const EpblParams &P, double T, double S,
                                                   double p, double *dsv_dt, double *dsv_ds) {
   if (P.eos_variant == EOS_VARIANT_ROQUET_SPV) {
      *dsv_dt = 0.0;  // stub branch: dead at eos="linear" (see epbl_stubs.F90)
      *dsv_ds = 0.0;
   } else if (P.eos_variant == EOS_VARIANT_WRIGHT_97) {
      double T_sq = T * T;
      double p_0 = WRIGHT_B0 + WRIGHT_B1 * T + WRIGHT_B2 * T_sq + WRIGHT_B3 * T_sq * T +
                   WRIGHT_B4 * S + WRIGHT_B5 * S * T;
      double lambda = WRIGHT_C0 + WRIGHT_C1 * T + WRIGHT_C2 * T_sq + WRIGHT_C3 * T_sq * T +
                      WRIGHT_C4 * S + WRIGHT_C5 * S * T;
      double dp0_dt = WRIGHT_B1 + 2.0 * WRIGHT_B2 * T + 3.0 * WRIGHT_B3 * T_sq + WRIGHT_B5 * S;
      double dlam_dt = WRIGHT_C1 + 2.0 * WRIGHT_C2 * T + 3.0 * WRIGHT_C3 * T_sq + WRIGHT_C5 * S;
      double dp0_ds = WRIGHT_B4 + WRIGHT_B5 * T;
      double dlam_ds = WRIGHT_C4 + WRIGHT_C5 * T;
      double big_p = p + p_0;
      double inv_p = 1.0 / big_p;
      double inv_p2 = inv_p * inv_p;
      *dsv_dt = WRIGHT_A1 + dlam_dt * inv_p - lambda * dp0_dt * inv_p2;
      *dsv_ds = WRIGHT_A2 + dlam_ds * inv_p - lambda * dp0_ds * inv_p2;
   } else {
      *dsv_dt = P.eos_alpha_T / (P.eos_rho0 * P.eos_rho0);
      *dsv_ds = -P.eos_beta_S / (P.eos_rho0 * P.eos_rho0);
   }
}

// ocean_epbl.F90:545-603
__device__ __forceinline__ void epbl_find_mstar(const EpblParams &P, double b0, double ustar,
                                                double bld, double absf, double *mstar_out) {
   double mstar, mstar_n, mstar_s, msn_term, absf_floor, mscr_t1, mscr_t2;
   absf_floor = fmax(absf, 1.0e-20);
   switch (P.mstar_scheme) {
   case EPBL_MSTAR_OM4:
      mstar_s = P.mstar_coef1 * sqrt(fmax(0.0, b0) / (ustar * ustar * absf_floor));
      mstar_n = 0.0;
      if (ustar > absf_floor * bld) {
         mstar_n = P.c_ek * log(ustar / (absf_floor * fmax(bld, 1.0e-10)));
      }
      mstar = fmax(mstar_s, fmin(1.25, mstar_n));
      if (P.mstar_cap > 0.0) mstar = fmin(P.mstar_cap, mstar);
      break;
   case EPBL_MSTAR_RH18: {
      // `**2` / `**5` are INTEGER exponents in Fortran -> multiplies, not pow.
      // `**cs2` is a REAL exponent -> pow. (Dead at mstar_scheme = om4.)
      double b0p = fmax(0.0, b0);
      double us2 = ustar * ustar;
      double us5 = us2 * us2 * ustar;
      msn_term = P.rh18_cn2 * exp(P.rh18_cn3 * bld * absf / ustar);
      mstar_n = P.rh18_cn1 * msn_term / (1.0 + msn_term);
      mstar_s = P.rh18_cs1 * pow(b0p * b0p * bld / (us5 * absf_floor), P.rh18_cs2);
      mstar = mstar_n + mstar_s;
      if (P.mstar_cap > 0.0) mstar = fmin(P.mstar_cap, mstar);
      break;
   }
   default:  // EPBL_MSTAR_CONSTANT
      mstar = P.mstar_const;
      break;
   }
   if (P.mstar_conv_adj > 0.0) {
      mscr_t1 = -bld * fmin(0.0, b0);
      mscr_t2 = 2.0 * mstar * ustar * ustar * ustar;
      if (mscr_t2 > 0.0) {
         mstar = mstar * ((1.0 - P.mstar_conv_adj) * mscr_t1 + mscr_t2) / (mscr_t1 + mscr_t2);
      } else {
         mstar = mstar * (1.0 - P.mstar_conv_adj);
      }
   }
   *mstar_out = mstar;
}

// ocean_epbl.F90:605-629
__device__ __forceinline__ double epbl_mixlen_shape(double z_depth, double mld_guess,
                                                    double translay_scale, double mixlen_exponent,
                                                    int shaped) {
   double fr;
   if ((!shaped) || translay_scale >= 1.0 || translay_scale < 0.0 || mld_guess <= 0.0) {
      return 1.0;
   }
   fr = fmax(0.0, (mld_guess - z_depth) / mld_guess);
   if (mixlen_exponent == 2.0) {
      return translay_scale + (1.0 - translay_scale) * fr * fr;
   }
   return translay_scale + (1.0 - translay_scale) * pow(fr, mixlen_exponent);
}

// ocean_epbl.F90:719-729
__device__ __forceinline__ double one_m_exp_x(double x) {
   if (x < 1.0e-4) return 1.0 - x * (0.5 - x / 6.0);
   return (1.0 - exp(-x)) / x;
}

// ocean_epbl.F90:631-680
__device__ __forceinline__ void epbl_lf17_wave_state(double ustar_w, double rho_ocn, double *u10_o,
                                                     double *ustokes_o, double *kphil_o) {
   double ustar_air, z0_smooth, z0w, alpha_ch, u10_new, u10;
   double hm0, f_mean, vstokes;
   int converged = 0;
   ustar_air = ustar_w * sqrt(rho_ocn / LT_RHO_AIR);
   z0_smooth = 0.11 * LT_NU_AIR / ustar_air;
   u10 = ustar_air * sqrt(1000.0);
   for (int itt = 1; itt <= 20; ++itt) {
      alpha_ch = fmin(LT_CHARNOCK_MIN, LT_CHARNOCK_SLOPE * u10 + LT_CHARNOCK_ICPT);
      z0w = fmax(z0_smooth + alpha_ch * ustar_air * ustar_air / GRAVITY, 1.0e-10);
      u10_new = ustar_air * log(10.0 / z0w) / LT_VONKAR_WAVES;
      if (fabs(u10_new - u10) <= 1.0e-3 * u10) {
         u10 = u10_new;
         converged = 1;
         break;
      }
      u10 = u10_new;
   }
   if (!converged) u10 = 25.82 * ustar_air;
   *ustokes_o = LT_US_TO_U10 * u10;
   hm0 = LT_SWH_FROM_U10SQ * u10 * u10;
   f_mean = LT_FM_INTO_FP * 0.877 * GRAVITY / (2.0 * LT_PI * LT_U19P5_TO_U10 * u10);
   vstokes = 0.125 * LT_PI * LT_R_LOSS * f_mean * hm0 * hm0;
   *kphil_o = 0.176 * (*ustokes_o) / fmax(vstokes, 1.0e-30);
   *u10_o = u10;
}

// ocean_epbl.F90:682-717
__device__ __forceinline__ double epbl_lf17_la(double ustar_w, double zsl, double ustokes,
                                               double kphil) {
   double xkz, root2kz, r1, r3, r5, ustokes_sl;
   xkz = kphil * zsl;
   root2kz = sqrt(2.0 * xkz);
   r1 = (0.302 - 1.68 * xkz) * one_m_exp_x(2.0 * xkz);
   r3 = (0.1264 + 0.64 * xkz) * one_m_exp_x(5.12 * xkz);
   if (root2kz > 1.0e-3) {
      r5 = sqrt(LT_PI) * (root2kz * (-0.84 * erfc(root2kz) + 0.2 * erfc(1.6 * root2kz)) +
                          0.1182 * (erfc(1.6 * root2kz) - erfc(root2kz)) / root2kz);
   } else {
      r5 = -0.64 * sqrt(LT_PI) * root2kz + (-0.14184 + 1.0839648 * root2kz * root2kz);
   }
   ustokes_sl = fmax(ustokes * (0.715 + r1 + r3 + r5), 1.0e-10);
   return sqrt(ustar_w / ustokes_sl);
}

// ocean_epbl.F90:731-771
__device__ __forceinline__ void epbl_lt_enhance(const EpblParams &P, double la, double b0,
                                                double ustar, double bld, double absf,
                                                double *mstar) {
   double mld_ek, ek_ob, mld_ob, la_mod, enh;
   mld_ek = bld * absf / ustar;
   ek_ob = fabs(b0 * P.von_karman) / (fmax(absf, 1.0e-20) * ustar * ustar);
   mld_ob = fabs(bld * b0 * P.von_karman) / (ustar * ustar * ustar);
   if (b0 >= 0.0) {
      la_mod = la * ((1.0 + fmax(-0.5, P.lt_lac1 * mld_ek)) + P.lt_lac4 * ek_ob + P.lt_lac2 * mld_ob);
   } else {
      la_mod = la * ((1.0 + fmax(-0.5, P.lt_lac1 * mld_ek)) + P.lt_lac5 * ek_ob + P.lt_lac3 * mld_ob);
   }
   la_mod = fmax(la_mod, 1.0e-10);
   if (P.lt_scheme == EPBL_LT_ADDITIVE) {
      *mstar = *mstar + P.lt_enhance_coef * pow(la_mod, P.lt_enhance_exp);
   } else {
      enh = fmin(P.lt_max_enhance, 1.0 + P.lt_enhance_coef * pow(la_mod, P.lt_enhance_exp));
      *mstar = *mstar * enh;
   }
}

// ===========================================================================
// The column solve. ONE THREAD = ONE COLUMN, mirroring the `do concurrent
// (j=1:ny, i=1:nx)` body of ocean_epbl.F90:905-1339 statement for
// statement. `__forceinline__` so both launchers compile the identical body.
// ===========================================================================
__device__ __forceinline__ void epbl_column_body(const EpblParams &P, const EpblPtrs &A, int i,
                                                 int j) {
   const int nx = P.nx, ny = P.ny, nz = P.nz;
   const double dt = P.dt;

   int k;
   double q_t_kin, q_s_kin;
   double hk, inv_h, t0k, s0k, dmass, dpres, p_mid, pres;
   double dsv_dt_k, dsv_ds_k, dsv_dt_sfc, dsv_ds_sfc, h_sum;
   double tau_xc, tau_yc, ustar, absf, idecay, mech_in;
   double b0, ctke_sfc;
   int obl_it, n_its;
   double min_mld, max_mld, mld_guess, mld_found, dmld_min, dmld_max;
   int have_min, have_max;
   double mstar_val, mech_tke, conv_perel, forcing_clip;
   int ki, ka, kb;
   double htot, z_int, pres_int, mld_output;
   int sfc_connected, sfc_disconnect;
   double hp_a, dpe_t_a, dpe_s_a, dch_t_a, dch_s_a;
   double th_a, sh_a, te_lag, se_lag, kddt_prev, kddt_cur;
   double exp_kh, nstar_fc, tot_tke, tke_here, vstar;
   double h_ka, h_kb, hb_hs, shape_fn, hbs, mixlen, kd_g0;
   double hp_b, th_b, sh_b, hps, bdt1, dt_c, ds_c;
   double pec_core, colht_core, dkddt, pe_max;
   double kd_val, tke_used, frac_bl, pe_g0, dpe_conv;
   double b1, c1, r_reduc, te_new, se_new, surf_scale;
   double lt_u10, lt_ustokes, lt_kphil, la_val;
   double d_wind, d_conv, d_forcing, d_mixing, d_mdecay, d_cdecay;

   // `dt_h` is a `local()` scalar in production and is READ at line 1261
   // (`kddt_cur = kd_val*dt_h`) on the static-stability short-circuit path,
   // where it was never assigned on that interface. It retains the previous
   // interface's value in both languages. Zero-init here is the only defined
   // choice; if the corner (short-circuit at ki=nz on obl_it=1) ever fired,
   // DC and CUDA would disagree -- the bit-identity check in the driver is
   // what proves it does not. See README "Caveats".
   double dt_h = 0.0;

   pres = 0.0;
   h_sum = 0.0;
   dsv_dt_sfc = 0.0;
   dsv_ds_sfc = 0.0;
   for (k = nz; k >= 1; --k) {
      hk = A.h_layer[IDX3(i, j, k, nx, ny)];
      if (hk > 0.0) {
         inv_h = 1.0 / (hk + H_NEGLECT);
         t0k = A.hT[IDX3(i, j, k, nx, ny)] * inv_h;
         s0k = A.hS[IDX3(i, j, k, nx, ny)] * inv_h;
      } else {
         t0k = 0.0;
         s0k = 0.0;
      }
      dmass = P.rho0 * hk;
      dpres = GRAVITY * dmass;
      p_mid = pres + 0.5 * dpres;
      eos_specvol_derivs(P, t0k, s0k, p_mid, &dsv_dt_k, &dsv_ds_k);
      A.t0[IDX3(i, j, k, nx, ny)] = t0k;
      A.s0[IDX3(i, j, k, nx, ny)] = s0k;
      A.dpe_t[IDX3(i, j, k, nx, ny)] = dmass * p_mid * dsv_dt_k;
      A.dpe_s[IDX3(i, j, k, nx, ny)] = dmass * p_mid * dsv_ds_k;
      A.dcolht_t[IDX3(i, j, k, nx, ny)] = dmass * dsv_dt_k;
      A.dcolht_s[IDX3(i, j, k, nx, ny)] = dmass * dsv_ds_k;
      if (k == nz) {
         dsv_dt_sfc = dsv_dt_k;
         dsv_ds_sfc = dsv_ds_k;
      }
      pres = pres + dpres;
      h_sum = h_sum + hk;
   }
   h_sum = h_sum + H_NEGLECT;

   q_t_kin = P.inv_rho0_cp * A.Q_heat[IDX2(i, j, nx)];
   q_s_kin = P.inv_rho0 * A.Q_salt[IDX2(i, j, nx)];
   b0 = GRAVITY * P.rho0 * (dsv_dt_sfc * q_t_kin + dsv_ds_sfc * q_s_kin);
   ctke_sfc = -0.5 * GRAVITY * P.rho0 * P.rho0 * A.h_layer[IDX3(i, j, nz, nx, ny)] *
              (q_t_kin * dt * dsv_dt_sfc + q_s_kin * dt * dsv_ds_sfc);

   tau_xc = 0.5 * (A.tau_x[IDX2(i, j, nx + 1)] + A.tau_x[IDX2(i + 1, j, nx + 1)]);
   tau_yc = 0.5 * (A.tau_y[IDX2(i, j, nx)] + A.tau_y[IDX2(i, j + 1, nx)]);
   ustar = fmax(sqrt(sqrt(tau_xc * tau_xc + tau_yc * tau_yc) / P.rho0), P.ustar_min);
   absf = sqrt((1.0 - P.omega_frac) * A.f_centre[IDX2(i, j, nx)] * A.f_centre[IDX2(i, j, nx)] +
               P.omega_frac * 4.0 * P.omega * P.omega);
   idecay = P.tke_decay * absf / ustar;
   mech_in = dt * P.rho0 * ustar * ustar * ustar;

   la_val = 0.0;
   lt_u10 = 0.0;
   lt_ustokes = 0.0;
   lt_kphil = 0.0;
   if (P.use_lt) epbl_lf17_wave_state(ustar, P.rho0, &lt_u10, &lt_ustokes, &lt_kphil);

   d_wind = 0.0;
   d_conv = 0.0;
   d_forcing = 0.0;
   d_mixing = 0.0;
   d_mdecay = 0.0;
   d_cdecay = 0.0;
   mld_found = 0.0;
   obl_it = 0;

   if (A.wet_mask[IDX2(i, j, nx)] <= 0.0 || h_sum <= 2.0 * H_NEGLECT) {
      for (k = 1; k <= nz + 1; ++k) A.kd_int[IDX3(i, j, k, nx, ny)] = 0.0;
   } else {
      min_mld = 0.0;
      max_mld = h_sum;
      dmld_min = 0.0;
      dmld_max = 0.0;
      have_min = 0;
      have_max = 0;
      mld_guess = 0.5 * (min_mld + max_mld);
      if (P.mld_use_prev_guess && A.mld[IDX2(i, j, nx)] > 0.0) {
         mld_guess = fmin(A.mld[IDX2(i, j, nx)], max_mld);
      }
      n_its = 1;
      if (P.mld_iteration) n_its = P.mld_max_its;

      for (obl_it = 1; obl_it <= n_its; ++obl_it) {
         d_wind = 0.0;
         d_conv = 0.0;
         d_forcing = 0.0;
         d_mixing = 0.0;
         d_mdecay = 0.0;
         d_cdecay = 0.0;

         // (A) mstar at the current MLD guess.
         epbl_find_mstar(P, b0, ustar, mld_guess, absf, &mstar_val);
         if (P.use_lt) {
            la_val = epbl_lf17_la(ustar, P.la_frac_hbl * mld_guess, lt_ustokes, lt_kphil);
            epbl_lt_enhance(P, la_val, b0, ustar, mld_guess, absf, &mstar_val);
         }
         mech_tke = mstar_val * mech_in;
         d_wind = mech_tke;

         // (B) seed the reservoirs from the surface forcing.
         if (ctke_sfc <= 0.0) {
            forcing_clip = fmax(ctke_sfc, -mech_tke);
            mech_tke = mech_tke + forcing_clip;
            conv_perel = 0.0;
            d_forcing = forcing_clip;
         } else {
            conv_perel = ctke_sfc;
            d_conv = P.nstar * ctke_sfc;
         }

         // (C) sweep initialization at the surface layer.
         h_ka = A.h_layer[IDX3(i, j, nz, nx, ny)] + H_NEGLECT;
         hp_a = h_ka;
         dpe_t_a = A.dpe_t[IDX3(i, j, nz, nx, ny)];
         dpe_s_a = A.dpe_s[IDX3(i, j, nz, nx, ny)];
         dch_t_a = A.dcolht_t[IDX3(i, j, nz, nx, ny)];
         dch_s_a = A.dcolht_s[IDX3(i, j, nz, nx, ny)];
         th_a = h_ka * A.t0[IDX3(i, j, nz, nx, ny)];
         sh_a = h_ka * A.s0[IDX3(i, j, nz, nx, ny)];
         te_lag = 0.0;
         se_lag = 0.0;
         kddt_prev = 0.0;
         htot = A.h_layer[IDX3(i, j, nz, nx, ny)];
         z_int = A.h_layer[IDX3(i, j, nz, nx, ny)];
         pres_int = GRAVITY * P.rho0 * A.h_layer[IDX3(i, j, nz, nx, ny)];
         mld_output = A.h_layer[IDX3(i, j, nz, nx, ny)];
         sfc_connected = 1;
         A.kd_int[IDX3(i, j, nz + 1, nx, ny)] = 0.0;

         // (D) downward sweep over interfaces Ki = nz .. 2.
         for (ki = nz; ki >= 2; --ki) {
            ka = ki;
            kb = ki - 1;
            h_ka = A.h_layer[IDX3(i, j, ka, nx, ny)] + H_NEGLECT;
            h_kb = A.h_layer[IDX3(i, j, kb, nx, ny)] + H_NEGLECT;
            sfc_disconnect = 0;

            // (1) mechanical TKE decays across the layer above.
            exp_kh = exp(-h_ka * idecay);
            d_mdecay = d_mdecay + (1.0 - exp_kh) * mech_tke;
            mech_tke = mech_tke * exp_kh;

            // (3) rotation-reduced convective efficiency.
            nstar_fc = P.nstar;
            if (conv_perel > 0.0 && absf > 0.0) {
               // (absf*htot)**3 in Fortran is an INTEGER exponent -> expands to
               // multiplies. pow(x,3.0) is a libm call and does NOT give the
               // same bits. Same rule for **2 / **5 in epbl_find_mstar.
               double afh = absf * htot;
               nstar_fc = P.nstar * conv_perel /
                          (conv_perel +
                           0.2 * sqrt(0.5 * dt * P.rho0 * (afh * afh * afh) * conv_perel));
            }
            tot_tke = mech_tke + nstar_fc * conv_perel;

            // (5) static-stability short-circuit.
            if (tot_tke <= 0.0 &&
                0.0 <= (A.dcolht_t[IDX3(i, j, kb, nx, ny)] + A.dcolht_t[IDX3(i, j, ka, nx, ny)]) *
                               (A.t0[IDX3(i, j, ka, nx, ny)] - A.t0[IDX3(i, j, kb, nx, ny)]) +
                           (A.dcolht_s[IDX3(i, j, kb, nx, ny)] +
                            A.dcolht_s[IDX3(i, j, ka, nx, ny)]) *
                               (A.s0[IDX3(i, j, ka, nx, ny)] - A.s0[IDX3(i, j, kb, nx, ny)])) {
               kd_val = 0.0;
               sfc_disconnect = 1;
            } else {
               // (6) velocity scale, mixing length, first-guess Kd.
               dt_h = dt / fmax(0.5 * (h_ka + h_kb), 1.0e-15 * h_sum);
               tke_here = mech_tke + P.wstar_ustar_coef * conv_perel;
               hb_hs = (h_sum - z_int) / h_sum;
               shape_fn = epbl_mixlen_shape(z_int, mld_guess, P.translay_scale, P.mixlen_exponent,
                                            P.mld_iteration);
               hbs = fmin(hb_hs, shape_fn);
               if (tke_here > 0.0) {
                  if (P.vstar_scheme == EPBL_VSTAR_RH18) {
                     surf_scale = fmax(0.05, 1.0 - htot / mld_guess);
                     vstar = P.vstar_scale_fac * surf_scale *
                             (P.vstar_surf_fac * ustar +
                              pow(P.wstar_ustar_coef * conv_perel / (dt * P.rho0), 1.0 / 3.0));
                  } else {
                     vstar = P.vstar_scale_fac * pow(tke_here / (dt * P.rho0), 1.0 / 3.0);
                  }
                  if (P.mld_iteration) {
                     mixlen = fmax(P.min_mix_len, (htot * hbs * vstar) /
                                                      (P.ekman_scale_coef * absf * htot * hbs + vstar));
                     kd_g0 = vstar * P.von_karman * mixlen;
                  } else {
                     kd_g0 = vstar * P.von_karman * (htot * hbs * vstar) /
                             (P.ekman_scale_coef * absf * htot * hbs + vstar);
                  }
               } else {
                  vstar = 0.0;
                  kd_g0 = 0.0;
               }

               // (7) pivot quantities for the layer below.
               hp_b = h_kb;
               th_b = h_kb * A.t0[IDX3(i, j, kb, nx, ny)];
               sh_b = h_kb * A.s0[IDX3(i, j, kb, nx, ny)];

               // (8) closed-form energy solve (direct path).
               hps = hp_a + hp_b;
               bdt1 = hp_a * hp_b;
               dt_c = hp_a * th_b - hp_b * th_a;
               ds_c = hp_a * sh_b - hp_b * sh_a;
               pec_core = hp_b * (dpe_t_a * dt_c + dpe_s_a * ds_c) -
                          hp_a * (A.dpe_t[IDX3(i, j, kb, nx, ny)] * dt_c +
                                  A.dpe_s[IDX3(i, j, kb, nx, ny)] * ds_c);
               colht_core = hp_b * (dch_t_a * dt_c + dch_s_a * ds_c) -
                            hp_a * (A.dcolht_t[IDX3(i, j, kb, nx, ny)] * dt_c +
                                    A.dcolht_s[IDX3(i, j, kb, nx, ny)] * ds_c);
               if (colht_core < 0.0) pec_core = pec_core - pres_int * colht_core;
               dkddt = kd_g0 * dt_h;
               pe_max = pec_core / (bdt1 * hps);

               if (pe_max < 0.0) {
                  // (8a) convectively unstable: mixing RELEASES PE.
                  tke_here = mech_tke + P.wstar_ustar_coef * (conv_perel - pe_max);
                  if (tke_here > 0.0) {
                     if (P.vstar_scheme == EPBL_VSTAR_RH18) {
                        surf_scale = fmax(0.05, 1.0 - htot / mld_guess);
                        vstar = P.vstar_scale_fac * surf_scale *
                                (P.vstar_surf_fac * ustar +
                                 pow(P.wstar_ustar_coef * conv_perel / (dt * P.rho0), 1.0 / 3.0));
                     } else {
                        vstar = P.vstar_scale_fac * pow(tke_here / (dt * P.rho0), 1.0 / 3.0);
                     }
                     if (P.mld_iteration) {
                        mixlen = fmax(P.min_mix_len,
                                      (htot * hbs * vstar) /
                                          (P.ekman_scale_coef * absf * htot * hbs + vstar));
                        kd_val = vstar * P.von_karman * mixlen;
                     } else {
                        kd_val = vstar * P.von_karman * (htot * hbs * vstar) /
                                 (P.ekman_scale_coef * absf * htot * hbs + vstar);
                     }
                  } else {
                     vstar = 0.0;
                     kd_val = 0.0;
                  }
                  pe_g0 = pec_core * dkddt / (bdt1 * (bdt1 + dkddt * hps));
                  dpe_conv = pec_core * (kd_val * dt_h) / (bdt1 * (bdt1 + (kd_val * dt_h) * hps));
                  if (dpe_conv > 0.0) {
                     kd_val = kd_g0;
                     dpe_conv = pe_g0;
                  }
                  conv_perel = conv_perel - dpe_conv;
                  d_conv = d_conv - P.nstar * dpe_conv;
                  if (sfc_connected) mld_output = mld_output + A.h_layer[IDX3(i, j, kb, nx, ny)];
               } else {
                  // (8b) stable: direct closed-form Kd; drain the reservoirs.
                  if ((pec_core * dkddt <= tot_tke * (bdt1 * (bdt1 + dkddt * hps))) ||
                      (pec_core <= 0.0)) {
                     kd_val = kd_g0;
                     tke_used = pec_core * dkddt / (bdt1 * (bdt1 + dkddt * hps));
                     frac_bl = 1.0;
                  } else {
                     kd_val = (bdt1 * bdt1 * tot_tke) / (dt_h * (pec_core - bdt1 * hps * tot_tke));
                     tke_used = tot_tke;
                     frac_bl = tot_tke * (bdt1 * (bdt1 + dkddt * hps)) / (pec_core * dkddt);
                  }
                  if (sfc_connected)
                     mld_output = mld_output + frac_bl * A.h_layer[IDX3(i, j, kb, nx, ny)];
                  if (frac_bl < 1.0) sfc_disconnect = 1;
                  r_reduc = 0.0;
                  if (tot_tke > 0.0 && tot_tke > tke_used) r_reduc = (tot_tke - tke_used) / tot_tke;
                  d_mixing = d_mixing + tke_used;
                  d_cdecay = d_cdecay + (1.0 - r_reduc) * (P.nstar - nstar_fc) * conv_perel;
                  mech_tke = r_reduc * mech_tke;
                  conv_perel = r_reduc * conv_perel;
               }
            }

            A.kd_int[IDX3(i, j, ki, nx, ny)] = kd_val;
            kddt_cur = kd_val * dt_h;

            // (9) advance the embedded forward elimination.
            b1 = 1.0 / (hp_a + kddt_cur);
            c1 = kddt_cur * b1;
            if (ki == nz) {
               te_lag = b1 * (h_ka * A.t0[IDX3(i, j, ka, nx, ny)]);
               se_lag = b1 * (h_ka * A.s0[IDX3(i, j, ka, nx, ny)]);
            } else {
               te_new = b1 * (h_ka * A.t0[IDX3(i, j, ka, nx, ny)] + kddt_prev * te_lag);
               se_new = b1 * (h_ka * A.s0[IDX3(i, j, ka, nx, ny)] + kddt_prev * se_lag);
               te_lag = te_new;
               se_lag = se_new;
            }
            hp_a = h_kb + (hp_a * b1) * kddt_cur;
            dpe_t_a = A.dpe_t[IDX3(i, j, kb, nx, ny)] + c1 * dpe_t_a;
            dpe_s_a = A.dpe_s[IDX3(i, j, kb, nx, ny)] + c1 * dpe_s_a;
            dch_t_a = A.dcolht_t[IDX3(i, j, kb, nx, ny)] + c1 * dch_t_a;
            dch_s_a = A.dcolht_s[IDX3(i, j, kb, nx, ny)] + c1 * dch_s_a;
            th_a = h_kb * A.t0[IDX3(i, j, kb, nx, ny)] + kddt_cur * te_lag;
            sh_a = h_kb * A.s0[IDX3(i, j, kb, nx, ny)] + kddt_cur * se_lag;
            kddt_prev = kddt_cur;
            if (sfc_disconnect) {
               htot = A.h_layer[IDX3(i, j, kb, nx, ny)];
               sfc_connected = 0;
            } else {
               htot = htot + A.h_layer[IDX3(i, j, kb, nx, ny)];
            }
            z_int = z_int + A.h_layer[IDX3(i, j, kb, nx, ny)];
            pres_int = pres_int + GRAVITY * P.rho0 * A.h_layer[IDX3(i, j, kb, nx, ny)];
         }
         A.kd_int[IDX3(i, j, 1, nx, ny)] = 0.0;

         d_mdecay = d_mdecay + mech_tke;
         d_cdecay = d_cdecay + P.nstar * conv_perel;

         mld_found = mld_output;
         if (!P.mld_iteration) break;
         if (fabs(mld_found - mld_guess) < P.mld_tol) break;
         if (obl_it == n_its) break;

         // Bracket update + next guess.
         if (mld_found > mld_guess) {
            min_mld = mld_guess;
            dmld_min = mld_found - mld_guess;
            have_min = 1;
         } else {
            max_mld = mld_guess;
            dmld_max = mld_found - mld_guess;
            have_max = 1;
         }
         if (P.mld_bisection) {
            mld_guess = 0.5 * (min_mld + max_mld);
         } else if (have_min && have_max && obl_it > 2 && ((obl_it - 1) % 4) > 0) {
            mld_guess = min_mld + dmld_min * (max_mld - min_mld) / (dmld_min - dmld_max);
         } else if (mld_found > min_mld && mld_found < max_mld) {
            mld_guess = mld_found;
         } else {
            mld_guess = 0.5 * (min_mld + max_mld);
         }
      }
   }

   A.mld[IDX2(i, j, nx)] = mld_found;
   A.la[IDX2(i, j, nx)] = la_val;
   if (P.tke_diags) {
      A.tke_wind[IDX2(i, j, nx)] = d_wind / dt;
      A.tke_conv[IDX2(i, j, nx)] = d_conv / dt;
      A.tke_forcing[IDX2(i, j, nx)] = d_forcing / dt;
      A.tke_mixing[IDX2(i, j, nx)] = d_mixing / dt;
      A.tke_mech_decay[IDX2(i, j, nx)] = d_mdecay / dt;
      A.tke_conv_decay[IDX2(i, j, nx)] = d_cdecay / dt;
   }
   // Diagnostic: how many MLD iterations this column actually consumed. NULL in
   // every timed run, so it costs nothing there. Used to quantify divergence.
   // (`obl_it` is left at 0 on the dry-column path and at the loop-exit value
   // otherwise; the C for-loop increments past n_its only when it runs to
   // completion, which the `obl_it == n_its` break already prevents.)
   if (A.its_used) A.its_used[IDX2(i, j, nx)] = obl_it;
}

// Variant 1: faithful. Block size fixed at 128 to match nvfortran's own choice
// for this loop; NO __launch_bounds__, so ptxas allocates registers exactly as
// it sees fit -- the same freedom nvfortran's backend has.
__global__ void epbl_faithful(EpblParams P, EpblPtrs A) {
   size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
   size_t ncol = (size_t)P.nx * (size_t)P.ny;
   if (t >= ncol) return;
   int i = (int)(t % (size_t)P.nx) + 1;
   int j = (int)(t / (size_t)P.nx) + 1;
   epbl_column_body(P, A, i, j);
}

// Variant 2: the CUDA-exclusive lever. Identical body; __launch_bounds__ caps
// registers to raise occupancy. MAXREG_PER_THREAD is set by -DEPBL_LB_* at
// compile time and the block size comes from the caller.
#ifndef EPBL_LB_THREADS
#define EPBL_LB_THREADS 128
#endif
#ifndef EPBL_LB_BLOCKS_PER_SM
#define EPBL_LB_BLOCKS_PER_SM 4
#endif
__global__ void __launch_bounds__(EPBL_LB_THREADS, EPBL_LB_BLOCKS_PER_SM)
    epbl_tuned(EpblParams P, EpblPtrs A) {
   size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
   size_t ncol = (size_t)P.nx * (size_t)P.ny;
   if (t >= ncol) return;
   int i = (int)(t % (size_t)P.nx) + 1;
   int j = (int)(t / (size_t)P.nx) + 1;
   epbl_column_body(P, A, i, j);
}

// ===========================================================================
// Host launcher. `variant`: 0 = faithful, 1 = tuned.
// `sync`: 0 = async, 1 = sync after launch.
// ===========================================================================
extern "C" void epbl_cuda_launch(const double *h_layer, const double *wet_mask, const double *hT,
                                 const double *hS, const double *tau_x, const double *tau_y,
                                 const double *Q_heat, const double *Q_salt, const double *f_centre,
                                 double *mld, double *kd_int, double *la, double *t0, double *s0,
                                 double *dpe_t, double *dpe_s, double *dcolht_t, double *dcolht_s,
                                 double *tke_wind, double *tke_conv, double *tke_forcing,
                                 double *tke_mixing, double *tke_mech_decay, double *tke_conv_decay,
                                 int *its_used, const EpblParams *Pin, int variant, int threads,
                                 int sync) {
   EpblParams P = *Pin;
   EpblPtrs A;
   A.h_layer = h_layer; A.wet_mask = wet_mask; A.hT = hT; A.hS = hS;
   A.tau_x = tau_x; A.tau_y = tau_y; A.Q_heat = Q_heat; A.Q_salt = Q_salt;
   A.f_centre = f_centre; A.mld = mld; A.kd_int = kd_int; A.la = la;
   A.t0 = t0; A.s0 = s0; A.dpe_t = dpe_t; A.dpe_s = dpe_s;
   A.dcolht_t = dcolht_t; A.dcolht_s = dcolht_s;
   A.tke_wind = tke_wind; A.tke_conv = tke_conv; A.tke_forcing = tke_forcing;
   A.tke_mixing = tke_mixing; A.tke_mech_decay = tke_mech_decay; A.tke_conv_decay = tke_conv_decay;
   A.its_used = its_used;

   size_t ncol = (size_t)P.nx * (size_t)P.ny;
   // variant 0 pins 128 to match nvfortran's own choice for this loop.
   // variant 1's block size is fixed by __launch_bounds__ at compile time --
   // launching it with anything else would silently break the register cap, so
   // `threads` is ignored here rather than trusted.
   (void)threads;
   int thr = (variant == 0) ? 128 : EPBL_LB_THREADS;
   size_t nblk = (ncol + (size_t)thr - 1) / (size_t)thr;

   if (variant == 0) {
      epbl_faithful<<<(unsigned)nblk, (unsigned)thr>>>(P, A);
   } else {
      epbl_tuned<<<(unsigned)nblk, (unsigned)thr>>>(P, A);
   }
   if (sync) {
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) printf("CUDA error (variant %d): %s\n", variant, cudaGetErrorString(e));
   }
}
