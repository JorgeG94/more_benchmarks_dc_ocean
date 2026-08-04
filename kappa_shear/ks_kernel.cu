// Faithful hand-written CUDA C port of the ocean model's kappa-shear (JHL08) column
// solver, for the `do concurrent` vs CUDA C benchmark.
//
// PORT CONTRACT
// -------------
// One thread per column -- EXACTLY what `do concurrent (j, i)` does. Every
// per-column array lives in the thread's local frame, dimensioned NZL/NZLI =
// NZ_STACK_MAX/(NZ_STACK_MAX+1), same as the Fortran. This variant is
// deliberately NOT clever: it exists to isolate CODEGEN. If it ties with
// `do concurrent`, nvfortran's codegen is fine and the question moves to
// parallelisation strategy (see ks_kernel_warp.cu).
//
// FORTRAN 1-BASED INDEXING IS PRESERVED. Every column array is declared
// [NZLI+1] and indexed [1..nz+1] exactly as the Fortran does. The wasted
// element 0 costs 8 bytes per array against a ~40 KB frame -- worth it to kill
// an entire class of off-by-one transcription bugs.
//
// GLOBAL ARRAYS are Fortran column-major, passed through `host_data
// use_device` -- the SAME device allocation the `do concurrent` variant reads.
//   h_layer(i,j,k)        -> [(i-1) + (j-1)*nx + (k-1)*nx*ny]
//   u_face_x_layer(i,j,k) -> [(i-1) + (j-1)*(nx+1) + (k-1)*(nx+1)*ny]
//   v_face_y_layer(i,j,k) -> [(i-1) + (j-1)*nx + (k-1)*nx*(ny+1)]
//
// ONE DELIBERATE DEVIATION from the Fortran, which does NOT change results:
// ks_solve_column's three whole-array assignments (`kappa_out = kappa`,
// `kq_tmp = k_q` x2) copy all NZLI elements in Fortran because that is the
// declared extent -- even though only 1..nz+1 are ever read. This port copies
// 1..nz+1. Elements above nz+1 are never read by either version, so results
// are unaffected; the Fortran simply moves ~4x more bytes at nz=30,
// NZ_STACK_MAX=128. That gap is measured separately by the `dcfix` Fortran
// variant (ks_fix.F90), not hidden here.

#include <cstdio>
#include "gpu_rt.h"

// Must match the Fortran's bound (Makefile passes -DMODEL_NZ_STACK_MAX to
// BOTH compilers). If these ever diverge, the two variants solve differently
// sized frames and the comparison is meaningless -- hence the static assert.
#ifndef MODEL_NZ_STACK_MAX
#define MODEL_NZ_STACK_MAX 128
#endif
#define NZL  MODEL_NZ_STACK_MAX
#define NZLI (MODEL_NZ_STACK_MAX + 1)

// ---------------------------------------------------------------------------
// ALGORITHMIC-COMPLEXITY PARITY WITH THE FORTRAN  (-DKS_FULL_COPY)
// ---------------------------------------------------------------------------
// ks_solve_column has three WHOLE-ARRAY assignments -- `kappa_out = kappa` and
// `kq_tmp = k_q` twice. In Fortran those copy the DECLARED extent (NZLI
// elements = NZ_STACK_MAX+1), not 1..nz+1, because that is what a whole-array
// assignment means; ks.F90:949 documents it as the ONLY reason wall time
// depends on NZ_STACK_MAX at all. This port originally bounded them to
// 1..nz+1, which is numerically identical (elements above nz+1 are never read)
// but is NOT the same algorithmic complexity: O(NZ_STACK_MAX) became O(nz).
//
// Timing a bounded-copy CUDA kernel against a declared-extent `do concurrent`
// kernel is therefore not a 1:1 comparison -- part of any measured gap is a
// copy one side simply does not perform. So the copy bound is now a build
// knob, and the two sides are compared in MATCHED pairs:
//
//   baseline  (true 1:1 with production Fortran):
//       make cpp FULLCOPY=1      vs   make dc BCOPY=0
//   optimised (both sides bound the copy):
//       make cpp FULLCOPY=0      vs   make dc BCOPY=1
//
// Mixing the pairs measures the copy, not the codegen.
#ifdef KS_FULL_COPY
#define KS_COPY_TOP NZLI
#else
#define KS_COPY_TOP (nz + 1)
#endif

#ifndef KS_BLOCK
#define KS_BLOCK 128
#endif

// Mirrors ocean_kappa_shear_t's knob set + the flat-POD EOS handle. Passed by
// value: on sm_70 that lands in constant/param space, register-resident.
struct KsPar {
   double dt, ri_crit, shearmix_rate, fri_curvature;
   double c_n, c_s, lambda, lz_rescale;
   double kappa_0, kappa_seed, kappa_trunc, tke_bg;
   double tol_err, src_max_chg, vel_underflow, rho0;
   int max_inner_it, max_substep_it;
   int eos_variant;
   double eos_rho0, eos_alpha_T, eos_beta_S;
};

#define GRAVITY    9.80665
#define H_VANISHED 1.5e-4

#define EOS_VARIANT_WRIGHT_97 2
#define WRIGHT_A1  3.480336e-7
#define WRIGHT_A2 -1.112733e-7
#define WRIGHT_B0  5.790749e8
#define WRIGHT_B1  3.516535e6
#define WRIGHT_B2 -4.002714e4
#define WRIGHT_B3  2.084372e2
#define WRIGHT_B4  5.944068e5
#define WRIGHT_B5 -9.643486e3
#define WRIGHT_C0  1.704853e5
#define WRIGHT_C1  7.904722e2
#define WRIGHT_C2 -7.984422e0
#define WRIGHT_C3  5.140652e-2
#define WRIGHT_C4 -2.302158e2
#define WRIGHT_C5 -3.079464e0

// ---- eos_specvol_derivs (ocean_eos.F90) -----------------------------
__device__ __forceinline__ void eos_specvol_derivs(const KsPar &p, double T,
                                                   double S, double pr,
                                                   double *dsv_dt,
                                                   double *dsv_ds) {
   if (p.eos_variant == EOS_VARIANT_WRIGHT_97) {
      double T_sq = T * T;
      double p_0 = WRIGHT_B0 + WRIGHT_B1 * T + WRIGHT_B2 * T_sq +
                   WRIGHT_B3 * T_sq * T + WRIGHT_B4 * S + WRIGHT_B5 * S * T;
      double lam = WRIGHT_C0 + WRIGHT_C1 * T + WRIGHT_C2 * T_sq +
                   WRIGHT_C3 * T_sq * T + WRIGHT_C4 * S + WRIGHT_C5 * S * T;
      double dp0_dt = WRIGHT_B1 + 2.0 * WRIGHT_B2 * T + 3.0 * WRIGHT_B3 * T_sq +
                      WRIGHT_B5 * S;
      double dlam_dt = WRIGHT_C1 + 2.0 * WRIGHT_C2 * T +
                       3.0 * WRIGHT_C3 * T_sq + WRIGHT_C5 * S;
      double dp0_ds = WRIGHT_B4 + WRIGHT_B5 * T;
      double dlam_ds = WRIGHT_C4 + WRIGHT_C5 * T;
      double big_p = pr + p_0;
      double inv_p = 1.0 / big_p;
      double inv_p2 = inv_p * inv_p;
      *dsv_dt = WRIGHT_A1 + dlam_dt * inv_p - lam * dp0_dt * inv_p2;
      *dsv_ds = WRIGHT_A2 + dlam_ds * inv_p - lam * dp0_ds * inv_p2;
   } else {
      *dsv_dt = p.eos_alpha_T / (p.eos_rho0 * p.eos_rho0);
      *dsv_ds = -p.eos_beta_S / (p.eos_rho0 * p.eos_rho0);
   }
}

// ---- ks_src_func (production :474-490) ----------------------------------
__device__ __forceinline__ double ks_src_func(double ri_crit,
                                              double shearmix_rate,
                                              double fri_curvature, double n2,
                                              double s2) {
   double ksrc = 0.0;
   if (n2 < ri_crit * s2) {
      double dnom = ri_crit * s2 + fri_curvature * n2;
      if (dnom != 0.0 && s2 > 0.0) {
         ksrc = 2.0 * shearmix_rate * sqrt(s2) * (ri_crit * s2 - n2) / dnom;
      }
   }
   return ksrc;
}

// ---- ks_precompute (production :492-553) --------------------------------
__device__ void ks_precompute(int nz, double lz_rescale, const double *h_sd,
                              double *idz_o, double *idz_int_o, double *hint_o,
                              double *il2_o, double *dbot) {
   double i_lz2 = 1.0 / (lz_rescale * lz_rescale);

   for (int k = 1; k <= nz; ++k) idz_o[k] = 1.0 / h_sd[k];

   idz_int_o[1] = 2.0 / h_sd[1];
   for (int k = 2; k <= nz; ++k) idz_int_o[k] = 2.0 / (h_sd[k - 1] + h_sd[k]);
   idz_int_o[nz + 1] = 2.0 / h_sd[nz];

   hint_o[1] = 0.0;
   if (nz >= 2) {
      hint_o[2] = h_sd[1];
      for (int k = 2; k <= nz - 1; ++k) {
         double hk = h_sd[k], hkm1 = h_sd[k - 1], hkp1 = h_sd[k + 1];
         double norm_l = 1.0 / (hk * (hkm1 + hkp1) + 2.0 * hkm1 * hkp1);
         double wt_a = (hk + hkp1) * hkm1 * norm_l;
         double wt_b = (hkm1 + hk) * hkp1 * norm_l;
         hint_o[k] = hint_o[k] + hk * wt_a;
         hint_o[k + 1] = hk * wt_b;
      }
      hint_o[nz] = hint_o[nz] + h_sd[nz];
   }
   hint_o[nz + 1] = 0.0;

   dbot[nz + 1] = 0.0;
   for (int k = nz; k >= 1; --k) dbot[k] = dbot[k + 1] + h_sd[k];
   il2_o[1] = 0.0;
   il2_o[nz + 1] = 0.0;
   double dtop = 0.0;
   for (int k = 2; k <= nz; ++k) {
      dtop = dtop + h_sd[k - 1];
      if (dtop > 0.0 && dbot[k] > 0.0) {
         double num = (dtop + dbot[k]);
         double den = (dtop * dbot[k]);
         il2_o[k] = i_lz2 * (num * num) / (den * den);
      } else {
         il2_o[k] = 0.0;
      }
   }
}

// ---- ks_projected_state (production :555-692) ---------------------------
__device__ void ks_projected_state(int nz, double dt_now, int ks, int ke,
                                   double vel_underflow, const double *dbuoy_t,
                                   const double *dbuoy_s, const double *h_sd,
                                   const double *idz_int_s, const double *u0,
                                   const double *v0, const double *t0,
                                   const double *s0, const double *kappa_ps,
                                   double *u_o, double *v_o, double *t_o,
                                   double *s_o, double *c1_o, double *n2_o,
                                   double *s2_o) {
   for (int k = 1; k <= nz; ++k) {
      u_o[k] = u0[k];
      v_o[k] = v0[k];
      t_o[k] = t0[k];
      s_o[k] = s0[k];
   }
   for (int k = 1; k <= nz + 1; ++k) c1_o[k] = 0.0;

   if (ks <= ke && dt_now > 0.0) {
      double a_b = dt_now * kappa_ps[ks + 1] * idz_int_s[ks + 1];
      double b1 = 1.0 / (h_sd[ks] + a_b);
      c1_o[ks + 1] = a_b * b1;
      double d1 = h_sd[ks] * b1;
      u_o[ks] = b1 * h_sd[ks] * u0[ks];
      v_o[ks] = b1 * h_sd[ks] * v0[ks];
      t_o[ks] = b1 * h_sd[ks] * t0[ks];
      s_o[ks] = b1 * h_sd[ks] * s0[ks];

      double a_a;
      for (int k = ks + 1; k <= ke - 1; ++k) {
         a_a = a_b;
         a_b = dt_now * kappa_ps[k + 1] * idz_int_s[k + 1];
         double bd1 = h_sd[k] + d1 * a_a;
         b1 = 1.0 / (bd1 + a_b);
         c1_o[k + 1] = a_b * b1;
         d1 = bd1 * b1;
         u_o[k] = b1 * (h_sd[k] * u0[k] + a_a * u_o[k - 1]);
         v_o[k] = b1 * (h_sd[k] * v0[k] + a_a * v_o[k - 1]);
         t_o[k] = b1 * (h_sd[k] * t0[k] + a_a * t_o[k - 1]);
         s_o[k] = b1 * (h_sd[k] * s0[k] + a_a * s_o[k - 1]);
      }

      a_a = a_b;
      double b1nz;
      if (ke > ks) {
         b1 = 1.0 / (h_sd[ke] + d1 * a_a);
         t_o[ke] = b1 * (h_sd[ke] * t0[ke] + a_a * t_o[ke - 1]);
         s_o[ke] = b1 * (h_sd[ke] * s0[ke] + a_a * s_o[ke - 1]);
         if (ke == nz) {
            b1nz = 1.0 / ((h_sd[ke] + d1 * a_a) +
                          dt_now * kappa_ps[nz + 1] * idz_int_s[nz + 1]);
         } else {
            b1nz = b1;
         }
         u_o[ke] = b1nz * (h_sd[ke] * u0[ke] + a_a * u_o[ke - 1]);
         v_o[ke] = b1nz * (h_sd[ke] * v0[ke] + a_a * v_o[ke - 1]);
      } else {
         b1 = 1.0 / (h_sd[ke] + a_a);
         t_o[ke] = b1 * h_sd[ke] * t0[ke];
         s_o[ke] = b1 * h_sd[ke] * s0[ke];
         if (ke == nz) {
            b1nz = 1.0 / (h_sd[ke] + a_a +
                          dt_now * kappa_ps[nz + 1] * idz_int_s[nz + 1]);
         } else {
            b1nz = b1;
         }
         u_o[ke] = b1nz * h_sd[ke] * u0[ke];
         v_o[ke] = b1nz * h_sd[ke] * v0[ke];
      }
      if (fabs(u_o[ke]) < vel_underflow) u_o[ke] = 0.0;
      if (fabs(v_o[ke]) < vel_underflow) v_o[ke] = 0.0;

      for (int k = ke - 1; k >= ks; --k) {
         u_o[k] = u_o[k] + c1_o[k + 1] * u_o[k + 1];
         v_o[k] = v_o[k] + c1_o[k + 1] * v_o[k + 1];
         t_o[k] = t_o[k] + c1_o[k + 1] * t_o[k + 1];
         s_o[k] = s_o[k] + c1_o[k + 1] * s_o[k + 1];
         if (fabs(u_o[k]) < vel_underflow) u_o[k] = 0.0;
         if (fabs(v_o[k]) < vel_underflow) v_o[k] = 0.0;
      }
   } else {
      for (int k = 1; k <= nz; ++k) {
         if (fabs(u_o[k]) < vel_underflow) u_o[k] = 0.0;
         if (fabs(v_o[k]) < vel_underflow) v_o[k] = 0.0;
      }
   }

   n2_o[1] = 0.0;
   n2_o[nz + 1] = 0.0;
   s2_o[1] = 0.0;
   s2_o[nz + 1] = 0.0;
   for (int kk = 2; kk <= nz; ++kk) {
      double ua, ub, va, vb, ta, tb, sa, sb;
      if (kk - 1 >= ks && kk - 1 <= ke) {
         ua = u_o[kk - 1]; va = v_o[kk - 1]; ta = t_o[kk - 1]; sa = s_o[kk - 1];
      } else {
         ua = u0[kk - 1]; va = v0[kk - 1]; ta = t0[kk - 1]; sa = s0[kk - 1];
      }
      if (kk >= ks && kk <= ke) {
         ub = u_o[kk]; vb = v_o[kk]; tb = t_o[kk]; sb = s_o[kk];
      } else {
         ub = u0[kk]; vb = v0[kk]; tb = t0[kk]; sb = s0[kk];
      }
      double n2v = idz_int_s[kk] *
                   (dbuoy_t[kk] * (ta - tb) + dbuoy_s[kk] * (sa - sb));
      if (n2v < 0.0) n2v = 0.0;
      n2_o[kk] = n2v;
      double du = ua - ub, dv = va - vb;
      s2_o[kk] = (du * du + dv * dv) * idz_int_s[kk] * idz_int_s[kk];
   }
}

// ---- ks_find_kappa_tke (production :694-934) ----------------------------
// `n_it` returns the Picard iteration count actually executed -- the
// instrumentation the divergence question needs. It is one store per call into
// a register; it does not perturb the schedule.
__device__ void ks_find_kappa_tke(int nz, double tke_min, double f2_val,
                                  double ri_crit, double shearmix_rate,
                                  double fri_curvature, double c_n2, double c_s2,
                                  double ilambda2, double kappa_0,
                                  double kappa_trunc, double tke_bg,
                                  double tol_err, int max_inner_it,
                                  const double *n2_in, const double *s2_in,
                                  const double *kappa_seed, double *k_q_io,
                                  const double *idz_s, const double *hint_s,
                                  const double *il2_s, const double *e1_s,
                                  double *tke_o, double *kappa_o,
                                  double *ksrc_sc, double *tkedec_sc,
                                  double *aq_sc, double *dq_sc, double *cq_sc,
                                  double *dk_sc, double *ck_sc, double *ild2_sc,
                                  int *n_it) {
   int ks_src = nz + 2, ke_src = 0;
   ksrc_sc[1] = 0.0;
   ksrc_sc[nz + 1] = 0.0;
   for (int kk = 2; kk <= nz; ++kk) {
      ksrc_sc[kk] = ks_src_func(ri_crit, shearmix_rate, fri_curvature,
                                n2_in[kk], s2_in[kk]);
      if (ksrc_sc[kk] > 0.0) {
         if (ks_src > kk) ks_src = kk;
         ke_src = kk;
      }
   }

   if (ks_src > ke_src) {
      for (int kk = 1; kk <= nz + 1; ++kk) {
         tke_o[kk] = tke_min;
         kappa_o[kk] = 0.0;
         k_q_io[kk] = 0.0;
      }
      *n_it = 0;
      return;
   }

   for (int kk = 2; kk <= nz; ++kk)
      tkedec_sc[kk] = sqrt(c_n2 * n2_in[kk] + c_s2 * s2_in[kk]);

   tke_o[1] = tke_bg;
   for (int kk = 2; kk <= nz; ++kk) {
      if (kappa_seed[kk] > 0.0 && k_q_io[kk] > 0.0)
         tke_o[kk] = kappa_seed[kk] / k_q_io[kk];
      else
         tke_o[kk] = tke_min;
   }
   tke_o[nz + 1] = tke_min;

   for (int kk = 1; kk <= nz + 1; ++kk) kappa_o[kk] = kappa_seed[kk];
   kappa_o[1] = 0.0;
   kappa_o[nz + 1] = 0.0;

   int ks_kap = 2, ke_kap = nz, ks_kp = 2, ke_kp = nz;
   int it;

   for (it = 1; it <= max_inner_it; ++it) {
      // (a) TKE tridiagonal sweep.
      int ke_tke = min(max(ke_kap, ke_kp) + 1, nz + 1);
      for (int k = 1; k <= min(ke_tke, nz); ++k)
         aq_sc[k] = (0.5 * (kappa_o[k] + kappa_o[k + 1]) + kappa_0) * idz_s[k];

      dq_sc[1] = -tke_o[1];
      tke_o[1] = tke_bg;
      cq_sc[2] = 0.0;
      double cqc_l = 1.0;
      for (int kk = 2; kk <= ke_tke - 1; ++kk) {
         dq_sc[kk] = -tke_o[kk];
         double tsrc_l =
             (kappa_o[kk] + kappa_0) * s2_in[kk] + tke_bg * tkedec_sc[kk];
         double bqd1_l =
             hint_s[kk] * (tkedec_sc[kk] + n2_in[kk] * k_q_io[kk]) +
             cqc_l * aq_sc[kk - 1];
         double bq_l = 1.0 / (bqd1_l + aq_sc[kk]);
         tke_o[kk] = bq_l * (hint_s[kk] * tsrc_l + aq_sc[kk - 1] * tke_o[kk - 1]);
         cq_sc[kk + 1] = aq_sc[kk] * bq_l;
         cqc_l = bqd1_l * bq_l;
      }

      if (ke_tke == nz + 1) {
         tke_o[nz + 1] = tke_min;
         dq_sc[nz + 1] = 0.0;
      } else {
         int kk = ke_tke;
         double tsrc_l = kappa_0 * s2_in[kk] + tke_bg * tkedec_sc[kk];
         double bq_l = 1.0 / (hint_s[kk] * tkedec_sc[kk] +
                              cqc_l * aq_sc[kk - 1] + aq_sc[kk]);
         cq_sc[kk + 1] = aq_sc[kk] * bq_l;
         dq_sc[kk] = -tke_o[kk];
         double raw_l =
             bq_l * (hint_s[kk] * tsrc_l + aq_sc[kk - 1] * tke_o[kk - 1]);
         double dnom_l = 1.0 - cq_sc[kk + 1] * e1_s[kk + 1];
         if (fabs(dnom_l) > 1.0e-30) {
            tke_o[kk] = fmax((raw_l + cq_sc[kk + 1] * (tke_o[kk + 1] -
                                                       e1_s[kk + 1] * tke_o[kk])) /
                                 dnom_l,
                             tke_min);
         } else {
            tke_o[kk] = fmax(raw_l, tke_min);
         }
         dq_sc[kk] = tke_o[kk] + dq_sc[kk];
         for (int k2 = ke_tke + 1; k2 <= nz + 1; ++k2) {
            dq_sc[k2] = e1_s[k2] * dq_sc[k2 - 1];
            tke_o[k2] = fmax(tke_o[k2] + dq_sc[k2], tke_min);
            if (fabs(dq_sc[k2]) < 1.0e-16 * tke_o[k2]) break;
         }
      }

      for (int kk = ke_tke - 1; kk >= 1; --kk) {
         tke_o[kk] = fmax(tke_o[kk] + cq_sc[kk + 1] * tke_o[kk + 1], tke_min);
         dq_sc[kk] = tke_o[kk] + dq_sc[kk];
      }

      // (b) kappa tridiagonal sweep with truncation ramp + range track.
      ks_kp = ks_kap;
      ke_kp = ke_kap;
      for (int kk = 2; kk <= nz; ++kk) {
         if (tke_o[kk] > 0.0)
            ild2_sc[kk] = (n2_in[kk] * ilambda2 + f2_val) / tke_o[kk] + il2_s[kk];
         else
            ild2_sc[kk] = 1.0e30;
      }

      dk_sc[1] = 0.0;
      ck_sc[2] = 0.0;
      double ckc_l = 1.0;
      int ke_new = 0; (void)ke_new;
      int ks_new = nz;
      for (int kk = 2; kk <= nz; ++kk) {
         dk_sc[kk] = -kappa_o[kk];
         double bkd1_l = hint_s[kk] * ild2_sc[kk] + ckc_l * idz_s[kk - 1];
         double bk_l = 1.0 / (bkd1_l + idz_s[kk]);
         kappa_o[kk] = bk_l * (idz_s[kk - 1] * kappa_o[kk - 1] +
                               hint_s[kk] * ksrc_sc[kk]);
         ck_sc[kk + 1] = idz_s[kk] * bk_l;
         ckc_l = bkd1_l * bk_l;

         double trv = ckc_l * kappa_trunc;
         double tr2v = 2.0 * trv;
         if (kappa_o[kk] < trv) {
            kappa_o[kk] = 0.0;
            if (kk > ke_src) {
               ke_kap = kk - 1;
               k_q_io[kk] = 0.0;
               break;
            }
         } else if (kappa_o[kk] < tr2v) {
            kappa_o[kk] = 2.0 * (kappa_o[kk] - trv);
         }
#ifndef KS_KE_MOM6
         ke_new = kk;
#endif
      }
      // Fortran `exit` leaves the loop and FALLS THROUGH to the statement
      // after `end do` -- so this runs even on the early exit, clobbering the
      // `ke_kap = kk-1` just assigned. Reproduced deliberately.
#ifndef KS_KE_MOM6
      if (ke_new > 0) ke_kap = ke_new;
#endif

      if (ke_kap >= 1 && tke_o[ke_kap] > 0.0)
         k_q_io[ke_kap] = kappa_o[ke_kap] / tke_o[ke_kap];
      dk_sc[ke_kap] = dk_sc[ke_kap] + kappa_o[ke_kap];

      for (int kk = ke_kap + 2; kk <= ke_kp + 1; ++kk) {
         if (kk > nz) break;
         dk_sc[kk] = -kappa_o[kk];
         kappa_o[kk] = 0.0;
         k_q_io[kk] = 0.0;
      }

      ks_new = 2;
      for (int kk = ke_kap - 1; kk >= 2; --kk) {
         kappa_o[kk] = kappa_o[kk] + ck_sc[kk + 1] * kappa_o[kk + 1];
         if (kappa_o[kk] <= kappa_trunc) {
            kappa_o[kk] = 0.0;
            if (kk < ks_src) {
               ks_kap = kk + 1;
               k_q_io[kk] = 0.0;
               break;
            }
         } else if (kappa_o[kk] < 2.0 * kappa_trunc) {
            kappa_o[kk] = 2.0 * (kappa_o[kk] - kappa_trunc);
         }
         dk_sc[kk] = dk_sc[kk] + kappa_o[kk];
         if (tke_o[kk] > 0.0)
            k_q_io[kk] = kappa_o[kk] / tke_o[kk];
         else
            k_q_io[kk] = 0.0;
         ks_new = kk;
      }
      // Same fall-through as the forward loop: Fortran's `exit` does NOT skip
      // this, so the `ks_kap = kk+1` assigned inside the loop is immediately
      // overwritten. Surprising, but it is the shipped semantics -- porting it
      // as a `goto` past this line silently changes the active range, and then
      // the iteration count, and then every number in this benchmark.
      ks_kap = max(ks_new, 2);

      for (int kk = ks_kp; kk <= ks_kap - 2; ++kk) {
         if (kk < 2) continue;
         kappa_o[kk] = 0.0;
         k_q_io[kk] = 0.0;
      }

      // (c) Picard convergence test.
      {
         int k_lo = min(ks_kap, ks_kp);
         int k_hi = max(ke_kap, ke_kp);
         bool conv = true;
         for (int kk = k_lo; kk <= k_hi; ++kk) {
            double lhs_l = fabs(dk_sc[kk]);
            double rhs_l = tol_err * (kappa_0 + kappa_o[kk] - 0.5 * dk_sc[kk]);
            if (lhs_l > rhs_l) {
               conv = false;
               break;
            }
         }
         if (conv) break;
      }
   }

   *n_it = (it > max_inner_it) ? max_inner_it : it;

   kappa_o[1] = 0.0;
   kappa_o[nz + 1] = 0.0;
}

// ---- ks_adaptive_dt (production :936-1064) ------------------------------
__device__ double ks_adaptive_dt(int nz, double dt_rem, int itt_outer,
                                 int max_substep_it, double ri_crit,
                                 double shearmix_rate, double fri_curvature,
                                 double src_max_chg, double tol_err,
                                 double vel_underflow, const double *dbuoy_t,
                                 const double *dbuoy_s, const double *h_s,
                                 const double *u_cur, const double *v_cur,
                                 const double *t_cur, const double *s_cur,
                                 const double *kappa_out_s,
                                 const double *kappa_src_s,
                                 const double *local_src_s,
                                 const double *local_src_avg_s, int ks_kap,
                                 int ke_kap, const double *idz_int_s,
                                 // caller-owned scratch (the Fortran declares
                                 // these as function locals; hoisting them
                                 // keeps one frame instead of two)
                                 double *tol_max, double *tol_min_a,
                                 double *tol_chg_a, double *u_pr, double *v_pr,
                                 double *t_pr, double *s_pr, double *c1_pr,
                                 double *n2_pr, double *s2_pr) {
   double tol_dksrc_low;
   if (src_max_chg == 10.0)
      tol_dksrc_low = 0.95;
   else
      tol_dksrc_low = (src_max_chg - 0.5) / src_max_chg;
   double tol2_l = 2.0 * tol_err;

   for (int kk = 1; kk <= nz + 1; ++kk) {
      tol_max[kk] = kappa_src_s[kk] + src_max_chg * local_src_s[kk];
      tol_min_a[kk] = kappa_src_s[kk] - tol_dksrc_low * local_src_s[kk];
      tol_chg_a[kk] = tol2_l * local_src_avg_s[kk];
   }

   int k_lo = max(ks_kap - 1, 2);
   int k_hi = min(ke_kap + 1, nz);
   int ks_lyr = max(ks_kap - 1, 1);
   int ke_lyr = min(ke_kap, nz);

   double dt_tst = dt_rem;
   bool valid_dt = false;
   int max_halvings = (max_substep_it + 1 - itt_outer) / 2;

   for (int ih = 1; ih <= max(max_halvings, 1); ++ih) {
      ks_projected_state(nz, 0.5 * dt_tst, ks_lyr, ke_lyr, vel_underflow,
                         dbuoy_t, dbuoy_s, h_s, idz_int_s, u_cur, v_cur, t_cur,
                         s_cur, kappa_out_s, u_pr, v_pr, t_pr, s_pr, c1_pr,
                         n2_pr, s2_pr);
      double idtt = 0.0;
      if (dt_tst > 0.0) idtt = 1.0 / dt_tst;
      valid_dt = true;
      for (int kk = k_lo; kk <= k_hi; ++kk) {
         if (n2_pr[kk] < ri_crit * s2_pr[kk]) {
            double ksrc_tst = ks_src_func(ri_crit, shearmix_rate, fri_curvature,
                                          n2_pr[kk], s2_pr[kk]);
            double upper_t =
                fmax(tol_max[kk], kappa_src_s[kk] + idtt * tol_chg_a[kk]);
            double lower_t =
                fmin(tol_min_a[kk], kappa_src_s[kk] - idtt * tol_chg_a[kk]);
            if (ksrc_tst > upper_t || ksrc_tst < lower_t) {
               valid_dt = false;
               break;
            }
         } else {
            double lower_t =
                fmin(tol_min_a[kk], kappa_src_s[kk] - idtt * tol_chg_a[kk]);
            if (0.0 < lower_t) {
               valid_dt = false;
               break;
            }
         }
      }
      if (valid_dt) break;
      dt_tst = 0.5 * dt_tst;
   }

   double dt_inc = 0.0;
   if (dt_tst < dt_rem && valid_dt) {
      dt_inc = 0.5 * dt_tst;
      for (int ir = 1; ir <= 5; ++ir) {
         double dt_try = dt_tst + dt_inc;
         ks_projected_state(nz, 0.5 * dt_try, ks_lyr, ke_lyr, vel_underflow,
                            dbuoy_t, dbuoy_s, h_s, idz_int_s, u_cur, v_cur,
                            t_cur, s_cur, kappa_out_s, u_pr, v_pr, t_pr, s_pr,
                            c1_pr, n2_pr, s2_pr);
         double idtt = 0.0;
         if (dt_try > 0.0) idtt = 1.0 / dt_try;
         bool valid_try = true;
         for (int kk = k_lo; kk <= k_hi; ++kk) {
            if (n2_pr[kk] < ri_crit * s2_pr[kk]) {
               double ksrc_tst = ks_src_func(ri_crit, shearmix_rate,
                                             fri_curvature, n2_pr[kk], s2_pr[kk]);
               double upper_t =
                   fmax(tol_max[kk], kappa_src_s[kk] + idtt * tol_chg_a[kk]);
               double lower_t =
                   fmin(tol_min_a[kk], kappa_src_s[kk] - idtt * tol_chg_a[kk]);
               if (ksrc_tst > upper_t || ksrc_tst < lower_t) {
                  valid_try = false;
                  break;
               }
            } else {
               double lower_t =
                   fmin(tol_min_a[kk], kappa_src_s[kk] - idtt * tol_chg_a[kk]);
               if (0.0 < lower_t) {
                  valid_try = false;
                  break;
               }
            }
         }
         if (valid_try) dt_tst = dt_try;
         dt_inc = 0.5 * dt_inc;
      }
   }

   return fmin(dt_tst * (1.0 + tol_err) + dt_inc, dt_rem);
}

// ---- ks_solve_column (production :1066-1430) ----------------------------
// `n_out` (outer substeps taken) and `n_in` (summed Picard iterations) are the
// divergence instrumentation: two int stores at the very end of the column.
__device__ void ks_solve_column(int nz, const KsPar &p, double f2_val,
                                const double *h_sd, const double *u_sd,
                                const double *v_sd, const double *t_sd,
                                const double *s_sd, const double *idz_s,
                                const double *idz_int_s, const double *hint_s,
                                const double *il2_s, double *kappa_avg_sd,
                                double *tke_avg_sd, int *n_out, int *n_in) {
   double u_c[NZL + 1], v_c[NZL + 1], t_c[NZL + 1], s_c[NZL + 1];
   double dbuoy_t[NZLI + 1], dbuoy_s[NZLI + 1];
   double e1[NZLI + 1];
   double kappa[NZLI + 1], k_q[NZLI + 1], kappa_avg[NZLI + 1], tke_avg[NZLI + 1];
   double n2[NZLI + 1], s2[NZLI + 1], tke[NZLI + 1];
   double ksrc[NZLI + 1], tkedec[NZLI + 1];
   double aq[NZL + 1], dq[NZLI + 1], cq[NZLI + 1];
   double dk[NZLI + 1], ck[NZLI + 1], ild2[NZLI + 1];
   double cqsav[NZLI + 1];
   double u_ps[NZL + 1], v_ps[NZL + 1], t_ps[NZL + 1], s_ps[NZL + 1];
   double c1_ps[NZLI + 1];
   double n2p[NZLI + 1], s2p[NZLI + 1], n2c[NZLI + 1], s2c[NZLI + 1];
   double kappa_out[NZLI + 1], kq_tmp[NZLI + 1];
   double tke_pred[NZLI + 1], kappa_pred[NZLI + 1], kappa_mid[NZLI + 1];
   double tke_fin[NZLI + 1], kappa_pred2[NZLI + 1];
   double local_src_avg[NZLI + 1], kappa_src[NZLI + 1], local_src[NZLI + 1];
   // ks_adaptive_dt's locals, hoisted (see its signature).
   double tol_max[NZLI + 1], tol_min_a[NZLI + 1], tol_chg_a[NZLI + 1];
   double u_pr[NZL + 1], v_pr[NZL + 1], t_pr[NZL + 1], s_pr[NZL + 1];
   double c1_pr[NZLI + 1], n2_pr[NZLI + 1], s2_pr[NZLI + 1];

   double k0dt = p.dt * p.kappa_0;
   double tke_min = fmax(p.tke_bg, 1.0e-20);
   double c_n2 = p.c_n * p.c_n;
   double c_s2 = p.c_s * p.c_s;
   double ilambda2 = 1.0 / (p.lambda * p.lambda);
   int nit_acc = 0;

   // ---- Background-diffusion pre-step (kappa_0) ----
   if (nz == 1) {
      double b1_l = 1.0 / (h_sd[1] + k0dt * idz_int_s[2]);
      u_c[1] = b1_l * h_sd[1] * u_sd[1];
      v_c[1] = b1_l * h_sd[1] * v_sd[1];
      t_c[1] = t_sd[1];
      s_c[1] = s_sd[1];
   } else {
      double a1n = k0dt * idz_int_s[2];
      double b1_l = 1.0 / (h_sd[1] + a1n);
      cqsav[2] = a1n * b1_l;
      double d1_l = h_sd[1] * b1_l;
      u_c[1] = b1_l * h_sd[1] * u_sd[1];
      v_c[1] = b1_l * h_sd[1] * v_sd[1];
      t_c[1] = b1_l * h_sd[1] * t_sd[1];
      s_c[1] = b1_l * h_sd[1] * s_sd[1];
      double a1c;
      for (int k = 2; k <= nz - 1; ++k) {
         a1c = a1n;
         a1n = k0dt * idz_int_s[k + 1];
         double bd1_l = h_sd[k] + d1_l * a1c;
         b1_l = 1.0 / (bd1_l + a1n);
         u_c[k] = b1_l * (h_sd[k] * u_sd[k] + a1c * u_c[k - 1]);
         v_c[k] = b1_l * (h_sd[k] * v_sd[k] + a1c * v_c[k - 1]);
         t_c[k] = b1_l * (h_sd[k] * t_sd[k] + a1c * t_c[k - 1]);
         s_c[k] = b1_l * (h_sd[k] * s_sd[k] + a1c * s_c[k - 1]);
         cqsav[k + 1] = a1n * b1_l;
         d1_l = bd1_l * b1_l;
      }
      a1c = a1n;
      double base_l = h_sd[nz] + d1_l * a1c;
      double b1ns = 1.0 / (base_l + k0dt * idz_int_s[nz + 1]);
      double b1in = 1.0 / base_l;
      u_c[nz] = b1ns * (h_sd[nz] * u_sd[nz] + a1c * u_c[nz - 1]);
      v_c[nz] = b1ns * (h_sd[nz] * v_sd[nz] + a1c * v_c[nz - 1]);
      t_c[nz] = b1in * (h_sd[nz] * t_sd[nz] + a1c * t_c[nz - 1]);
      s_c[nz] = b1in * (h_sd[nz] * s_sd[nz] + a1c * s_c[nz - 1]);
      cqsav[nz + 1] = 0.0;
      for (int k = nz - 1; k >= 1; --k) {
         u_c[k] = u_c[k] + cqsav[k + 1] * u_c[k + 1];
         v_c[k] = v_c[k] + cqsav[k + 1] * v_c[k + 1];
         t_c[k] = t_c[k] + cqsav[k + 1] * t_c[k + 1];
         s_c[k] = s_c[k] + cqsav[k + 1] * s_c[k + 1];
      }
   }

   // ---- Frozen interface buoyancy derivatives ----
   dbuoy_t[1] = 0.0;
   dbuoy_s[1] = 0.0;
   dbuoy_t[nz + 1] = 0.0;
   dbuoy_s[nz + 1] = 0.0;
   double p_int = 0.0;
   for (int kk = 2; kk <= nz; ++kk) {
      double dpres = GRAVITY * p.rho0 * h_sd[kk - 1];
      p_int = p_int + dpres;
      double t_int = 0.5 * (t_c[kk - 1] + t_c[kk]);
      double s_int = 0.5 * (s_c[kk - 1] + s_c[kk]);
      double dsv_dt_k, dsv_ds_k;
      eos_specvol_derivs(p, t_int, s_int, p_int, &dsv_dt_k, &dsv_ds_k);
      dbuoy_t[kk] = GRAVITY * p.rho0 * dsv_dt_k;
      dbuoy_s[kk] = GRAVITY * p.rho0 * dsv_ds_k;
   }

   // ---- Initial N^2, S^2 ----
   n2[1] = 0.0;
   n2[nz + 1] = 0.0;
   s2[1] = 0.0;
   s2[nz + 1] = 0.0;
   for (int kk = 2; kk <= nz; ++kk) {
      double n2v = idz_int_s[kk] * (dbuoy_t[kk] * (t_c[kk - 1] - t_c[kk]) +
                                    dbuoy_s[kk] * (s_c[kk - 1] - s_c[kk]));
      if (n2v < 0.0) n2v = 0.0;
      n2[kk] = n2v;
      double du = u_c[kk - 1] - u_c[kk], dv = v_c[kk - 1] - v_c[kk];
      s2[kk] = (du * du + dv * dv) * idz_int_s[kk] * idz_int_s[kk];
   }

   // ---- e1 tail recursion ----
   e1[nz + 1] = 0.0;
   double ome_l = 1.0;
   double eden2_l = p.kappa_0 * idz_s[nz];
   for (int kk = nz; kk >= 2; --kk) {
      double eden1_l =
          hint_s[kk] * sqrt(c_n2 * n2[kk] + c_s2 * s2[kk]) + ome_l * eden2_l;
      eden2_l = p.kappa_0 * idz_s[kk - 1];
      double i_eden_l = 1.0 / (eden2_l + eden1_l);
      e1[kk] = eden2_l * i_eden_l;
      ome_l = eden1_l * i_eden_l;
   }
   e1[1] = 0.0;

   // ---- Outer-loop init ----
   for (int kk = 1; kk <= nz + 1; ++kk) {
      k_q[kk] = 0.0;
      kappa_avg[kk] = 0.0;
      tke_avg[kk] = 0.0;
      kappa[kk] = p.kappa_seed;
   }
   kappa[1] = 0.0;
   kappa[nz + 1] = 0.0;
   double dt_rem = p.dt;
   local_src_avg[1] = 0.0;
   local_src_avg[nz + 1] = 0.0;
   for (int kk = 2; kk <= nz; ++kk) {
      if (hint_s[kk] > 0.0)
         local_src_avg[kk] = 0.1 * k0dt * idz_int_s[kk] / hint_s[kk];
      else
         local_src_avg[kk] = 0.0;
   }

   // ---- Adaptive outer substepping ----
   int io;
   for (io = 1; io <= p.max_substep_it; ++io) {
      int ks_src = nz + 2, ke_src = 0;
      ksrc[1] = 0.0;
      ksrc[nz + 1] = 0.0;
      for (int kk = 2; kk <= nz; ++kk) {
         ksrc[kk] = ks_src_func(p.ri_crit, p.shearmix_rate, p.fri_curvature,
                                n2[kk], s2[kk]);
         if (ksrc[kk] > 0.0) {
            if (ks_src > kk) ks_src = kk;
            ke_src = kk;
         }
      }
      for (int kk = 1; kk <= nz + 1; ++kk) kappa_src[kk] = ksrc[kk];

      for (int kk = 2; kk <= nz; ++kk)
         tkedec[kk] = sqrt(c_n2 * n2[kk] + c_s2 * s2[kk]);
      tke[1] = p.tke_bg;
      for (int kk = 2; kk <= nz; ++kk) {
         if (kappa[kk] > 0.0 && k_q[kk] > 0.0)
            tke[kk] = kappa[kk] / k_q[kk];
         else
            tke[kk] = tke_min;
      }
      tke[nz + 1] = tke_min;

      for (int kk = 1; kk <= KS_COPY_TOP; ++kk) kappa_out[kk] = kappa[kk];

      int nit_here = 0;
      if (ks_src > ke_src) {
         for (int kk = 1; kk <= nz + 1; ++kk) kappa_out[kk] = 0.0;
         for (int kk = 2; kk <= nz; ++kk) ild2[kk] = 0.0;
      } else {
         for (int kk = 1; kk <= KS_COPY_TOP; ++kk) kq_tmp[kk] = k_q[kk];
         ks_find_kappa_tke(nz, tke_min, f2_val, p.ri_crit, p.shearmix_rate,
                           p.fri_curvature, c_n2, c_s2, ilambda2, p.kappa_0,
                           p.kappa_trunc, p.tke_bg, p.tol_err, p.max_inner_it,
                           n2, s2, kappa, kq_tmp, idz_s, hint_s, il2_s, e1, tke,
                           kappa_out, ksrc, tkedec, aq, dq, cq, dk, ck, ild2,
                           &nit_here);
         nit_acc += nit_here;
         for (int kk = 1; kk <= nz + 1; ++kk) k_q[kk] = kq_tmp[kk];
      }

      local_src[1] = 0.0;
      local_src[nz + 1] = 0.0;
      for (int kk = 2; kk <= nz; ++kk) {
         if (hint_s[kk] > 0.0) {
            local_src[kk] = ksrc[kk] + p.kappa_0 * ((idz_s[kk - 1] + idz_s[kk]) /
                                                        fmax(hint_s[kk], 1.0e-30) +
                                                    ild2[kk]);
            double n2v = idz_s[kk - 1] * (kappa_out[kk - 1] - kappa_out[kk]) +
                         idz_s[kk] * (kappa_out[kk + 1] - kappa_out[kk]);
            if (n2v > 0.0)
               local_src[kk] = local_src[kk] + n2v / fmax(hint_s[kk], 1.0e-30);
         } else {
            local_src[kk] = ksrc[kk];
         }
      }

      int ks_kap = nz + 2, ke_kap = 0;
      for (int kk = 2; kk <= nz; ++kk) {
         if (kappa_out[kk] > 0.0) {
            if (ks_kap > kk) ks_kap = kk;
            ke_kap = kk;
         }
      }
      if (ke_kap == nz) kappa_out[nz + 1] = 0.0;
      bool no_mixing = (ke_kap < ks_kap);

      double dt_now;
      if (no_mixing || io == p.max_substep_it) {
         dt_now = dt_rem;
      } else {
         dt_now = ks_adaptive_dt(nz, dt_rem, io, p.max_substep_it, p.ri_crit,
                                 p.shearmix_rate, p.fri_curvature, p.src_max_chg,
                                 p.tol_err, p.vel_underflow, dbuoy_t, dbuoy_s,
                                 h_sd, u_c, v_c, t_c, s_c, kappa_out, kappa_src,
                                 local_src, local_src_avg, ks_kap, ke_kap,
                                 idz_int_s, tol_max, tol_min_a, tol_chg_a, u_pr,
                                 v_pr, t_pr, s_pr, c1_pr, n2_pr, s2_pr);
      }
      for (int kk = 2; kk <= nz; ++kk)
         local_src_avg[kk] = local_src_avg[kk] + dt_now * local_src[kk];
      double dt_wt = dt_now / p.dt;

      if (no_mixing) {
         for (int kk = 1; kk <= nz + 1; ++kk)
            tke_avg[kk] = tke_avg[kk] + dt_wt * tke[kk];
         dt_rem = 0.0;
      } else {
         int ks_ps = max(ks_kap - 1, 1);
         int ke_ps = min(ke_kap, nz);
         ks_projected_state(nz, dt_now, ks_ps, ke_ps, p.vel_underflow, dbuoy_t,
                            dbuoy_s, h_sd, idz_int_s, u_c, v_c, t_c, s_c,
                            kappa_out, u_ps, v_ps, t_ps, s_ps, c1_ps, n2p, s2p);
         for (int kk = 1; kk <= KS_COPY_TOP; ++kk) kq_tmp[kk] = k_q[kk];
         ks_find_kappa_tke(nz, tke_min, f2_val, p.ri_crit, p.shearmix_rate,
                           p.fri_curvature, c_n2, c_s2, ilambda2, p.kappa_0,
                           p.kappa_trunc, p.tke_bg, p.tol_err, p.max_inner_it,
                           n2p, s2p, kappa_out, kq_tmp, idz_s, hint_s, il2_s, e1,
                           tke_pred, kappa_pred, ksrc, tkedec, aq, dq, cq, dk,
                           ck, ild2, &nit_here);
         nit_acc += nit_here;
         for (int kk = 1; kk <= nz + 1; ++kk)
            kappa_mid[kk] = 0.5 * (kappa_out[kk] + kappa_pred[kk]);
         int ks_mid = nz + 2, ke_mid = 0;
         for (int kk = 1; kk <= nz + 1; ++kk) {
            if (kappa_mid[kk] > 0.0) {
               if (ks_mid > kk) ks_mid = kk;
               ke_mid = kk;
            }
         }
         ks_ps = max(ks_mid - 1, 1);
         ke_ps = min(ke_mid, nz);

         ks_projected_state(nz, dt_now, ks_ps, ke_ps, p.vel_underflow, dbuoy_t,
                            dbuoy_s, h_sd, idz_int_s, u_c, v_c, t_c, s_c,
                            kappa_mid, u_ps, v_ps, t_ps, s_ps, c1_ps, n2c, s2c);
         ks_find_kappa_tke(nz, tke_min, f2_val, p.ri_crit, p.shearmix_rate,
                           p.fri_curvature, c_n2, c_s2, ilambda2, p.kappa_0,
                           p.kappa_trunc, p.tke_bg, p.tol_err, p.max_inner_it,
                           n2c, s2c, kappa_out, k_q, idz_s, hint_s, il2_s, e1,
                           tke_fin, kappa_pred2, ksrc, tkedec, aq, dq, cq, dk,
                           ck, ild2, &nit_here);
         nit_acc += nit_here;
         dt_rem = dt_rem - dt_now;
         for (int kk = 1; kk <= nz + 1; ++kk) {
            kappa_avg[kk] =
                kappa_avg[kk] + dt_wt * 0.5 * (kappa_out[kk] + kappa_pred2[kk]);
            tke_avg[kk] =
                tke_avg[kk] + dt_wt * 0.5 * (tke_pred[kk] + tke_fin[kk]);
            kappa[kk] = kappa_pred2[kk];
         }
         kappa[1] = 0.0;
         kappa[nz + 1] = 0.0;

         if (dt_rem > 0.0) {
            for (int kk = 1; kk <= nz + 1; ++kk)
               kappa_mid[kk] = 0.5 * (kappa_out[kk] + kappa_pred2[kk]);
            ks_projected_state(nz, dt_now, 1, nz, p.vel_underflow, dbuoy_t,
                               dbuoy_s, h_sd, idz_int_s, u_c, v_c, t_c, s_c,
                               kappa_mid, u_ps, v_ps, t_ps, s_ps, c1_ps, n2, s2);
            for (int k = 1; k <= nz; ++k) {
               u_c[k] = u_ps[k];
               v_c[k] = v_ps[k];
               t_c[k] = t_ps[k];
               s_c[k] = s_ps[k];
            }
         }
      }

      if (dt_rem <= 0.0) break;
   }

   for (int kk = 1; kk <= nz + 1; ++kk) {
      kappa_avg_sd[kk] = kappa_avg[kk];
      tke_avg_sd[kk] = tke_avg[kk];
   }
   kappa_avg_sd[1] = 0.0;
   kappa_avg_sd[nz + 1] = 0.0;

   *n_out = (io > p.max_substep_it) ? p.max_substep_it : io;
   *n_in = nit_acc;
}

// ---- kappa_shear_column_kernel (production :296-466) --------------------
__global__ __launch_bounds__(KS_BLOCK) void k_kappa_shear(
    const double *__restrict__ h_layer, const double *__restrict__ u_face,
    const double *__restrict__ v_face, const double *__restrict__ hT,
    const double *__restrict__ hS, const double *__restrict__ wet_mask,
    const double *__restrict__ f_centre, double *__restrict__ kd_int,
    double *__restrict__ tke_int, int *__restrict__ n_out_a,
    int *__restrict__ n_in_a, int nx, int ny, int nz, KsPar p) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid >= nx * ny) return;
   int i = tid % nx + 1;  // 1-based, matching the Fortran (i fastest)
   int j = tid / nx + 1;

   double h_sd[NZL + 1], u_sd[NZL + 1], v_sd[NZL + 1], t_sd[NZL + 1],
       s_sd[NZL + 1];
   double idz_s[NZL + 1], idz_int_s[NZLI + 1], hint_s[NZLI + 1], il2_s[NZLI + 1];
   double kappa_avg_sd[NZLI + 1], tke_avg_sd[NZLI + 1];
   double dbot[NZLI + 1];  // ks_precompute's local, hoisted

   const size_t nT = (size_t)nx * ny;
   const size_t nU = (size_t)(nx + 1) * ny;
   const size_t nV = (size_t)nx * (ny + 1);

   for (int k = 1; k <= nz + 1; ++k) {
      kappa_avg_sd[k] = 0.0;
      tke_avg_sd[k] = 0.0;
   }

   int n_out = 0, n_in = 0;
   if (wet_mask[(i - 1) + (size_t)(j - 1) * nx] > 0.0) {
      for (int k = 1; k <= nz; ++k) {
         int kg = nz + 1 - k;
         h_sd[k] = h_layer[(i - 1) + (size_t)(j - 1) * nx + (size_t)(kg - 1) * nT];
         u_sd[k] = 0.5 * (u_face[(i - 1) + (size_t)(j - 1) * (nx + 1) +
                                 (size_t)(kg - 1) * nU] +
                          u_face[i + (size_t)(j - 1) * (nx + 1) +
                                 (size_t)(kg - 1) * nU]);
         v_sd[k] = 0.5 * (v_face[(i - 1) + (size_t)(j - 1) * nx +
                                 (size_t)(kg - 1) * nV] +
                          v_face[(i - 1) + (size_t)j * nx + (size_t)(kg - 1) * nV]);
      }

      double fc = f_centre[(i - 1) + (size_t)(j - 1) * nx];
      double f2_val = fc * fc;

      for (int k = 1; k <= nz; ++k) {
         int kg = nz + 1 - k;
         double hk = fmax(h_sd[k], H_VANISHED);
         double inv_h = 1.0 / hk;
         h_sd[k] = hk;
         t_sd[k] = hT[(i - 1) + (size_t)(j - 1) * nx + (size_t)(kg - 1) * nT] * inv_h;
         s_sd[k] = hS[(i - 1) + (size_t)(j - 1) * nx + (size_t)(kg - 1) * nT] * inv_h;
      }

      ks_precompute(nz, p.lz_rescale, h_sd, idz_s, idz_int_s, hint_s, il2_s, dbot);

      ks_solve_column(nz, p, f2_val, h_sd, u_sd, v_sd, t_sd, s_sd, idz_s,
                      idz_int_s, hint_s, il2_s, kappa_avg_sd, tke_avg_sd, &n_out,
                      &n_in);
   }

   const size_t nI = (size_t)nx * ny;
   for (int k = 1; k <= nz + 1; ++k) {
      int kg = nz + 2 - k;
      kd_int[(i - 1) + (size_t)(j - 1) * nx + (size_t)(kg - 1) * nI] = kappa_avg_sd[k];
      tke_int[(i - 1) + (size_t)(j - 1) * nx + (size_t)(kg - 1) * nI] = tke_avg_sd[k];
   }
   kd_int[(i - 1) + (size_t)(j - 1) * nx] = 0.0;
   kd_int[(i - 1) + (size_t)(j - 1) * nx + (size_t)nz * nI] = 0.0;
   tke_int[(i - 1) + (size_t)(j - 1) * nx] = 0.0;
   tke_int[(i - 1) + (size_t)(j - 1) * nx + (size_t)nz * nI] = 0.0;

   if (n_out_a) {
      n_out_a[tid] = n_out;
      n_in_a[tid] = n_in;
   }
}

extern "C" void ks_cuda_launch(const double *h_layer, const double *u_face,
                               const double *v_face, const double *hT,
                               const double *hS, const double *wet_mask,
                               const double *f_centre, double *kd_int,
                               double *tke_int, int *n_out_a, int *n_in_a,
                               int nx, int ny, int nz, const KsPar *pin,
                               int sync) {
   KsPar p = *pin;
   int n = nx * ny;
   int nb = (n + KS_BLOCK - 1) / KS_BLOCK;
   k_kappa_shear<<<nb, KS_BLOCK>>>(h_layer, u_face, v_face, hT, hS, wet_mask,
                                   f_centre, kd_int, tke_int, n_out_a, n_in_a,
                                   nx, ny, nz, p);
   if (sync) {
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess)
         printf("  *** CUDA error (faithful): %s\n", cudaGetErrorString(e));
   }
}

// Barrier only. Lets a Fortran driver time N ASYNCHRONOUS launches followed by
// one synchronise -- the same timing window the `do concurrent` side gets from
// `!$acc wait`. The alternatives both bias the measurement: a sync per rep taxes
// small grids, and an extra forced-sync launch inflates the rep count.
extern "C" void ks_cuda_sync(void) {
   cudaError_t e = cudaDeviceSynchronize();
   if (e != cudaSuccess)
      printf("  *** CUDA error (sync): %s\n", cudaGetErrorString(e));
}

// Reports the compiled frame size / register count straight from the driver --
// the authoritative number, not a ptxas dump that has to be name-matched.
extern "C" void ks_cuda_attrs(int *nregs, int *local_bytes, int *shared_bytes,
                              int *max_tpb) {
   cudaFuncAttributes a;
   cudaFuncGetAttributes(&a, (const void *)k_kappa_shear);
   *nregs = a.numRegs;
   *local_bytes = (int)a.localSizeBytes;
   *shared_bytes = (int)a.sharedSizeBytes;
   *max_tpb = a.maxThreadsPerBlock;
}
