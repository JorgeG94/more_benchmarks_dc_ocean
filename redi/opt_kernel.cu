// ===========================================================================
// OPTIMIZED CUDA C port of the Redi neutral-diffusion apply-flux kernel.
//
// Freed from the "faithful transliteration" rule, but held BIT-IDENTICAL to
// redi_kernel.cu: every floating-point expression is evaluated in the SAME
// order, so ab_main.cu checks max|diff| == 0 on both tracers.
//
// THE WIN -- precompute hoist in redi_apply_flux (the analogue of the
// continuity fusion win, and the exact lever the README calls out):
//
//   The faithful kRediApplyFlux visits every cell (i,j) and evaluates FOUR
//   face fluxes (W/E/S/N). Each face flux reconstructs its LEFT and its RIGHT
//   tracer column from scratch (redi_tracer_column: PLM slopes + PPM edges +
//   per-layer limiter over the whole column). That is 8 column rebuilds per
//   cell -- but only 5 DISTINCT columns are involved:
//         (i,j)   <- used by ALL FOUR faces (W-right, E-left, S-right, N-left)
//         (i-1,j) <- W-left      (i+1,j) <- E-right
//         (i,j-1) <- S-left      (i,j+1) <- N-right
//   The centre column is rebuilt 4x. Hoisting it -- reconstruct (i,j) ONCE,
//   reuse it in all four faces -- cuts 8 rebuilds to 5. redi_tracer_column is
//   a deterministic pure function of the column data, so computing it once vs
//   four times gives bit-identical values: the result is bit-exact, not just
//   <1e-12. The accumulation order into dTr (W,E,S,N; ks=1..ns-1 within each)
//   is preserved exactly.
//
// Secondary: 32-bit indexing (IDX32, default on) -- the domain fits int32
// (u-face x nsurf ~ 9.0e6 << 2.1e9), so the size_t address math in the
// faithful port is pure overhead. FP values are unchanged -> still bit-exact.
//
// The Phase-A coeff kernels, the snapshot and the face-copy are transcribed
// unchanged (renamed to avoid symbol clash with redi_kernel.cu at link time);
// only kOptApplyFlux carries the hoist.
// ===========================================================================
#include "gpu_rt.h"
#include <math.h>

#ifndef NZ_STACK_MAX
#define NZ_STACK_MAX 128
#endif
#define NZS NZ_STACK_MAX

#define GRAVITY   9.80665
#define H_DIV_EPS 1.0e-20

// 32-bit indexing (default). Valid while every flat index < 2^31; the largest
// here is (nx+1)*ny*nsurf ~ 9.0e6 at the 473x297x30 config. FP-neutral: only
// address arithmetic changes, so bit-identicality to the faithful port holds.
#ifndef IDX32
#define IDX32 1
#endif
#if IDX32
#define I3(p, i, j, k, n1, n2) ((p)[(((int)((k)-1)*(int)(n2) + (int)((j)-1)) * (int)(n1)) + (int)((i)-1)])
#define I2(p, i, j, n1)        ((p)[((int)((j)-1) * (int)(n1)) + (int)((i)-1)])
#else
#define I3(p, i, j, k, n1, n2) ((p)[(((size_t)((k)-1)*(size_t)(n2) + (size_t)((j)-1)) * (size_t)(n1)) + (size_t)((i)-1)])
#define I2(p, i, j, n1)        ((p)[((size_t)((j)-1) * (size_t)(n1)) + (size_t)((i)-1)])
#endif

// Everything except the extern "C" bridge lives in an anonymous namespace, so
// the device helpers and __global__s get INTERNAL linkage and do not collide
// with redi_kernel.cu's identically-named symbols when both objects link.
namespace {

struct EosLin {
   double rho0, T_ref, S_ref, alpha_T, beta_S;
};

// ---- device helpers: transcribed VERBATIM from redi_kernel.cu --------------
// (same math, same order -> bit-identical). Only apply-flux changes below.
__device__ __forceinline__ double eos_density_point(const EosLin &e, double T, double S, double p) {
   return e.rho0 + e.beta_S * (S - e.S_ref) - e.alpha_T * (T - e.T_ref);
}
__device__ __forceinline__ void eos_specvol_derivs(const EosLin &e, double T, double S, double p,
                                                   double &dsv_dt, double &dsv_ds) {
   dsv_dt = e.alpha_T / (e.rho0 * e.rho0);
   dsv_ds = -e.beta_S / (e.rho0 * e.rho0);
}
__device__ __forceinline__ double redi_fv_diff(double hkm1, double hk, double hkp1,
                                               double skm1, double sk, double skp1) {
   double h_sum = (hkm1 + hkp1) + hk;
   if (h_sum != 0.0) h_sum = 1.0 / h_sum;
   double hm = hkm1 + hk;
   if (hm != 0.0) hm = 1.0 / hm;
   double hp = hkp1 + hk;
   if (hp != 0.0) hp = 1.0 / hp;
   return (hk * h_sum) * ((2.0 * hkm1 + hk) * hp * (skp1 - sk) + (2.0 * hkp1 + hk) * hm * (sk - skm1));
}
__device__ __forceinline__ double redi_signum(double a, double x) {
   double s = copysign(a, x);
   if (x == 0.0) s = 0.0;
   return s;
}
__device__ __forceinline__ double redi_signum1(double x) {
   double s = copysign(1.0, x);
   if (x == 0.0) s = 0.0;
   return s;
}
__device__ void redi_plm_diff(int nk, const double *h, const double *s, double *diff) {
   for (int k = 2; k <= nk - 1; ++k) {
      if ((h[k] + h[k - 1]) * (h[k - 2] + h[k - 1]) > 0.0) {
         double diff_c = redi_fv_diff(h[k - 2], h[k - 1], h[k], s[k - 2], s[k - 1], s[k]);
         double diff_l = 2.0 * (s[k - 1] - s[k - 2]);
         double diff_r = 2.0 * (s[k] - s[k - 1]);
         if (redi_signum(1.0, diff_l) * redi_signum(1.0, diff_r) <= 0.0) {
            diff[k - 1] = 0.0;
         } else {
            double am = fmin(fmin(fabs(diff_l), fabs(diff_c)), fabs(diff_r));
            diff[k - 1] = copysign(am, diff_c);
         }
      } else {
         diff[k - 1] = 0.0;
      }
   }
   diff[0] = 0.0;
   diff[nk - 1] = 0.0;
}
__device__ double redi_ppm_edge(double hkm1, double hk, double hkp1, double hkp2,
                                double ak, double akp1, double pk, double pkp1, double h_neglect) {
   double r_hk_hkp1 = hk + hkp1;
   if (r_hk_hkp1 <= 0.0) return 0.5 * (ak + akp1);
   r_hk_hkp1 = 1.0 / r_hk_hkp1;
   double e;
   if (hk < hkp1) e = ak + (hk * r_hk_hkp1) * (akp1 - ak);
   else           e = akp1 + (hkp1 * r_hk_hkp1) * (ak - akp1);

   double r_2hk_hkp1 = 1.0 / ((2.0 * hk + hkp1) + h_neglect);
   double r_hk_2hkp1 = 1.0 / ((hk + 2.0 * hkp1) + h_neglect);
   double f1 = 1.0 / ((hk + hkp1) + (hkm1 + hkp2));
   double f2 = 2.0 * (hkp1 * hk) * r_hk_hkp1 * ((hkm1 + hk) * r_2hk_hkp1 - (hkp2 + hkp1) * r_hk_2hkp1);
   double f3 = hk * (hkm1 + hk) * r_2hk_hkp1;
   double f4 = hkp1 * (hkp1 + hkp2) * r_hk_2hkp1;
   return e + f1 * (f2 * (akp1 - ak) - (f3 * pkp1 - f4 * pk));
}
__device__ void redi_interface_scalar(int nk, const double *h, const double *tr, double *edge) {
   double diff[NZS];
   const double hn = 1.0e-30;
   redi_plm_diff(nk, h, tr, diff);
   edge[0] = tr[0] - 0.5 * diff[0];
   for (int k = 2; k <= nk; ++k) {
      int km2 = max(1, k - 2);
      int kp1 = min(nk, k + 1);
      edge[k - 1] = redi_ppm_edge(h[km2 - 1], h[k - 2], h[k - 1], h[kp1 - 1],
                                  tr[k - 2], tr[k - 1], diff[k - 2], diff[k - 1], hn);
   }
   edge[nk] = tr[nk - 1] + 0.5 * diff[nk - 1];
}
__device__ double redi_interpolate_position(double dRhoNeg, double Pneg, double dRhoPos, double Ppos) {
   if ((Ppos > Pneg) && (dRhoPos - dRhoNeg >= 0.0)) {
      if (dRhoPos - dRhoNeg > 0.0) return fmin(1.0, fmax(0.0, -dRhoNeg / (dRhoPos - dRhoNeg)));
      if (dRhoNeg > 0.0) return 0.0;
      if (dRhoNeg < 0.0) return 1.0;
      return 0.5;
   } else if (Ppos == Pneg) {
      return 0.5;
   }
   return 0.5;
}
__device__ __forceinline__ double redi_absolute_position(const double *Pint, int Karr, double frac) {
   return Pint[Karr - 1] + frac * (Pint[Karr] - Pint[Karr - 1]);
}
__device__ void redi_neutral_positions_continuous(
    int nk, const double *Pl, const double *Tl, const double *Sl, const double *dRdTl, const double *dRdSl,
    const double *Pr, const double *Tr, const double *Sr, const double *dRdTr, const double *dRdSr,
    double *PoL, double *PoR, int *KoL, int *KoR, double *hEff) {
   int ns = 2 * nk + 2;
   int kr = 1, kl = 1;
   double lastP_right = 0.0, lastP_left = 0.0;
   int lastK_right = 1, lastK_left = 1;
   bool reached_bottom = false, searching_left_column = false, searching_right_column = false;

   for (int k_surface = 1; k_surface <= ns; ++k_surface) {
      int klm1 = max(kl - 1, 1);
      int krm1 = max(kr - 1, 1);
      double dRho = 0.5 * ((dRdTr[kr - 1] + dRdTl[kl - 1]) * (Tr[kr - 1] - Tl[kl - 1]) +
                           (dRdSr[kr - 1] + dRdSl[kl - 1]) * (Sr[kr - 1] - Sl[kl - 1]));
      if (!reached_bottom) {
         if (dRho < 0.0) { searching_left_column = true;  searching_right_column = false; }
         else if (dRho > 0.0) { searching_right_column = true; searching_left_column = false; }
         else {
            if (kl + kr == 2) { searching_left_column = true; searching_right_column = false; }
            else { searching_left_column = !searching_left_column; searching_right_column = !searching_right_column; }
         }
      }

      if (searching_left_column) {
         double dRhoTop = 0.5 * ((dRdTl[klm1 - 1] + dRdTr[kr - 1]) * (Tl[klm1 - 1] - Tr[kr - 1]) +
                                 (dRdSl[klm1 - 1] + dRdSr[kr - 1]) * (Sl[klm1 - 1] - Sr[kr - 1]));
         double dRhoBot = 0.5 * ((dRdTl[klm1] + dRdTr[kr - 1]) * (Tl[klm1] - Tr[kr - 1]) +
                                 (dRdSl[klm1] + dRdSr[kr - 1]) * (Sl[klm1] - Sr[kr - 1]));
         if (dRhoTop > 0.0 || kr + kl == 2) PoL[k_surface - 1] = 0.0;
         else if (dRhoTop >= dRhoBot)       PoL[k_surface - 1] = 1.0;
         else PoL[k_surface - 1] = redi_interpolate_position(dRhoTop, Pl[klm1 - 1], dRhoBot, Pl[klm1]);
         if (PoL[k_surface - 1] >= 1.0 && klm1 < nk) { klm1 += 1; PoL[k_surface - 1] -= 1.0; }
         if ((double)(klm1 - lastK_left) + (PoL[k_surface - 1] - lastP_left) < 0.0) {
            PoL[k_surface - 1] = lastP_left; klm1 = lastK_left;
         }
         KoL[k_surface - 1] = klm1;
         if (kr <= nk) { PoR[k_surface - 1] = 0.0; KoR[k_surface - 1] = kr; }
         else          { PoR[k_surface - 1] = 1.0; KoR[k_surface - 1] = nk; }
         if (kr <= nk) kr += 1;
         else { reached_bottom = true; searching_right_column = true; searching_left_column = false; }
      } else if (searching_right_column) {
         double dRhoTop = 0.5 * ((dRdTr[krm1 - 1] + dRdTl[kl - 1]) * (Tr[krm1 - 1] - Tl[kl - 1]) +
                                 (dRdSr[krm1 - 1] + dRdSl[kl - 1]) * (Sr[krm1 - 1] - Sl[kl - 1]));
         double dRhoBot = 0.5 * ((dRdTr[krm1] + dRdTl[kl - 1]) * (Tr[krm1] - Tl[kl - 1]) +
                                 (dRdSr[krm1] + dRdSl[kl - 1]) * (Sr[krm1] - Sl[kl - 1]));
         if (dRhoTop >= 0.0 || kr + kl == 2) PoR[k_surface - 1] = 0.0;
         else if (dRhoTop >= dRhoBot)        PoR[k_surface - 1] = 1.0;
         else PoR[k_surface - 1] = redi_interpolate_position(dRhoTop, Pr[krm1 - 1], dRhoBot, Pr[krm1]);
         if (PoR[k_surface - 1] >= 1.0 && krm1 < nk) { krm1 += 1; PoR[k_surface - 1] -= 1.0; }
         if ((double)(krm1 - lastK_right) + (PoR[k_surface - 1] - lastP_right) < 0.0) {
            PoR[k_surface - 1] = lastP_right; krm1 = lastK_right;
         }
         KoR[k_surface - 1] = krm1;
         if (kl <= nk) { PoL[k_surface - 1] = 0.0; KoL[k_surface - 1] = kl; }
         else          { PoL[k_surface - 1] = 1.0; KoL[k_surface - 1] = nk; }
         if (kl <= nk) kl += 1;
         else { reached_bottom = true; searching_right_column = false; searching_left_column = true; }
      }

      lastK_left  = KoL[k_surface - 1];
      lastP_left  = PoL[k_surface - 1];
      lastK_right = KoR[k_surface - 1];
      lastP_right = PoR[k_surface - 1];

      if (k_surface > 1) {
         double hL = redi_absolute_position(Pl, KoL[k_surface - 1], PoL[k_surface - 1]) -
                     redi_absolute_position(Pl, KoL[k_surface - 2], PoL[k_surface - 2]);
         double hR = redi_absolute_position(Pr, KoR[k_surface - 1], PoR[k_surface - 1]) -
                     redi_absolute_position(Pr, KoR[k_surface - 2], PoR[k_surface - 2]);
         if (hL + hR > 0.0) hEff[k_surface - 2] = 2.0 * hL * hR / (hL + hR);
         else               hEff[k_surface - 2] = 0.0;
      }
   }
}
__device__ void redi_build_column(int nz, const double *h_col, const double *thtr_col,
                                  const double *shtr_col, const EosLin &eos,
                                  double *Pint, double *Tint, double *Sint, double *dRdT, double *dRdS) {
   double htd[NZS], ttd[NZS], std_[NZS];
   for (int k = 1; k <= nz; ++k) {
      int kf = nz + 1 - k;
      double he = fmax(h_col[k - 1], H_DIV_EPS);
      htd[kf - 1] = h_col[k - 1];
      ttd[kf - 1] = thtr_col[k - 1] / he;
      std_[kf - 1] = shtr_col[k - 1] / he;
   }
   redi_interface_scalar(nz, htd, ttd, Tint);
   redi_interface_scalar(nz, htd, std_, Sint);
   Pint[0] = 0.0;
   for (int k = 1; k <= nz; ++k) Pint[k] = Pint[k - 1] + GRAVITY * eos.rho0 * htd[k - 1];
   for (int k = 1; k <= nz + 1; ++k) {
      double rho_i = eos_density_point(eos, Tint[k - 1], Sint[k - 1], Pint[k - 1]);
      double dsv_dt, dsv_ds;
      eos_specvol_derivs(eos, Tint[k - 1], Sint[k - 1], Pint[k - 1], dsv_dt, dsv_ds);
      dRdT[k - 1] = -(rho_i * rho_i) * dsv_dt;
      dRdS[k - 1] = -(rho_i * rho_i) * dsv_ds;
   }
}
__device__ void redi_face_coeffs(int nz, int ns, const double *hL, const double *tL, const double *sL,
                                 const double *hR, const double *tR, const double *sR, const EosLin &eos,
                                 double *PoLo, double *PoRo, int *KoLo, int *KoRo, double *hEffo) {
   double Pl[NZS + 1], Tli[NZS + 1], Sli[NZS + 1], dRdTl[NZS + 1], dRdSl[NZS + 1];
   double Pr[NZS + 1], Tri[NZS + 1], Sri[NZS + 1], dRdTr[NZS + 1], dRdSr[NZS + 1];
   redi_build_column(nz, hL, tL, sL, eos, Pl, Tli, Sli, dRdTl, dRdSl);
   redi_build_column(nz, hR, tR, sR, eos, Pr, Tri, Sri, dRdTr, dRdSr);
   redi_neutral_positions_continuous(nz, Pl, Tli, Sli, dRdTl, dRdSl,
                                     Pr, Tri, Sri, dRdTr, dRdSr, PoLo, PoRo, KoLo, KoRo, hEffo);
   double pa_to_h = 1.0 / (GRAVITY * eos.rho0);
   for (int ks = 1; ks <= ns - 1; ++ks) hEffo[ks - 1] *= pa_to_h;
}
__device__ __forceinline__ double redi_ppm_ave(double xL, double xR, double aL, double aR, double aMean) {
   double dx = xR - xL;
   double xave = 0.5 * (xR + xL);
   double a6o3 = 2.0 * aMean - (aL + aR);
   double a6 = 3.0 * a6o3;
   if (dx > 0.0 && dx <= 1.0) return (aL + xave * ((aR - aL) + a6)) - a6o3 * (xR * xR + xR * xL + xL * xL);
   return aL + (aR - aL) * xR + a6 * xR * (1.0 - xR);
}
__device__ void redi_tracer_column(int nz, const double *h_col, const double *htr_col,
                                   double *Tlay, double *Tint, double *aLe, double *aRe) {
   double htd[NZS], tedge[NZS + 1];
   for (int k = 1; k <= nz; ++k) {
      int kf = nz + 1 - k;
      double he = fmax(h_col[k - 1], H_DIV_EPS);
      htd[kf - 1] = h_col[k - 1];
      Tlay[kf - 1] = htr_col[k - 1] / he;
   }
   redi_interface_scalar(nz, htd, Tlay, tedge);
   for (int k = 1; k <= nz + 1; ++k) Tint[k - 1] = tedge[k - 1];
   for (int k = 1; k <= nz; ++k) {
      double alk = Tint[k - 1];
      double ark = Tint[k];
      double tlk = Tlay[k - 1];
      if (redi_signum1(ark - tlk) * redi_signum1(tlk - alk) <= 0.0) {
         alk = tlk; ark = tlk;
      } else if (copysign(3.0, ark - alk) * ((tlk - alk) + (tlk - ark)) > fabs(ark - alk)) {
         alk = tlk + 2.0 * (tlk - ark);
      } else if (copysign(3.0, ark - alk) * ((tlk - alk) + (tlk - ark)) < -fabs(ark - alk)) {
         ark = tlk + 2.0 * (tlk - alk);
      }
      aLe[k - 1] = alk;
      aRe[k - 1] = ark;
   }
}
__device__ double redi_sublayer_dT(int klt, int klb, int krt, int krb,
                                   double PoLt, double PoLb, double PoRt, double PoRb,
                                   const double *TlL, const double *TiL, const double *aLL, const double *aRL,
                                   const double *TlR, const double *TiR, const double *aLR, const double *aRR) {
   double tlt = (1.0 - PoLt) * TiL[klt - 1] + PoLt * TiL[klt];
   double tlb = (1.0 - PoLb) * TiL[klb - 1] + PoLb * TiL[klb];
   double trt = (1.0 - PoRt) * TiR[krt - 1] + PoRt * TiR[krt];
   double trb = (1.0 - PoRb) * TiR[krb - 1] + PoRb * TiR[krb];
   double tlay  = redi_ppm_ave(PoLt, PoLb + (double)(klb - klt), aLL[klt - 1], aRL[klt - 1], TlL[klt - 1]);
   double trlay = redi_ppm_ave(PoRt, PoRb + (double)(krb - krt), aLR[krt - 1], aRR[krt - 1], TlR[krt - 1]);
   double dT_top = trt - tlt;
   double dT_bot = trb - tlb;
   double dT_ave = 0.5 * (dT_top + dT_bot);
   double dT_layer = trlay - tlay;
   if (redi_signum1(dT_top) * redi_signum1(dT_bot) <= 0.0 ||
       redi_signum1(dT_ave) * redi_signum1(dT_layer) <= 0.0) return 0.0;
   return dT_layer;
}

// ===========================================================================
// Phase-A coeff kernels + snapshot + face-copy: transcribed unchanged from
// redi_kernel.cu, renamed (kOpt*) so the two object files link side by side.
// ===========================================================================
__global__ void kOptCalcCoeffsX(int nx, int ny, int nz, int ns, EosLin eos,
                                const double * __restrict__ h_layer, const double * __restrict__ t_htr, const double * __restrict__ s_htr,
                                const double * __restrict__ wet_u, double * __restrict__ uPoL, double * __restrict__ uPoR,
                                int *uKoL, int *uKoR, double * __restrict__ uhEff) {
   long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
   long long ni = nx - 1;
   if (tid >= ni * (long long)ny) return;
   int i = (int)(tid % ni) + 2;
   int j = (int)(tid / ni) + 1;
   const int nfa = nx + 1;

   for (int s = 1; s <= ns; ++s) {
      I3(uPoL, i, j, s, nfa, ny) = 0.0;
      I3(uPoR, i, j, s, nfa, ny) = 0.0;
      I3(uKoL, i, j, s, nfa, ny) = 1;
      I3(uKoR, i, j, s, nfa, ny) = 1;
   }
   for (int s = 1; s <= ns - 1; ++s) I3(uhEff, i, j, s, nfa, ny) = 0.0;

   if (I2(wet_u, i, j, nfa) > 0.0) {
      double hL[NZS], tcL[NZS], scL[NZS], hR[NZS], tcR[NZS], scR[NZS];
      double PoLc[2 * NZS + 2], PoRc[2 * NZS + 2], hEc[2 * NZS + 1];
      int KoLc[2 * NZS + 2], KoRc[2 * NZS + 2];
      for (int k = 1; k <= nz; ++k) {
         hL[k - 1]  = I3(h_layer, i - 1, j, k, nx, ny);
         tcL[k - 1] = I3(t_htr,   i - 1, j, k, nx, ny);
         scL[k - 1] = I3(s_htr,   i - 1, j, k, nx, ny);
         hR[k - 1]  = I3(h_layer, i,     j, k, nx, ny);
         tcR[k - 1] = I3(t_htr,   i,     j, k, nx, ny);
         scR[k - 1] = I3(s_htr,   i,     j, k, nx, ny);
      }
      redi_face_coeffs(nz, ns, hL, tcL, scL, hR, tcR, scR, eos, PoLc, PoRc, KoLc, KoRc, hEc);
      for (int s = 1; s <= ns; ++s) {
         I3(uPoL, i, j, s, nfa, ny) = PoLc[s - 1];
         I3(uPoR, i, j, s, nfa, ny) = PoRc[s - 1];
         I3(uKoL, i, j, s, nfa, ny) = KoLc[s - 1];
         I3(uKoR, i, j, s, nfa, ny) = KoRc[s - 1];
      }
      for (int s = 1; s <= ns - 1; ++s) I3(uhEff, i, j, s, nfa, ny) = hEc[s - 1];
   }
}
__global__ void kOptCalcCoeffsY(int nx, int ny, int nz, int ns, EosLin eos,
                                const double * __restrict__ h_layer, const double * __restrict__ t_htr, const double * __restrict__ s_htr,
                                const double * __restrict__ wet_v, double * __restrict__ vPoL, double * __restrict__ vPoR,
                                int *vKoL, int *vKoR, double * __restrict__ vhEff) {
   long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
   long long nj = ny - 1;
   if (tid >= (long long)nx * nj) return;
   int i = (int)(tid % nx) + 1;
   int j = (int)(tid / nx) + 2;
   const int nfb = ny + 1;

   for (int s = 1; s <= ns; ++s) {
      I3(vPoL, i, j, s, nx, nfb) = 0.0;
      I3(vPoR, i, j, s, nx, nfb) = 0.0;
      I3(vKoL, i, j, s, nx, nfb) = 1;
      I3(vKoR, i, j, s, nx, nfb) = 1;
   }
   for (int s = 1; s <= ns - 1; ++s) I3(vhEff, i, j, s, nx, nfb) = 0.0;

   if (I2(wet_v, i, j, nx) > 0.0) {
      double hL[NZS], tcL[NZS], scL[NZS], hR[NZS], tcR[NZS], scR[NZS];
      double PoLc[2 * NZS + 2], PoRc[2 * NZS + 2], hEc[2 * NZS + 1];
      int KoLc[2 * NZS + 2], KoRc[2 * NZS + 2];
      for (int k = 1; k <= nz; ++k) {
         hL[k - 1]  = I3(h_layer, i, j - 1, k, nx, ny);
         tcL[k - 1] = I3(t_htr,   i, j - 1, k, nx, ny);
         scL[k - 1] = I3(s_htr,   i, j - 1, k, nx, ny);
         hR[k - 1]  = I3(h_layer, i, j,     k, nx, ny);
         tcR[k - 1] = I3(t_htr,   i, j,     k, nx, ny);
         scR[k - 1] = I3(s_htr,   i, j,     k, nx, ny);
      }
      redi_face_coeffs(nz, ns, hL, tcL, scL, hR, tcR, scR, eos, PoLc, PoRc, KoLc, KoRc, hEc);
      for (int s = 1; s <= ns; ++s) {
         I3(vPoL, i, j, s, nx, nfb) = PoLc[s - 1];
         I3(vPoR, i, j, s, nx, nfb) = PoRc[s - 1];
         I3(vKoL, i, j, s, nx, nfb) = KoLc[s - 1];
         I3(vKoR, i, j, s, nx, nfb) = KoRc[s - 1];
      }
      for (int s = 1; s <= ns - 1; ++s) I3(vhEff, i, j, s, nx, nfb) = hEc[s - 1];
   }
}
__global__ void kOptSnapshot(int n, const double * __restrict__ src, double * __restrict__ dst) {
   long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
   if (t < n) dst[t] = src[t];
}
__global__ void kOptFaceCopy(int n, const double * __restrict__ src, double * __restrict__ dst) {
   long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
   if (t < n) dst[t] = src[t];
}

// ===========================================================================
// OPTIMIZED apply-flux. The precompute hoist lives here.
// ===========================================================================

// Reconstruct one cell's tracer column (Tlay/Tint/aLe/aRe) from raw layer data.
__device__ __forceinline__ void redi_recon_cell(int nz, const double *h_layer, const double *hTr_in,
                                                int i, int j, int nxc, int nyc,
                                                double *Tlay, double *Tint, double *aLe, double *aRe) {
   double hcol[NZS], trcol[NZS];
   for (int k = 1; k <= nz; ++k) {
      hcol[k - 1]  = I3(h_layer, i, j, k, nxc, nyc);
      trcol[k - 1] = I3(hTr_in,  i, j, k, nxc, nyc);
   }
   redi_tracer_column(nz, hcol, trcol, Tlay, Tint, aLe, aRe);
}

// The sublayer sweep for ONE face, given BOTH columns already reconstructed.
// Same ks order and same dTr accumulation as the faithful redi_face_flux.
__device__ __forceinline__ void redi_face_flux_pre(int nz, int ns, int nfa, int nfb,
                               int fa, int fb,
                               const double *PoL, const double *PoR, const int *KoL, const int *KoR,
                               const double *hEff, double coef, bool is_left,
                               const double *TlL, const double *TiL, const double *aLL, const double *aRL,
                               const double *TlR, const double *TiR, const double *aLR, const double *aRR,
                               double *dTr) {
   for (int ks = 1; ks <= ns - 1; ++ks) {
      double he = I3(hEff, fa, fb, ks, nfa, nfb);
      if (he != 0.0) {
         double dtdiff = redi_sublayer_dT(I3(KoL, fa, fb, ks, nfa, nfb), I3(KoL, fa, fb, ks + 1, nfa, nfb),
                                          I3(KoR, fa, fb, ks, nfa, nfb), I3(KoR, fa, fb, ks + 1, nfa, nfb),
                                          I3(PoL, fa, fb, ks, nfa, nfb), I3(PoL, fa, fb, ks + 1, nfa, nfb),
                                          I3(PoR, fa, fb, ks, nfa, nfb), I3(PoR, fa, fb, ks + 1, nfa, nfb),
                                          TlL, TiL, aLL, aRL, TlR, TiR, aLR, aRR);
         double flx = dtdiff * he * coef;
         if (is_left) {
            int knat = nz + 1 - I3(KoL, fa, fb, ks, nfa, nfb);
            dTr[knat - 1] += flx;
         } else {
            int knat = nz + 1 - I3(KoR, fa, fb, ks, nfa, nfb);
            dTr[knat - 1] -= flx;
         }
      }
   }
}

__global__ void kOptApplyFlux(int nx, int ny, int nz, int ns, double dt,
                              int nghost, int nxp, int nyp,
                              int wall_w, int wall_e, int wall_s, int wall_n,
                              const double * __restrict__ khtr_u, const double * __restrict__ khtr_v,
                              const double * __restrict__ dy_cu, const double * __restrict__ dx_cv,
                              const double * __restrict__ idxCu, const double * __restrict__ idyCv, const double * __restrict__ areaT,
                              const double * __restrict__ h_layer, const double * __restrict__ hTr_in, double * __restrict__ hTr,
                              const double * __restrict__ uPoL, const double * __restrict__ uPoR, const int *uKoL, const int *uKoR,
                              const double * __restrict__ uhEff,
                              const double * __restrict__ vPoL, const double * __restrict__ vPoR, const int *vKoL, const int *vKoR,
                              const double * __restrict__ vhEff) {
   long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
   if (tid >= (long long)nx * (long long)ny) return;
   int i = (int)(tid % nx) + 1;
   int j = (int)(tid / nx) + 1;

   const int wuf_w = nghost + 1;
   const int wuf_e = nghost + nxp + 1;
   const int wvf_s = nghost + 1;
   const int wvf_n = nghost + nyp + 1;

   double dTr[NZS];
   for (int k = 1; k <= nz; ++k) dTr[k - 1] = 0.0;

   // --- THE HOIST: reconstruct the centre column (i,j) ONCE. It is the RIGHT
   //     column of the W/S faces and the LEFT column of the E/N faces. ---
   double TlC[NZS], TiC[NZS + 1], aLC[NZS], aRC[NZS];
   redi_recon_cell(nz, h_layer, hTr_in, i, j, nx, ny, TlC, TiC, aLC, aRC);
   // A single reusable buffer for the (distinct) neighbour column per face.
   double TlN[NZS], TiN[NZS + 1], aLN[NZS], aRN[NZS];

   // WEST u-face (face i): this cell is the RIGHT column; neighbour (i-1,j) LEFT.
   if (i >= 2 && !((wall_w && i == wuf_w) || (wall_e && i == wuf_e))) {
      redi_recon_cell(nz, h_layer, hTr_in, i - 1, j, nx, ny, TlN, TiN, aLN, aRN);
      double coef = dt * I2(khtr_u, i, j, nx + 1) * I2(dy_cu, i, j, nx + 1) * I2(idxCu, i, j, nx + 1);
      redi_face_flux_pre(nz, ns, nx + 1, ny, i, j, uPoL, uPoR, uKoL, uKoR, uhEff, coef, false,
                         TlN, TiN, aLN, aRN, TlC, TiC, aLC, aRC, dTr);
   }
   // EAST u-face (face i+1): this cell is the LEFT column; neighbour (i+1,j) RIGHT.
   if (i <= nx - 1 && !((wall_w && i + 1 == wuf_w) || (wall_e && i + 1 == wuf_e))) {
      redi_recon_cell(nz, h_layer, hTr_in, i + 1, j, nx, ny, TlN, TiN, aLN, aRN);
      double coef = dt * I2(khtr_u, i + 1, j, nx + 1) * I2(dy_cu, i + 1, j, nx + 1) * I2(idxCu, i + 1, j, nx + 1);
      redi_face_flux_pre(nz, ns, nx + 1, ny, i + 1, j, uPoL, uPoR, uKoL, uKoR, uhEff, coef, true,
                         TlC, TiC, aLC, aRC, TlN, TiN, aLN, aRN, dTr);
   }
   // SOUTH v-face (face j): this cell is the RIGHT column; neighbour (i,j-1) LEFT.
   if (j >= 2 && !((wall_s && j == wvf_s) || (wall_n && j == wvf_n))) {
      redi_recon_cell(nz, h_layer, hTr_in, i, j - 1, nx, ny, TlN, TiN, aLN, aRN);
      double coef = dt * I2(khtr_v, i, j, nx) * I2(dx_cv, i, j, nx) * I2(idyCv, i, j, nx);
      redi_face_flux_pre(nz, ns, nx, ny + 1, i, j, vPoL, vPoR, vKoL, vKoR, vhEff, coef, false,
                         TlN, TiN, aLN, aRN, TlC, TiC, aLC, aRC, dTr);
   }
   // NORTH v-face (face j+1): this cell is the LEFT column; neighbour (i,j+1) RIGHT.
   if (j <= ny - 1 && !((wall_s && j + 1 == wvf_s) || (wall_n && j + 1 == wvf_n))) {
      redi_recon_cell(nz, h_layer, hTr_in, i, j + 1, nx, ny, TlN, TiN, aLN, aRN);
      double coef = dt * I2(khtr_v, i, j + 1, nx) * I2(dx_cv, i, j + 1, nx) * I2(idyCv, i, j + 1, nx);
      redi_face_flux_pre(nz, ns, nx, ny + 1, i, j + 1, vPoL, vPoR, vKoL, vKoR, vhEff, coef, true,
                         TlC, TiC, aLC, aRC, TlN, TiN, aLN, aRN, dTr);
   }

   double iaij = 1.0 / I2(areaT, i, j, nx);
   for (int k = 1; k <= nz; ++k) I3(hTr, i, j, k, nx, ny) += dTr[k - 1] * iaij;
}

} // anonymous namespace

// ===========================================================================
// Host bridge -- same signature and same launch sequence as redi_cuda_launch_,
// but Phase B uses the hoisted kOptApplyFlux.
// ===========================================================================
extern "C" void redi_opt_launch(
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
    const double *areaT, double *tr_snap) {
   const int nx = *nx_, ny = *ny_, nz = *nz_, ns = *ns_;
   const int nghost = *nghost_, nxp = *nxp_, nyp = *nyp_;
   const double dt = *dt_;
   EosLin eos{*rho0_, *T_ref_, *S_ref_, *alpha_T_, *beta_S_};

   const int TPB = 128;
   auto grid1 = [&](long long n) { return (int)((n + TPB - 1) / TPB); };

   kOptCalcCoeffsX<<<grid1((long long)(nx - 1) * ny), TPB>>>(nx, ny, nz, ns, eos, h_layer, t_htr, s_htr,
                                                             wet_u, uPoL, uPoR, uKoL, uKoR, uhEff);
   kOptCalcCoeffsY<<<grid1((long long)nx * (ny - 1)), TPB>>>(nx, ny, nz, ns, eos, h_layer, t_htr, s_htr,
                                                             wet_v, vPoL, vPoR, vKoL, vKoR, vhEff);

   kOptFaceCopy<<<grid1((long long)(nx + 1) * ny), TPB>>>((nx + 1) * ny, khtr_u_ext, khtr_u);
   kOptFaceCopy<<<grid1((long long)nx * (ny + 1)), TPB>>>(nx * (ny + 1), khtr_v_ext, khtr_v);

   double *tr[2] = {t_htr, s_htr};
   for (int it = 0; it < 2; ++it) {
      const long long ncell = (long long)nx * ny * nz;
      kOptSnapshot<<<grid1(ncell), TPB>>>((int)ncell, tr[it], tr_snap);
      kOptApplyFlux<<<grid1((long long)nx * ny), TPB>>>(
          nx, ny, nz, ns, dt, nghost, nxp, nyp, *ww_, *we_, *wsz_, *wn_,
          khtr_u, khtr_v, dy_cu, dx_cv, idxCu, idyCv, areaT,
          h_layer, tr_snap, tr[it], uPoL, uPoR, uKoL, uKoR, uhEff,
          vPoL, vPoR, vKoL, vKoR, vhEff);
   }
   cudaDeviceSynchronize();
}
