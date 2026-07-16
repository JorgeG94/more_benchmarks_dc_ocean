// Optimized MEKE-step CUDA. A best-practice pass over meke_kernel.cu, freed from
// the "faithful transliteration" rule but held BIT-IDENTICAL to the faithful
// 16-kernel launcher (same per-cell arithmetic order). ab_main.cu checks
// max|diff| == 0 on the primary output field `meke`.
//
// RESULT (V100, 473x297x30): see OPTIMIZATION.md. Three ideas, in order of
// payoff, all "remove genuine waste" rather than cleverness:
//
//   1. LAUNCH FUSION 16 -> 3 kernels. The dependency graph of the step has
//      exactly three barriers:
//        (A) a per-cell pass  : mass/length/source/drag           -> meke_A
//        (B) a per-cell pass  : flux divergence + drag + kh + ku   -> meke
//      with the sn_u/sn_v zeroing (read by A when a_eady>0) as a pre-pass.
//      meke_kernel.cu's own "fused" path already collapses to 6; this goes to 3
//      by (a) merging the two sn-zero launches into one and (b) folding the two
//      flux launches + the divergence launch into pass B (see #2).
//
//   2. ELIMINATE the uflux/vflux global arrays (nu+nv doubles, written then
//      read). Pass B recomputes the 4 surrounding face fluxes IN REGISTERS from
//      the post-drag meke (meke_A) + mass_ws stencils. The flux is cheap (a
//      harmonic mean + a few mults), so the 2x redundant flux math costs far
//      less than the round-trip it removes -- unlike the continuity kernel,
//      where recompute of the expensive PPM lost.
//
//   3. 32-bit indexing. nx*ny*nz < 2^31 for every realistic MEKE grid
//      (479*303*30 ~ 4.35e6 << 2.1e9), so the size_t address math in
//      meke_args.h is pure overhead here.
//
// DOUBLE-BUFFER: pass A reads meke(old) and writes a scratch meke_A (post-drag);
// pass B reads the meke_A 5-point stencil and writes the final meke. One extra
// nt-sized scratch, but it removes the nu+nv flux arrays AND lets the flux see a
// race-free neighbourhood (meke_A is finished by the kernel boundary). Net
// device memory goes DOWN.
//
// CONFIG CAVEAT (the one non-general assumption): pass B recomputes the flux,
// which reads kh_diff, while pass B also writes kh_diff -- a read/write hazard in
// the general case. It is benign here because the nml sets khmeke_fac = 0, so
// the kh_diff term is `0.0 * (finite)` == 0.0 regardless of which copy is read;
// kh_u collapses to fmax(0,kh_bg) exactly. With khmeke_fac != 0 the flux would
// have to stay in its own kernel (OPTVER 2 below does exactly that, and is only
// ~a launch slower). Every other input pass B reads (meke_A, mass_ws, the
// per-cell factors) is produced by pass A, so it is hazard-free by construction.
#include <cstdio>
#include "gpu_rt.h"
#include "meke_args.h"

// --- 32-bit indexing: override meke_args.h's size_t macros (see #3 above) ---
#undef IT
#undef IU
#undef IV
#undef I3
#define IT(i, j)    ((int)((i)-1) + (nx)    * (int)((j)-1))
#define IU(i, j)    ((int)((i)-1) + (nx + 1)* (int)((j)-1))
#define IV(i, j)    ((int)((i)-1) + (nx)    * (int)((j)-1))
#define I3(i, j, k) ((int)((i)-1) + (nx) * ((int)((j)-1) + (ny) * (int)((k)-1)))

#ifndef TPB
#define TPB 128
#endif
#define NBLK(n) (int)(((n) + TPB - 1) / TPB)

#ifndef OPTVER
#define OPTVER 1     // 1 = recompute-fused (3 kernels, default); 2 = staged flux (4 kernels)
#endif

// meke_inv_lmix (`!$acc routine seq`) -- private copy, identical to meke_kernel.cu
__device__ __forceinline__ double o_inv_lmix(double ueddy, double sn, double beta,
                                             double area, double rd_over_dx, double depth,
                                             double cdrag, double a_deform, double a_rhines,
                                             double a_eady, double a_frict, double a_grid) {
    double lgrid = sqrt(fmax(area, 0.0));
    double ldeform = lgrid * rd_over_dx;
    double lfrict = (cdrag > 0.0) ? depth / cdrag : 0.0;
    double lrhines = (beta > 0.0) ? sqrt(fmax(ueddy, 0.0) / beta) : 0.0;
    double leady = (sn > 1.0e-15) ? ueddy / sn : 0.0;
    double inv_l = 0.0;
    if (a_deform * ldeform > 0.0) inv_l += 1.0 / (a_deform * ldeform);
    if (a_frict * lfrict > 0.0)   inv_l += 1.0 / (a_frict * lfrict);
    if (a_rhines * lrhines > 0.0) inv_l += 1.0 / (a_rhines * lrhines);
    if (a_eady * leady > 0.0)     inv_l += 1.0 / (a_eady * leady);
    if (a_grid * lgrid > 0.0)     inv_l += 1.0 / (a_grid * lgrid);
    return inv_l;
}

// ===================== pass A: loops 1,4,5,6,7,8 -> meke_A ===================
// Identical arithmetic to k_fused_A, but the post-drag energy is written to the
// scratch buffer mA instead of overwriting meke (so pass B sees a stable stencil).
__global__ void o_phaseA(MekeArgs a, double * __restrict__ mA) {
    int nx = a.nx, ny = a.ny, nz = a.nz;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx * ny) return;
    int i = t % nx + 1, j = t / nx + 1;
    double sdt = a.dt * a.dtscale;
    double damp_step = (a.kh_bg >= 0.0 || a.k4 >= 0.0) ? 0.5 : 1.0;
    double sdt_damp = sdt * damp_step;
    double cd2 = a.cdrag * a.cdrag;
    // loop 1
    a.rd_ws[IT(i, j)] = 0.0;
    double rd = a.rd_ws[IT(i, j)];
    // loop 6
    a.ke_diss_ws[IT(i, j)] = a.ke_diss_ext[IT(i, j)];
    // loop 4
    double mass = 0.0, dsum = 0.0;
    for (int k = 1; k <= nz; ++k) {
        double hk = fmax(a.h_layer[I3(i, j, k)], H_VANISHED);
        mass += a.rho_layer[I3(i, j, k)] * hk;
        dsum += a.h_layer[I3(i, j, k)];
    }
    a.depth_tot[IT(i, j)] = dsum;
    a.mass_ws[IT(i, j)] = mass;
    double imk = (mass > 0.0) ? 1.0 / mass : 0.0;
    a.i_mass[IT(i, j)] = imk;
    // loop 5
    double lgrid = sqrt(fmax(a.areaT[IT(i, j)], 0.0));
    double ldeform = lgrid * rd;
    double lfrict = (a.cdrag > 0.0) ? dsum / a.cdrag : 0.0;
    double bf2 = a.cd_scale * a.cd_scale;
    if (lfrict * a.cb > 0.0) { double r = ldeform / lfrict; bf2 += 1.0 / pow(1.0 + a.cb * r, 0.8); }
    bf2 = fmax(bf2, a.min_gamma2);
    a.bottom_fac2[IT(i, j)] = bf2;
    double tf2 = 1.0;
    if (lfrict * a.ct > 0.0) { double r = ldeform / lfrict; tf2 = 1.0 / pow(1.0 + a.ct * r, 0.25); }
    tf2 = fmax(tf2, a.min_gamma2);
    a.barotr_fac2[IT(i, j)] = tf2;
    double ueddy = sqrt(2.0 * fmax(0.0, tf2 * a.meke[IT(i, j)]));
    double beta = 0.0;
    if (i > 1 && i < nx) { double d = 0.5 * (a.f_centre[IT(i+1,j)] - a.f_centre[IT(i-1,j)]) * a.idxT[IT(i,j)]; beta += d * d; }
    if (j > 1 && j < ny) { double d = 0.5 * (a.f_centre[IT(i,j+1)] - a.f_centre[IT(i,j-1)]) * a.idyT[IT(i,j)]; beta += d * d; }
    beta = sqrt(beta);
    double sn = 0.0;
    if (a.a_eady > 0.0) sn = 0.25 * ((a.sn_u_ws[IU(i,j)] + a.sn_u_ws[IU(i+1,j)]) +
                                     (a.sn_v_ws[IV(i,j)] + a.sn_v_ws[IV(i,j+1)]));
    double il = o_inv_lmix(ueddy, sn, beta, a.areaT[IT(i,j)], rd, dsum,
                           a.cdrag, a.a_deform, a.a_rhines, a.a_eady, a.a_frict, a.a_grid);
    a.le[IT(i, j)] = (il > 0.0) ? 1.0 / il : 0.0;
    // loop 7
    double s = a.bgsrc;
    if (a.gmcoeff >= 0.0) s += a.gmcoeff * imk * a.gm_src[IT(i,j)];
    if (a.frcoeff >= 0.0) s -= a.frcoeff * imk * a.ke_diss_ws[IT(i,j)];
    a.src[IT(i, j)] = s;
    double e = a.meke[IT(i, j)] + sdt * s;
    // loop 8
    double drag_rate = imk * sqrt(cd2 * (fmax(0.0, 2.0 * bf2 * e) + a.u_bbl2[IT(i,j)] + a.uscale * a.uscale));
    double damp_rate = a.damping + drag_rate * bf2;
    if (e < 0.0) damp_rate = 0.0;
    mA[IT(i, j)] = e / (1.0 + sdt_damp * damp_rate);
}

// ---- per-face flux, recomputed from post-drag meke (mA) + mass_ws ----------
// Bit-identical to k_uflux/k_vflux (interior) with the wall value folded in as
// the 0.0 return -- reproduces the k_zero(uflux) + k_uflux split exactly.
__device__ __forceinline__ double o_flux_u(const MekeArgs &a, const double * __restrict__ mA, int i, int j) {
    int nx = a.nx;
    if (i < 2 || i > nx) return 0.0;                        // wall face -> 0 (was k_zero)
    double sdt = a.dt * a.dtscale;
    double geo = a.dy_cu[IU(i,j)] * a.idxCu[IU(i,j)];
    // kh_diff term: khmeke_fac == 0 here, so this is 0.0 regardless of the (racy)
    // kh_diff read -- see CONFIG CAVEAT at the top of the file.
    double kh_u = fmax(0.0, a.kh_bg) + a.khmeke_fac * 0.5 * (a.kh_diff[IT(i-1,j)] + a.kh_diff[IT(i,j)]);
    double inv_max = 2.0 * sdt * (geo * fmax(a.iareaT[IT(i-1,j)], a.iareaT[IT(i,j)]));
    if (kh_u * inv_max > 0.25) kh_u = 0.25 / inv_max;
    double hm = 2.0 * a.mass_ws[IT(i-1,j)] * a.mass_ws[IT(i,j)] /
                ((a.mass_ws[IT(i-1,j)] + a.mass_ws[IT(i,j)]) + MASS_NEGLECT);
    return (kh_u * geo) * hm * (mA[IT(i-1,j)] - mA[IT(i,j)]);
}
__device__ __forceinline__ double o_flux_v(const MekeArgs &a, const double * __restrict__ mA, int i, int j) {
    int nx = a.nx, ny = a.ny;
    if (j < 2 || j > ny) return 0.0;
    double sdt = a.dt * a.dtscale;
    double geo = a.dx_cv[IV(i,j)] * a.idyCv[IV(i,j)];
    double kh_v = fmax(0.0, a.kh_bg) + a.khmeke_fac * 0.5 * (a.kh_diff[IT(i,j-1)] + a.kh_diff[IT(i,j)]);
    double inv_max = 2.0 * sdt * (geo * fmax(a.iareaT[IT(i,j-1)], a.iareaT[IT(i,j)]));
    if (kh_v * inv_max > 0.25) kh_v = 0.25 / inv_max;
    double hm = 2.0 * a.mass_ws[IT(i,j-1)] * a.mass_ws[IT(i,j)] /
                ((a.mass_ws[IT(i,j-1)] + a.mass_ws[IT(i,j)]) + MASS_NEGLECT);
    return (kh_v * geo) * hm * (mA[IT(i,j-1)] - mA[IT(i,j)]);
}

// ===================== pass B: loops 13,14,15,16 -> meke =====================
// Recomputes the 4 surrounding fluxes in registers (no uflux/vflux global),
// then divergence + drag + kh + ku. Reads meke_A, writes the final meke.
__global__ void o_phaseB(MekeArgs a, const double * __restrict__ mA) {
    int nx = a.nx, ny = a.ny;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx * ny) return;
    int i = t % nx + 1, j = t / nx + 1;
    double sdt = a.dt * a.dtscale;
    double damp_step = (a.kh_bg >= 0.0 || a.k4 >= 0.0) ? 0.5 : 1.0;
    double sdt_damp = sdt * damp_step;
    double cd2 = a.cdrag * a.cdrag;
    // loop 13: conservative divergence, fluxes recomputed
    double ufx_i  = o_flux_u(a, mA, i,   j);
    double ufx_ip = o_flux_u(a, mA, i+1, j);
    double vfy_j  = o_flux_v(a, mA, i, j);
    double vfy_jp = o_flux_v(a, mA, i, j+1);
    double mke = sdt * (a.iareaT[IT(i,j)] * a.i_mass[IT(i,j)]) *
                 ((ufx_i - ufx_ip) + (vfy_j - vfy_jp));
    double e = mA[IT(i, j)] + mke;
    // loop 14: drag
    double bf2 = a.bottom_fac2[IT(i, j)];
    double drag_rate = a.i_mass[IT(i,j)] * sqrt(cd2 * (fmax(0.0, 2.0 * bf2 * e)
                                                       + a.u_bbl2[IT(i,j)] + a.uscale * a.uscale));
    double damp_rate = a.damping + drag_rate * bf2;
    if (e < 0.0) damp_rate = 0.0;
    e = e / (1.0 + sdt_damp * damp_rate);
    a.meke[IT(i, j)] = e;
    // loop 15: kh closure
    if (a.khcoeff > 0.0) {
        double ueddy = sqrt(2.0 * fmax(0.0, a.barotr_fac2[IT(i,j)] * e));
        a.kh_diff[IT(i, j)] = a.khcoeff * ueddy * a.le[IT(i,j)];
    } else a.kh_diff[IT(i, j)] = 0.0;
    // loop 16: ku closure
    if (a.backscatter && a.visc_coeff_ku != 0.0) {
        double ueddy = sqrt(2.0 * fmax(0.0, e));
        a.ku[IT(i, j)] = a.visc_coeff_ku * ueddy * a.le[IT(i,j)];
    } else a.ku[IT(i, j)] = 0.0;
}

// ---- OPTVER 2: staged flux (no recompute). Keeps uflux/vflux, but still 32-bit
//      indexed and merged where safe. General wrt khmeke_fac. One launch slower.
__global__ void o_phaseU(MekeArgs a, const double * __restrict__ mA) {
    int nx = a.nx, ny = a.ny;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (nx + 1) * ny) return;
    int i = t % (nx + 1) + 1, j = t / (nx + 1) + 1;
    a.uflux[IU(i, j)] = o_flux_u(a, mA, i, j);
}
__global__ void o_phaseV(MekeArgs a, const double * __restrict__ mA) {
    int nx = a.nx, ny = a.ny;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx * (ny + 1)) return;
    int i = t % nx + 1, j = t / nx + 1;
    a.vflux[IV(i, j)] = o_flux_v(a, mA, i, j);
}
__global__ void o_phaseB_staged(MekeArgs a, const double * __restrict__ mA) {
    int nx = a.nx, ny = a.ny;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx * ny) return;
    int i = t % nx + 1, j = t / nx + 1;
    double sdt = a.dt * a.dtscale;
    double damp_step = (a.kh_bg >= 0.0 || a.k4 >= 0.0) ? 0.5 : 1.0;
    double sdt_damp = sdt * damp_step;
    double cd2 = a.cdrag * a.cdrag;
    double mke = sdt * (a.iareaT[IT(i,j)] * a.i_mass[IT(i,j)]) *
                 ((a.uflux[IU(i,j)] - a.uflux[IU(i+1,j)]) + (a.vflux[IV(i,j)] - a.vflux[IV(i,j+1)]));
    double e = mA[IT(i, j)] + mke;
    double bf2 = a.bottom_fac2[IT(i, j)];
    double drag_rate = a.i_mass[IT(i,j)] * sqrt(cd2 * (fmax(0.0, 2.0 * bf2 * e)
                                                       + a.u_bbl2[IT(i,j)] + a.uscale * a.uscale));
    double damp_rate = a.damping + drag_rate * bf2;
    if (e < 0.0) damp_rate = 0.0;
    e = e / (1.0 + sdt_damp * damp_rate);
    a.meke[IT(i, j)] = e;
    if (a.khcoeff > 0.0) {
        double ueddy = sqrt(2.0 * fmax(0.0, a.barotr_fac2[IT(i,j)] * e));
        a.kh_diff[IT(i, j)] = a.khcoeff * ueddy * a.le[IT(i,j)];
    } else a.kh_diff[IT(i, j)] = 0.0;
    if (a.backscatter && a.visc_coeff_ku != 0.0) {
        double ueddy = sqrt(2.0 * fmax(0.0, e));
        a.ku[IT(i, j)] = a.visc_coeff_ku * ueddy * a.le[IT(i,j)];
    } else a.ku[IT(i, j)] = 0.0;
}

// ---- merged sn_u/sn_v zeroing (loops 2,3): one launch over max(nu,nv) --------
__global__ void o_zero_sn(double * __restrict__ su, double * __restrict__ sv, int nu, int nv) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < nu) su[t] = 0.0;
    if (t < nv) sv[t] = 0.0;
}

// ===================== launcher =============================================
// Synchronous per call, exactly like meke_cuda_launch, so the A/B timing is
// apples-to-apples (both pay one device sync per step).
extern "C" void meke_opt_launch(const MekeArgs *ap, double *meke_scratch) {
    const MekeArgs &a = *ap;
    int nx = a.nx, ny = a.ny;
    int nt = nx * ny, nu = (nx + 1) * ny, nv = nx * (ny + 1);
    int nmax = nu > nv ? nu : nv;
    o_zero_sn<<<NBLK(nmax), TPB>>>(a.sn_u_ws, a.sn_v_ws, nu, nv);   // loops 2,3
    o_phaseA<<<NBLK(nt), TPB>>>(a, meke_scratch);                   // loops 1,4,5,6,7,8
#if OPTVER == 2
    o_phaseU<<<NBLK(nu), TPB>>>(a, meke_scratch);                   // loops 9,10
    o_phaseV<<<NBLK(nv), TPB>>>(a, meke_scratch);                   // loops 11,12
    o_phaseB_staged<<<NBLK(nt), TPB>>>(a, meke_scratch);            // loops 13,14,15,16
#else
    o_phaseB<<<NBLK(nt), TPB>>>(a, meke_scratch);                   // loops 9-16, fluxes in regs
#endif
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) fprintf(stderr, "CUDA error (opt): %s\n", cudaGetErrorString(e));
}
