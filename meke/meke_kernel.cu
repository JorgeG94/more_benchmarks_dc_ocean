// CUDA C port of the live MEKE step (meke.F90 / meke_fused.F90).
//
// THREE MODES, answering different questions:
//   MODE_FAITHFUL (0) : 16 kernels/step, one per `do concurrent` loop, same
//                       order. Measures nvfortran's CODEGEN against nvcc at
//                       equal launch count.
//   MODE_FUSED    (1) :  6 kernels/step, the SAME fusion meke_fused.F90
//                       does (see that file's header for the dependency
//                       analysis). Measures what fewer launches buy in C.
//   MODE_GRAPH    (2) : the 6 fused kernels captured into a cudaGraph and
//                       replayed. Measures the residual launch cost --
//                       something `do concurrent` cannot express at all.
//
// This is a FAITHFUL port, not a tuned one: no tiling, no __ldg, no
// launch_bounds, no shared memory. The question is "does nvfortran keep up
// with straightforward CUDA?", not "what is the fastest possible MEKE?".
//
// INDEXING / ABI: `struct MekeArgs` and the IT/IU/IV/I3 stride macros now live in
// meke_args.h, so meke_bench.F90 (via bind(C)) and meke_native.cu share ONE
// definition instead of each keeping a copy that could drift. See that header.
//
// FP CONTRACT: nvcc contracts a*b+c into FMA by default and so does nvfortran
// at -O3 -fast; the two do not always pick the same groupings, so DC-vs-CUDA
// agreement is checked at ~1e-15 relative, not at 0. DC-vs-DC-fused IS checked
// at exactly 0 (same compiler, same expressions).
#include <cstdio>
#include "gpu_rt.h"
#include "meke_args.h"

#define TPB 128
static inline int nblk(size_t n) { return (int)((n + TPB - 1) / TPB); }

// ===================== FAITHFUL: one kernel per DC loop =====================

// -- loop 1 / 2 / 3: meke_zero_2d on rd_ws (nx,ny), sn_u (nx+1,ny), sn_v (nx,ny+1)
__global__ void k_zero(double * __restrict__ a, size_t n) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) a[t] = 0.0;
}

// -- loop 4: meke_mass
__global__ void k_mass(MekeArgs a) {
    int nx = a.nx, ny = a.ny, nz = a.nz;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nx * ny) return;
    int i = (int)(t % nx) + 1, j = (int)(t / nx) + 1;
    double mass = 0.0, dsum = 0.0;
    for (int k = 1; k <= nz; ++k) {
        double hk = fmax(a.h_layer[I3(i, j, k)], H_VANISHED);
        double rhok = a.rho_layer[I3(i, j, k)];   // have_rho = .true.
        mass += rhok * hk;
        dsum += a.h_layer[I3(i, j, k)];
    }
    a.depth_tot[IT(i, j)] = dsum;
    a.mass_ws[IT(i, j)] = mass;
    a.i_mass[IT(i, j)] = (mass > 0.0) ? 1.0 / mass : 0.0;
}

// -- meke_inv_lmix, `!$acc routine seq` in Fortran -> __device__ here
__device__ __forceinline__ double inv_lmix(double ueddy, double sn, double beta,
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

// -- loop 5: meke_length_scales
__global__ void k_length(MekeArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nx * ny) return;
    int i = (int)(t % nx) + 1, j = (int)(t / nx) + 1;
    double rd = a.rd_ws[IT(i, j)];
    double lgrid = sqrt(fmax(a.areaT[IT(i, j)], 0.0));
    double ldeform = lgrid * rd;
    double lfrict = (a.cdrag > 0.0) ? a.depth_tot[IT(i, j)] / a.cdrag : 0.0;
    double bf2 = a.cd_scale * a.cd_scale;
    // Fortran writes `1.0_wp/(1.0_wp + cb*ratio)**0.8_wp` -- reciprocal of a
    // pow, NOT pow(x,-0.8). Those are not the same in the last bits.
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
    double il = inv_lmix(ueddy, sn, beta, a.areaT[IT(i,j)], rd, a.depth_tot[IT(i,j)],
                         a.cdrag, a.a_deform, a.a_rhines, a.a_eady, a.a_frict, a.a_grid);
    a.le[IT(i, j)] = (il > 0.0) ? 1.0 / il : 0.0;
}

// -- loop 6: meke_stage_rd(ke_diss_ext -> ke_diss_ws)
__global__ void k_stage_kediss(MekeArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nx * ny) return;
    a.ke_diss_ws[t] = a.ke_diss_ext[t];
}

// -- loop 7: meke_source
__global__ void k_source(MekeArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nx * ny) return;
    int i = (int)(t % nx) + 1, j = (int)(t / nx) + 1;
    double sdt = a.dt * a.dtscale;
    double s = a.bgsrc;
    if (a.gmcoeff >= 0.0) s += a.gmcoeff * a.i_mass[IT(i,j)] * a.gm_src[IT(i,j)];
    if (a.frcoeff >= 0.0) s -= a.frcoeff * a.i_mass[IT(i,j)] * a.ke_diss_ws[IT(i,j)];
    a.src[IT(i, j)] = s;
    a.meke[IT(i, j)] += sdt * s;
}

// -- loops 8 and 14: meke_drag
__global__ void k_drag(MekeArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nx * ny) return;
    int i = (int)(t % nx) + 1, j = (int)(t / nx) + 1;
    double sdt = a.dt * a.dtscale;
    double damp_step = (a.kh_bg >= 0.0 || a.k4 >= 0.0) ? 0.5 : 1.0;
    double sdt_damp = sdt * damp_step;
    double cd2 = a.cdrag * a.cdrag;
    double bf2 = a.bottom_fac2[IT(i, j)];
    double e = a.meke[IT(i, j)];
    double drag_rate = a.i_mass[IT(i,j)] * sqrt(cd2 * (fmax(0.0, 2.0 * bf2 * e)
                                                       + a.u_bbl2[IT(i,j)] + a.uscale * a.uscale));
    double damp_rate = a.damping + drag_rate * bf2;
    if (e < 0.0) damp_rate = 0.0;
    a.meke[IT(i, j)] = e / (1.0 + sdt_damp * damp_rate);
}

// -- loop 9: zero uflux ; loop 11: zero vflux  -> k_zero

// -- loop 10: uflux interior (harmonic-mass Laplacian flux)
__global__ void k_uflux(MekeArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)(nx - 1) * ny) return;          // i = 2..nx
    int i = (int)(t % (nx - 1)) + 2, j = (int)(t / (nx - 1)) + 1;
    double sdt = a.dt * a.dtscale;
    double geo = a.dy_cu[IU(i,j)] * a.idxCu[IU(i,j)];
    double kh_u = fmax(0.0, a.kh_bg) + a.khmeke_fac * 0.5 * (a.kh_diff[IT(i-1,j)] + a.kh_diff[IT(i,j)]);
    double inv_max = 2.0 * sdt * (geo * fmax(a.iareaT[IT(i-1,j)], a.iareaT[IT(i,j)]));
    if (kh_u * inv_max > 0.25) kh_u = 0.25 / inv_max;
    double hm = 2.0 * a.mass_ws[IT(i-1,j)] * a.mass_ws[IT(i,j)] /
                ((a.mass_ws[IT(i-1,j)] + a.mass_ws[IT(i,j)]) + MASS_NEGLECT);
    a.uflux[IU(i, j)] = (kh_u * geo) * hm * (a.meke[IT(i-1,j)] - a.meke[IT(i,j)]);
}

// -- loop 12: vflux interior
__global__ void k_vflux(MekeArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nx * (ny - 1)) return;          // j = 2..ny
    int i = (int)(t % nx) + 1, j = (int)(t / nx) + 2;
    double sdt = a.dt * a.dtscale;
    double geo = a.dx_cv[IV(i,j)] * a.idyCv[IV(i,j)];
    double kh_v = fmax(0.0, a.kh_bg) + a.khmeke_fac * 0.5 * (a.kh_diff[IT(i,j-1)] + a.kh_diff[IT(i,j)]);
    double inv_max = 2.0 * sdt * (geo * fmax(a.iareaT[IT(i,j-1)], a.iareaT[IT(i,j)]));
    if (kh_v * inv_max > 0.25) kh_v = 0.25 / inv_max;
    double hm = 2.0 * a.mass_ws[IT(i,j-1)] * a.mass_ws[IT(i,j)] /
                ((a.mass_ws[IT(i,j-1)] + a.mass_ws[IT(i,j)]) + MASS_NEGLECT);
    a.vflux[IV(i, j)] = (kh_v * geo) * hm * (a.meke[IT(i,j-1)] - a.meke[IT(i,j)]);
}

// -- loop 13: conservative divergence
__global__ void k_div(MekeArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nx * ny) return;
    int i = (int)(t % nx) + 1, j = (int)(t / nx) + 1;
    double sdt = a.dt * a.dtscale;
    double mke = sdt * (a.iareaT[IT(i,j)] * a.i_mass[IT(i,j)]) *
                 ((a.uflux[IU(i,j)] - a.uflux[IU(i+1,j)]) + (a.vflux[IV(i,j)] - a.vflux[IV(i,j+1)]));
    a.meke[IT(i, j)] += mke;
}

// -- loop 15: meke_kh_closure
__global__ void k_kh(MekeArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nx * ny) return;
    int i = (int)(t % nx) + 1, j = (int)(t / nx) + 1;
    if (a.khcoeff > 0.0) {
        double ueddy = sqrt(2.0 * fmax(0.0, a.barotr_fac2[IT(i,j)] * a.meke[IT(i,j)]));
        a.kh_diff[IT(i, j)] = a.khcoeff * ueddy * a.le[IT(i,j)];
    } else a.kh_diff[IT(i, j)] = 0.0;
}

// -- loop 16: meke_ku_closure
__global__ void k_ku(MekeArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nx * ny) return;
    int i = (int)(t % nx) + 1, j = (int)(t / nx) + 1;
    if (a.backscatter && a.visc_coeff_ku != 0.0) {
        double ueddy = sqrt(2.0 * fmax(0.0, a.meke[IT(i,j)]));
        a.ku[IT(i, j)] = a.visc_coeff_ku * ueddy * a.le[IT(i,j)];
    } else a.ku[IT(i, j)] = 0.0;
}

// ===================== FUSED: 6 kernels ====================================
// Mirrors meke_fused.F90 exactly -- same grouping, same barriers.

// -- L3 = loops 1,4,5,6,7,8
__global__ void k_fused_A(MekeArgs a) {
    int nx = a.nx, ny = a.ny, nz = a.nz;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nx * ny) return;
    int i = (int)(t % nx) + 1, j = (int)(t / nx) + 1;
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
    // Fortran writes `1.0_wp/(1.0_wp + cb*ratio)**0.8_wp` -- reciprocal of a
    // pow, NOT pow(x,-0.8). Those are not the same in the last bits.
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
    double il = inv_lmix(ueddy, sn, beta, a.areaT[IT(i,j)], rd, dsum,
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
    a.meke[IT(i, j)] = e / (1.0 + sdt_damp * damp_rate);
}

// -- L4 = loops 9,10 (wall + producer merge)
__global__ void k_fused_U(MekeArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)(nx + 1) * ny) return;
    int i = (int)(t % (nx + 1)) + 1, j = (int)(t / (nx + 1)) + 1;
    if (i >= 2 && i <= nx) {
        double sdt = a.dt * a.dtscale;
        double geo = a.dy_cu[IU(i,j)] * a.idxCu[IU(i,j)];
        double kh_u = fmax(0.0, a.kh_bg) + a.khmeke_fac * 0.5 * (a.kh_diff[IT(i-1,j)] + a.kh_diff[IT(i,j)]);
        double inv_max = 2.0 * sdt * (geo * fmax(a.iareaT[IT(i-1,j)], a.iareaT[IT(i,j)]));
        if (kh_u * inv_max > 0.25) kh_u = 0.25 / inv_max;
        double hm = 2.0 * a.mass_ws[IT(i-1,j)] * a.mass_ws[IT(i,j)] /
                    ((a.mass_ws[IT(i-1,j)] + a.mass_ws[IT(i,j)]) + MASS_NEGLECT);
        a.uflux[IU(i, j)] = (kh_u * geo) * hm * (a.meke[IT(i-1,j)] - a.meke[IT(i,j)]);
    } else a.uflux[IU(i, j)] = 0.0;
}

// -- L5 = loops 11,12
__global__ void k_fused_V(MekeArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nx * (ny + 1)) return;
    int i = (int)(t % nx) + 1, j = (int)(t / nx) + 1;
    if (j >= 2 && j <= ny) {
        double sdt = a.dt * a.dtscale;
        double geo = a.dx_cv[IV(i,j)] * a.idyCv[IV(i,j)];
        double kh_v = fmax(0.0, a.kh_bg) + a.khmeke_fac * 0.5 * (a.kh_diff[IT(i,j-1)] + a.kh_diff[IT(i,j)]);
        double inv_max = 2.0 * sdt * (geo * fmax(a.iareaT[IT(i,j-1)], a.iareaT[IT(i,j)]));
        if (kh_v * inv_max > 0.25) kh_v = 0.25 / inv_max;
        double hm = 2.0 * a.mass_ws[IT(i,j-1)] * a.mass_ws[IT(i,j)] /
                    ((a.mass_ws[IT(i,j-1)] + a.mass_ws[IT(i,j)]) + MASS_NEGLECT);
        a.vflux[IV(i, j)] = (kh_v * geo) * hm * (a.meke[IT(i,j-1)] - a.meke[IT(i,j)]);
    } else a.vflux[IV(i, j)] = 0.0;
}

// -- L6 = loops 13,14,15,16
__global__ void k_fused_B(MekeArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nx * ny) return;
    int i = (int)(t % nx) + 1, j = (int)(t / nx) + 1;
    double sdt = a.dt * a.dtscale;
    double damp_step = (a.kh_bg >= 0.0 || a.k4 >= 0.0) ? 0.5 : 1.0;
    double sdt_damp = sdt * damp_step;
    double cd2 = a.cdrag * a.cdrag;
    double mke = sdt * (a.iareaT[IT(i,j)] * a.i_mass[IT(i,j)]) *
                 ((a.uflux[IU(i,j)] - a.uflux[IU(i+1,j)]) + (a.vflux[IV(i,j)] - a.vflux[IV(i,j+1)]));
    double e = a.meke[IT(i, j)] + mke;
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

// ===================== launchers ===========================================

static void launch_faithful(const MekeArgs &a, cudaStream_t s) {
    int nx = a.nx, ny = a.ny;
    size_t nt = (size_t)nx * ny, nu = (size_t)(nx + 1) * ny, nv = (size_t)nx * (ny + 1);
    k_zero<<<nblk(nt), TPB, 0, s>>>(a.rd_ws, nt);              // 1
    k_zero<<<nblk(nu), TPB, 0, s>>>(a.sn_u_ws, nu);            // 2
    k_zero<<<nblk(nv), TPB, 0, s>>>(a.sn_v_ws, nv);            // 3
    k_mass<<<nblk(nt), TPB, 0, s>>>(a);                        // 4
    k_length<<<nblk(nt), TPB, 0, s>>>(a);                      // 5
    k_stage_kediss<<<nblk(nt), TPB, 0, s>>>(a);                // 6
    k_source<<<nblk(nt), TPB, 0, s>>>(a);                      // 7
    k_drag<<<nblk(nt), TPB, 0, s>>>(a);                        // 8
    k_zero<<<nblk(nu), TPB, 0, s>>>(a.uflux, nu);              // 9
    k_uflux<<<nblk((size_t)(nx-1)*ny), TPB, 0, s>>>(a);        // 10
    k_zero<<<nblk(nv), TPB, 0, s>>>(a.vflux, nv);              // 11
    k_vflux<<<nblk((size_t)nx*(ny-1)), TPB, 0, s>>>(a);        // 12
    k_div<<<nblk(nt), TPB, 0, s>>>(a);                         // 13
    k_drag<<<nblk(nt), TPB, 0, s>>>(a);                        // 14
    k_kh<<<nblk(nt), TPB, 0, s>>>(a);                          // 15
    k_ku<<<nblk(nt), TPB, 0, s>>>(a);                          // 16
}

static void launch_fused(const MekeArgs &a, cudaStream_t s) {
    int nx = a.nx, ny = a.ny;
    size_t nt = (size_t)nx * ny, nu = (size_t)(nx + 1) * ny, nv = (size_t)nx * (ny + 1);
    k_zero<<<nblk(nu), TPB, 0, s>>>(a.sn_u_ws, nu);            // L1
    k_zero<<<nblk(nv), TPB, 0, s>>>(a.sn_v_ws, nv);            // L2
    k_fused_A<<<nblk(nt), TPB, 0, s>>>(a);                     // L3
    k_fused_U<<<nblk(nu), TPB, 0, s>>>(a);                     // L4
    k_fused_V<<<nblk(nv), TPB, 0, s>>>(a);                     // L5
    k_fused_B<<<nblk(nt), TPB, 0, s>>>(a);                     // L6
}

static cudaGraphExec_t g_exec = nullptr;
static cudaStream_t    g_stream = nullptr;

extern "C" void meke_cuda_launch(const MekeArgs *ap, int mode) {
    const MekeArgs &a = *ap;
    if (mode == 0) {
        launch_faithful(a, 0);
        cudaDeviceSynchronize();
    } else if (mode == 1) {
        launch_fused(a, 0);
        cudaDeviceSynchronize();
    } else {
        // MODE_GRAPH: capture once, replay thereafter. The capture is done on a
        // non-default stream (the legacy default stream cannot be captured).
        if (!g_exec) {
            cudaGraph_t g;
            cudaStreamCreate(&g_stream);
            cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeGlobal);
            launch_fused(a, g_stream);
            cudaStreamEndCapture(g_stream, &g);
            cudaGraphInstantiate(&g_exec, g, nullptr, nullptr, 0);
            cudaGraphDestroy(g);
        }
        cudaGraphLaunch(g_exec, g_stream);
        cudaStreamSynchronize(g_stream);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) fprintf(stderr, "CUDA error (mode %d): %s\n", mode, cudaGetErrorString(e));
}

extern "C" void meke_cuda_graph_reset(void) {
    if (g_exec) { cudaGraphExecDestroy(g_exec); g_exec = nullptr; }
    if (g_stream) { cudaStreamDestroy(g_stream); g_stream = nullptr; }
}
