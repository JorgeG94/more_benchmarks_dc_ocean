// Faithful CUDA C transliteration of continuity_compute_fluxes_barotropic.
//
// PROVENANCE: hand-ported from continuity.F90 sitting next to this file,
// itself a verbatim extract of <model>-sea-ice @ feat/sea-ice,
// src/core/ocean/kernels/structured/continuity_ppm/continuity.F90:428-609
// (+ the four PPM helpers). If the Fortran changes, this must change too — a
// silent divergence would make the benchmark compare two different algorithms
// and report the difference as a compiler result.
//
// FAITHFULNESS RULES (the point is to measure the COMPILER, not my cleverness):
//   * Same operation order, same parenthesisation, same temporaries. Fortran
//     pins evaluation order inside parens; reassociating moves the last bits
//     and makes "do they agree?" unanswerable.
//   * Same branch structure. Same helper decomposition (the helpers stay
//     separate __device__ functions, exactly as the Fortran calls them).
//   * One thread per cell, matching each `do concurrent(j,i)` iteration space.
//   * Nine kernels, one per Fortran loop, launched in the same order. The
//     4 one-dimensional boundary loops are ~0.4% of runtime but they are part
//     of the subroutine, so they are part of the port.
//   * No shared-memory tiling, no __ldg, no launch_bounds, no fast-math.
//
// INDEXING: the device arrays ARE the Fortran allocations (via host_data
// use_device), so this must linearise identically. Fortran is column-major,
// 1-based. Face arrays carry one extra element in their own direction, so
// they need their own stride — mixing them up is silent corruption, not a
// compile error:
//     h, flux_h, iareaT, wet_T          (nx,   ny)     -> IDX_T
//     u_face_x, mass_flux_x, dy_cu      (nx+1, ny)     -> IDX_U
//     v_face_y, mass_flux_y, dx_cv      (nx,   ny+1)   -> IDX_V
//     h_face_{left,right}_x%data        (nx+1, ny, 1)  -> IDX_U  (3rd dim = 1)
//     h_face_{left,right}_y%data        (nx,   ny+1,1) -> IDX_V

#include <cstdio>
#include <cuda_runtime.h>

// Fortran 1-based (i,j) -> 0-based linear, column-major, per stride class.
#define IDX_T(i, j) ((size_t)((i) - 1) + (size_t)(nx) * (size_t)((j) - 1))
#define IDX_U(i, j) ((size_t)((i) - 1) + (size_t)(nx + 1) * (size_t)((j) - 1))
#define IDX_V(i, j) ((size_t)((i) - 1) + (size_t)(nx) * (size_t)((j) - 1))

// --- ppm_mirror_h (continuity.F90:259) — elemental, branchless ---
__device__ __forceinline__ double ppm_mirror_h(double h_nbr, double h_loc, double w_nbr) {
    return w_nbr * h_nbr + (1.0 - w_nbr) * h_loc;
}

// --- ppm_limited_slope (:280) ---
__device__ __forceinline__ void ppm_limited_slope(double h_im1, double h_i, double h_ip1,
                                                  double *dh) {
    double dh_centered, dh_left, dh_right;
    dh_left = h_i - h_im1;
    dh_right = h_ip1 - h_i;
    dh_centered = 0.5 * (dh_left + dh_right);
    if (dh_left * dh_right > 0.0) {
        // Fortran: sign(min(|dc|, 2|dl|, 2|dr|), dc) — magnitude of the first
        // argument, sign of the second. copysign is the exact analogue.
        *dh = copysign(fmin(fabs(dh_centered), fmin(2.0 * fabs(dh_left), 2.0 * fabs(dh_right))),
                       dh_centered);
    } else {
        *dh = 0.0;
    }
}

// --- ppm_cell_limiter (:307) — Colella-Woodward 1984 eq 1.10 ---
__device__ __forceinline__ void ppm_cell_limiter(double h_centre, double *h_left,
                                                 double *h_right) {
    double dh_lr, h_six;
    dh_lr = *h_right - *h_left;
    h_six = 6.0 * (h_centre - 0.5 * (*h_left + *h_right));
    if ((*h_right - h_centre) * (h_centre - *h_left) <= 0.0) {
        *h_left = h_centre;
        *h_right = h_centre;
    } else if (dh_lr * h_six > dh_lr * dh_lr) {
        *h_left = 3.0 * h_centre - 2.0 * (*h_right);
    } else if (dh_lr * h_six < -dh_lr * dh_lr) {
        *h_right = 3.0 * h_centre - 2.0 * (*h_left);
    }
}

// --- ppm_limit_pos (:341) — off by default (use_ppm_limit_pos = .false.) ---
__device__ __forceinline__ void ppm_limit_pos(double h_centre, double *h_left,
                                              double *h_right, double h_min) {
    double curv, dh, loc_scale;
    curv = 3.0 * ((*h_left + *h_right) - 2.0 * h_centre);
    if (curv > 0.0) {
        dh = *h_right - *h_left;
        if (fabs(dh) < curv) {
            if (h_centre <= h_min) {
                *h_left = h_centre;
                *h_right = h_centre;
            } else if (12.0 * curv * (h_centre - h_min) < (curv * curv + 3.0 * dh * dh)) {
                loc_scale = 12.0 * curv * (h_centre - h_min) / (curv * curv + 3.0 * dh * dh);
                *h_left = h_centre + loc_scale * (*h_left - h_centre);
                *h_right = h_centre + loc_scale * (*h_right - h_centre);
            }
        }
    }
}

// ===================== 1. X-direction PPM (F90:136) =====================
__global__ void k_ppm_x(const double * __restrict__ h, const double * __restrict__ wet_T, double * __restrict__ hfl_x, double * __restrict__ hfr_x,
                        int nx, int ny, int do_pos, double h_min_pos) {
    // do concurrent(j=1:ny, i=3:nx-2)
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t ni = (size_t)(nx - 4);          // i = 3 .. nx-2
    if (t >= ni * (size_t)ny) return;
    int i = (int)(t % ni) + 3;
    int j = (int)(t / ni) + 1;

    double dh_m1, dh_0, dh_p1, h_left, h_right, hm2, hm1, h0, hp1, hp2;
    h0 = h[IDX_T(i, j)];
    hm1 = ppm_mirror_h(h[IDX_T(i - 1, j)], h0, wet_T[IDX_T(i - 1, j)]);
    hp1 = ppm_mirror_h(h[IDX_T(i + 1, j)], h0, wet_T[IDX_T(i + 1, j)]);
    hm2 = ppm_mirror_h(h[IDX_T(i - 2, j)], hm1, wet_T[IDX_T(i - 2, j)]);
    hp2 = ppm_mirror_h(h[IDX_T(i + 2, j)], hp1, wet_T[IDX_T(i + 2, j)]);
    ppm_limited_slope(hm2, hm1, h0, &dh_m1);
    ppm_limited_slope(hm1, h0, hp1, &dh_0);
    ppm_limited_slope(h0, hp1, hp2, &dh_p1);
    dh_0 = dh_0 * wet_T[IDX_T(i - 1, j)] * wet_T[IDX_T(i, j)] * wet_T[IDX_T(i + 1, j)];
    h_left = 0.5 * (hm1 + h0) - (dh_0 - dh_m1) / 6.0;
    h_right = 0.5 * (h0 + hp1) - (dh_p1 - dh_0) / 6.0;
    ppm_cell_limiter(h0, &h_left, &h_right);
    if (do_pos) ppm_limit_pos(h0, &h_left, &h_right, h_min_pos);
    hfr_x[IDX_U(i, j)] = h_left;
    hfl_x[IDX_U(i + 1, j)] = h_right;
}

// ===================== 2. X boundary cells (F90:164) ====================
__global__ void k_bnd_x(const double * __restrict__ h, double * __restrict__ hfl_x, double * __restrict__ hfr_x, int nx, int ny) {
    // do concurrent(j=1:ny)
    int j = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x) + 1;
    if (j > ny) return;
    hfl_x[IDX_U(1, j)] = h[IDX_T(1, j)];
    hfr_x[IDX_U(1, j)] = h[IDX_T(1, j)];
    hfl_x[IDX_U(2, j)] = h[IDX_T(1, j)];
    hfr_x[IDX_U(2, j)] = h[IDX_T(2, j)];
    hfl_x[IDX_U(3, j)] = h[IDX_T(2, j)];
    hfr_x[IDX_U(3, j)] = h[IDX_T(2, j)];
    hfr_x[IDX_U(nx - 1, j)] = h[IDX_T(nx - 1, j)];
    hfl_x[IDX_U(nx, j)] = h[IDX_T(nx - 1, j)];
    hfr_x[IDX_U(nx, j)] = h[IDX_T(nx, j)];
    hfl_x[IDX_U(nx + 1, j)] = h[IDX_T(nx, j)];
    hfr_x[IDX_U(nx + 1, j)] = h[IDX_T(nx, j)];
}

// ===================== 3. X face transport (F90:185) ====================
__global__ void k_trans_x(const double * __restrict__ u_face_x, const double * __restrict__ hfl_x, const double * __restrict__ hfr_x,
                          const double * __restrict__ dy_cu, double * __restrict__ mass_flux_x, int nx, int ny) {
    // do concurrent(j=1:ny, i=2:nx)
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t ni = (size_t)(nx - 1);          // i = 2 .. nx
    if (t >= ni * (size_t)ny) return;
    int i = (int)(t % ni) + 2;
    int j = (int)(t / ni) + 1;

    double u, h_face;
    u = u_face_x[IDX_U(i, j)];
    if (u >= 0.0) h_face = hfl_x[IDX_U(i, j)];
    else          h_face = hfr_x[IDX_U(i, j)];
    mass_flux_x[IDX_U(i, j)] = u * h_face * dy_cu[IDX_U(i, j)];
}

// ===================== 4. X walls (F90:194) =============================
__global__ void k_wall_x(double * __restrict__ mass_flux_x, int nx, int ny) {
    // do concurrent(j=1:ny)
    int j = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x) + 1;
    if (j > ny) return;
    mass_flux_x[IDX_U(1, j)] = 0.0;
    mass_flux_x[IDX_U(nx + 1, j)] = 0.0;
}

// ===================== 5. Y-direction PPM (F90:202) =====================
__global__ void k_ppm_y(const double * __restrict__ h, const double * __restrict__ wet_T, double * __restrict__ hfl_y, double * __restrict__ hfr_y,
                        int nx, int ny, int do_pos, double h_min_pos) {
    // do concurrent(j=3:ny-2, i=1:nx)
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nx * (size_t)(ny - 4)) return;
    int i = (int)(t % (size_t)nx) + 1;
    int j = (int)(t / (size_t)nx) + 3;

    double dh_m1, dh_0, dh_p1, h_left, h_right, hm2, hm1, h0, hp1, hp2;
    h0 = h[IDX_T(i, j)];
    hm1 = ppm_mirror_h(h[IDX_T(i, j - 1)], h0, wet_T[IDX_T(i, j - 1)]);
    hp1 = ppm_mirror_h(h[IDX_T(i, j + 1)], h0, wet_T[IDX_T(i, j + 1)]);
    hm2 = ppm_mirror_h(h[IDX_T(i, j - 2)], hm1, wet_T[IDX_T(i, j - 2)]);
    hp2 = ppm_mirror_h(h[IDX_T(i, j + 2)], hp1, wet_T[IDX_T(i, j + 2)]);
    ppm_limited_slope(hm2, hm1, h0, &dh_m1);
    ppm_limited_slope(hm1, h0, hp1, &dh_0);
    ppm_limited_slope(h0, hp1, hp2, &dh_p1);
    dh_0 = dh_0 * wet_T[IDX_T(i, j - 1)] * wet_T[IDX_T(i, j)] * wet_T[IDX_T(i, j + 1)];
    h_left = 0.5 * (hm1 + h0) - (dh_0 - dh_m1) / 6.0;
    h_right = 0.5 * (h0 + hp1) - (dh_p1 - dh_0) / 6.0;
    ppm_cell_limiter(h0, &h_left, &h_right);
    if (do_pos) ppm_limit_pos(h0, &h_left, &h_right, h_min_pos);
    hfr_y[IDX_V(i, j)] = h_left;
    hfl_y[IDX_V(i, j + 1)] = h_right;
}

// ===================== 6. Y boundary cells (F90:221) ====================
__global__ void k_bnd_y(const double * __restrict__ h, double * __restrict__ hfl_y, double * __restrict__ hfr_y, int nx, int ny) {
    // do concurrent(i=1:nx)
    int i = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x) + 1;
    if (i > nx) return;
    hfl_y[IDX_V(i, 1)] = h[IDX_T(i, 1)];
    hfr_y[IDX_V(i, 1)] = h[IDX_T(i, 1)];
    hfl_y[IDX_V(i, 2)] = h[IDX_T(i, 1)];
    hfr_y[IDX_V(i, 2)] = h[IDX_T(i, 2)];
    hfl_y[IDX_V(i, 3)] = h[IDX_T(i, 2)];
    hfr_y[IDX_V(i, 3)] = h[IDX_T(i, 2)];
    hfr_y[IDX_V(i, ny - 1)] = h[IDX_T(i, ny - 1)];
    hfl_y[IDX_V(i, ny)] = h[IDX_T(i, ny - 1)];
    hfr_y[IDX_V(i, ny)] = h[IDX_T(i, ny)];
    hfl_y[IDX_V(i, ny + 1)] = h[IDX_T(i, ny)];
    hfr_y[IDX_V(i, ny + 1)] = h[IDX_T(i, ny)];
}

// ===================== 7. Y face transport (F90:235) ====================
__global__ void k_trans_y(const double * __restrict__ v_face_y, const double * __restrict__ hfl_y, const double * __restrict__ hfr_y,
                          const double * __restrict__ dx_cv, double * __restrict__ mass_flux_y, int nx, int ny) {
    // do concurrent(j=2:ny, i=1:nx)
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nx * (size_t)(ny - 1)) return;
    int i = (int)(t % (size_t)nx) + 1;
    int j = (int)(t / (size_t)nx) + 2;

    double v, h_face;
    v = v_face_y[IDX_V(i, j)];
    if (v >= 0.0) h_face = hfl_y[IDX_V(i, j)];
    else          h_face = hfr_y[IDX_V(i, j)];
    mass_flux_y[IDX_V(i, j)] = v * h_face * dx_cv[IDX_V(i, j)];
}

// ===================== 8. Y walls (F90:244) =============================
__global__ void k_wall_y(double * __restrict__ mass_flux_y, int nx, int ny) {
    // do concurrent(i=1:nx)
    int i = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x) + 1;
    if (i > nx) return;
    mass_flux_y[IDX_V(i, 1)] = 0.0;
    mass_flux_y[IDX_V(i, ny + 1)] = 0.0;
}

// ===================== 9. Flux divergence (F90:252) =====================
__global__ void k_div(const double * __restrict__ mass_flux_x, const double * __restrict__ mass_flux_y,
                      const double * __restrict__ iareaT, double * __restrict__ flux_h, int nx, int ny) {
    // do concurrent(j=1:ny, i=1:nx)
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)nx * (size_t)ny) return;
    int i = (int)(t % (size_t)nx) + 1;
    int j = (int)(t / (size_t)nx) + 1;

    flux_h[IDX_T(i, j)] = ((mass_flux_x[IDX_U(i + 1, j)] - mass_flux_x[IDX_U(i, j)]) +
                           (mass_flux_y[IDX_V(i, j + 1)] - mass_flux_y[IDX_V(i, j)])) *
                          iareaT[IDX_T(i, j)];
}

// ===================== launcher =========================================
// Block size 128 to match what nvfortran picks for these loops (-Minfo says
// `CUDA threads(128)`), so the comparison isn't a launch-geometry artifact.
#define TPB 128
#define NBLK(n) (int)(((size_t)(n) + TPB - 1) / TPB)

// `sync` selects the SYNCHRONISATION MODEL, which at small domains matters far
// more than codegen and makes this comparison honest or dishonest:
//
//   0 = async. Queue all 9 kernels, return. The caller syncs when it needs the
//       data. Maximum pipelining.
//   1 = sync once, after the 9th kernel (one timestep's worth of work).
//   2 = sync after EVERY kernel. This is what nvfortran's `do concurrent`
//       appears to do: at 256^2 its kernels are within 12% of these, yet its
//       wall-clock is 3.5x, because ~10 us of dead time sits between each of
//       its 9 loops. Mode 2 pays the same tax, so `dc vs cuda:2` isolates
//       CODEGEN from the launch model.
//
// Reporting mode 0 against `do concurrent` without saying so would credit CUDA
// with a pipelining advantage that has nothing to do with generated code.
static inline void maybe_sync(int sync, int level) {
    if (sync >= level) {
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
    }
}

extern "C" void continuity_cuda_launch(const double *h, const double *u_face_x,
                                       const double *v_face_y, const double *wet_T,
                                       const double *dy_cu, const double *dx_cv,
                                       const double *iareaT, double *hfl_x, double *hfr_x,
                                       double *hfl_y, double *hfr_y, double *mass_flux_x,
                                       double *mass_flux_y, double *flux_h, int nx, int ny,
                                       int do_pos, double h_min_pos, int sync) {
    k_ppm_x<<<NBLK((size_t)(nx - 4) * ny), TPB>>>(h, wet_T, hfl_x, hfr_x, nx, ny, do_pos, h_min_pos);
    maybe_sync(sync, 2);
    k_bnd_x<<<NBLK(ny), TPB>>>(h, hfl_x, hfr_x, nx, ny);
    maybe_sync(sync, 2);
    k_trans_x<<<NBLK((size_t)(nx - 1) * ny), TPB>>>(u_face_x, hfl_x, hfr_x, dy_cu, mass_flux_x, nx, ny);
    maybe_sync(sync, 2);
    k_wall_x<<<NBLK(ny), TPB>>>(mass_flux_x, nx, ny);
    maybe_sync(sync, 2);
    k_ppm_y<<<NBLK((size_t)nx * (ny - 4)), TPB>>>(h, wet_T, hfl_y, hfr_y, nx, ny, do_pos, h_min_pos);
    maybe_sync(sync, 2);
    k_bnd_y<<<NBLK(nx), TPB>>>(h, hfl_y, hfr_y, nx, ny);
    maybe_sync(sync, 2);
    k_trans_y<<<NBLK((size_t)nx * (ny - 1)), TPB>>>(v_face_y, hfl_y, hfr_y, dx_cv, mass_flux_y, nx, ny);
    maybe_sync(sync, 2);
    k_wall_y<<<NBLK(nx), TPB>>>(mass_flux_y, nx, ny);
    maybe_sync(sync, 2);
    k_div<<<NBLK((size_t)nx * ny), TPB>>>(mass_flux_x, mass_flux_y, iareaT, flux_h, nx, ny);
    maybe_sync(sync, 1);
}
