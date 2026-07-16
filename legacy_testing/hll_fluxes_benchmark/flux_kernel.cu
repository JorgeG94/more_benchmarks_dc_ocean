// Faithful CUDA C transliteration of kernel_flux.F90's HLL path.
//
// PROVENANCE: hand-ported from <model>-sea-ice @ feat/sea-ice,
// src/core/coastal/kernels/structured/barotropic/kernel_flux.F90
// (md5 1ea40efad8567a85914561db5bfc3a55). The Fortran file sits next to this
// one, byte-identical to production. If it changes, this must change too — a
// silent divergence would make the benchmark compare two different algorithms
// and report the difference as a compiler result.
//
// FAITHFULNESS RULES observed here, because the point is to measure the
// COMPILER, not my cleverness:
//   * Same operation order, same parenthesisation, same intermediate temporaries.
//     Fortran pins evaluation order inside parens; reassociating would change
//     the last bits and make "do they agree?" unanswerable.
//   * Same branch structure (no flattening a branch into arithmetic).
//   * One thread per cell, exactly like the `do concurrent(j,i)` iteration space.
//   * No shared-memory tiling, no __ldg, no launch_bounds. Those are the TUNED
//     variant's job; mixing them in here would answer a different question.
//
// INDEXING: Fortran h(nx,ny) is column-major and 1-based. The device arrays are
// the SAME allocation (via host_data use_device), so this must linearise
// identically: h(i,j) -> h[(i-1) + nx*(j-1)]. IDX below keeps the Fortran 1-based
// (i,j) at every call site so the port reads line-by-line against the original.

#include <cstdio>
#include <cuda_runtime.h>

// Verbatim from constants.F90 — see the Fortran stub for line refs.
#define GRAVITY              9.80665
#define DRY_TOLERANCE        1.0e-6
#define THIN_LAYER_THRESHOLD 1.0e-3

// Fortran 1-based (i,j) -> 0-based linear, column-major.
#define IDX(i, j) ((size_t)((i) - 1) + (size_t)nx * (size_t)((j) - 1))

// --- minmod (kernel_flux.F90:380) ---
__device__ __forceinline__ double minmod(double a, double b) {
    if (a * b <= 0.0)          return 0.0;
    else if (fabs(a) < fabs(b)) return a;
    else                        return b;
}

// --- extract_velocity (:395) ---
__device__ __forceinline__ void extract_velocity(double h_val, double hu_val, double hv_val,
                                                 double *u_val, double *v_val) {
    if (h_val > DRY_TOLERANCE) {
        *u_val = hu_val / h_val;
        *v_val = hv_val / h_val;
    } else {
        *u_val = 0.0;
        *v_val = 0.0;
    }
}

// --- hll_flux_x (:410) ---
__device__ __forceinline__ void hll_flux_x(double h_l, double hu_l, double hv_l,
                                           double h_r, double hu_r, double hv_r,
                                           double *f_h, double *f_hu, double *f_hv) {
    double u_l, u_r, v_l, v_r, a_l, a_r, s_l, s_r, denom;
    double fl_h, fl_hu, fl_hv, fr_h, fr_hu, fr_hv;

    if (h_l > DRY_TOLERANCE) { u_l = hu_l / h_l; v_l = hv_l / h_l; }
    else                     { u_l = 0.0;        v_l = 0.0;        }
    if (h_r > DRY_TOLERANCE) { u_r = hu_r / h_r; v_r = hv_r / h_r; }
    else                     { u_r = 0.0;        v_r = 0.0;        }

    a_l = sqrt(GRAVITY * fmax(h_l, 0.0));
    a_r = sqrt(GRAVITY * fmax(h_r, 0.0));
    s_l = fmin(u_l - a_l, u_r - a_r);
    s_r = fmax(u_l + a_l, u_r + a_r);

    fl_h  = hu_l;
    fl_hu = hu_l * u_l + 0.5 * GRAVITY * h_l * h_l;
    fl_hv = hu_l * v_l;

    fr_h  = hu_r;
    fr_hu = hu_r * u_r + 0.5 * GRAVITY * h_r * h_r;
    fr_hv = hu_r * v_r;

    if (s_l >= 0.0) {
        *f_h = fl_h; *f_hu = fl_hu; *f_hv = fl_hv;
    } else if (s_r <= 0.0) {
        *f_h = fr_h; *f_hu = fr_hu; *f_hv = fr_hv;
    } else {
        denom = 1.0 / (s_r - s_l);
        *f_h  = (s_r * fl_h  - s_l * fr_h  + s_l * s_r * (h_r  - h_l)) * denom;
        *f_hu = (s_r * fl_hu - s_l * fr_hu + s_l * s_r * (hu_r - hu_l)) * denom;
        *f_hv = (s_r * fl_hv - s_l * fr_hv + s_l * s_r * (hv_r - hv_l)) * denom;
    }
}

// --- hll_flux_y (:471) ---
__device__ __forceinline__ void hll_flux_y(double h_b, double hu_b, double hv_b,
                                           double h_t, double hu_t, double hv_t,
                                           double *g_h, double *g_hu, double *g_hv) {
    double u_b, u_t, v_b, v_t, a_b, a_t, s_l, s_r, denom;
    double gl_h, gl_hu, gl_hv, gr_h, gr_hu, gr_hv;

    if (h_b > DRY_TOLERANCE) { u_b = hu_b / h_b; v_b = hv_b / h_b; }
    else                     { u_b = 0.0;        v_b = 0.0;        }
    if (h_t > DRY_TOLERANCE) { u_t = hu_t / h_t; v_t = hv_t / h_t; }
    else                     { u_t = 0.0;        v_t = 0.0;        }

    a_b = sqrt(GRAVITY * fmax(h_b, 0.0));
    a_t = sqrt(GRAVITY * fmax(h_t, 0.0));
    s_l = fmin(v_b - a_b, v_t - a_t);
    s_r = fmax(v_b + a_b, v_t + a_t);

    gl_h  = hv_b;
    gl_hu = hu_b * v_b;
    gl_hv = hv_b * v_b + 0.5 * GRAVITY * h_b * h_b;

    gr_h  = hv_t;
    gr_hu = hu_t * v_t;
    gr_hv = hv_t * v_t + 0.5 * GRAVITY * h_t * h_t;

    if (s_l >= 0.0) {
        *g_h = gl_h; *g_hu = gl_hu; *g_hv = gl_hv;
    } else if (s_r <= 0.0) {
        *g_h = gr_h; *g_hu = gr_hu; *g_hv = gr_hv;
    } else {
        denom = 1.0 / (s_r - s_l);
        *g_h  = (s_r * gl_h  - s_l * gr_h  + s_l * s_r * (h_t  - h_b)) * denom;
        *g_hu = (s_r * gl_hu - s_l * gr_hu + s_l * s_r * (hu_t - hu_b)) * denom;
        *g_hv = (s_r * gl_hv - s_l * gr_hv + s_l * s_r * (hv_t - hv_b)) * denom;
    }
}

// --- flux_cell (:104) + the do-concurrent body (:47) fused into one kernel ---
// The Fortran calls flux_cell from inside `do concurrent`; here the body is
// inlined by __forceinline__ on the helpers, which is what we WANT nvfortran to
// do too. Whether it does is precisely the thing under test.
__global__ void flux_hll_kernel(const double *__restrict__ h,
                                const double *__restrict__ hu,
                                const double *__restrict__ hv,
                                const double *__restrict__ b,
                                double *__restrict__ flux_h,
                                double *__restrict__ flux_hu,
                                double *__restrict__ flux_hv,
                                double *__restrict__ mass_flux_x,
                                double *__restrict__ mass_flux_y,
                                const int nx, const int ny, const int nghost,
                                const double dx, const double dy)
{
    // Fortran: do concurrent(j = nghost+1 : ny-nghost, i = nghost+1 : nx-nghost)
    const int i = nghost + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    const int j = nghost + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (i > nx - nghost || j > ny - nghost) return;

    double eta_c, eta_im1, eta_ip1, eta_im2, eta_ip2;
    double eta_jm1, eta_jp1, eta_jm2, eta_jp2;
    double seta_x, shu_x, shv_x, seta_y, shu_y, shv_y;
    double seta_x_nb, shu_x_nb, shv_x_nb, seta_y_nb, shu_y_nb, shv_y_nb;
    double eta_L, eta_R, hu_L, hv_L, hu_R, hv_R, h_L, h_R;
    double b_max, h_hyd_l, h_hyd_r;
    double u_l, v_l, u_r, v_r;
    double fx_h_l, fx_hu_l, fx_hv_l, fx_h_r, fx_hu_r, fx_hv_r;
    double fy_h_b, fy_hu_b, fy_hv_b, fy_h_t, fy_hu_t, fy_hv_t;
    double h_hyd_left, h_pre_left, h_hyd_right, h_pre_right;
    double h_hyd_bottom, h_pre_bottom, h_hyd_top, h_pre_top;

    const double half_g = 0.5 * GRAVITY;

    // eta = h + B  (:148)
    eta_c   = h[IDX(i, j)]     + b[IDX(i, j)];
    eta_im1 = h[IDX(i - 1, j)] + b[IDX(i - 1, j)];
    eta_ip1 = h[IDX(i + 1, j)] + b[IDX(i + 1, j)];
    eta_im2 = h[IDX(i - 2, j)] + b[IDX(i - 2, j)];
    eta_ip2 = h[IDX(i + 2, j)] + b[IDX(i + 2, j)];
    eta_jm1 = h[IDX(i, j - 1)] + b[IDX(i, j - 1)];
    eta_jp1 = h[IDX(i, j + 1)] + b[IDX(i, j + 1)];
    eta_jm2 = h[IDX(i, j - 2)] + b[IDX(i, j - 2)];
    eta_jp2 = h[IDX(i, j + 2)] + b[IDX(i, j + 2)];

    // first-order fallback near wet/dry (:159)
    if (h[IDX(i, j)] < THIN_LAYER_THRESHOLD ||
        h[IDX(i - 1, j)] < DRY_TOLERANCE ||
        h[IDX(i + 1, j)] < DRY_TOLERANCE) {
        seta_x = 0.0; shu_x = 0.0; shv_x = 0.0;
    } else {
        seta_x = minmod(eta_c - eta_im1, eta_ip1 - eta_c);
        shu_x  = minmod(hu[IDX(i, j)] - hu[IDX(i - 1, j)], hu[IDX(i + 1, j)] - hu[IDX(i, j)]);
        shv_x  = minmod(hv[IDX(i, j)] - hv[IDX(i - 1, j)], hv[IDX(i + 1, j)] - hv[IDX(i, j)]);
    }

    if (h[IDX(i, j)] < THIN_LAYER_THRESHOLD ||
        h[IDX(i, j - 1)] < DRY_TOLERANCE ||
        h[IDX(i, j + 1)] < DRY_TOLERANCE) {
        seta_y = 0.0; shu_y = 0.0; shv_y = 0.0;
    } else {
        seta_y = minmod(eta_c - eta_jm1, eta_jp1 - eta_c);
        shu_y  = minmod(hu[IDX(i, j)] - hu[IDX(i, j - 1)], hu[IDX(i, j + 1)] - hu[IDX(i, j)]);
        shv_y  = minmod(hv[IDX(i, j)] - hv[IDX(i, j - 1)], hv[IDX(i, j + 1)] - hv[IDX(i, j)]);
    }

    // ---- x: LEFT interface (i-1/2)  (:186) ----
    eta_R = eta_c - 0.5 * seta_x;
    hu_R  = hu[IDX(i, j)] - 0.5 * shu_x;
    hv_R  = hv[IDX(i, j)] - 0.5 * shv_x;
    h_R   = fmax(0.0, eta_R - b[IDX(i, j)]);
    h_pre_left = h_R;

    if (h[IDX(i - 1, j)] < THIN_LAYER_THRESHOLD ||
        h[IDX(i - 2, j)] < DRY_TOLERANCE ||
        h[IDX(i, j)] < DRY_TOLERANCE) {
        seta_x_nb = 0.0; shu_x_nb = 0.0; shv_x_nb = 0.0;
    } else {
        seta_x_nb = minmod(eta_im1 - eta_im2, eta_c - eta_im1);
        shu_x_nb  = minmod(hu[IDX(i - 1, j)] - hu[IDX(i - 2, j)], hu[IDX(i, j)] - hu[IDX(i - 1, j)]);
        shv_x_nb  = minmod(hv[IDX(i - 1, j)] - hv[IDX(i - 2, j)], hv[IDX(i, j)] - hv[IDX(i - 1, j)]);
    }
    eta_L = eta_im1 + 0.5 * seta_x_nb;
    hu_L  = hu[IDX(i - 1, j)] + 0.5 * shu_x_nb;
    hv_L  = hv[IDX(i - 1, j)] + 0.5 * shv_x_nb;
    h_L   = fmax(0.0, eta_L - b[IDX(i - 1, j)]);

    b_max   = fmax(b[IDX(i - 1, j)], b[IDX(i, j)]);
    h_hyd_l = fmax(0.0, h_L + b[IDX(i - 1, j)] - b_max);
    h_hyd_r = fmax(0.0, h_R + b[IDX(i, j)] - b_max);
    h_hyd_left = h_hyd_r;

    extract_velocity(h_L, hu_L, hv_L, &u_l, &v_l);
    extract_velocity(h_R, hu_R, hv_R, &u_r, &v_r);
    hll_flux_x(h_hyd_l, h_hyd_l * u_l, h_hyd_l * v_l,
               h_hyd_r, h_hyd_r * u_r, h_hyd_r * v_r,
               &fx_h_l, &fx_hu_l, &fx_hv_l);

    // ---- x: RIGHT interface (i+1/2)  (:229) ----
    eta_L = eta_c + 0.5 * seta_x;
    hu_L  = hu[IDX(i, j)] + 0.5 * shu_x;
    hv_L  = hv[IDX(i, j)] + 0.5 * shv_x;
    h_L   = fmax(0.0, eta_L - b[IDX(i, j)]);
    h_pre_right = h_L;

    if (h[IDX(i + 1, j)] < THIN_LAYER_THRESHOLD ||
        h[IDX(i, j)] < DRY_TOLERANCE ||
        h[IDX(i + 2, j)] < DRY_TOLERANCE) {
        seta_x_nb = 0.0; shu_x_nb = 0.0; shv_x_nb = 0.0;
    } else {
        seta_x_nb = minmod(eta_ip1 - eta_c, eta_ip2 - eta_ip1);
        shu_x_nb  = minmod(hu[IDX(i + 1, j)] - hu[IDX(i, j)], hu[IDX(i + 2, j)] - hu[IDX(i + 1, j)]);
        shv_x_nb  = minmod(hv[IDX(i + 1, j)] - hv[IDX(i, j)], hv[IDX(i + 2, j)] - hv[IDX(i + 1, j)]);
    }
    eta_R = eta_ip1 - 0.5 * seta_x_nb;
    hu_R  = hu[IDX(i + 1, j)] - 0.5 * shu_x_nb;
    hv_R  = hv[IDX(i + 1, j)] - 0.5 * shv_x_nb;
    h_R   = fmax(0.0, eta_R - b[IDX(i + 1, j)]);

    b_max   = fmax(b[IDX(i, j)], b[IDX(i + 1, j)]);
    h_hyd_l = fmax(0.0, h_L + b[IDX(i, j)] - b_max);
    h_hyd_r = fmax(0.0, h_R + b[IDX(i + 1, j)] - b_max);
    h_hyd_right = h_hyd_l;

    extract_velocity(h_L, hu_L, hv_L, &u_l, &v_l);
    extract_velocity(h_R, hu_R, hv_R, &u_r, &v_r);
    hll_flux_x(h_hyd_l, h_hyd_l * u_l, h_hyd_l * v_l,
               h_hyd_r, h_hyd_r * u_r, h_hyd_r * v_r,
               &fx_h_r, &fx_hu_r, &fx_hv_r);

    // ---- y: BOTTOM interface (j-1/2)  (:272) ----
    eta_R = eta_c - 0.5 * seta_y;
    hu_R  = hu[IDX(i, j)] - 0.5 * shu_y;
    hv_R  = hv[IDX(i, j)] - 0.5 * shv_y;
    h_R   = fmax(0.0, eta_R - b[IDX(i, j)]);
    h_pre_bottom = h_R;

    if (h[IDX(i, j - 1)] < THIN_LAYER_THRESHOLD ||
        h[IDX(i, j - 2)] < DRY_TOLERANCE ||
        h[IDX(i, j)] < DRY_TOLERANCE) {
        seta_y_nb = 0.0; shu_y_nb = 0.0; shv_y_nb = 0.0;
    } else {
        seta_y_nb = minmod(eta_jm1 - eta_jm2, eta_c - eta_jm1);
        shu_y_nb  = minmod(hu[IDX(i, j - 1)] - hu[IDX(i, j - 2)], hu[IDX(i, j)] - hu[IDX(i, j - 1)]);
        shv_y_nb  = minmod(hv[IDX(i, j - 1)] - hv[IDX(i, j - 2)], hv[IDX(i, j)] - hv[IDX(i, j - 1)]);
    }
    eta_L = eta_jm1 + 0.5 * seta_y_nb;
    hu_L  = hu[IDX(i, j - 1)] + 0.5 * shu_y_nb;
    hv_L  = hv[IDX(i, j - 1)] + 0.5 * shv_y_nb;
    h_L   = fmax(0.0, eta_L - b[IDX(i, j - 1)]);

    b_max   = fmax(b[IDX(i, j - 1)], b[IDX(i, j)]);
    h_hyd_l = fmax(0.0, h_L + b[IDX(i, j - 1)] - b_max);
    h_hyd_r = fmax(0.0, h_R + b[IDX(i, j)] - b_max);
    h_hyd_bottom = h_hyd_r;

    extract_velocity(h_L, hu_L, hv_L, &u_l, &v_l);
    extract_velocity(h_R, hu_R, hv_R, &u_r, &v_r);
    hll_flux_y(h_hyd_l, h_hyd_l * u_l, h_hyd_l * v_l,
               h_hyd_r, h_hyd_r * u_r, h_hyd_r * v_r,
               &fy_h_b, &fy_hu_b, &fy_hv_b);

    // ---- y: TOP interface (j+1/2)  (:315) ----
    eta_L = eta_c + 0.5 * seta_y;
    hu_L  = hu[IDX(i, j)] + 0.5 * shu_y;
    hv_L  = hv[IDX(i, j)] + 0.5 * shv_y;
    h_L   = fmax(0.0, eta_L - b[IDX(i, j)]);
    h_pre_top = h_L;

    if (h[IDX(i, j + 1)] < THIN_LAYER_THRESHOLD ||
        h[IDX(i, j)] < DRY_TOLERANCE ||
        h[IDX(i, j + 2)] < DRY_TOLERANCE) {
        seta_y_nb = 0.0; shu_y_nb = 0.0; shv_y_nb = 0.0;
    } else {
        seta_y_nb = minmod(eta_jp1 - eta_c, eta_jp2 - eta_jp1);
        shu_y_nb  = minmod(hu[IDX(i, j + 1)] - hu[IDX(i, j)], hu[IDX(i, j + 2)] - hu[IDX(i, j + 1)]);
        shv_y_nb  = minmod(hv[IDX(i, j + 1)] - hv[IDX(i, j)], hv[IDX(i, j + 2)] - hv[IDX(i, j + 1)]);
    }
    eta_R = eta_jp1 - 0.5 * seta_y_nb;
    hu_R  = hu[IDX(i, j + 1)] - 0.5 * shu_y_nb;
    hv_R  = hv[IDX(i, j + 1)] - 0.5 * shv_y_nb;
    h_R   = fmax(0.0, eta_R - b[IDX(i, j + 1)]);

    b_max   = fmax(b[IDX(i, j)], b[IDX(i, j + 1)]);
    h_hyd_l = fmax(0.0, h_L + b[IDX(i, j)] - b_max);
    h_hyd_r = fmax(0.0, h_R + b[IDX(i, j + 1)] - b_max);
    h_hyd_top = h_hyd_l;

    extract_velocity(h_L, hu_L, hv_L, &u_l, &v_l);
    extract_velocity(h_R, hu_R, hv_R, &u_r, &v_r);
    hll_flux_y(h_hyd_l, h_hyd_l * u_l, h_hyd_l * v_l,
               h_hyd_r, h_hyd_r * u_r, h_hyd_r * v_r,
               &fy_h_t, &fy_hu_t, &fy_hv_t);

    // ---- net flux divergence + well-balanced correction  (:358) ----
    // Parenthesisation copied exactly: Fortran pins evaluation order in parens,
    // so reassociating here would move the last bits and break the comparison.
    flux_h[IDX(i, j)] = -(fx_h_r - fx_h_l) / dx
                        - (fy_h_t - fy_h_b) / dy;

    flux_hu[IDX(i, j)] = -(fx_hu_r - fx_hu_l) / dx
                         - (fy_hu_t - fy_hu_b) / dy
                         - half_g * (h_hyd_left * h_hyd_left - h_pre_left * h_pre_left
                                     - h_hyd_right * h_hyd_right + h_pre_right * h_pre_right) / dx;

    flux_hv[IDX(i, j)] = -(fx_hv_r - fx_hv_l) / dx
                         - (fy_hv_t - fy_hv_b) / dy
                         - half_g * (h_hyd_bottom * h_hyd_bottom - h_pre_bottom * h_pre_bottom
                                     - h_hyd_top * h_hyd_top + h_pre_top * h_pre_top) / dy;

    // Per-face mass fluxes (:373 and the DC body at :52).
    mass_flux_x[IDX(i, j)] = fx_h_r;
    mass_flux_y[IDX(i, j)] = fy_h_t;
    // Only i == nghost+1 writes (i-1, j), so this is not a race — one writer.
    if (i == nghost + 1) mass_flux_x[IDX(i - 1, j)] = fx_h_l;
    if (j == nghost + 1) mass_flux_y[IDX(i, j - 1)] = fy_h_b;
}

extern "C" void flux_hll_cuda_launch(const double *h, const double *hu, const double *hv,
                                     const double *b,
                                     double *flux_h, double *flux_hu, double *flux_hv,
                                     double *mass_flux_x, double *mass_flux_y,
                                     int nx, int ny, int nghost,
                                     double dx, double dy, int sync)
{
    const int ni = nx - 2 * nghost;
    const int nj = ny - 2 * nghost;
    // 32 in x = one warp along Fortran's fast index -> coalesced.
    const dim3 block(32, 8, 1);
    const dim3 grid((ni + block.x - 1) / block.x, (nj + block.y - 1) / block.y, 1);

    flux_hll_kernel<<<grid, block>>>(h, hu, hv, b, flux_h, flux_hu, flux_hv,
                                     mass_flux_x, mass_flux_y, nx, ny, nghost, dx, dy);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "flux_hll launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    if (sync) {
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            std::fprintf(stderr, "flux_hll sync failed: %s\n", cudaGetErrorString(err));
    }
}
