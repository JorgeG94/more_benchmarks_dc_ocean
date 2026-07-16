// OPTIMIZED CUDA for the ocean ALE remap (zstar, PPM, 2 tracers).
//
// Baseline being beaten: the FAITHFUL port in ale_kernel.cu -- ale_remap_cuda(...,
// fused=0), one kernel per production do-concurrent loop (10 launches, T and S
// each remapped by their own k_tracer that rebuilds the whole column geometry
// from scratch, plus total_h/h_ref written to global as intermediates).
//
// This file keeps EVERY per-cell arithmetic expression in the SAME ORDER as the
// faithful path, so the result is bit-identical (ab_main.cu checks max|diff|==0).
// The wins are pure waste-removal, exactly the three levers the task names:
//
//   1. T and S share the column geometry. The faithful path builds z_old/z_new +
//      the PPM edge reconstruction + the overlap sweep TWICE (once per tracer).
//      Here it is built ONCE and reused for both tracers in a single kernel
//      (k_opt_tracer2). Per-tracer arithmetic is untouched.
//   2. No intermediate global scratch. total_h and h_ref are consumed only
//      inside the pre-pass, so they live in registers and never reach global
//      memory (the faithful path round-trips both through DRAM). The cheap
//      pre/post do-concurrent loops are each fused into one per-column kernel.
//   3. 32-bit indexing. nx*ny*nz < 2^31 for every realistic domain (479*303*30
//      = 4.35e6 << 2.1e9), so the size_t address math in the faithful IDX3 is
//      pure overhead -- fewer address registers, cheaper indexing.
//
// Fortran column-major layout, 0-based indices, mirrors ale_kernel.cu exactly.

#include "gpu_rt.h"
#include <cstdio>

#ifndef NZ_STACK_MAX
#define NZ_STACK_MAX 128
#endif

#ifndef TPB
#define TPB 128
#endif

#define H_FLOOR 1.5e-4

// 32-bit indexing (default): valid while nx*ny*nz < 2^31 (true at every
// realistic size). -DIDX32=0 selects the size_t path, to isolate its cost.
#ifndef IDX32
#define IDX32 1
#endif
#if IDX32
#define IDX2(i, j, nx) ((int)(i) + (int)(j) * (int)(nx))
#define IDX3(i, j, k, nx, ny) ((int)(i) + (int)(j) * (int)(nx) + (int)(k) * (int)(nx) * (int)(ny))
#else
#define IDX2(i, j, nx) ((size_t)(i) + (size_t)(j) * (size_t)(nx))
#define IDX3(i, j, k, nx, ny) ((size_t)(i) + (size_t)(j) * (size_t)(nx) + (size_t)(k) * (size_t)(nx) * (size_t)(ny))
#endif

typedef double real_t;

// ---------------------------------------------------------------------------
// Per-column PPM remap -- copied verbatim from ale_kernel.cu so the arithmetic
// order is identical. These operate on per-thread stack columns (indexed by k
// only), so there is no global-index change to make and no bit difference.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void remap_column_plm_d(int nz, const real_t *dz_old,
                                                   const real_t *dz_new,
                                                   const real_t *q_old, real_t *q_new) {
    real_t z_old[NZ_STACK_MAX + 1], z_new[NZ_STACK_MAX + 1], slope[NZ_STACK_MAX];
    if (nz == 1) { q_new[0] = q_old[0]; return; }
    z_old[0] = 0.0; z_new[0] = 0.0;
    for (int k = 1; k <= nz; ++k) {
        z_old[k] = z_old[k - 1] + dz_old[k - 1];
        z_new[k] = z_new[k - 1] + dz_new[k - 1];
    }
    slope[0] = 0.0;
    for (int k = 2; k <= nz - 1; ++k) {
        real_t dq_l = q_old[k - 1] - q_old[k - 2];
        real_t dq_r = q_old[k] - q_old[k - 1];
        if (dq_l * dq_r > 0.0)
            slope[k - 1] = 0.5 * copysign(fmin(fabs(dq_l), fabs(dq_r)), dq_l);
        else
            slope[k - 1] = 0.0;
    }
    slope[nz - 1] = 0.0;
    int ko_start = 1;
    for (int k = 1; k <= nz; ++k) {
        if (dz_new[k - 1] <= 0.0) { q_new[k - 1] = 0.0; continue; }
        real_t integral = 0.0;
        for (int ko = ko_start; ko <= nz; ++ko) {
            real_t z_lo = fmax(z_new[k - 1], z_old[ko - 1]);
            real_t z_hi = fmin(z_new[k], z_old[ko]);
            real_t overlap = z_hi - z_lo;
            if (overlap <= 0.0) { if (z_old[ko] > z_new[k]) break; else continue; }
            if (dz_old[ko - 1] > 0.0) {
                real_t xi_lo = (z_lo - z_old[ko - 1]) / dz_old[ko - 1];
                real_t xi_hi = (z_hi - z_old[ko - 1]) / dz_old[ko - 1];
                integral += overlap * (q_old[ko - 1] + slope[ko - 1] * (xi_lo + xi_hi - 1.0));
            } else {
                integral += q_old[ko - 1] * overlap;
            }
            if (z_old[ko] <= z_new[k]) ko_start = ko;
        }
        q_new[k - 1] = integral / dz_new[k - 1];
    }
}

__device__ __forceinline__ void remap_column_ppm_d(int nz, const real_t *dz_old,
                                                   const real_t *dz_new,
                                                   const real_t *q_old, real_t *q_new) {
    real_t z_old[NZ_STACK_MAX + 1], z_new[NZ_STACK_MAX + 1];
    real_t q_L[NZ_STACK_MAX], q_R[NZ_STACK_MAX], q6[NZ_STACK_MAX];

    if (nz == 1) { q_new[0] = q_old[0]; return; }
    if (nz == 2) { remap_column_plm_d(nz, dz_old, dz_new, q_old, q_new); return; }

    z_old[0] = 0.0; z_new[0] = 0.0;
    for (int k = 1; k <= nz; ++k) {
        z_old[k] = z_old[k - 1] + dz_old[k - 1];
        z_new[k] = z_new[k - 1] + dz_new[k - 1];
    }
    q_L[0] = q_old[0];
    q_R[nz - 1] = q_old[nz - 1];
    q_R[0] = 0.5 * (q_old[0] + q_old[1]);
    q_L[nz - 1] = 0.5 * (q_old[nz - 2] + q_old[nz - 1]);
    for (int k = 2; k <= nz - 1; ++k) {
        real_t edge = 0.5 * (q_old[k - 1] + q_old[k]);
        if (k - 1 >= 1 && k + 2 <= nz) {
            edge = (7.0 / 12.0) * (q_old[k - 1] + q_old[k]) -
                   (1.0 / 12.0) * (q_old[k - 2] + q_old[k + 1]);
        }
        q_R[k - 1] = edge;
        q_L[k] = edge;
    }
    if (nz >= 3) q_L[1] = q_R[0];

    for (int k = 1; k <= nz; ++k) {
        real_t q_min = q_old[k - 1], q_max = q_old[k - 1];
        if (k > 1) { q_min = fmin(q_min, q_old[k - 2]); q_max = fmax(q_max, q_old[k - 2]); }
        if (k < nz) { q_min = fmin(q_min, q_old[k]); q_max = fmax(q_max, q_old[k]); }
        q_L[k - 1] = fmax(q_min, fmin(q_max, q_L[k - 1]));
        q_R[k - 1] = fmax(q_min, fmin(q_max, q_R[k - 1]));
        real_t dq = q_R[k - 1] - q_L[k - 1];
        real_t dq_l = q_old[k - 1] - q_L[k - 1];
        real_t dq_r = q_R[k - 1] - q_old[k - 1];
        if (dq_l * dq_r <= 0.0) {
            q_L[k - 1] = q_old[k - 1];
            q_R[k - 1] = q_old[k - 1];
        } else {
            q6[k - 1] = 6.0 * q_old[k - 1] - 3.0 * (q_L[k - 1] + q_R[k - 1]);
            if (fabs(q6[k - 1]) > fabs(dq)) {
                if (q6[k - 1] * dq > 0.0) q_L[k - 1] = 3.0 * q_old[k - 1] - 2.0 * q_R[k - 1];
                else q_R[k - 1] = 3.0 * q_old[k - 1] - 2.0 * q_L[k - 1];
            }
        }
        q6[k - 1] = 6.0 * q_old[k - 1] - 3.0 * (q_L[k - 1] + q_R[k - 1]);
    }

    int ko_start = 1;
    for (int k = 1; k <= nz; ++k) {
        if (dz_new[k - 1] <= 0.0) { q_new[k - 1] = 0.0; continue; }
        real_t integral = 0.0;
        for (int ko = ko_start; ko <= nz; ++ko) {
            real_t z_lo = fmax(z_new[k - 1], z_old[ko - 1]);
            real_t z_hi = fmin(z_new[k], z_old[ko]);
            real_t overlap = z_hi - z_lo;
            if (overlap <= 0.0) { if (z_old[ko] > z_new[k]) break; else continue; }
            if (dz_old[ko - 1] > 0.0) {
                real_t xi_lo = (z_lo - z_old[ko - 1]) / dz_old[ko - 1];
                real_t xi_hi = (z_hi - z_old[ko - 1]) / dz_old[ko - 1];
                integral += dz_old[ko - 1] *
                            ((xi_hi - xi_lo) * q_L[ko - 1] +
                             0.5 * (xi_hi * xi_hi - xi_lo * xi_lo) * (q_R[ko - 1] - q_L[ko - 1] + q6[ko - 1]) -
                             (xi_hi * xi_hi * xi_hi - xi_lo * xi_lo * xi_lo) * q6[ko - 1] / 3.0);
            } else {
                integral += q_old[ko - 1] * overlap;
            }
            if (z_old[ko] <= z_new[k]) ko_start = ko;
        }
        q_new[k - 1] = integral / dz_new[k - 1];
    }
}

// PPM edge reconstruction, factored out so the fused kernel builds it per tracer
// with identical arithmetic. Copied verbatim from ale_kernel.cu::ppm_edges_d.
__device__ __forceinline__ void ppm_edges_d(int nz, const real_t *q_old,
                                            real_t *q_L, real_t *q_R, real_t *q6) {
    q_L[0] = q_old[0];
    q_R[nz - 1] = q_old[nz - 1];
    q_R[0] = 0.5 * (q_old[0] + q_old[1]);
    q_L[nz - 1] = 0.5 * (q_old[nz - 2] + q_old[nz - 1]);
    for (int k = 2; k <= nz - 1; ++k) {
        real_t edge = 0.5 * (q_old[k - 1] + q_old[k]);
        if (k - 1 >= 1 && k + 2 <= nz) {
            edge = (7.0 / 12.0) * (q_old[k - 1] + q_old[k]) -
                   (1.0 / 12.0) * (q_old[k - 2] + q_old[k + 1]);
        }
        q_R[k - 1] = edge;
        q_L[k] = edge;
    }
    if (nz >= 3) q_L[1] = q_R[0];
    for (int k = 1; k <= nz; ++k) {
        real_t q_min = q_old[k - 1], q_max = q_old[k - 1];
        if (k > 1) { q_min = fmin(q_min, q_old[k - 2]); q_max = fmax(q_max, q_old[k - 2]); }
        if (k < nz) { q_min = fmin(q_min, q_old[k]); q_max = fmax(q_max, q_old[k]); }
        q_L[k - 1] = fmax(q_min, fmin(q_max, q_L[k - 1]));
        q_R[k - 1] = fmax(q_min, fmin(q_max, q_R[k - 1]));
        real_t dq = q_R[k - 1] - q_L[k - 1];
        real_t dq_l = q_old[k - 1] - q_L[k - 1];
        real_t dq_r = q_R[k - 1] - q_old[k - 1];
        if (dq_l * dq_r <= 0.0) {
            q_L[k - 1] = q_old[k - 1]; q_R[k - 1] = q_old[k - 1];
        } else {
            q6[k - 1] = 6.0 * q_old[k - 1] - 3.0 * (q_L[k - 1] + q_R[k - 1]);
            if (fabs(q6[k - 1]) > fabs(dq)) {
                if (q6[k - 1] * dq > 0.0) q_L[k - 1] = 3.0 * q_old[k - 1] - 2.0 * q_R[k - 1];
                else q_R[k - 1] = 3.0 * q_old[k - 1] - 2.0 * q_L[k - 1];
            }
        }
        q6[k - 1] = 6.0 * q_old[k - 1] - 3.0 * (q_L[k - 1] + q_R[k - 1]);
    }
}

// ---------------------------------------------------------------------------
// PRE pass: total_h + h_ref + h_old snapshot + target_h in ONE per-column
// kernel. total_h and h_ref never leave registers (faithful writes both to
// global). target_h == (h_ref + eta) * dsig; h_ref == total_h - eta, so
// target_h keeps the faithful association ((total_h - eta) + eta) exactly.
// ---------------------------------------------------------------------------
__global__ void k_opt_pre(int nx, int ny, int nz, const real_t * __restrict__ h_layer,
                          real_t * __restrict__ h_old, real_t * __restrict__ target_h,
                          const real_t * __restrict__ eta, const real_t * __restrict__ dsig) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx * ny) return;
    int i = t % nx, j = t / nx;
    real_t s = 0.0;
    for (int k = 0; k < nz; ++k) {
        int o = IDX3(i, j, k, nx, ny);
        real_t h = h_layer[o];
        h_old[o] = h;
        s += h;
    }
    real_t e = eta[IDX2(i, j, nx)];
    real_t hr = s - e;               // h_ref  (register only)
    real_t ct = hr + e;              // == total column, faithful order preserved
    for (int k = 0; k < nz; ++k) target_h[IDX3(i, j, k, nx, ny)] = ct * dsig[k];
}

// ---------------------------------------------------------------------------
// Fused T+S tracer remap: geometry (z_old/z_new + overlap sweep) and PPM edges
// built ONCE, reused for both tracers. Per-tracer arithmetic is bit-identical
// to the faithful single-tracer k_tracer. Copied structure from ale_kernel.cu's
// k_tracer2 (the fused reference), only the index macros are 32-bit.
// ---------------------------------------------------------------------------
__global__ void k_opt_tracer2(int nx, int ny, int nz, const real_t * __restrict__ h_old,
                              const real_t * __restrict__ h_new, real_t * __restrict__ hTr_t,
                              real_t * __restrict__ hTr_s, real_t * __restrict__ bud_t,
                              real_t * __restrict__ bud_s) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx * ny) return;
    int i = t % nx, j = t / nx;
    real_t dz_old[NZ_STACK_MAX], dz_new[NZ_STACK_MAX];
    real_t cT[NZ_STACK_MAX], cS[NZ_STACK_MAX];
    real_t z_old[NZ_STACK_MAX + 1], z_new[NZ_STACK_MAX + 1];
    real_t qLT[NZ_STACK_MAX], qRT[NZ_STACK_MAX], q6T[NZ_STACK_MAX];
    real_t qLS[NZ_STACK_MAX], qRS[NZ_STACK_MAX], q6S[NZ_STACK_MAX];

    for (int k = 0; k < nz; ++k) {
        int o = IDX3(i, j, k, nx, ny);
        dz_old[k] = h_old[o];
        dz_new[k] = h_new[o];
        cT[k] = (dz_old[k] > H_FLOOR) ? hTr_t[o] / dz_old[k] : 0.0;
        cS[k] = (dz_old[k] > H_FLOOR) ? hTr_s[o] / dz_old[k] : 0.0;
    }
    if (nz < 3) {  // PPM degenerates below nz=3 -> faithful per-tracer fallback
        real_t oT[NZ_STACK_MAX], oS[NZ_STACK_MAX];
        remap_column_ppm_d(nz, dz_old, dz_new, cT, oT);
        remap_column_ppm_d(nz, dz_old, dz_new, cS, oS);
        for (int k = 0; k < nz; ++k) {
            int o = IDX3(i, j, k, nx, ny);
            real_t a = oT[k] * dz_new[k], b = oS[k] * dz_new[k];
            if (bud_t) bud_t[o] += (a - hTr_t[o]);
            if (bud_s) bud_s[o] += (b - hTr_s[o]);
            hTr_t[o] = a; hTr_s[o] = b;
        }
        return;
    }

    // ---- geometry built ONCE ----
    z_old[0] = 0.0; z_new[0] = 0.0;
    for (int k = 1; k <= nz; ++k) {
        z_old[k] = z_old[k - 1] + dz_old[k - 1];
        z_new[k] = z_new[k - 1] + dz_new[k - 1];
    }
    ppm_edges_d(nz, cT, qLT, qRT, q6T);
    ppm_edges_d(nz, cS, qLS, qRS, q6S);

    // ---- one overlap sweep, both tracers ----
    int ko_start = 1;
    for (int k = 1; k <= nz; ++k) {
        int o = IDX3(i, j, k - 1, nx, ny);
        if (dz_new[k - 1] <= 0.0) {
            real_t a = 0.0 * dz_new[k - 1], b = 0.0 * dz_new[k - 1];
            if (bud_t) bud_t[o] += (a - hTr_t[o]);
            if (bud_s) bud_s[o] += (b - hTr_s[o]);
            hTr_t[o] = a; hTr_s[o] = b;
            continue;
        }
        real_t iT = 0.0, iS = 0.0;
        for (int ko = ko_start; ko <= nz; ++ko) {
            real_t z_lo = fmax(z_new[k - 1], z_old[ko - 1]);
            real_t z_hi = fmin(z_new[k], z_old[ko]);
            real_t overlap = z_hi - z_lo;
            if (overlap <= 0.0) { if (z_old[ko] > z_new[k]) break; else continue; }
            if (dz_old[ko - 1] > 0.0) {
                real_t xi_lo = (z_lo - z_old[ko - 1]) / dz_old[ko - 1];
                real_t xi_hi = (z_hi - z_old[ko - 1]) / dz_old[ko - 1];
                real_t d1 = xi_hi - xi_lo;
                real_t d2 = 0.5 * (xi_hi * xi_hi - xi_lo * xi_lo);
                real_t d3 = xi_hi * xi_hi * xi_hi - xi_lo * xi_lo * xi_lo;
                iT += dz_old[ko - 1] * (d1 * qLT[ko - 1] + d2 * (qRT[ko - 1] - qLT[ko - 1] + q6T[ko - 1]) - d3 * q6T[ko - 1] / 3.0);
                iS += dz_old[ko - 1] * (d1 * qLS[ko - 1] + d2 * (qRS[ko - 1] - qLS[ko - 1] + q6S[ko - 1]) - d3 * q6S[ko - 1] / 3.0);
            } else {
                iT += cT[ko - 1] * overlap;
                iS += cS[ko - 1] * overlap;
            }
            if (z_old[ko] <= z_new[k]) ko_start = ko;
        }
        real_t a = (iT / dz_new[k - 1]) * dz_new[k - 1];
        real_t b = (iS / dz_new[k - 1]) * dz_new[k - 1];
        if (bud_t) bud_t[o] += (a - hTr_t[o]);
        if (bud_s) bud_s[o] += (b - hTr_s[o]);
        hTr_t[o] = a; hTr_s[o] = b;
    }
}

// ---- x-face momentum remap (32-bit indexing; arithmetic verbatim) ----------
__global__ void k_opt_xface(int nx, int ny, int nz, const real_t * __restrict__ h_old,
                            const real_t * __restrict__ h_new, real_t * __restrict__ u) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (nx + 1) * ny) return;
    int I = t % (nx + 1), j = t / (nx + 1);
    real_t h_old_face[NZ_STACK_MAX], h_new_face[NZ_STACK_MAX];
    real_t u_old[NZ_STACK_MAX], u_new[NZ_STACK_MAX];
    for (int k = 0; k < nz; ++k) {
        if (I == 0) {
            h_old_face[k] = h_old[IDX3(0, j, k, nx, ny)];
            h_new_face[k] = h_new[IDX3(0, j, k, nx, ny)];
        } else if (I == nx) {
            h_old_face[k] = h_old[IDX3(nx - 1, j, k, nx, ny)];
            h_new_face[k] = h_new[IDX3(nx - 1, j, k, nx, ny)];
        } else {
            h_old_face[k] = 0.5 * (h_old[IDX3(I - 1, j, k, nx, ny)] + h_old[IDX3(I, j, k, nx, ny)]);
            h_new_face[k] = 0.5 * (h_new[IDX3(I - 1, j, k, nx, ny)] + h_new[IDX3(I, j, k, nx, ny)]);
        }
        u_old[k] = u[IDX3(I, j, k, nx + 1, ny)];
    }
    remap_column_ppm_d(nz, h_old_face, h_new_face, u_old, u_new);
    for (int k = 0; k < nz; ++k) u[IDX3(I, j, k, nx + 1, ny)] = u_new[k];
}

__global__ void k_opt_yface(int nx, int ny, int nz, const real_t * __restrict__ h_old,
                            const real_t * __restrict__ h_new, real_t * __restrict__ v) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx * (ny + 1)) return;
    int i = t % nx, J = t / nx;
    real_t h_old_face[NZ_STACK_MAX], h_new_face[NZ_STACK_MAX];
    real_t v_old[NZ_STACK_MAX], v_new[NZ_STACK_MAX];
    for (int k = 0; k < nz; ++k) {
        if (J == 0) {
            h_old_face[k] = h_old[IDX3(i, 0, k, nx, ny)];
            h_new_face[k] = h_new[IDX3(i, 0, k, nx, ny)];
        } else if (J == ny) {
            h_old_face[k] = h_old[IDX3(i, ny - 1, k, nx, ny)];
            h_new_face[k] = h_new[IDX3(i, ny - 1, k, nx, ny)];
        } else {
            h_old_face[k] = 0.5 * (h_old[IDX3(i, J - 1, k, nx, ny)] + h_old[IDX3(i, J, k, nx, ny)]);
            h_new_face[k] = 0.5 * (h_new[IDX3(i, J - 1, k, nx, ny)] + h_new[IDX3(i, J, k, nx, ny)]);
        }
        v_old[k] = v[IDX3(i, J, k, nx, ny + 1)];
    }
    remap_column_ppm_d(nz, h_old_face, h_new_face, v_old, v_new);
    for (int k = 0; k < nz; ++k) v[IDX3(i, J, k, nx, ny + 1)] = v_new[k];
}

// ---- POST pass: mass budget + h_layer commit + bt_eta, ONE per-column kernel.
__global__ void k_opt_post(int nx, int ny, int nz, const real_t * __restrict__ target_h,
                           const real_t * __restrict__ h_old, real_t * __restrict__ mass_b,
                           real_t * __restrict__ h_layer, const real_t * __restrict__ H_ref,
                           real_t * __restrict__ eta) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx * ny) return;
    int i = t % nx, j = t / nx;
    real_t s = -H_ref[IDX2(i, j, nx)];
    for (int k = 0; k < nz; ++k) {
        int o = IDX3(i, j, k, nx, ny);
        real_t th = target_h[o];
        mass_b[o] += (th - h_old[o]);
        h_layer[o] = th;
        s += th;
    }
    eta[IDX2(i, j, nx)] = s;
}

// ---------------------------------------------------------------------------
extern "C" void ale_remap_opt(int nx, int ny, int nz,
                              real_t *h_layer, real_t *h_old, real_t *target_h,
                              real_t *hTr_t, real_t *hTr_s, real_t *u, real_t *v,
                              real_t *mass_b, real_t *heat_b, real_t *salt_b,
                              real_t *eta, real_t *H_ref, real_t *dsig) {
    int n2 = nx * ny;
    int nxf = (nx + 1) * ny, nyf = nx * (ny + 1);
    auto G = [&](int n) { return (n + TPB - 1) / TPB; };

    k_opt_pre<<<G(n2), TPB>>>(nx, ny, nz, h_layer, h_old, target_h, eta, dsig);
    k_opt_tracer2<<<G(n2), TPB>>>(nx, ny, nz, h_old, target_h, hTr_t, hTr_s, heat_b, salt_b);
    k_opt_xface<<<G(nxf), TPB>>>(nx, ny, nz, h_old, target_h, u);
    k_opt_yface<<<G(nyf), TPB>>>(nx, ny, nz, h_old, target_h, v);
    k_opt_post<<<G(n2), TPB>>>(nx, ny, nz, target_h, h_old, mass_b, h_layer, H_ref, eta);
}
