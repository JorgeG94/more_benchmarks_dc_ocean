// Hand-written CUDA C port of the ocean model's ALE remap (zstar, PPM, 2 tracers).
//
// Faithful port: same one-thread-per-column mapping, same per-thread column
// arrays, same arithmetic in the same order as
//   <model>/src/ALE/kernel_remap.F90::remap_column_ppm
//   <model>/src/ALE/ocean_remap.F90::ocean_apply_ale_remap_step
//
// Two modes:
//   fused=0  "faithful"  -- one kernel per production do-concurrent loop.
//   fused=1  "fused"     -- the cheap loops folded together, and (the real
//                          lever) T and S remapped in ONE kernel that builds
//                          the column geometry (z_old/z_new + the overlap
//                          sweep) once and reuses it for both tracers.
//
// Fortran arrays are column-major: a(i,j,k) -> a[(i-1) + (j-1)*nx + (k-1)*nx*ny].
// Indices below are 0-based; `IDX3` mirrors the Fortran layout exactly.
//
// PPM_DIRECT: like the Fortran `PPM_DIRECT` variant, this calls the PPM
// kernel directly rather than dispatching through a 5-way select case inside
// the loop body. A CUDA port would never write the select-case form, so the
// honest comparison for "CUDA vs best Fortran" is against the Fortran variant
// that also hoists the dispatch.

#include "gpu_rt.h"
#include <cstdio>

#ifndef NZ_STACK_MAX
#define NZ_STACK_MAX 128
#endif

#define H_FLOOR 1.5e-4
#define IDX2(i, j, nx) ((i) + (j) * (nx))
#define IDX3(i, j, k, nx, ny) ((i) + (j) * (nx) + (k) * (nx) * (ny))

typedef double real_t;

extern "C" void cuda_sync() { cudaDeviceSynchronize(); }

// ---------------------------------------------------------------------------
// remap_column_ppm -- faithful port of kernel_remap.F90:213-375.
// Operation order preserved so results match the Fortran bit-for-bit modulo
// FMA contraction (nvcc and nvfortran may contract differently; see README).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void remap_column_plm_d(int nz, const real_t *dz_old,
                                                   const real_t *dz_new,
                                                   const real_t *q_old, real_t *q_new) {
    // Faithful port of kernel_remap.F90:124-211.
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

// ---------------------------------------------------------------------------
// FUSED two-tracer PPM: geometry (z_old/z_new + the overlap sweep) built ONCE
// and reused for both tracers. Per-tracer arithmetic is unchanged and stays in
// the same order, so each tracer's result is bit-identical to the 1-tracer
// kernel. This is the one real algorithmic saving available here -- T and S
// share h_old/h_new exactly.
// ---------------------------------------------------------------------------
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
// Kernels
// ---------------------------------------------------------------------------
__global__ void k_total_h(int nx, int ny, int nz, const real_t * __restrict__ h_layer, real_t * __restrict__ total_h) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx * ny) return;
    int i = t % nx, j = t / nx;
    real_t s = 0.0;
    for (int k = 0; k < nz; ++k) s += h_layer[IDX3(i, j, k, nx, ny)];
    total_h[IDX2(i, j, nx)] = s;
}

__global__ void k_h_ref(int nx, int ny, const real_t * __restrict__ total_h, const real_t * __restrict__ eta, real_t * __restrict__ h_ref) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx * ny) return;
    h_ref[t] = total_h[t] - eta[t];
}

__global__ void k_h_old(int nx, int ny, int nz, const real_t * __restrict__ h_layer, real_t * __restrict__ h_old) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx * ny * nz) return;
    h_old[t] = h_layer[t];
}

__global__ void k_target_h(int nx, int ny, int nz, const real_t * __restrict__ h_ref,
                           const real_t * __restrict__ eta, const real_t * __restrict__ dsig, real_t * __restrict__ target_h) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx * ny * nz) return;
    int ij = t % (nx * ny), k = t / (nx * ny);
    target_h[t] = (h_ref[ij] + eta[ij]) * dsig[k];
}

__global__ void k_tracer(int nx, int ny, int nz, const real_t * __restrict__ h_old, const real_t * __restrict__ h_new,
                         real_t * __restrict__ hTr, real_t * __restrict__ budget) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx * ny) return;
    int i = t % nx, j = t / nx;
    real_t h_old_col[NZ_STACK_MAX], h_new_col[NZ_STACK_MAX];
    real_t c_old_col[NZ_STACK_MAX], c_new_col[NZ_STACK_MAX];
    for (int k = 0; k < nz; ++k) {
        int o = IDX3(i, j, k, nx, ny);
        h_old_col[k] = h_old[o];
        h_new_col[k] = h_new[o];
        real_t hc = hTr[o];
        c_old_col[k] = (h_old_col[k] > H_FLOOR) ? hc / h_old_col[k] : 0.0;
    }
    remap_column_ppm_d(nz, h_old_col, h_new_col, c_old_col, c_new_col);
    for (int k = 0; k < nz; ++k) {
        int o = IDX3(i, j, k, nx, ny);
        real_t hTr_new = c_new_col[k] * h_new_col[k];
        if (budget) budget[o] += (hTr_new - hTr[o]);
        hTr[o] = hTr_new;
    }
}

// Fused T+S: shared geometry, both tracers in one pass.
__global__ void k_tracer2(int nx, int ny, int nz, const real_t * __restrict__ h_old, const real_t * __restrict__ h_new,
                          real_t * __restrict__ hTr_t, real_t * __restrict__ hTr_s, real_t * __restrict__ bud_t, real_t * __restrict__ bud_s) {
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
    if (nz < 3) {  // fall back: PPM degenerates below nz=3
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
            // Fortran sets q_new(k)=0 then hTr_new = q_new*h_new_col(k); keep
            // the multiply so a negative dz_new yields -0.0 exactly as it does.
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
                // d3 = RAW cube difference; the `*q6/3` keeps production's
                // association order (pre-dividing re-rounds, costs ~1 ulp).
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

__global__ void k_xface(int nx, int ny, int nz, const real_t * __restrict__ h_old, const real_t * __restrict__ h_new, real_t * __restrict__ u) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (nx + 1) * ny) return;
    int I = t % (nx + 1), j = t / (nx + 1);   // I is 0-based; Fortran I = I+1
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

__global__ void k_yface(int nx, int ny, int nz, const real_t * __restrict__ h_old, const real_t * __restrict__ h_new, real_t * __restrict__ v) {
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

__global__ void k_mass_h(int nx, int ny, int nz, const real_t * __restrict__ target_h, const real_t * __restrict__ h_old,
                         real_t * __restrict__ mass_b, real_t * __restrict__ h_layer) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx * ny * nz) return;
    mass_b[t] += (target_h[t] - h_old[t]);
    h_layer[t] = target_h[t];
}

__global__ void k_bt_eta(int nx, int ny, int nz, const real_t * __restrict__ h_layer, const real_t * __restrict__ H_ref, real_t * __restrict__ eta) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx * ny) return;
    int i = t % nx, j = t / nx;
    real_t s = -H_ref[IDX2(i, j, nx)];
    for (int k = 0; k < nz; ++k) s += h_layer[IDX3(i, j, k, nx, ny)];
    eta[IDX2(i, j, nx)] = s;
}

// Fused cheap-loop kernels -------------------------------------------------
// G1: total_h + h_ref + h_old snapshot + target_h, one 2-D thread per column.
// L1->L2->L4 are same-index chains and L3 rides the L1 k-sweep, so one thread
// per column can do all four with no cross-thread dependence.
__global__ void k_pre_fused(int nx, int ny, int nz, real_t * __restrict__ h_layer, real_t * __restrict__ h_old,
                            real_t * __restrict__ target_h, real_t * __restrict__ total_h, real_t * __restrict__ h_ref,
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
    total_h[IDX2(i, j, nx)] = s;
    real_t hr = s - eta[IDX2(i, j, nx)];
    h_ref[IDX2(i, j, nx)] = hr;
    real_t ct = hr + eta[IDX2(i, j, nx)];
    for (int k = 0; k < nz; ++k) target_h[IDX3(i, j, k, nx, ny)] = ct * dsig[k];
}

// G5: mass budget + h_layer + bt_eta.
__global__ void k_post_fused(int nx, int ny, int nz, const real_t * __restrict__ target_h, const real_t * __restrict__ h_old,
                             real_t * __restrict__ mass_b, real_t * __restrict__ h_layer, const real_t * __restrict__ H_ref, real_t * __restrict__ eta) {
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
extern "C" void ale_remap_cuda(int nx, int ny, int nz, int method, int fused,
                               real_t *h_layer, real_t *h_old, real_t *target_h,
                               real_t *total_h, real_t *h_ref,
                               real_t *hTr_t, real_t *hTr_s, real_t *u, real_t *v,
                               real_t *mass_b, real_t *heat_b, real_t *salt_b,
                               real_t *eta, real_t *H_ref, real_t *dsig) {
    const int TPB = 128;
    int n2 = nx * ny, n3 = nx * ny * nz;
    int nxf = (nx + 1) * ny, nyf = nx * (ny + 1);
    auto G = [&](int n) { return (n + TPB - 1) / TPB; };

    if (!fused) {
        k_total_h<<<G(n2), TPB>>>(nx, ny, nz, h_layer, total_h);
        k_h_ref<<<G(n2), TPB>>>(nx, ny, total_h, eta, h_ref);
        k_h_old<<<G(n3), TPB>>>(nx, ny, nz, h_layer, h_old);
        k_target_h<<<G(n3), TPB>>>(nx, ny, nz, h_ref, eta, dsig, target_h);
        k_tracer<<<G(n2), TPB>>>(nx, ny, nz, h_old, target_h, hTr_t, heat_b);
        k_tracer<<<G(n2), TPB>>>(nx, ny, nz, h_old, target_h, hTr_s, salt_b);
        k_xface<<<G(nxf), TPB>>>(nx, ny, nz, h_old, target_h, u);
        k_yface<<<G(nyf), TPB>>>(nx, ny, nz, h_old, target_h, v);
        k_mass_h<<<G(n3), TPB>>>(nx, ny, nz, target_h, h_old, mass_b, h_layer);
        k_bt_eta<<<G(n2), TPB>>>(nx, ny, nz, h_layer, H_ref, eta);
    } else {
        k_pre_fused<<<G(n2), TPB>>>(nx, ny, nz, h_layer, h_old, target_h, total_h, h_ref, eta, dsig);
        k_tracer2<<<G(n2), TPB>>>(nx, ny, nz, h_old, target_h, hTr_t, hTr_s, heat_b, salt_b);
        k_xface<<<G(nxf), TPB>>>(nx, ny, nz, h_old, target_h, u);
        k_yface<<<G(nyf), TPB>>>(nx, ny, nz, h_old, target_h, v);
        k_post_fused<<<G(n2), TPB>>>(nx, ny, nz, target_h, h_old, mass_b, h_layer, H_ref, eta);
    }
}
