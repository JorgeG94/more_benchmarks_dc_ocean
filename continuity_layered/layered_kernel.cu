// Faithful CUDA C transliteration of the LAYERED continuity_compute_fluxes.
//
// PROVENANCE: hand-ported from continuity_layered.F90 next to this file,
// itself a verbatim extract of <model> @ HEAD,
// src/core/ocean/kernels/structured/continuity_ppm/continuity.F90:612-775.
// If the Fortran changes, this must change too -- a silent divergence would
// make the benchmark compare two different algorithms and report the
// difference as a compiler result.
//
// FAITHFULNESS RULES (the point is to measure the COMPILER, not my cleverness):
//   * Same operation order, parenthesisation, temporaries. Fortran pins
//     evaluation order inside parens; reassociating moves the last bits and
//     makes "do they agree?" unanswerable.
//   * Same branch structure, same helper decomposition.
//   * One thread per (i,j,k), matching each `do concurrent(k,j,i)` space --
//     nvfortran auto-collapses all five 3-D loops to collapse(3), so this
//     matches its iteration space exactly.
//   * Eleven kernels, one per Fortran loop, same order. No tiling, no __ldg,
//     no launch_bounds, no fast-math.
//
// INDEXING: the device arrays ARE the Fortran allocations (host_data
// use_device). Fortran is column-major, 1-based. THREE stride classes -- the
// face arrays carry one extra element in their own direction, and mixing them
// up is silent corruption, not a compile error:
//     h_layer, flux_h_layer           (nx,   ny,   nz)  -> IDX_T
//     u_face_x_layer, mass_flux_x_layer, h_face_*_x%data (nx+1, ny,   nz) -> IDX_U
//     v_face_y_layer, mass_flux_y_layer, h_face_*_y%data (nx,   ny+1, nz) -> IDX_V
// The metrics stay 2-D: wet_T/iareaT (nx,ny), dy_cu (nx+1,ny), dx_cv (nx,ny+1).

#include <cstdio>
#include "gpu_rt.h"

#define IDX_T(i, j, k) ((size_t)((i)-1) + (size_t)(nx)*((size_t)((j)-1) + (size_t)(ny)*(size_t)((k)-1)))
#define IDX_U(i, j, k) ((size_t)((i)-1) + (size_t)(nx+1)*((size_t)((j)-1) + (size_t)(ny)*(size_t)((k)-1)))
#define IDX_V(i, j, k) ((size_t)((i)-1) + (size_t)(nx)*((size_t)((j)-1) + (size_t)(ny+1)*(size_t)((k)-1)))
#define M_T(i, j) ((size_t)((i)-1) + (size_t)(nx)*(size_t)((j)-1))
#define M_U(i, j) ((size_t)((i)-1) + (size_t)(nx+1)*(size_t)((j)-1))
#define M_V(i, j) ((size_t)((i)-1) + (size_t)(nx)*(size_t)((j)-1))

__device__ __forceinline__ double ppm_mirror_h(double h_nbr, double h_loc, double w_nbr) {
    return w_nbr * h_nbr + (1.0 - w_nbr) * h_loc;
}
__device__ __forceinline__ void ppm_limited_slope(double h_im1, double h_i, double h_ip1, double *dh) {
    double dh_centered, dh_left, dh_right;
    dh_left = h_i - h_im1;
    dh_right = h_ip1 - h_i;
    dh_centered = 0.5 * (dh_left + dh_right);
    if (dh_left * dh_right > 0.0)
        *dh = copysign(fmin(fabs(dh_centered), fmin(2.0*fabs(dh_left), 2.0*fabs(dh_right))), dh_centered);
    else
        *dh = 0.0;
}
__device__ __forceinline__ void ppm_cell_limiter(double h_centre, double *h_left, double *h_right) {
    double dh_lr = *h_right - *h_left;
    double h_six = 6.0 * (h_centre - 0.5 * (*h_left + *h_right));
    if ((*h_right - h_centre) * (h_centre - *h_left) <= 0.0) { *h_left = h_centre; *h_right = h_centre; }
    else if (dh_lr * h_six >  dh_lr * dh_lr) *h_left  = 3.0*h_centre - 2.0*(*h_right);
    else if (dh_lr * h_six < -dh_lr * dh_lr) *h_right = 3.0*h_centre - 2.0*(*h_left);
}
__device__ __forceinline__ void ppm_limit_pos(double h_centre, double *h_left, double *h_right, double h_min) {
    double curv = 3.0 * ((*h_left + *h_right) - 2.0 * h_centre);
    if (curv > 0.0) {
        double dh = *h_right - *h_left;
        if (fabs(dh) < curv) {
            if (h_centre <= h_min) { *h_left = h_centre; *h_right = h_centre; }
            else if (12.0*curv*(h_centre-h_min) < (curv*curv + 3.0*dh*dh)) {
                double s = 12.0*curv*(h_centre-h_min)/(curv*curv + 3.0*dh*dh);
                *h_left  = h_centre + s*(*h_left  - h_centre);
                *h_right = h_centre + s*(*h_right - h_centre);
            }
        }
    }
}

// ---- 1. X PPM (F90:128) : do concurrent(k=1:nz, j=1:ny, i=3:nx-2) ----
__global__ void kL_ppm_x(const double * __restrict__ h, const double * __restrict__ wet_T, double * __restrict__ hfl_x, double * __restrict__ hfr_x,
                         int nx, int ny, int nz, int do_pos, double h_min_pos) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t ni = (size_t)(nx-4);
    if (t >= ni*(size_t)ny*(size_t)nz) return;
    int i = (int)(t % ni) + 3;
    int j = (int)((t/ni) % (size_t)ny) + 1;
    int k = (int)(t/(ni*(size_t)ny)) + 1;
    double dh_m1, dh_0, dh_p1, h_left, h_right, hm2, hm1, h0, hp1, hp2;
    h0  = h[IDX_T(i,j,k)];
    hm1 = ppm_mirror_h(h[IDX_T(i-1,j,k)], h0,  wet_T[M_T(i-1,j)]);
    hp1 = ppm_mirror_h(h[IDX_T(i+1,j,k)], h0,  wet_T[M_T(i+1,j)]);
    hm2 = ppm_mirror_h(h[IDX_T(i-2,j,k)], hm1, wet_T[M_T(i-2,j)]);
    hp2 = ppm_mirror_h(h[IDX_T(i+2,j,k)], hp1, wet_T[M_T(i+2,j)]);
    ppm_limited_slope(hm2, hm1, h0, &dh_m1);
    ppm_limited_slope(hm1, h0, hp1, &dh_0);
    ppm_limited_slope(h0, hp1, hp2, &dh_p1);
    dh_0 = dh_0 * wet_T[M_T(i-1,j)] * wet_T[M_T(i,j)] * wet_T[M_T(i+1,j)];
    h_left  = 0.5*(hm1 + h0) - (dh_0 - dh_m1)/6.0;
    h_right = 0.5*(h0 + hp1) - (dh_p1 - dh_0)/6.0;
    ppm_cell_limiter(h0, &h_left, &h_right);
    if (do_pos) ppm_limit_pos(h0, &h_left, &h_right, h_min_pos);
    hfr_x[IDX_U(i,j,k)]   = h_left;
    hfl_x[IDX_U(i+1,j,k)] = h_right;
}
// ---- 2. X boundary (F90:148) ----
__global__ void kL_bnd_x(const double * __restrict__ h, double * __restrict__ hfl_x, double * __restrict__ hfr_x, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)ny*(size_t)nz) return;
    int j = (int)(t % (size_t)ny) + 1;
    int k = (int)(t / (size_t)ny) + 1;
    hfl_x[IDX_U(1,j,k)] = h[IDX_T(1,j,k)];
    hfr_x[IDX_U(1,j,k)] = h[IDX_T(1,j,k)];
    hfl_x[IDX_U(2,j,k)] = h[IDX_T(1,j,k)];
    hfr_x[IDX_U(2,j,k)] = h[IDX_T(2,j,k)];
    hfl_x[IDX_U(3,j,k)] = h[IDX_T(2,j,k)];
    hfr_x[IDX_U(3,j,k)] = h[IDX_T(2,j,k)];
    hfr_x[IDX_U(nx-1,j,k)] = h[IDX_T(nx-1,j,k)];
    hfl_x[IDX_U(nx,j,k)]   = h[IDX_T(nx-1,j,k)];
    hfr_x[IDX_U(nx,j,k)]   = h[IDX_T(nx,j,k)];
    hfl_x[IDX_U(nx+1,j,k)] = h[IDX_T(nx,j,k)];
    hfr_x[IDX_U(nx+1,j,k)] = h[IDX_T(nx,j,k)];
}
// ---- 3. X transport (F90:169) ----
__global__ void kL_trans_x(const double * __restrict__ u, const double * __restrict__ hfl_x, const double * __restrict__ hfr_x,
                           const double * __restrict__ dy_cu, double * __restrict__ mfx, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t ni = (size_t)(nx-1);
    if (t >= ni*(size_t)ny*(size_t)nz) return;
    int i = (int)(t % ni) + 2;
    int j = (int)((t/ni) % (size_t)ny) + 1;
    int k = (int)(t/(ni*(size_t)ny)) + 1;
    double uu = u[IDX_U(i,j,k)], h_face;
    if (uu >= 0.0) h_face = hfl_x[IDX_U(i,j,k)];
    else           h_face = hfr_x[IDX_U(i,j,k)];
    mfx[IDX_U(i,j,k)] = uu * h_face * dy_cu[M_U(i,j)];
}
// ---- 4. X array-edge walls (F90:178) ----
__global__ void kL_wall_x(double * __restrict__ mfx, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)ny*(size_t)nz) return;
    int j = (int)(t % (size_t)ny) + 1;
    int k = (int)(t / (size_t)ny) + 1;
    mfx[IDX_U(1,j,k)]    = 0.0;
    mfx[IDX_U(nx+1,j,k)] = 0.0;
}
// ---- 5. X PHYSICAL walls (F90:185) — at nghost+1 / nghost+nx_phys+1 ----
__global__ void kL_pwall_x(double * __restrict__ mfx, int nx, int ny, int nz, int nghost, int nx_phys) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)ny*(size_t)nz) return;
    int j = (int)(t % (size_t)ny) + 1;
    int k = (int)(t / (size_t)ny) + 1;
    mfx[IDX_U(nghost+1, j, k)]           = 0.0;
    mfx[IDX_U(nghost+nx_phys+1, j, k)]   = 0.0;
}
// ---- 6. Y PPM (F90:193) : do concurrent(k=1:nz, j=3:ny-2, i=1:nx) ----
__global__ void kL_ppm_y(const double * __restrict__ h, const double * __restrict__ wet_T, double * __restrict__ hfl_y, double * __restrict__ hfr_y,
                         int nx, int ny, int nz, int do_pos, double h_min_pos) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t nj = (size_t)(ny-4);
    if (t >= (size_t)nx*nj*(size_t)nz) return;
    int i = (int)(t % (size_t)nx) + 1;
    int j = (int)((t/(size_t)nx) % nj) + 3;
    int k = (int)(t/((size_t)nx*nj)) + 1;
    double dh_m1, dh_0, dh_p1, h_left, h_right, hm2, hm1, h0, hp1, hp2;
    h0  = h[IDX_T(i,j,k)];
    hm1 = ppm_mirror_h(h[IDX_T(i,j-1,k)], h0,  wet_T[M_T(i,j-1)]);
    hp1 = ppm_mirror_h(h[IDX_T(i,j+1,k)], h0,  wet_T[M_T(i,j+1)]);
    hm2 = ppm_mirror_h(h[IDX_T(i,j-2,k)], hm1, wet_T[M_T(i,j-2)]);
    hp2 = ppm_mirror_h(h[IDX_T(i,j+2,k)], hp1, wet_T[M_T(i,j+2)]);
    ppm_limited_slope(hm2, hm1, h0, &dh_m1);
    ppm_limited_slope(hm1, h0, hp1, &dh_0);
    ppm_limited_slope(h0, hp1, hp2, &dh_p1);
    dh_0 = dh_0 * wet_T[M_T(i,j-1)] * wet_T[M_T(i,j)] * wet_T[M_T(i,j+1)];
    h_left  = 0.5*(hm1 + h0) - (dh_0 - dh_m1)/6.0;
    h_right = 0.5*(h0 + hp1) - (dh_p1 - dh_0)/6.0;
    ppm_cell_limiter(h0, &h_left, &h_right);
    if (do_pos) ppm_limit_pos(h0, &h_left, &h_right, h_min_pos);
    hfr_y[IDX_V(i,j,k)]   = h_left;
    hfl_y[IDX_V(i,j+1,k)] = h_right;
}
// ---- 7. Y boundary (F90:211) ----
__global__ void kL_bnd_y(const double * __restrict__ h, double * __restrict__ hfl_y, double * __restrict__ hfr_y, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)nx*(size_t)nz) return;
    int i = (int)(t % (size_t)nx) + 1;
    int k = (int)(t / (size_t)nx) + 1;
    hfl_y[IDX_V(i,1,k)] = h[IDX_T(i,1,k)];
    hfr_y[IDX_V(i,1,k)] = h[IDX_T(i,1,k)];
    hfl_y[IDX_V(i,2,k)] = h[IDX_T(i,1,k)];
    hfr_y[IDX_V(i,2,k)] = h[IDX_T(i,2,k)];
    hfl_y[IDX_V(i,3,k)] = h[IDX_T(i,2,k)];
    hfr_y[IDX_V(i,3,k)] = h[IDX_T(i,2,k)];
    hfr_y[IDX_V(i,ny-1,k)] = h[IDX_T(i,ny-1,k)];
    hfl_y[IDX_V(i,ny,k)]   = h[IDX_T(i,ny-1,k)];
    hfr_y[IDX_V(i,ny,k)]   = h[IDX_T(i,ny,k)];
    hfl_y[IDX_V(i,ny+1,k)] = h[IDX_T(i,ny,k)];
    hfr_y[IDX_V(i,ny+1,k)] = h[IDX_T(i,ny,k)];
}
// ---- 8. Y transport (F90:225) ----
__global__ void kL_trans_y(const double * __restrict__ v, const double * __restrict__ hfl_y, const double * __restrict__ hfr_y,
                           const double * __restrict__ dx_cv, double * __restrict__ mfy, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t nj = (size_t)(ny-1);
    if (t >= (size_t)nx*nj*(size_t)nz) return;
    int i = (int)(t % (size_t)nx) + 1;
    int j = (int)((t/(size_t)nx) % nj) + 2;
    int k = (int)(t/((size_t)nx*nj)) + 1;
    double vv = v[IDX_V(i,j,k)], h_face;
    if (vv >= 0.0) h_face = hfl_y[IDX_V(i,j,k)];
    else           h_face = hfr_y[IDX_V(i,j,k)];
    mfy[IDX_V(i,j,k)] = vv * h_face * dx_cv[M_V(i,j)];
}
// ---- 9. Y array-edge walls (F90:234) ----
__global__ void kL_wall_y(double * __restrict__ mfy, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)nx*(size_t)nz) return;
    int i = (int)(t % (size_t)nx) + 1;
    int k = (int)(t / (size_t)nx) + 1;
    mfy[IDX_V(i,1,k)]    = 0.0;
    mfy[IDX_V(i,ny+1,k)] = 0.0;
}
// ---- 10. Y PHYSICAL walls (F90:240) ----
__global__ void kL_pwall_y(double * __restrict__ mfy, int nx, int ny, int nz, int nghost, int ny_phys) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)nx*(size_t)nz) return;
    int i = (int)(t % (size_t)nx) + 1;
    int k = (int)(t / (size_t)nx) + 1;
    mfy[IDX_V(i, nghost+1, k)]         = 0.0;
    mfy[IDX_V(i, nghost+ny_phys+1, k)] = 0.0;
}
// ---- 11. Divergence (F90:248) ----
__global__ void kL_div(const double * __restrict__ mfx, const double * __restrict__ mfy, const double * __restrict__ iareaT,
                       double * __restrict__ fh, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)nx*(size_t)ny*(size_t)nz) return;
    int i = (int)(t % (size_t)nx) + 1;
    int j = (int)((t/(size_t)nx) % (size_t)ny) + 1;
    int k = (int)(t/((size_t)nx*(size_t)ny)) + 1;
    fh[IDX_T(i,j,k)] = ((mfx[IDX_U(i+1,j,k)] - mfx[IDX_U(i,j,k)]) +
                        (mfy[IDX_V(i,j+1,k)] - mfy[IDX_V(i,j,k)])) * iareaT[M_T(i,j)];
}

// ===================== launcher =========================================
// 128 threads/block to match nvfortran's choice for these loops (-Minfo says
// `CUDA threads(128)`), so this is not a launch-geometry artifact.
//
// `sync` selects the SYNCHRONISATION MODEL -- see ../continuity_ppm_benchmark:
//   0 = async (queue all 11, return), 1 = sync once per call,
//   2 = sync after EVERY kernel (DEFAULT: matches what `do concurrent` does,
//       so the ratio reflects CODEGEN rather than the launch model).
#define TPB 128
#define NBLK(n) (int)(((size_t)(n) + TPB - 1) / TPB)
static inline void maybe_sync(int sync, int level) {
    if (sync >= level) {
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
    }
}

extern "C" void continuity_layered_cuda_launch(
    const double *h, const double *u, const double *v, const double *wet_T,
    const double *dy_cu, const double *dx_cv, const double *iareaT,
    double *hfl_x, double *hfr_x, double *hfl_y, double *hfr_y,
    double *mfx, double *mfy, double *fh,
    int nx, int ny, int nz, int nghost, int nx_phys, int ny_phys,
    int do_pos, double h_min_pos, int sync) {
    kL_ppm_x  <<<NBLK((size_t)(nx-4)*ny*nz), TPB>>>(h, wet_T, hfl_x, hfr_x, nx, ny, nz, do_pos, h_min_pos);
    maybe_sync(sync, 2);
    kL_bnd_x  <<<NBLK((size_t)ny*nz), TPB>>>(h, hfl_x, hfr_x, nx, ny, nz);
    maybe_sync(sync, 2);
    kL_trans_x<<<NBLK((size_t)(nx-1)*ny*nz), TPB>>>(u, hfl_x, hfr_x, dy_cu, mfx, nx, ny, nz);
    maybe_sync(sync, 2);
    kL_wall_x <<<NBLK((size_t)ny*nz), TPB>>>(mfx, nx, ny, nz);
    maybe_sync(sync, 2);
    kL_pwall_x<<<NBLK((size_t)ny*nz), TPB>>>(mfx, nx, ny, nz, nghost, nx_phys);
    maybe_sync(sync, 2);
    kL_ppm_y  <<<NBLK((size_t)nx*(ny-4)*nz), TPB>>>(h, wet_T, hfl_y, hfr_y, nx, ny, nz, do_pos, h_min_pos);
    maybe_sync(sync, 2);
    kL_bnd_y  <<<NBLK((size_t)nx*nz), TPB>>>(h, hfl_y, hfr_y, nx, ny, nz);
    maybe_sync(sync, 2);
    kL_trans_y<<<NBLK((size_t)nx*(ny-1)*nz), TPB>>>(v, hfl_y, hfr_y, dx_cv, mfy, nx, ny, nz);
    maybe_sync(sync, 2);
    kL_wall_y <<<NBLK((size_t)nx*nz), TPB>>>(mfy, nx, ny, nz);
    maybe_sync(sync, 2);
    kL_pwall_y<<<NBLK((size_t)nx*nz), TPB>>>(mfy, nx, ny, nz, nghost, ny_phys);
    maybe_sync(sync, 2);
    kL_div    <<<NBLK((size_t)nx*ny*nz), TPB>>>(mfx, mfy, iareaT, fh, nx, ny, nz);
    maybe_sync(sync, 1);
}
