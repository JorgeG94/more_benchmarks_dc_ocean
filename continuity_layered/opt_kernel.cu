// Optimized layered-continuity CUDA. Best-practice pass over layered_kernel.cu.
// Freed from the "faithful transliteration" rule -- fuse aggressively, but stay
// bit-identical to the original (same per-cell arithmetic order), which the A/B
// harness (ab_main.cu) checks as max|diff| == 0.
//
// RESULT (V100, 473x297x30): 0.93 -> 0.62 ms/rep, 1.50x, bit-identical.
// (1.40-1.70x across sizes; more on smaller grids where launches matter more.)
//
// The winner is the DEFAULT build (OPTVER=2, IDX32=1) -- two changes, both
// "remove genuine waste", not cleverness:
//   * fuse each direction's PPM + boundary + transport + walls into ONE kernel
//     that writes only mfx (resp. mfy); the hfl/hfr face arrays never reach
//     global memory. 11 kernels -> 3, ~280 MB of intermediate traffic gone.
//   * 32-bit indexing (the domain fits int32): fewer address regs, ~1.06x.
// kO_div is already ~84% of DRAM peak -- near the memory-bandwidth floor.
//
// Everything cleverer LOST (kept here as OPTVER 3-8 for the record, see
// OPTIMIZATION.md): full recompute-fusion, k-loop/k-block for ILP,
// launch_bounds, div+flux fusion, and shared-memory tiling all cost more
// (recompute / lost parallelism / smem+sync overhead) than they saved. This
// kernel is memory-bound and the flat one-thread-per-face layout lets the GPU's
// own L2 + occupancy do the reuse better than any hand-rolled scheme.
#include <cstdio>
#include "gpu_rt.h"

#ifndef TPB
#define TPB 128
#endif
#ifndef LB
#define LB 0
#endif
// 32-bit indexing (default): valid while nx*ny*nz < 2^31, i.e. every realistic
// ocean domain (4096^2 x 50 = 0.84e9 << 2.1e9). Worth ~1.06x -- set -DIDX32=0
// for the size_t-safe path on pathologically large grids.
#ifndef IDX32
#define IDX32 1
#endif
#if LB
#define LBOUNDS __launch_bounds__(TPB, LB)
#else
#define LBOUNDS
#endif

#if IDX32   // 32-bit indexing: valid while nx*ny*nz < 2^31 (fine at these sizes)
#define IDX_T(i, j, k) ((int)((i)-1) + (nx)*((int)((j)-1) + (ny)*((int)((k)-1))))
#define IDX_U(i, j, k) ((int)((i)-1) + (nx+1)*((int)((j)-1) + (ny)*((int)((k)-1))))
#define IDX_V(i, j, k) ((int)((i)-1) + (nx)*((int)((j)-1) + (ny+1)*((int)((k)-1))))
#define M_T(i, j) ((int)((i)-1) + (nx)*((int)((j)-1)))
#define M_U(i, j) ((int)((i)-1) + (nx+1)*((int)((j)-1)))
#define M_V(i, j) ((int)((i)-1) + (nx)*((int)((j)-1)))
#else
#define IDX_T(i, j, k) ((size_t)((i)-1) + (size_t)(nx)*((size_t)((j)-1) + (size_t)(ny)*(size_t)((k)-1)))
#define IDX_U(i, j, k) ((size_t)((i)-1) + (size_t)(nx+1)*((size_t)((j)-1) + (size_t)(ny)*(size_t)((k)-1)))
#define IDX_V(i, j, k) ((size_t)((i)-1) + (size_t)(nx)*((size_t)((j)-1) + (size_t)(ny+1)*(size_t)((k)-1)))
#define M_T(i, j) ((size_t)((i)-1) + (size_t)(nx)*(size_t)((j)-1))
#define M_U(i, j) ((size_t)((i)-1) + (size_t)(nx+1)*(size_t)((j)-1))
#define M_V(i, j) ((size_t)((i)-1) + (size_t)(nx)*(size_t)((j)-1))
#endif

__device__ __forceinline__ double o_mirror(double h_nbr, double h_loc, double w_nbr) {
    return w_nbr * h_nbr + (1.0 - w_nbr) * h_loc;
}
__device__ __forceinline__ void o_slope(double h_im1, double h_i, double h_ip1, double *dh) {
    double dl = h_i - h_im1, dr = h_ip1 - h_i, dc = 0.5 * (dl + dr);
    if (dl * dr > 0.0)
        *dh = copysign(fmin(fabs(dc), fmin(2.0*fabs(dl), 2.0*fabs(dr))), dc);
    else
        *dh = 0.0;
}
__device__ __forceinline__ void o_cell_lim(double hc, double *hl, double *hr) {
    double dlr = *hr - *hl, h6 = 6.0 * (hc - 0.5 * (*hl + *hr));
    if ((*hr - hc) * (hc - *hl) <= 0.0) { *hl = hc; *hr = hc; }
    else if (dlr * h6 >  dlr * dlr) *hl = 3.0*hc - 2.0*(*hr);
    else if (dlr * h6 < -dlr * dlr) *hr = 3.0*hc - 2.0*(*hl);
}
__device__ __forceinline__ void o_limpos(double hc, double *hl, double *hr, double hmin) {
    double curv = 3.0 * ((*hl + *hr) - 2.0 * hc);
    if (curv > 0.0) {
        double dh = *hr - *hl;
        if (fabs(dh) < curv) {
            if (hc <= hmin) { *hl = hc; *hr = hc; }
            else if (12.0*curv*(hc-hmin) < (curv*curv + 3.0*dh*dh)) {
                double s = 12.0*curv*(hc-hmin)/(curv*curv + 3.0*dh*dh);
                *hl = hc + s*(*hl - hc); *hr = hc + s*(*hr - hc);
            }
        }
    }
}

// PPM reconstruct cell c along x (needs h[c-2..c+2]); returns h_left,h_right.
__device__ __forceinline__ void recon_x(const double *__restrict__ h, const double *__restrict__ wet,
                                         int c, int j, int k, int nx, int ny, int nz,
                                         int do_pos, double hmin, double *hl, double *hr) {
    double h0 = h[IDX_T(c,j,k)];
    double hm1 = o_mirror(h[IDX_T(c-1,j,k)], h0,  wet[M_T(c-1,j)]);
    double hp1 = o_mirror(h[IDX_T(c+1,j,k)], h0,  wet[M_T(c+1,j)]);
    double hm2 = o_mirror(h[IDX_T(c-2,j,k)], hm1, wet[M_T(c-2,j)]);
    double hp2 = o_mirror(h[IDX_T(c+2,j,k)], hp1, wet[M_T(c+2,j)]);
    double dm1, d0, dp1;
    o_slope(hm2, hm1, h0, &dm1); o_slope(hm1, h0, hp1, &d0); o_slope(h0, hp1, hp2, &dp1);
    d0 = d0 * wet[M_T(c-1,j)] * wet[M_T(c,j)] * wet[M_T(c+1,j)];
    double L = 0.5*(hm1 + h0) - (d0 - dm1)/6.0;
    double R = 0.5*(h0 + hp1) - (dp1 - d0)/6.0;
    o_cell_lim(h0, &L, &R);
    if (do_pos) o_limpos(h0, &L, &R, hmin);
    *hl = L; *hr = R;
}
__device__ __forceinline__ void recon_y(const double *__restrict__ h, const double *__restrict__ wet,
                                         int i, int c, int k, int nx, int ny, int nz,
                                         int do_pos, double hmin, double *hl, double *hr) {
    double h0 = h[IDX_T(i,c,k)];
    double hm1 = o_mirror(h[IDX_T(i,c-1,k)], h0,  wet[M_T(i,c-1)]);
    double hp1 = o_mirror(h[IDX_T(i,c+1,k)], h0,  wet[M_T(i,c+1)]);
    double hm2 = o_mirror(h[IDX_T(i,c-2,k)], hm1, wet[M_T(i,c-2)]);
    double hp2 = o_mirror(h[IDX_T(i,c+2,k)], hp1, wet[M_T(i,c+2)]);
    double dm1, d0, dp1;
    o_slope(hm2, hm1, h0, &dm1); o_slope(hm1, h0, hp1, &d0); o_slope(h0, hp1, hp2, &dp1);
    d0 = d0 * wet[M_T(i,c-1)] * wet[M_T(i,c)] * wet[M_T(i,c+1)];
    double L = 0.5*(hm1 + h0) - (d0 - dm1)/6.0;
    double R = 0.5*(h0 + hp1) - (dp1 - d0)/6.0;
    o_cell_lim(h0, &L, &R);
    if (do_pos) o_limpos(h0, &L, &R, hmin);
    *hl = L; *hr = R;
}

// ---- per-face flux (shared by the fused and staged variants) --------------
__device__ __forceinline__ double flux_x(const double *__restrict__ h, const double *__restrict__ u,
        const double *__restrict__ wet, const double *__restrict__ dy_cu,
        int f, int j, int k, int nx, int ny, int nz, int nghost, int nx_phys, int do_pos, double hmin) {
    if (f == 1 || f == nx+1 || f == nghost+1 || f == nghost+nx_phys+1) return 0.0;
    double uu = u[IDX_U(f,j,k)], h_face, hl, hr;
    if (uu >= 0.0) {
        if      (f == 2)     h_face = h[IDX_T(1,j,k)];
        else if (f == 3)     h_face = h[IDX_T(2,j,k)];
        else if (f == nx)    h_face = h[IDX_T(nx-1,j,k)];
        else { recon_x(h, wet, f-1, j, k, nx, ny, nz, do_pos, hmin, &hl, &hr); h_face = hr; }
    } else {
        if      (f == 2)     h_face = h[IDX_T(2,j,k)];
        else if (f == 3)     h_face = h[IDX_T(2,j,k)];
        else if (f == nx-1)  h_face = h[IDX_T(nx-1,j,k)];
        else if (f == nx)    h_face = h[IDX_T(nx,j,k)];
        else { recon_x(h, wet, f,   j, k, nx, ny, nz, do_pos, hmin, &hl, &hr); h_face = hl; }
    }
    return uu * h_face * dy_cu[M_U(f,j)];
}
__device__ __forceinline__ double flux_y(const double *__restrict__ h, const double *__restrict__ v,
        const double *__restrict__ wet, const double *__restrict__ dx_cv,
        int i, int f, int k, int nx, int ny, int nz, int nghost, int ny_phys, int do_pos, double hmin) {
    if (f == 1 || f == ny+1 || f == nghost+1 || f == nghost+ny_phys+1) return 0.0;
    double vv = v[IDX_V(i,f,k)], h_face, hl, hr;
    if (vv >= 0.0) {
        if      (f == 2)     h_face = h[IDX_T(i,1,k)];
        else if (f == 3)     h_face = h[IDX_T(i,2,k)];
        else if (f == ny)    h_face = h[IDX_T(i,ny-1,k)];
        else { recon_y(h, wet, i, f-1, k, nx, ny, nz, do_pos, hmin, &hl, &hr); h_face = hr; }
    } else {
        if      (f == 2)     h_face = h[IDX_T(i,2,k)];
        else if (f == 3)     h_face = h[IDX_T(i,2,k)];
        else if (f == ny-1)  h_face = h[IDX_T(i,ny-1,k)];
        else if (f == ny)    h_face = h[IDX_T(i,ny,k)];
        else { recon_y(h, wet, i, f,   k, nx, ny, nz, do_pos, hmin, &hl, &hr); h_face = hl; }
    }
    return vv * h_face * dx_cv[M_V(i,f)];
}
__global__ void LBOUNDS kO_xflux(const double *__restrict__ h, const double *__restrict__ u,
                         const double *__restrict__ wet, const double *__restrict__ dy_cu,
                         double *__restrict__ mfx, int nx, int ny, int nz,
                         int nghost, int nx_phys, int do_pos, double hmin) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t nf = (size_t)(nx+1);
    if (t >= nf*(size_t)ny*(size_t)nz) return;
    int f = (int)(t % nf) + 1, j = (int)((t/nf) % (size_t)ny) + 1, k = (int)(t/(nf*(size_t)ny)) + 1;
    mfx[IDX_U(f,j,k)] = flux_x(h,u,wet,dy_cu,f,j,k,nx,ny,nz,nghost,nx_phys,do_pos,hmin);
}
__global__ void LBOUNDS kO_yflux(const double *__restrict__ h, const double *__restrict__ v,
                         const double *__restrict__ wet, const double *__restrict__ dx_cv,
                         double *__restrict__ mfy, int nx, int ny, int nz,
                         int nghost, int ny_phys, int do_pos, double hmin) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t nf = (size_t)(ny+1);
    if (t >= (size_t)nx*nf*(size_t)nz) return;
    int i = (int)(t % (size_t)nx) + 1, f = (int)((t/(size_t)nx) % nf) + 1, k = (int)(t/((size_t)nx*nf)) + 1;
    mfy[IDX_V(i,f,k)] = flux_y(h,v,wet,dx_cv,i,f,k,nx,ny,nz,nghost,ny_phys,do_pos,hmin);
}
// ---- v3: fully fused, one thread per cell -> fh (no mfx/mfy global) --------
__global__ void kO_fused(const double *__restrict__ h, const double *__restrict__ u, const double *__restrict__ v,
                         const double *__restrict__ wet, const double *__restrict__ dy_cu, const double *__restrict__ dx_cv,
                         const double *__restrict__ iareaT, double *__restrict__ fh,
                         int nx, int ny, int nz, int nghost, int nx_phys, int ny_phys, int do_pos, double hmin) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)nx*(size_t)ny*(size_t)nz) return;
    int i = (int)(t % (size_t)nx) + 1, j = (int)((t/(size_t)nx) % (size_t)ny) + 1, k = (int)(t/((size_t)nx*(size_t)ny)) + 1;
    double mfx_i  = flux_x(h,u,wet,dy_cu,i,  j,k,nx,ny,nz,nghost,nx_phys,do_pos,hmin);
    double mfx_ip = flux_x(h,u,wet,dy_cu,i+1,j,k,nx,ny,nz,nghost,nx_phys,do_pos,hmin);
    double mfy_j  = flux_y(h,v,wet,dx_cv,i,j,  k,nx,ny,nz,nghost,ny_phys,do_pos,hmin);
    double mfy_jp = flux_y(h,v,wet,dx_cv,i,j+1,k,nx,ny,nz,nghost,ny_phys,do_pos,hmin);
    fh[IDX_T(i,j,k)] = ((mfx_ip - mfx_i) + (mfy_jp - mfy_j)) * iareaT[M_T(i,j)];
}
// ---- divergence (unchanged) ----------------------------------------------
__global__ void kO_div(const double *__restrict__ mfx, const double *__restrict__ mfy,
                       const double *__restrict__ iareaT, double *__restrict__ fh,
                       int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)nx*(size_t)ny*(size_t)nz) return;
    int i = (int)(t % (size_t)nx) + 1;
    int j = (int)((t/(size_t)nx) % (size_t)ny) + 1;
    int k = (int)(t/((size_t)nx*(size_t)ny)) + 1;
    fh[IDX_T(i,j,k)] = ((mfx[IDX_U(i+1,j,k)] - mfx[IDX_U(i,j,k)]) +
                        (mfy[IDX_V(i,j+1,k)] - mfy[IDX_V(i,j,k)])) * iareaT[M_T(i,j)];
}

#ifndef TPB
#define TPB 128
#endif
#define NBLK(n) (int)(((size_t)(n) + TPB - 1) / TPB)

// ---- v4: k-looping flux kernels (one thread per face-line, loop k) --------
// Independent k-iterations expose ILP (helps the latency-bound x kernel); the
// wall test + 2-D metric address amortise over all nz layers.
__global__ void LBOUNDS kO_xflux_k(const double *__restrict__ h, const double *__restrict__ u,
                         const double *__restrict__ wet, const double *__restrict__ dy_cu,
                         double *__restrict__ mfx, int nx, int ny, int nz,
                         int nghost, int nx_phys, int do_pos, double hmin) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t nf = (size_t)(nx+1);
    if (t >= nf*(size_t)ny) return;
    int f = (int)(t % nf) + 1, j = (int)(t / nf) + 1;
    bool wall = (f == 1 || f == nx+1 || f == nghost+1 || f == nghost+nx_phys+1);
    for (int k = 1; k <= nz; ++k)
        mfx[IDX_U(f,j,k)] = wall ? 0.0 : flux_x(h,u,wet,dy_cu,f,j,k,nx,ny,nz,nghost,nx_phys,do_pos,hmin);
}
__global__ void LBOUNDS kO_yflux_k(const double *__restrict__ h, const double *__restrict__ v,
                         const double *__restrict__ wet, const double *__restrict__ dx_cv,
                         double *__restrict__ mfy, int nx, int ny, int nz,
                         int nghost, int ny_phys, int do_pos, double hmin) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t nf = (size_t)(ny+1);
    if (t >= (size_t)nx*nf) return;
    int i = (int)(t % (size_t)nx) + 1, f = (int)(t / (size_t)nx) + 1;
    bool wall = (f == 1 || f == ny+1 || f == nghost+1 || f == nghost+ny_phys+1);
    for (int k = 1; k <= nz; ++k)
        mfy[IDX_V(i,f,k)] = wall ? 0.0 : flux_y(h,v,wet,dx_cv,i,f,k,nx,ny,nz,nghost,ny_phys,do_pos,hmin);
}

// ---- v5: k-blocked flux (KB consecutive k per thread: ILP without losing parallelism)
#ifndef KB
#define KB 4
#endif
__global__ void LBOUNDS kO_xflux_kb(const double *__restrict__ h, const double *__restrict__ u,
                         const double *__restrict__ wet, const double *__restrict__ dy_cu,
                         double *__restrict__ mfx, int nx, int ny, int nz,
                         int nghost, int nx_phys, int do_pos, double hmin) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t nf = (size_t)(nx+1); int nkb = (nz + KB - 1) / KB;
    if (t >= nf*(size_t)ny*(size_t)nkb) return;
    int f = (int)(t % nf) + 1, j = (int)((t/nf) % (size_t)ny) + 1, k0 = (int)(t/(nf*(size_t)ny))*KB + 1;
    bool wall = (f == 1 || f == nx+1 || f == nghost+1 || f == nghost+nx_phys+1);
    #pragma unroll
    for (int kk = 0; kk < KB; ++kk) { int k = k0 + kk; if (k > nz) break;
        mfx[IDX_U(f,j,k)] = wall ? 0.0 : flux_x(h,u,wet,dy_cu,f,j,k,nx,ny,nz,nghost,nx_phys,do_pos,hmin); }
}
__global__ void LBOUNDS kO_yflux_kb(const double *__restrict__ h, const double *__restrict__ v,
                         const double *__restrict__ wet, const double *__restrict__ dx_cv,
                         double *__restrict__ mfy, int nx, int ny, int nz,
                         int nghost, int ny_phys, int do_pos, double hmin) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t nf = (size_t)(ny+1); int nkb = (nz + KB - 1) / KB;
    if (t >= (size_t)nx*nf*(size_t)nkb) return;
    int i = (int)(t % (size_t)nx) + 1, f = (int)((t/(size_t)nx) % nf) + 1, k0 = (int)(t/((size_t)nx*nf))*KB + 1;
    bool wall = (f == 1 || f == ny+1 || f == nghost+1 || f == nghost+ny_phys+1);
    #pragma unroll
    for (int kk = 0; kk < KB; ++kk) { int k = k0 + kk; if (k > nz) break;
        mfy[IDX_V(i,f,k)] = wall ? 0.0 : flux_y(h,v,wet,dx_cv,i,f,k,nx,ny,nz,nghost,ny_phys,do_pos,hmin); }
}

// ---- v6/v7: fuse divergence with ONE flux direction (kill an mf round-trip) ----
// v6: keep mfy in global; recompute the 2 x-faces per cell (x-recon is the cheap
// contiguous one). v7: symmetric, keep mfx and recompute the 2 y-faces.
__global__ void LBOUNDS kO_xdiv(const double *__restrict__ h, const double *__restrict__ u,
        const double *__restrict__ wet, const double *__restrict__ dy_cu, const double *__restrict__ mfy,
        const double *__restrict__ iareaT, double *__restrict__ fh,
        int nx, int ny, int nz, int nghost, int nx_phys, int do_pos, double hmin) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)nx*(size_t)ny*(size_t)nz) return;
    int i = (int)(t % (size_t)nx) + 1, j = (int)((t/(size_t)nx) % (size_t)ny) + 1, k = (int)(t/((size_t)nx*(size_t)ny)) + 1;
    double mfx_i  = flux_x(h,u,wet,dy_cu,i,  j,k,nx,ny,nz,nghost,nx_phys,do_pos,hmin);
    double mfx_ip = flux_x(h,u,wet,dy_cu,i+1,j,k,nx,ny,nz,nghost,nx_phys,do_pos,hmin);
    fh[IDX_T(i,j,k)] = ((mfx_ip - mfx_i) + (mfy[IDX_V(i,j+1,k)] - mfy[IDX_V(i,j,k)])) * iareaT[M_T(i,j)];
}
__global__ void LBOUNDS kO_ydiv(const double *__restrict__ h, const double *__restrict__ v,
        const double *__restrict__ wet, const double *__restrict__ dx_cv, const double *__restrict__ mfx,
        const double *__restrict__ iareaT, double *__restrict__ fh,
        int nx, int ny, int nz, int nghost, int ny_phys, int do_pos, double hmin) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)nx*(size_t)ny*(size_t)nz) return;
    int i = (int)(t % (size_t)nx) + 1, j = (int)((t/(size_t)nx) % (size_t)ny) + 1, k = (int)(t/((size_t)nx*(size_t)ny)) + 1;
    double mfy_j  = flux_y(h,v,wet,dx_cv,i,j,  k,nx,ny,nz,nghost,ny_phys,do_pos,hmin);
    double mfy_jp = flux_y(h,v,wet,dx_cv,i,j+1,k,nx,ny,nz,nghost,ny_phys,do_pos,hmin);
    fh[IDX_T(i,j,k)] = ((mfx[IDX_U(i+1,j,k)] - mfx[IDX_U(i,j,k)]) + (mfy_jp - mfy_j)) * iareaT[M_T(i,j)];
}

// recon from pre-loaded raw neighbour values (used by the smem tiled kernel)
__device__ __forceinline__ void recon_vals(double h0, double hm1r, double hp1r, double hm2r, double hp2r,
        double wm2, double wm1, double w0, double wp1, double wp2, int do_pos, double hmin,
        double *hl, double *hr) {
    double hm1 = o_mirror(hm1r, h0, wm1);
    double hp1 = o_mirror(hp1r, h0, wp1);
    double hm2 = o_mirror(hm2r, hm1, wm2);
    double hp2 = o_mirror(hp2r, hp1, wp2);
    double dm1, d0, dp1;
    o_slope(hm2, hm1, h0, &dm1); o_slope(hm1, h0, hp1, &d0); o_slope(h0, hp1, hp2, &dp1);
    d0 = d0 * wm1 * w0 * wp1;
    double L = 0.5*(hm1 + h0) - (d0 - dm1)/6.0, R = 0.5*(h0 + hp1) - (dp1 - d0)/6.0;
    o_cell_lim(h0, &L, &R);
    if (do_pos) o_limpos(h0, &L, &R, hmin);
    *hl = L; *hr = R;
}
// ---- v8: shared-memory-tiled Y flux (kills the strided-j L2 thrash) --------
#ifndef BXS
#define BXS 32
#endif
#ifndef BYS
#define BYS 8
#endif
__global__ void kO_yflux_smem(const double *__restrict__ h, const double *__restrict__ v,
        const double *__restrict__ wet, const double *__restrict__ dx_cv, double *__restrict__ mfy,
        int nx, int ny, int nz, int nghost, int ny_phys, int do_pos, double hmin) {
    __shared__ double sh[BYS+5][BXS];
    int tx = threadIdx.x, ty = threadIdx.y;
    int k  = blockIdx.z + 1;
    int i  = blockIdx.x*BXS + tx + 1;
    int f0 = blockIdx.y*BYS;
    int rowbase = f0 - 2;                 // smem row r -> 1-based cell (rowbase+r)
    for (int r = ty; r < BYS+5; r += BYS) {
        int cc = rowbase + r;
        sh[r][tx] = (i <= nx && cc >= 1 && cc <= ny) ? h[IDX_T(i,cc,k)] : 0.0;
    }
    __syncthreads();
    int f = f0 + ty + 1;
    if (i > nx || f > ny+1) return;
    if (f == 1 || f == ny+1 || f == nghost+1 || f == nghost+ny_phys+1) { mfy[IDX_V(i,f,k)] = 0.0; return; }
    double vv = v[IDX_V(i,f,k)], h_face, hl, hr;
    if (vv >= 0.0) {
        if      (f == 2)     h_face = h[IDX_T(i,1,k)];
        else if (f == 3)     h_face = h[IDX_T(i,2,k)];
        else if (f == ny)    h_face = h[IDX_T(i,ny-1,k)];
        else { int c=f-1, lc=c-rowbase;
            recon_vals(sh[lc][tx],sh[lc-1][tx],sh[lc+1][tx],sh[lc-2][tx],sh[lc+2][tx],
                wet[M_T(i,c-2)],wet[M_T(i,c-1)],wet[M_T(i,c)],wet[M_T(i,c+1)],wet[M_T(i,c+2)],do_pos,hmin,&hl,&hr); h_face=hr; }
    } else {
        if      (f == 2)     h_face = h[IDX_T(i,2,k)];
        else if (f == 3)     h_face = h[IDX_T(i,2,k)];
        else if (f == ny-1)  h_face = h[IDX_T(i,ny-1,k)];
        else if (f == ny)    h_face = h[IDX_T(i,ny,k)];
        else { int c=f, lc=c-rowbase;
            recon_vals(sh[lc][tx],sh[lc-1][tx],sh[lc+1][tx],sh[lc-2][tx],sh[lc+2][tx],
                wet[M_T(i,c-2)],wet[M_T(i,c-1)],wet[M_T(i,c)],wet[M_T(i,c+1)],wet[M_T(i,c+2)],do_pos,hmin,&hl,&hr); h_face=hl; }
    }
    mfy[IDX_V(i,f,k)] = vv * h_face * dx_cv[M_V(i,f)];
}

#ifndef OPTVER
#define OPTVER 2
#endif
extern "C" void continuity_opt_launch(
    const double *h, const double *u, const double *v, const double *wet_T,
    const double *dy_cu, const double *dx_cv, const double *iareaT,
    double *mfx, double *mfy, double *fh,
    int nx, int ny, int nz, int nghost, int nx_phys, int ny_phys,
    int do_pos, double h_min_pos) {
#if OPTVER == 3
    kO_fused<<<NBLK((size_t)nx*ny*nz), TPB>>>(h, u, v, wet_T, dy_cu, dx_cv, iareaT, fh,
                                              nx, ny, nz, nghost, nx_phys, ny_phys, do_pos, h_min_pos);
#elif OPTVER == 4
    kO_xflux_k<<<NBLK((size_t)(nx+1)*ny), TPB>>>(h, u, wet_T, dy_cu, mfx, nx, ny, nz, nghost, nx_phys, do_pos, h_min_pos);
    kO_yflux_k<<<NBLK((size_t)nx*(ny+1)), TPB>>>(h, v, wet_T, dx_cv, mfy, nx, ny, nz, nghost, ny_phys, do_pos, h_min_pos);
    kO_div    <<<NBLK((size_t)nx*ny*nz), TPB>>>(mfx, mfy, iareaT, fh, nx, ny, nz);
#elif OPTVER == 5
    { int nkb = (nz+KB-1)/KB;
    kO_xflux_kb<<<NBLK((size_t)(nx+1)*ny*nkb), TPB>>>(h, u, wet_T, dy_cu, mfx, nx, ny, nz, nghost, nx_phys, do_pos, h_min_pos);
    kO_yflux_kb<<<NBLK((size_t)nx*(ny+1)*nkb), TPB>>>(h, v, wet_T, dx_cv, mfy, nx, ny, nz, nghost, ny_phys, do_pos, h_min_pos);
    kO_div     <<<NBLK((size_t)nx*ny*nz), TPB>>>(mfx, mfy, iareaT, fh, nx, ny, nz); }
#elif OPTVER == 6
    kO_yflux<<<NBLK((size_t)nx*(ny+1)*nz), TPB>>>(h, v, wet_T, dx_cv, mfy, nx, ny, nz, nghost, ny_phys, do_pos, h_min_pos);
    kO_xdiv <<<NBLK((size_t)nx*ny*nz), TPB>>>(h, u, wet_T, dy_cu, mfy, iareaT, fh, nx, ny, nz, nghost, nx_phys, do_pos, h_min_pos);
#elif OPTVER == 7
    kO_xflux<<<NBLK((size_t)(nx+1)*ny*nz), TPB>>>(h, u, wet_T, dy_cu, mfx, nx, ny, nz, nghost, nx_phys, do_pos, h_min_pos);
    kO_ydiv <<<NBLK((size_t)nx*ny*nz), TPB>>>(h, v, wet_T, dx_cv, mfx, iareaT, fh, nx, ny, nz, nghost, ny_phys, do_pos, h_min_pos);
#elif OPTVER == 8
    kO_xflux<<<NBLK((size_t)(nx+1)*ny*nz), TPB>>>(h, u, wet_T, dy_cu, mfx, nx, ny, nz, nghost, nx_phys, do_pos, h_min_pos);
    { dim3 blk(BXS,BYS); dim3 grd((nx+BXS-1)/BXS, (ny+1+BYS-1)/BYS, nz);
      kO_yflux_smem<<<grd,blk>>>(h, v, wet_T, dx_cv, mfy, nx, ny, nz, nghost, ny_phys, do_pos, h_min_pos); }
    kO_div<<<NBLK((size_t)nx*ny*nz), TPB>>>(mfx, mfy, iareaT, fh, nx, ny, nz);
#else
    kO_xflux<<<NBLK((size_t)(nx+1)*ny*nz), TPB>>>(h, u, wet_T, dy_cu, mfx, nx, ny, nz, nghost, nx_phys, do_pos, h_min_pos);
    kO_yflux<<<NBLK((size_t)nx*(ny+1)*nz), TPB>>>(h, v, wet_T, dx_cv, mfy, nx, ny, nz, nghost, ny_phys, do_pos, h_min_pos);
    kO_div  <<<NBLK((size_t)nx*ny*nz), TPB>>>(mfx, mfy, iareaT, fh, nx, ny, nz);
#endif
}
