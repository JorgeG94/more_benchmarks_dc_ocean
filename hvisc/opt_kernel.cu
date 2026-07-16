// Optimized CUDA for the ocean horizontal-viscosity Smagorinsky closure.
// Best-practice pass over hvisc_kernel.cu (the faithful 12-kernel port), freed
// from the transliteration rule but held to FMA-level agreement (the strain has
// two sqrt calls, so the bar is field-relative max|diff|/max|field| < 1e-12,
// not bitwise; the arithmetic ORDER is kept identical so contraction is the
// only source of difference).
//
// RESULT (V100, 473x297x30): see OPTIMIZATION.md.
//
// The winner (OPTVER=2, IDX32=1) is a full fusion of the whole closure, "remove
// genuine waste":
//   * FUSE the entire closure into ONE kernel. The faithful path is
//     smag(strain) -> ah_face global array -> apply(ah_face x Laplacian),
//     12 kernels writing/reading two full face-sized intermediate arrays
//     (ah_face_x on U, ah_face_y on V). We compute ah_face in registers and
//     feed it straight into the Laplacian, so ah_face NEVER touches global
//     memory. 12 kernels -> 1 (OPTVER 2) / 2 (OPTVER 1), two face arrays of
//     DRAM traffic gone.
//     KEY: the faithful ah_face boundary/reuse kernels (kS_u_reuse/_bg,
//     kS_v_reuse/_bg) only fill ah_face's HALO, and kH_u/kH_v read ah_face
//     solely at interior cells they themselves produced -> those 4 kernels are
//     dead code for the du/dv output and are dropped.
//   * 32-bit indexing (the domain fits int32) -- fewer address registers.
//
// The single interior region also owns the du/dv wall zeros (boundary threads
// write 0.0), folding the 4 kH_*_b* wall kernels in for free -- no separate
// memset per rep.
//
// OPTVER 2 (one kernel over U-then-V faces, single launch) beats OPTVER 1 (a
// kernel each for U and V) most on small grids where launch overhead dominates
// (1.83x vs 1.73x at 108x137x30); the two tie on large memory-bound grids.
// OPTVER 1 kept for the record -- see the switch and OPTIMIZATION.md.
#include <cstdio>
#include "gpu_rt.h"

#ifndef TPB
#define TPB 128
#endif
#ifndef IDX32
#define IDX32 1          // valid while nx*ny*nz < 2^31 (every realistic domain)
#endif

#if IDX32
#define IDX_U(i, j, k) ((int)((i)-1) + (nx+1)*((int)((j)-1) + (ny)*((int)((k)-1))))
#define IDX_V(i, j, k) ((int)((i)-1) + (nx)*((int)((j)-1) + (ny+1)*((int)((k)-1))))
#define M_T(i, j)  ((int)((i)-1) + (nx)*((int)((j)-1)))
#define M_BU(i, j) ((int)((i)-1) + (nx+1)*((int)((j)-1)))
#define M_CU(i, j) ((int)((i)-1) + (nx+1)*((int)((j)-1)))
#define M_CV(i, j) ((int)((i)-1) + (nx)*((int)((j)-1)))
#else
#define IDX_U(i, j, k) ((size_t)((i)-1) + (size_t)(nx+1)*((size_t)((j)-1) + (size_t)(ny)*(size_t)((k)-1)))
#define IDX_V(i, j, k) ((size_t)((i)-1) + (size_t)(nx)*((size_t)((j)-1) + (size_t)(ny+1)*(size_t)((k)-1)))
#define M_T(i, j)  ((size_t)((i)-1) + (size_t)(nx)*(size_t)((j)-1))
#define M_BU(i, j) ((size_t)((i)-1) + (size_t)(nx+1)*(size_t)((j)-1))
#define M_CU(i, j) ((size_t)((i)-1) + (size_t)(nx+1)*(size_t)((j)-1))
#define M_CV(i, j) ((size_t)((i)-1) + (size_t)(nx)*(size_t)((j)-1))
#endif

// ---- fused ah_face_x (strain) + u-face Laplacian -> du_visc, in registers ----
// Reproduces kS_u's body then kH_u's body, same arithmetic order.
__device__ __forceinline__ double du_interior(
        const double *__restrict__ u, const double *__restrict__ v,
        const double *__restrict__ dxT, const double *__restrict__ dyT,
        const double *__restrict__ idxT, const double *__restrict__ idyT,
        const double *__restrict__ dy_dxBu, const double *__restrict__ dx_dyBu,
        const double *__restrict__ idyCv, const double *__restrict__ idxCu,
        const double *__restrict__ wet_q, const double *__restrict__ dy_dxT,
        const double *__restrict__ iareaCu,
        double c_smag, double ah_bg, double ah_max, double ns,
        int i, int j, int k, int nx, int ny, int nz) {
    // --- ah_face_x (Smagorinsky strain) ---
    double s = c_smag*sqrt(dxT[M_T(i,j)]*dyT[M_T(i,j)]); double smag_scale = s*s;
    double D_T_W = (u[IDX_U(i,j,k)] - u[IDX_U(i-1,j,k)])*idxT[M_T(i-1,j)]
                 - (v[IDX_V(i-1,j+1,k)] - v[IDX_V(i-1,j,k)])*idyT[M_T(i-1,j)];
    double D_T_E = (u[IDX_U(i+1,j,k)] - u[IDX_U(i,j,k)])*idxT[M_T(i,j)]
                 - (v[IDX_V(i,j+1,k)] - v[IDX_V(i,j,k)])*idyT[M_T(i,j)];
    double D_T_face = 0.5*(D_T_W + D_T_E);
    double D_S_S = ((1.0 - 2.0*ns)*wet_q[M_BU(i,j)] + 2.0*ns)*
        (dy_dxBu[M_BU(i,j)]*(v[IDX_V(i,j,k)]*idyCv[M_CV(i,j)] - v[IDX_V(i-1,j,k)]*idyCv[M_CV(i-1,j)]) +
         dx_dyBu[M_BU(i,j)]*(u[IDX_U(i,j,k)]*idxCu[M_CU(i,j)] - u[IDX_U(i,j-1,k)]*idxCu[M_CU(i,j-1)]));
    double D_S_N = ((1.0 - 2.0*ns)*wet_q[M_BU(i,j+1)] + 2.0*ns)*
        (dy_dxBu[M_BU(i,j+1)]*(v[IDX_V(i,j+1,k)]*idyCv[M_CV(i,j+1)] - v[IDX_V(i-1,j+1,k)]*idyCv[M_CV(i-1,j+1)]) +
         dx_dyBu[M_BU(i,j+1)]*(u[IDX_U(i,j+1,k)]*idxCu[M_CU(i,j+1)] - u[IDX_U(i,j,k)]*idxCu[M_CU(i,j)]));
    double D_S_face = 0.5*(D_S_S + D_S_N);
    double strain = sqrt(D_T_face*D_T_face + D_S_face*D_S_face);
    double ah = fmin(ah_max, fmax(ah_bg, smag_scale*strain));
    // --- u-face Laplacian ---
    double lap_u = iareaCu[M_CU(i,j)]*(
        (dy_dxT[M_T(i,j)]*(u[IDX_U(i+1,j,k)] - u[IDX_U(i,j,k)]) -
         dy_dxT[M_T(i-1,j)]*(u[IDX_U(i,j,k)] - u[IDX_U(i-1,j,k)])) +
        (dx_dyBu[M_BU(i,j+1)]*(u[IDX_U(i,j+1,k)] - u[IDX_U(i,j,k)]) -
         dx_dyBu[M_BU(i,j)]*(u[IDX_U(i,j,k)] - u[IDX_U(i,j-1,k)])));
    return ah*lap_u;
}

// ---- fused ah_face_y (strain) + v-face Laplacian -> dv_visc, in registers ----
__device__ __forceinline__ double dv_interior(
        const double *__restrict__ u, const double *__restrict__ v,
        const double *__restrict__ dxT, const double *__restrict__ dyT,
        const double *__restrict__ idxT, const double *__restrict__ idyT,
        const double *__restrict__ dy_dxBu, const double *__restrict__ dx_dyBu,
        const double *__restrict__ idyCv, const double *__restrict__ idxCu,
        const double *__restrict__ wet_q, const double *__restrict__ dx_dyT,
        const double *__restrict__ iareaCv,
        double c_smag, double ah_bg, double ah_max, double ns,
        int i, int j, int k, int nx, int ny, int nz) {
    // --- ah_face_y (Smagorinsky strain) ---
    double s = c_smag*sqrt(dxT[M_T(i,j)]*dyT[M_T(i,j)]); double smag_scale = s*s;
    double D_T_S = (u[IDX_U(i+1,j-1,k)] - u[IDX_U(i,j-1,k)])*idxT[M_T(i,j-1)]
                 - (v[IDX_V(i,j,k)] - v[IDX_V(i,j-1,k)])*idyT[M_T(i,j-1)];
    double D_T_N = (u[IDX_U(i+1,j,k)] - u[IDX_U(i,j,k)])*idxT[M_T(i,j)]
                 - (v[IDX_V(i,j+1,k)] - v[IDX_V(i,j,k)])*idyT[M_T(i,j)];
    double D_T_face = 0.5*(D_T_S + D_T_N);
    double D_S_W = ((1.0 - 2.0*ns)*wet_q[M_BU(i,j)] + 2.0*ns)*
        (dy_dxBu[M_BU(i,j)]*(v[IDX_V(i,j,k)]*idyCv[M_CV(i,j)] - v[IDX_V(i-1,j,k)]*idyCv[M_CV(i-1,j)]) +
         dx_dyBu[M_BU(i,j)]*(u[IDX_U(i,j,k)]*idxCu[M_CU(i,j)] - u[IDX_U(i,j-1,k)]*idxCu[M_CU(i,j-1)]));
    double D_S_E = ((1.0 - 2.0*ns)*wet_q[M_BU(i+1,j)] + 2.0*ns)*
        (dy_dxBu[M_BU(i+1,j)]*(v[IDX_V(i+1,j,k)]*idyCv[M_CV(i+1,j)] - v[IDX_V(i,j,k)]*idyCv[M_CV(i,j)]) +
         dx_dyBu[M_BU(i+1,j)]*(u[IDX_U(i+1,j,k)]*idxCu[M_CU(i+1,j)] - u[IDX_U(i+1,j-1,k)]*idxCu[M_CU(i+1,j-1)]));
    double D_S_face = 0.5*(D_S_W + D_S_E);
    double strain = sqrt(D_T_face*D_T_face + D_S_face*D_S_face);
    double ah = fmin(ah_max, fmax(ah_bg, smag_scale*strain));
    // --- v-face Laplacian ---
    double lap_v = iareaCv[M_CV(i,j)]*(
        (dx_dyT[M_T(i,j)]*(v[IDX_V(i,j+1,k)] - v[IDX_V(i,j,k)]) -
         dx_dyT[M_T(i,j-1)]*(v[IDX_V(i,j,k)] - v[IDX_V(i,j-1,k)])) +
        (dy_dxBu[M_BU(i+1,j)]*(v[IDX_V(i+1,j,k)] - v[IDX_V(i,j,k)]) -
         dy_dxBu[M_BU(i,j)]*(v[IDX_V(i,j,k)] - v[IDX_V(i-1,j,k)])));
    return ah*lap_v;
}

// ---- one thread per U-face cell: interior fused, boundary -> 0 --------------
__global__ void kU_fused(
        const double *__restrict__ u, const double *__restrict__ v,
        const double *__restrict__ dxT, const double *__restrict__ dyT,
        const double *__restrict__ idxT, const double *__restrict__ idyT,
        const double *__restrict__ dy_dxBu, const double *__restrict__ dx_dyBu,
        const double *__restrict__ idyCv, const double *__restrict__ idxCu,
        const double *__restrict__ wet_q, const double *__restrict__ dy_dxT,
        const double *__restrict__ iareaCu, double *__restrict__ du_visc,
        double c_smag, double ah_bg, double ah_max, double ns, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t nf = (size_t)(nx+1);
    if (t >= nf*(size_t)ny*(size_t)nz) return;
    int i = (int)(t % nf) + 1, j = (int)((t/nf) % (size_t)ny) + 1, k = (int)(t/(nf*(size_t)ny)) + 1;
    double out;
    if (i >= 2 && i <= nx && j >= 2 && j <= ny-1)
        out = du_interior(u,v,dxT,dyT,idxT,idyT,dy_dxBu,dx_dyBu,idyCv,idxCu,wet_q,dy_dxT,iareaCu,
                          c_smag,ah_bg,ah_max,ns,i,j,k,nx,ny,nz);
    else
        out = 0.0;                                   // walls: kH_u_bj / kH_u_bi
    du_visc[IDX_U(i,j,k)] = out;
}

// ---- one thread per V-face cell: interior fused, boundary -> 0 --------------
__global__ void kV_fused(
        const double *__restrict__ u, const double *__restrict__ v,
        const double *__restrict__ dxT, const double *__restrict__ dyT,
        const double *__restrict__ idxT, const double *__restrict__ idyT,
        const double *__restrict__ dy_dxBu, const double *__restrict__ dx_dyBu,
        const double *__restrict__ idyCv, const double *__restrict__ idxCu,
        const double *__restrict__ wet_q, const double *__restrict__ dx_dyT,
        const double *__restrict__ iareaCv, double *__restrict__ dv_visc,
        double c_smag, double ah_bg, double ah_max, double ns, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t nf = (size_t)nx;
    if (t >= nf*(size_t)(ny+1)*(size_t)nz) return;
    int i = (int)(t % nf) + 1, j = (int)((t/nf) % (size_t)(ny+1)) + 1, k = (int)(t/(nf*(size_t)(ny+1))) + 1;
    double out;
    if (i >= 2 && i <= nx-1 && j >= 2 && j <= ny)
        out = dv_interior(u,v,dxT,dyT,idxT,idyT,dy_dxBu,dx_dyBu,idyCv,idxCu,wet_q,dx_dyT,iareaCv,
                          c_smag,ah_bg,ah_max,ns,i,j,k,nx,ny,nz);
    else
        out = 0.0;                                   // walls: kH_v_bi / kH_v_bj
    dv_visc[IDX_V(i,j,k)] = out;
}

// ---- OPTVER 2: one merged kernel over U-then-V faces (single launch) --------
__global__ void kUV_fused(
        const double *__restrict__ u, const double *__restrict__ v,
        const double *__restrict__ dxT, const double *__restrict__ dyT,
        const double *__restrict__ idxT, const double *__restrict__ idyT,
        const double *__restrict__ dy_dxBu, const double *__restrict__ dx_dyBu,
        const double *__restrict__ idyCv, const double *__restrict__ idxCu,
        const double *__restrict__ wet_q, const double *__restrict__ dy_dxT,
        const double *__restrict__ iareaCu, const double *__restrict__ dx_dyT,
        const double *__restrict__ iareaCv, double *__restrict__ du_visc,
        double *__restrict__ dv_visc,
        double c_smag, double ah_bg, double ah_max, double ns, int nx, int ny, int nz) {
    size_t t  = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t nU = (size_t)(nx+1)*(size_t)ny*(size_t)nz;
    size_t nV = (size_t)nx*(size_t)(ny+1)*(size_t)nz;
    if (t >= nU + nV) return;
    if (t < nU) {
        size_t nf = (size_t)(nx+1);
        int i = (int)(t % nf) + 1, j = (int)((t/nf) % (size_t)ny) + 1, k = (int)(t/(nf*(size_t)ny)) + 1;
        double out = (i >= 2 && i <= nx && j >= 2 && j <= ny-1)
            ? du_interior(u,v,dxT,dyT,idxT,idyT,dy_dxBu,dx_dyBu,idyCv,idxCu,wet_q,dy_dxT,iareaCu,
                          c_smag,ah_bg,ah_max,ns,i,j,k,nx,ny,nz)
            : 0.0;
        du_visc[IDX_U(i,j,k)] = out;
    } else {
        size_t tv = t - nU, nf = (size_t)nx;
        int i = (int)(tv % nf) + 1, j = (int)((tv/nf) % (size_t)(ny+1)) + 1, k = (int)(tv/(nf*(size_t)(ny+1))) + 1;
        double out = (i >= 2 && i <= nx-1 && j >= 2 && j <= ny)
            ? dv_interior(u,v,dxT,dyT,idxT,idyT,dy_dxBu,dx_dyBu,idyCv,idxCu,wet_q,dx_dyT,iareaCv,
                          c_smag,ah_bg,ah_max,ns,i,j,k,nx,ny,nz)
            : 0.0;
        dv_visc[IDX_V(i,j,k)] = out;
    }
}

#define NBLK(n) (int)(((size_t)(n) + TPB - 1) / TPB)
static inline void msync(int sync, int level) {
    if (sync >= level) { cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); }
}

#ifndef OPTVER
#define OPTVER 2
#endif

// Full fused closure: smag(strain, sqrt) folded into apply(Laplacian) -> du/dv.
extern "C" void hvisc_opt_launch(
    const double *u_face, const double *v_face,
    const double *dxT, const double *dyT, const double *idxT, const double *idyT,
    const double *dy_dxBu, const double *dx_dyBu, const double *idyCv, const double *idxCu,
    const double *wet_q, const double *dy_dxT, const double *iareaCu,
    const double *dx_dyT, const double *iareaCv,
    double *du_visc, double *dv_visc,
    double c_smag, double ah_bg, double ah_max, double ns, int nx, int ny, int nz, int sync) {
#if OPTVER == 2
    size_t nUV = (size_t)(nx+1)*ny*nz + (size_t)nx*(ny+1)*nz;
    kUV_fused<<<NBLK(nUV), TPB>>>(u_face,v_face,dxT,dyT,idxT,idyT,dy_dxBu,dx_dyBu,idyCv,idxCu,wet_q,
                                  dy_dxT,iareaCu,dx_dyT,iareaCv,du_visc,dv_visc,
                                  c_smag,ah_bg,ah_max,ns,nx,ny,nz); msync(sync,1);
#else
    kU_fused<<<NBLK((size_t)(nx+1)*ny*nz), TPB>>>(u_face,v_face,dxT,dyT,idxT,idyT,dy_dxBu,dx_dyBu,idyCv,idxCu,wet_q,
                                                  dy_dxT,iareaCu,du_visc,c_smag,ah_bg,ah_max,ns,nx,ny,nz); msync(sync,2);
    kV_fused<<<NBLK((size_t)nx*(ny+1)*nz), TPB>>>(u_face,v_face,dxT,dyT,idxT,idyT,dy_dxBu,dx_dyBu,idyCv,idxCu,wet_q,
                                                  dx_dyT,iareaCv,dv_visc,c_smag,ah_bg,ah_max,ns,nx,ny,nz); msync(sync,1);
#endif
}
