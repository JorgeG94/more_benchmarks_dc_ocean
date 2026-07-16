// Faithful CUDA C port of hvisc_compute_face (ocean horizontal viscosity:
// per-face curvilinear FV Laplacian x per-face viscosity). Same loops, same
// order, same arithmetic grouping as hvisc.F90 -> bit-identical (FMA aside).
//
// Column-major (the device arrays ARE the Fortran allocations via host_data):
//   u_face, ah_face_x, du_visc : (nx+1, ny,   nz)  -> IDX_U
//   v_face, ah_face_y, dv_visc : (nx,   ny+1, nz)  -> IDX_V
//   dy_dxT, dx_dyT             : (nx,   ny)        -> M_T
//   dx_dyBu, dy_dxBu           : (nx+1, ny+1)      -> M_BU
//   iareaCu                    : (nx+1, ny)        -> M_CU
//   iareaCv                    : (nx,   ny+1)      -> M_CV
#include <cstdio>
#include "gpu_rt.h"

#define IDX_U(i, j, k) ((size_t)((i)-1) + (size_t)(nx+1)*((size_t)((j)-1) + (size_t)(ny)*(size_t)((k)-1)))
#define IDX_V(i, j, k) ((size_t)((i)-1) + (size_t)(nx)*((size_t)((j)-1) + (size_t)(ny+1)*(size_t)((k)-1)))
#define M_T(i, j)  ((size_t)((i)-1) + (size_t)(nx)*(size_t)((j)-1))
#define M_BU(i, j) ((size_t)((i)-1) + (size_t)(nx+1)*(size_t)((j)-1))
#define M_CU(i, j) ((size_t)((i)-1) + (size_t)(nx+1)*(size_t)((j)-1))
#define M_CV(i, j) ((size_t)((i)-1) + (size_t)(nx)*(size_t)((j)-1))

// ======================= Smagorinsky ah_face (strain) ======================
// wet_q is corner-shaped (nx+1, ny+1) -> M_BU.
__global__ void kS_u(const double *__restrict__ u_face, const double *__restrict__ v_face,
                     const double *__restrict__ dxT, const double *__restrict__ dyT,
                     const double *__restrict__ idxT, const double *__restrict__ idyT,
                     const double *__restrict__ dy_dxBu, const double *__restrict__ dx_dyBu,
                     const double *__restrict__ idyCv, const double *__restrict__ idxCu,
                     const double *__restrict__ wet_q, double *__restrict__ ah_face_x,
                     double c_smag, double ah_bg, double ah_max, double ns, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t ni = (size_t)(nx-1), nj = (size_t)(ny-2);
    if (t >= ni*nj*(size_t)nz) return;
    int i = (int)(t % ni) + 2, j = (int)((t/ni) % nj) + 2, k = (int)(t/(ni*nj)) + 1;
    double s = c_smag*sqrt(dxT[M_T(i,j)]*dyT[M_T(i,j)]); double smag_scale = s*s;
    double D_T_W = (u_face[IDX_U(i,j,k)] - u_face[IDX_U(i-1,j,k)])*idxT[M_T(i-1,j)]
                 - (v_face[IDX_V(i-1,j+1,k)] - v_face[IDX_V(i-1,j,k)])*idyT[M_T(i-1,j)];
    double D_T_E = (u_face[IDX_U(i+1,j,k)] - u_face[IDX_U(i,j,k)])*idxT[M_T(i,j)]
                 - (v_face[IDX_V(i,j+1,k)] - v_face[IDX_V(i,j,k)])*idyT[M_T(i,j)];
    double D_T_face = 0.5*(D_T_W + D_T_E);
    double D_S_S = ((1.0 - 2.0*ns)*wet_q[M_BU(i,j)] + 2.0*ns)*
        (dy_dxBu[M_BU(i,j)]*(v_face[IDX_V(i,j,k)]*idyCv[M_CV(i,j)] - v_face[IDX_V(i-1,j,k)]*idyCv[M_CV(i-1,j)]) +
         dx_dyBu[M_BU(i,j)]*(u_face[IDX_U(i,j,k)]*idxCu[M_CU(i,j)] - u_face[IDX_U(i,j-1,k)]*idxCu[M_CU(i,j-1)]));
    double D_S_N = ((1.0 - 2.0*ns)*wet_q[M_BU(i,j+1)] + 2.0*ns)*
        (dy_dxBu[M_BU(i,j+1)]*(v_face[IDX_V(i,j+1,k)]*idyCv[M_CV(i,j+1)] - v_face[IDX_V(i-1,j+1,k)]*idyCv[M_CV(i-1,j+1)]) +
         dx_dyBu[M_BU(i,j+1)]*(u_face[IDX_U(i,j+1,k)]*idxCu[M_CU(i,j+1)] - u_face[IDX_U(i,j,k)]*idxCu[M_CU(i,j)]));
    double D_S_face = 0.5*(D_S_S + D_S_N);
    double strain = sqrt(D_T_face*D_T_face + D_S_face*D_S_face);
    ah_face_x[IDX_U(i,j,k)] = fmin(ah_max, fmax(ah_bg, smag_scale*strain));
}
__global__ void kS_u_reuse(double *__restrict__ ah_face_x, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t ni = (size_t)(nx-1);
    if (t >= ni*(size_t)nz) return;
    int i = (int)(t % ni) + 2, k = (int)(t / ni) + 1;
    ah_face_x[IDX_U(i,1,k)] = ah_face_x[IDX_U(i,2,k)];
    ah_face_x[IDX_U(i,ny,k)] = ah_face_x[IDX_U(i,ny-1,k)];
}
__global__ void kS_u_bg(double *__restrict__ ah_face_x, double ah_bg, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)ny*(size_t)nz) return;
    int j = (int)(t % (size_t)ny) + 1, k = (int)(t / (size_t)ny) + 1;
    ah_face_x[IDX_U(1,j,k)] = ah_bg; ah_face_x[IDX_U(nx+1,j,k)] = ah_bg;
}
__global__ void kS_v(const double *__restrict__ u_face, const double *__restrict__ v_face,
                     const double *__restrict__ dxT, const double *__restrict__ dyT,
                     const double *__restrict__ idxT, const double *__restrict__ idyT,
                     const double *__restrict__ dy_dxBu, const double *__restrict__ dx_dyBu,
                     const double *__restrict__ idyCv, const double *__restrict__ idxCu,
                     const double *__restrict__ wet_q, double *__restrict__ ah_face_y,
                     double c_smag, double ah_bg, double ah_max, double ns, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t ni = (size_t)(nx-2), nj = (size_t)(ny-1);
    if (t >= ni*nj*(size_t)nz) return;
    int i = (int)(t % ni) + 2, j = (int)((t/ni) % nj) + 2, k = (int)(t/(ni*nj)) + 1;
    double s = c_smag*sqrt(dxT[M_T(i,j)]*dyT[M_T(i,j)]); double smag_scale = s*s;
    double D_T_S = (u_face[IDX_U(i+1,j-1,k)] - u_face[IDX_U(i,j-1,k)])*idxT[M_T(i,j-1)]
                 - (v_face[IDX_V(i,j,k)] - v_face[IDX_V(i,j-1,k)])*idyT[M_T(i,j-1)];
    double D_T_N = (u_face[IDX_U(i+1,j,k)] - u_face[IDX_U(i,j,k)])*idxT[M_T(i,j)]
                 - (v_face[IDX_V(i,j+1,k)] - v_face[IDX_V(i,j,k)])*idyT[M_T(i,j)];
    double D_T_face = 0.5*(D_T_S + D_T_N);
    double D_S_W = ((1.0 - 2.0*ns)*wet_q[M_BU(i,j)] + 2.0*ns)*
        (dy_dxBu[M_BU(i,j)]*(v_face[IDX_V(i,j,k)]*idyCv[M_CV(i,j)] - v_face[IDX_V(i-1,j,k)]*idyCv[M_CV(i-1,j)]) +
         dx_dyBu[M_BU(i,j)]*(u_face[IDX_U(i,j,k)]*idxCu[M_CU(i,j)] - u_face[IDX_U(i,j-1,k)]*idxCu[M_CU(i,j-1)]));
    double D_S_E = ((1.0 - 2.0*ns)*wet_q[M_BU(i+1,j)] + 2.0*ns)*
        (dy_dxBu[M_BU(i+1,j)]*(v_face[IDX_V(i+1,j,k)]*idyCv[M_CV(i+1,j)] - v_face[IDX_V(i,j,k)]*idyCv[M_CV(i,j)]) +
         dx_dyBu[M_BU(i+1,j)]*(u_face[IDX_U(i+1,j,k)]*idxCu[M_CU(i+1,j)] - u_face[IDX_U(i+1,j-1,k)]*idxCu[M_CU(i+1,j-1)]));
    double D_S_face = 0.5*(D_S_W + D_S_E);
    double strain = sqrt(D_T_face*D_T_face + D_S_face*D_S_face);
    ah_face_y[IDX_V(i,j,k)] = fmin(ah_max, fmax(ah_bg, smag_scale*strain));
}
__global__ void kS_v_reuse(double *__restrict__ ah_face_y, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t nj = (size_t)(ny-1);
    if (t >= nj*(size_t)nz) return;
    int j = (int)(t % nj) + 2, k = (int)(t / nj) + 1;
    ah_face_y[IDX_V(1,j,k)] = ah_face_y[IDX_V(2,j,k)];
    ah_face_y[IDX_V(nx,j,k)] = ah_face_y[IDX_V(nx-1,j,k)];
}
__global__ void kS_v_bg(double *__restrict__ ah_face_y, double ah_bg, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)nx*(size_t)nz) return;
    int i = (int)(t % (size_t)nx) + 1, k = (int)(t / (size_t)nx) + 1;
    ah_face_y[IDX_V(i,1,k)] = ah_bg; ah_face_y[IDX_V(i,ny+1,k)] = ah_bg;
}

// ---- u-face interior Laplacian (k, j=2:ny-1, i=2:nx) ----
__global__ void kH_u(const double *__restrict__ u_face, const double *__restrict__ ah_face_x,
                     const double *__restrict__ dy_dxT, const double *__restrict__ dx_dyBu,
                     const double *__restrict__ iareaCu, double *__restrict__ du_visc,
                     int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t ni = (size_t)(nx-1), nj = (size_t)(ny-2);
    if (t >= ni*nj*(size_t)nz) return;
    int i = (int)(t % ni) + 2;
    int j = (int)((t/ni) % nj) + 2;
    int k = (int)(t/(ni*nj)) + 1;
    double lap_u = iareaCu[M_CU(i,j)]*(
        (dy_dxT[M_T(i,j)]*(u_face[IDX_U(i+1,j,k)] - u_face[IDX_U(i,j,k)]) -
         dy_dxT[M_T(i-1,j)]*(u_face[IDX_U(i,j,k)] - u_face[IDX_U(i-1,j,k)])) +
        (dx_dyBu[M_BU(i,j+1)]*(u_face[IDX_U(i,j+1,k)] - u_face[IDX_U(i,j,k)]) -
         dx_dyBu[M_BU(i,j)]*(u_face[IDX_U(i,j,k)] - u_face[IDX_U(i,j-1,k)])));
    du_visc[IDX_U(i,j,k)] = ah_face_x[IDX_U(i,j,k)]*lap_u;
}
// ---- u boundaries: (k, j=1:ny) i-walls, and (k, i=1:nx+1) j-walls ----
__global__ void kH_u_bj(double *__restrict__ du_visc, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)ny*(size_t)nz) return;
    int j = (int)(t % (size_t)ny) + 1, k = (int)(t / (size_t)ny) + 1;
    du_visc[IDX_U(1,j,k)] = 0.0; du_visc[IDX_U(nx+1,j,k)] = 0.0;
}
__global__ void kH_u_bi(double *__restrict__ du_visc, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)(nx+1)*(size_t)nz) return;
    int i = (int)(t % (size_t)(nx+1)) + 1, k = (int)(t / (size_t)(nx+1)) + 1;
    du_visc[IDX_U(i,1,k)] = 0.0; du_visc[IDX_U(i,ny,k)] = 0.0;
}
// ---- v-face interior Laplacian (k, j=2:ny, i=2:nx-1) ----
__global__ void kH_v(const double *__restrict__ v_face, const double *__restrict__ ah_face_y,
                     const double *__restrict__ dx_dyT, const double *__restrict__ dy_dxBu,
                     const double *__restrict__ iareaCv, double *__restrict__ dv_visc,
                     int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t ni = (size_t)(nx-2), nj = (size_t)(ny-1);
    if (t >= ni*nj*(size_t)nz) return;
    int i = (int)(t % ni) + 2;
    int j = (int)((t/ni) % nj) + 2;
    int k = (int)(t/(ni*nj)) + 1;
    double lap_v = iareaCv[M_CV(i,j)]*(
        (dx_dyT[M_T(i,j)]*(v_face[IDX_V(i,j+1,k)] - v_face[IDX_V(i,j,k)]) -
         dx_dyT[M_T(i,j-1)]*(v_face[IDX_V(i,j,k)] - v_face[IDX_V(i,j-1,k)])) +
        (dy_dxBu[M_BU(i+1,j)]*(v_face[IDX_V(i+1,j,k)] - v_face[IDX_V(i,j,k)]) -
         dy_dxBu[M_BU(i,j)]*(v_face[IDX_V(i,j,k)] - v_face[IDX_V(i-1,j,k)])));
    dv_visc[IDX_V(i,j,k)] = ah_face_y[IDX_V(i,j,k)]*lap_v;
}
__global__ void kH_v_bi(double *__restrict__ dv_visc, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)nx*(size_t)nz) return;
    int i = (int)(t % (size_t)nx) + 1, k = (int)(t / (size_t)nx) + 1;
    dv_visc[IDX_V(i,1,k)] = 0.0; dv_visc[IDX_V(i,ny+1,k)] = 0.0;
}
__global__ void kH_v_bj(double *__restrict__ dv_visc, int nx, int ny, int nz) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)(ny+1)*(size_t)nz) return;
    int j = (int)(t % (size_t)(ny+1)) + 1, k = (int)(t / (size_t)(ny+1)) + 1;
    dv_visc[IDX_V(1,j,k)] = 0.0; dv_visc[IDX_V(nx,j,k)] = 0.0;
}

#define TPB 128
#define NBLK(n) (int)(((size_t)(n) + TPB - 1) / TPB)
static inline void msync(int sync, int level) {
    if (sync >= level) { cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); }
}

extern "C" void hvisc_compute_face_cuda_launch(
    const double *u_face, const double *v_face, const double *ah_face_x, const double *ah_face_y,
    double *du_visc, double *dv_visc,
    const double *dy_dxT, const double *dx_dyBu, const double *iareaCu,
    const double *dx_dyT, const double *dy_dxBu, const double *iareaCv,
    int nx, int ny, int nz, int sync) {
    kH_u   <<<NBLK((size_t)(nx-1)*(ny-2)*nz), TPB>>>(u_face, ah_face_x, dy_dxT, dx_dyBu, iareaCu, du_visc, nx, ny, nz); msync(sync,2);
    kH_u_bj<<<NBLK((size_t)ny*nz), TPB>>>(du_visc, nx, ny, nz); msync(sync,2);
    kH_u_bi<<<NBLK((size_t)(nx+1)*nz), TPB>>>(du_visc, nx, ny, nz); msync(sync,2);
    kH_v   <<<NBLK((size_t)(nx-2)*(ny-1)*nz), TPB>>>(v_face, ah_face_y, dx_dyT, dy_dxBu, iareaCv, dv_visc, nx, ny, nz); msync(sync,2);
    kH_v_bi<<<NBLK((size_t)nx*nz), TPB>>>(dv_visc, nx, ny, nz); msync(sync,2);
    kH_v_bj<<<NBLK((size_t)(ny+1)*nz), TPB>>>(dv_visc, nx, ny, nz); msync(sync,1);
}

extern "C" void hvisc_compute_smag_cuda_launch(
    const double *u_face, const double *v_face,
    const double *dxT, const double *dyT, const double *idxT, const double *idyT,
    const double *dy_dxBu, const double *dx_dyBu, const double *idyCv, const double *idxCu,
    const double *wet_q, double *ah_face_x, double *ah_face_y,
    double c_smag, double ah_bg, double ah_max, double ns, int nx, int ny, int nz, int sync) {
    kS_u      <<<NBLK((size_t)(nx-1)*(ny-2)*nz), TPB>>>(u_face,v_face,dxT,dyT,idxT,idyT,dy_dxBu,dx_dyBu,idyCv,idxCu,wet_q,ah_face_x,c_smag,ah_bg,ah_max,ns,nx,ny,nz); msync(sync,2);
    kS_u_reuse<<<NBLK((size_t)(nx-1)*nz), TPB>>>(ah_face_x, nx, ny, nz); msync(sync,2);
    kS_u_bg   <<<NBLK((size_t)ny*nz), TPB>>>(ah_face_x, ah_bg, nx, ny, nz); msync(sync,2);
    kS_v      <<<NBLK((size_t)(nx-2)*(ny-1)*nz), TPB>>>(u_face,v_face,dxT,dyT,idxT,idyT,dy_dxBu,dx_dyBu,idyCv,idxCu,wet_q,ah_face_y,c_smag,ah_bg,ah_max,ns,nx,ny,nz); msync(sync,2);
    kS_v_reuse<<<NBLK((size_t)(ny-1)*nz), TPB>>>(ah_face_y, nx, ny, nz); msync(sync,2);
    kS_v_bg   <<<NBLK((size_t)nx*nz), TPB>>>(ah_face_y, ah_bg, nx, ny, nz); msync(sync,1);
}
