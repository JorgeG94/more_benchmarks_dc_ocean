// CUDA port of the CLOSED-BASIN barotropic substep (btstep.F90).
//
// THREE VARIANTS, because they answer different questions:
//   MODE_FAITHFUL (0) : 11 kernels/substep, one per `do concurrent` loop, same
//                       order. Measures nvfortran's CODEGEN against nvcc.
//   MODE_FUSED    (1) :  5 kernels/substep. Walls and accumulators fold into
//                       their producers as single-assignment merges
//                       (`ubt = wall ? 0 : computed; ubt_sum += ubt`) -- same
//                       values, same order, so still bit-identical. Measures
//                       what FEWER LAUNCHES buys.
//   MODE_GRAPH    (2) : the fused kernels captured into a cudaGraph and
//                       replayed. Same work, same kernel count, but the launch
//                       cost is paid once at capture. Measures how much of the
//                       remaining time is launch overhead -- something
//                       `do concurrent` cannot do at all.
//
// WHAT CANNOT BE FUSED (checked, not assumed):
//   * Pass 2b -> Pass 2c is SEQUENTIAL: 2c's `u_at_v` reads the `bt_ubt` that
//     2b just wrote (Gauss-Seidel, not Jacobi). Fusing them would change the
//     answer.
//   * Pass 1 -> eta-swap -> Pass 2b are separated by real barriers: Pass 2b
//     reads eta/zeta NEIGHBOURS, so every thread's producer must have landed.
//   The 5-kernel floor is that dependency structure, not an implementation limit.
//
// INDEXING: the device arrays ARE the Fortran allocations (host_data
// use_device). Column-major, 1-based. Four stride classes -- mixing them is
// silent corruption, not a compile error:
//     eta, eta_new, H_ref, ke_centre, iareaT   (nx,   ny)   -> IT
//     ubt, ubt_prev, rem_u, ubt_sum, uhbt_sum,
//       dy_cu, areaCu, dxCu, idxCu, force_u    (nx+1, ny)   -> IU
//     vbt, vbt_prev, rem_v, vbt_sum, vhbt_sum,
//       dx_cv, areaCv, dyCv, idyCv, force_v    (nx,   ny+1) -> IV
//     zeta_corner, f_corner, iareaBu           (nx+1, ny+1) -> IB
#include <cstdio>
#include <cuda_runtime.h>

// `struct BtArgs` and the btstep_cuda_launch() prototype. Shared verbatim with
// btstep_native.cu so the two callers cannot drift apart -- see the header.
#include "btstep_args.h"

#define IT(i, j) ((size_t)((i)-1) + (size_t)(nx)  *(size_t)((j)-1))
#define IU(i, j) ((size_t)((i)-1) + (size_t)(nx+1)*(size_t)((j)-1))
#define IV(i, j) ((size_t)((i)-1) + (size_t)(nx)  *(size_t)((j)-1))
#define IB(i, j) ((size_t)((i)-1) + (size_t)(nx+1)*(size_t)((j)-1))

// ---- prev-save (2 loops in Fortran; 1 kernel here -- disjoint arrays) ----
__global__ void k_prev(BtArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t nu = (size_t)(nx+1)*ny, nv = (size_t)nx*(ny+1);
    if (t < nu)      { int i = t % (nx+1) + 1, j = t/(nx+1) + 1; a.ubt_prev[IU(i,j)] = a.ubt[IU(i,j)]; }
    else if (t < nu+nv) { size_t s = t-nu; int i = s % nx + 1, j = s/nx + 1; a.vbt_prev[IV(i,j)] = a.vbt[IV(i,j)]; }
}
__global__ void k_prev_u(BtArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)(nx+1)*ny) return;
    int i = t % (nx+1) + 1, j = t/(nx+1) + 1;
    a.ubt_prev[IU(i,j)] = a.ubt[IU(i,j)];
}
__global__ void k_prev_v(BtArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)nx*(ny+1)) return;
    int i = t % nx + 1, j = t/nx + 1;
    a.vbt_prev[IV(i,j)] = a.vbt[IV(i,j)];
}

// ---- Pass 1: eta update + uhbt/vhbt accumulate + KE + interior zeta ----
__global__ void k_pass1(BtArgs a) {
    int nx = a.nx, ny = a.ny;
    const double bebt = a.bebt, dt = a.dt;
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)nx*ny) return;
    int i = t % nx + 1, j = t/nx + 1;
    double h_face_E, h_face_W, h_face_N, h_face_S;
    if (i < nx) h_face_E = 0.5*((a.H_ref[IT(i,j)] + a.eta[IT(i,j)]) + (a.H_ref[IT(i+1,j)] + a.eta[IT(i+1,j)]));
    else        h_face_E = a.H_ref[IT(i,j)] + a.eta[IT(i,j)];
    if (i > 1)  h_face_W = 0.5*((a.H_ref[IT(i-1,j)] + a.eta[IT(i-1,j)]) + (a.H_ref[IT(i,j)] + a.eta[IT(i,j)]));
    else        h_face_W = a.H_ref[IT(i,j)] + a.eta[IT(i,j)];
    if (j < ny) h_face_N = 0.5*((a.H_ref[IT(i,j)] + a.eta[IT(i,j)]) + (a.H_ref[IT(i,j+1)] + a.eta[IT(i,j+1)]));
    else        h_face_N = a.H_ref[IT(i,j)] + a.eta[IT(i,j)];
    if (j > 1)  h_face_S = 0.5*((a.H_ref[IT(i,j-1)] + a.eta[IT(i,j-1)]) + (a.H_ref[IT(i,j)] + a.eta[IT(i,j)]));
    else        h_face_S = a.H_ref[IT(i,j)] + a.eta[IT(i,j)];
    double ubt_R = (1.0+bebt)*a.ubt[IU(i+1,j)] - bebt*a.ubt_prev[IU(i+1,j)];
    double ubt_L = (1.0+bebt)*a.ubt[IU(i,j)]   - bebt*a.ubt_prev[IU(i,j)];
    double vbt_N = (1.0+bebt)*a.vbt[IV(i,j+1)] - bebt*a.vbt_prev[IV(i,j+1)];
    double vbt_S = (1.0+bebt)*a.vbt[IV(i,j)]   - bebt*a.vbt_prev[IV(i,j)];
    double flux_x_R = h_face_E*ubt_R*a.dy_cu[IU(i+1,j)];
    double flux_x_L = h_face_W*ubt_L*a.dy_cu[IU(i,j)];
    double flux_y_N = h_face_N*vbt_N*a.dx_cv[IV(i,j+1)];
    double flux_y_S = h_face_S*vbt_S*a.dx_cv[IV(i,j)];
    double div_h_u = ((flux_x_R - flux_x_L) + (flux_y_N - flux_y_S))*a.iareaT[IT(i,j)];
    a.eta_new[IT(i,j)] = a.eta[IT(i,j)] - dt*div_h_u;
    // cell (i,j) owns its east/north face -> race-free
    a.uhbt_sum[IU(i+1,j)] += flux_x_R;
    a.vhbt_sum[IV(i,j+1)] += flux_y_N;
    a.ke[IT(i,j)] = 0.25*a.iareaT[IT(i,j)]*(
        a.areaCu[IU(i,j)]*a.ubt[IU(i,j)]*a.ubt[IU(i,j)] +
        a.areaCu[IU(i+1,j)]*a.ubt[IU(i+1,j)]*a.ubt[IU(i+1,j)] +
        a.areaCv[IV(i,j)]*a.vbt[IV(i,j)]*a.vbt[IV(i,j)] +
        a.areaCv[IV(i,j+1)]*a.vbt[IV(i,j+1)]*a.vbt[IV(i,j+1)]);
    if (i >= 2 && j >= 2)
        a.zeta[IB(i,j)] = ((a.vbt[IV(i,j)]*a.dyCv[IV(i,j)] - a.vbt[IV(i-1,j)]*a.dyCv[IV(i-1,j)]) -
                           (a.ubt[IU(i,j)]*a.dxCu[IU(i,j)] - a.ubt[IU(i,j-1)]*a.dxCu[IU(i,j-1)]))*a.iareaBu[IB(i,j)];
}

// ---- eta-swap + zeta wall closure  (FUSED variant also does eta_sum) ----
template<bool FUSE_SUM>
__global__ void k_swap(BtArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)(nx+1)*(ny+1)) return;
    int i = t % (nx+1) + 1, j = t/(nx+1) + 1;
    if (i <= nx && j <= ny) {
        a.eta[IT(i,j)] = a.eta_new[IT(i,j)];
        // eta is final for the substep here (Pass 2 does not write it), so
        // accumulating now is identical to accumulating at the end.
        if (FUSE_SUM) a.eta_sum[IT(i,j)] += a.eta[IT(i,j)];
    }
    if (i == 1 || i == nx+1) a.zeta[IB(i,j)] = 0.0;
    if (j == 1 || j == ny+1) a.zeta[IB(i,j)] = 0.0;
    if (i == a.nghost+1)              a.zeta[IB(i,j)] = 0.0;
    if (i == a.nghost+a.nx_phys+1)    a.zeta[IB(i,j)] = 0.0;
    if (j == a.nghost+1)              a.zeta[IB(i,j)] = 0.0;
    if (j == a.nghost+a.ny_phys+1)    a.zeta[IB(i,j)] = 0.0;
}

// ---- Pass 2b: u update (FUSED variant also does walls + ubt_sum) ----
template<bool FUSE_WALL>
__global__ void k_pass2b(BtArgs a) {
    int nx = a.nx, ny = a.ny;
    const double G = a.G, dt = a.dt;
    size_t span = FUSE_WALL ? (size_t)(nx+1)*ny : (size_t)(nx-1)*ny;
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= span) return;
    int i, j;
    if (FUSE_WALL) { i = t % (nx+1) + 1; j = t/(nx+1) + 1; }
    else           { i = t % (nx-1) + 2; j = t/(nx-1) + 1; }
    bool is_wall = (i == 1) || (i == nx+1) || (i == a.nghost+1) || (i == a.nghost+a.nx_phys+1);
    if (!FUSE_WALL && is_wall) { /* faithful mode: walls are a separate kernel */ }
    if (FUSE_WALL && is_wall) {
        a.ubt[IU(i,j)] = 0.0;                       // single-assignment merge
    } else if (i >= 2 && i <= nx) {
        double zeta_at_u = 0.5*(a.zeta[IB(i,j)] + a.zeta[IB(i,j+1)]);
        double f_at_u    = 0.5*(a.f_corner[IB(i,j)] + a.f_corner[IB(i,j+1)]);
        double v_at_u;
        if (j > 1 && j < ny) v_at_u = 0.25*(a.vbt[IV(i-1,j)] + a.vbt[IV(i-1,j+1)] + a.vbt[IV(i,j)] + a.vbt[IV(i,j+1)]);
        else if (j == 1)     v_at_u = 0.5*(a.vbt[IV(i-1,j+1)] + a.vbt[IV(i,j+1)]);
        else                 v_at_u = 0.5*(a.vbt[IV(i-1,j)] + a.vbt[IV(i,j)]);
        double ke_grad_x = (a.ke[IT(i,j)] - a.ke[IT(i-1,j)])*a.idxCu[IU(i,j)];
        double d_eta = a.eta[IT(i,j)] - a.eta[IT(i-1,j)];
        a.ubt[IU(i,j)] = a.rem_u[IU(i,j)]*(a.ubt[IU(i,j)] + dt*((zeta_at_u + f_at_u)*v_at_u
                          - G*d_eta*a.idxCu[IU(i,j)] - ke_grad_x + a.force_u[IU(i,j)]));
    }
    if (FUSE_WALL) a.ubt_sum[IU(i,j)] += a.ubt[IU(i,j)];
}
__global__ void k_wall_u(BtArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)ny) return;
    int j = (int)t + 1;
    a.ubt[IU(1,j)] = 0.0;
    a.ubt[IU(nx+1,j)] = 0.0;
    a.ubt[IU(a.nghost+1,j)] = 0.0;
    a.ubt[IU(a.nghost+a.nx_phys+1,j)] = 0.0;
}

// ---- Pass 2c: v update (FUSED variant also does walls + vbt_sum) ----
template<bool FUSE_WALL>
__global__ void k_pass2c(BtArgs a) {
    int nx = a.nx, ny = a.ny;
    const double G = a.G, dt = a.dt;
    size_t span = FUSE_WALL ? (size_t)nx*(ny+1) : (size_t)nx*(ny-1);
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= span) return;
    int i = t % nx + 1, j = FUSE_WALL ? (int)(t/nx) + 1 : (int)(t/nx) + 2;
    bool is_wall = (j == 1) || (j == ny+1) || (j == a.nghost+1) || (j == a.nghost+a.ny_phys+1);
    if (FUSE_WALL && is_wall) {
        a.vbt[IV(i,j)] = 0.0;
    } else if (j >= 2 && j <= ny) {
        double zeta_at_v = 0.5*(a.zeta[IB(i,j)] + a.zeta[IB(i+1,j)]);
        double f_at_v    = 0.5*(a.f_corner[IB(i,j)] + a.f_corner[IB(i+1,j)]);
        double u_at_v;
        if (i > 1 && i < nx) u_at_v = 0.25*(a.ubt[IU(i,j-1)] + a.ubt[IU(i+1,j-1)] + a.ubt[IU(i,j)] + a.ubt[IU(i+1,j)]);
        else if (i == 1)     u_at_v = 0.5*(a.ubt[IU(i+1,j-1)] + a.ubt[IU(i+1,j)]);
        else                 u_at_v = 0.5*(a.ubt[IU(i,j-1)] + a.ubt[IU(i,j)]);
        double ke_grad_y = (a.ke[IT(i,j)] - a.ke[IT(i,j-1)])*a.idyCv[IV(i,j)];
        double d_eta = a.eta[IT(i,j)] - a.eta[IT(i,j-1)];
        a.vbt[IV(i,j)] = a.rem_v[IV(i,j)]*(a.vbt[IV(i,j)] + dt*(-(zeta_at_v + f_at_v)*u_at_v
                          - G*d_eta*a.idyCv[IV(i,j)] - ke_grad_y + a.force_v[IV(i,j)]));
    }
    if (FUSE_WALL) a.vbt_sum[IV(i,j)] += a.vbt[IV(i,j)];
}
__global__ void k_wall_v(BtArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)nx) return;
    int i = (int)t + 1;
    a.vbt[IV(i,1)] = 0.0;
    a.vbt[IV(i,ny+1)] = 0.0;
    a.vbt[IV(i,a.nghost+1)] = 0.0;
    a.vbt[IV(i,a.nghost+a.ny_phys+1)] = 0.0;
}

// ---- accumulators (faithful mode only; fused mode folds them in) ----
__global__ void k_acc_eta(BtArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)nx*ny) return;
    int i = t % nx + 1, j = t/nx + 1;
    a.eta_sum[IT(i,j)] += a.eta[IT(i,j)];
}
__global__ void k_acc_u(BtArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)(nx+1)*ny) return;
    int i = t % (nx+1) + 1, j = t/(nx+1) + 1;
    a.ubt_sum[IU(i,j)] += a.ubt[IU(i,j)];
}
__global__ void k_acc_v(BtArgs a) {
    int nx = a.nx, ny = a.ny;
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (size_t)nx*(ny+1)) return;
    int i = t % nx + 1, j = t/nx + 1;
    a.vbt_sum[IV(i,j)] += a.vbt[IV(i,j)];
}
__global__ void k_zero(double * __restrict__ p, size_t n) {
    size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (t < n) p[t] = 0.0;
}

#define TPB 128
#define NB(n) (int)(((size_t)(n) + TPB - 1) / TPB)

static void issue_substep(const BtArgs &a, int mode, cudaStream_t s) {
    int nx = a.nx, ny = a.ny;
    if (a.bebt > 0.0) {
        if (mode == 0) {
            k_prev_u<<<NB((size_t)(nx+1)*ny), TPB, 0, s>>>(a);
            k_prev_v<<<NB((size_t)nx*(ny+1)), TPB, 0, s>>>(a);
        } else {
            k_prev<<<NB((size_t)(nx+1)*ny + (size_t)nx*(ny+1)), TPB, 0, s>>>(a);
        }
    }
    k_pass1<<<NB((size_t)nx*ny), TPB, 0, s>>>(a);
    if (mode == 0) {
        k_swap<false><<<NB((size_t)(nx+1)*(ny+1)), TPB, 0, s>>>(a);
        k_pass2b<false><<<NB((size_t)(nx-1)*ny), TPB, 0, s>>>(a);
        k_wall_u<<<NB(ny), TPB, 0, s>>>(a);
        k_pass2c<false><<<NB((size_t)nx*(ny-1)), TPB, 0, s>>>(a);
        k_wall_v<<<NB(nx), TPB, 0, s>>>(a);
        k_acc_eta<<<NB((size_t)nx*ny), TPB, 0, s>>>(a);
        k_acc_u<<<NB((size_t)(nx+1)*ny), TPB, 0, s>>>(a);
        k_acc_v<<<NB((size_t)nx*(ny+1)), TPB, 0, s>>>(a);
    } else {
        k_swap<true><<<NB((size_t)(nx+1)*(ny+1)), TPB, 0, s>>>(a);
        k_pass2b<true><<<NB((size_t)(nx+1)*ny), TPB, 0, s>>>(a);
        k_pass2c<true><<<NB((size_t)nx*(ny+1)), TPB, 0, s>>>(a);
    }
}

extern "C" void btstep_cuda_launch(BtArgs *ap, int n_steps, int mode) {
    BtArgs a = *ap;
    int nx = a.nx, ny = a.ny;
    static cudaGraphExec_t gexec = nullptr;
    static int g_steps = -1, g_nx = -1, g_ny = -1;

    k_zero<<<NB((size_t)nx*ny), TPB>>>(a.eta_sum, (size_t)nx*ny);
    k_zero<<<NB((size_t)(nx+1)*ny), TPB>>>(a.ubt_sum, (size_t)(nx+1)*ny);
    k_zero<<<NB((size_t)(nx+1)*ny), TPB>>>(a.uhbt_sum, (size_t)(nx+1)*ny);
    k_zero<<<NB((size_t)nx*(ny+1)), TPB>>>(a.vbt_sum, (size_t)nx*(ny+1));
    k_zero<<<NB((size_t)nx*(ny+1)), TPB>>>(a.vhbt_sum, (size_t)nx*(ny+1));

    if (mode == 2) {
        // Capture the whole n_steps substep loop into ONE graph. The work is
        // identical to MODE_FUSED; only the launch cost changes. `do concurrent`
        // has no equivalent -- this is the CUDA-only lever.
        if (!gexec || g_steps != n_steps || g_nx != nx || g_ny != ny) {
            if (gexec) { cudaGraphExecDestroy(gexec); gexec = nullptr; }
            cudaStream_t cap;
            cudaStreamCreate(&cap);
            cudaGraph_t graph;
            cudaStreamBeginCapture(cap, cudaStreamCaptureModeGlobal);
            for (int n = 0; n < n_steps; ++n) issue_substep(a, 1, cap);
            cudaStreamEndCapture(cap, &graph);
            cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0);
            cudaGraphDestroy(graph);
            cudaStreamDestroy(cap);
            g_steps = n_steps; g_nx = nx; g_ny = ny;
        }
        cudaGraphLaunch(gexec, 0);
    } else {
        for (int n = 0; n < n_steps; ++n) issue_substep(a, mode, 0);
    }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) fprintf(stderr, "CUDA btstep: %s\n", cudaGetErrorString(e));
}
