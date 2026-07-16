// Optimized barotropic substep CUDA. Best-practice pass over btstep_kernel.cu
// (the faithful 11-kernel-per-substep transliteration), freed from the
// "faithful transliteration" rule but held BIT-IDENTICAL to the faithful port
// (same per-cell arithmetic order). ab_main.cu checks max|d eta|, judged
// field-relative (bt_eta is divergence-driven, interior ~0).
//
// The optimization is TWO changes, both "remove genuine waste", exactly the
// pair that won for continuity_layered:
//
//   1. FUSE 11 kernels/substep -> 5. Walls and the eta/u/v/-sum accumulators
//      fold into their producers as single-assignment merges
//      (`ubt = wall ? 0 : computed; ubt_sum += ubt`). Same values, same order,
//      so still bit-identical -- the merges just remove 6 separate launches and
//      their global round-trips per substep (n_inner=24 -> 264 launches become
//      120). This is the same fusion btstep_kernel.cu offers as MODE_FUSED; it
//      is reproduced here so the optimized artifact is standalone.
//
//   2. 32-bit indexing. The faithful port addresses with size_t (safe for any
//      grid); every realistic barotropic domain fits int32 (479*303 = 145k <<
//      2^31), so the size_t address math is pure overhead. Fewer address
//      registers, cheaper indexing. Numerically INERT -- only the address
//      computation changes, never the double arithmetic -- so still bit-exact.
//
//   3. (opt_mode 1) capture the whole n_steps substep loop into ONE cudaGraph
//      and replay it: the 120 per-call launches are issued once at capture and
//      replayed with ~0 host cost thereafter. Same kernels, same work, same
//      values -- only the launch overhead moves out of the timed loop. This is
//      the CUDA-only lever `do concurrent` cannot express.
//
// WHAT CANNOT BE FUSED FURTHER (checked, not assumed -- same structure the
// faithful port documents):
//   * Pass 2b -> Pass 2c is SEQUENTIAL: 2c's u_at_v reads the ubt that 2b just
//     wrote (Gauss-Seidel, not Jacobi). Fusing them changes the answer.
//   * Pass 1 -> eta-swap -> Pass 2b are separated by real barriers: Pass 2b
//     reads eta/zeta NEIGHBOURS, so every producer must have landed first.
//   The 5-kernel floor is that dependency structure, not an implementation limit.
#include <cstdio>
#include "gpu_rt.h"
#include "btstep_args.h"

// 32-bit indexing (default). Valid while every stride-class count < 2^31 --
// true for any realistic barotropic grid. The doubles read/written are
// byte-identical to the faithful port's size_t path; only the address
// arithmetic differs. Build -DIDX32=0 for the size_t path (isolates the win).
#ifndef IDX32
#define IDX32 1
#endif
#if IDX32
#define IT(i, j) ((int)((i)-1) + (nx)  *(int)((j)-1))
#define IU(i, j) ((int)((i)-1) + (nx+1)*(int)((j)-1))
#define IV(i, j) ((int)((i)-1) + (nx)  *(int)((j)-1))
#define IB(i, j) ((int)((i)-1) + (nx+1)*(int)((j)-1))
#else
#define IT(i, j) ((size_t)((i)-1) + (size_t)(nx)  *(size_t)((j)-1))
#define IU(i, j) ((size_t)((i)-1) + (size_t)(nx+1)*(size_t)((j)-1))
#define IV(i, j) ((size_t)((i)-1) + (size_t)(nx)  *(size_t)((j)-1))
#define IB(i, j) ((size_t)((i)-1) + (size_t)(nx+1)*(size_t)((j)-1))
#endif

// ---- prev-save (2 Fortran loops; 1 kernel -- disjoint arrays) --------------
__global__ void ko_prev(BtArgs a) {
    int nx = a.nx, ny = a.ny;
    int t = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    int nu = (nx+1)*ny, nv = nx*(ny+1);
    if (t < nu)         { int i = t % (nx+1) + 1, j = t/(nx+1) + 1; a.ubt_prev[IU(i,j)] = a.ubt[IU(i,j)]; }
    else if (t < nu+nv) { int s = t-nu; int i = s % nx + 1, j = s/nx + 1; a.vbt_prev[IV(i,j)] = a.vbt[IV(i,j)]; }
}

// ---- Pass 1: eta update + uhbt/vhbt accumulate + KE + interior zeta --------
__global__ void ko_pass1(BtArgs a) {
    int nx = a.nx, ny = a.ny;
    const double bebt = a.bebt, dt = a.dt;
    int t = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    if (t >= nx*ny) return;
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

// ---- eta-swap + eta_sum + zeta wall closure (fused) ------------------------
__global__ void ko_swap(BtArgs a) {
    int nx = a.nx, ny = a.ny;
    int t = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    if (t >= (nx+1)*(ny+1)) return;
    int i = t % (nx+1) + 1, j = t/(nx+1) + 1;
    if (i <= nx && j <= ny) {
        a.eta[IT(i,j)] = a.eta_new[IT(i,j)];
        a.eta_sum[IT(i,j)] += a.eta[IT(i,j)];
    }
    if (i == 1 || i == nx+1) a.zeta[IB(i,j)] = 0.0;
    if (j == 1 || j == ny+1) a.zeta[IB(i,j)] = 0.0;
    if (i == a.nghost+1)              a.zeta[IB(i,j)] = 0.0;
    if (i == a.nghost+a.nx_phys+1)    a.zeta[IB(i,j)] = 0.0;
    if (j == a.nghost+1)              a.zeta[IB(i,j)] = 0.0;
    if (j == a.nghost+a.ny_phys+1)    a.zeta[IB(i,j)] = 0.0;
}

// ---- Pass 2b: u update + walls + ubt_sum (fused) ---------------------------
__global__ void ko_pass2b(BtArgs a) {
    int nx = a.nx, ny = a.ny;
    const double G = a.G, dt = a.dt;
    int t = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    if (t >= (nx+1)*ny) return;
    int i = t % (nx+1) + 1, j = t/(nx+1) + 1;
    bool is_wall = (i == 1) || (i == nx+1) || (i == a.nghost+1) || (i == a.nghost+a.nx_phys+1);
    if (is_wall) {
        a.ubt[IU(i,j)] = 0.0;
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
    a.ubt_sum[IU(i,j)] += a.ubt[IU(i,j)];
}

// ---- Pass 2c: v update + walls + vbt_sum (fused) ---------------------------
__global__ void ko_pass2c(BtArgs a) {
    int nx = a.nx, ny = a.ny;
    const double G = a.G, dt = a.dt;
    int t = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    if (t >= nx*(ny+1)) return;
    int i = t % nx + 1, j = t/nx + 1;
    bool is_wall = (j == 1) || (j == ny+1) || (j == a.nghost+1) || (j == a.nghost+a.ny_phys+1);
    if (is_wall) {
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
    a.vbt_sum[IV(i,j)] += a.vbt[IV(i,j)];
}

__global__ void ko_zero(double * __restrict__ p, int n) {
    int t = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    if (t < n) p[t] = 0.0;
}

#define TPB 128
#define NB(n) (int)(((size_t)(n) + TPB - 1) / TPB)

static void issue_substep(const BtArgs &a, cudaStream_t s) {
    int nx = a.nx, ny = a.ny;
    if (a.bebt > 0.0)
        ko_prev<<<NB((nx+1)*ny + nx*(ny+1)), TPB, 0, s>>>(a);
    ko_pass1 <<<NB(nx*ny),         TPB, 0, s>>>(a);
    ko_swap  <<<NB((nx+1)*(ny+1)), TPB, 0, s>>>(a);
    ko_pass2b<<<NB((nx+1)*ny),     TPB, 0, s>>>(a);
    ko_pass2c<<<NB(nx*(ny+1)),     TPB, 0, s>>>(a);
}

// opt_mode: 0 = fused (5 kern/substep, idx32);  1 = fused captured in a cudaGraph.
extern "C" void btstep_opt_launch(BtArgs *ap, int n_steps, int opt_mode) {
    BtArgs a = *ap;
    int nx = a.nx, ny = a.ny;
    static cudaGraphExec_t gexec = nullptr;
    static int g_steps = -1, g_nx = -1, g_ny = -1;

    ko_zero<<<NB(nx*ny),      TPB>>>(a.eta_sum,  nx*ny);
    ko_zero<<<NB((nx+1)*ny),  TPB>>>(a.ubt_sum,  (nx+1)*ny);
    ko_zero<<<NB((nx+1)*ny),  TPB>>>(a.uhbt_sum, (nx+1)*ny);
    ko_zero<<<NB(nx*(ny+1)),  TPB>>>(a.vbt_sum,  nx*(ny+1));
    ko_zero<<<NB(nx*(ny+1)),  TPB>>>(a.vhbt_sum, nx*(ny+1));

    if (opt_mode == 1) {
        if (!gexec || g_steps != n_steps || g_nx != nx || g_ny != ny) {
            if (gexec) { cudaGraphExecDestroy(gexec); gexec = nullptr; }
            cudaStream_t cap;
            cudaStreamCreate(&cap);
            cudaGraph_t graph;
            cudaStreamBeginCapture(cap, cudaStreamCaptureModeGlobal);
            for (int n = 0; n < n_steps; ++n) issue_substep(a, cap);
            cudaStreamEndCapture(cap, &graph);
            cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0);
            cudaGraphDestroy(graph);
            cudaStreamDestroy(cap);
            g_steps = n_steps; g_nx = nx; g_ny = ny;
        }
        cudaGraphLaunch(gexec, 0);
    } else {
        for (int n = 0; n < n_steps; ++n) issue_substep(a, 0);
    }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) fprintf(stderr, "CUDA btstep_opt: %s\n", cudaGetErrorString(e));
}
