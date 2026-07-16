// Shared ABI for the barotropic substep CUDA port: the argument struct and the
// launch entry point. Nothing else -- no kernels.
//
// WHY THIS FILE EXISTS: `struct BtArgs` used to live inside btstep_kernel.cu.
// It now has TWO C-side callers (btstep_kernel.cu itself and btstep_native.cu)
// plus a Fortran mirror (`type, bind(C) :: bt_args_t` in btstep_bench.F90).
// Field order is load-bearing and a mismatch is NOT a compile error -- it is
// silent corruption. One definition, included everywhere, removes that hazard
// for the C side; the Fortran mirror is unavoidable and must be kept in sync
// by hand.
//
// INDEXING (see btstep_kernel.cu for the full note): the arrays are plain
// column-major, 1-based Fortran allocations. Four stride classes:
//     (nx,   ny)    eta, eta_new, H_ref, ke, eta_sum, iareaT
//     (nx+1, ny)    ubt, ubt_prev, rem_u, ubt_sum, uhbt_sum, dy_cu, areaCu,
//                     dxCu, idxCu, force_u
//     (nx,   ny+1)  vbt, vbt_prev, rem_v, vbt_sum, vhbt_sum, dx_cv, areaCv,
//                     dyCv, idyCv, force_v
//     (nx+1, ny+1)  zeta, f_corner, iareaBu
#ifndef BTSTEP_ARGS_H
#define BTSTEP_ARGS_H

struct BtArgs {
    double * __restrict__ eta, * __restrict__ eta_new, * __restrict__ H_ref, * __restrict__ ubt, * __restrict__ vbt, * __restrict__ ubt_prev, * __restrict__ vbt_prev;
    double * __restrict__ rem_u, * __restrict__ rem_v, * __restrict__ zeta, * __restrict__ ke, * __restrict__ ubt_sum, * __restrict__ vbt_sum, * __restrict__ eta_sum;
    double * __restrict__ uhbt_sum, * __restrict__ vhbt_sum;
    const double * __restrict__ dy_cu, * __restrict__ dx_cv, * __restrict__ iareaT, * __restrict__ areaCu, * __restrict__ areaCv, * __restrict__ dxCu, * __restrict__ dyCv;
    const double * __restrict__ idxCu, * __restrict__ idyCv, * __restrict__ iareaBu, * __restrict__ f_corner, * __restrict__ force_u, * __restrict__ force_v;
    int nx, ny, nghost, nx_phys, ny_phys;
    double G, bebt, dt;
};

// mode: 0 = faithful (11 kern/substep), 1 = fused (5), 2 = fused captured in a
// cudaGraph. Runs n_steps substeps and synchronises before returning.
extern "C" void btstep_cuda_launch(BtArgs *ap, int n_steps, int mode);

#endif // BTSTEP_ARGS_H
