// Shared ABI + memory layout for the MEKE CUDA port.
//
// WHY THIS FILE EXISTS: `struct MekeArgs` and the index macros are the contract
// between meke_kernel.cu (the kernels) and its callers (meke_bench.F90 via
// bind(C), and meke_native.cu). They were previously defined inside
// meke_kernel.cu, so the native driver would have had to COPY them. A copy that
// drifted -- one reordered field, one wrong stride class -- would silently
// corrupt the comparison rather than fail to compile, which is the same failure
// mode the "do not duplicate the kernel" rule exists to prevent. One definition,
// two includers.
//
// ⚠ FIELD ORDER IS LOAD-BEARING: `type, bind(C) :: meke_args_t` in
// meke_bench.F90:51-64 mirrors this struct field-for-field. Changing the order
// here without changing it there is undefined behaviour, not a compile error.
//
// INDEXING: the device arrays are Fortran allocations (column-major, 1-based).
// Three stride classes -- mixing them is silent corruption, not a compile error:
//     meke, kh_diff, le, ku, i_mass, depth_tot, bottom_fac2, barotr_fac2,
//       src, mass_ws, u_bbl2, ke_diss_ws, rd_ws, f_centre, gm_src,
//       areaT, iareaT, idxT, idyT                       (nx,   ny)   -> IT
//     uflux, sn_u_ws, dy_cu, idxCu                      (nx+1, ny)   -> IU
//     vflux, sn_v_ws, dx_cv, idyCv                      (nx,   ny+1) -> IV
//     h_layer, rho_layer                                (nx, ny, nz) -> I3
#ifndef MEKE_ARGS_H
#define MEKE_ARGS_H

#define IT(i, j) ((size_t)((i)-1) + (size_t)(nx)  *(size_t)((j)-1))
#define IU(i, j) ((size_t)((i)-1) + (size_t)(nx+1)*(size_t)((j)-1))
#define IV(i, j) ((size_t)((i)-1) + (size_t)(nx)  *(size_t)((j)-1))
#define I3(i, j, k) ((size_t)((i)-1) + (size_t)(nx)*((size_t)((j)-1) + (size_t)(ny)*(size_t)((k)-1)))

#define MASS_NEGLECT 1.0e-30
#define H_VANISHED   1.5e-4

struct MekeArgs {
    // workspaces / prognostic
    double * __restrict__ meke, * __restrict__ kh_diff, * __restrict__ le, * __restrict__ ku, * __restrict__ i_mass, * __restrict__ depth_tot, * __restrict__ bottom_fac2;
    double * __restrict__ barotr_fac2, * __restrict__ src, * __restrict__ uflux, * __restrict__ vflux, * __restrict__ mass_ws, * __restrict__ rd_ws;
    double * __restrict__ sn_u_ws, * __restrict__ sn_v_ws, * __restrict__ ke_diss_ws;
    // read-only inputs
    const double * __restrict__ u_bbl2, * __restrict__ f_centre, * __restrict__ gm_src, * __restrict__ ke_diss_ext;
    const double * __restrict__ h_layer, * __restrict__ rho_layer;
    const double * __restrict__ areaT, * __restrict__ iareaT, * __restrict__ idxT, * __restrict__ idyT, * __restrict__ dy_cu, * __restrict__ dx_cv, * __restrict__ idxCu, * __restrict__ idyCv;
    int nx, ny, nz;
    // config scalars (mirror ocean_meke_t)
    double dt, dtscale, cd_scale, cb, ct, min_gamma2, cdrag, uscale;
    double a_deform, a_rhines, a_eady, a_frict, a_grid;
    double bgsrc, gmcoeff, frcoeff, damping;
    double kh_bg, k4, khmeke_fac, khcoeff, visc_coeff_ku, rho0;
    int backscatter;
};

#ifdef __cplusplus
extern "C" {
#endif
// mode: 0 = faithful (16 kernels), 1 = fused (6), 2 = graph (6 in a cudaGraph).
// Each call is synchronous on return (cudaDeviceSynchronize / stream sync).
void meke_cuda_launch(const MekeArgs *ap, int mode);
void meke_cuda_graph_reset(void);
#ifdef __cplusplus
}
#endif

#endif // MEKE_ARGS_H
