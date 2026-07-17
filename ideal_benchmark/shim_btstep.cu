// shim_btstep.cu -- flat extern "C" wrapper around btstep_kernel.cu's FAITHFUL
// btstep_cuda_launch(BtArgs*, n_steps, mode). Mirrors the opt flat wrapper
// (btstep_opt_launch_flat in opt_kernel.cu): assemble the BtArgs struct from a
// uniform pointer + scalar list, then launch. mode=0 = faithful (11 kern/substep).
#include "btstep_args.h"   // struct BtArgs + btstep_cuda_launch prototype

extern "C" void btstep_cuda_flat(
    double *eta, double *eta_new, double *H_ref, double *ubt, double *vbt,
    double *ubt_prev, double *vbt_prev, double *rem_u, double *rem_v,
    double *zeta, double *ke, double *ubt_sum, double *vbt_sum, double *eta_sum,
    double *uhbt_sum, double *vhbt_sum,
    double *dy_cu, double *dx_cv, double *iareaT, double *areaCu, double *areaCv,
    double *dxCu, double *dyCv, double *idxCu, double *idyCv, double *iareaBu,
    double *f_corner, double *force_u, double *force_v,
    int nx, int ny, int nghost, int nx_phys, int ny_phys,
    double G, double bebt, double dt, int n_steps, int mode)
{
    BtArgs a;
    a.eta = eta; a.eta_new = eta_new; a.H_ref = H_ref;
    a.ubt = ubt; a.vbt = vbt; a.ubt_prev = ubt_prev; a.vbt_prev = vbt_prev;
    a.rem_u = rem_u; a.rem_v = rem_v; a.zeta = zeta; a.ke = ke;
    a.ubt_sum = ubt_sum; a.vbt_sum = vbt_sum; a.eta_sum = eta_sum;
    a.uhbt_sum = uhbt_sum; a.vhbt_sum = vhbt_sum;
    a.dy_cu = dy_cu; a.dx_cv = dx_cv; a.iareaT = iareaT;
    a.areaCu = areaCu; a.areaCv = areaCv; a.dxCu = dxCu; a.dyCv = dyCv;
    a.idxCu = idxCu; a.idyCv = idyCv; a.iareaBu = iareaBu;
    a.f_corner = f_corner; a.force_u = force_u; a.force_v = force_v;
    a.nx = nx; a.ny = ny; a.nghost = nghost; a.nx_phys = nx_phys; a.ny_phys = ny_phys;
    a.G = G; a.bebt = bebt; a.dt = dt;
    btstep_cuda_launch(&a, n_steps, mode);
}
