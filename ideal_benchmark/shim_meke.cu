// shim_meke.cu -- flat extern "C" wrapper around meke_kernel.cu's FAITHFUL
// meke_cuda_launch(MekeArgs*, mode). Mirrors the opt flat wrapper
// (meke_opt_launch_flat in opt_kernel.cu): assemble MekeArgs from a uniform
// pointer + scalar list, then launch. mode=0 = faithful (16 kernels). The
// faithful path has no double-buffer scratch, so there is no meke_scratch arg.
#include "meke_args.h"   // struct MekeArgs + meke_cuda_launch prototype

extern "C" void meke_cuda_flat(
    double *meke, double *kh_diff, double *le, double *ku, double *i_mass,
    double *depth_tot, double *bottom_fac2, double *barotr_fac2, double *src,
    double *uflux, double *vflux, double *mass_ws, double *rd_ws,
    double *sn_u_ws, double *sn_v_ws, double *ke_diss_ws,
    const double *u_bbl2, const double *f_centre, const double *gm_src,
    const double *ke_diss_ext, const double *h_layer, const double *rho_layer,
    const double *areaT, const double *iareaT, const double *idxT,
    const double *idyT, const double *dy_cu, const double *dx_cv,
    const double *idxCu, const double *idyCv,
    int nx, int ny, int nz,
    double dt, double dtscale, double cd_scale, double cb, double ct,
    double min_gamma2, double cdrag, double uscale,
    double a_deform, double a_rhines, double a_eady, double a_frict, double a_grid,
    double bgsrc, double gmcoeff, double frcoeff, double damping,
    double kh_bg, double k4, double khmeke_fac, double khcoeff,
    double visc_coeff_ku, double rho0, int backscatter)
{
    MekeArgs a;
    a.meke = meke; a.kh_diff = kh_diff; a.le = le; a.ku = ku; a.i_mass = i_mass;
    a.depth_tot = depth_tot; a.bottom_fac2 = bottom_fac2; a.barotr_fac2 = barotr_fac2;
    a.src = src; a.uflux = uflux; a.vflux = vflux; a.mass_ws = mass_ws; a.rd_ws = rd_ws;
    a.sn_u_ws = sn_u_ws; a.sn_v_ws = sn_v_ws; a.ke_diss_ws = ke_diss_ws;
    a.u_bbl2 = u_bbl2; a.f_centre = f_centre; a.gm_src = gm_src; a.ke_diss_ext = ke_diss_ext;
    a.h_layer = h_layer; a.rho_layer = rho_layer;
    a.areaT = areaT; a.iareaT = iareaT; a.idxT = idxT; a.idyT = idyT;
    a.dy_cu = dy_cu; a.dx_cv = dx_cv; a.idxCu = idxCu; a.idyCv = idyCv;
    a.nx = nx; a.ny = ny; a.nz = nz;
    a.dt = dt; a.dtscale = dtscale; a.cd_scale = cd_scale; a.cb = cb; a.ct = ct;
    a.min_gamma2 = min_gamma2; a.cdrag = cdrag; a.uscale = uscale;
    a.a_deform = a_deform; a.a_rhines = a_rhines; a.a_eady = a_eady;
    a.a_frict = a_frict; a.a_grid = a_grid;
    a.bgsrc = bgsrc; a.gmcoeff = gmcoeff; a.frcoeff = frcoeff; a.damping = damping;
    a.kh_bg = kh_bg; a.k4 = k4; a.khmeke_fac = khmeke_fac; a.khcoeff = khcoeff;
    a.visc_coeff_ku = visc_coeff_ku; a.rho0 = rho0; a.backscatter = backscatter;
    meke_cuda_launch(&a, 0);
}
