// Isolates the cost of the OpenACC `host_data use_device` present-table lookup.
//
// WHY THIS EXISTS: redi_native.cu answers "does the host_data path cost
// anything?" by A/B against the Fortran-driven CUDA variant. But that A/B is a
// GPU measurement, and on a shared V100 it is worth ~nothing while siblings are
// resident (measured: the same binary reads 70-112 ms/call under contention vs
// ~45 ms idle). The lookup itself, though, is pure HOST work: the runtime walks
// its present table once per array per call, before any kernel launches.
//
// So: run the IDENTICAL host_data region, but call this no-op instead of the
// kernels. What is left is the lookup, and it is immune to GPU contention.
// This number is trustworthy even on a busy node -- which is the point.
//
// The parameters are taken and sunk so the runtime must genuinely resolve all
// 24 device pointers; a no-op with unused args could let the region collapse.
extern "C" void redi_noop_(
    const int *nx_, const int *ny_, const int *nz_, const int *ns_, const double *dt_,
    const int *nghost_, const int *nxp_, const int *nyp_,
    const int *ww_, const int *we_, const int *wsz_, const int *wn_,
    const double *rho0_, const double *T_ref_, const double *S_ref_,
    const double *alpha_T_, const double *beta_S_,
    const double *h_layer, double *t_htr, double *s_htr,
    const double *wet_u, const double *wet_v,
    double *uPoL, double *uPoR, int *uKoL, int *uKoR, double *uhEff,
    double *vPoL, double *vPoR, int *vKoL, int *vKoR, double *vhEff,
    const double *khtr_u_ext, const double *khtr_v_ext, double *khtr_u, double *khtr_v,
    const double *dy_cu, const double *dx_cv, const double *idxCu, const double *idyCv,
    const double *areaT, double *tr_snap);

// volatile: the sink must survive -O3, or the compiler is entitled to discard
// every argument and we would be timing an empty function.
static volatile const void *g_sink = nullptr;

void redi_noop_(
    const int *nx_, const int *ny_, const int *nz_, const int *ns_, const double *dt_,
    const int *nghost_, const int *nxp_, const int *nyp_,
    const int *ww_, const int *we_, const int *wsz_, const int *wn_,
    const double *rho0_, const double *T_ref_, const double *S_ref_,
    const double *alpha_T_, const double *beta_S_,
    const double *h_layer, double *t_htr, double *s_htr,
    const double *wet_u, const double *wet_v,
    double *uPoL, double *uPoR, int *uKoL, int *uKoR, double *uhEff,
    double *vPoL, double *vPoR, int *vKoL, int *vKoR, double *vhEff,
    const double *khtr_u_ext, const double *khtr_v_ext, double *khtr_u, double *khtr_v,
    const double *dy_cu, const double *dx_cv, const double *idxCu, const double *idyCv,
    const double *areaT, double *tr_snap) {
    (void)nx_; (void)ny_; (void)nz_; (void)ns_; (void)dt_;
    (void)nghost_; (void)nxp_; (void)nyp_;
    (void)ww_; (void)we_; (void)wsz_; (void)wn_;
    (void)rho0_; (void)T_ref_; (void)S_ref_; (void)alpha_T_; (void)beta_S_;
    // Touch exactly the 24 array pointers host_data resolves.
    g_sink = h_layer;    g_sink = t_htr;      g_sink = s_htr;
    g_sink = wet_u;      g_sink = wet_v;
    g_sink = uPoL;       g_sink = uPoR;       g_sink = uKoL;
    g_sink = uKoR;       g_sink = uhEff;
    g_sink = vPoL;       g_sink = vPoR;       g_sink = vKoL;
    g_sink = vKoR;       g_sink = vhEff;
    g_sink = khtr_u_ext; g_sink = khtr_v_ext; g_sink = khtr_u; g_sink = khtr_v;
    g_sink = dy_cu;      g_sink = dx_cv;      g_sink = idxCu;  g_sink = idyCv;
    g_sink = areaT;      g_sink = tr_snap;
}
