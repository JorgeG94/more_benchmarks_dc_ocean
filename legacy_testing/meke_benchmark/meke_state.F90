!! MRE stubs for the MEKE benchmark.
!!
!! Field names, shapes and defaults follow production:
!!   src/core/ocean/state/ocean_metrics.F90   (ocean_metrics_t)
!!   src/core/structured/multilayer_cgrid_state.F90 (multilayer_cgrid_state_t)
!!   src/parameterizations/lateral/structured/ocean_gm.F90 (ocean_gm_t)
!!   src/core/structured/grid.F90             (hgrid_t)
!! ONLY the fields the live `meke_step` path touches are here.
!!
!! `ocean_meke_t` below is a FIELD-FOR-FIELD copy of production's type
!! (ocean_meke.F90:49-226) minus the type-bound procedures and the
!! `bytes` accounting (which needs mem_report). The scalar defaults are
!! copied verbatim so a config knob left unset here behaves as it does in a
!! real run.
module meke_state
   use constants, only: wp
   implicit none
   public

   type :: hgrid_t
      integer :: nx_total = 0, ny_total = 0
      integer :: nx_phys = 0, ny_phys = 0
      integer :: nghost = 3
   end type hgrid_t

   !! Curvilinear in production; this MRE fills them with the uniform-Cartesian
   !! values, which is what the gabight spherical configs are close to.
   type :: ocean_metrics_t
      real(wp), allocatable :: areaT(:, :)    !! cell area [m^2]     (nx, ny)
      real(wp), allocatable :: iareaT(:, :)   !! 1/areaT   [1/m^2]   (nx, ny)
      real(wp), allocatable :: idxT(:, :)     !! 1/dxT     [1/m]     (nx, ny)
      real(wp), allocatable :: idyT(:, :)     !! 1/dyT     [1/m]     (nx, ny)
      real(wp), allocatable :: dy_cu(:, :)    !! east-face width [m] (nx+1, ny)
      real(wp), allocatable :: dx_cv(:, :)    !! north-face width[m] (nx, ny+1)
      real(wp), allocatable :: idxCu(:, :)    !! 1/dxCu    [1/m]     (nx+1, ny)
      real(wp), allocatable :: idyCv(:, :)    !! 1/dyCv    [1/m]     (nx, ny+1)
   end type ocean_metrics_t

   type :: multilayer_cgrid_state_t
      integer :: nz_ml = 0
      real(wp), allocatable :: h_layer(:, :, :)    !! (nx, ny, nz)
      real(wp), allocatable :: rho_layer(:, :, :)  !! (nx, ny, nz)
   end type multilayer_cgrid_state_t

   type :: ocean_gm_t
      logical :: is_init = .false.
      real(wp) :: rho0 = 1035.0_wp
      real(wp), allocatable :: gm_src(:, :)        !! (nx, ny)
   end type ocean_gm_t

   !! Production's ocean_meke_t, fields + defaults verbatim (TBPs removed).
   type :: ocean_meke_t
      logical :: is_init = .false.
      logical :: enable = .false.
      real(wp) :: gmcoeff = -1.0_wp
      real(wp) :: frcoeff = -1.0_wp
      real(wp) :: bgsrc = 0.0_wp
      real(wp) :: damping = 0.0_wp
      real(wp) :: kh = -1.0_wp
      real(wp) :: k4 = -1.0_wp
      real(wp) :: khmeke_fac = 0.0_wp
      real(wp) :: advection_factor = 0.0_wp
      real(wp) :: khcoeff = 1.0_wp
      real(wp) :: cd_scale = 0.0_wp
      real(wp) :: cb = 25.0_wp
      real(wp) :: ct = 50.0_wp
      real(wp) :: min_gamma2 = 1.0e-4_wp
      real(wp) :: uscale = 0.0_wp
      real(wp) :: dtscale = 1.0_wp
      real(wp) :: cdrag = 2.5e-3_wp
      logical :: use_bbl_drag = .false.
      real(wp) :: alpha_deform = 0.0_wp
      real(wp) :: alpha_rhines = 0.0_wp
      real(wp) :: alpha_eady = 0.0_wp
      real(wp) :: alpha_frict = 0.0_wp
      real(wp) :: alpha_grid = 0.0_wp
      real(wp) :: khth_fac = 0.0_wp
      real(wp) :: khtr_fac = 0.0_wp
      logical :: backscatter = .false.
      real(wp) :: visc_coeff_ku = 0.0_wp
      integer :: nx_total = 0
      integer :: ny_total = 0
      integer :: nz_ml = 0
      real(wp), allocatable :: meke(:, :)
      real(wp), allocatable :: kh_diff(:, :)
      real(wp), allocatable :: le(:, :)
      real(wp), allocatable :: ku(:, :)
      real(wp), allocatable :: i_mass(:, :)
      real(wp), allocatable :: depth_tot(:, :)
      real(wp), allocatable :: bottom_fac2(:, :)
      real(wp), allocatable :: barotr_fac2(:, :)
      real(wp), allocatable :: src(:, :)
      real(wp), allocatable :: uflux(:, :)
      real(wp), allocatable :: vflux(:, :)
      real(wp), allocatable :: del2(:, :)
      real(wp), allocatable :: mass_ws(:, :)
      real(wp), allocatable :: u_bbl2(:, :)
      real(wp), allocatable :: ke_diss_ws(:, :)
      real(wp), allocatable :: rd_ws(:, :)
      real(wp), allocatable :: f_centre(:, :)
      real(wp), allocatable :: sn_u_ws(:, :)
      real(wp), allocatable :: sn_v_ws(:, :)
      real(wp), allocatable :: baro_hu(:, :)
      real(wp), allocatable :: baro_hv(:, :)
   end type ocean_meke_t

end module meke_state
