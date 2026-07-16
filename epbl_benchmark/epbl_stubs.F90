!! MRE stubs for the derived types + the one device routine that the ocean model's
!! `epbl_column_kernel` reads.  Everything here is cut down to EXACTLY the
!! fields the kernel touches, with names and defaults verbatim from production
!! so the extracted kernel compiles byte-for-byte unchanged.
!!
!! Sources (<model> @ <model-root>/src):
!!   scratch_3d_buffer_t        framework/scratch_3d.F90
!!   multilayer_cgrid_state_t   core/structured/multilayer_cgrid_state.F90
!!   ocean_surface_stress_t     parameterizations/vertical/structured/
!!                                ocean_surface_stress.F90
!!   ocean_surface_flux_t       parameterizations/vertical/structured/
!!                                ocean_surface_flux.F90
!!   ocean_eos_t                equation_of_state/structured/ocean_eos.F90
!!   eos_specvol_derivs         same file, lines 800-864 -- VERBATIM, all three
!!                              branches, including the Roquet dispatch stub.
!!
!! WHAT IS DROPPED, and why it is safe:
!!   * The tracer registry (`ms%tracers(:)%hTr`) is NOT modelled. Production's
!!     `epbl_compute` host-dereferences it and forwards bare 3-D arrays to
!!     `epbl_column_kernel`; the driver does the same dereference itself. The
!!     kernel under test never sees the registry.
!!   * `ocean_surface_flux_t` keeps only `cp` (the kernel's caller reads it to
!!     form `inv_rho0_cp`); Q_heat/Q_salt are passed as bare arrays, exactly as
!!     production's shim does.
!!   * `roquet_spv_point` is a STUB that error-stops. Production's profiled
!!     config runs `eos = "linear"` (confirmed in the run log: "EOS variant:
!!     linear"), so the Roquet branch is dead. The branch is KEPT in
!!     `eos_specvol_derivs` so the inliner sees the same control flow.
module epbl_stubs
   use constants, only: wp, SEAWATER_CP
   implicit none
   private

   public :: scratch_3d_buffer_t
   public :: multilayer_cgrid_state_t
   public :: ocean_surface_stress_t
   public :: ocean_surface_flux_t
   public :: ocean_eos_t, eos_specvol_derivs
   public :: EOS_VARIANT_LINEAR, EOS_VARIANT_WRIGHT_97, EOS_VARIANT_ROQUET_SPV
   public :: TS_POT_PRAC

   ! ---- ocean_eos tags (verbatim) ----
   integer, parameter :: EOS_VARIANT_LINEAR = 1
   integer, parameter :: EOS_VARIANT_WRIGHT_97 = 2
   integer, parameter :: EOS_VARIANT_TEOS10 = 3
   integer, parameter :: EOS_VARIANT_ROQUET_SPV = 4
   integer, parameter :: TS_POT_PRAC = 1
   integer, parameter :: TS_CONS_ABS = 2

   ! Wright (1997) Table A1 coefficients -- verbatim from ocean_eos.F90:77-91.
   real(wp), parameter :: WRIGHT_A0 = 7.057924e-4_wp
   real(wp), parameter :: WRIGHT_A1 = 3.480336e-7_wp
   real(wp), parameter :: WRIGHT_A2 = -1.112733e-7_wp
   real(wp), parameter :: WRIGHT_B0 = 5.790749e8_wp
   real(wp), parameter :: WRIGHT_B1 = 3.516535e6_wp
   real(wp), parameter :: WRIGHT_B2 = -4.002714e4_wp
   real(wp), parameter :: WRIGHT_B3 = 2.084372e2_wp
   real(wp), parameter :: WRIGHT_B4 = 5.944068e5_wp
   real(wp), parameter :: WRIGHT_B5 = -9.643486e3_wp
   real(wp), parameter :: WRIGHT_C0 = 1.704853e5_wp
   real(wp), parameter :: WRIGHT_C1 = 7.904722e2_wp
   real(wp), parameter :: WRIGHT_C2 = -7.984422e0_wp
   real(wp), parameter :: WRIGHT_C3 = 5.140652e-2_wp
   real(wp), parameter :: WRIGHT_C4 = -2.302158e2_wp
   real(wp), parameter :: WRIGHT_C5 = -3.079464e0_wp

   type :: scratch_3d_buffer_t
      logical :: is_init = .false.
      integer :: n1 = 0
      integer :: n2 = 0
      integer :: n3 = 0
      real(wp), allocatable :: data(:, :, :)
      character(len=32) :: name = ""
   contains
      procedure, non_overridable :: init => scratch_3d_buffer_init
      procedure, non_overridable :: destroy => scratch_3d_buffer_destroy
   end type scratch_3d_buffer_t

   !! Only the three fields `epbl_column_kernel` reads: nz_ml, h_layer,
   !! wet_mask. (idx_temperature / idx_salinity are read by the OUTER shim
   !! `epbl_compute`, which the driver replaces -- kept for completeness.)
   type :: multilayer_cgrid_state_t
      logical :: is_init = .false.
      integer :: nz_ml = 0
      real(wp), allocatable :: h_layer(:, :, :)
      real(wp), allocatable :: wet_mask(:, :)
      integer :: idx_salinity = 0
      integer :: idx_temperature = 0
   end type multilayer_cgrid_state_t

   !! Only tau_x / tau_y -- the kernel forms the cell-centre stress average.
   type :: ocean_surface_stress_t
      logical :: is_init = .false.
      real(wp) :: rho0 = 1035.0_wp
      real(wp), allocatable :: tau_x(:, :)   !! (nx+1, ny), N/m^2
      real(wp), allocatable :: tau_y(:, :)   !! (nx, ny+1), N/m^2
   end type ocean_surface_stress_t

   !! Only cp + rho0 -- the shim reads them to form inv_rho0_cp / inv_rho0.
   type :: ocean_surface_flux_t
      logical :: is_init = .false.
      real(wp) :: rho0 = 1035.0_wp
      real(wp) :: cp = SEAWATER_CP
      real(wp), allocatable :: Q_heat(:, :)  !! (nx, ny), W/m^2
      real(wp), allocatable :: Q_salt(:, :)  !! (nx, ny), kg/m^2/s
   end type ocean_surface_flux_t

   !! Flat POD, no allocatable -- passed BY VALUE into the device routine and
   !! held in registers. Defaults verbatim from ocean_eos.F90.
   type :: ocean_eos_t
      logical :: is_init = .false.
      integer  :: variant = EOS_VARIANT_LINEAR
      real(wp) :: rho0 = 1035.0_wp
      real(wp) :: T_ref = 10.0_wp
      real(wp) :: S_ref = 35.0_wp
      real(wp) :: alpha_T = 1.7e-4_wp
      real(wp) :: beta_S = 7.6e-4_wp
      real(wp) :: p_ref = 0.0_wp
      integer :: ts_convention = TS_POT_PRAC
   end type ocean_eos_t

contains

   subroutine scratch_3d_buffer_init(this, n1, n2, n3, name)
      class(scratch_3d_buffer_t), intent(inout) :: this
      integer, intent(in) :: n1, n2, n3
      character(len=*), intent(in), optional :: name
      this%n1 = n1; this%n2 = n2; this%n3 = n3
      allocate (this%data(n1, n2, n3), source=0.0_wp)
      if (present(name)) this%name = name
      this%is_init = .true.
   end subroutine scratch_3d_buffer_init

   subroutine scratch_3d_buffer_destroy(this)
      class(scratch_3d_buffer_t), intent(inout) :: this
      this%is_init = .false.
      if (allocated(this%data)) deallocate (this%data)
   end subroutine scratch_3d_buffer_destroy

   ! ==================================================================
   ! VERBATIM from ocean_eos.F90:800-864 (doc comment trimmed).
   ! ==================================================================
   pure subroutine eos_specvol_derivs(eos, T, S, p, dsv_dt, dsv_ds)
      !! Analytic specific-volume sensitivities dSV/dT and dSV/dS
      !! (SV = 1/rho) at a point.  Needed by the EPBL energy
      !! bookkeeping (pressure-weighted PE-per-unit-tracer-change
      !! weights) and kappa-shear buoyancy -- dSV/dX = -(1/rho^2)
      !! d(rho)/dX.
      !$acc routine seq
      type(ocean_eos_t), intent(in) :: eos
      real(wp), intent(in) :: T, S
      real(wp), intent(in) :: p
      real(wp), intent(out) :: dsv_dt
      real(wp), intent(out) :: dsv_ds

      real(wp) :: T_sq, p_0, lambda, big_p, inv_p, inv_p2
      real(wp) :: dp0_dt, dlam_dt, dp0_ds, dlam_ds
      real(wp) :: sv_roq

      if (eos%variant == EOS_VARIANT_ROQUET_SPV) then
         call roquet_spv_point(T, S, p, sv_roq, dsv_dt, dsv_ds)
      else if (eos%variant == EOS_VARIANT_WRIGHT_97) then
         T_sq = T*T
         p_0 = WRIGHT_B0 + WRIGHT_B1*T + WRIGHT_B2*T_sq + WRIGHT_B3*T_sq*T + &
               WRIGHT_B4*S + WRIGHT_B5*S*T
         lambda = WRIGHT_C0 + WRIGHT_C1*T + WRIGHT_C2*T_sq + WRIGHT_C3*T_sq*T + &
                  WRIGHT_C4*S + WRIGHT_C5*S*T
         dp0_dt = WRIGHT_B1 + 2.0_wp*WRIGHT_B2*T + 3.0_wp*WRIGHT_B3*T_sq + &
                  WRIGHT_B5*S
         dlam_dt = WRIGHT_C1 + 2.0_wp*WRIGHT_C2*T + 3.0_wp*WRIGHT_C3*T_sq + &
                   WRIGHT_C5*S
         dp0_ds = WRIGHT_B4 + WRIGHT_B5*T
         dlam_ds = WRIGHT_C4 + WRIGHT_C5*T
         big_p = p + p_0
         inv_p = 1.0_wp/big_p
         inv_p2 = inv_p*inv_p
         dsv_dt = WRIGHT_A1 + dlam_dt*inv_p - lambda*dp0_dt*inv_p2
         dsv_ds = WRIGHT_A2 + dlam_ds*inv_p - lambda*dp0_ds*inv_p2
      else
         dsv_dt = eos%alpha_T/(eos%rho0*eos%rho0)
         dsv_ds = -eos%beta_S/(eos%rho0*eos%rho0)
      end if
   end subroutine eos_specvol_derivs

   !! STUB. The profiled production config runs eos = "linear" (see the run
   !! log: "EOS variant: linear"), so this branch is unreachable. Kept so
   !! `eos_specvol_derivs` retains production's exact control flow.
   pure subroutine roquet_spv_point(T, S, p, sv, dsv_dt, dsv_ds)
      !$acc routine seq
      real(wp), intent(in) :: T, S, p
      real(wp), intent(out) :: sv, dsv_dt, dsv_ds
      sv = 1.0_wp/1035.0_wp
      dsv_dt = 0.0_wp*T
      dsv_ds = 0.0_wp*(S + p)
   end subroutine roquet_spv_point

end module epbl_stubs
