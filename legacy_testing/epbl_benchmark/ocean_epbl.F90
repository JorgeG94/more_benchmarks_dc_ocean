!! ============================================================================
!! MRE EXTRACT of the ocean model's production EPBL scheme.
!!
!!   source : <model>/src/parameterizations/vertical/structured/ocean_epbl.F90
!!   md5    : 33e307f821c1729434f05fbc8b33550e   (re-extract if this drifts)
!!
!! THE KERNEL UNDER TEST -- `epbl_column_kernel`, production lines 818-1340 --
!! IS SPLICED IN VERBATIM, byte for byte. So are `epbl_find_mstar`,
!! `epbl_mixlen_shape`, `epbl_lf17_wave_state`, `epbl_lf17_la`,
!! `one_m_exp_x`, `epbl_lt_enhance` (lines 545-771), `epbl_merge_into_kv_kt`
!! (1342-1369), the `ocean_epbl_t` type + all knob defaults (46-303), the
!! parsers (487-539), `ocean_epbl_init` / `_destroy` (311-365) and
!! `ocean_epbl_set_f_centre` (463-481).
!!
!! ---- WHAT WAS REMOVED, exhaustively ----------------------------------------
!! Nothing inside the timed kernel. Only host-side scaffolding:
!!
!!  1. `epbl_compute` (production 777-816) -- the OUTER SHIM. It host-
!!     dereferences the tracer registry (`ms%tracers(idx)%hTr`) and the
!!     surface-flux slot, then forwards bare arrays to `epbl_column_kernel`.
!!     The driver performs the identical dereference and calls
!!     `epbl_column_kernel` with the identical argument list. The shim runs on
!!     the host, launches nothing, and is not the thing being measured.
!!     CONSEQUENCE: the tracer registry (`tracer_t`) is not modelled at all.
!!
!!  2. `ocean_epbl_enter_data{,_impl}` / `_exit_data{,_impl}` (367-461) and the
!!     matching type-bound `procedure`s. RESUME_GPU_MRE gotcha 3: an OpenACC
!!     deep copy must live in the scope that OWNS the variable, or the
!!     components never attach and every region falls back to implicit copies.
!!     The driver therefore owns and performs the deep copy explicitly.
!!
!!  3. `ocean_epbl_bytes` (1371-1393) + its type-bound `procedure`. Pure
!!     host-side accounting; needs `mem_report`, which is not stubbed.
!!
!! ---- WHAT IS STUBBED (see epbl_stubs.F90) -------------------------------
!! `hgrid_t`, `multilayer_cgrid_state_t`, `ocean_surface_stress_t`,
!! `ocean_surface_flux_t`, `scratch_3d_buffer_t`, `ocean_eos_t`, and
!! `eos_specvol_derivs` (itself a verbatim splice of ocean_eos.F90:800-864,
!! all three branches). Each type carries exactly the fields this kernel reads,
!! with production's names and defaults.
!!
!! ---- WHICH KNOBS ARE LIVE ---------------------------------------------------
!! The profiled config (gabight_sph_acc_year_v100.nml, which produced
!! <scratch>/<user>/tassie_runs/gabight_acc_120d.173705307.<sched>.log)
!! sets ONLY `enable = .true.` in `&ocean_epbl_nml`. Every other EPBL knob is
!! at the default baked into the type below, so the live path is:
!!     mstar_scheme       = om4          (EPBL_MSTAR_OM4)
!!     vstar_scheme       = cube_root    (EPBL_VSTAR_CUBE_ROOT)
!!     mld_iteration      = .true.  mld_max_its = 20  mld_tol = 1.0 m
!!     mld_bisection      = .false. (false position)
!!     mld_use_prev_guess = .false. (ALWAYS restart at half depth)
!!     use_lt             = .false. (the whole LF17/Langmuir path is DEAD)
!!     tke_diags          = .false. (the budget stores are DEAD)
!!     mixlen_exponent    = 2.0     (the fast, non-`**` shape path)
!!     combine_mode       = add
!! and EOS = linear (log: "EOS variant: linear"), rho0 = 1035, alpha_T = 0.2
!! (from &ocean_ic_nml), beta_S = 7.6e-4.
!! NONE of these dead branches were deleted -- they are all still compiled, so
!! the register pressure and control flow are production's.
!! ============================================================================
module ocean_epbl
   use constants, only: wp, GRAVITY, H_DIV_EPS
   use grid, only: hgrid_t
   use epbl_stubs, only: multilayer_cgrid_state_t, &
                             ocean_surface_stress_t, &
                             ocean_surface_flux_t, &
                             ocean_eos_t, eos_specvol_derivs, &
                             scratch_3d_buffer_t
   use, intrinsic :: iso_fortran_env, only: int64
   implicit none
   private

   public :: ocean_epbl_t
   public :: epbl_column_kernel
      !! SCAFFOLDING CHANGE (the only one): production keeps this private and
      !! exports the outer shim `epbl_compute` instead. The shim is host-side
      !! tracer-registry dereferencing (see this file's header, removal 1); the
      !! driver does that dereference itself and calls the kernel directly, so
      !! the kernel must be visible. Nothing about the kernel itself changes.
   public :: epbl_merge_into_kv_kt
   public :: epbl_find_mstar
   public :: epbl_mixlen_shape
   public :: epbl_lf17_wave_state
   public :: epbl_lf17_la
   public :: epbl_lt_enhance
   public :: parse_epbl_mstar_scheme
   public :: parse_epbl_vstar_scheme
   public :: parse_epbl_combine
   public :: parse_epbl_lt_scheme

   ! mstar (mechanical TKE / u*^3) scheme tags.
   integer, parameter, public :: EPBL_MSTAR_CONSTANT = 1
      !! Fixed mstar (the simplest credible configuration).
   integer, parameter, public :: EPBL_MSTAR_OM4 = 2
      !! Ekman/Obukhov-balance form used by MOM6's OM4 production
      !! config (Reichl & Hallberg 2018 Appendix).
   integer, parameter, public :: EPBL_MSTAR_RH18 = 3
      !! Reichl & Hallberg (2018) eq. fits (cN1..cS2 coefficients).

   ! Turbulent-velocity-scale scheme tags.
   integer, parameter, public :: EPBL_VSTAR_CUBE_ROOT = 1
      !! vstar = vstar_scale_fac * (TKE / (dt rho0))^(1/3).
   integer, parameter, public :: EPBL_VSTAR_RH18 = 2
      !! Separate mechanical (u*-proportional, surface-decaying) +
      !! convective cube-root contributions (RH18).

   ! Combination of kd_epbl with the interior closure's kv/kt.
   integer, parameter, public :: EPBL_COMBINE_ADD = 1
      !! kt += kd ; kv += prandtl*kd  (MOM6 EPBL_IS_ADDITIVE=.true.)
   integer, parameter, public :: EPBL_COMBINE_MAX = 2
      !! kt = max(kt, kd) ; kv = max(kv, prandtl*kd)

   ! Langmuir-enhancement scheme tags (Reichl & Li 2019 forms).
   integer, parameter, public :: EPBL_LT_NONE = 0
   integer, parameter, public :: EPBL_LT_RESCALE = 1
      !! mstar *= min(max_enh, 1 + coef * La^exp)  (multiplicative)
   integer, parameter, public :: EPBL_LT_ADDITIVE = 2
      !! mstar += coef * La^exp

   ! Thickness regulariser added to every layer thickness used inside
   ! the energy solve (MOM6 H_subroundoff role): keeps the pivot
   ! products bdt1 = hp_a*hp_b representable when a layer vanishes.
   ! Division-safety only (D4 role 2): aliases the constant of record
   ! `H_DIV_EPS` from constants — the local name is kept so the many
   ! use sites below read unchanged (bitwise no-op: 1e-20 == 1e-20).
   real(wp), parameter :: H_NEGLECT = H_DIV_EPS

   ! ---- LF17 / COARE 3.5 empirical constants ----
   ! Parts of the cited formulas, NOT namelist knobs.  Li & Fox-Kemper
   ! (2017) statistical wave state; Edson et al. (2013) COARE 3.5
   ! drag; Webb & Fox-Kemper (2015) / Breivik et al. (2016)
   ! surface-layer-averaged Stokes drift over a Phillips spectrum.
   real(wp), parameter :: LT_VONKAR_WAVES = 0.40_wp
      !! von Karman constant used by the COARE u*->U10 inversion.
   real(wp), parameter :: LT_NU_AIR = 1.0e-6_wp
      !! Kinematic viscosity of air (m^2/s) for the smooth-flow z0.
   real(wp), parameter :: LT_RHO_AIR = 1.225_wp
      !! Air density (kg/m^3) for the water->air u* conversion.
   real(wp), parameter :: LT_CHARNOCK_MIN = 0.028_wp
      !! Cap on the Charnock parameter.
   real(wp), parameter :: LT_CHARNOCK_SLOPE = 0.0017_wp
      !! d(alpha_Charnock)/dU10 (s/m).
   real(wp), parameter :: LT_CHARNOCK_ICPT = -0.005_wp
      !! Charnock intercept at U10 = 0.
   real(wp), parameter :: LT_US_TO_U10 = 0.0162_wp
      !! Surface Stokes drift / U10 (Webb 2011).
   real(wp), parameter :: LT_SWH_FROM_U10SQ = 0.0246_wp
      !! Significant wave height = c * U10^2 (s^2/m).
   real(wp), parameter :: LT_U19P5_TO_U10 = 1.075_wp
      !! Pierson-Moskowitz U19.5/U10 ratio.
   real(wp), parameter :: LT_FM_INTO_FP = 1.296_wp
      !! Mean / peak frequency ratio (Webb 2011).
   real(wp), parameter :: LT_R_LOSS = 0.667_wp
      !! Stokes-transport loss ratio.
   real(wp), parameter :: LT_PI = 4.0_wp*atan(1.0_wp)

   type :: ocean_epbl_t
      logical :: is_init = .false.
         !! True between `init` and `destroy`.

      ! ---- Scheme selection + master switch ----
      logical :: enable = .false.
         !! Master switch.  Default off — all existing namelists and
         !! tests stay bit-identical.  Requires `vmix%use_closure`;
         !! disables the KPP overlay (configure logs the override).
      integer :: mstar_scheme = EPBL_MSTAR_OM4
      integer :: vstar_scheme = EPBL_VSTAR_CUBE_ROOT
      integer :: combine_mode = EPBL_COMBINE_ADD

      ! ---- mstar knobs ----
      real(wp) :: mstar_const = 1.2_wp
         !! Constant-scheme mstar (MOM6 MSTAR).
      real(wp) :: mstar_cap = -1.0_wp
         !! Cap on mstar for the OM4/RH18 schemes; off when < 0.
      real(wp) :: mstar_coef1 = 0.3_wp
         !! OM4 stabilizing-balance coefficient (MOM6 MSTAR2_COEF1).
      real(wp) :: c_ek = 0.085_wp
         !! OM4 Ekman-limit coefficient (MOM6 MSTAR2_COEF2).
      real(wp) :: mstar_conv_adj = 0.0_wp
         !! Reduce mechanical mstar when convection dominates, in
         !! [0,1]; 0 = off (MOM6 MSTAR_CONV_ADJ).
      real(wp) :: rh18_cn1 = 0.275_wp
      real(wp) :: rh18_cn2 = 8.0_wp
      real(wp) :: rh18_cn3 = -5.0_wp
      real(wp) :: rh18_cs1 = 0.2_wp
      real(wp) :: rh18_cs2 = 0.4_wp
         !! RH18 mstar fit coefficients (paper values).

      ! ---- Energetics knobs ----
      real(wp) :: nstar = 0.2_wp
         !! Fraction of convectively released PE that becomes
         !! entrainment-driving TKE (MOM6 NSTAR).
      real(wp) :: tke_decay = 2.5_wp
         !! Ratio of the natural Ekman depth to the mechanical-TKE
         !! decay scale (MOM6 TKE_DECAY).
      real(wp) :: wstar_ustar_coef = 1.0_wp
         !! Weight of the convective reservoir in the velocity scale.
      real(wp) :: vstar_scale_fac = 1.0_wp
         !! Overall vstar multiplier (∝ diffusivity).
      real(wp) :: vstar_surf_fac = 1.2_wp
         !! RH18 vstar scheme: mechanical surface vstar ∝ u* factor.
      real(wp) :: von_karman = 0.41_wp
         !! kappa in Kd = vstar * kappa * mixing_length.
      real(wp) :: ekman_scale_coef = 1.0_wp
         !! Rotational inhibition of the mixing length.
      real(wp) :: min_mix_len = 0.0_wp
         !! Mixing-length floor (m).
      real(wp) :: mixlen_exponent = 2.0_wp
         !! Shape-function exponent (2 = KPP-like; fast path).
      real(wp) :: translay_scale = 0.1_wp
         !! Transition-layer floor of the shape function; must be in
         !! [0,1) when `mld_iteration` is on.

      ! ---- MLD iteration knobs ----
      logical :: mld_iteration = .true.
         !! Iterate the boundary-layer depth to self-consistency with
         !! the mixing-length shape function (MOM6 USE_MLD_ITERATION).
      real(wp) :: mld_tol = 1.0_wp
         !! Convergence tolerance on the MLD root-find (m).
      integer :: mld_max_its = 20
      logical :: mld_bisection = .false.
         !! Plain bisection instead of false-position.
      logical :: mld_use_prev_guess = .false.
         !! Seed the iteration from the previous step's MLD (faster,
         !! ~1-3 iterations; off = always start at half depth).

      ! ---- Environment ----
      real(wp) :: rho0 = 1035.0_wp
         !! Boussinesq reference density (kg/m^3).
      real(wp) :: omega = 7.2921e-5_wp
         !! Earth rotation rate (1/s), for the omega_frac blend.
      real(wp) :: omega_frac = 0.0_wp
         !! Blend |f| with 2*Omega: absf = sqrt((1-of) f^2 + of 4 Om^2).
      real(wp) :: ustar_min = 1.0e-8_wp
         !! Floor on u* (deliberate <model> floor — pure denominator
         !! guard in the TKE decay scale; MOM6 derives ~1e-10).
      real(wp) :: prandtl = 1.0_wp
         !! Kv = prandtl * Kd into the momentum solve.
      logical :: tke_diags = .false.
         !! Compute + store the per-column TKE budget terms (W/m^2).
         !! The column ledger closes to round-off — see the design
         !! doc §7; `test_ocean_epbl` asserts it.

      ! ---- Langmuir turbulence (LF17 wind-only path) ----
      ! One scalar La per column per MLD iteration, computed from u*
      ! and the current MLD guess (Li & Fox-Kemper 2017 statistical
      ! waves; no wave model, no Stokes arrays).  Enhancement applies
      ! to mstar only (Reichl & Li 2019).  NOTE: the LF17 branch uses
      ! NEITHER MOM6's MIN_LANGMUIR nor LA_DEPTH_MIN floors — those
      ! belong to the profile-averaging wave paths only.
      logical :: use_lt = .false.
         !! Master Langmuir switch (default off — bit-identity).
      integer :: lt_scheme = EPBL_LT_RESCALE
         !! Enhancement form: rescale (multiplicative) or additive.
      real(wp) :: lt_enhance_coef = 0.447_wp
         !! Enhancement coefficient (MOM6 LT_ENHANCE_COEF).
      real(wp) :: lt_enhance_exp = -1.33_wp
         !! La exponent (MOM6 LT_ENHANCE_EXP).
      real(wp) :: lt_max_enhance = 5.0_wp
         !! Cap on the multiplicative enhancement factor.
      real(wp) :: la_frac_hbl = 0.04_wp
         !! Stokes surface-layer-average depth as a fraction of the
         !! BLD (MOM6 LA_DEPTH_RATIO).
      real(wp) :: lt_lac1 = -0.87_wp
      real(wp) :: lt_lac2 = 0.0_wp
      real(wp) :: lt_lac3 = 0.0_wp
      real(wp) :: lt_lac4 = 0.95_wp
      real(wp) :: lt_lac5 = 0.95_wp
         !! Stability-modified Langmuir number coefficients (MOM6
         !! LT_MOD_LAC1..5: MLD/Ekman, MLD/Obukhov stable, unstable,
         !! Ekman/Obukhov stable, unstable).  All-zero => La_mod = La.

      ! ---- EOS hookup (shared handle from the eos slot) ----
      ! The energy weights use the SAME EOS the dyn-core runs.  We
      ! carry a value copy of the flat-POD `ocean_eos_t` (no
      ! allocatable) set once at configure from `ocean_state%eos`;
      ! it maps onto the device with the parent `this` for free and
      ! the per-column `eos_specvol_derivs(this%eos, ...)` call reads
      ! it from registers.  One source of truth — no private scalar
      ! copies that can drift from the dyn-core EOS.
      type(ocean_eos_t) :: eos

      ! ---- Persistent fields ----
      real(wp), allocatable :: f_centre(:, :)
         !! |f| at cell centres (1/s); filled by `set_f_centre`.
      real(wp), allocatable :: mld(:, :)
         !! Converged active-mixing-layer depth (m) from the last
         !! call — the next step's first guess (when
         !! `mld_use_prev_guess`) and the primary diagnostic.
      real(wp), allocatable :: kd_int(:, :, :)
         !! EPBL diapycnal diffusivity at interfaces (m^2/s),
         !! (nx, ny, nz+1); zero at bed (k=1) and surface (k=nz+1).
      real(wp), allocatable :: la(:, :)
         !! Turbulent Langmuir number from the last call (diagnostic;
         !! 0 where `use_lt` is off or the column is dry).

      ! ---- Per-step TKE budget diagnostics (W/m^2) ----
      ! Overwritten each `epbl_compute` (final MLD iteration).  The
      ! ledger: wind + conv + forcing - mixing - mech_decay -
      ! conv_decay = 0 exactly (forcing <= 0, all others >= 0).
      real(wp), allocatable :: tke_wind(:, :)
      real(wp), allocatable :: tke_conv(:, :)
      real(wp), allocatable :: tke_forcing(:, :)
      real(wp), allocatable :: tke_mixing(:, :)
      real(wp), allocatable :: tke_mech_decay(:, :)
      real(wp), allocatable :: tke_conv_decay(:, :)

      ! ---- Iteration-invariant column workspaces ----
      ! Filled once per call by the column prep sweep; read by every
      ! MLD iteration.  All (nx, ny, nz).  Everything else the sweep
      ! needs is carried as scalars (see the design doc D6) — no
      ! per-column work arrays, no NZ_STACK_MAX locals.
      type(scratch_3d_buffer_t) :: t0
         !! Pre-mixing layer temperature (degC).
      type(scratch_3d_buffer_t) :: s0
         !! Pre-mixing layer salinity (PSU).
      type(scratch_3d_buffer_t) :: dpe_t
         !! d(column PE)/d(T_k): rho0*h_k * p_mid * dSV_dT (J/m^2/degC).
      type(scratch_3d_buffer_t) :: dpe_s
         !! d(column PE)/d(S_k) (J/m^2/PSU).
      type(scratch_3d_buffer_t) :: dcolht_t
         !! d(column height)/d(T_k): rho0*h_k * dSV_dT (m/degC) —
         !! steric sensitivity for the gravity-wave radiation term.
      type(scratch_3d_buffer_t) :: dcolht_s
         !! d(column height)/d(S_k) (m/PSU).
   contains
      procedure, non_overridable :: init => ocean_epbl_init
      procedure, non_overridable :: destroy => ocean_epbl_destroy
      procedure, non_overridable :: set_f_centre => ocean_epbl_set_f_centre
   end type ocean_epbl_t

contains

   ! =================================================================
   ! Lifecycle
   ! =================================================================

   subroutine ocean_epbl_init(this, grid, nz_ml)
      !! Allocate the persistent fields + column workspaces.  Always
      !! allocates (configure runs after init, so `enable` isn't
      !! known yet); the memory cost when off is the same 7-field
      !! footprint the vmix slot already pays.
      class(ocean_epbl_t), intent(inout) :: this
      type(hgrid_t), intent(in) :: grid
      integer, intent(in), optional :: nz_ml
      integer :: nx, ny, nz

      nx = grid%nx_total
      ny = grid%ny_total
      nz = 1
      if (present(nz_ml)) nz = nz_ml

      allocate (this%f_centre(nx, ny), source=0.0_wp)
      allocate (this%mld(nx, ny), source=0.0_wp)
      allocate (this%kd_int(nx, ny, nz + 1), source=0.0_wp)
      allocate (this%la(nx, ny), source=0.0_wp)
      allocate (this%tke_wind(nx, ny), source=0.0_wp)
      allocate (this%tke_conv(nx, ny), source=0.0_wp)
      allocate (this%tke_forcing(nx, ny), source=0.0_wp)
      allocate (this%tke_mixing(nx, ny), source=0.0_wp)
      allocate (this%tke_mech_decay(nx, ny), source=0.0_wp)
      allocate (this%tke_conv_decay(nx, ny), source=0.0_wp)
      call this%t0%init(nx, ny, nz, "epbl_t0")
      call this%s0%init(nx, ny, nz, "epbl_s0")
      call this%dpe_t%init(nx, ny, nz, "epbl_dpe_t")
      call this%dpe_s%init(nx, ny, nz, "epbl_dpe_s")
      call this%dcolht_t%init(nx, ny, nz, "epbl_dcolht_t")
      call this%dcolht_s%init(nx, ny, nz, "epbl_dcolht_s")

      this%is_init = .true.
   end subroutine ocean_epbl_init

   subroutine ocean_epbl_destroy(this)
      class(ocean_epbl_t), intent(inout) :: this
      this%is_init = .false.
      if (allocated(this%f_centre)) deallocate (this%f_centre)
      if (allocated(this%mld)) deallocate (this%mld)
      if (allocated(this%kd_int)) deallocate (this%kd_int)
      if (allocated(this%la)) deallocate (this%la)
      if (allocated(this%tke_wind)) deallocate (this%tke_wind)
      if (allocated(this%tke_conv)) deallocate (this%tke_conv)
      if (allocated(this%tke_forcing)) deallocate (this%tke_forcing)
      if (allocated(this%tke_mixing)) deallocate (this%tke_mixing)
      if (allocated(this%tke_mech_decay)) deallocate (this%tke_mech_decay)
      if (allocated(this%tke_conv_decay)) deallocate (this%tke_conv_decay)
      call this%t0%destroy()
      call this%s0%destroy()
      call this%dpe_t%destroy()
      call this%dpe_s%destroy()
      call this%dcolht_t%destroy()
      call this%dcolht_s%destroy()
   end subroutine ocean_epbl_destroy

   subroutine ocean_epbl_set_f_centre(this, grid, f_0, beta, y_ref)
      !! Fill `f_centre` with the beta-plane Coriolis magnitude at
      !! cell centres: |f_0 + beta*(y - y_ref)|.  Mirror of
      !! `coriolis_adv_set_beta_plane` (which fills corners).  Call
      !! after `init`, before `enter_data`.
      class(ocean_epbl_t), intent(inout) :: this
      type(hgrid_t), intent(in) :: grid
      real(wp), intent(in) :: f_0, beta, y_ref
      integer :: i, j, ng
      real(wp) :: y

      ng = grid%nghost
      do j = 1, size(this%f_centre, 2)
         y = (real(j - ng, wp) - 0.5_wp)*grid%dy
         do i = 1, size(this%f_centre, 1)
            this%f_centre(i, j) = abs(f_0 + beta*(y - y_ref))
         end do
      end do
   end subroutine ocean_epbl_set_f_centre

   ! =================================================================
   ! Parsers (host-side, configure time)
   ! =================================================================

   pure function parse_epbl_mstar_scheme(name) result(tag)
      character(len=*), intent(in) :: name
      integer :: tag
      select case (trim(name))
      case ("constant")
         tag = EPBL_MSTAR_CONSTANT
      case ("om4", "OM4")
         tag = EPBL_MSTAR_OM4
      case ("rh18", "RH18", "reichl_h18")
         tag = EPBL_MSTAR_RH18
      case default
         tag = -1
      end select
   end function parse_epbl_mstar_scheme

   pure function parse_epbl_vstar_scheme(name) result(tag)
      character(len=*), intent(in) :: name
      integer :: tag
      select case (trim(name))
      case ("cube_root", "cube_root_tke")
         tag = EPBL_VSTAR_CUBE_ROOT
      case ("rh18", "RH18", "reichl_h18")
         tag = EPBL_VSTAR_RH18
      case default
         tag = -1
      end select
   end function parse_epbl_vstar_scheme

   pure function parse_epbl_combine(name) result(tag)
      character(len=*), intent(in) :: name
      integer :: tag
      select case (trim(name))
      case ("add", "additive")
         tag = EPBL_COMBINE_ADD
      case ("max")
         tag = EPBL_COMBINE_MAX
      case default
         tag = -1
      end select
   end function parse_epbl_combine

   pure function parse_epbl_lt_scheme(name) result(tag)
      character(len=*), intent(in) :: name
      integer :: tag
      select case (trim(name))
      case ("rescale")
         tag = EPBL_LT_RESCALE
      case ("additive", "add")
         tag = EPBL_LT_ADDITIVE
      case default
         tag = -1
      end select
   end function parse_epbl_lt_scheme

   ! =================================================================
   ! Pure helpers (device-callable; unit-tested directly)
   ! =================================================================

   pure subroutine epbl_find_mstar(scheme, mstar_const, mstar_cap, &
                                   mstar_coef1, c_ek, mstar_conv_adj, &
                                   cn1, cn2, cn3, cs1, cs2, &
                                   b0, ustar, bld, absf, mstar)
      !! mstar = (mechanical TKE available for entrainment) / u*^3.
      !! Schemes: constant; OM4 Ekman/Obukhov balance; RH18 fits.
      !! All followed by the optional convective reduction
      !! (mstar_conv_adj in [0,1]; the u*=0 corner multiplies by
      !! (1 - adj), matching the reference behaviour).
      !$acc routine seq
      integer, intent(in) :: scheme
      real(wp), intent(in) :: mstar_const, mstar_cap
      real(wp), intent(in) :: mstar_coef1, c_ek, mstar_conv_adj
      real(wp), intent(in) :: cn1, cn2, cn3, cs1, cs2
      real(wp), intent(in) :: b0
         !! Surface buoyancy flux (m^2/s^3); > 0 stabilizing.
      real(wp), intent(in) :: ustar
         !! Surface friction velocity (m/s), already floored.
      real(wp), intent(in) :: bld
         !! Boundary-layer depth guess (m).
      real(wp), intent(in) :: absf
         !! |Coriolis| (1/s), possibly omega-blended.
      real(wp), intent(out) :: mstar

      real(wp) :: mstar_n, mstar_s, msn_term, absf_floor
      real(wp) :: mscr_t1, mscr_t2

      absf_floor = max(absf, 1.0e-20_wp)
      select case (scheme)
      case (EPBL_MSTAR_OM4)
         mstar_s = mstar_coef1*sqrt(max(0.0_wp, b0)/(ustar*ustar*absf_floor))
         mstar_n = 0.0_wp
         if (ustar > absf_floor*bld) then
            mstar_n = c_ek*log(ustar/(absf_floor*max(bld, 1.0e-10_wp)))
         end if
         mstar = max(mstar_s, min(1.25_wp, mstar_n))
         if (mstar_cap > 0.0_wp) mstar = min(mstar_cap, mstar)
      case (EPBL_MSTAR_RH18)
         msn_term = cn2*exp(cn3*bld*absf/ustar)
         mstar_n = cn1*msn_term/(1.0_wp + msn_term)
         mstar_s = cs1*(max(0.0_wp, b0)**2*bld/(ustar**5*absf_floor))**cs2
         mstar = mstar_n + mstar_s
         if (mstar_cap > 0.0_wp) mstar = min(mstar_cap, mstar)
      case default  ! EPBL_MSTAR_CONSTANT
         mstar = mstar_const
      end select

      ! Convective reduction (no-op at the default adj = 0).
      if (mstar_conv_adj > 0.0_wp) then
         mscr_t1 = -bld*min(0.0_wp, b0)
         mscr_t2 = 2.0_wp*mstar*ustar**3
         if (mscr_t2 > 0.0_wp) then
            mstar = mstar*((1.0_wp - mstar_conv_adj)*mscr_t1 + mscr_t2)/ &
                    (mscr_t1 + mscr_t2)
         else
            mstar = mstar*(1.0_wp - mstar_conv_adj)
         end if
      end if
   end subroutine epbl_find_mstar

   pure function epbl_mixlen_shape(z_depth, mld_guess, translay_scale, &
                                   mixlen_exponent, shaped) result(shape_val)
      !! Mixing-length shape factor in [translay_scale, 1]: 1 at the
      !! surface, decaying to the transition-layer floor at the MLD.
      !! `shaped = .false.` (no MLD iteration) returns 1.
      !$acc routine seq
      real(wp), intent(in) :: z_depth
         !! Unconditional interface depth below the surface (m).
      real(wp), intent(in) :: mld_guess, translay_scale, mixlen_exponent
      logical, intent(in) :: shaped
      real(wp) :: shape_val
      real(wp) :: fr

      if ((.not. shaped) .or. translay_scale >= 1.0_wp .or. &
          translay_scale < 0.0_wp .or. mld_guess <= 0.0_wp) then
         shape_val = 1.0_wp
      else
         fr = max(0.0_wp, (mld_guess - z_depth)/mld_guess)
         if (mixlen_exponent == 2.0_wp) then
            shape_val = translay_scale + (1.0_wp - translay_scale)*fr*fr
         else
            shape_val = translay_scale + (1.0_wp - translay_scale)*fr**mixlen_exponent
         end if
      end if
   end function epbl_mixlen_shape

   pure subroutine epbl_lf17_wave_state(ustar_w, rho_ocn, u10, ustokes, kphil)
      !! LF17 statistical wave state from the water-side friction
      !! velocity alone: COARE 3.5 fixed-point inversion u* -> U10
      !! (Edson et al. 2013), then the Pierson-Moskowitz-based
      !! surface Stokes drift and Phillips peak wavenumber (Li &
      !! Fox-Kemper 2017; Breivik et al. 2016).  BLD-independent —
      !! call once per column, outside the MLD iteration.
      !$acc routine seq
      real(wp), intent(in) :: ustar_w
         !! Water-side u* (m/s), > 0 (caller floors it).
      real(wp), intent(in) :: rho_ocn
         !! Seawater reference density (kg/m^3).
      real(wp), intent(out) :: u10
         !! 10-m wind speed (m/s).
      real(wp), intent(out) :: ustokes
         !! Surface Stokes drift (m/s).
      real(wp), intent(out) :: kphil
         !! Phillips-spectrum peak wavenumber (1/m).

      real(wp) :: ustar_air, z0_smooth, z0w, alpha_ch, u10_new
      real(wp) :: hm0, f_mean, vstokes
      integer :: itt
      logical :: converged

      ustar_air = ustar_w*sqrt(rho_ocn/LT_RHO_AIR)
      z0_smooth = 0.11_wp*LT_NU_AIR/ustar_air
      u10 = ustar_air*sqrt(1000.0_wp)
      converged = .false.
      do itt = 1, 20
         alpha_ch = min(LT_CHARNOCK_MIN, LT_CHARNOCK_SLOPE*u10 + LT_CHARNOCK_ICPT)
         ! alpha can go negative below U10 ~ 3 m/s; clamp z0 positive
         ! before the log.
         z0w = max(z0_smooth + alpha_ch*ustar_air**2/GRAVITY, 1.0e-10_wp)
         u10_new = ustar_air*log(10.0_wp/z0w)/LT_VONKAR_WAVES
         if (abs(u10_new - u10) <= 1.0e-3_wp*u10) then
            u10 = u10_new
            converged = .true.
            exit
         end if
         u10 = u10_new
      end do
      if (.not. converged) u10 = 25.82_wp*ustar_air

      ustokes = LT_US_TO_U10*u10
      hm0 = LT_SWH_FROM_U10SQ*u10*u10
      f_mean = LT_FM_INTO_FP*0.877_wp*GRAVITY/ &
               (2.0_wp*LT_PI*LT_U19P5_TO_U10*u10)
      vstokes = 0.125_wp*LT_PI*LT_R_LOSS*f_mean*hm0*hm0
      kphil = 0.176_wp*ustokes/max(vstokes, 1.0e-30_wp)
   end subroutine epbl_lf17_wave_state

   pure function epbl_lf17_la(ustar_w, zsl, ustokes, kphil) result(la)
      !! Turbulent Langmuir number La = sqrt(u*/u_s_SL): the Stokes
      !! drift averaged over the surface layer of thickness `zsl`
      !! under the Phillips spectrum, in the singularity-safe form
      !! (Breivik et al. 2016 with Webb & Fox-Kemper 2015 directional
      !! spreading).  No MIN_LANGMUIR / LA_DEPTH_MIN floors — those
      !! belong to MOM6's profile-averaging wave paths, not LF17.
      !$acc routine seq
      real(wp), intent(in) :: ustar_w
         !! Water-side u* (m/s).
      real(wp), intent(in) :: zsl
         !! Surface-layer average depth (m) = la_frac_hbl * BLD.
      real(wp), intent(in) :: ustokes, kphil
         !! From `epbl_lf17_wave_state`.
      real(wp) :: la

      real(wp) :: xkz, root2kz, r1, r3, r5, ustokes_sl

      xkz = kphil*zsl
      root2kz = sqrt(2.0_wp*xkz)
      r1 = (0.302_wp - 1.68_wp*xkz)*one_m_exp_x(2.0_wp*xkz)
      r3 = (0.1264_wp + 0.64_wp*xkz)*one_m_exp_x(5.12_wp*xkz)
      if (root2kz > 1.0e-3_wp) then
         r5 = sqrt(LT_PI)*(root2kz*(-0.84_wp*erfc(root2kz) + &
                                    0.2_wp*erfc(1.6_wp*root2kz)) + &
                           0.1182_wp*(erfc(1.6_wp*root2kz) - erfc(root2kz))/root2kz)
      else
         r5 = -0.64_wp*sqrt(LT_PI)*root2kz + &
              (-0.14184_wp + 1.0839648_wp*root2kz*root2kz)
      end if
      ! Floor is a pure numerical guard (UStokes_sl -> 0+ for very
      ! deep surface layers); La then goes huge and the enhancement
      ! vanishes, which is the correct limit.
      ustokes_sl = max(ustokes*(0.715_wp + r1 + r3 + r5), 1.0e-10_wp)
      la = sqrt(ustar_w/ustokes_sl)
   end function epbl_lf17_la

   pure function one_m_exp_x(x) result(f)
      !! (1 - exp(-x)) / x, Taylor-safe at small x.
      !$acc routine seq
      real(wp), intent(in) :: x
      real(wp) :: f
      if (x < 1.0e-4_wp) then
         f = 1.0_wp - x*(0.5_wp - x/6.0_wp)
      else
         f = (1.0_wp - exp(-x))/x
      end if
   end function one_m_exp_x

   pure subroutine epbl_lt_enhance(scheme, coef, expo, max_enh, vonkar, &
                                   lac1, lac2, lac3, lac4, lac5, &
                                   la, b0, ustar, bld, absf, mstar)
      !! Apply the Langmuir enhancement to mstar (Reichl & Li 2019;
      !! Li et al. 2016 stability modification).  The modified
      !! Langmuir number folds the boundary-layer stability regime in
      !! via Ekman / Obukhov / MLD length-scale ratios, split by the
      !! sign of the surface buoyancy flux; all-zero lac coefficients
      !! give La_mod = La exactly.
      !$acc routine seq
      integer, intent(in) :: scheme
      real(wp), intent(in) :: coef, expo, max_enh, vonkar
      real(wp), intent(in) :: lac1, lac2, lac3, lac4, lac5
      real(wp), intent(in) :: la
         !! Raw turbulent Langmuir number (> 0).
      real(wp), intent(in) :: b0
         !! Surface buoyancy flux (m^2/s^3); > 0 stabilizing.
      real(wp), intent(in) :: ustar, bld, absf
      real(wp), intent(inout) :: mstar

      real(wp) :: mld_ek, ek_ob, mld_ob, la_mod, enh

      mld_ek = bld*absf/ustar
      ek_ob = abs(b0*vonkar)/(max(absf, 1.0e-20_wp)*ustar*ustar)
      mld_ob = abs(bld*b0*vonkar)/ustar**3
      if (b0 >= 0.0_wp) then
         la_mod = la*((1.0_wp + max(-0.5_wp, lac1*mld_ek)) + &
                      lac4*ek_ob + lac2*mld_ob)
      else
         la_mod = la*((1.0_wp + max(-0.5_wp, lac1*mld_ek)) + &
                      lac5*ek_ob + lac3*mld_ob)
      end if
      la_mod = max(la_mod, 1.0e-10_wp)

      if (scheme == EPBL_LT_ADDITIVE) then
         mstar = mstar + coef*la_mod**expo
      else
         enh = min(max_enh, 1.0_wp + coef*la_mod**expo)
         mstar = mstar*enh
      end if
   end subroutine epbl_lt_enhance

   pure subroutine epbl_column_kernel(grid, this, ms, hT, hS, ss, dt, &
                                      Q_heat_field, inv_rho0_cp, &
                                      Q_salt_field, inv_rho0, &
                                      nx_arg, ny_arg)
      !! Per-column EPBL solve.  One `do concurrent (j, i)` with the
      !! serial work in k inside (j -> i -> k ordering); ALL sweep
      !! state is carried in scalars (design doc D6) — the only
      !! column arrays are the six iteration-invariant workspaces
      !! filled by the prep sweep.
      !!
      !! Kinematic surface fluxes are read per-column inside the DC:
      !!   q_t_kin = Q_heat_field(i,j) / (rho0 · cp)  [degC·m/s]
      !!   q_s_kin = Q_salt_field(i,j) / rho0           [PSU·m/s]
      !! For the constant-fill default both fields are uniform, giving
      !! arithmetic identical to the former scalar-broadcast path.
      !!
      !! Algorithm:
      !!   prep:  T0, S0 and the pressure-weighted PE / steric
      !!          sensitivities per layer (downward pressure sum).
      !!   outer: MLD root-find (false position / bisection) —
      !!          mstar and the mixing-length shape depend on MLD.
      !!   sweep: interfaces Ki = nz..2 downward.  Decay mech TKE
      !!          across the layer above; rotation-reduce the
      !!          convective reservoir; closed-form energy solve for
      !!          the largest affordable Kd (the gravity-wave
      !!          column-height correction folds into PEc_core);
      !!          advance the embedded tridiagonal forward
      !!          elimination (hp_a / dX_to_dPE_a / Th_a recursions).
      type(hgrid_t), intent(in) :: grid
      type(ocean_epbl_t), intent(inout) :: this
      type(multilayer_cgrid_state_t), intent(in) :: ms
      ! assumed-shape-ok: tracer registry outer-shim — caller host-dereferences
      ! ms%tracers(idx)%hTr before passing; size varies per tracer slot
      ! (see CLAUDE.md "outer-shim + flat-impl" pattern); thermo cadence.
      real(wp), intent(in) :: hT(:, :, :)
         !! Temperature tracer hTr (degC*m), host-dereferenced.
      real(wp), intent(in) :: hS(:, :, :)  ! assumed-shape-ok: tracer registry outer-shim; thermo cadence
         !! Salinity tracer hTr (PSU*m), host-dereferenced.
      type(ocean_surface_stress_t), intent(in) :: ss
      real(wp), intent(in) :: dt
      real(wp), intent(in) :: Q_heat_field(:, :)   ! assumed-shape-ok: outer-shim pass; thermo cadence
         !! 2D heat-flux field (W/m²), host-dereferenced from sf%Q_heat.
      real(wp), intent(in) :: inv_rho0_cp
         !! Precomputed 1/(rho0·cp) multiplier.
      real(wp), intent(in) :: Q_salt_field(:, :)   ! assumed-shape-ok: outer-shim pass; thermo cadence
         !! 2D salt-flux field (kg/m²/s), host-dereferenced from sf%Q_salt.
      real(wp), intent(in) :: inv_rho0
         !! Precomputed 1/rho0 multiplier.
      integer, intent(in) :: nx_arg, ny_arg
         !! Grid extents — used to bounds-check Q_* indexing.

      integer :: i, j, nx, ny, nz
      ! prep locals
      integer :: k
      real(wp) :: q_t_kin, q_s_kin
      real(wp) :: hk, hk_eff, inv_h, t0k, s0k, dmass, dpres, p_mid, pres
      real(wp) :: dsv_dt_k, dsv_ds_k, dsv_dt_sfc, dsv_ds_sfc, h_sum
      ! forcing / environment locals
      real(wp) :: tau_xc, tau_yc, ustar, absf, idecay, mech_in
      real(wp) :: b0, ctke_sfc
      ! MLD iteration locals
      integer :: obl_it, n_its
      real(wp) :: min_mld, max_mld, mld_guess, mld_found
      real(wp) :: dmld_min, dmld_max
      logical :: have_min, have_max
      real(wp) :: mstar_val, mech_tke, conv_perel, forcing_clip
      ! sweep carried scalars
      integer :: ki, ka, kb
      real(wp) :: htot, z_int, pres_int, mld_output
      logical :: sfc_connected, sfc_disconnect
      real(wp) :: hp_a, dpe_t_a, dpe_s_a, dch_t_a, dch_s_a
      real(wp) :: th_a, sh_a, te_lag, se_lag, kddt_prev, kddt_cur
      real(wp) :: exp_kh, nstar_fc, tot_tke, dt_h, tke_here, vstar
      real(wp) :: h_ka, h_kb, hb_hs, shape_fn, hbs, mixlen, kd_g0
      real(wp) :: hp_b, th_b, sh_b, hps, bdt1, dt_c, ds_c
      real(wp) :: pec_core, colht_core, dkddt, pe_max
      real(wp) :: kd_val, tke_used, frac_bl, pe_g0, dpe_conv
      real(wp) :: b1, c1, r_reduc, te_new, se_new, surf_scale
      ! Langmuir (LF17) wave state + Langmuir number
      real(wp) :: lt_u10, lt_ustokes, lt_kphil, la_val
      ! TKE budget ledger (final iteration's values survive)
      real(wp) :: d_wind, d_conv, d_forcing, d_mixing, d_mdecay, d_cdecay

      nx = grid%nx_total
      ny = grid%ny_total
      nz = ms%nz_ml

      do concurrent(j=1:ny, i=1:nx) &
         local(k, hk, hk_eff, inv_h, t0k, s0k, dmass, dpres, p_mid, pres, &
               dsv_dt_k, dsv_ds_k, dsv_dt_sfc, dsv_ds_sfc, h_sum, &
               tau_xc, tau_yc, ustar, absf, idecay, mech_in, b0, ctke_sfc, &
               obl_it, n_its, min_mld, max_mld, mld_guess, mld_found, &
               dmld_min, dmld_max, have_min, have_max, &
               mstar_val, mech_tke, conv_perel, forcing_clip, &
               ki, ka, kb, htot, z_int, pres_int, mld_output, &
               sfc_connected, sfc_disconnect, &
               hp_a, dpe_t_a, dpe_s_a, dch_t_a, dch_s_a, &
               th_a, sh_a, te_lag, se_lag, kddt_prev, kddt_cur, &
               exp_kh, nstar_fc, tot_tke, dt_h, tke_here, vstar, &
               h_ka, h_kb, hb_hs, shape_fn, hbs, mixlen, kd_g0, &
               hp_b, th_b, sh_b, hps, bdt1, dt_c, ds_c, &
               pec_core, colht_core, dkddt, pe_max, &
               kd_val, tke_used, frac_bl, pe_g0, dpe_conv, &
               b1, c1, r_reduc, te_new, se_new, surf_scale, &
               lt_u10, lt_ustokes, lt_kphil, la_val, &
               d_wind, d_conv, d_forcing, d_mixing, d_mdecay, d_cdecay, &
               q_t_kin, q_s_kin)

         ! ---- Column prep: T0/S0 + PE/steric weights (downward) ----
         pres = 0.0_wp
         h_sum = 0.0_wp
         dsv_dt_sfc = 0.0_wp
         dsv_ds_sfc = 0.0_wp
         do k = nz, 1, -1
            hk = ms%h_layer(i, j, k)
            if (hk > 0.0_wp) then
               inv_h = 1.0_wp/(hk + H_NEGLECT)
               t0k = hT(i, j, k)*inv_h
               s0k = hS(i, j, k)*inv_h
            else
               t0k = 0.0_wp
               s0k = 0.0_wp
            end if
            dmass = this%rho0*hk
            dpres = GRAVITY*dmass
            p_mid = pres + 0.5_wp*dpres
            call eos_specvol_derivs(this%eos, t0k, s0k, p_mid, dsv_dt_k, dsv_ds_k)
            this%t0%data(i, j, k) = t0k
            this%s0%data(i, j, k) = s0k
            this%dpe_t%data(i, j, k) = dmass*p_mid*dsv_dt_k
            this%dpe_s%data(i, j, k) = dmass*p_mid*dsv_ds_k
            this%dcolht_t%data(i, j, k) = dmass*dsv_dt_k
            this%dcolht_s%data(i, j, k) = dmass*dsv_ds_k
            if (k == nz) then
               dsv_dt_sfc = dsv_dt_k
               dsv_ds_sfc = dsv_ds_k
            end if
            pres = pres + dpres
            h_sum = h_sum + hk
         end do
         h_sum = h_sum + H_NEGLECT

         ! ---- Surface forcing energetics ----
         ! Kinematic fluxes at this column:
         !   q_T_kin = Q_heat(i,j) / (rho0·cp)  [degC·m/s]
         !   q_S_kin = Q_salt(i,j) / rho0         [PSU·m/s]
         q_t_kin = inv_rho0_cp*Q_heat_field(i, j)
         q_s_kin = inv_rho0*Q_salt_field(i, j)
         ! B0 = g rho0 (dSV_dT q_T + dSV_dS q_S); > 0 stabilizing.
         ! cTKE_sfc: PE released (> 0) / required (< 0) to homogenize
         ! the freshly applied skin fluxes through the surface layer
         ! — reduces to -0.5 rho0 h_sfc B0 dt.
         b0 = GRAVITY*this%rho0*(dsv_dt_sfc*q_t_kin + dsv_ds_sfc*q_s_kin)
         ctke_sfc = -0.5_wp*GRAVITY*this%rho0**2*ms%h_layer(i, j, nz)* &
                    (q_t_kin*dt*dsv_dt_sfc + q_s_kin*dt*dsv_ds_sfc)

         tau_xc = 0.5_wp*(ss%tau_x(i, j) + ss%tau_x(i + 1, j))
         tau_yc = 0.5_wp*(ss%tau_y(i, j) + ss%tau_y(i, j + 1))
         ustar = max(sqrt(sqrt(tau_xc*tau_xc + tau_yc*tau_yc)/this%rho0), &
                     this%ustar_min)
         absf = sqrt((1.0_wp - this%omega_frac)*this%f_centre(i, j)**2 + &
                     this%omega_frac*4.0_wp*this%omega**2)
         idecay = this%tke_decay*absf/ustar
         mech_in = dt*this%rho0*ustar**3

         ! LF17 wave state is BLD-independent: compute once per
         ! column; only the surface-layer average inside the MLD
         ! iteration depends on the guess.
         la_val = 0.0_wp
         lt_u10 = 0.0_wp
         lt_ustokes = 0.0_wp
         lt_kphil = 0.0_wp
         if (this%use_lt) then
            call epbl_lf17_wave_state(ustar, this%rho0, lt_u10, lt_ustokes, lt_kphil)
         end if

         d_wind = 0.0_wp
         d_conv = 0.0_wp
         d_forcing = 0.0_wp
         d_mixing = 0.0_wp
         d_mdecay = 0.0_wp
         d_cdecay = 0.0_wp
         mld_found = 0.0_wp

         if (ms%wet_mask(i, j) <= 0.0_wp .or. h_sum <= 2.0_wp*H_NEGLECT) then
            ! Dry / land column: no mixing.
            do k = 1, nz + 1
               this%kd_int(i, j, k) = 0.0_wp
            end do
         else

            ! ---- Outer MLD iteration ----
            min_mld = 0.0_wp
            max_mld = h_sum
            dmld_min = 0.0_wp
            dmld_max = 0.0_wp
            have_min = .false.
            have_max = .false.
            mld_guess = 0.5_wp*(min_mld + max_mld)
            if (this%mld_use_prev_guess .and. this%mld(i, j) > 0.0_wp) then
               mld_guess = min(this%mld(i, j), max_mld)
            end if
            n_its = 1
            if (this%mld_iteration) n_its = this%mld_max_its

            do obl_it = 1, n_its
               ! Budget ledger restarts each iteration; the last
               ! iteration's values are the ones reported.
               d_wind = 0.0_wp
               d_conv = 0.0_wp
               d_forcing = 0.0_wp
               d_mixing = 0.0_wp
               d_mdecay = 0.0_wp
               d_cdecay = 0.0_wp

               ! (A) mstar at the current MLD guess.
               call epbl_find_mstar(this%mstar_scheme, this%mstar_const, &
                                    this%mstar_cap, this%mstar_coef1, &
                                    this%c_ek, this%mstar_conv_adj, &
                                    this%rh18_cn1, this%rh18_cn2, this%rh18_cn3, &
                                    this%rh18_cs1, this%rh18_cs2, &
                                    b0, ustar, mld_guess, absf, mstar_val)
               if (this%use_lt) then
                  la_val = epbl_lf17_la(ustar, this%la_frac_hbl*mld_guess, &
                                        lt_ustokes, lt_kphil)
                  call epbl_lt_enhance(this%lt_scheme, this%lt_enhance_coef, &
                                       this%lt_enhance_exp, this%lt_max_enhance, &
                                       this%von_karman, &
                                       this%lt_lac1, this%lt_lac2, this%lt_lac3, &
                                       this%lt_lac4, this%lt_lac5, &
                                       la_val, b0, ustar, mld_guess, absf, mstar_val)
               end if
               mech_tke = mstar_val*mech_in
               d_wind = mech_tke

               ! (B) seed the reservoirs from the surface forcing.
               if (ctke_sfc <= 0.0_wp) then
                  forcing_clip = max(ctke_sfc, -mech_tke)
                  mech_tke = mech_tke + forcing_clip
                  conv_perel = 0.0_wp
                  d_forcing = forcing_clip
               else
                  conv_perel = ctke_sfc
                  d_conv = this%nstar*ctke_sfc
               end if

               ! (C) sweep initialization at the surface layer.
               h_ka = ms%h_layer(i, j, nz) + H_NEGLECT
               hp_a = h_ka
               dpe_t_a = this%dpe_t%data(i, j, nz)
               dpe_s_a = this%dpe_s%data(i, j, nz)
               dch_t_a = this%dcolht_t%data(i, j, nz)
               dch_s_a = this%dcolht_s%data(i, j, nz)
               th_a = h_ka*this%t0%data(i, j, nz)
               sh_a = h_ka*this%s0%data(i, j, nz)
               te_lag = 0.0_wp
               se_lag = 0.0_wp
               kddt_prev = 0.0_wp
               htot = ms%h_layer(i, j, nz)
               z_int = ms%h_layer(i, j, nz)
               pres_int = GRAVITY*this%rho0*ms%h_layer(i, j, nz)
               mld_output = ms%h_layer(i, j, nz)
               sfc_connected = .true.
               this%kd_int(i, j, nz + 1) = 0.0_wp

               ! (D) downward sweep over interfaces Ki = nz .. 2.
               ! Layer above the interface: ka = Ki; below: kb = Ki-1.
               do ki = nz, 2, -1
                  ka = ki
                  kb = ki - 1
                  h_ka = ms%h_layer(i, j, ka) + H_NEGLECT
                  h_kb = ms%h_layer(i, j, kb) + H_NEGLECT
                  sfc_disconnect = .false.

                  ! (1) mechanical TKE decays across the layer above
                  !     (Ekman-scale e-folding; no decay at f = 0).
                  exp_kh = exp(-h_ka*idecay)
                  d_mdecay = d_mdecay + (1.0_wp - exp_kh)*mech_tke
                  mech_tke = mech_tke*exp_kh

                  ! (2) per-layer convective forcing accrual: only the
                  !     surface layer carries cTKE today (scalar
                  !     fluxes, no penetrating shortwave), and it
                  !     seeds the reservoirs in (B).  When SW lands,
                  !     positive cTKE(kb) accrues into conv_perel here
                  !     and negative cTKE(kb) drains tot_tke after (3).

                  ! (3) rotation-reduced convective efficiency.
                  nstar_fc = this%nstar
                  if (conv_perel > 0.0_wp .and. absf > 0.0_wp) then
                     nstar_fc = this%nstar*conv_perel/ &
                                (conv_perel + 0.2_wp* &
                                 sqrt(0.5_wp*dt*this%rho0*(absf*htot)**3*conv_perel))
                  end if
                  tot_tke = mech_tke + nstar_fc*conv_perel

                  ! (5) static-stability short-circuit: no energy and
                  !     a stable interface => no mixing here.
                  if (tot_tke <= 0.0_wp .and. &
                      0.0_wp <= (this%dcolht_t%data(i, j, kb) + this%dcolht_t%data(i, j, ka))* &
                      (this%t0%data(i, j, ka) - this%t0%data(i, j, kb)) + &
                      (this%dcolht_s%data(i, j, kb) + this%dcolht_s%data(i, j, ka))* &
                      (this%s0%data(i, j, ka) - this%s0%data(i, j, kb))) then
                     kd_val = 0.0_wp
                     sfc_disconnect = .true.
                  else
                     ! (6) velocity scale, mixing length, first-guess Kd.
                     dt_h = dt/max(0.5_wp*(h_ka + h_kb), 1.0e-15_wp*h_sum)
                     tke_here = mech_tke + this%wstar_ustar_coef*conv_perel
                     hb_hs = (h_sum - z_int)/h_sum
                     shape_fn = epbl_mixlen_shape(z_int, mld_guess, &
                                                  this%translay_scale, &
                                                  this%mixlen_exponent, &
                                                  this%mld_iteration)
                     hbs = min(hb_hs, shape_fn)
                     if (tke_here > 0.0_wp) then
                        if (this%vstar_scheme == EPBL_VSTAR_RH18) then
                           surf_scale = max(0.05_wp, 1.0_wp - htot/mld_guess)
                           vstar = this%vstar_scale_fac*surf_scale* &
                                   (this%vstar_surf_fac*ustar + &
                                    (this%wstar_ustar_coef*conv_perel/ &
                                     (dt*this%rho0))**(1.0_wp/3.0_wp))
                        else
                           vstar = this%vstar_scale_fac* &
                                   (tke_here/(dt*this%rho0))**(1.0_wp/3.0_wp)
                        end if
                        if (this%mld_iteration) then
                           mixlen = max(this%min_mix_len, &
                                        (htot*hbs*vstar)/ &
                                        (this%ekman_scale_coef*absf*htot*hbs + vstar))
                           kd_g0 = vstar*this%von_karman*mixlen
                        else
                           kd_g0 = vstar*this%von_karman*(htot*hbs*vstar)/ &
                                   (this%ekman_scale_coef*absf*htot*hbs + vstar)
                        end if
                     else
                        vstar = 0.0_wp
                        kd_g0 = 0.0_wp
                     end if

                     ! (7) pivot quantities for the layer below.
                     hp_b = h_kb
                     th_b = h_kb*this%t0%data(i, j, kb)
                     sh_b = h_kb*this%s0%data(i, j, kb)

                     ! (8) closed-form energy solve (direct path).
                     hps = hp_a + hp_b
                     bdt1 = hp_a*hp_b
                     dt_c = hp_a*th_b - hp_b*th_a
                     ds_c = hp_a*sh_b - hp_b*sh_a
                     pec_core = hp_b*(dpe_t_a*dt_c + dpe_s_a*ds_c) - &
                                hp_a*(this%dpe_t%data(i, j, kb)*dt_c + &
                                      this%dpe_s%data(i, j, kb)*ds_c)
                     colht_core = hp_b*(dch_t_a*dt_c + dch_s_a*ds_c) - &
                                  hp_a*(this%dcolht_t%data(i, j, kb)*dt_c + &
                                        this%dcolht_s%data(i, j, kb)*ds_c)
                     ! Gravity-wave radiation correction: a shrinking
                     ! column radiates energy that cannot drive mixing.
                     if (colht_core < 0.0_wp) then
                        pec_core = pec_core - pres_int*colht_core
                     end if
                     dkddt = kd_g0*dt_h
                     pe_max = pec_core/(bdt1*hps)

                     if (pe_max < 0.0_wp) then
                        ! (8a) convectively unstable: mixing RELEASES
                        ! PE.  Recompute vstar with the released
                        ! energy included; Kd from the mixing length
                        ! (not energy-limited); bank the release.
                        tke_here = mech_tke + &
                                   this%wstar_ustar_coef*(conv_perel - pe_max)
                        if (tke_here > 0.0_wp) then
                           if (this%vstar_scheme == EPBL_VSTAR_RH18) then
                              surf_scale = max(0.05_wp, 1.0_wp - htot/mld_guess)
                              vstar = this%vstar_scale_fac*surf_scale* &
                                      (this%vstar_surf_fac*ustar + &
                                       (this%wstar_ustar_coef*conv_perel/ &
                                        (dt*this%rho0))**(1.0_wp/3.0_wp))
                           else
                              vstar = this%vstar_scale_fac* &
                                      (tke_here/(dt*this%rho0))**(1.0_wp/3.0_wp)
                           end if
                           if (this%mld_iteration) then
                              mixlen = max(this%min_mix_len, &
                                           (htot*hbs*vstar)/ &
                                           (this%ekman_scale_coef*absf*htot*hbs + vstar))
                              kd_val = vstar*this%von_karman*mixlen
                           else
                              kd_val = vstar*this%von_karman*(htot*hbs*vstar)/ &
                                       (this%ekman_scale_coef*absf*htot*hbs + vstar)
                           end if
                        else
                           vstar = 0.0_wp
                           kd_val = 0.0_wp
                        end if
                        pe_g0 = pec_core*dkddt/(bdt1*(bdt1 + dkddt*hps))
                        dpe_conv = pec_core*(kd_val*dt_h)/ &
                                   (bdt1*(bdt1 + (kd_val*dt_h)*hps))
                        if (dpe_conv > 0.0_wp) then
                           kd_val = kd_g0
                           dpe_conv = pe_g0
                        end if
                        ! dpe_conv < 0 => the reservoir grows; the
                        ! reservoirs are NOT proportionally drained on
                        ! this branch.
                        conv_perel = conv_perel - dpe_conv
                        d_conv = d_conv - this%nstar*dpe_conv
                        if (sfc_connected) then
                           mld_output = mld_output + ms%h_layer(i, j, kb)
                        end if
                     else
                        ! (8b) stable: direct closed-form Kd from the
                        ! energy budget; drain the reservoirs.
                        if ((pec_core*dkddt <= &
                             tot_tke*(bdt1*(bdt1 + dkddt*hps))) .or. &
                            (pec_core <= 0.0_wp)) then
                           kd_val = kd_g0
                           tke_used = pec_core*dkddt/(bdt1*(bdt1 + dkddt*hps))
                           frac_bl = 1.0_wp
                        else
                           kd_val = (bdt1**2*tot_tke)/ &
                                    (dt_h*(pec_core - bdt1*hps*tot_tke))
                           tke_used = tot_tke
                           frac_bl = tot_tke*(bdt1*(bdt1 + dkddt*hps))/ &
                                     (pec_core*dkddt)
                        end if
                        if (sfc_connected) then
                           mld_output = mld_output + frac_bl*ms%h_layer(i, j, kb)
                        end if
                        if (frac_bl < 1.0_wp) sfc_disconnect = .true.
                        r_reduc = 0.0_wp
                        if (tot_tke > 0.0_wp .and. tot_tke > tke_used) then
                           r_reduc = (tot_tke - tke_used)/tot_tke
                        end if
                        d_mixing = d_mixing + tke_used
                        d_cdecay = d_cdecay + &
                                   (1.0_wp - r_reduc)*(this%nstar - nstar_fc)*conv_perel
                        mech_tke = r_reduc*mech_tke
                        conv_perel = r_reduc*conv_perel
                     end if
                  end if

                  this%kd_int(i, j, ki) = kd_val
                  kddt_cur = kd_val*dt_h

                  ! (9) advance the embedded forward elimination.
                  b1 = 1.0_wp/(hp_a + kddt_cur)
                  c1 = kddt_cur*b1
                  if (ki == nz) then
                     te_lag = b1*(h_ka*this%t0%data(i, j, ka))
                     se_lag = b1*(h_ka*this%s0%data(i, j, ka))
                  else
                     te_new = b1*(h_ka*this%t0%data(i, j, ka) + kddt_prev*te_lag)
                     se_new = b1*(h_ka*this%s0%data(i, j, ka) + kddt_prev*se_lag)
                     te_lag = te_new
                     se_lag = se_new
                  end if
                  hp_a = h_kb + (hp_a*b1)*kddt_cur
                  dpe_t_a = this%dpe_t%data(i, j, kb) + c1*dpe_t_a
                  dpe_s_a = this%dpe_s%data(i, j, kb) + c1*dpe_s_a
                  dch_t_a = this%dcolht_t%data(i, j, kb) + c1*dch_t_a
                  dch_s_a = this%dcolht_s%data(i, j, kb) + c1*dch_s_a
                  th_a = h_kb*this%t0%data(i, j, kb) + kddt_cur*te_lag
                  sh_a = h_kb*this%s0%data(i, j, kb) + kddt_cur*se_lag
                  kddt_prev = kddt_cur
                  if (sfc_disconnect) then
                     htot = ms%h_layer(i, j, kb)
                     sfc_connected = .false.
                  else
                     htot = htot + ms%h_layer(i, j, kb)
                  end if
                  z_int = z_int + ms%h_layer(i, j, kb)
                  pres_int = pres_int + GRAVITY*this%rho0*ms%h_layer(i, j, kb)
               end do
               this%kd_int(i, j, 1) = 0.0_wp

               ! Leftover stocks at the bed count as dissipated.
               d_mdecay = d_mdecay + mech_tke
               d_cdecay = d_cdecay + this%nstar*conv_perel

               mld_found = mld_output
               if (.not. this%mld_iteration) exit
               if (abs(mld_found - mld_guess) < this%mld_tol) exit
               if (obl_it == n_its) exit

               ! Bracket update + next guess (false position with a
               ! bisection fallback; bisection-only when configured).
               if (mld_found > mld_guess) then
                  min_mld = mld_guess
                  dmld_min = mld_found - mld_guess
                  have_min = .true.
               else
                  max_mld = mld_guess
                  dmld_max = mld_found - mld_guess
                  have_max = .true.
               end if
               if (this%mld_bisection) then
                  mld_guess = 0.5_wp*(min_mld + max_mld)
               else if (have_min .and. have_max .and. obl_it > 2 .and. &
                        mod(obl_it - 1, 4) > 0) then
                  mld_guess = min_mld + dmld_min*(max_mld - min_mld)/ &
                              (dmld_min - dmld_max)
               else if (mld_found > min_mld .and. mld_found < max_mld) then
                  mld_guess = mld_found
               else
                  mld_guess = 0.5_wp*(min_mld + max_mld)
               end if
            end do

         end if

         this%mld(i, j) = mld_found
         this%la(i, j) = la_val
         if (this%tke_diags) then
            this%tke_wind(i, j) = d_wind/dt
            this%tke_conv(i, j) = d_conv/dt
            this%tke_forcing(i, j) = d_forcing/dt
            this%tke_mixing(i, j) = d_mixing/dt
            this%tke_mech_decay(i, j) = d_mdecay/dt
            this%tke_conv_decay(i, j) = d_cdecay/dt
         end if
      end do
   end subroutine epbl_column_kernel

   pure subroutine epbl_merge_into_kv_kt(this, nx, ny, nzp1, kv, kt)
      !! Fold the EPBL diffusivity into the vmix interface fields.
      !! Called EVERY stage (the interior closure rewrites kv/kt each
      !! stage; `kd_int` itself refreshes at thermo cadence).
      !! Interior interfaces only — k=1 (bed) and k=nz+1 (surface)
      !! stay at the closed-boundary zero in both source and target.
      type(ocean_epbl_t), intent(in) :: this
      integer, intent(in) :: nx, ny, nzp1
         !! Interface-field extents (explicit shape: assumed-shape
         !! dummies in a `do concurrent` kernel make NVHPC walk the
         !! descriptor with per-launch memcpys — this runs every stage).
      real(wp), intent(inout) :: kv(nx, ny, nzp1)
         !! Momentum viscosity at interfaces; gets prandtl*kd.
      real(wp), intent(inout) :: kt(nx, ny, nzp1)
         !! Tracer diffusivity at interfaces; gets kd.

      integer :: i, j, k

      do concurrent(k=2:nzp1 - 1, j=1:ny, i=1:nx)
         if (this%combine_mode == EPBL_COMBINE_MAX) then
            kt(i, j, k) = max(kt(i, j, k), this%kd_int(i, j, k))
            kv(i, j, k) = max(kv(i, j, k), this%prandtl*this%kd_int(i, j, k))
         else
            kt(i, j, k) = kt(i, j, k) + this%kd_int(i, j, k)
            kv(i, j, k) = kv(i, j, k) + this%prandtl*this%kd_int(i, j, k)
         end if
      end do
   end subroutine epbl_merge_into_kv_kt

end module ocean_epbl
