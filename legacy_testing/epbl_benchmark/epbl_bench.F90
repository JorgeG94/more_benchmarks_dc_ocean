!! Driver for the ocean model's PRODUCTION EPBL column kernel:
!! `do concurrent` vs hand-written CUDA C.
!!
!! WHY THIS EXISTS: `ocean_vmix_compute` is 39.1% of production runtime -- the
!! single largest region in the profile
!! (<scratch>/<user>/tassie_runs/gabight_acc_120d.173705307.<sched>.log,
!! 0.1 deg, 473x297x30, 34560 steps) -- and it wraps EPBL + kappa-shear. It had
!! never been measured. Every kernel measured before this one was
!! one-thread-per-CELL in both languages, which is why they all tied. EPBL is
!! one-thread-per-COLUMN with ~630 sequential steps inside, so it is the first
!! kernel where the two languages could plausibly diverge.
!!
!! STRUCTURE (measured, not assumed -- `make collapse`): the whole scheme is ONE
!! `do concurrent (j=1:ny, i=1:nx)`, auto-collapsed to collapse(2) with 128
!! threads. Parallel width = nx_total*ny_total columns. Inside each thread:
!!   prep sweep   k = nz..1                          (nz steps)
!!   MLD iteration obl_it = 1..20, each redoing
!!     the interface sweep ki = nz..2                (<= 20*(nz-1) steps)
!!
!! STATE -- this matters more here than in any previous benchmark, because a
!! column scheme's cost is a function of its stratification and forcing, and a
!! trivially-stable or trivially-unstable column takes a fast path production
!! never takes. So the state is built to match the profiled config:
!!   * REAL bathymetry, read from gabight_bathy_0p1_473x297.f64 (dumped from
!!     <home>/analysis_gebco/gabight_bathy_sph_0p1_smooth.nc, the file
!!     the profiled run actually used). 10.9% land, mean wet depth 3983 m.
!!     Nearest-neighbour resampled to whatever nx_phys/ny_phys is asked for, so
!!     the land fraction and the shelf/abyss structure survive at every size.
!!   * zstar-like layers: ~5 m at the surface (zstar_h_surf_target = 5.0),
!!     stretched to the local bottom -- so per-column layer thicknesses vary by
!!     ~3 orders of magnitude across the domain, as in production.
!!   * T: a ~100 m mixed layer over a stratified thermocline, 14 degC surface to
!!     2 degC bottom (matching &tracer_nml T_init_surface/T_init_bottom). The
!!     profiled run reports MLD_EPBL mean = 120 m, max = 5000 m, so the target
!!     is a state that produces mixed layers of that order, not a fast path.
!!   * S = 35 (&tracer_nml initial_salinity), so B0 is temperature-driven.
!!   * tau: the MOM6 2gyre profile, taux_magnitude = 0.15 (&ocean_topo_nml),
!!     transcribed from ocean_surfstress_set_2gyre.
!!   * Q_heat: SST restoring toward 14 degC with piston_t = 1 m/day
!!     (&ocean_restore_nml). SST varies spatially, so Q_heat CHANGES SIGN across
!!     the domain -- some columns are stabilizing (8b), some convecting (8a).
!!     Both energy branches fire. Q_salt = 0.
!! The `--- state ---` block the driver prints reports what this actually
!! produced, so the claim is checkable rather than asserted.
!!
!! MEMORY: -gpu=mem:separate + MANUAL DEEP COPY, done HERE, in the scope that
!! owns the variables (RESUME_GPU_MRE gotcha 3: a deep copy inside a helper maps
!! the DUMMY, the components never attach, and you time PCIe). Both toolchains
!! then read the SAME device allocation via host_data use_device.
!!
!! SEPARATE OUTPUT STATE PER VARIANT (gotcha 5): each variant gets its own
!! ocean_epbl_t, so nothing can silently inherit "agreement OK" from the last
!! writer.
!!
!! Usage: ./epbl_bench [nx_phys] [ny_phys] [nz] [nreps] [nwarm] [cuda_sync] [its]
!!   ./epbl_bench                    -> 473 297 30, the profiled 0.1 deg config
!!   ./epbl_bench 945 594 30         -> the 0.05 deg config
!!   ./epbl_bench 473 297 30 50 10 2 1  -> also dump the MLD-iteration histogram
!! nx_phys/ny_phys are INTERIOR cells; nx_total = nx_phys + 2*nghost, nghost = 3.
program epbl_bench
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr, c_null_ptr, c_loc
   use constants, only: wp, SEAWATER_CP
   use grid, only: hgrid_t
   use epbl_stubs, only: multilayer_cgrid_state_t, ocean_surface_stress_t, &
                             ocean_surface_flux_t, EOS_VARIANT_LINEAR
   use ocean_epbl, only: ocean_epbl_t, epbl_column_kernel
   use ocean_epbl_plain, only: epbl_column_kernel_plain
   implicit none

   ! Mirror of EpblParams in epbl_kernel.cu. Field ORDER and TYPES must match
   ! exactly -- this is passed as a raw struct, so a reordering here is silent
   ! corruption, not a link error.
   type, bind(C) :: epbl_params_t
      integer(c_int) :: mstar_scheme, vstar_scheme, combine_mode
      real(c_double) :: mstar_const, mstar_cap, mstar_coef1, c_ek, mstar_conv_adj
      real(c_double) :: rh18_cn1, rh18_cn2, rh18_cn3, rh18_cs1, rh18_cs2
      real(c_double) :: nstar, tke_decay, wstar_ustar_coef, vstar_scale_fac, vstar_surf_fac
      real(c_double) :: von_karman, ekman_scale_coef, min_mix_len, mixlen_exponent, translay_scale
      integer(c_int) :: mld_iteration, mld_max_its, mld_bisection, mld_use_prev_guess
      real(c_double) :: mld_tol
      real(c_double) :: rho0, omega, omega_frac, ustar_min, prandtl
      integer(c_int) :: tke_diags
      integer(c_int) :: use_lt, lt_scheme
      real(c_double) :: lt_enhance_coef, lt_enhance_exp, lt_max_enhance, la_frac_hbl
      real(c_double) :: lt_lac1, lt_lac2, lt_lac3, lt_lac4, lt_lac5
      integer(c_int) :: eos_variant
      real(c_double) :: eos_rho0, eos_T_ref, eos_S_ref, eos_alpha_T, eos_beta_S, eos_p_ref
      real(c_double) :: inv_rho0_cp, inv_rho0, dt
      integer(c_int) :: nx, ny, nz
   end type epbl_params_t

   interface
      subroutine epbl_cuda_launch(h_layer, wet_mask, hT, hS, tau_x, tau_y, Q_heat, Q_salt, &
                                  f_centre, mld, kd_int, la, t0, s0, dpe_t, dpe_s, dcolht_t, &
                                  dcolht_s, tke_wind, tke_conv, tke_forcing, tke_mixing, &
                                  tke_mech_decay, tke_conv_decay, its_used, P, variant, &
                                  threads, sync) bind(C, name="epbl_cuda_launch")
         import :: c_double, c_int, c_ptr, epbl_params_t
         implicit none
         real(c_double), intent(in) :: h_layer(*), wet_mask(*), hT(*), hS(*)
         real(c_double), intent(in) :: tau_x(*), tau_y(*), Q_heat(*), Q_salt(*), f_centre(*)
         real(c_double), intent(inout) :: mld(*), kd_int(*), la(*)
         real(c_double), intent(inout) :: t0(*), s0(*), dpe_t(*), dpe_s(*), dcolht_t(*), dcolht_s(*)
         real(c_double), intent(inout) :: tke_wind(*), tke_conv(*), tke_forcing(*)
         real(c_double), intent(inout) :: tke_mixing(*), tke_mech_decay(*), tke_conv_decay(*)
         type(c_ptr), value :: its_used
         type(epbl_params_t), intent(in) :: P
         integer(c_int), value :: variant, threads, sync
      end subroutine
   end interface

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30
   integer, parameter :: REPS_DEF = 50, WARM_DEF = 10, NGHOST = 3
   integer, parameter :: BATHY_NX = 473, BATHY_NY = 297
   character(len=*), parameter :: BATHY_FILE = "gabight_bathy_0p1_473x297.f64"
   !! Reference dump consumed by epbl_native.cu -- see `want_dump`.
   character(len=*), parameter :: REF_FILE = "epbl_ref.f64"

   ! Profiled config: &time_nml dt_fixed = 300.0 (thermo cadence).
   real(wp), parameter :: DT = 300.0_wp
   ! &ocean_topo_nml taux_magnitude = 0.15 ; &ocean_restore_nml piston_t = 1.0,
   ! restore_sst = 14.0 ; &vcoord_nml zstar_h_surf_target = 5.0
   real(wp), parameter :: TAUX_MAG = 0.15_wp
   real(wp), parameter :: PISTON_T = 1.0_wp
   real(wp), parameter :: RESTORE_SST = 14.0_wp
   real(wp), parameter :: H_SURF_TARGET = 5.0_wp
   ! &tracer_nml
   real(wp), parameter :: T_SFC = 14.0_wp, T_BOT = 2.0_wp, S_INIT = 35.0_wp
   ! &ocean_ic_nml
   real(wp), parameter :: RHO_0 = 1035.0_wp, ALPHA_T = 0.2_wp
   real(wp), parameter :: MIXED_LAYER_M = 100.0_wp
   real(wp), parameter :: THERMOCLINE_M = 800.0_wp

   type(hgrid_t) :: grid
   type(multilayer_cgrid_state_t) :: ms
   type(ocean_surface_stress_t) :: ss
   type(ocean_surface_flux_t) :: sf
   type(ocean_epbl_t) :: e_dc, e_cf, e_ct, e_pl   ! one per variant (gotcha 5)
   type(epbl_params_t), target :: P
   real(wp), allocatable :: hT(:, :, :), hS(:, :, :)
   real(wp), allocatable :: depth(:, :), bathy(:, :)
   integer, allocatable, target :: its_used(:, :)
   integer :: hist(0:32)

   real(wp) :: t0w, t1w, ms_dc, ms_cf, ms_ct, ms_pl, mx_dc, mx_cf, mx_ct, mx_pl, tr, noise
   integer :: trial, n_trials
   real(wp) :: inv_rho0_cp, inv_rho0, gib
   real(wp) :: dmax_kd, rmax_kd, dmax_mld, rmax_mld, scale
   real(wp) :: dmax_kd_t, rmax_kd_t, dmax_mld_t, rmax_mld_t
   real(wp) :: dmax_kd_p, rmax_kd_p, dmax_mld_p, rmax_mld_p
   integer :: i, j, k, rep, nx, ny, nz, nxp, nyp, n_reps, n_warm, cu_sync, want_its
   integer :: max_its_ovr, want_dump
   integer :: nbad, ncol, nwet
   integer :: bi, bj
   real(wp) :: D, z_top, z_bot, hk, zc, tt, sst_loc, y_rel, wet
   real(wp) :: hsurf, r, denom, mld_mean, mld_max_v, mld_min_v, kd_max_v
   integer :: nmld

   nxp = iarg(1, NXP_DEF)
   nyp = iarg(2, NYP_DEF)
   nz = iarg(3, NZ_DEF)
   n_reps = iarg(4, REPS_DEF)
   n_warm = iarg(5, WARM_DEF)
   cu_sync = iarg(6, 1)
   want_its = iarg(7, 0)
   ! Diagnostic knob, NOT a production value. mld_max_its = 1 runs the sweep
   ! ONCE, which removes the root-find entirely. It is how the DC-vs-CUDA
   ! disagreement gets classified: if the kernels agree to FMA precision at 1
   ! iteration and diverge at 20, the sweep is faithful and the root-find is
   ! amplifying -- a property of the scheme. If they disagree at 1, it is a
   ! port bug. See README "Bit-identity".
   max_its_ovr = iarg(8, 20)
   ! Interleaved trials; the MIN of these is the reported time. See the timing
   ! block below for why a single pass is not trustworthy on a shared GPU.
   n_trials = iarg(9, 5)
   ! Dump the reference state for epbl_native.cu (arg 10 = 1). Writes BOTH the
   ! inputs and the outputs to REF_FILE, so the native driver can check its
   ! hand-mirrored init against this one ELEMENTWISE rather than inferring init
   ! correctness from the outputs agreeing. An init that is wrong in the last
   ! ulp and an init that is wrong structurally both show up as "MLD disagrees",
   ! and EPBL is chaotic enough (README "Bit-identity") that only the input dump
   ! can tell them apart. Costs nothing when 0.
   want_dump = iarg(10, 0)

   nx = nxp + 2*NGHOST
   ny = nyp + 2*NGHOST
   if (nxp < 1 .or. nyp < 1 .or. nz < 2) then
      write (output_unit, '(a)') 'ERROR: need nx_phys,ny_phys >= 1 and nz >= 2'
      stop 1
   end if
   grid%nx_total = nx; grid%ny_total = ny
   grid%nx_phys = nxp; grid%ny_phys = nyp
   grid%nghost = NGHOST
   grid%dx = 0.1_wp; grid%dy = 0.1_wp
   ncol = nx*ny

   ! 3 variants x (6 scratch + kd_int) 3-D arrays + h_layer/hT/hS.
   gib = (3.0_wp*7.0_wp + 3.0_wp)*real(nx, wp)*real(ny, wp)*real(nz, wp)*8.0_wp/(1024.0_wp**3)

   ! ---------------- allocate ----------------
   allocate (ms%h_layer(nx, ny, nz), ms%wet_mask(nx, ny))
   allocate (hT(nx, ny, nz), hS(nx, ny, nz))
   allocate (ss%tau_x(nx + 1, ny), ss%tau_y(nx, ny + 1))
   allocate (sf%Q_heat(nx, ny), sf%Q_salt(nx, ny))
   allocate (depth(nx, ny), bathy(BATHY_NX, BATHY_NY))
   allocate (its_used(nx, ny))
   ms%nz_ml = nz
   ms%idx_temperature = 1; ms%idx_salinity = 2
   sf%cp = SEAWATER_CP; sf%rho0 = RHO_0
   call e_dc%init(grid, nz); call e_cf%init(grid, nz)
   call e_ct%init(grid, nz); call e_pl%init(grid, nz)

   ! ---------------- real bathymetry ----------------
   call read_bathy(bathy)
   ! Nearest-neighbour resample onto the physical interior; ghosts replicate the
   ! nearest interior column (production fills ghosts from halo exchange / land).
   do j = 1, ny
      bj = clampi(nint((real(j - NGHOST, wp) - 0.5_wp)/real(nyp, wp)*real(BATHY_NY, wp) + 0.5_wp), &
                  1, BATHY_NY)
      do i = 1, nx
         bi = clampi(nint((real(i - NGHOST, wp) - 0.5_wp)/real(nxp, wp)*real(BATHY_NX, wp) &
                          + 0.5_wp), 1, BATHY_NX)
         depth(i, j) = bathy(bi, bj)
      end do
   end do

   ! ---------------- state ----------------
   nwet = 0
   do j = 1, ny
      do i = 1, nx
         D = depth(i, j)
         wet = 0.0_wp
         if (D > 1.0_wp) then
            wet = 1.0_wp
            nwet = nwet + 1
         else
            D = 0.0_wp
         end if
         ms%wet_mask(i, j) = wet
         ! zstar-like stretch: k = nz (surface) is ~H_SURF_TARGET thick, growing
         ! geometrically to the bed at k = 1. Solve the ratio so the layers sum
         ! to D exactly. Layers vary ~5 m -> ~1000 m over a 5 km column, which is
         ! what makes h_ka/h_kb (and hence the TKE decay) column-dependent.
         if (wet > 0.0_wp) then
            hsurf = min(H_SURF_TARGET, D/real(nz, wp))
            r = stretch_ratio(D, hsurf, nz)
            denom = 0.0_wp
            do k = 1, nz
               denom = denom + r**(nz - k)
            end do
            do k = 1, nz
               ms%h_layer(i, j, k) = D*r**(nz - k)/denom
            end do
         else
            do k = 1, nz
               ms%h_layer(i, j, k) = 0.0_wp
            end do
         end if
         ! T(z): a mixed layer over an exponential thermocline, with the SURFACE
         ! temperature varying by latitude. This is what sets the MLD spread and
         ! it is the single most important state choice in this benchmark.
         !
         ! The profiled domain is the ACC sector, lat -60.6 .. -30.9, and the run
         ! reports MLD_EPBL mean 120 m / max ~5000 m -- i.e. the north is
         ! stratified with a shallow mixed layer while the far south is nearly
         ! NEUTRAL and mixes to the bed. A horizontally-uniform T profile cannot
         ! reproduce that: it gives every column the same stratification and the
         ! same MLD, so every column converges in the same number of iterations
         ! and the MLD-iteration DIVERGENCE -- the entire question for a column
         ! scheme -- is artificially zero.
         !
         ! So T_surface ramps from ~T_BOT at the south wall (neutral column ->
         ! deep mixing) to ~T_SFC+4 at the north (sharp thermocline -> ~5-50 m
         ! MLD). The `--- state ---` block reports the MLD distribution that
         ! results, so this is checkable against the profiled run rather than
         ! taken on trust.
         ! The far-south band is made weakly STATICALLY UNSTABLE (surface colder,
         ! hence denser, than the water below it). That is not a fudge: it is
         ! what sustained surface buoyancy loss does, and it is the ONLY way to
         ! reach the ~5000 m mixed layers the profiled run reports. A stably
         ! stratified column is always energy-limited (branch 8b) and can never
         ! mix deeper than the wind TKE pays for -- ~110 m here. An unstable
         ! interface takes branch 8a, where mixing RELEASES PE and Kd is set by
         ! the mixing length rather than the energy budget, so it propagates to
         ! the bed. Without this band the (8a) branch NEVER FIRES and the
         ! benchmark measures half the scheme.
         y_rel = (real(j - NGHOST, wp) - 0.5_wp)/real(nyp, wp)   ! 0 south, 1 north
         y_rel = max(0.0_wp, min(1.0_wp, y_rel))
         tt = T_BOT - 0.30_wp + (T_SFC + 4.0_wp - T_BOT + 0.30_wp)*y_rel**1.5_wp
         z_top = 0.0_wp
         do k = nz, 1, -1
            hk = ms%h_layer(i, j, k)
            z_bot = z_top + hk
            zc = 0.5_wp*(z_top + z_bot)
            if (zc <= MIXED_LAYER_M) then
               hT(i, j, k) = hk*tt
            else
               hT(i, j, k) = hk*(T_BOT + (tt - T_BOT)* &
                                 exp(-(zc - MIXED_LAYER_M)/THERMOCLINE_M))
            end if
            hS(i, j, k) = hk*S_INIT
            z_top = z_bot
         end do
      end do
   end do

   ! 2gyre wind: transcribed from ocean_surfstress_set_2gyre. Physical rows only.
   ss%tau_x = 0.0_wp; ss%tau_y = 0.0_wp
   do j = NGHOST + 1, NGHOST + nyp
      y_rel = (real(j - NGHOST, wp) - 0.5_wp)/real(nyp, wp)
      do i = 1, nx + 1
         ss%tau_x(i, j) = TAUX_MAG*(1.0_wp - cos(8.0_wp*atan(1.0_wp)*y_rel))
      end do
   end do

   ! SST restoring: Q = rho0*cp*(piston/86400)*(T_target - SST). SST is the top
   ! layer's temperature plus a spatial anomaly, so Q_heat changes SIGN across
   ! the domain -> both the stabilizing (8b) and convecting (8a) branches fire.
   sf%Q_salt = 0.0_wp
   do j = 1, ny
      do i = 1, nx
         if (ms%wet_mask(i, j) > 0.0_wp) then
            sst_loc = hT(i, j, nz)/max(ms%h_layer(i, j, nz), 1.0e-10_wp) &
                      + 3.0_wp*sin(0.05_wp*real(i, wp))*cos(0.037_wp*real(j, wp))
            sf%Q_heat(i, j) = RHO_0*SEAWATER_CP*(PISTON_T/86400.0_wp)*(RESTORE_SST - sst_loc)
         else
            sf%Q_heat(i, j) = 0.0_wp
         end if
      end do
   end do

   ! Coriolis: the profiled run is spherical/planetary over lat -60.61 .. -30.9.
   ! set_f_centre is a beta-plane, so seed |f| directly across that band.
   do j = 1, ny
      do i = 1, nx
         e_dc%f_centre(i, j) = abs(2.0_wp*7.2921e-5_wp* &
                                   sin((-60.61_wp + 0.1_wp*real(j - NGHOST, wp))* &
                                       atan(1.0_wp)/45.0_wp))
      end do
   end do
   e_cf%f_centre = e_dc%f_centre; e_ct%f_centre = e_dc%f_centre
   e_pl%f_centre = e_dc%f_centre

   ! ---- knobs: the profiled config sets ONLY `enable`, so everything else is
   ! the type default. Set them explicitly anyway so the CUDA mirror and the
   ! Fortran type are provably reading the same values.
   call set_knobs(e_dc); call set_knobs(e_cf)
   call set_knobs(e_ct); call set_knobs(e_pl)
   inv_rho0_cp = 1.0_wp/(e_dc%rho0*sf%cp)
   inv_rho0 = 1.0_wp/e_dc%rho0
   call fill_params(P, e_dc, nx, ny, nz, inv_rho0_cp, inv_rho0, DT)

   write (output_unit, '(a)') repeat('=', 74)
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  domain: ', nxp, ' x ', nyp, ' x ', nz, &
      ' interior (+', NGHOST, ' ghosts/side)'
   write (output_unit, '(a,i0,a,i0,a,i0,a)') '  arrays: ', nx, ' x ', ny, ' -> ', ncol, ' COLUMNS'
   write (output_unit, '(a,i0,a,f5.1,a)') '  wet columns: ', nwet, '  (', &
      100.0_wp*real(nwet, wp)/real(ncol, wp), ' % -- from the real gabight bathymetry)'
   write (output_unit, '(a,i0,a,i0,a)') '  reps:   ', n_reps, ' timed, ', n_warm, ' warm-up'
   write (output_unit, '(a,f0.2,a,i0)') '  device memory ~', gib, ' GiB;  cuda_sync = ', cu_sync
   write (output_unit, '(a)') repeat('=', 74)

   ! ---------------- deep copy: HERE, where the variables are owned ----------
   !$acc enter data copyin(ms)
   !$acc enter data copyin(ms%h_layer, ms%wet_mask)
   !$acc enter data copyin(ss)
   !$acc enter data copyin(ss%tau_x, ss%tau_y)
   !$acc enter data copyin(sf)
   !$acc enter data copyin(sf%Q_heat, sf%Q_salt)
   !$acc enter data copyin(hT, hS)
   !$acc enter data create(its_used)
   call map_epbl(e_dc)
   call map_epbl(e_cf)
   call map_epbl(e_ct)
   call map_epbl(e_pl)

   ! ---------------- timing ----------------
   ! ⚠ THIS GPU IS SHARED. The benchmark was developed on a The HPC system analysis node
   ! whose single V100 was simultaneously running another job; the SAME binary
   ! measured 4.9 ms and 23.5 ms in consecutive runs. A single timed pass is
   ! therefore worthless here.
   !
   ! Contention can only ever ADD time, so the MINIMUM over repeated trials is
   ! the best estimator of the uncontended cost. Trials are INTERLEAVED
   ! (dc, cf, ct, dc, cf, ct, ...) rather than blocked, so no variant can be
   ! systematically advantaged by a quiet window. Both min and max are reported:
   ! if max/min is near 1 the machine was quiet and the numbers stand on their
   ! own; if it is large, only the min means anything -- and the driver says so.
   do rep = 1, n_warm
      call epbl_column_kernel(grid, e_dc, ms, hT, hS, ss, DT, sf%Q_heat, inv_rho0_cp, &
                              sf%Q_salt, inv_rho0, nx, ny)
   end do
   !$acc wait

   do rep = 1, n_warm
      call run_plain()
   end do
   !$acc wait
   ms_dc = huge(1.0_wp); ms_cf = huge(1.0_wp); ms_ct = huge(1.0_wp); ms_pl = huge(1.0_wp)
   mx_dc = 0.0_wp; mx_cf = 0.0_wp; mx_ct = 0.0_wp; mx_pl = 0.0_wp
   do trial = 1, n_trials
      t0w = wall()
      do rep = 1, n_reps
         call epbl_column_kernel(grid, e_dc, ms, hT, hS, ss, DT, sf%Q_heat, inv_rho0_cp, &
                                 sf%Q_salt, inv_rho0, nx, ny)
      end do
      !$acc wait
      t1w = wall()
      tr = (t1w - t0w)*1000.0_wp/real(n_reps, wp)
      ms_dc = min(ms_dc, tr); mx_dc = max(mx_dc, tr)

      t0w = wall()
      do rep = 1, n_reps
         call run_plain()
      end do
      !$acc wait
      t1w = wall()
      tr = (t1w - t0w)*1000.0_wp/real(n_reps, wp)
      ms_pl = min(ms_pl, tr); mx_pl = max(mx_pl, tr)

      tr = time_cuda(e_cf, 0)
      ms_cf = min(ms_cf, tr); mx_cf = max(mx_cf, tr)
      tr = time_cuda(e_ct, 1)
      ms_ct = min(ms_ct, tr); mx_ct = max(mx_ct, tr)
   end do

   !$acc update self(e_dc%kd_int, e_dc%mld)
   !$acc update self(e_cf%kd_int, e_cf%mld)
   !$acc update self(e_ct%kd_int, e_ct%mld)
   !$acc update self(e_pl%kd_int, e_pl%mld)

   ! ---------------- agreement ----------------
   call cmp3(e_dc%kd_int, e_cf%kd_int, nx, ny, nz + 1, dmax_kd, rmax_kd, nbad)
   call cmp2(e_dc%mld, e_cf%mld, nx, ny, dmax_mld, rmax_mld)
   call cmp3(e_dc%kd_int, e_ct%kd_int, nx, ny, nz + 1, dmax_kd_t, rmax_kd_t, nbad)
   call cmp2(e_dc%mld, e_ct%mld, nx, ny, dmax_mld_t, rmax_mld_t)
   ! DC vs DC-plain is a SOURCE transform in ONE language -- it must be EXACTLY
   ! 0.0, not merely small. Anything else is a bug in the signature fix.
   call cmp3(e_dc%kd_int, e_pl%kd_int, nx, ny, nz + 1, dmax_kd_p, rmax_kd_p, nbad)
   call cmp2(e_dc%mld, e_pl%mld, nx, ny, dmax_mld_p, rmax_mld_p)

   ! ---------------- state report ----------------
   mld_mean = 0.0_wp; mld_max_v = 0.0_wp; mld_min_v = huge(1.0_wp); nmld = 0
   do j = 1, ny
      do i = 1, nx
         if (ms%wet_mask(i, j) > 0.0_wp) then
            mld_mean = mld_mean + e_dc%mld(i, j)
            mld_max_v = max(mld_max_v, e_dc%mld(i, j))
            mld_min_v = min(mld_min_v, e_dc%mld(i, j))
            nmld = nmld + 1
         end if
      end do
   end do
   if (nmld > 0) mld_mean = mld_mean/real(nmld, wp)
   kd_max_v = maxval(e_dc%kd_int)

   write (output_unit, '(a)') '  --- state (compare to the profiled run: MLD mean 120 m, max ~5000 m) ---'
   write (output_unit, '(a,f10.3,a)') '  MLD_EPBL mean (wet)    : ', mld_mean, ' m'
   write (output_unit, '(a,f10.3,a)') '  MLD_EPBL min  (wet)    : ', mld_min_v, ' m'
   write (output_unit, '(a,f10.3,a)') '  MLD_EPBL max  (wet)    : ', mld_max_v, ' m'
   write (output_unit, '(a,es12.5,a)') '  max Kd_EPBL            : ', kd_max_v, ' m^2/s'
   write (output_unit, '(a)') ''
   write (output_unit, '(a,i0,a)') '  --- timings: MIN of ', n_trials, &
      ' interleaved trials (see note below) ---'
   write (output_unit, '(a,f10.4,a,f9.4,a)') '  EPBL do concurrent (production)       : ', ms_dc, &
      ' ms/rep   (worst ', mx_dc, ')'
   write (output_unit, '(a,f10.4,a,f9.4,a)') '  DC + signature fix (plain dummies)    : ', ms_pl, &
      ' ms/rep   (worst ', mx_pl, ')'
   write (output_unit, '(a,f10.4,a,f9.4,a)') '  CUDA C faithful (1 thr/col, blk 128)  : ', ms_cf, &
      ' ms/rep   (worst ', mx_cf, ')'
   write (output_unit, '(a,f10.4,a,f9.4,a)') '  CUDA C + __launch_bounds__ (occupancy): ', ms_ct, &
      ' ms/rep   (worst ', mx_ct, ')'
   write (output_unit, '(a,f10.3,a)') '  ratio  dc / dc_signature_fix         : ', ms_dc/ms_pl, ' x'
   write (output_unit, '(a,f10.3,a)') '  ratio  dc / cuda_faithful             : ', ms_dc/ms_cf, ' x'
   write (output_unit, '(a,f10.3,a)') '  ratio  dc / cuda_tuned                : ', ms_dc/ms_ct, ' x'
   ! Contention self-check. If the machine was busy, say so loudly rather than
   ! letting a reader quote a number that a neighbouring job produced.
   noise = max(max(mx_dc/max(ms_dc, 1.0e-12_wp), mx_pl/max(ms_pl, 1.0e-12_wp)), &
               max(mx_cf/max(ms_cf, 1.0e-12_wp), mx_ct/max(ms_ct, 1.0e-12_wp)))
   if (noise > 1.15_wp) then
      write (output_unit, '(a,f5.2,a)') '  ⚠ GPU CONTENDED: worst/best = ', noise, &
         ' x across trials. Another process was'
      write (output_unit, '(a)') '    using the device. The MIN values above are still the best'
      write (output_unit, '(a)') '    available estimate (contention only adds time), but re-run'
      write (output_unit, '(a)') '    on a quiet GPU before quoting them.'
   else
      write (output_unit, '(a,f5.2,a)') '  GPU quiet: worst/best = ', noise, &
         ' x across trials -- timings are clean.'
   end if
   write (output_unit, '(a)') ''
   write (output_unit, '(a,es12.5,a,es12.5)') '  DC vs CUDA-faithful  kd_int  max|d| ', dmax_kd, &
      '   max rel ', rmax_kd
   write (output_unit, '(a,es12.5,a,es12.5)') '                       mld     max|d| ', dmax_mld, &
      '   max rel ', rmax_mld
   write (output_unit, '(a,es12.5,a,es12.5)') '  DC vs CUDA-tuned     kd_int  max|d| ', dmax_kd_t, &
      '   max rel ', rmax_kd_t
   write (output_unit, '(a,es12.5,a,es12.5)') '                       mld     max|d| ', dmax_mld_t, &
      '   max rel ', rmax_mld_t
   write (output_unit, '(a,es12.5,a,es12.5)') '  DC vs DC-signature-fix  kd_int max|d| ', dmax_kd_p, &
      '   mld max|d| ', dmax_mld_p
   if (max(dmax_kd_p, dmax_mld_p) == 0.0_wp) then
      write (output_unit, '(a)') '     -> BIT-IDENTICAL (exactly 0.0), as a same-language source'
      write (output_unit, '(a)') '        transform must be.'
   else
      write (output_unit, '(a)') '     -> *** BUG in the signature fix: a same-language source'
      write (output_unit, '(a)') '        transform MUST be exactly 0.0. ***'
   end if
   write (output_unit, '(a)') ''
   if (max(rmax_kd, rmax_mld) < 1.0e-12_wp) then
      write (output_unit, '(a)') '  agreement : OK (<1e-12 rel -> FMA contraction only)'
   else
      call classify_disagreement()
   end if

   if (want_its /= 0) call report_divergence()
   if (want_dump /= 0) call write_ref()

   write (output_unit, '(a)') repeat('=', 74)

contains

   !! Knob values: production's `ocean_epbl_t` defaults, because the profiled
   !! namelist sets only `enable`. Written out so the CUDA mirror is provably
   !! reading the same numbers.
   subroutine set_knobs(e)
      type(ocean_epbl_t), intent(inout) :: e
      e%enable = .true.
      e%rho0 = RHO_0
      e%eos%variant = EOS_VARIANT_LINEAR   ! run log: "EOS variant: linear"
      e%eos%rho0 = RHO_0                   ! &ocean_ic_nml rho_0
      e%eos%alpha_T = ALPHA_T              ! &ocean_ic_nml alpha_T
      e%eos%beta_S = 7.6e-4_wp             ! type default (nml does not set it)
      e%eos%T_ref = 10.0_wp
      e%eos%S_ref = 35.0_wp
      e%mld_max_its = max_its_ovr        ! 20 = production default
      e%mld = 0.0_wp
      e%kd_int = 0.0_wp
      e%la = 0.0_wp
   end subroutine set_knobs

   subroutine fill_params(Pp, e, nxl, nyl, nzl, ircp, ir0, dtl)
      type(epbl_params_t), intent(out) :: Pp
      type(ocean_epbl_t), intent(in) :: e
      integer, intent(in) :: nxl, nyl, nzl
      real(wp), intent(in) :: ircp, ir0, dtl
      Pp%mstar_scheme = e%mstar_scheme; Pp%vstar_scheme = e%vstar_scheme
      Pp%combine_mode = e%combine_mode
      Pp%mstar_const = e%mstar_const; Pp%mstar_cap = e%mstar_cap
      Pp%mstar_coef1 = e%mstar_coef1; Pp%c_ek = e%c_ek; Pp%mstar_conv_adj = e%mstar_conv_adj
      Pp%rh18_cn1 = e%rh18_cn1; Pp%rh18_cn2 = e%rh18_cn2; Pp%rh18_cn3 = e%rh18_cn3
      Pp%rh18_cs1 = e%rh18_cs1; Pp%rh18_cs2 = e%rh18_cs2
      Pp%nstar = e%nstar; Pp%tke_decay = e%tke_decay
      Pp%wstar_ustar_coef = e%wstar_ustar_coef
      Pp%vstar_scale_fac = e%vstar_scale_fac; Pp%vstar_surf_fac = e%vstar_surf_fac
      Pp%von_karman = e%von_karman; Pp%ekman_scale_coef = e%ekman_scale_coef
      Pp%min_mix_len = e%min_mix_len; Pp%mixlen_exponent = e%mixlen_exponent
      Pp%translay_scale = e%translay_scale
      Pp%mld_iteration = merge(1, 0, e%mld_iteration)
      Pp%mld_max_its = e%mld_max_its
      Pp%mld_bisection = merge(1, 0, e%mld_bisection)
      Pp%mld_use_prev_guess = merge(1, 0, e%mld_use_prev_guess)
      Pp%mld_tol = e%mld_tol
      Pp%rho0 = e%rho0; Pp%omega = e%omega; Pp%omega_frac = e%omega_frac
      Pp%ustar_min = e%ustar_min; Pp%prandtl = e%prandtl
      Pp%tke_diags = merge(1, 0, e%tke_diags)
      Pp%use_lt = merge(1, 0, e%use_lt); Pp%lt_scheme = e%lt_scheme
      Pp%lt_enhance_coef = e%lt_enhance_coef; Pp%lt_enhance_exp = e%lt_enhance_exp
      Pp%lt_max_enhance = e%lt_max_enhance; Pp%la_frac_hbl = e%la_frac_hbl
      Pp%lt_lac1 = e%lt_lac1; Pp%lt_lac2 = e%lt_lac2; Pp%lt_lac3 = e%lt_lac3
      Pp%lt_lac4 = e%lt_lac4; Pp%lt_lac5 = e%lt_lac5
      Pp%eos_variant = e%eos%variant; Pp%eos_rho0 = e%eos%rho0
      Pp%eos_T_ref = e%eos%T_ref; Pp%eos_S_ref = e%eos%S_ref
      Pp%eos_alpha_T = e%eos%alpha_T; Pp%eos_beta_S = e%eos%beta_S
      Pp%eos_p_ref = e%eos%p_ref
      Pp%inv_rho0_cp = ircp; Pp%inv_rho0 = ir0; Pp%dt = dtl
      Pp%nx = nxl; Pp%ny = nyl; Pp%nz = nzl
   end subroutine fill_params

   !! Deep copy in the OWNING scope: parent struct first, then each payload,
   !! then each scratch buffer's parent and ITS payload (gotcha 3).
   subroutine map_epbl(e)
      type(ocean_epbl_t), intent(inout) :: e
      !$acc enter data copyin(e)
      !$acc enter data copyin(e%f_centre, e%mld, e%kd_int, e%la)
      !$acc enter data copyin(e%tke_wind, e%tke_conv, e%tke_forcing)
      !$acc enter data copyin(e%tke_mixing, e%tke_mech_decay, e%tke_conv_decay)
      !$acc enter data copyin(e%t0, e%s0, e%dpe_t, e%dpe_s, e%dcolht_t, e%dcolht_s)
      !$acc enter data create(e%t0%data, e%s0%data, e%dpe_t%data, e%dpe_s%data)
      !$acc enter data create(e%dcolht_t%data, e%dcolht_s%data)
   end subroutine map_epbl

   !! Time a CUDA variant. Warm-up is untimed and always synchronised; the
   !! trailing sync'd launch closes the timing window when cuda_sync = 0.
   function time_cuda(e, variant) result(msr)
      type(ocean_epbl_t), intent(inout) :: e
      integer, intent(in) :: variant
      real(wp) :: msr, ta, tb
      integer :: r
      do r = 1, n_warm
         call fire(e, variant, 1)
      end do
      ta = wall()
      do r = 1, n_reps
         call fire(e, variant, cu_sync)
      end do
      call fire(e, variant, 1)
      tb = wall()
      msr = (tb - ta)*1000.0_wp/real(n_reps + 1, wp)
   end function time_cuda

   !! The signature-fix variant: identical body, plain explicit-shape dummies.
   subroutine run_plain()
      call epbl_column_kernel_plain(grid, e_pl, ms, ss, DT, inv_rho0_cp, inv_rho0, &
                                    nx, ny, nx, ny, nz, hT, hS, ms%h_layer, ms%wet_mask, &
                                    ss%tau_x, ss%tau_y, sf%Q_heat, sf%Q_salt, e_pl%f_centre, &
                                    e_pl%mld, e_pl%la, e_pl%kd_int, e_pl%t0%data, e_pl%s0%data, &
                                    e_pl%dpe_t%data, e_pl%dpe_s%data, e_pl%dcolht_t%data, &
                                    e_pl%dcolht_s%data, e_pl%tke_wind, e_pl%tke_conv, &
                                    e_pl%tke_forcing, e_pl%tke_mixing, e_pl%tke_mech_decay, &
                                    e_pl%tke_conv_decay)
   end subroutine run_plain

   subroutine fire(e, variant, sync)
      type(ocean_epbl_t), intent(inout) :: e
      integer, intent(in) :: variant, sync
      !$acc host_data use_device(ms%h_layer, ms%wet_mask, hT, hS, ss%tau_x, ss%tau_y, &
      !$acc                      sf%Q_heat, sf%Q_salt, e%f_centre, e%mld, e%kd_int, e%la, &
      !$acc                      e%t0%data, e%s0%data, e%dpe_t%data, e%dpe_s%data, &
      !$acc                      e%dcolht_t%data, e%dcolht_s%data, e%tke_wind, e%tke_conv, &
      !$acc                      e%tke_forcing, e%tke_mixing, e%tke_mech_decay, e%tke_conv_decay)
      call epbl_cuda_launch(ms%h_layer, ms%wet_mask, hT, hS, ss%tau_x, ss%tau_y, &
                            sf%Q_heat, sf%Q_salt, e%f_centre, e%mld, e%kd_int, e%la, &
                            e%t0%data, e%s0%data, e%dpe_t%data, e%dpe_s%data, &
                            e%dcolht_t%data, e%dcolht_s%data, e%tke_wind, e%tke_conv, &
                            e%tke_forcing, e%tke_mixing, e%tke_mech_decay, e%tke_conv_decay, &
                            c_null_ptr, P, variant, 128, sync)
      !$acc end host_data
   end subroutine fire

   !! Quantify the divergence hypothesis: run the CUDA kernel once with the
   !! its_used diagnostic enabled and report the per-column iteration count and,
   !! crucially, the WARP MAX -- a warp costs its slowest lane, so the ratio of
   !! (mean warp max) to (mean per column) IS the divergence tax.
   subroutine report_divergence()
      integer :: ii, jj, w, lane, cnt, mx, tot_col, tot_warp, nwarp, mx_all
      real(wp) :: mean_col, mean_warp
      !$acc host_data use_device(ms%h_layer, ms%wet_mask, hT, hS, ss%tau_x, ss%tau_y, &
      !$acc                      sf%Q_heat, sf%Q_salt, e_cf%f_centre, e_cf%mld, e_cf%kd_int, &
      !$acc                      e_cf%la, e_cf%t0%data, e_cf%s0%data, e_cf%dpe_t%data, &
      !$acc                      e_cf%dpe_s%data, e_cf%dcolht_t%data, e_cf%dcolht_s%data, &
      !$acc                      e_cf%tke_wind, e_cf%tke_conv, e_cf%tke_forcing, &
      !$acc                      e_cf%tke_mixing, e_cf%tke_mech_decay, e_cf%tke_conv_decay, &
      !$acc                      its_used)
      call epbl_cuda_launch(ms%h_layer, ms%wet_mask, hT, hS, ss%tau_x, ss%tau_y, &
                            sf%Q_heat, sf%Q_salt, e_cf%f_centre, e_cf%mld, e_cf%kd_int, &
                            e_cf%la, e_cf%t0%data, e_cf%s0%data, e_cf%dpe_t%data, &
                            e_cf%dpe_s%data, e_cf%dcolht_t%data, e_cf%dcolht_s%data, &
                            e_cf%tke_wind, e_cf%tke_conv, e_cf%tke_forcing, &
                            e_cf%tke_mixing, e_cf%tke_mech_decay, e_cf%tke_conv_decay, &
                            c_loc(its_used), P, 0, 128, 1)
      !$acc end host_data
      !$acc update self(its_used)

      hist = 0
      do jj = 1, ny
         do ii = 1, nx
            cnt = min(max(its_used(ii, jj), 0), 32)
            hist(cnt) = hist(cnt) + 1
         end do
      end do
      write (output_unit, '(a)') ''
      write (output_unit, '(a)') '  --- MLD-iteration divergence (the whole point of a column scheme) ---'
      write (output_unit, '(a)') '   iters   columns'
      do w = 0, 32
         if (hist(w) > 0) write (output_unit, '(i8,i10)') w, hist(w)
      end do
      ! Warps are laid out exactly as the kernel indexes them: linear over
      ! (i + nx*j), 32 consecutive columns per warp.
      tot_col = 0; tot_warp = 0; nwarp = 0; mx_all = 0
      do w = 0, (ncol - 1)/32
         mx = 0
         do lane = 0, 31
            cnt = w*32 + lane
            if (cnt >= ncol) cycle
            ii = mod(cnt, nx) + 1
            jj = cnt/nx + 1
            tot_col = tot_col + its_used(ii, jj)
            mx = max(mx, its_used(ii, jj))
         end do
         tot_warp = tot_warp + mx*32
         nwarp = nwarp + 1
         mx_all = max(mx_all, mx)
      end do
      mean_col = real(tot_col, wp)/real(ncol, wp)
      mean_warp = real(tot_warp, wp)/real(ncol, wp)
      write (output_unit, '(a,f8.3)') '  mean iterations per COLUMN        : ', mean_col
      write (output_unit, '(a,f8.3)') '  mean iterations per WARP LANE-SLOT: ', mean_warp
      write (output_unit, '(a,i8)') '  worst warp max                    : ', mx_all
      if (mean_col > 0.0_wp) then
         write (output_unit, '(a,f8.3,a)') '  DIVERGENCE TAX (warp/column)      : ', &
            mean_warp/mean_col, ' x  <- work the warp does but throws away'
      end if
   end subroutine report_divergence

   !! A raw max|diff| cannot tell a PORT BUG from a marginal-branch flip, and
   !! this scheme has both a hard branch (the static-stability short-circuit,
   !! kd_val -> 0) and a layer-quantised output (mld_output accrues WHOLE
   !! layers via frac_bl). So classify instead of asserting:
   !!   * a port bug is SYSTEMATIC -- most wet columns move, and they move by
   !!     amounts unrelated to the layer grid;
   !!   * branch sensitivity is RARE -- only columns sitting within an ulp of a
   !!     branch boundary move, and mld moves by ~one layer thickness.
   subroutine classify_disagreement()
      integer :: ii, jj, ndiff, nwet_l, nlayer_sized
      real(wp) :: d, relmld, hbase, frac
      ndiff = 0; nwet_l = 0; nlayer_sized = 0
      do jj = 1, ny
         do ii = 1, nx
            if (ms%wet_mask(ii, jj) <= 0.0_wp) cycle
            nwet_l = nwet_l + 1
            d = abs(e_dc%mld(ii, jj) - e_cf%mld(ii, jj))
            relmld = d/max(abs(e_dc%mld(ii, jj)), 1.0e-30_wp)
            if (relmld > 1.0e-12_wp) then
               ndiff = ndiff + 1
               ! Is the jump ~one layer thick? Compare to the thickest layer
               ! within the mixed layer -- mld_output accrues h_layer(kb).
               hbase = maxval(ms%h_layer(ii, jj, max(nz - 6, 1):nz))
               if (d <= 3.0_wp*hbase) nlayer_sized = nlayer_sized + 1
            end if
         end do
      end do
      frac = 100.0_wp*real(ndiff, wp)/real(max(nwet_l, 1), wp)
      write (output_unit, '(a)') '  agreement : NOT bit-comparable -- classifying:'
      write (output_unit, '(a,i0,a,i0,a,f7.4,a)') '    wet columns whose MLD moved >1e-12 rel: ', &
         ndiff, ' / ', nwet_l, '  (', frac, ' %)'
      write (output_unit, '(a,i0,a,i0,a)') '    of those, moved by <= ~1 layer thickness : ', &
         nlayer_sized, ' / ', ndiff, ''
      if (frac < 5.0_wp .and. nlayer_sized >= (95*ndiff)/100) then
         write (output_unit, '(a)') '    VERDICT: branch sensitivity, NOT a port bug -- rare, and'
         write (output_unit, '(a)') '             layer-quantised. A sub-ulp difference flips the'
         write (output_unit, '(a)') '             marginal short-circuit at the mixed-layer base,'
         write (output_unit, '(a)') '             so mld jumps by one WHOLE layer. PROVEN, not'
         write (output_unit, '(a)') '             assumed: `make nofma` rebuilds both sides with'
         write (output_unit, '(a)') '             FMA contraction off and the disagreement'
         write (output_unit, '(a)') '             collapses to ~1e-10 m. See README "Bit-identity".'
      else
         write (output_unit, '(a)') '    VERDICT: *** SUSPECT *** -- too widespread and/or not'
         write (output_unit, '(a)') '             layer-quantised. Treat as a PORT BUG.'
      end if
   end subroutine classify_disagreement

   subroutine cmp3(a, b, n1, n2, n3, dmax, rmax, nb)
      real(wp), intent(in) :: a(:, :, :), b(:, :, :)
      integer, intent(in) :: n1, n2, n3
      real(wp), intent(out) :: dmax, rmax
      integer, intent(out) :: nb
      integer :: ii, jj, kk
      real(wp) :: sc, d
      dmax = 0.0_wp; rmax = 0.0_wp; nb = 0
      do kk = 1, n3
         do jj = 1, n2
            do ii = 1, n1
               d = abs(a(ii, jj, kk) - b(ii, jj, kk))
               dmax = max(dmax, d)
               sc = max(abs(a(ii, jj, kk)), abs(b(ii, jj, kk)))
               if (sc > 1.0e-30_wp) then
                  rmax = max(rmax, d/sc)
                  if (d/sc > 1.0e-12_wp) nb = nb + 1
               end if
            end do
         end do
      end do
   end subroutine cmp3

   subroutine cmp2(a, b, n1, n2, dmax, rmax)
      real(wp), intent(in) :: a(:, :), b(:, :)
      integer, intent(in) :: n1, n2
      real(wp), intent(out) :: dmax, rmax
      integer :: ii, jj
      real(wp) :: sc, d
      dmax = 0.0_wp; rmax = 0.0_wp
      do jj = 1, n2
         do ii = 1, n1
            d = abs(a(ii, jj) - b(ii, jj))
            dmax = max(dmax, d)
            sc = max(abs(a(ii, jj)), abs(b(ii, jj)))
            if (sc > 1.0e-30_wp) rmax = max(rmax, d/sc)
         end do
      end do
   end subroutine cmp2

   !! Dump the inputs AND the outputs for epbl_native.cu to check itself
   !! against. Raw stream, native endianness, Fortran array order -- the native
   !! driver linearises identically, so a mismatch is a real disagreement and
   !! not a layout artefact.
   !!
   !! The INPUTS are the load-bearing half. epbl_native.cu re-derives the whole
   !! state from scratch in C++ (bathymetry resample, zstar stretch, T profile,
   !! 2gyre wind, SST restoring, Coriolis). If any of that lands even one ulp
   !! away from what nvfortran computed, the kernels are being fed different
   !! problems and every downstream number is meaningless -- so the native
   !! driver requires these to be BIT-IDENTICAL, not merely close.
   subroutine write_ref()
      integer :: u, st
      open (newunit=u, file=REF_FILE, form='unformatted', access='stream', &
            status='replace', iostat=st)
      if (st /= 0) then
         write (output_unit, '(a)') 'ERROR: cannot write '//REF_FILE
         stop 1
      end if
      write (u) int(nx, c_int), int(ny, c_int), int(nz, c_int)
      ! inputs, in the order epbl_native.cu reads them
      write (u) ms%h_layer, ms%wet_mask, hT, hS
      write (u) ss%tau_x, ss%tau_y, sf%Q_heat, sf%Q_salt, e_dc%f_centre
      ! outputs, per variant
      write (u) e_cf%mld, e_cf%kd_int      ! CUDA faithful, launched from Fortran
      write (u) e_ct%mld, e_ct%kd_int      ! CUDA tuned, launched from Fortran
      write (u) e_dc%mld, e_dc%kd_int      ! do concurrent
      close (u)
      write (output_unit, '(a)') '  wrote '//REF_FILE//' (inputs + mld/kd_int for cf, ct, dc)'
      write (output_unit, '(a)') '     -> ./epbl_native checks itself against this.'
   end subroutine write_ref

   subroutine read_bathy(b)
      real(wp), intent(out) :: b(BATHY_NX, BATHY_NY)
      integer :: u, st
      open (newunit=u, file=BATHY_FILE, form='unformatted', access='stream', &
            status='old', iostat=st)
      if (st /= 0) then
         write (output_unit, '(a)') 'ERROR: cannot open '//BATHY_FILE//'.'
         write (output_unit, '(a)') '  It is the REAL gabight bathymetry and the state depends'
         write (output_unit, '(a)') '  on it. Regenerate it -- see README.md "Files".'
         stop 1
      end if
      read (u, iostat=st) b
      close (u)
      if (st /= 0) then
         write (output_unit, '(a)') 'ERROR: short read on '//BATHY_FILE
         stop 1
      end if
   end subroutine read_bathy

   !! Geometric layer ratio r such that hsurf*(1 + r + ... + r^(nz-1)) = D.
   !! Bisection; r = 1 (uniform) when the column is too shallow to stretch.
   real(wp) function stretch_ratio(D, hsurf, nzl)
      real(wp), intent(in) :: D, hsurf
      integer, intent(in) :: nzl
      real(wp) :: lo, hi, mid, s
      integer :: it, kk
      if (hsurf*real(nzl, wp) >= D) then
         stretch_ratio = 1.0_wp
         return
      end if
      lo = 1.0_wp; hi = 3.0_wp
      do it = 1, 60
         mid = 0.5_wp*(lo + hi)
         s = 0.0_wp
         do kk = 0, nzl - 1
            s = s + mid**kk
         end do
         if (hsurf*s < D) then
            lo = mid
         else
            hi = mid
         end if
      end do
      stretch_ratio = 0.5_wp*(lo + hi)
   end function stretch_ratio

   integer function clampi(v, lo, hi)
      integer, intent(in) :: v, lo, hi
      clampi = max(lo, min(hi, v))
   end function clampi

   integer function iarg(kk, dflt)
      integer, intent(in) :: kk, dflt
      character(len=32) :: buf
      integer :: ln, st
      iarg = dflt
      if (command_argument_count() < kk) return
      call get_command_argument(kk, buf, ln, st)
      if (st /= 0 .or. ln == 0) return
      read (buf, *, iostat=st) iarg
      if (st /= 0) iarg = dflt
   end function iarg

   function wall() result(t)
      real(wp) :: t
      integer(int64) :: cnt, rate
      call system_clock(cnt, rate)
      t = real(cnt, wp)/real(rate, wp)
   end function wall

end program epbl_bench
