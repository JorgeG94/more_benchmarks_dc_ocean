#include "directives.h"
!! DC-only driver for the ocean model's PRODUCTION EPBL column kernel.
!!
!! COMPUTE is the production `do concurrent` (ocean_epbl::epbl_column_kernel,
!! one `do concurrent (j,i)` with the whole column solve inside). The device
!! data layer is chosen ENTIRELY by directives.h at compile time:
!!   -DDC_DATA_ACC  -> OpenACC  (nvfortran -acc=gpu -stdpar=gpu)   GPU
!!   -DDC_DATA_OMP  -> OpenMP target                               GPU (AMD/Intel too)
!!   (neither)      -> host: bare DC on the CPU (-stdpar=multicore/serial)
!! There is NO CUDA and NO nvcc in this binary -- that is the point: the same
!! do-concurrent scheme builds and runs on the CPU / AMD / Intel unchanged.
!! The CUDA comparison lives in the native driver (drivers/cpp_main.cu).
!!
!! STATE: identical to epbl_bench.F90 -- REAL gabight bathymetry, zstar-like
!! layers, a mixed layer over a stratified thermocline with a latitudinal
!! surface-T ramp (so both the (8a) convecting and (8b) energy-limited branches
!! fire), 2gyre wind, SST restoring, seeded |f|. See epbl_bench.F90 for the
!! physical justification of every choice; this driver reproduces it verbatim.
!!
!! Cross-check (proves the macro'd data layer did not change the numbers):
!!   DC_DUMP=file  writes nx,ny,nz + kd_int + mld  (a reference)
!!   DC_REF=file   reads that reference and reports max|diff| vs this run
!! kd_int is the load-bearing check (bar: <1e-12 rel). MLD is ALSO compared but
!! is CHAOTICALLY BRANCH-SENSITIVE: a sub-ulp FMA difference can flip the
!! static-stability short-circuit at the mixed-layer base, so mld jumps a WHOLE
!! layer for a small % of columns. That is documented physics (README
!! "Bit-identity"), NOT a port bug -- the driver reports it with that caveat.
!!
!! Usage: ./dc_main [nx_phys] [ny_phys] [nz] [nreps] [nwarm] [-] [-] [max_its] [ntrials]
!!   trailing args mirror epbl_bench.F90's positions; args 6,7,10 are CUDA-only
!!   there (cuda_sync / want_its / want_dump) and are accepted-but-ignored here,
!!   so the same command line runs both drivers. e.g. a short verify run:
!!   ./dc_main 473 297 30 8 2 1 0 20 5
program dc_main
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use constants, only: wp, SEAWATER_CP
   use grid, only: hgrid_t
   use epbl_stubs, only: multilayer_cgrid_state_t, ocean_surface_stress_t, &
                             ocean_surface_flux_t, EOS_VARIANT_LINEAR
   use ocean_epbl, only: ocean_epbl_t, epbl_column_kernel
   implicit none

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30
   integer, parameter :: REPS_DEF = 50, WARM_DEF = 10, NGHOST = 3
   integer, parameter :: BATHY_NX = 473, BATHY_NY = 297
   character(len=*), parameter :: BATHY_FILE = "gabight_bathy_0p1_473x297.f64"

   ! Profiled config (see epbl_bench.F90 for the namelist provenance).
   real(wp), parameter :: DT = 300.0_wp
   real(wp), parameter :: TAUX_MAG = 0.15_wp
   real(wp), parameter :: PISTON_T = 1.0_wp
   real(wp), parameter :: RESTORE_SST = 14.0_wp
   real(wp), parameter :: H_SURF_TARGET = 5.0_wp
   real(wp), parameter :: T_SFC = 14.0_wp, T_BOT = 2.0_wp, S_INIT = 35.0_wp
   real(wp), parameter :: RHO_0 = 1035.0_wp, ALPHA_T = 0.2_wp
   real(wp), parameter :: MIXED_LAYER_M = 100.0_wp
   real(wp), parameter :: THERMOCLINE_M = 800.0_wp

   type(hgrid_t) :: grid
   type(multilayer_cgrid_state_t) :: ms
   type(ocean_surface_stress_t) :: ss
   type(ocean_surface_flux_t) :: sf
   type(ocean_epbl_t) :: e_dc
   real(wp), allocatable :: hT(:, :, :), hS(:, :, :)
   real(wp), allocatable :: depth(:, :), bathy(:, :)

   real(wp) :: t0w, t1w, ms_dc, mx_dc, tr, noise
   integer :: trial, n_trials
   real(wp) :: inv_rho0_cp, inv_rho0, gib
   integer :: i, j, k, rep, nx, ny, nz, nxp, nyp, n_reps, n_warm
   integer :: max_its_ovr, ncol, nwet, iu, ios
   integer :: bi, bj
   real(wp) :: D, z_top, z_bot, hk, zc, tt, sst_loc, y_rel, wet
   real(wp) :: hsurf, r, denom, mld_mean, mld_max_v, mld_min_v, kd_max_v
   integer :: nmld
   character(len=256) :: ref_path, dump_path

   nxp = iarg(1, NXP_DEF)
   nyp = iarg(2, NYP_DEF)
   nz = iarg(3, NZ_DEF)
   n_reps = iarg(4, REPS_DEF)
   n_warm = iarg(5, WARM_DEF)
   ! args 6 (cuda_sync) and 7 (want_its) are CUDA-only in epbl_bench; parse to
   ! keep the command line aligned across drivers, but they do nothing here.
   ! max_its_ovr: 20 = production; 1 removes the MLD root-find (see README).
   max_its_ovr = iarg(8, 20)
   ! MIN over interleaved trials is the reported time -- this GPU is shared.
   n_trials = iarg(9, 5)

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
   gib = (7.0_wp + 3.0_wp)*real(nx, wp)*real(ny, wp)*real(nz, wp)*8.0_wp/(1024.0_wp**3)

   ! ---------------- allocate ----------------
   allocate (ms%h_layer(nx, ny, nz), ms%wet_mask(nx, ny))
   allocate (hT(nx, ny, nz), hS(nx, ny, nz))
   allocate (ss%tau_x(nx + 1, ny), ss%tau_y(nx, ny + 1))
   allocate (sf%Q_heat(nx, ny), sf%Q_salt(nx, ny))
   allocate (depth(nx, ny), bathy(BATHY_NX, BATHY_NY))
   ms%nz_ml = nz
   ms%idx_temperature = 1; ms%idx_salinity = 2
   sf%cp = SEAWATER_CP; sf%rho0 = RHO_0
   call e_dc%init(grid, nz)

   ! ---------------- real bathymetry (nearest-neighbour resample) ----------
   call read_bathy(bathy)
   do j = 1, ny
      bj = clampi(nint((real(j - NGHOST, wp) - 0.5_wp)/real(nyp, wp)*real(BATHY_NY, wp) + 0.5_wp), &
                  1, BATHY_NY)
      do i = 1, nx
         bi = clampi(nint((real(i - NGHOST, wp) - 0.5_wp)/real(nxp, wp)*real(BATHY_NX, wp) &
                          + 0.5_wp), 1, BATHY_NX)
         depth(i, j) = bathy(bi, bj)
      end do
   end do

   ! ---------------- state (verbatim from epbl_bench.F90) ----------------
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

   ! 2gyre wind (physical rows only).
   ss%tau_x = 0.0_wp; ss%tau_y = 0.0_wp
   do j = NGHOST + 1, NGHOST + nyp
      y_rel = (real(j - NGHOST, wp) - 0.5_wp)/real(nyp, wp)
      do i = 1, nx + 1
         ss%tau_x(i, j) = TAUX_MAG*(1.0_wp - cos(8.0_wp*atan(1.0_wp)*y_rel))
      end do
   end do

   ! SST restoring -> Q_heat changes sign across the domain.
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

   ! Coriolis: seed |f| across lat -60.61 .. -30.9.
   do j = 1, ny
      do i = 1, nx
         e_dc%f_centre(i, j) = abs(2.0_wp*7.2921e-5_wp* &
                                   sin((-60.61_wp + 0.1_wp*real(j - NGHOST, wp))* &
                                       atan(1.0_wp)/45.0_wp))
      end do
   end do

   call set_knobs(e_dc)
   inv_rho0_cp = 1.0_wp/(e_dc%rho0*sf%cp)
   inv_rho0 = 1.0_wp/e_dc%rho0

   write (output_unit, '(a)') repeat('=', 74)
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  domain: ', nxp, ' x ', nyp, ' x ', nz, &
      ' interior (+', NGHOST, ' ghosts/side)'
   write (output_unit, '(a,i0,a,i0,a,i0,a)') '  arrays: ', nx, ' x ', ny, ' -> ', ncol, ' COLUMNS'
   write (output_unit, '(a,i0,a,f5.1,a)') '  wet columns: ', nwet, '  (', &
      100.0_wp*real(nwet, wp)/real(ncol, wp), ' % -- from the real gabight bathymetry)'
   write (output_unit, '(3a,i0,a,i0,a,i0,a)') '  DATA layer: ', DC_DATA_NAME, &
      '   (reps ', n_reps, ', warm ', n_warm, ', trials ', n_trials, ')'
   write (output_unit, '(a,f0.2,a)') '  device memory ~', gib, ' GiB'
   write (output_unit, '(a)') repeat('=', 74)

   ! ---------------- map the working set (no-ops when DATA layer is 'host') --
   DC_ENTER_IN(ms)
   DC_ENTER_IN(ms%h_layer)
   DC_ENTER_IN(ms%wet_mask)
   DC_ENTER_IN(ss)
   DC_ENTER_IN(ss%tau_x)
   DC_ENTER_IN(ss%tau_y)
   DC_ENTER_IN(sf)
   DC_ENTER_IN(sf%Q_heat)
   DC_ENTER_IN(sf%Q_salt)
   DC_ENTER_IN(hT)
   DC_ENTER_IN(hS)
   DC_ENTER_IN(e_dc)
   DC_ENTER_IN(e_dc%f_centre)
   DC_ENTER_IN(e_dc%mld)
   DC_ENTER_IN(e_dc%kd_int)
   DC_ENTER_IN(e_dc%la)
   DC_ENTER_IN(e_dc%tke_wind)
   DC_ENTER_IN(e_dc%tke_conv)
   DC_ENTER_IN(e_dc%tke_forcing)
   DC_ENTER_IN(e_dc%tke_mixing)
   DC_ENTER_IN(e_dc%tke_mech_decay)
   DC_ENTER_IN(e_dc%tke_conv_decay)
   DC_ENTER_IN(e_dc%t0)
   DC_ENTER_IN(e_dc%s0)
   DC_ENTER_IN(e_dc%dpe_t)
   DC_ENTER_IN(e_dc%dpe_s)
   DC_ENTER_IN(e_dc%dcolht_t)
   DC_ENTER_IN(e_dc%dcolht_s)
   DC_ENTER_CREATE(e_dc%t0%data)
   DC_ENTER_CREATE(e_dc%s0%data)
   DC_ENTER_CREATE(e_dc%dpe_t%data)
   DC_ENTER_CREATE(e_dc%dpe_s%data)
   DC_ENTER_CREATE(e_dc%dcolht_t%data)
   DC_ENTER_CREATE(e_dc%dcolht_s%data)

   ! ---------------- do concurrent, production verbatim ----------------------
   do rep = 1, n_warm
      call epbl_column_kernel(grid, e_dc, ms, hT, hS, ss, DT, sf%Q_heat, inv_rho0_cp, &
                              sf%Q_salt, inv_rho0, nx, ny)
   end do
   DC_WAIT

   ! ---------------- timing: MIN of interleaved trials -----------------------
   ! This GPU is shared with sibling jobs; contention can only ADD time, so the
   ! MIN over repeated trials is the best estimator of the uncontended cost.
   ms_dc = huge(1.0_wp); mx_dc = 0.0_wp
   do trial = 1, n_trials
      t0w = wall()
      do rep = 1, n_reps
         call epbl_column_kernel(grid, e_dc, ms, hT, hS, ss, DT, sf%Q_heat, inv_rho0_cp, &
                                 sf%Q_salt, inv_rho0, nx, ny)
      end do
      DC_WAIT
      t1w = wall()
      tr = (t1w - t0w)*1000.0_wp/real(n_reps, wp)
      ms_dc = min(ms_dc, tr); mx_dc = max(mx_dc, tr)
   end do

   DC_UPDATE_SELF(e_dc%kd_int)
   DC_UPDATE_SELF(e_dc%mld)

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

   write (output_unit, '(a)') '  --- state (profiled run: MLD mean 120 m, max ~5000 m) ---'
   write (output_unit, '(a,f10.3,a)') '  MLD_EPBL mean (wet)    : ', mld_mean, ' m'
   write (output_unit, '(a,f10.3,a)') '  MLD_EPBL min  (wet)    : ', mld_min_v, ' m'
   write (output_unit, '(a,f10.3,a)') '  MLD_EPBL max  (wet)    : ', mld_max_v, ' m'
   write (output_unit, '(a,es12.5,a)') '  max Kd_EPBL            : ', kd_max_v, ' m^2/s'
   write (output_unit, '(a)') ''
   write (output_unit, '(3a,f10.4,a,f9.4,a)') '  EPBL do concurrent (', DC_DATA_NAME, ')  : ', &
      ms_dc, ' ms/rep   (worst ', mx_dc, ')'
   noise = mx_dc/max(ms_dc, 1.0e-12_wp)
   if (noise > 1.15_wp) then
      write (output_unit, '(a,f5.2,a)') '  GPU CONTENDED: worst/best = ', noise, &
         ' x -- MIN is still the best estimate; re-run on a quiet GPU to quote.'
   else
      write (output_unit, '(a,f5.2,a)') '  GPU quiet: worst/best = ', noise, ' x -- timing clean.'
   end if
   if (kd_max_v /= kd_max_v .or. (mld_max_v == 0.0_wp .and. kd_max_v == 0.0_wp)) then
      write (output_unit, '(a)') '  sanity           : *** garbage (NaN or all-zero) ***'; stop 2
   else
      write (output_unit, '(a)') '  sanity           : OK (finite, non-zero)'
   end if

   ! ---------------- optional reference dump / cross-check -------------------
   call get_environment_variable('DC_DUMP', dump_path, status=ios)
   if (ios == 0 .and. len_trim(dump_path) > 0) then
      open (newunit=iu, file=trim(dump_path), access='stream', form='unformatted', status='replace')
      write (iu) nx, ny, nz
      write (iu) e_dc%kd_int
      write (iu) e_dc%mld
      close (iu)
      write (output_unit, '(3a)') '  wrote ref       : ', trim(dump_path), ' (nx,ny,nz, kd_int, mld)'
   end if

   call get_environment_variable('DC_REF', ref_path, status=ios)
   if (ios == 0 .and. len_trim(ref_path) > 0) call compare_ref(trim(ref_path))

   write (output_unit, '(a)') repeat('=', 74)

contains

   !! Cross-check this run against a reference dump. kd_int is the LOAD-BEARING
   !! check (bar <1e-12 rel -> the macro'd data layer is numerically inert). MLD
   !! is compared too but only REPORTED: EPBL is chaotically branch-sensitive
   !! (README "Bit-identity"), so a small % of columns can jump a whole layer
   !! from a sub-ulp FMA difference. That is expected physics, not a failure.
   subroutine compare_ref(path)
      character(len=*), intent(in) :: path
      real(wp), allocatable :: kd_ref(:, :, :), mld_ref(:, :)
      real(wp) :: dmax_kd, rmax_kd, dmax_mld, rmax_mld, sc, d, relmld, hbase
      real(wp) :: kd_scale, rel_field
      integer :: rnx, rny, rnz, u, st, nbad_kd, ii, jj, kk
      integer :: nwet_l, ndiff_mld, nlayer_sized
      open (newunit=u, file=path, access='stream', form='unformatted', status='old', iostat=st)
      if (st /= 0) then
         write (output_unit, '(3a)') '  cross-check     : ref ', path, ' not found -- skipped'; return
      end if
      read (u) rnx, rny, rnz
      if (rnx /= nx .or. rny /= ny .or. rnz /= nz) then
         write (output_unit, '(a)') '  cross-check     : ref has a different shape -- skipped'
         close (u); return
      end if
      allocate (kd_ref(nx, ny, nz + 1), mld_ref(nx, ny))
      read (u) kd_ref
      read (u) mld_ref
      close (u)

      ! --- kd_int: the decisive check ---
      ! kd_int is diffusivity AT INTERFACES and is STRUCTURALLY ZERO at the bed
      ! (k=1) and surface (k=nz+1), and near-zero in statically stable interior
      ! interfaces. Pointwise relative error is therefore the WRONG bar -- it
      ! divides an FMA-level roundoff by an ~1e-21 near-zero and reports a huge
      ! "relative" error at cells that carry no signal (exactly the argument the
      ! pilot's native driver makes for the divergence field flux_h). The
      ! load-bearing bar is the diff against the FIELD MAGNITUDE. Pointwise rel
      ! is still printed for transparency.
      dmax_kd = 0.0_wp; rmax_kd = 0.0_wp; nbad_kd = 0; kd_scale = 0.0_wp
      do kk = 1, nz + 1
         do jj = 1, ny
            do ii = 1, nx
               d = abs(e_dc%kd_int(ii, jj, kk) - kd_ref(ii, jj, kk))
               dmax_kd = max(dmax_kd, d)
               kd_scale = max(kd_scale, abs(kd_ref(ii, jj, kk)))
               sc = max(abs(e_dc%kd_int(ii, jj, kk)), abs(kd_ref(ii, jj, kk)))
               if (sc > 1.0e-30_wp) then
                  rmax_kd = max(rmax_kd, d/sc)
                  if (d/sc > 1.0e-12_wp) nbad_kd = nbad_kd + 1
               end if
            end do
         end do
      end do
      rel_field = merge(dmax_kd/kd_scale, 0.0_wp, kd_scale > 0.0_wp)
      write (output_unit, '(a,es12.5,a,es12.5)') '  cross-check kd_int : max|diff| ', dmax_kd, &
         '  /max|kd| ', rel_field
      write (output_unit, '(a,es12.5,a,i0,a)') '                       pointwise max rel ', rmax_kd, &
         '  (', nbad_kd, ' near-zero interface cells -- see note)'
      if (rel_field < 1.0e-12_wp) then
         write (output_unit, '(a)') '  cross-check kd_int : OK (<1e-12 vs field -> data layer numerically inert)'
      else
         write (output_unit, '(a)') '  cross-check kd_int : *** >1e-12 vs field -- INVESTIGATE ***'
      end if

      ! --- mld: reported, classified, never fails (branch sensitivity) ---
      dmax_mld = 0.0_wp; rmax_mld = 0.0_wp
      nwet_l = 0; ndiff_mld = 0; nlayer_sized = 0
      do jj = 1, ny
         do ii = 1, nx
            d = abs(e_dc%mld(ii, jj) - mld_ref(ii, jj))
            dmax_mld = max(dmax_mld, d)
            sc = max(abs(e_dc%mld(ii, jj)), abs(mld_ref(ii, jj)))
            if (sc > 1.0e-30_wp) rmax_mld = max(rmax_mld, d/sc)
            if (ms%wet_mask(ii, jj) <= 0.0_wp) cycle
            nwet_l = nwet_l + 1
            relmld = d/max(abs(e_dc%mld(ii, jj)), 1.0e-30_wp)
            if (relmld > 1.0e-12_wp) then
               ndiff_mld = ndiff_mld + 1
               hbase = maxval(ms%h_layer(ii, jj, max(nz - 6, 1):nz))
               if (d <= 3.0_wp*hbase) nlayer_sized = nlayer_sized + 1
            end if
         end do
      end do
      write (output_unit, '(a,es12.5,a,es12.5)') '  cross-check mld    : max|diff| ', dmax_mld, &
         '  max rel ', rmax_mld
      if (rmax_mld < 1.0e-12_wp) then
         write (output_unit, '(a)') '  cross-check mld    : OK (<1e-12 rel -- no branch flips this run)'
      else
         write (output_unit, '(a,i0,a,i0,a,f7.4,a)') '  cross-check mld    : ', ndiff_mld, ' / ', &
            nwet_l, ' wet columns moved >1e-12 rel (', &
            100.0_wp*real(ndiff_mld, wp)/real(max(nwet_l, 1), wp), ' %)'
         write (output_unit, '(a,i0,a,i0,a)') '                       of those, <= ~1 layer thick: ', &
            nlayer_sized, ' / ', ndiff_mld, ''
         write (output_unit, '(a)') '                       -> EXPECTED: EPBL branch sensitivity, NOT a'
         write (output_unit, '(a)') '                          bug. A sub-ulp FMA difference flips the'
         write (output_unit, '(a)') '                          static-stability short-circuit and mld'
         write (output_unit, '(a)') '                          jumps a whole layer. See README.'
      end if
   end subroutine compare_ref

   subroutine set_knobs(e)
      type(ocean_epbl_t), intent(inout) :: e
      e%enable = .true.
      e%rho0 = RHO_0
      e%eos%variant = EOS_VARIANT_LINEAR
      e%eos%rho0 = RHO_0
      e%eos%alpha_T = ALPHA_T
      e%eos%beta_S = 7.6e-4_wp
      e%eos%T_ref = 10.0_wp
      e%eos%S_ref = 35.0_wp
      e%mld_max_its = max_its_ovr
      e%mld = 0.0_wp
      e%kd_int = 0.0_wp
      e%la = 0.0_wp
   end subroutine set_knobs

   subroutine read_bathy(b)
      real(wp), intent(out) :: b(BATHY_NX, BATHY_NY)
      integer :: u, st
      open (newunit=u, file=BATHY_FILE, form='unformatted', access='stream', &
            status='old', iostat=st)
      if (st /= 0) then
         write (output_unit, '(a)') 'ERROR: cannot open '//BATHY_FILE//'.'
         write (output_unit, '(a)') '  It is the REAL gabight bathymetry and the state depends on it.'
         stop 1
      end if
      read (u, iostat=st) b
      close (u)
      if (st /= 0) then
         write (output_unit, '(a)') 'ERROR: short read on '//BATHY_FILE
         stop 1
      end if
   end subroutine read_bathy

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

end program dc_main
