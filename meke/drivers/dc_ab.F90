#include "directives.h"
!! DC A/B driver for the production MEKE step: faithful meke_step_ext (16 `do
!! concurrent` loops) vs the fused meke_step_ext_fused (6 loops), held
!! BIT-IDENTICAL. Same state setup as dc_main.F90; adds a timed A/B comparison on
!! the primary output field `meke` (max rel diff < 1e-12 is the bar).
!!
!! COMPUTE is bare `do concurrent`; the device data layer is chosen ENTIRELY by
!! directives.h at compile time (DC_DATA_ACC -> OpenACC GPU; DC_DATA_OMP -> OpenMP
!! target; neither -> host CPU). No CUDA, no nvcc in this binary.
!!
!! Usage: ./dc_ab [nx] [ny] [nz] [nreps] [nwarm]
program dc_main
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use constants, only: wp
   use meke_state, only: ocean_metrics_t, multilayer_cgrid_state_t, ocean_gm_t, ocean_meke_t
   use meke, only: meke_step_ext, meke_step_ext_fused
   implicit none

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30
   integer, parameter :: REPS_DEF = 200, WARM_DEF = 20, NGH = 3
   real(wp), parameter :: DXM = 8000.0_wp, DYM = 8000.0_wp, DT_THERMO = 1800.0_wp

   type(ocean_metrics_t) :: met
   type(multilayer_cgrid_state_t) :: ms
   type(ocean_gm_t) :: gm
   type(ocean_meke_t) :: m
   real(wp), allocatable, target :: ke_diss_ext(:, :)
   real(wp), allocatable :: meke_prod(:, :)
   real(wp) :: t0, t1, ms_dc, ms_fu, dmax, rmax, sc, e_min, e_max
   integer :: i, j, k, rep, nx, ny, nz, nxp, nyp, n_reps, n_warm, nbad

   nxp = iarg(1, NXP_DEF); nyp = iarg(2, NYP_DEF); nz = iarg(3, NZ_DEF)
   n_reps = iarg(4, REPS_DEF); n_warm = iarg(5, WARM_DEF)
   nx = nxp + 2*NGH; ny = nyp + 2*NGH
   if (nx < 6 .or. ny < 6 .or. nz < 1) then
      write (output_unit, '(a)') 'ERROR: need nx,ny >= 1 and nz >= 1'; stop 1
   end if

   allocate (met%areaT(nx, ny), met%iareaT(nx, ny), met%idxT(nx, ny), met%idyT(nx, ny))
   allocate (met%dy_cu(nx + 1, ny), met%dx_cv(nx, ny + 1))
   allocate (met%idxCu(nx + 1, ny), met%idyCv(nx, ny + 1))
   allocate (ms%h_layer(nx, ny, nz), ms%rho_layer(nx, ny, nz))
   allocate (gm%gm_src(nx, ny), ke_diss_ext(nx, ny))
   ms%nz_ml = nz; gm%is_init = .true.

   ! uniform-Cartesian metrics (the gabight spherical configs are close to this)
   met%areaT = DXM*DYM; met%iareaT = 1.0_wp/(DXM*DYM)
   met%idxT = 1.0_wp/DXM; met%idyT = 1.0_wp/DYM
   met%dy_cu = DYM; met%dx_cv = DXM
   met%idxCu = 1.0_wp/DXM; met%idyCv = 1.0_wp/DYM
   do k = 1, nz
      do j = 1, ny
         do i = 1, nx
            ms%h_layer(i, j, k) = 4000.0_wp/real(nz, wp)
            ms%rho_layer(i, j, k) = 1025.0_wp + 5.0_wp*real(nz - k, wp)/real(nz, wp)
         end do
      end do
   end do
   do j = 1, ny
      do i = 1, nx
         gm%gm_src(i, j) = 1.0e-2_wp*exp(-((real(i - nx/2, wp)/real(max(nx/6, 1), wp))**2 &
                                           + (real(j - ny/2, wp)/real(max(ny/6, 1), wp))**2))
         ke_diss_ext(i, j) = -1.0e-3_wp
      end do
   end do

   call alloc_meke(m)
   call init_meke(m)
   allocate (meke_prod(nx, ny))

   write (output_unit, '(a)') repeat('=', 74)
   write (output_unit, '(a,i0,a,i0,a,i0,a)') '  config: ', nxp, ' x ', nyp, &
      ' points (+', NGH, ' ghosts), gabight_sph_meke_v100.nml'
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0)') '  arrays: ', nx, ' x ', ny, &
      ' = ', nx*ny, ' cells;  nz = ', nz
   write (output_unit, '(3a,i0,a,i0,a)') '  DATA layer: ', DC_DATA_NAME, &
      '   (reps ', n_reps, ', warm ', n_warm, ')'
   write (output_unit, '(a)') repeat('=', 74)

   ! ---- map the working set (no-ops when the DATA layer is 'host') ----------
   DC_ENTER_IN(met)
   DC_ENTER_IN(met%areaT)
   DC_ENTER_IN(met%iareaT)
   DC_ENTER_IN(met%idxT)
   DC_ENTER_IN(met%idyT)
   DC_ENTER_IN(met%dy_cu)
   DC_ENTER_IN(met%dx_cv)
   DC_ENTER_IN(met%idxCu)
   DC_ENTER_IN(met%idyCv)
   DC_ENTER_IN(ms)
   DC_ENTER_IN(ms%h_layer)
   DC_ENTER_IN(ms%rho_layer)
   DC_ENTER_IN(gm)
   DC_ENTER_IN(gm%gm_src)
   DC_ENTER_IN(ke_diss_ext)
   DC_ENTER_IN(m)
   DC_ENTER_IN(m%meke)
   DC_ENTER_IN(m%kh_diff)
   DC_ENTER_IN(m%u_bbl2)
   DC_ENTER_IN(m%f_centre)
   DC_ENTER_CREATE(m%le)
   DC_ENTER_CREATE(m%ku)
   DC_ENTER_CREATE(m%i_mass)
   DC_ENTER_CREATE(m%depth_tot)
   DC_ENTER_CREATE(m%bottom_fac2)
   DC_ENTER_CREATE(m%barotr_fac2)
   DC_ENTER_CREATE(m%src)
   DC_ENTER_CREATE(m%uflux)
   DC_ENTER_CREATE(m%vflux)
   DC_ENTER_CREATE(m%del2)
   DC_ENTER_CREATE(m%mass_ws)
   DC_ENTER_CREATE(m%ke_diss_ws)
   DC_ENTER_CREATE(m%rd_ws)
   DC_ENTER_CREATE(m%sn_u_ws)
   DC_ENTER_CREATE(m%sn_v_ws)

   ! ---- A: faithful do concurrent (16 loops) -------------------------------
   do rep = 1, n_warm
      call meke_step_ext(nx, ny, nz, m, met, gm, ms, DT_THERMO, ke_diss_ext)
   end do
   DC_WAIT
   t0 = wall()
   do rep = 1, n_reps
      call meke_step_ext(nx, ny, nz, m, met, gm, ms, DT_THERMO, ke_diss_ext)
   end do
   DC_WAIT
   t1 = wall()
   ms_dc = (t1 - t0)*1000.0_wp/real(n_reps, wp)
   DC_UPDATE_SELF(m%meke)
   meke_prod = m%meke

   ! ---- B: fused do concurrent (6 loops) -----------------------------------
   ! Re-seed the prognostic meke so B starts from the SAME state A did (the
   ! step mutates meke in place, so A left it advanced n_warm+n_reps steps).
   call reseed_meke(m)
   DC_UPDATE_DEVICE(m%meke)
   do rep = 1, n_warm
      call meke_step_ext_fused(nx, ny, nz, m, met, gm, ms, DT_THERMO, ke_diss_ext)
   end do
   DC_WAIT
   t0 = wall()
   do rep = 1, n_reps
      call meke_step_ext_fused(nx, ny, nz, m, met, gm, ms, DT_THERMO, ke_diss_ext)
   end do
   DC_WAIT
   t1 = wall()
   ms_fu = (t1 - t0)*1000.0_wp/real(n_reps, wp)
   DC_UPDATE_SELF(m%meke)

   dmax = 0.0_wp; rmax = 0.0_wp; nbad = 0
   do j = 1, ny
      do i = 1, nx
         dmax = max(dmax, abs(m%meke(i, j) - meke_prod(i, j)))
         sc = max(abs(m%meke(i, j)), abs(meke_prod(i, j)))
         if (sc > 1.0e-30_wp) then
            rmax = max(rmax, abs(m%meke(i, j) - meke_prod(i, j))/sc)
            if (abs(m%meke(i, j) - meke_prod(i, j))/sc > 1.0e-12_wp) nbad = nbad + 1
         end if
      end do
   end do
   e_min = minval(m%meke); e_max = maxval(m%meke)

   write (output_unit, '(a,i0,a,i0,a,i0,a)') '  grid ', nx, ' x ', ny, ' (', nx*ny, ' cells)'
   write (output_unit, '(3a)') '  data layer       : ', DC_DATA_NAME, ''
   write (output_unit, '(a,es14.6,a,es14.6)') '  output (fused)   : min ', e_min, '  max ', e_max
   write (output_unit, '(a,es12.5,a,es12.5)') '  correctness      : max|diff| ', dmax, '  max rel ', rmax
   if (e_min /= e_min .or. (e_min == 0.0_wp .and. e_max == 0.0_wp)) then
      write (output_unit, '(a)') '  verdict          : *** garbage (NaN or all-zero) ***'; stop 2
   else if (rmax < 1.0e-12_wp) then
      write (output_unit, '(a)') '  verdict          : OK (fused == faithful, <1e-12 rel)'
   else
      write (output_unit, '(a,i0,a)') '  verdict          : *** ', nbad, ' cells >1e-12 rel -- fused bug ***'
   end if
   write (output_unit, '(a,f10.4,a)') '  faithful DC      : ', ms_dc, ' ms/rep'
   write (output_unit, '(a,f10.4,a,f7.3,a)') '  fused DC         : ', ms_fu, ' ms/rep   -> ', ms_dc/ms_fu, 'x'

   write (output_unit, '(a)') repeat('=', 74)

contains

   subroutine alloc_meke(s)
      type(ocean_meke_t), intent(inout) :: s
      allocate (s%meke(nx, ny), s%kh_diff(nx, ny), s%le(nx, ny), s%ku(nx, ny))
      allocate (s%i_mass(nx, ny), s%depth_tot(nx, ny))
      allocate (s%bottom_fac2(nx, ny), s%barotr_fac2(nx, ny), s%src(nx, ny))
      allocate (s%uflux(nx + 1, ny), s%vflux(nx, ny + 1), s%del2(nx, ny))
      allocate (s%mass_ws(nx, ny), s%u_bbl2(nx, ny), s%ke_diss_ws(nx, ny))
      allocate (s%rd_ws(nx, ny), s%f_centre(nx, ny))
      allocate (s%sn_u_ws(nx + 1, ny), s%sn_v_ws(nx, ny + 1))
      allocate (s%baro_hu(nx + 1, ny), s%baro_hv(nx, ny + 1))
      s%nx_total = nx; s%ny_total = ny; s%nz_ml = nz; s%is_init = .true.
   end subroutine alloc_meke

   !! Knobs from gabight_sph_meke_v100.nml; everything else keeps the default.
   subroutine init_meke(s)
      type(ocean_meke_t), intent(inout) :: s
      s%enable = .true.
      s%gmcoeff = 0.9_wp
      s%khcoeff = 1.0_wp
      s%cdrag = 2.5e-3_wp
      s%kh = 100.0_wp
      call reseed_meke(s)
      s%kh_diff = 0.0_wp; s%le = 0.0_wp; s%ku = 0.0_wp
      s%i_mass = 0.0_wp; s%depth_tot = 0.0_wp
      s%bottom_fac2 = 0.0_wp; s%barotr_fac2 = 0.0_wp; s%src = 0.0_wp
      s%uflux = 0.0_wp; s%vflux = 0.0_wp; s%del2 = 0.0_wp
      s%mass_ws = 0.0_wp
      s%u_bbl2 = 0.0_wp
      s%ke_diss_ws = 0.0_wp; s%rd_ws = 0.0_wp
      s%sn_u_ws = 0.0_wp; s%sn_v_ws = 0.0_wp
      s%baro_hu = 0.0_wp; s%baro_hv = 0.0_wp
   end subroutine init_meke

   !! (Re)seed the prognostic energy + Coriolis so A and B start identically.
   subroutine reseed_meke(s)
      type(ocean_meke_t), intent(inout) :: s
      integer :: ii, jj
      do jj = 1, ny
         do ii = 1, nx
            s%meke(ii, jj) = 1.0e-3_wp + 1.0e-2_wp* &
               exp(-((real(ii - nx/3, wp)/real(max(nx/8, 1), wp))**2 &
                     + (real(jj - ny/3, wp)/real(max(ny/8, 1), wp))**2))
            s%f_centre(ii, jj) = abs(-1.0e-4_wp + 2.0e-11_wp*real(jj - ny/2, wp)*DYM)
         end do
      end do
   end subroutine reseed_meke

   integer function iarg(k, dflt)
      integer, intent(in) :: k, dflt
      character(len=32) :: buf
      integer :: ln, st
      iarg = dflt
      if (command_argument_count() < k) return
      call get_command_argument(k, buf, ln, st)
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
