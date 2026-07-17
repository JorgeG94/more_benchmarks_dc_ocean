#include "directives.h"
!! RK2 wrapper for the EPBL column kernel.
!! Entry compute: epbl_column_kernel (faithful production do-concurrent).
!! Init copied verbatim from epbl/drivers/dc_main.F90.
!! Reads the REAL gabight bathymetry file "gabight_bathy_0p1_473x297.f64" from
!! the current working directory (the Makefile stages it next to the binary).
module rk2_epbl_mod
   use, intrinsic :: iso_c_binding, only: c_int, c_double
   use, intrinsic :: iso_fortran_env, only: output_unit
   use constants, only: wp, SEAWATER_CP
   use grid, only: hgrid_t
   use epbl_stubs, only: multilayer_cgrid_state_t, ocean_surface_stress_t, &
                         ocean_surface_flux_t, EOS_VARIANT_LINEAR
   use ocean_epbl, only: ocean_epbl_t, epbl_column_kernel
   implicit none
   private
   ! epbl has NO opt DC variant: opt DC == unopt DC == epbl_column_kernel.
   public :: rk2_epbl_init, rk2_epbl_stage, rk2_epbl_stage_unopt, rk2_epbl_probe
#ifndef RK2_NO_CUDA
   public :: rk2_epbl_stage_cuda, rk2_epbl_stage_cuda_unopt
#endif

#ifndef RK2_NO_CUDA
   interface
      ! flat shims (ideal_benchmark/shim_epbl.cu). Identical signatures; opt ->
      ! epbl_opt_launch (tuned), unopt -> epbl_cuda_launch (faithful, variant 0).
      subroutine epbl_opt_flat(h_layer, wet_mask, hT, hS, tau_x, tau_y, Q_heat, Q_salt, &
                               f_centre, mld, kd_int, la, t0, s0, dpe_t, dpe_s, dcolht_t, dcolht_s, &
                               tke_wind, tke_conv, tke_forcing, tke_mixing, tke_mech_decay, tke_conv_decay, &
                               inv_rho0_cp, inv_rho0, dt, nx, ny, nz, mld_max_its, sync) &
         bind(C, name="epbl_opt_flat")
         import :: wp, c_int, c_double
         real(wp) :: h_layer(*), wet_mask(*), hT(*), hS(*), tau_x(*), tau_y(*)
         real(wp) :: Q_heat(*), Q_salt(*), f_centre(*), mld(*), kd_int(*), la(*)
         real(wp) :: t0(*), s0(*), dpe_t(*), dpe_s(*), dcolht_t(*), dcolht_s(*)
         real(wp) :: tke_wind(*), tke_conv(*), tke_forcing(*), tke_mixing(*)
         real(wp) :: tke_mech_decay(*), tke_conv_decay(*)
         real(c_double), value :: inv_rho0_cp, inv_rho0, dt
         integer(c_int), value :: nx, ny, nz, mld_max_its, sync
      end subroutine epbl_opt_flat
      subroutine epbl_cuda_flat(h_layer, wet_mask, hT, hS, tau_x, tau_y, Q_heat, Q_salt, &
                               f_centre, mld, kd_int, la, t0, s0, dpe_t, dpe_s, dcolht_t, dcolht_s, &
                               tke_wind, tke_conv, tke_forcing, tke_mixing, tke_mech_decay, tke_conv_decay, &
                               inv_rho0_cp, inv_rho0, dt, nx, ny, nz, mld_max_its, sync) &
         bind(C, name="epbl_cuda_flat")
         import :: wp, c_int, c_double
         real(wp) :: h_layer(*), wet_mask(*), hT(*), hS(*), tau_x(*), tau_y(*)
         real(wp) :: Q_heat(*), Q_salt(*), f_centre(*), mld(*), kd_int(*), la(*)
         real(wp) :: t0(*), s0(*), dpe_t(*), dpe_s(*), dcolht_t(*), dcolht_s(*)
         real(wp) :: tke_wind(*), tke_conv(*), tke_forcing(*), tke_mixing(*)
         real(wp) :: tke_mech_decay(*), tke_conv_decay(*)
         real(c_double), value :: inv_rho0_cp, inv_rho0, dt
         integer(c_int), value :: nx, ny, nz, mld_max_its, sync
      end subroutine epbl_cuda_flat
   end interface
#endif

   integer, parameter :: NGHOST = 3
   integer, parameter :: BATHY_NX = 473, BATHY_NY = 297
   character(len=*), parameter :: BATHY_FILE = "gabight_bathy_0p1_473x297.f64"
   real(wp), parameter :: DT = 300.0_wp, TAUX_MAG = 0.15_wp, PISTON_T = 1.0_wp
   real(wp), parameter :: RESTORE_SST = 14.0_wp, H_SURF_TARGET = 5.0_wp
   real(wp), parameter :: T_SFC = 14.0_wp, T_BOT = 2.0_wp, S_INIT = 35.0_wp
   real(wp), parameter :: RHO_0 = 1035.0_wp, ALPHA_T = 0.2_wp
   real(wp), parameter :: MIXED_LAYER_M = 100.0_wp, THERMOCLINE_M = 800.0_wp
   integer, parameter :: MAX_ITS = 20

   type(hgrid_t), save :: grid
   type(multilayer_cgrid_state_t), save :: ms
   type(ocean_surface_stress_t), save :: ss
   type(ocean_surface_flux_t), save :: sf
   type(ocean_epbl_t), save :: e_dc
   real(wp), allocatable, save :: hT(:, :, :), hS(:, :, :)
   real(wp), save :: inv_rho0_cp, inv_rho0
   integer, save :: gnx, gny

contains

   subroutine rk2_epbl_init(nxp, nyp, nz) bind(C, name="rk2_epbl_init")
      integer(c_int), value :: nxp, nyp, nz
      real(wp), allocatable :: depth(:, :), bathy(:, :)
      integer :: i, j, k, nx, ny, bi, bj
      real(wp) :: D, z_top, z_bot, hk, zc, tt, sst_loc, y_rel, wet, hsurf, r, denom
      nx = nxp + 2*NGHOST; ny = nyp + 2*NGHOST
      gnx = nx; gny = ny
      grid%nx_total = nx; grid%ny_total = ny
      grid%nx_phys = nxp; grid%ny_phys = nyp
      grid%nghost = NGHOST; grid%dx = 0.1_wp; grid%dy = 0.1_wp

      allocate (ms%h_layer(nx, ny, nz), ms%wet_mask(nx, ny))
      allocate (hT(nx, ny, nz), hS(nx, ny, nz))
      allocate (ss%tau_x(nx + 1, ny), ss%tau_y(nx, ny + 1))
      allocate (sf%Q_heat(nx, ny), sf%Q_salt(nx, ny))
      allocate (depth(nx, ny), bathy(BATHY_NX, BATHY_NY))
      ms%nz_ml = nz
      ms%idx_temperature = 1; ms%idx_salinity = 2
      sf%cp = SEAWATER_CP; sf%rho0 = RHO_0
      call e_dc%init(grid, nz)

      call read_bathy(bathy)
      do j = 1, ny
         bj = clampi(nint((real(j - NGHOST, wp) - 0.5_wp)/real(nyp, wp)*real(BATHY_NY, wp) + 0.5_wp), 1, BATHY_NY)
         do i = 1, nx
            bi = clampi(nint((real(i - NGHOST, wp) - 0.5_wp)/real(nxp, wp)*real(BATHY_NX, wp) + 0.5_wp), 1, BATHY_NX)
            depth(i, j) = bathy(bi, bj)
         end do
      end do

      do j = 1, ny
         do i = 1, nx
            D = depth(i, j)
            wet = 0.0_wp
            if (D > 1.0_wp) then
               wet = 1.0_wp
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
            y_rel = (real(j - NGHOST, wp) - 0.5_wp)/real(nyp, wp)
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
                  hT(i, j, k) = hk*(T_BOT + (tt - T_BOT)*exp(-(zc - MIXED_LAYER_M)/THERMOCLINE_M))
               end if
               hS(i, j, k) = hk*S_INIT
               z_top = z_bot
            end do
         end do
      end do

      ss%tau_x = 0.0_wp; ss%tau_y = 0.0_wp
      do j = NGHOST + 1, NGHOST + nyp
         y_rel = (real(j - NGHOST, wp) - 0.5_wp)/real(nyp, wp)
         do i = 1, nx + 1
            ss%tau_x(i, j) = TAUX_MAG*(1.0_wp - cos(8.0_wp*atan(1.0_wp)*y_rel))
         end do
      end do

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

      do j = 1, ny
         do i = 1, nx
            e_dc%f_centre(i, j) = abs(2.0_wp*7.2921e-5_wp* &
                                      sin((-60.61_wp + 0.1_wp*real(j - NGHOST, wp))*atan(1.0_wp)/45.0_wp))
         end do
      end do

      e_dc%enable = .true.
      e_dc%rho0 = RHO_0
      e_dc%eos%variant = EOS_VARIANT_LINEAR
      e_dc%eos%rho0 = RHO_0
      e_dc%eos%alpha_T = ALPHA_T
      e_dc%eos%beta_S = 7.6e-4_wp
      e_dc%eos%T_ref = 10.0_wp
      e_dc%eos%S_ref = 35.0_wp
      e_dc%mld_max_its = MAX_ITS
      e_dc%mld = 0.0_wp
      e_dc%kd_int = 0.0_wp
      e_dc%la = 0.0_wp

      inv_rho0_cp = 1.0_wp/(e_dc%rho0*sf%cp)
      inv_rho0 = 1.0_wp/e_dc%rho0

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
      DC_WAIT
      deallocate (depth, bathy)
   end subroutine rk2_epbl_init

   subroutine rk2_epbl_stage() bind(C, name="rk2_epbl_stage")
      call epbl_column_kernel(grid, e_dc, ms, hT, hS, ss, DT, sf%Q_heat, inv_rho0_cp, &
                              sf%Q_salt, inv_rho0, gnx, gny)
   end subroutine rk2_epbl_stage

   subroutine rk2_epbl_stage_unopt() bind(C, name="rk2_epbl_stage_unopt")
      call epbl_column_kernel(grid, e_dc, ms, hT, hS, ss, DT, sf%Q_heat, inv_rho0_cp, &
                              sf%Q_salt, inv_rho0, gnx, gny)
   end subroutine rk2_epbl_stage_unopt

#ifndef RK2_NO_CUDA
   subroutine rk2_epbl_stage_cuda() bind(C, name="rk2_epbl_stage_cuda")
      !$acc host_data use_device(ms%h_layer, ms%wet_mask, hT, hS, ss%tau_x, ss%tau_y, &
      !$acc                      sf%Q_heat, sf%Q_salt, e_dc%f_centre, e_dc%mld, e_dc%kd_int, &
      !$acc                      e_dc%la, e_dc%t0%data, e_dc%s0%data, e_dc%dpe_t%data, &
      !$acc                      e_dc%dpe_s%data, e_dc%dcolht_t%data, e_dc%dcolht_s%data, &
      !$acc                      e_dc%tke_wind, e_dc%tke_conv, e_dc%tke_forcing, e_dc%tke_mixing, &
      !$acc                      e_dc%tke_mech_decay, e_dc%tke_conv_decay)
      call epbl_opt_flat(ms%h_layer, ms%wet_mask, hT, hS, ss%tau_x, ss%tau_y, &
                         sf%Q_heat, sf%Q_salt, e_dc%f_centre, e_dc%mld, e_dc%kd_int, e_dc%la, &
                         e_dc%t0%data, e_dc%s0%data, e_dc%dpe_t%data, e_dc%dpe_s%data, &
                         e_dc%dcolht_t%data, e_dc%dcolht_s%data, &
                         e_dc%tke_wind, e_dc%tke_conv, e_dc%tke_forcing, e_dc%tke_mixing, &
                         e_dc%tke_mech_decay, e_dc%tke_conv_decay, &
                         inv_rho0_cp, inv_rho0, DT, gnx, gny, ms%nz_ml, MAX_ITS, 0)
      !$acc end host_data
   end subroutine rk2_epbl_stage_cuda

   subroutine rk2_epbl_stage_cuda_unopt() bind(C, name="rk2_epbl_stage_cuda_unopt")
      !$acc host_data use_device(ms%h_layer, ms%wet_mask, hT, hS, ss%tau_x, ss%tau_y, &
      !$acc                      sf%Q_heat, sf%Q_salt, e_dc%f_centre, e_dc%mld, e_dc%kd_int, &
      !$acc                      e_dc%la, e_dc%t0%data, e_dc%s0%data, e_dc%dpe_t%data, &
      !$acc                      e_dc%dpe_s%data, e_dc%dcolht_t%data, e_dc%dcolht_s%data, &
      !$acc                      e_dc%tke_wind, e_dc%tke_conv, e_dc%tke_forcing, e_dc%tke_mixing, &
      !$acc                      e_dc%tke_mech_decay, e_dc%tke_conv_decay)
      call epbl_cuda_flat(ms%h_layer, ms%wet_mask, hT, hS, ss%tau_x, ss%tau_y, &
                          sf%Q_heat, sf%Q_salt, e_dc%f_centre, e_dc%mld, e_dc%kd_int, e_dc%la, &
                          e_dc%t0%data, e_dc%s0%data, e_dc%dpe_t%data, e_dc%dpe_s%data, &
                          e_dc%dcolht_t%data, e_dc%dcolht_s%data, &
                          e_dc%tke_wind, e_dc%tke_conv, e_dc%tke_forcing, e_dc%tke_mixing, &
                          e_dc%tke_mech_decay, e_dc%tke_conv_decay, &
                          inv_rho0_cp, inv_rho0, DT, gnx, gny, ms%nz_ml, MAX_ITS, 0)
      !$acc end host_data
   end subroutine rk2_epbl_stage_cuda_unopt
#endif

   subroutine rk2_epbl_probe(vmin, vmax) bind(C, name="rk2_epbl_probe")
      real(c_double), intent(out) :: vmin, vmax
      DC_UPDATE_SELF(e_dc%kd_int)
      vmin = minval(e_dc%kd_int); vmax = maxval(e_dc%kd_int)
   end subroutine rk2_epbl_probe

   subroutine read_bathy(b)
      real(wp), intent(out) :: b(BATHY_NX, BATHY_NY)
      integer :: u, st
      open (newunit=u, file=BATHY_FILE, form='unformatted', access='stream', status='old', iostat=st)
      if (st /= 0) then
         write (output_unit, '(a)') 'ERROR: cannot open '//BATHY_FILE//' (epbl needs it in CWD).'
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
         stretch_ratio = 1.0_wp; return
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

end module rk2_epbl_mod
