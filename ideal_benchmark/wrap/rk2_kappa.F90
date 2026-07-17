#include "directives.h"
!! RK2 wrapper for the kappa_shear (JHL08) column kernel.
!! Entry compute: kappa_shear_column_kernel (faithful, no opt variant).
!! Init copied verbatim from kappa_shear/drivers/dc_main.F90 (build_state).
module rk2_kappa_mod
   use, intrinsic :: iso_c_binding, only: c_int, c_double
   use constants, only: wp, NZ_STACK_MAX
   use grid, only: hgrid_t
   use multilayer_cgrid_state, only: multilayer_cgrid_state_t
   use ocean_eos, only: EOS_VARIANT_LINEAR
   use ks, only: ocean_kappa_shear_t, kappa_shear_column_kernel
   implicit none
   private
   public :: rk2_kappa_init, rk2_kappa_stage, rk2_kappa_stage_cuda, rk2_kappa_probe

   integer, parameter :: NGHOST = 3
   real(wp), parameter :: DT_THERM = 300.0_wp

   interface
      ! flat shim (ideal_benchmark/shim_ks.cu) -> ks_opt_launch (opt_kernel.cu).
      subroutine ks_opt_flat(h, u, v, hT, hS, wet, fc, kd, tke, nx, ny, nz, dt, sync) &
         bind(C, name="ks_opt_flat")
         import :: wp, c_int, c_double
         real(wp) :: h(*), u(*), v(*), hT(*), hS(*), wet(*), fc(*), kd(*), tke(*)
         integer(c_int), value :: nx, ny, nz, sync
         real(c_double), value :: dt
      end subroutine ks_opt_flat
   end interface

   type(hgrid_t), save :: grid
   type(multilayer_cgrid_state_t), save :: ms
   type(ocean_kappa_shear_t), save :: ks
   real(wp), allocatable, save :: hT(:, :, :), hS(:, :, :)

contains

   subroutine rk2_kappa_init(nxp, nyp, nz) bind(C, name="rk2_kappa_init")
      integer(c_int), value :: nxp, nyp, nz
      integer :: nx, ny
      integer, parameter :: land_pct = 0
      integer, parameter :: NZ_DEF_MAX = 512
      real(wp) :: depth, mld, us, vs, zt, zb, zm, hh, tt, ss, uu, vv
      real(wp) :: w(NZ_DEF_MAX), wsum, r, xr, yr, lat
      integer :: i, j, m, kg
      nx = nxp + 2*NGHOST; ny = nyp + 2*NGHOST
      grid%nx_total = nx; grid%ny_total = ny
      grid%nx_phys = nxp; grid%ny_phys = nyp
      grid%nghost = NGHOST; grid%dx = 0.1_wp; grid%dy = 0.1_wp

      allocate (ms%h_layer(nx, ny, nz))
      allocate (ms%u_face_x_layer(nx + 1, ny, nz), ms%v_face_y_layer(nx, ny + 1, nz))
      allocate (ms%wet_mask(nx, ny))
      allocate (hT(nx, ny, nz), hS(nx, ny, nz))
      allocate (ks%f_centre(nx, ny))
      allocate (ks%kd_int(nx, ny, nz + 1), ks%tke_int(nx, ny, nz + 1))
      ms%nz_ml = nz

      r = 1.18_wp
      wsum = 0.0_wp
      do m = 1, nz
         w(m) = r**(m - 1); wsum = wsum + w(m)
      end do
      do m = 1, nz
         w(m) = w(m)/wsum
      end do
      do j = 1, ny
         yr = real(j - NGHOST, wp)/real(max(nyp, 1), wp)
         lat = -60.61_wp + 0.1_wp*real(j - NGHOST, wp)
         do i = 1, nx
            xr = real(i - NGHOST, wp)/real(max(nxp, 1), wp)
            ks%f_centre(i, j) = abs(2.0_wp*7.2921e-5_wp*sin(lat*3.141592653589793_wp/180.0_wp))
            depth = 200.0_wp + 4300.0_wp* &
                    (0.5_wp*(1.0_wp + tanh(3.0_wp*(yr - 0.25_wp))))* &
                    (0.85_wp + 0.15_wp*sin(6.0_wp*xr))
            if (land_pct > 0) then
               if (xr < 0.01_wp*real(land_pct, wp) .and. yr > 0.55_wp) then
                  ms%wet_mask(i, j) = 0.0_wp
               else
                  ms%wet_mask(i, j) = 1.0_wp
               end if
            else
               ms%wet_mask(i, j) = 1.0_wp
            end if
            mld = 40.0_wp + 60.0_wp*(0.5_wp*(1.0_wp + sin(5.0_wp*xr)*cos(4.0_wp*yr)))
            us = 0.25_wp + 0.60_wp*(0.5_wp*(1.0_wp + sin(7.0_wp*xr + 2.0_wp*yr)*cos(3.0_wp*yr)))
            vs = 0.10_wp*sin(4.0_wp*xr)*cos(5.0_wp*yr)
            zt = 0.0_wp
            do m = 1, nz
               hh = max(depth*w(m), 1.0e-3_wp)
               zb = zt + hh
               zm = 0.5_wp*(zt + zb)
               if (zm <= mld) then
                  tt = 14.0_wp
               else
                  tt = 2.0_wp + 12.0_wp*exp(-(zm - mld)/600.0_wp)
               end if
               ss = 35.0_wp - 0.5_wp*exp(-zm/300.0_wp)
               uu = us*0.5_wp*(1.0_wp - tanh((zm - mld)/25.0_wp))
               vv = vs*0.5_wp*(1.0_wp - tanh((zm - mld)/25.0_wp))
               kg = nz + 1 - m
               ms%h_layer(i, j, kg) = hh
               hT(i, j, kg) = tt*hh
               hS(i, j, kg) = ss*hh
               ms%u_face_x_layer(i, j, kg) = uu
               ms%v_face_y_layer(i, j, kg) = vv
               zt = zb
            end do
         end do
      end do
      do kg = 1, nz
         do j = 1, ny
            ms%u_face_x_layer(nx + 1, j, kg) = ms%u_face_x_layer(nx, j, kg)
         end do
         do i = 1, nx
            ms%v_face_y_layer(i, ny + 1, kg) = ms%v_face_y_layer(i, ny, kg)
         end do
      end do

      ks%enable = .true.
      ks%eos%variant = EOS_VARIANT_LINEAR
      ks%eos%rho0 = 1035.0_wp
      ks%eos%alpha_T = 0.2_wp
      ks%eos%beta_S = 7.6e-4_wp
      ks%rho0 = 1035.0_wp
      ks%kd_int = 0.0_wp; ks%tke_int = 0.0_wp

      DC_ENTER_IN(ms)
      DC_ENTER_IN(ms%h_layer)
      DC_ENTER_IN(ms%u_face_x_layer)
      DC_ENTER_IN(ms%v_face_y_layer)
      DC_ENTER_IN(ms%wet_mask)
      DC_ENTER_IN(ks)
      DC_ENTER_IN(ks%f_centre)
      DC_ENTER_CREATE(ks%kd_int)
      DC_ENTER_CREATE(ks%tke_int)
      DC_ENTER_IN(hT)
      DC_ENTER_IN(hS)
      DC_WAIT
   end subroutine rk2_kappa_init

   subroutine rk2_kappa_stage() bind(C, name="rk2_kappa_stage")
      call kappa_shear_column_kernel(grid, ks, ms, hT, hS, DT_THERM)
   end subroutine rk2_kappa_stage

   subroutine rk2_kappa_stage_cuda() bind(C, name="rk2_kappa_stage_cuda")
      !$acc host_data use_device(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, &
      !$acc                      hT, hS, ms%wet_mask, ks%f_centre, ks%kd_int, ks%tke_int)
      call ks_opt_flat(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, hT, hS, &
                       ms%wet_mask, ks%f_centre, ks%kd_int, ks%tke_int, &
                       grid%nx_total, grid%ny_total, ms%nz_ml, DT_THERM, 0)
      !$acc end host_data
   end subroutine rk2_kappa_stage_cuda

   subroutine rk2_kappa_probe(vmin, vmax) bind(C, name="rk2_kappa_probe")
      real(c_double), intent(out) :: vmin, vmax
      DC_UPDATE_SELF(ks%kd_int)
      vmin = minval(ks%kd_int); vmax = maxval(ks%kd_int)
   end subroutine rk2_kappa_probe

end module rk2_kappa_mod
