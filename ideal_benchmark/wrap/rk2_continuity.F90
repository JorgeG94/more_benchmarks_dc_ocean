#include "directives.h"
!! RK2 wrapper for the continuity_layered kernel.
!! Entry compute: continuity_compute_fluxes_fused (fused opt-DC, 3 loops).
!! Init copied verbatim from continuity_layered/drivers/dc_ab.F90.
module rk2_continuity_mod
   use, intrinsic :: iso_c_binding, only: c_int, c_double
   use constants, only: wp
   use grid, only: hgrid_t
   use ocean_metrics, only: ocean_metrics_t
   use multilayer_cgrid_state, only: multilayer_cgrid_state_t
   use continuity_layered, only: continuity_t, continuity_compute_fluxes_fused
   implicit none
   private
   public :: rk2_continuity_init, rk2_continuity_stage, rk2_continuity_stage_cuda, rk2_continuity_probe

   integer, parameter :: NGHOST = 3
   real(wp), parameter :: DX = 10.0_wp, DY = 10.0_wp

   interface
      ! extern "C" in continuity_layered/opt_kernel.cu (all scalars by value).
      subroutine continuity_opt_launch(h, u, v, wet_T, dy_cu, dx_cv, iareaT, &
                                       mfx, mfy, fh, nx, ny, nz, nghost, nx_phys, ny_phys, &
                                       do_pos, h_min_pos) bind(C, name="continuity_opt_launch")
         import :: wp, c_int, c_double
         real(wp) :: h(*), u(*), v(*), wet_T(*), dy_cu(*), dx_cv(*), iareaT(*)
         real(wp) :: mfx(*), mfy(*), fh(*)
         integer(c_int), value :: nx, ny, nz, nghost, nx_phys, ny_phys, do_pos
         real(c_double), value :: h_min_pos
      end subroutine continuity_opt_launch
   end interface

   type(hgrid_t), save :: grid
   type(ocean_metrics_t), save :: metrics
   type(multilayer_cgrid_state_t), save :: ms
   type(continuity_t), save :: cont

contains

   subroutine rk2_continuity_init(nxp, nyp, nz) bind(C, name="rk2_continuity_init")
      integer(c_int), value :: nxp, nyp, nz
      integer :: i, j, k, nx, ny
      nx = nxp + 2*NGHOST; ny = nyp + 2*NGHOST
      grid%nx_total = nx; grid%ny_total = ny
      grid%nx_phys = nxp; grid%ny_phys = nyp
      grid%nghost = NGHOST; grid%dx = DX; grid%dy = DY

      allocate (ms%h_layer(nx, ny, nz))
      allocate (ms%u_face_x_layer(nx + 1, ny, nz), ms%v_face_y_layer(nx, ny + 1, nz))
      allocate (ms%mass_flux_x_layer(nx + 1, ny, nz), ms%mass_flux_y_layer(nx, ny + 1, nz))
      allocate (ms%flux_h_layer(nx, ny, nz))
      allocate (metrics%dy_cu(nx + 1, ny), metrics%dx_cv(nx, ny + 1))
      allocate (metrics%iareaT(nx, ny), metrics%wet_T(nx, ny))
      allocate (cont%h_face_left_x%data(nx + 1, ny, nz), cont%h_face_right_x%data(nx + 1, ny, nz))
      allocate (cont%h_face_left_y%data(nx, ny + 1, nz), cont%h_face_right_y%data(nx, ny + 1, nz))
      ms%nz_ml = nz

      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               ms%h_layer(i, j, k) = 20.0_wp + real(k, wp)*0.5_wp &
                                     + 5.0_wp*exp(-((real(i - nx/2, wp)/real(max(nx/8, 1), wp))**2 &
                                                    + (real(j - ny/2, wp)/real(max(ny/8, 1), wp))**2)) &
                                     + 0.4_wp*sin(0.01_wp*real(i, wp))*cos(0.013_wp*real(j, wp))
            end do
         end do
      end do
      do j = 1, ny
         do i = 1, nx
            metrics%wet_T(i, j) = 1.0_wp; metrics%iareaT(i, j) = 1.0_wp/(DX*DY)
         end do
      end do
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx + 1
               ms%u_face_x_layer(i, j, k) = 0.5_wp*sin(0.02_wp*real(i, wp))*cos(0.03_wp*real(k, wp))
            end do
         end do
      end do
      do k = 1, nz
         do j = 1, ny + 1
            do i = 1, nx
               ms%v_face_y_layer(i, j, k) = 0.3_wp*cos(0.017_wp*real(j, wp))
            end do
         end do
      end do
      do j = 1, ny
         do i = 1, nx + 1
            metrics%dy_cu(i, j) = DY
         end do
      end do
      do j = 1, ny + 1
         do i = 1, nx
            metrics%dx_cv(i, j) = DX
         end do
      end do
      ms%mass_flux_x_layer = 0.0_wp; ms%mass_flux_y_layer = 0.0_wp; ms%flux_h_layer = 0.0_wp
      cont%h_face_left_x%data = 0.0_wp; cont%h_face_right_x%data = 0.0_wp
      cont%h_face_left_y%data = 0.0_wp; cont%h_face_right_y%data = 0.0_wp
      cont%h_min = 1.0e-6_wp; cont%use_ppm_limit_pos = .false.

      DC_ENTER_IN(ms)
      DC_ENTER_IN(ms%h_layer)
      DC_ENTER_IN(ms%u_face_x_layer)
      DC_ENTER_IN(ms%v_face_y_layer)
      DC_ENTER_CREATE(ms%mass_flux_x_layer)
      DC_ENTER_CREATE(ms%mass_flux_y_layer)
      DC_ENTER_CREATE(ms%flux_h_layer)
      DC_ENTER_IN(metrics)
      DC_ENTER_IN(metrics%dy_cu)
      DC_ENTER_IN(metrics%dx_cv)
      DC_ENTER_IN(metrics%iareaT)
      DC_ENTER_IN(metrics%wet_T)
      DC_ENTER_IN(cont)
      DC_ENTER_IN(cont%h_face_left_x)
      DC_ENTER_IN(cont%h_face_right_x)
      DC_ENTER_IN(cont%h_face_left_y)
      DC_ENTER_IN(cont%h_face_right_y)
      DC_ENTER_CREATE(cont%h_face_left_x%data)
      DC_ENTER_CREATE(cont%h_face_right_x%data)
      DC_ENTER_CREATE(cont%h_face_left_y%data)
      DC_ENTER_CREATE(cont%h_face_right_y%data)
      DC_WAIT
   end subroutine rk2_continuity_init

   subroutine rk2_continuity_stage() bind(C, name="rk2_continuity_stage")
      call continuity_compute_fluxes_fused(grid, metrics, cont, ms)
   end subroutine rk2_continuity_stage

   subroutine rk2_continuity_stage_cuda() bind(C, name="rk2_continuity_stage_cuda")
      integer :: do_pos
      do_pos = 0
      if (cont%use_ppm_limit_pos) do_pos = 1
      !$acc host_data use_device(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, &
      !$acc                      metrics%wet_T, metrics%dy_cu, metrics%dx_cv, metrics%iareaT, &
      !$acc                      ms%mass_flux_x_layer, ms%mass_flux_y_layer, ms%flux_h_layer)
      call continuity_opt_launch(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, &
                                 metrics%wet_T, metrics%dy_cu, metrics%dx_cv, metrics%iareaT, &
                                 ms%mass_flux_x_layer, ms%mass_flux_y_layer, ms%flux_h_layer, &
                                 grid%nx_total, grid%ny_total, ms%nz_ml, grid%nghost, &
                                 grid%nx_phys, grid%ny_phys, do_pos, cont%h_min)
      !$acc end host_data
   end subroutine rk2_continuity_stage_cuda

   subroutine rk2_continuity_probe(vmin, vmax) bind(C, name="rk2_continuity_probe")
      real(c_double), intent(out) :: vmin, vmax
      DC_UPDATE_SELF(ms%flux_h_layer)
      vmin = minval(ms%flux_h_layer); vmax = maxval(ms%flux_h_layer)
   end subroutine rk2_continuity_probe

end module rk2_continuity_mod
