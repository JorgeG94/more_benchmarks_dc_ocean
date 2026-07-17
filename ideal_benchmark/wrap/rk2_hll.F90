#include "directives.h"
!! RK2 wrapper for the HLL flux kernel (2-D shallow water).
!! Entry compute: compute_flux_hll (faithful production kernel).
!! Init copied verbatim from hll_fluxes/drivers/dc_main.F90. NGHOST=2 (2-D).
module rk2_hll_mod
   use, intrinsic :: iso_c_binding, only: c_int, c_double
   use constants, only: wp
   use grid, only: hgrid_t
   use kernel_flux, only: compute_flux_hll
   implicit none
   private
   public :: rk2_hll_init, rk2_hll_stage, rk2_hll_stage_cuda, rk2_hll_probe

   integer, parameter :: NGHOST = 2
   real(wp), parameter :: DX = 10.0_wp, DY = 10.0_wp

   interface
      ! extern "C" flux_hll_opt_launch (hll_fluxes/opt_kernel.cu).
      subroutine flux_hll_opt_launch(h, hu, hv, b, flux_h, flux_hu, flux_hv, &
                                     mass_flux_x, mass_flux_y, nx, ny, nghost, &
                                     dx, dy, sync) bind(C, name="flux_hll_opt_launch")
         import :: wp, c_int, c_double
         real(wp) :: h(*), hu(*), hv(*), b(*)
         real(wp) :: flux_h(*), flux_hu(*), flux_hv(*), mass_flux_x(*), mass_flux_y(*)
         integer(c_int), value :: nx, ny, nghost, sync
         real(c_double), value :: dx, dy
      end subroutine flux_hll_opt_launch
   end interface

   type(hgrid_t), save :: grid
   real(wp), allocatable, save :: h(:, :), hu(:, :), hv(:, :), b(:, :)
   real(wp), allocatable, save :: flux_h(:, :), flux_hu(:, :), flux_hv(:, :)
   real(wp), allocatable, save :: mass_flux_x(:, :), mass_flux_y(:, :)

contains

   subroutine rk2_hll_init(nxp, nyp, nz) bind(C, name="rk2_hll_init")
      integer(c_int), value :: nxp, nyp, nz
      integer :: i, j, nx, ny
      nx = nxp + 2*NGHOST; ny = nyp + 2*NGHOST
      grid%nghost = NGHOST; grid%nx_total = nx; grid%ny_total = ny
      grid%dx = DX; grid%dy = DY

      allocate (h(nx, ny), hu(nx, ny), hv(nx, ny), b(nx, ny))
      allocate (flux_h(nx, ny), flux_hu(nx, ny), flux_hv(nx, ny))
      allocate (mass_flux_x(nx, ny), mass_flux_y(nx, ny))

      do j = 1, ny
         do i = 1, nx
            b(i, j) = 0.0_wp
            if ((real(i - nx/2, wp)**2 + real(j - ny/2, wp)**2) < real(nx/4, wp)**2) then
               h(i, j) = 10.0_wp
            else
               h(i, j) = 2.0_wp
            end if
            hu(i, j) = 0.05_wp*real(i, wp)/real(nx, wp)
            hv(i, j) = -0.03_wp*real(j, wp)/real(ny, wp)
         end do
      end do
      flux_h = 0.0_wp; flux_hu = 0.0_wp; flux_hv = 0.0_wp
      mass_flux_x = 0.0_wp; mass_flux_y = 0.0_wp

      DC_ENTER_IN(h)
      DC_ENTER_IN(hu)
      DC_ENTER_IN(hv)
      DC_ENTER_IN(b)
      DC_ENTER_CREATE(flux_h)
      DC_ENTER_CREATE(flux_hu)
      DC_ENTER_CREATE(flux_hv)
      DC_ENTER_CREATE(mass_flux_x)
      DC_ENTER_CREATE(mass_flux_y)
      DC_WAIT
   end subroutine rk2_hll_init

   subroutine rk2_hll_stage() bind(C, name="rk2_hll_stage")
      call compute_flux_hll(h, hu, hv, b, flux_h, flux_hu, flux_hv, &
                            mass_flux_x, mass_flux_y, grid)
   end subroutine rk2_hll_stage

   subroutine rk2_hll_stage_cuda() bind(C, name="rk2_hll_stage_cuda")
      !$acc host_data use_device(h, hu, hv, b, flux_h, flux_hu, flux_hv, &
      !$acc                      mass_flux_x, mass_flux_y)
      call flux_hll_opt_launch(h, hu, hv, b, flux_h, flux_hu, flux_hv, &
                               mass_flux_x, mass_flux_y, grid%nx_total, grid%ny_total, &
                               grid%nghost, grid%dx, grid%dy, 0)
      !$acc end host_data
   end subroutine rk2_hll_stage_cuda

   subroutine rk2_hll_probe(vmin, vmax) bind(C, name="rk2_hll_probe")
      real(c_double), intent(out) :: vmin, vmax
      DC_UPDATE_SELF(flux_h)
      vmin = minval(flux_h); vmax = maxval(flux_h)
   end subroutine rk2_hll_probe

end module rk2_hll_mod
