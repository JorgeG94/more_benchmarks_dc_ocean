#include "directives.h"
!! RK2 wrapper for the btstep barotropic substep kernel.
!! Entry compute: btstep_nonlinear_closed_fused (fused opt-DC, 5 loops/substep).
!! Init copied verbatim from btstep/drivers/dc_ab.F90 (alloc_state + init_state).
module rk2_btstep_mod
   use, intrinsic :: iso_c_binding, only: c_int, c_double
   use constants, only: wp
   use bt_state, only: hgrid_t, ocean_metrics_t, coriolis_t, bt_work_t
   use btstep, only: btstep_nonlinear_closed_fused, btstep_nonlinear_closed
   implicit none
   private
   public :: rk2_btstep_init, rk2_btstep_stage, rk2_btstep_stage_unopt, rk2_btstep_probe
#ifndef RK2_NO_CUDA
   public :: rk2_btstep_stage_cuda, rk2_btstep_stage_cuda_unopt
#endif

   integer, parameter :: NGH = 3, N_INNER = 24
   real(wp), parameter :: DXM = 10000.0_wp, DYM = 10000.0_wp, DT_INNER = 12.0_wp

#ifndef RK2_NO_CUDA
   interface
      ! opt: extern "C" btstep_opt_launch_flat (opt_kernel.cu). Last int selects
      ! the opt CUDA path (0=fused). faithful: btstep_cuda_flat (shim_btstep.cu),
      ! SAME flat arg list, last int = btstep_cuda_launch mode (0=faithful 11-loop).
      subroutine btstep_opt_launch_flat(eta, eta_new, H_ref, ubt, vbt, &
                                        ubt_prev, vbt_prev, rem_u, rem_v, &
                                        zeta, ke, ubt_sum, vbt_sum, eta_sum, &
                                        uhbt_sum, vhbt_sum, &
                                        dy_cu, dx_cv, iareaT, areaCu, areaCv, &
                                        dxCu, dyCv, idxCu, idyCv, iareaBu, &
                                        f_corner, force_u, force_v, &
                                        nx, ny, nghost, nx_phys, ny_phys, &
                                        G, bebt, dt, n_steps, opt_mode) &
         bind(C, name="btstep_opt_launch_flat")
         import :: wp
         real(wp) :: eta(*), eta_new(*), H_ref(*), ubt(*), vbt(*)
         real(wp) :: ubt_prev(*), vbt_prev(*), rem_u(*), rem_v(*)
         real(wp) :: zeta(*), ke(*), ubt_sum(*), vbt_sum(*), eta_sum(*)
         real(wp) :: uhbt_sum(*), vhbt_sum(*)
         real(wp) :: dy_cu(*), dx_cv(*), iareaT(*), areaCu(*), areaCv(*)
         real(wp) :: dxCu(*), dyCv(*), idxCu(*), idyCv(*), iareaBu(*)
         real(wp) :: f_corner(*), force_u(*), force_v(*)
         integer, value :: nx, ny, nghost, nx_phys, ny_phys, n_steps, opt_mode
         real(wp), value :: G, bebt, dt
      end subroutine btstep_opt_launch_flat
      subroutine btstep_cuda_flat(eta, eta_new, H_ref, ubt, vbt, &
                                        ubt_prev, vbt_prev, rem_u, rem_v, &
                                        zeta, ke, ubt_sum, vbt_sum, eta_sum, &
                                        uhbt_sum, vhbt_sum, &
                                        dy_cu, dx_cv, iareaT, areaCu, areaCv, &
                                        dxCu, dyCv, idxCu, idyCv, iareaBu, &
                                        f_corner, force_u, force_v, &
                                        nx, ny, nghost, nx_phys, ny_phys, &
                                        G, bebt, dt, n_steps, mode) &
         bind(C, name="btstep_cuda_flat")
         import :: wp
         real(wp) :: eta(*), eta_new(*), H_ref(*), ubt(*), vbt(*)
         real(wp) :: ubt_prev(*), vbt_prev(*), rem_u(*), rem_v(*)
         real(wp) :: zeta(*), ke(*), ubt_sum(*), vbt_sum(*), eta_sum(*)
         real(wp) :: uhbt_sum(*), vhbt_sum(*)
         real(wp) :: dy_cu(*), dx_cv(*), iareaT(*), areaCu(*), areaCv(*)
         real(wp) :: dxCu(*), dyCv(*), idxCu(*), idyCv(*), iareaBu(*)
         real(wp) :: f_corner(*), force_u(*), force_v(*)
         integer, value :: nx, ny, nghost, nx_phys, ny_phys, n_steps, mode
         real(wp), value :: G, bebt, dt
      end subroutine btstep_cuda_flat
   end interface
#endif

   type(hgrid_t), save :: grid
   type(ocean_metrics_t), save :: met
   type(coriolis_t), save :: cor
   type(bt_work_t), save :: w
   real(wp), allocatable, save :: force_u(:, :), force_v(:, :)

contains

   subroutine rk2_btstep_init(nxp, nyp, nz) bind(C, name="rk2_btstep_init")
      integer(c_int), value :: nxp, nyp, nz
      integer :: i, j, nx, ny
      nx = nxp + 2*NGH; ny = nyp + 2*NGH
      grid%nx_total = nx; grid%ny_total = ny
      grid%nx_phys = nxp; grid%ny_phys = nyp; grid%nghost = NGH

      allocate (w%bt_eta(nx, ny), w%bt_eta_new(nx, ny), w%bt_H_ref(nx, ny))
      allocate (w%bt_ubt(nx+1, ny), w%bt_vbt(nx, ny+1))
      allocate (w%bt_ubt_prev(nx+1, ny), w%bt_vbt_prev(nx, ny+1))
      allocate (w%bt_rem_u(nx+1, ny), w%bt_rem_v(nx, ny+1))
      allocate (w%bt_zeta_corner(nx+1, ny+1), w%bt_ke_centre(nx, ny))
      allocate (w%ubt_sum(nx+1, ny), w%vbt_sum(nx, ny+1), w%eta_sum(nx, ny))
      allocate (w%uhbt_sum(nx+1, ny), w%vhbt_sum(nx, ny+1))
      allocate (met%dy_cu(nx+1, ny), met%dx_cv(nx, ny+1), met%iareaT(nx, ny))
      allocate (met%areaCu(nx+1, ny), met%areaCv(nx, ny+1))
      allocate (met%dxCu(nx+1, ny), met%dyCv(nx, ny+1))
      allocate (met%idxCu(nx+1, ny), met%idyCv(nx, ny+1), met%iareaBu(nx+1, ny+1))
      allocate (cor%f_corner(nx+1, ny+1))
      allocate (force_u(nx+1, ny), force_v(nx, ny+1))

      met%dy_cu = DYM; met%dx_cv = DXM
      met%iareaT = 1.0_wp/(DXM*DYM)
      met%areaCu = DXM*DYM; met%areaCv = DXM*DYM
      met%dxCu = DXM; met%dyCv = DYM
      met%idxCu = 1.0_wp/DXM; met%idyCv = 1.0_wp/DYM
      met%iareaBu = 1.0_wp/(DXM*DYM)
      do j = 1, ny + 1
         do i = 1, nx + 1
            cor%f_corner(i, j) = -1.0e-4_wp + 2.0e-11_wp*real(j - ny/2, wp)*DYM
         end do
      end do
      do j = 1, ny
         do i = 1, nx + 1
            force_u(i, j) = 1.0e-6_wp*sin(3.14159_wp*real(j, wp)/real(ny, wp))
         end do
      end do
      force_v = 0.0_wp

      call init_state(nx, ny)

      DC_ENTER_IN(met)
      DC_ENTER_IN(cor)
      DC_ENTER_IN(w)
      DC_ENTER_IN(w%bt_eta)
      DC_ENTER_IN(w%bt_eta_new)
      DC_ENTER_IN(w%bt_H_ref)
      DC_ENTER_IN(w%bt_ubt)
      DC_ENTER_IN(w%bt_vbt)
      DC_ENTER_IN(w%bt_ubt_prev)
      DC_ENTER_IN(w%bt_vbt_prev)
      DC_ENTER_IN(w%bt_rem_u)
      DC_ENTER_IN(w%bt_rem_v)
      DC_ENTER_IN(w%bt_zeta_corner)
      DC_ENTER_IN(w%bt_ke_centre)
      DC_ENTER_IN(w%ubt_sum)
      DC_ENTER_IN(w%vbt_sum)
      DC_ENTER_IN(w%eta_sum)
      DC_ENTER_IN(w%uhbt_sum)
      DC_ENTER_IN(w%vhbt_sum)
      DC_ENTER_IN(met%dy_cu)
      DC_ENTER_IN(met%dx_cv)
      DC_ENTER_IN(met%iareaT)
      DC_ENTER_IN(met%areaCu)
      DC_ENTER_IN(met%areaCv)
      DC_ENTER_IN(met%dxCu)
      DC_ENTER_IN(met%dyCv)
      DC_ENTER_IN(met%idxCu)
      DC_ENTER_IN(met%idyCv)
      DC_ENTER_IN(met%iareaBu)
      DC_ENTER_IN(cor%f_corner)
      DC_ENTER_IN(force_u)
      DC_ENTER_IN(force_v)
      DC_WAIT
   end subroutine rk2_btstep_init

   subroutine init_state(nx, ny)
      integer, intent(in) :: nx, ny
      integer :: ii, jj
      w%g_bt = 9.81_wp
      w%bebt = 0.2_wp
      do jj = 1, ny
         do ii = 1, nx
            w%bt_H_ref(ii, jj) = 4000.0_wp
            w%bt_eta(ii, jj) = 0.5_wp*exp(-((real(ii - nx/2, wp)/real(max(nx/8, 1), wp))**2 &
                                            + (real(jj - ny/2, wp)/real(max(ny/8, 1), wp))**2))
         end do
      end do
      w%bt_eta_new = 0.0_wp
      w%bt_ubt = 0.0_wp; w%bt_vbt = 0.0_wp
      w%bt_ubt_prev = 0.0_wp; w%bt_vbt_prev = 0.0_wp
      w%bt_rem_u = 1.0_wp; w%bt_rem_v = 1.0_wp
      w%bt_zeta_corner = 0.0_wp; w%bt_ke_centre = 0.0_wp
      w%ubt_sum = 0.0_wp; w%vbt_sum = 0.0_wp; w%eta_sum = 0.0_wp
      w%uhbt_sum = 0.0_wp; w%vhbt_sum = 0.0_wp
   end subroutine init_state

   subroutine rk2_btstep_stage() bind(C, name="rk2_btstep_stage")
      call btstep_nonlinear_closed_fused(grid, met, w, cor, force_u, force_v, N_INNER, DT_INNER)
   end subroutine rk2_btstep_stage

   subroutine rk2_btstep_stage_unopt() bind(C, name="rk2_btstep_stage_unopt")
      call btstep_nonlinear_closed(grid, met, w, cor, force_u, force_v, N_INNER, DT_INNER)
   end subroutine rk2_btstep_stage_unopt

#ifndef RK2_NO_CUDA
   subroutine rk2_btstep_stage_cuda() bind(C, name="rk2_btstep_stage_cuda")
      integer :: nx, ny
      nx = grid%nx_total; ny = grid%ny_total
      !$acc host_data use_device(w%bt_eta, w%bt_eta_new, w%bt_H_ref, w%bt_ubt, w%bt_vbt, &
      !$acc                      w%bt_ubt_prev, w%bt_vbt_prev, w%bt_rem_u, w%bt_rem_v, &
      !$acc                      w%bt_zeta_corner, w%bt_ke_centre, w%ubt_sum, w%vbt_sum, &
      !$acc                      w%eta_sum, w%uhbt_sum, w%vhbt_sum, &
      !$acc                      met%dy_cu, met%dx_cv, met%iareaT, met%areaCu, met%areaCv, &
      !$acc                      met%dxCu, met%dyCv, met%idxCu, met%idyCv, met%iareaBu, &
      !$acc                      cor%f_corner, force_u, force_v)
      call btstep_opt_launch_flat(w%bt_eta, w%bt_eta_new, w%bt_H_ref, w%bt_ubt, w%bt_vbt, &
                                  w%bt_ubt_prev, w%bt_vbt_prev, w%bt_rem_u, w%bt_rem_v, &
                                  w%bt_zeta_corner, w%bt_ke_centre, w%ubt_sum, w%vbt_sum, &
                                  w%eta_sum, w%uhbt_sum, w%vhbt_sum, &
                                  met%dy_cu, met%dx_cv, met%iareaT, met%areaCu, met%areaCv, &
                                  met%dxCu, met%dyCv, met%idxCu, met%idyCv, met%iareaBu, &
                                  cor%f_corner, force_u, force_v, &
                                  nx, ny, grid%nghost, grid%nx_phys, grid%ny_phys, &
                                  w%g_bt, w%bebt, DT_INNER, N_INNER, 0)
      !$acc end host_data
   end subroutine rk2_btstep_stage_cuda

   subroutine rk2_btstep_stage_cuda_unopt() bind(C, name="rk2_btstep_stage_cuda_unopt")
      ! Faithful CUDA: btstep_cuda_flat -> btstep_cuda_launch(&BtArgs, N_INNER, 0).
      integer :: nx, ny
      nx = grid%nx_total; ny = grid%ny_total
      !$acc host_data use_device(w%bt_eta, w%bt_eta_new, w%bt_H_ref, w%bt_ubt, w%bt_vbt, &
      !$acc                      w%bt_ubt_prev, w%bt_vbt_prev, w%bt_rem_u, w%bt_rem_v, &
      !$acc                      w%bt_zeta_corner, w%bt_ke_centre, w%ubt_sum, w%vbt_sum, &
      !$acc                      w%eta_sum, w%uhbt_sum, w%vhbt_sum, &
      !$acc                      met%dy_cu, met%dx_cv, met%iareaT, met%areaCu, met%areaCv, &
      !$acc                      met%dxCu, met%dyCv, met%idxCu, met%idyCv, met%iareaBu, &
      !$acc                      cor%f_corner, force_u, force_v)
      call btstep_cuda_flat(w%bt_eta, w%bt_eta_new, w%bt_H_ref, w%bt_ubt, w%bt_vbt, &
                                  w%bt_ubt_prev, w%bt_vbt_prev, w%bt_rem_u, w%bt_rem_v, &
                                  w%bt_zeta_corner, w%bt_ke_centre, w%ubt_sum, w%vbt_sum, &
                                  w%eta_sum, w%uhbt_sum, w%vhbt_sum, &
                                  met%dy_cu, met%dx_cv, met%iareaT, met%areaCu, met%areaCv, &
                                  met%dxCu, met%dyCv, met%idxCu, met%idyCv, met%iareaBu, &
                                  cor%f_corner, force_u, force_v, &
                                  nx, ny, grid%nghost, grid%nx_phys, grid%ny_phys, &
                                  w%g_bt, w%bebt, DT_INNER, N_INNER, 0)
      !$acc end host_data
   end subroutine rk2_btstep_stage_cuda_unopt
#endif

   subroutine rk2_btstep_probe(vmin, vmax) bind(C, name="rk2_btstep_probe")
      real(c_double), intent(out) :: vmin, vmax
      DC_UPDATE_SELF(w%bt_eta)
      vmin = minval(w%bt_eta); vmax = maxval(w%bt_eta)
   end subroutine rk2_btstep_probe

end module rk2_btstep_mod
