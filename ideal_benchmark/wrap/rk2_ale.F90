#include "directives.h"
!! RK2 wrapper for the ALE-remap kernel.
!! Entry compute: ale_remap_step_opt (optimized do-concurrent variant).
!! Init copied verbatim from ale_remap/drivers/dc_ab.F90.
!! NOTE: the remap has a FIXED POINT (after one call h_old == target_h), so a
!! repeated call does strictly less work than the first. A timing harness that
!! calls it once per stage therefore measures the fixed-point (cheap) cost after
!! the first step. This matches the task's "call the compute routine once per
!! stage" contract; see README for the caveat.
module rk2_ale_mod
   use, intrinsic :: iso_c_binding, only: c_int, c_double
   use constants, only: wp, REMAP_PPM, VCOORD_ZSTAR
   use remap_state, only: hgrid_t, ocean_vcoord_t, multilayer_cgrid_state_t
   use ale_remap, only: ale_remap_step_opt
   implicit none
   private
   public :: rk2_ale_init, rk2_ale_stage, rk2_ale_stage_cuda, rk2_ale_probe

   integer, parameter :: NGHOST = 3
   integer, parameter :: PERT_PCT = 25

   interface
      ! extern "C" ale_remap_opt (opt_kernel.cu) -- lifted from ale_bridge.F90.
      subroutine ale_remap_opt(nx, ny, nz, h_layer, h_old, target_h, &
                               hTr_t, hTr_s, u, v, mass_b, heat_b, salt_b, &
                               eta, H_ref, dsig) bind(C, name="ale_remap_opt")
         import :: wp
         integer, value :: nx, ny, nz
         real(wp) :: h_layer(*), h_old(*), target_h(*)
         real(wp) :: hTr_t(*), hTr_s(*), u(*), v(*)
         real(wp) :: mass_b(*), heat_b(*), salt_b(*)
         real(wp) :: eta(*), H_ref(*), dsig(*)
      end subroutine ale_remap_opt
   end interface

   type(hgrid_t), save :: grid
   type(ocean_vcoord_t), target, save :: vc
   type(multilayer_cgrid_state_t), target, save :: ms
   real(wp), allocatable, target, save :: bt_eta(:, :), bt_H_ref(:, :)

contains

   subroutine rk2_ale_init(nxp, nyp, nz) bind(C, name="rk2_ale_init")
      integer(c_int), value :: nxp, nyp, nz
      integer :: i, j, k, nx, ny
      real(wp) :: hbed, sig, pert, cT, cS, csum
      nx = nxp + 2*NGHOST; ny = nyp + 2*NGHOST
      grid%nx_total = nx; grid%ny_total = ny
      vc%nx_total = nx; vc%ny_total = ny; vc%nz_ml = nz
      vc%coord_type = VCOORD_ZSTAR; vc%is_init = .true.; vc%remap_method = REMAP_PPM
      vc%regrid_time_scale = 0.0_wp; vc%remap_vel_conserve_ke = .false.
      ms%nz_ml = nz; ms%idx_temperature = 1; ms%idx_salinity = 2

      allocate (vc%dsig(nz), vc%target_h(nx, ny, nz), vc%remap_h_old(nx, ny, nz))
      allocate (vc%remap_total_h(nx, ny), vc%remap_h_ref(nx, ny))
      allocate (ms%h_layer(nx, ny, nz), ms%mass_budget_remap(nx, ny, nz))
      allocate (ms%heat_budget_remap(nx, ny, nz), ms%salt_budget_remap(nx, ny, nz))
      allocate (ms%u_face_x_layer(nx + 1, ny, nz), ms%v_face_y_layer(nx, ny + 1, nz))
      allocate (ms%tracers(2))
      allocate (ms%tracers(1)%hTr(nx, ny, nz), ms%tracers(2)%hTr(nx, ny, nz))
      allocate (bt_eta(nx, ny), bt_H_ref(nx, ny))

      do k = 1, nz
         vc%dsig(k) = 1.0_wp + 3.0_wp*real(k - 1, wp)/real(max(1, nz - 1), wp)
      end do
      vc%dsig = vc%dsig/sum(vc%dsig)

      do j = 1, ny
         do i = 1, nx
            hbed = 15.0_wp + 4485.0_wp*abs(sin(0.013_wp*real(i, wp))*cos(0.017_wp*real(j, wp)))
            if (mod(i + j, 97) == 0) hbed = 3.0_wp
            bt_H_ref(i, j) = hbed
            bt_eta(i, j) = 0.4_wp*sin(0.05_wp*real(i, wp) + 0.03_wp*real(j, wp))
            do k = 1, nz
               sig = vc%dsig(k)
               pert = (0.01_wp*real(PERT_PCT, wp))*sin(3.0_wp*real(k, wp) + 0.02_wp*real(i, wp) + 0.03_wp*real(j, wp))
               ms%h_layer(i, j, k) = (hbed + bt_eta(i, j))*sig*(1.0_wp + pert)
            end do
            csum = sum(ms%h_layer(i, j, 1:nz))
            ms%h_layer(i, j, 1:nz) = ms%h_layer(i, j, 1:nz)*(hbed + bt_eta(i, j))/csum
            do k = 1, nz
               cT = 2.0_wp + 12.0_wp*exp(-real(nz - k, wp)/6.0_wp) + 0.5_wp*sin(0.02_wp*real(i, wp))
               cS = 34.5_wp + 0.8_wp*real(k, wp)/real(nz, wp)
               ms%tracers(1)%hTr(i, j, k) = ms%h_layer(i, j, k)*cT
               ms%tracers(2)%hTr(i, j, k) = ms%h_layer(i, j, k)*cS
            end do
         end do
      end do
      ms%mass_budget_remap = 0.0_wp; ms%heat_budget_remap = 0.0_wp; ms%salt_budget_remap = 0.0_wp
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx + 1
               ms%u_face_x_layer(i, j, k) = 0.3_wp*sin(0.01_wp*real(i, wp))* &
                                            cos(0.02_wp*real(j, wp))*(1.0_wp + 0.1_wp*real(k, wp))
            end do
         end do
         do j = 1, ny + 1
            do i = 1, nx
               ms%v_face_y_layer(i, j, k) = 0.2_wp*cos(0.015_wp*real(i, wp))* &
                                            sin(0.018_wp*real(j, wp))*(1.0_wp - 0.05_wp*real(k, wp))
            end do
         end do
      end do

      DC_ENTER_IN(grid)
      DC_ENTER_IN(vc)
      DC_ENTER_IN(ms)
      DC_ENTER_IN(vc%dsig)
      DC_ENTER_CREATE(vc%target_h)
      DC_ENTER_CREATE(vc%remap_h_old)
      DC_ENTER_CREATE(vc%remap_total_h)
      DC_ENTER_CREATE(vc%remap_h_ref)
      DC_ENTER_IN(ms%h_layer)
      DC_ENTER_IN(ms%mass_budget_remap)
      DC_ENTER_IN(ms%heat_budget_remap)
      DC_ENTER_IN(ms%salt_budget_remap)
      DC_ENTER_IN(ms%u_face_x_layer)
      DC_ENTER_IN(ms%v_face_y_layer)
      DC_ENTER_IN(ms%tracers)
      DC_ENTER_IN(ms%tracers(1)%hTr)
      DC_ENTER_IN(ms%tracers(2)%hTr)
      DC_ENTER_IN(bt_eta)
      DC_ENTER_IN(bt_H_ref)
      DC_WAIT
   end subroutine rk2_ale_init

   subroutine rk2_ale_stage() bind(C, name="rk2_ale_stage")
      call ale_remap_step_opt(grid, vc, ms, bt_eta, bt_H_ref)
   end subroutine rk2_ale_stage

   subroutine rk2_ale_stage_cuda() bind(C, name="rk2_ale_stage_cuda")
      integer :: nx, ny, nz
      nx = grid%nx_total; ny = grid%ny_total; nz = ms%nz_ml
      !$acc host_data use_device(ms%h_layer, vc%remap_h_old, vc%target_h, &
      !$acc                      ms%tracers(1)%hTr, ms%tracers(2)%hTr, &
      !$acc                      ms%u_face_x_layer, ms%v_face_y_layer, &
      !$acc                      ms%mass_budget_remap, ms%heat_budget_remap, ms%salt_budget_remap, &
      !$acc                      bt_eta, bt_H_ref, vc%dsig)
      call ale_remap_opt(nx, ny, nz, &
                         ms%h_layer, vc%remap_h_old, vc%target_h, &
                         ms%tracers(1)%hTr, ms%tracers(2)%hTr, &
                         ms%u_face_x_layer, ms%v_face_y_layer, &
                         ms%mass_budget_remap, ms%heat_budget_remap, ms%salt_budget_remap, &
                         bt_eta, bt_H_ref, vc%dsig)
      !$acc end host_data
   end subroutine rk2_ale_stage_cuda

   subroutine rk2_ale_probe(vmin, vmax) bind(C, name="rk2_ale_probe")
      real(c_double), intent(out) :: vmin, vmax
      DC_UPDATE_SELF(ms%h_layer)
      vmin = minval(ms%h_layer); vmax = maxval(ms%h_layer)
   end subroutine rk2_ale_probe

end module rk2_ale_mod
