#include "directives.h"
!! Bridge from the shared cmp driver to the OPTIMIZED CUDA launcher.
!!
!! Hands the CUDA kernels the SAME device allocations the `do concurrent`
!! variant uses, via `!$acc host_data use_device`. No copies: both toolchains
!! read one truth -- this is what makes cmp_main a single-binary head-to-head.
!!
!! ⚠ THIS FILE MUST NOT BE COMPILED WITH `-cuda`. That flag is LINK-ONLY here:
!! it also switches nvfortran into CUDA Fortran mode, which type-checks device
!! attributes -- inside `host_data use_device` the arrays carry the device
!! attribute while the bind(C) interface declares plain host arrays, giving
!! NVFORTRAN-S-0528 device attribute mismatch.
!!
!! The layout handed over MUST match what ale_remap_step_opt uses on the SAME
!! device arrays: target_h = (Sum h_layer - eta + eta)*dsig(k) (k_opt_pre);
!! bt_eta re-derived from -H_ref + Sum target_h (k_opt_post). Verified by the
!! < 1e-12 agreement check in cmp_main.
module ale_bridge
   use constants, only: wp
   use remap_state, only: hgrid_t, ocean_vcoord_t, multilayer_cgrid_state_t
   implicit none
   private

   public :: ale_remap_opt_step

   interface
      !! extern "C" void ale_remap_opt(int nx, int ny, int nz, real_t* ...)
      !! nx/ny/nz are BY VALUE (int); the real_t* arrays are device pointers
      !! supplied inside host_data. real_t is double == wp here.
      subroutine ale_remap_opt(nx, ny, nz, h_layer, h_old, target_h, &
                               hTr_t, hTr_s, u, v, mass_b, heat_b, salt_b, &
                               eta, H_ref, dsig) &
         bind(C, name="ale_remap_opt")
         import :: wp
         integer, value :: nx, ny, nz
         real(wp) :: h_layer(*), h_old(*), target_h(*)
         real(wp) :: hTr_t(*), hTr_s(*), u(*), v(*)
         real(wp) :: mass_b(*), heat_b(*), salt_b(*)
         real(wp) :: eta(*), H_ref(*), dsig(*)
      end subroutine ale_remap_opt
   end interface

contains

   !! Launch the optimized CUDA remap on the SAME device arrays the optimized
   !! do-concurrent path (ale_remap_step_opt) uses. Argument order and array
   !! roles mirror ale_remap_step_opt exactly:
   !!   scratch:  vcoord%remap_h_old (h_old), vcoord%target_h (h_new)
   !!   tracers:  ms%tracers(1)%hTr = T (heat), ms%tracers(2)%hTr = S (salt)
   !!   budgets:  heat_budget_remap (T), salt_budget_remap (S), mass_budget_remap
   subroutine ale_remap_opt_step(grid, vcoord, ms, bt_eta, bt_H_ref)
      type(hgrid_t), intent(in) :: grid
      type(ocean_vcoord_t), intent(inout) :: vcoord
      type(multilayer_cgrid_state_t), intent(inout) :: ms
      real(wp), intent(inout) :: bt_eta(:, :)
      real(wp), intent(in) :: bt_H_ref(:, :)

      integer :: nx, ny, nz

      nx = grid%nx_total; ny = grid%ny_total; nz = ms%nz_ml

      !$acc host_data use_device(ms%h_layer, vcoord%remap_h_old, vcoord%target_h, &
      !$acc                      ms%tracers(1)%hTr, ms%tracers(2)%hTr, &
      !$acc                      ms%u_face_x_layer, ms%v_face_y_layer, &
      !$acc                      ms%mass_budget_remap, ms%heat_budget_remap, ms%salt_budget_remap, &
      !$acc                      bt_eta, bt_H_ref, vcoord%dsig)
      call ale_remap_opt(nx, ny, nz, &
                         ms%h_layer, vcoord%remap_h_old, vcoord%target_h, &
                         ms%tracers(1)%hTr, ms%tracers(2)%hTr, &
                         ms%u_face_x_layer, ms%v_face_y_layer, &
                         ms%mass_budget_remap, ms%heat_budget_remap, ms%salt_budget_remap, &
                         bt_eta, bt_H_ref, vcoord%dsig)
      !$acc end host_data
   end subroutine ale_remap_opt_step

end module ale_bridge
