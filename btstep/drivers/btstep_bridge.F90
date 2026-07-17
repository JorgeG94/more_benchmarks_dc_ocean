!! Bridge from the shared head-to-head driver to the optimized CUDA launcher.
!!
!! Hands the CUDA kernels the SAME device allocations the `do concurrent`
!! routine (btstep_nonlinear_closed_fused) uses, via `!$acc host_data
!! use_device`. No copies, no second harness: both toolchains read one truth,
!! so opt-DC vs opt-CUDA is a fused-vs-fused comparison on identical device
!! arrays in a single binary.
!!
!! The flat entry point btstep_opt_launch_flat (opt_kernel.cu) takes one device
!! pointer per array plus the scalars/ints by value, then assembles the BtArgs
!! struct C-side -- so this Fortran interface stays a uniform flat bind(C), like
!! redi_cuda.F90's.
!!
!! THIS FILE MUST NOT BE COMPILED WITH `-cuda`. That flag is LINK-ONLY here: it
!! also switches nvfortran into CUDA Fortran mode, which type-checks device
!! attributes -- inside `host_data use_device` the arrays carry the device
!! attribute while the bind(C) interface declares plain host arrays, giving
!! NVFORTRAN-S-0528 device attribute mismatch.
module btstep_bridge
   use constants, only: wp
   use bt_state, only: hgrid_t, ocean_metrics_t, coriolis_t, bt_work_t
   implicit none
   private

   public :: btstep_opt_cuda

   interface
      !! Flat mirror of `struct BtArgs` (btstep_args.h): every array a device
      !! pointer (assumed-size, passed inside host_data), scalars/ints by value.
      !! opt_mode: 0 = fused (5 kern/substep), 1 = fused captured in a cudaGraph.
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
   end interface

contains

   subroutine btstep_opt_cuda(grid, met, w, cor, force_u, force_v, n_steps, dt_inner, opt_mode)
      !! One optimized-CUDA call: n_steps substeps on the device arrays w/met/cor
      !! already own. host_data resolves each to its device pointer; the launcher
      !! synchronises before returning.
      type(hgrid_t), intent(in) :: grid
      type(ocean_metrics_t), intent(in) :: met
      type(bt_work_t), intent(inout) :: w
      type(coriolis_t), intent(in) :: cor
      real(wp), intent(in) :: force_u(:, :), force_v(:, :)
      integer, intent(in) :: n_steps, opt_mode
      real(wp), intent(in) :: dt_inner

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
                                  w%g_bt, w%bebt, dt_inner, n_steps, opt_mode)
      !$acc end host_data
   end subroutine btstep_opt_cuda

end module btstep_bridge
