!! Bridge from the shared head-to-head driver to the OPTIMIZED CUDA launcher.
!!
!! Hands the CUDA kernels the SAME device allocations the `do concurrent`
!! variant uses, via `!$acc host_data use_device`. No copies, no second harness:
!! both toolchains read one truth in one binary. This is the redi analogue of
!! legacy_testing/redi_benchmark/redi_cuda.F90, with the bind(C) target changed
!! from the faithful `redi_cuda_launch_` to the optimized `redi_opt_launch`
!! (opt_kernel.cu) -- same signature, same launch sequence (Phase A calc-coeffs
!! + Phase B hoisted apply-flux).
!!
!! WARNING THIS FILE MUST NOT BE COMPILED WITH `-cuda`. That flag is LINK-ONLY
!! here: it also switches nvfortran into CUDA Fortran mode, which type-checks
!! device attributes -- inside `host_data use_device` the arrays carry the
!! device attribute while the bind(C) interface declares plain host arrays,
!! giving NVFORTRAN-S-0528 device attribute mismatch.
module redi_bridge
   use constants, only: wp
   use grid, only: hgrid_t
   use ocean_metrics, only: ocean_metrics_t
   use multilayer_cgrid_state, only: multilayer_cgrid_state_t
   use ocean_eos, only: ocean_eos_t
   use ocean_redi, only: ocean_redi_t
   implicit none
   private

   public :: redi_opt_step

   interface
      !! extern "C" void redi_opt_launch(...) in opt_kernel.cu. Note the bind(C)
      !! name is the BARE symbol `redi_opt_launch` (no trailing underscore): the
      !! CUDA side declares it `extern "C"` with exactly that spelling, unlike
      !! the legacy faithful launcher which used the `_`-suffixed Fortran name.
      subroutine redi_opt_launch(nx, ny, nz, ns, dt, nghost, nxp, nyp, &
                                 ww, we, wsz, wn, rho0, T_ref, S_ref, alpha_T, beta_S, &
                                 h_layer, t_htr, s_htr, wet_u, wet_v, &
                                 uPoL, uPoR, uKoL, uKoR, uhEff, &
                                 vPoL, vPoR, vKoL, vKoR, vhEff, &
                                 khtr_u_ext, khtr_v_ext, khtr_u, khtr_v, &
                                 dy_cu, dx_cv, idxCu, idyCv, areaT, tr_snap) &
         bind(C, name="redi_opt_launch")
         import :: wp
         integer :: nx, ny, nz, ns, nghost, nxp, nyp, ww, we, wsz, wn
         real(wp) :: dt, rho0, T_ref, S_ref, alpha_T, beta_S
         real(wp) :: h_layer(*), t_htr(*), s_htr(*), wet_u(*), wet_v(*)
         real(wp) :: uPoL(*), uPoR(*), uhEff(*), vPoL(*), vPoR(*), vhEff(*)
         integer :: uKoL(*), uKoR(*), vKoL(*), vKoR(*)
         real(wp) :: khtr_u_ext(*), khtr_v_ext(*), khtr_u(*), khtr_v(*)
         real(wp) :: dy_cu(*), dx_cv(*), idxCu(*), idyCv(*), areaT(*), tr_snap(*)
      end subroutine redi_opt_launch
   end interface

contains

   subroutine redi_opt_step(grid, metrics, eos, this, ms, dt, khtr_u_ext, khtr_v_ext)
      !! ONE optimized launch = Phase A (redi_calc_coeffs) + face copy + per
      !! tracer (snapshot + hoisted apply-flux). Byte-for-byte the region the DC
      !! side times as `redi_calc_coeffs` + `redi_apply_flux_hoist`.
      type(hgrid_t), intent(in) :: grid
      type(ocean_metrics_t), intent(inout) :: metrics
      type(ocean_eos_t), intent(in) :: eos
      type(ocean_redi_t), intent(inout) :: this
      type(multilayer_cgrid_state_t), intent(inout) :: ms
      real(wp), intent(in) :: dt
      real(wp), intent(inout) :: khtr_u_ext(:, :), khtr_v_ext(:, :)

      integer :: nx, ny, nz, ns

      nx = grid%nx_total; ny = grid%ny_total; nz = ms%nz_ml; ns = this%nsurf

      !$acc host_data use_device(ms%h_layer, ms%tracers(1)%hTr, ms%tracers(2)%hTr, &
      !$acc                      metrics%wet_u, metrics%wet_v, &
      !$acc                      this%uPoL, this%uPoR, this%uKoL, this%uKoR, this%uhEff, &
      !$acc                      this%vPoL, this%vPoR, this%vKoL, this%vKoR, this%vhEff, &
      !$acc                      khtr_u_ext, khtr_v_ext, this%khtr_u, this%khtr_v, &
      !$acc                      metrics%dy_cu, metrics%dx_cv, metrics%idxCu, metrics%idyCv, &
      !$acc                      metrics%areaT, this%tr_snap)
      call redi_opt_launch(nx, ny, nz, ns, dt, grid%nghost, grid%nx_phys, grid%ny_phys, &
                           1, 1, 1, 1, &
                           eos%rho0, eos%T_ref, eos%S_ref, eos%alpha_T, eos%beta_S, &
                           ms%h_layer, ms%tracers(1)%hTr, ms%tracers(2)%hTr, &
                           metrics%wet_u, metrics%wet_v, &
                           this%uPoL, this%uPoR, this%uKoL, this%uKoR, this%uhEff, &
                           this%vPoL, this%vPoR, this%vKoL, this%vKoR, this%vhEff, &
                           khtr_u_ext, khtr_v_ext, this%khtr_u, this%khtr_v, &
                           metrics%dy_cu, metrics%dx_cv, metrics%idxCu, metrics%idyCv, &
                           metrics%areaT, this%tr_snap)
      !$acc end host_data
   end subroutine redi_opt_step

end module redi_bridge
