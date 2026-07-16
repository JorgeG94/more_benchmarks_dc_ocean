!! Bridge from the driver to the CUDA C port.
!!
!! Hands the CUDA kernels the SAME device allocations the `do concurrent`
!! variant uses, via `!$acc host_data use_device`. No copies: both toolchains
!! read one truth.
!!
!! ⚠ THIS FILE MUST NOT BE COMPILED WITH `-cuda`. That flag is LINK-ONLY here:
!! it also switches nvfortran into CUDA Fortran mode, which type-checks device
!! attributes -- inside `host_data use_device` the arrays carry the device
!! attribute while the bind(C) interface declares plain host arrays, giving
!! NVFORTRAN-S-0528 device attribute mismatch.
module redi_cuda
   use constants, only: wp
   use grid, only: hgrid_t
   use ocean_metrics, only: ocean_metrics_t
   use multilayer_cgrid_state, only: multilayer_cgrid_state_t
   use ocean_eos, only: ocean_eos_t
   use ocean_redi, only: ocean_redi_t
   implicit none
   private

   public :: redi_cuda_step
   public :: redi_cuda_probe

   interface
      !! Same signature as redi_cuda_launch, but redi_probe.cu's no-op body.
      !! Lets redi_cuda_probe time the host_data present-table lookup ALONE.
      subroutine redi_noop(nx, ny, nz, ns, dt, nghost, nxp, nyp, &
                           ww, we, wsz, wn, rho0, T_ref, S_ref, alpha_T, beta_S, &
                           h_layer, t_htr, s_htr, wet_u, wet_v, &
                           uPoL, uPoR, uKoL, uKoR, uhEff, &
                           vPoL, vPoR, vKoL, vKoR, vhEff, &
                           khtr_u_ext, khtr_v_ext, khtr_u, khtr_v, &
                           dy_cu, dx_cv, idxCu, idyCv, areaT, tr_snap) &
         bind(C, name="redi_noop_")
         import :: wp
         integer :: nx, ny, nz, ns, nghost, nxp, nyp, ww, we, wsz, wn
         real(wp) :: dt, rho0, T_ref, S_ref, alpha_T, beta_S
         real(wp) :: h_layer(*), t_htr(*), s_htr(*), wet_u(*), wet_v(*)
         real(wp) :: uPoL(*), uPoR(*), uhEff(*), vPoL(*), vPoR(*), vhEff(*)
         integer :: uKoL(*), uKoR(*), vKoL(*), vKoR(*)
         real(wp) :: khtr_u_ext(*), khtr_v_ext(*), khtr_u(*), khtr_v(*)
         real(wp) :: dy_cu(*), dx_cv(*), idxCu(*), idyCv(*), areaT(*), tr_snap(*)
      end subroutine redi_noop
   end interface

   interface
      subroutine redi_cuda_launch(nx, ny, nz, ns, dt, nghost, nxp, nyp, &
                                  ww, we, wsz, wn, rho0, T_ref, S_ref, alpha_T, beta_S, &
                                  h_layer, t_htr, s_htr, wet_u, wet_v, &
                                  uPoL, uPoR, uKoL, uKoR, uhEff, &
                                  vPoL, vPoR, vKoL, vKoR, vhEff, &
                                  khtr_u_ext, khtr_v_ext, khtr_u, khtr_v, &
                                  dy_cu, dx_cv, idxCu, idyCv, areaT, tr_snap) &
         bind(C, name="redi_cuda_launch_")
         import :: wp
         integer :: nx, ny, nz, ns, nghost, nxp, nyp, ww, we, wsz, wn
         real(wp) :: dt, rho0, T_ref, S_ref, alpha_T, beta_S
         real(wp) :: h_layer(*), t_htr(*), s_htr(*), wet_u(*), wet_v(*)
         real(wp) :: uPoL(*), uPoR(*), uhEff(*), vPoL(*), vPoR(*), vhEff(*)
         integer :: uKoL(*), uKoR(*), vKoL(*), vKoR(*)
         real(wp) :: khtr_u_ext(*), khtr_v_ext(*), khtr_u(*), khtr_v(*)
         real(wp) :: dy_cu(*), dx_cv(*), idxCu(*), idyCv(*), areaT(*), tr_snap(*)
      end subroutine redi_cuda_launch
   end interface

contains

   subroutine redi_cuda_step(grid, metrics, eos, this, ms, dt, khtr_u_ext, khtr_v_ext)
      !! Mirrors ONE production `redi_calc_coeffs` + `redi_apply_flux` pair.
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
      call redi_cuda_launch(nx, ny, nz, ns, dt, grid%nghost, grid%nx_phys, grid%ny_phys, &
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
   end subroutine redi_cuda_step

   subroutine redi_cuda_probe(grid, metrics, eos, this, ms, dt, khtr_u_ext, khtr_v_ext)
      !! BYTE-FOR-BYTE redi_cuda_step, except the callee is redi_probe.cu's
      !! no-op instead of redi_cuda_launch. The difference between timing this
      !! and timing redi_cuda_step is therefore exactly the 8 kernels; what
      !! THIS costs is the `host_data use_device` present-table lookup for 24
      !! arrays, plus the bind(C) call.
      !!
      !! It is a HOST-side measurement, so unlike the redi_native A/B it stays
      !! valid while sibling jobs saturate the GPU. Keep the host_data list
      !! below identical to redi_cuda_step's or the probe measures nothing.
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
      call redi_noop(nx, ny, nz, ns, dt, grid%nghost, grid%nx_phys, grid%ny_phys, &
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
   end subroutine redi_cuda_probe

end module redi_cuda
