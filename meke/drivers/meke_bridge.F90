!! Bridge from the shared cmp driver to the OPTIMIZED MEKE CUDA launcher.
!!
!! Hands `meke_opt_launch_flat` (opt_kernel.cu) the SAME device allocations the
!! `do concurrent` variant (`meke_step_ext_fused`) steps, via `!$acc host_data
!! use_device`. No copies: both toolchains read and write one truth, so the
!! head-to-head has a single memory image and the two-harness caveat is gone.
!!
!! WHY A FLAT WRAPPER: the real entry point is
!!   extern "C" void meke_opt_launch(const MekeArgs*, double *meke_scratch)
!! which takes a struct of device pointers -- awkward to build from Fortran. The
!! C side adds `meke_opt_launch_flat`, every array + scalar as a plain argument;
!! this module mirrors it as one uniform bind(C) interface (arrays as device
!! pointers, scalars/ints by value). Argument ORDER here must match the flat
!! wrapper in opt_kernel.cu field-for-field.
!!
!! ⚠ THIS FILE MUST NOT BE COMPILED WITH `-cuda`. That flag is LINK-ONLY: it
!! switches nvfortran into CUDA Fortran mode, which type-checks device attributes
!! -- inside `host_data use_device` the arrays carry the device attribute while
!! this bind(C) interface declares plain host arrays, giving NVFORTRAN-S-0528.
!! (Same rule as legacy_testing/redi_benchmark/redi_cuda.F90.)
module meke_bridge
   use constants, only: wp
   use meke_state, only: ocean_metrics_t, multilayer_cgrid_state_t, ocean_gm_t, ocean_meke_t
   implicit none
   private
   public :: meke_opt_cuda_step

   interface
      subroutine meke_opt_launch_flat( &
         meke, kh_diff, le, ku, i_mass, depth_tot, bottom_fac2, barotr_fac2, src, &
         uflux, vflux, mass_ws, rd_ws, sn_u_ws, sn_v_ws, ke_diss_ws, &
         u_bbl2, f_centre, gm_src, ke_diss_ext, h_layer, rho_layer, &
         areaT, iareaT, idxT, idyT, dy_cu, dx_cv, idxCu, idyCv, &
         nx, ny, nz, &
         dt, dtscale, cd_scale, cb, ct, min_gamma2, cdrag, uscale, &
         a_deform, a_rhines, a_eady, a_frict, a_grid, &
         bgsrc, gmcoeff, frcoeff, damping, &
         kh_bg, k4, khmeke_fac, khcoeff, visc_coeff_ku, rho0, backscatter, &
         meke_scratch) &
         bind(C, name="meke_opt_launch_flat")
         use, intrinsic :: iso_c_binding, only: c_int
         import :: wp
         ! device arrays -- passed by reference (the host_data device address)
         real(wp) :: meke(*), kh_diff(*), le(*), ku(*), i_mass(*), depth_tot(*)
         real(wp) :: bottom_fac2(*), barotr_fac2(*), src(*), uflux(*), vflux(*)
         real(wp) :: mass_ws(*), rd_ws(*), sn_u_ws(*), sn_v_ws(*), ke_diss_ws(*)
         real(wp) :: u_bbl2(*), f_centre(*), gm_src(*), ke_diss_ext(*)
         real(wp) :: h_layer(*), rho_layer(*)
         real(wp) :: areaT(*), iareaT(*), idxT(*), idyT(*)
         real(wp) :: dy_cu(*), dx_cv(*), idxCu(*), idyCv(*)
         real(wp) :: meke_scratch(*)
         ! sizes + scalars -- by value
         integer(c_int), value :: nx, ny, nz, backscatter
         real(wp), value :: dt, dtscale, cd_scale, cb, ct, min_gamma2, cdrag, uscale
         real(wp), value :: a_deform, a_rhines, a_eady, a_frict, a_grid
         real(wp), value :: bgsrc, gmcoeff, frcoeff, damping
         real(wp), value :: kh_bg, k4, khmeke_fac, khcoeff, visc_coeff_ku, rho0
      end subroutine meke_opt_launch_flat
   end interface

contains

   !! One optimized-CUDA MEKE step on the SAME device arrays meke_step_ext_fused
   !! uses. `ext%ke_diss_ext` and `scratch` are the driver-owned arrays not held
   !! inside the derived types; both are already on the device (enter_data).
   subroutine meke_opt_cuda_step(nx, ny, nz, m, met, gm, ms, dt, ke_diss_ext, scratch)
      use, intrinsic :: iso_c_binding, only: c_int
      integer, intent(in) :: nx, ny, nz
      type(ocean_meke_t), intent(inout) :: m
      type(ocean_metrics_t), intent(in) :: met
      type(ocean_gm_t), intent(in) :: gm
      type(multilayer_cgrid_state_t), intent(in) :: ms
      real(wp), intent(in) :: dt
      real(wp), intent(inout) :: ke_diss_ext(:, :)
      real(wp), intent(inout) :: scratch(:, :)
      integer(c_int) :: bscat

      bscat = 0
      if (m%backscatter) bscat = 1

      !$acc host_data use_device(m%meke, m%kh_diff, m%le, m%ku, m%i_mass, &
      !$acc   m%depth_tot, m%bottom_fac2, m%barotr_fac2, m%src, m%uflux, m%vflux, &
      !$acc   m%mass_ws, m%rd_ws, m%sn_u_ws, m%sn_v_ws, m%ke_diss_ws, &
      !$acc   m%u_bbl2, m%f_centre, gm%gm_src, ke_diss_ext, ms%h_layer, ms%rho_layer, &
      !$acc   met%areaT, met%iareaT, met%idxT, met%idyT, met%dy_cu, met%dx_cv, &
      !$acc   met%idxCu, met%idyCv, scratch)
      call meke_opt_launch_flat( &
         m%meke, m%kh_diff, m%le, m%ku, m%i_mass, m%depth_tot, m%bottom_fac2, &
         m%barotr_fac2, m%src, m%uflux, m%vflux, m%mass_ws, m%rd_ws, &
         m%sn_u_ws, m%sn_v_ws, m%ke_diss_ws, &
         m%u_bbl2, m%f_centre, gm%gm_src, ke_diss_ext, ms%h_layer, ms%rho_layer, &
         met%areaT, met%iareaT, met%idxT, met%idyT, met%dy_cu, met%dx_cv, &
         met%idxCu, met%idyCv, &
         int(nx, c_int), int(ny, c_int), int(nz, c_int), &
         dt, m%dtscale, m%cd_scale, m%cb, m%ct, m%min_gamma2, m%cdrag, m%uscale, &
         m%alpha_deform, m%alpha_rhines, m%alpha_eady, m%alpha_frict, m%alpha_grid, &
         m%bgsrc, m%gmcoeff, m%frcoeff, m%damping, &
         m%kh, m%k4, m%khmeke_fac, m%khcoeff, m%visc_coeff_ku, gm%rho0, bscat, &
         scratch)
      !$acc end host_data
   end subroutine meke_opt_cuda_step

end module meke_bridge
