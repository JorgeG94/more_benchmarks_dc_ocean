!! Bridge from the DC driver to the optimized CUDA launcher.
!!
!! Hands `hvisc_opt_launch` the SAME device allocations the `do concurrent`
!! variant uses, via `!$acc host_data use_device`. No copies: both toolchains
!! read one truth, so the head-to-head times two paths on one set of device
!! arrays and can diff their outputs in-place.
!!
!! WARNING: THIS FILE MUST NOT BE COMPILED WITH `-cuda`. That flag is LINK-ONLY
!! here: it switches nvfortran into CUDA Fortran mode, which type-checks device
!! attributes -- inside `host_data use_device` the arrays carry the device
!! attribute while the bind(C) interface declares plain host arrays, giving
!! NVFORTRAN-S-0528 device attribute mismatch.
module hvisc_bridge
   use constants, only: wp
   use, intrinsic :: iso_c_binding, only: c_int, c_double
   implicit none
   private
   public :: hvisc_opt_step

   interface
      !! extern "C" void hvisc_opt_launch(const double*..., double c_smag,
      !!   double ah_bg, double ah_max, double ns, int nx, int ny, int nz, int sync)
      !! Arrays are device pointers (assumed-size, passed inside host_data);
      !! the trailing scalars are BY VALUE, hence the `value` attribute.
      subroutine hvisc_opt_launch(u_face, v_face, dxT, dyT, idxT, idyT, &
                                  dy_dxBu, dx_dyBu, idyCv, idxCu, wet_q, &
                                  dy_dxT, iareaCu, dx_dyT, iareaCv, &
                                  du_visc, dv_visc, &
                                  c_smag, ah_bg, ah_max, ns, nx, ny, nz, sync) &
         bind(C, name="hvisc_opt_launch")
         import :: wp, c_int, c_double
         real(wp) :: u_face(*), v_face(*)
         real(wp) :: dxT(*), dyT(*), idxT(*), idyT(*)
         real(wp) :: dy_dxBu(*), dx_dyBu(*), idyCv(*), idxCu(*), wet_q(*)
         real(wp) :: dy_dxT(*), iareaCu(*), dx_dyT(*), iareaCv(*)
         real(wp) :: du_visc(*), dv_visc(*)
         real(c_double), value :: c_smag, ah_bg, ah_max, ns
         integer(c_int), value :: nx, ny, nz, sync
      end subroutine hvisc_opt_launch
   end interface

contains

   subroutine hvisc_opt_step(u_face, v_face, du_visc, dv_visc, &
                             dxT, dyT, idxT, idyT, dy_dxBu, dx_dyBu, &
                             idyCv, idxCu, wet_q, dy_dxT, iareaCu, &
                             dx_dyT, iareaCv, c_smag, ah_bg, ah_max, ns, &
                             nx, ny, nz)
      !! Same argument shape as hvisc_compute_fused. Wraps the CUDA launch in a
      !! host_data region so the launcher gets the device copies the DC routine
      !! wrote/reads. sync=1 -> the launcher does one cudaDeviceSynchronize so
      !! the host-side system_clock window bounds the kernel.
      integer, intent(in) :: nx, ny, nz
      real(wp), intent(in)    :: u_face(nx + 1, ny, nz), v_face(nx, ny + 1, nz)
      real(wp), intent(inout) :: du_visc(nx + 1, ny, nz), dv_visc(nx, ny + 1, nz)
      real(wp), intent(in) :: dxT(nx, ny), dyT(nx, ny), idxT(nx, ny), idyT(nx, ny)
      real(wp), intent(in) :: dy_dxBu(nx + 1, ny + 1), dx_dyBu(nx + 1, ny + 1)
      real(wp), intent(in) :: idyCv(nx, ny + 1), idxCu(nx + 1, ny), wet_q(nx + 1, ny + 1)
      real(wp), intent(in) :: dy_dxT(nx, ny), iareaCu(nx + 1, ny)
      real(wp), intent(in) :: dx_dyT(nx, ny), iareaCv(nx, ny + 1)
      real(wp), intent(in) :: c_smag, ah_bg, ah_max, ns

      !$acc host_data use_device(u_face, v_face, dxT, dyT, idxT, idyT, &
      !$acc                      dy_dxBu, dx_dyBu, idyCv, idxCu, wet_q, &
      !$acc                      dy_dxT, iareaCu, dx_dyT, iareaCv, du_visc, dv_visc)
      call hvisc_opt_launch(u_face, v_face, dxT, dyT, idxT, idyT, &
                            dy_dxBu, dx_dyBu, idyCv, idxCu, wet_q, &
                            dy_dxT, iareaCu, dx_dyT, iareaCv, &
                            du_visc, dv_visc, &
                            c_smag, ah_bg, ah_max, ns, nx, ny, nz, 1)
      !$acc end host_data
   end subroutine hvisc_opt_step

end module hvisc_bridge
