#include "directives.h"
!! RK2 wrapper for the horizontal-viscosity (Smagorinsky) kernel.
!! Entry compute: hvisc_compute_fused (fused single-pass opt-DC).
!! Init copied verbatim from hvisc/drivers/dc_ab.F90.
module rk2_hvisc_mod
   use, intrinsic :: iso_c_binding, only: c_int, c_double
   use constants, only: wp
   use ocean_horizontal_viscosity, only: hvisc_compute_fused, hvisc_compute_smag, hvisc_compute_face
   implicit none
   private
   public :: rk2_hvisc_init, rk2_hvisc_stage, rk2_hvisc_stage_unopt, rk2_hvisc_probe
#ifndef RK2_NO_CUDA
   public :: rk2_hvisc_stage_cuda, rk2_hvisc_stage_cuda_unopt
#endif

   integer, parameter :: NGHOST = 3
   real(wp), parameter :: DX = 1000.0_wp, DY = 1000.0_wp
   real(wp), parameter :: C_SMAG = 0.2_wp, AH_BG = 1.0_wp, AH_MAX = 1.0e5_wp, NS = 0.0_wp

#ifndef RK2_NO_CUDA
   interface
      ! opt: extern "C" hvisc_opt_launch (opt_kernel.cu) -- from hvisc_bridge.F90.
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
      ! faithful two-pass: smag -> ah_face, then face -> du/dv (hvisc_kernel.cu).
      subroutine hvisc_compute_smag_cuda_launch(u_face, v_face, dxT, dyT, idxT, idyT, &
                                  dy_dxBu, dx_dyBu, idyCv, idxCu, wet_q, ah_face_x, ah_face_y, &
                                  c_smag, ah_bg, ah_max, ns, nx, ny, nz, sync) &
         bind(C, name="hvisc_compute_smag_cuda_launch")
         import :: wp, c_int, c_double
         real(wp) :: u_face(*), v_face(*), dxT(*), dyT(*), idxT(*), idyT(*)
         real(wp) :: dy_dxBu(*), dx_dyBu(*), idyCv(*), idxCu(*), wet_q(*)
         real(wp) :: ah_face_x(*), ah_face_y(*)
         real(c_double), value :: c_smag, ah_bg, ah_max, ns
         integer(c_int), value :: nx, ny, nz, sync
      end subroutine hvisc_compute_smag_cuda_launch
      subroutine hvisc_compute_face_cuda_launch(u_face, v_face, ah_face_x, ah_face_y, &
                                  du_visc, dv_visc, dy_dxT, dx_dyBu, iareaCu, &
                                  dx_dyT, dy_dxBu, iareaCv, nx, ny, nz, sync) &
         bind(C, name="hvisc_compute_face_cuda_launch")
         import :: wp, c_int
         real(wp) :: u_face(*), v_face(*), ah_face_x(*), ah_face_y(*)
         real(wp) :: du_visc(*), dv_visc(*)
         real(wp) :: dy_dxT(*), dx_dyBu(*), iareaCu(*), dx_dyT(*), dy_dxBu(*), iareaCv(*)
         integer(c_int), value :: nx, ny, nz, sync
      end subroutine hvisc_compute_face_cuda_launch
   end interface
#endif

   real(wp), allocatable, save :: u_face(:, :, :), v_face(:, :, :)
   real(wp), allocatable, save :: du(:, :, :), dv(:, :, :), ahx(:, :, :), ahy(:, :, :)
   real(wp), allocatable, save :: dxT(:, :), dyT(:, :), idxT(:, :), idyT(:, :)
   real(wp), allocatable, save :: dy_dxBu(:, :), dx_dyBu(:, :), idyCv(:, :), idxCu(:, :), wet_q(:, :)
   real(wp), allocatable, save :: dy_dxT(:, :), iareaCu(:, :), dx_dyT(:, :), iareaCv(:, :)
   integer, save :: gnx, gny, gnz

contains

   subroutine rk2_hvisc_init(nxp, nyp, nz) bind(C, name="rk2_hvisc_init")
      integer(c_int), value :: nxp, nyp, nz
      integer :: i, j, k, nx, ny
      real(wp) :: id, ia
      nx = nxp + 2*NGHOST; ny = nyp + 2*NGHOST
      gnx = nx; gny = ny; gnz = nz
      id = 1.0_wp/DX; ia = 1.0_wp/(DX*DY)

      allocate (u_face(nx + 1, ny, nz), v_face(nx, ny + 1, nz))
      allocate (du(nx + 1, ny, nz), dv(nx, ny + 1, nz))
      allocate (ahx(nx + 1, ny, nz), ahy(nx, ny + 1, nz))
      allocate (dxT(nx, ny), dyT(nx, ny), idxT(nx, ny), idyT(nx, ny))
      allocate (dy_dxBu(nx + 1, ny + 1), dx_dyBu(nx + 1, ny + 1), idyCv(nx, ny + 1), idxCu(nx + 1, ny), wet_q(nx + 1, ny + 1))
      allocate (dy_dxT(nx, ny), iareaCu(nx + 1, ny), dx_dyT(nx, ny), iareaCv(nx, ny + 1))

      do k = 1, nz
         do j = 1, ny
            do i = 1, nx + 1
               u_face(i, j, k) = (1.0_wp + 0.1_wp*real(k, wp))*sin(0.02_wp*real(i, wp))*cos(0.03_wp*real(j, wp))
            end do
         end do
      end do
      do k = 1, nz
         do j = 1, ny + 1
            do i = 1, nx
               v_face(i, j, k) = (1.0_wp + 0.1_wp*real(k, wp))*cos(0.017_wp*real(i, wp))*sin(0.023_wp*real(j, wp))
            end do
         end do
      end do
      dxT = DX; dyT = DY; idxT = id; idyT = id
      dy_dxBu = 1.0_wp; dx_dyBu = 1.0_wp; idyCv = id; idxCu = id; wet_q = 1.0_wp
      dy_dxT = 1.0_wp; iareaCu = ia; dx_dyT = 1.0_wp; iareaCv = ia
      du = 0.0_wp; dv = 0.0_wp; ahx = 0.0_wp; ahy = 0.0_wp

      DC_ENTER_IN(u_face)
      DC_ENTER_IN(v_face)
      DC_ENTER_IN(dxT)
      DC_ENTER_IN(dyT)
      DC_ENTER_IN(idxT)
      DC_ENTER_IN(idyT)
      DC_ENTER_IN(dy_dxBu)
      DC_ENTER_IN(dx_dyBu)
      DC_ENTER_IN(idyCv)
      DC_ENTER_IN(idxCu)
      DC_ENTER_IN(wet_q)
      DC_ENTER_IN(dy_dxT)
      DC_ENTER_IN(iareaCu)
      DC_ENTER_IN(dx_dyT)
      DC_ENTER_IN(iareaCv)
      DC_ENTER_CREATE(du)
      DC_ENTER_CREATE(dv)
      DC_ENTER_CREATE(ahx)
      DC_ENTER_CREATE(ahy)
      DC_WAIT
   end subroutine rk2_hvisc_init

   subroutine rk2_hvisc_stage() bind(C, name="rk2_hvisc_stage")
      call hvisc_compute_fused(u_face, v_face, du, dv, dxT, dyT, idxT, idyT, &
                               dy_dxBu, dx_dyBu, idyCv, idxCu, wet_q, dy_dxT, iareaCu, &
                               dx_dyT, iareaCv, C_SMAG, AH_BG, AH_MAX, NS, gnx, gny, gnz)
   end subroutine rk2_hvisc_stage

   subroutine rk2_hvisc_stage_unopt() bind(C, name="rk2_hvisc_stage_unopt")
      ! Faithful two-pass: smag -> ah_face, then face -> du/dv.
      call hvisc_compute_smag(u_face, v_face, ahx, ahy, dxT, dyT, idxT, idyT, &
                              dy_dxBu, dx_dyBu, idyCv, idxCu, wet_q, C_SMAG, AH_BG, AH_MAX, NS, gnx, gny, gnz)
      call hvisc_compute_face(u_face, v_face, ahx, ahy, du, dv, &
                              dy_dxT, dx_dyBu, iareaCu, dx_dyT, dy_dxBu, iareaCv, gnx, gny, gnz)
   end subroutine rk2_hvisc_stage_unopt

#ifndef RK2_NO_CUDA
   subroutine rk2_hvisc_stage_cuda() bind(C, name="rk2_hvisc_stage_cuda")
      !$acc host_data use_device(u_face, v_face, dxT, dyT, idxT, idyT, &
      !$acc                      dy_dxBu, dx_dyBu, idyCv, idxCu, wet_q, &
      !$acc                      dy_dxT, iareaCu, dx_dyT, iareaCv, du, dv)
      call hvisc_opt_launch(u_face, v_face, dxT, dyT, idxT, idyT, &
                            dy_dxBu, dx_dyBu, idyCv, idxCu, wet_q, &
                            dy_dxT, iareaCu, dx_dyT, iareaCv, &
                            du, dv, C_SMAG, AH_BG, AH_MAX, NS, gnx, gny, gnz, 0)
      !$acc end host_data
   end subroutine rk2_hvisc_stage_cuda

   subroutine rk2_hvisc_stage_cuda_unopt() bind(C, name="rk2_hvisc_stage_cuda_unopt")
      ! Faithful two-pass CUDA: smag launch then face launch, ah_face materialised.
      !$acc host_data use_device(u_face, v_face, dxT, dyT, idxT, idyT, &
      !$acc                      dy_dxBu, dx_dyBu, idyCv, idxCu, wet_q, ahx, ahy)
      call hvisc_compute_smag_cuda_launch(u_face, v_face, dxT, dyT, idxT, idyT, &
                            dy_dxBu, dx_dyBu, idyCv, idxCu, wet_q, ahx, ahy, &
                            C_SMAG, AH_BG, AH_MAX, NS, gnx, gny, gnz, 0)
      !$acc end host_data
      !$acc host_data use_device(u_face, v_face, ahx, ahy, du, dv, &
      !$acc                      dy_dxT, dx_dyBu, iareaCu, dx_dyT, dy_dxBu, iareaCv)
      call hvisc_compute_face_cuda_launch(u_face, v_face, ahx, ahy, du, dv, &
                            dy_dxT, dx_dyBu, iareaCu, dx_dyT, dy_dxBu, iareaCv, gnx, gny, gnz, 0)
      !$acc end host_data
   end subroutine rk2_hvisc_stage_cuda_unopt
#endif

   subroutine rk2_hvisc_probe(vmin, vmax) bind(C, name="rk2_hvisc_probe")
      real(c_double), intent(out) :: vmin, vmax
      DC_UPDATE_SELF(du)
      vmin = minval(du); vmax = maxval(du)
   end subroutine rk2_hvisc_probe

end module rk2_hvisc_mod
