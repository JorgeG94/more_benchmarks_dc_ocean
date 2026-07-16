!! Driver: FULL ocean horizontal-viscosity Smagorinsky closure --
!!   stage 1  hvisc_compute_smag : strain -> per-face viscosity ah_face
!!   stage 2  hvisc_compute_face : ah_face x curvilinear-FV velocity Laplacian
!! `do concurrent` vs a faithful CUDA C port, both reading the SAME device
!! allocation (OpenACC owns it, CUDA via host_data use_device). Compares BOTH
!! the viscosity field and the tendency, bit-for-bit.
!!
!! Uniform square metrics (dx=dy): the curvilinear ratios are 1, iareaC =
!! 1/(dx*dy), id* = 1/dx, all-wet corners -> the common production case; the
!! kernels still evaluate the full form. Free-slip (ns=0).
!!
!! Usage: ./hvisc_bench [nx_phys] [ny_phys] [nz] [nreps] [nwarm] [cuda_sync]
program hvisc_bench
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use, intrinsic :: iso_c_binding, only: c_double, c_int
   use constants, only: wp
   use ocean_horizontal_viscosity, only: hvisc_compute_smag, hvisc_compute_face
   implicit none

   interface
      subroutine hvisc_compute_smag_cuda_launch(u_face, v_face, dxT, dyT, idxT, idyT, &
                 dy_dxBu, dx_dyBu, idyCv, idxCu, wet_q, ah_face_x, ah_face_y, &
                 c_smag, ah_bg, ah_max, ns, nx, ny, nz, sync) bind(C, name="hvisc_compute_smag_cuda_launch")
         import :: c_double, c_int
         real(c_double), intent(in) :: u_face(*), v_face(*), dxT(*), dyT(*), idxT(*), idyT(*)
         real(c_double), intent(in) :: dy_dxBu(*), dx_dyBu(*), idyCv(*), idxCu(*), wet_q(*)
         real(c_double), intent(inout) :: ah_face_x(*), ah_face_y(*)
         real(c_double), value :: c_smag, ah_bg, ah_max, ns
         integer(c_int), value :: nx, ny, nz, sync
      end subroutine
      subroutine hvisc_compute_face_cuda_launch(u_face, v_face, ah_face_x, ah_face_y, &
                 du_visc, dv_visc, dy_dxT, dx_dyBu, iareaCu, dx_dyT, dy_dxBu, iareaCv, &
                 nx, ny, nz, sync) bind(C, name="hvisc_compute_face_cuda_launch")
         import :: c_double, c_int
         real(c_double), intent(in) :: u_face(*), v_face(*), ah_face_x(*), ah_face_y(*)
         real(c_double), intent(inout) :: du_visc(*), dv_visc(*)
         real(c_double), intent(in) :: dy_dxT(*), dx_dyBu(*), iareaCu(*), dx_dyT(*), dy_dxBu(*), iareaCv(*)
         integer(c_int), value :: nx, ny, nz, sync
      end subroutine
   end interface

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30
   integer, parameter :: REPS_DEF = 200, WARM_DEF = 10, NGHOST = 3
   real(wp), parameter :: DX = 1000.0_wp, DY = 1000.0_wp
   real(wp), parameter :: C_SMAG = 0.2_wp, AH_BG = 1.0_wp, AH_MAX = 1.0e5_wp, NS = 0.0_wp

   real(wp), allocatable :: u_face(:, :, :), v_face(:, :, :)
   real(wp), allocatable :: ahx_dc(:, :, :), ahy_dc(:, :, :), ahx_cu(:, :, :), ahy_cu(:, :, :)
   real(wp), allocatable :: du_dc(:, :, :), dv_dc(:, :, :), du_cu(:, :, :), dv_cu(:, :, :)
   real(wp), allocatable :: dxT(:, :), dyT(:, :), idxT(:, :), idyT(:, :)
   real(wp), allocatable :: dy_dxBu(:, :), dx_dyBu(:, :), idyCv(:, :), idxCu(:, :), wet_q(:, :)
   real(wp), allocatable :: dy_dxT(:, :), iareaCu(:, :), dx_dyT(:, :), iareaCv(:, :)
   real(wp) :: t0, t1, ms_dc, ms_cu, id, ia, ahmax, dumax, rmax, sc
   integer :: i, j, k, rep, nx, ny, nz, nxp, nyp, n_reps, n_warm, cu_sync

   nxp = iarg(1, NXP_DEF); nyp = iarg(2, NYP_DEF); nz = iarg(3, NZ_DEF)
   n_reps = iarg(4, REPS_DEF); n_warm = iarg(5, WARM_DEF); cu_sync = iarg(6, 2)
   nx = nxp + 2*NGHOST; ny = nyp + 2*NGHOST
   if (nx < 4 .or. ny < 4 .or. nz < 1) then
      write (output_unit, '(a)') 'ERROR: need nx_phys,ny_phys >= 1 and nz >= 1'; stop 1
   end if
   id = 1.0_wp/DX; ia = 1.0_wp/(DX*DY)

   allocate (u_face(nx + 1, ny, nz), v_face(nx, ny + 1, nz))
   allocate (ahx_dc(nx + 1, ny, nz), ahy_dc(nx, ny + 1, nz), ahx_cu(nx + 1, ny, nz), ahy_cu(nx, ny + 1, nz))
   allocate (du_dc(nx + 1, ny, nz), dv_dc(nx, ny + 1, nz), du_cu(nx + 1, ny, nz), dv_cu(nx, ny + 1, nz))
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
   ahx_dc = 0.0_wp; ahy_dc = 0.0_wp; ahx_cu = 0.0_wp; ahy_cu = 0.0_wp
   du_dc = 0.0_wp; dv_dc = 0.0_wp; du_cu = 0.0_wp; dv_cu = 0.0_wp

   write (output_unit, '(a)') repeat('=', 70)
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  domain: ', nxp, ' x ', nyp, ' x ', nz, &
      ' interior (+', NGHOST, ' ghosts/side)'
   write (output_unit, '(a,i0,a,i0,a,i0)') '  arrays: ', nx, ' x ', ny, ' x ', nz
   write (output_unit, '(a,i0,a,i0,a,i0)') '  full closure (smag + apply); reps ', n_reps, ' timed, ', n_warm, &
      ' warm; cuda_sync = ', cu_sync
   write (output_unit, '(a)') repeat('=', 70)

   !$acc enter data copyin(u_face, v_face, dxT, dyT, idxT, idyT, dy_dxBu, dx_dyBu, &
   !$acc                   idyCv, idxCu, wet_q, dy_dxT, iareaCu, dx_dyT, iareaCv)
   !$acc enter data create(ahx_dc, ahy_dc, ahx_cu, ahy_cu, du_dc, dv_dc, du_cu, dv_cu)

   ! ---- A: do concurrent full closure --------------------------------------
   do rep = 1, n_warm
      call dc_closure()
   end do
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call dc_closure()
   end do
   !$acc wait
   t1 = wall()
   ms_dc = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- B: faithful CUDA C full closure ------------------------------------
   do rep = 1, n_warm
      call cu_closure(1)
   end do
   t0 = wall()
   do rep = 1, n_reps
      call cu_closure(cu_sync)
   end do
   !$acc wait
   t1 = wall()
   ms_cu = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   !$acc update self(ahx_dc, ahy_dc, ahx_cu, ahy_cu, du_dc, dv_dc, du_cu, dv_cu)

   ! stage-1 (viscosity) and stage-2 (tendency) agreement, over both faces
   ahmax = 0.0_wp; dumax = 0.0_wp; rmax = 0.0_wp
   call cmp(ahx_dc, ahx_cu, ahmax, rmax)
   call cmp(ahy_dc, ahy_cu, ahmax, rmax)
   call cmp(du_dc, du_cu, dumax, rmax)
   call cmp(dv_dc, dv_cu, dumax, rmax)

   write (output_unit, '(a,f10.4,a)') '  do concurrent (production)   : ', ms_dc, ' ms/rep'
   write (output_unit, '(a,f10.4,a)') '  CUDA C (faithful port)       : ', ms_cu, ' ms/rep'
   write (output_unit, '(a,f10.3,a)') '  ratio  dc / cuda             : ', ms_dc/ms_cu, ' x'
   write (output_unit, '(a)') ''
   write (output_unit, '(a,es12.5)') '  DC vs CUDA  max|d ah_face| ', ahmax
   write (output_unit, '(a,es12.5)') '  DC vs CUDA  max|d tend|    ', dumax
   write (output_unit, '(a,es12.5)') '              max rel diff   ', rmax
   if (rmax < 1.0e-12_wp) then
      write (output_unit, '(a)') '              agreement      : OK (<1e-12 rel -> FMA contraction only)'
   else
      write (output_unit, '(a)') '              agreement      : *** SUSPECT -- likely a PORT BUG ***'
   end if
   write (output_unit, '(a,es14.6,a,es14.6)') '  sanity: sum ah_x ', sum(ahx_dc), '  sum du ', sum(du_dc)
   write (output_unit, '(a)') repeat('=', 70)

contains

   subroutine dc_closure()
      call hvisc_compute_smag(u_face, v_face, ahx_dc, ahy_dc, dxT, dyT, idxT, idyT, &
                              dy_dxBu, dx_dyBu, idyCv, idxCu, wet_q, C_SMAG, AH_BG, AH_MAX, NS, nx, ny, nz)
      call hvisc_compute_face(u_face, v_face, ahx_dc, ahy_dc, du_dc, dv_dc, &
                              dy_dxT, dx_dyBu, iareaCu, dx_dyT, dy_dxBu, iareaCv, nx, ny, nz)
   end subroutine dc_closure

   subroutine cu_closure(sync)
      integer, intent(in) :: sync
      !$acc host_data use_device(u_face, v_face, dxT, dyT, idxT, idyT, dy_dxBu, dx_dyBu, &
      !$acc                      idyCv, idxCu, wet_q, ahx_cu, ahy_cu, du_cu, dv_cu, dy_dxT, iareaCu, dx_dyT, iareaCv)
      call hvisc_compute_smag_cuda_launch(u_face, v_face, dxT, dyT, idxT, idyT, dy_dxBu, dx_dyBu, &
                                          idyCv, idxCu, wet_q, ahx_cu, ahy_cu, C_SMAG, AH_BG, AH_MAX, NS, nx, ny, nz, sync)
      call hvisc_compute_face_cuda_launch(u_face, v_face, ahx_cu, ahy_cu, du_cu, dv_cu, &
                                          dy_dxT, dx_dyBu, iareaCu, dx_dyT, dy_dxBu, iareaCv, nx, ny, nz, sync)
      !$acc end host_data
   end subroutine cu_closure

   subroutine cmp(a, b, amax, relmax)
      real(wp), intent(in) :: a(:, :, :), b(:, :, :)
      real(wp), intent(inout) :: amax, relmax
      integer :: ii, jj, kk
      real(wp) :: d, s
      do kk = 1, size(a, 3)
         do jj = 1, size(a, 2)
            do ii = 1, size(a, 1)
               d = abs(a(ii, jj, kk) - b(ii, jj, kk)); amax = max(amax, d)
               s = max(abs(a(ii, jj, kk)), abs(b(ii, jj, kk)))
               if (s > 1.0e-30_wp) relmax = max(relmax, d/s)
            end do
         end do
      end do
   end subroutine cmp

   integer function iarg(k, dflt)
      integer, intent(in) :: k, dflt
      character(len=32) :: buf
      integer :: ln, st
      iarg = dflt
      if (command_argument_count() < k) return
      call get_command_argument(k, buf, ln, st)
      if (st /= 0 .or. ln == 0) return
      read (buf, *, iostat=st) iarg
      if (st /= 0) iarg = dflt
   end function iarg

   function wall() result(t)
      real(wp) :: t
      integer(int64) :: cnt, rate
      call system_clock(cnt, rate)
      t = real(cnt, wp)/real(rate, wp)
   end function wall
end program hvisc_bench
