#include "directives.h"
!! DC-only driver for the ocean horizontal-viscosity Smagorinsky closure
!! (hvisc_compute_smag -> ah_face, then hvisc_compute_face -> tendency).
!! COMPUTE is bare `do concurrent`; the device data layer is chosen by
!! directives.h:  -DDC_DATA_ACC (OpenACC GPU) / -DDC_DATA_OMP (OpenMP target) /
!! neither (CPU, -stdpar=multicore). No CUDA in this binary.
!!
!!   DC_DUMP=file  writes nx,ny,nz + du_visc  (a reference)
!!   DC_REF=file   reads it and reports max|diff| vs this run
!! Usage: ./dc_main [nx_phys] [ny_phys] [nz] [nreps] [nwarm]
program dc_main
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use constants, only: wp
   use ocean_horizontal_viscosity, only: hvisc_compute_smag, hvisc_compute_face
   implicit none

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30
   integer, parameter :: REPS_DEF = 200, WARM_DEF = 10, NGHOST = 3
   real(wp), parameter :: DX = 1000.0_wp, DY = 1000.0_wp
   real(wp), parameter :: C_SMAG = 0.2_wp, AH_BG = 1.0_wp, AH_MAX = 1.0e5_wp, NS = 0.0_wp

   real(wp), allocatable :: u_face(:, :, :), v_face(:, :, :), ahx(:, :, :), ahy(:, :, :)
   real(wp), allocatable :: du(:, :, :), dv(:, :, :)
   real(wp), allocatable :: dxT(:, :), dyT(:, :), idxT(:, :), idyT(:, :)
   real(wp), allocatable :: dy_dxBu(:, :), dx_dyBu(:, :), idyCv(:, :), idxCu(:, :), wet_q(:, :)
   real(wp), allocatable :: dy_dxT(:, :), iareaCu(:, :), dx_dyT(:, :), iareaCv(:, :)
   real(wp) :: t0, t1, ms_dc, id, ia, s1, s2
   integer :: i, j, k, rep, nx, ny, nz, nxp, nyp, n_reps, n_warm, iu, ios
   character(len=256) :: ref_path, dump_path

   nxp = iarg(1, NXP_DEF); nyp = iarg(2, NYP_DEF); nz = iarg(3, NZ_DEF)
   n_reps = iarg(4, REPS_DEF); n_warm = iarg(5, WARM_DEF)
   nx = nxp + 2*NGHOST; ny = nyp + 2*NGHOST
   if (nx < 4 .or. ny < 4 .or. nz < 1) then
      write (output_unit, '(a)') 'ERROR: need nx_phys,ny_phys >= 1 and nz >= 1'; stop 1
   end if
   id = 1.0_wp/DX; ia = 1.0_wp/(DX*DY)

   allocate (u_face(nx + 1, ny, nz), v_face(nx, ny + 1, nz), ahx(nx + 1, ny, nz), ahy(nx, ny + 1, nz))
   allocate (du(nx + 1, ny, nz), dv(nx, ny + 1, nz))
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
   ahx = 0.0_wp; ahy = 0.0_wp; du = 0.0_wp; dv = 0.0_wp

   write (output_unit, '(a)') repeat('=', 70)
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  domain: ', nxp, ' x ', nyp, ' x ', nz, &
      ' interior (+', NGHOST, ' ghosts/side)'
   write (output_unit, '(a,i0)') '  cells : ', nx*ny*nz
   write (output_unit, '(3a,i0,a,i0,a)') '  DATA layer: ', DC_DATA_NAME, '   (reps ', n_reps, ', warm ', n_warm, ')'
   write (output_unit, '(a)') repeat('=', 70)

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
   DC_ENTER_CREATE(ahx)
   DC_ENTER_CREATE(ahy)
   DC_ENTER_CREATE(du)
   DC_ENTER_CREATE(dv)

   do rep = 1, n_warm
      call closure()
   end do
   DC_WAIT
   t0 = wall()
   do rep = 1, n_reps
      call closure()
   end do
   DC_WAIT
   t1 = wall()
   ms_dc = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   DC_UPDATE_SELF(du)
   DC_UPDATE_SELF(dv)
   s1 = sum(du); s2 = sum(dv)

   write (output_unit, '(3a,f10.4,a)') '  Smag closure (', DC_DATA_NAME, ')  : ', ms_dc, ' ms/rep'
   write (output_unit, '(a,es14.6,a,es14.6)') '  sanity: sum du ', s1, '  sum dv ', s2
   if (s1 /= s1 .or. (s1 == 0.0_wp .and. s2 == 0.0_wp)) then
      write (output_unit, '(a)') '  sanity           : *** garbage ***'; stop 2
   else
      write (output_unit, '(a)') '  sanity           : OK (finite, non-zero)'
   end if

   call get_environment_variable('DC_DUMP', dump_path, status=ios)
   if (ios == 0 .and. len_trim(dump_path) > 0) then
      ! Hand over the transcendental inputs (u_face/v_face use sin/cos) so a
      ! native C++ driver ADOPTS them instead of recomputing via a different
      ! libm -- otherwise the cross-check measures libm, not the kernel.
      DC_UPDATE_SELF(u_face)
      DC_UPDATE_SELF(v_face)
      open (newunit=iu, file=trim(dump_path), access='stream', form='unformatted', status='replace')
      write (iu) nx, ny, nz; write (iu) du; write (iu) u_face; write (iu) v_face; close (iu)
      write (output_unit, '(3a)') '  wrote ref       : ', trim(dump_path), ' (nx,ny,nz, du, u, v)'
   end if
   call get_environment_variable('DC_REF', ref_path, status=ios)
   if (ios == 0 .and. len_trim(ref_path) > 0) call compare_ref(trim(ref_path))
   write (output_unit, '(a)') repeat('=', 70)

contains

   subroutine closure()
      call hvisc_compute_smag(u_face, v_face, ahx, ahy, dxT, dyT, idxT, idyT, &
                              dy_dxBu, dx_dyBu, idyCv, idxCu, wet_q, C_SMAG, AH_BG, AH_MAX, NS, nx, ny, nz)
      call hvisc_compute_face(u_face, v_face, ahx, ahy, du, dv, &
                              dy_dxT, dx_dyBu, iareaCu, dx_dyT, dy_dxBu, iareaCv, nx, ny, nz)
   end subroutine closure

   subroutine compare_ref(path)
      character(len=*), intent(in) :: path
      real(wp), allocatable :: ref(:, :, :)
      real(wp) :: dmax, rmax, sc
      integer :: rnx, rny, rnz, u, st, nbad
      open (newunit=u, file=path, access='stream', form='unformatted', status='old', iostat=st)
      if (st /= 0) then
         write (output_unit, '(3a)') '  cross-check     : ref ', path, ' not found -- skipped'; return
      end if
      read (u) rnx, rny, rnz
      if (rnx /= nx .or. rny /= ny .or. rnz /= nz) then
         write (output_unit, '(a)') '  cross-check     : ref shape mismatch -- skipped'; close (u); return
      end if
      allocate (ref(nx + 1, ny, nz)); read (u) ref; close (u)
      dmax = 0.0_wp; rmax = 0.0_wp; nbad = 0
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx + 1
               dmax = max(dmax, abs(du(i, j, k) - ref(i, j, k)))
               sc = max(abs(du(i, j, k)), abs(ref(i, j, k)))
               if (sc > 1.0e-30_wp) then
                  rmax = max(rmax, abs(du(i, j, k) - ref(i, j, k))/sc)
                  if (abs(du(i, j, k) - ref(i, j, k))/sc > 1.0e-12_wp) nbad = nbad + 1
               end if
            end do
         end do
      end do
      write (output_unit, '(a,es12.5,a,es12.5)') '  cross-check vs ref: max|diff| ', dmax, '  max rel ', rmax
      if (rmax < 1.0e-12_wp) then
         write (output_unit, '(a)') '  cross-check     : OK (<1e-12 rel -> data layer numerically inert)'
      else
         write (output_unit, '(a,i0,a)') '  cross-check     : *** ', nbad, ' cells >1e-12 rel -- INVESTIGATE ***'
      end if
   end subroutine compare_ref

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
end program dc_main
