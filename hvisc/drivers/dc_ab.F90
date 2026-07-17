#include "directives.h"
!! DC A/B driver for the ocean horizontal-viscosity Smagorinsky closure.
!!
!! Same-binary, same-libm comparison of the FAITHFUL two-pass closure
!!   (hvisc_compute_smag -> ah_face, then hvisc_compute_face -> du/dv)
!! against the FUSED single-pass closure
!!   (hvisc_compute_fused -> du/dv, ah_face never materialised).
!!
!! COMPUTE is bare `do concurrent`; the device data layer is chosen by
!! directives.h:  -DDC_DATA_ACC (OpenACC GPU) / -DDC_DATA_OMP (OpenMP target) /
!! neither (CPU, -stdpar=multicore). No CUDA in this binary.
!!
!! A: warm + timed faithful -> snapshot du0/dv0.  B: warm + timed fused -> du/dv.
!! Then compare both fields field-relative (max|diff|/max|field|); bar 1e-12.
!! Both sides share this binary's libm, so du/v are compared directly in memory
!! -- no reference dump, no cross-language libm trap.
!! Usage: ./dc_ab [nx_phys] [ny_phys] [nz] [nreps] [nwarm]
program dc_ab
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use constants, only: wp
   use ocean_horizontal_viscosity, only: hvisc_compute_smag, hvisc_compute_face, hvisc_compute_fused
   implicit none

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30
   integer, parameter :: REPS_DEF = 200, WARM_DEF = 10, NGHOST = 3
   real(wp), parameter :: DX = 1000.0_wp, DY = 1000.0_wp
   real(wp), parameter :: C_SMAG = 0.2_wp, AH_BG = 1.0_wp, AH_MAX = 1.0e5_wp, NS = 0.0_wp

   real(wp), allocatable :: u_face(:, :, :), v_face(:, :, :), ahx(:, :, :), ahy(:, :, :)
   real(wp), allocatable :: du(:, :, :), dv(:, :, :), du0(:, :, :), dv0(:, :, :)
   real(wp), allocatable :: dxT(:, :), dyT(:, :), idxT(:, :), idyT(:, :)
   real(wp), allocatable :: dy_dxBu(:, :), dx_dyBu(:, :), idyCv(:, :), idxCu(:, :), wet_q(:, :)
   real(wp), allocatable :: dy_dxT(:, :), iareaCu(:, :), dx_dyT(:, :), iareaCv(:, :)
   real(wp) :: t0, t1, ms_fa, ms_fu, id, ia, s1, s2, dmax, rmax, sc, df
   integer :: i, j, k, rep, nx, ny, nz, nxp, nyp, n_reps, n_warm, nbad

   nxp = iarg(1, NXP_DEF); nyp = iarg(2, NYP_DEF); nz = iarg(3, NZ_DEF)
   n_reps = iarg(4, REPS_DEF); n_warm = iarg(5, WARM_DEF)
   nx = nxp + 2*NGHOST; ny = nyp + 2*NGHOST
   if (nx < 4 .or. ny < 4 .or. nz < 1) then
      write (output_unit, '(a)') 'ERROR: need nx_phys,ny_phys >= 1 and nz >= 1'; stop 1
   end if
   id = 1.0_wp/DX; ia = 1.0_wp/(DX*DY)

   allocate (u_face(nx + 1, ny, nz), v_face(nx, ny + 1, nz), ahx(nx + 1, ny, nz), ahy(nx, ny + 1, nz))
   allocate (du(nx + 1, ny, nz), dv(nx, ny + 1, nz), du0(nx + 1, ny, nz), dv0(nx, ny + 1, nz))
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
   ahx = 0.0_wp; ahy = 0.0_wp
   du = 0.0_wp; dv = 0.0_wp; du0 = 0.0_wp; dv0 = 0.0_wp

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
   DC_ENTER_CREATE(du0)
   DC_ENTER_CREATE(dv0)

   ! ---- A: faithful two-pass closure (smag -> ah_face -> face) --------------
   do rep = 1, n_warm
      call faithful()
   end do
   DC_WAIT
   t0 = wall()
   do rep = 1, n_reps
      call faithful()
   end do
   DC_WAIT
   t1 = wall()
   ms_fa = (t1 - t0)*1000.0_wp/real(n_reps, wp)
   ! snapshot faithful result on the device (du -> du0, dv -> dv0)
   DC_UPDATE_SELF(du)
   DC_UPDATE_SELF(dv)
   du0 = du
   dv0 = dv

   ! ---- B: fused single-pass closure (ah_face never materialised) ----------
   do rep = 1, n_warm
      call fused()
   end do
   DC_WAIT
   t0 = wall()
   do rep = 1, n_reps
      call fused()
   end do
   DC_WAIT
   t1 = wall()
   ms_fu = (t1 - t0)*1000.0_wp/real(n_reps, wp)
   DC_UPDATE_SELF(du)
   DC_UPDATE_SELF(dv)

   s1 = sum(du); s2 = sum(dv)

   ! ---- compare both fields (du and dv) field-relative ---------------------
   dmax = 0.0_wp; rmax = 0.0_wp; nbad = 0
   do k = 1, nz
      do j = 1, ny
         do i = 1, nx + 1
            df = abs(du(i, j, k) - du0(i, j, k))
            dmax = max(dmax, df)
            sc = max(abs(du(i, j, k)), abs(du0(i, j, k)))
            if (sc > 1.0e-30_wp) then
               rmax = max(rmax, df/sc)
               if (df/sc > 1.0e-12_wp) nbad = nbad + 1
            end if
         end do
      end do
   end do
   do k = 1, nz
      do j = 1, ny + 1
         do i = 1, nx
            df = abs(dv(i, j, k) - dv0(i, j, k))
            dmax = max(dmax, df)
            sc = max(abs(dv(i, j, k)), abs(dv0(i, j, k)))
            if (sc > 1.0e-30_wp) then
               rmax = max(rmax, df/sc)
               if (df/sc > 1.0e-12_wp) nbad = nbad + 1
            end if
         end do
      end do
   end do

   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  grid ', nxp, 'x', nyp, 'x', nz, ' (', nx*ny*nz, ' cells)'
   write (output_unit, '(3a)') '  data layer       : ', DC_DATA_NAME, ''
   write (output_unit, '(a,es14.6,a,es14.6)') '  sanity: sum du ', s1, '  sum dv ', s2
   if (s1 /= s1 .or. (s1 == 0.0_wp .and. s2 == 0.0_wp)) then
      write (output_unit, '(a)') '  sanity           : *** garbage ***'; stop 2
   else
      write (output_unit, '(a)') '  sanity           : OK (finite, non-zero)'
   end if
   write (output_unit, '(a,es12.5,a,es12.5)') '  correctness      : max|diff| ', dmax, '  max rel ', rmax
   if (rmax < 1.0e-12_wp) then
      write (output_unit, '(a)') '  verdict          : OK (fused == faithful, <1e-12 rel)'
   else
      write (output_unit, '(a,i0,a)') '  verdict          : *** ', nbad, ' cells >1e-12 rel -- fused kernel bug ***'
   end if
   write (output_unit, '(a,f10.4,a)') '  faithful DC      : ', ms_fa, ' ms/rep'
   write (output_unit, '(a,f10.4,a,f7.3,a)') '  fused DC         : ', ms_fu, ' ms/rep   -> ', ms_fa/ms_fu, 'x'
   write (output_unit, '(a)') repeat('=', 70)

contains

   subroutine faithful()
      call hvisc_compute_smag(u_face, v_face, ahx, ahy, dxT, dyT, idxT, idyT, &
                              dy_dxBu, dx_dyBu, idyCv, idxCu, wet_q, C_SMAG, AH_BG, AH_MAX, NS, nx, ny, nz)
      call hvisc_compute_face(u_face, v_face, ahx, ahy, du, dv, &
                              dy_dxT, dx_dyBu, iareaCu, dx_dyT, dy_dxBu, iareaCv, nx, ny, nz)
   end subroutine faithful

   subroutine fused()
      call hvisc_compute_fused(u_face, v_face, du, dv, dxT, dyT, idxT, idyT, &
                               dy_dxBu, dx_dyBu, idyCv, idxCu, wet_q, dy_dxT, iareaCu, &
                               dx_dyT, iareaCv, C_SMAG, AH_BG, AH_MAX, NS, nx, ny, nz)
   end subroutine fused

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
end program dc_ab
