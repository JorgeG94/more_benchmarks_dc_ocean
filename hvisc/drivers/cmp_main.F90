#include "directives.h"
!! SHARED single-binary head-to-head for the hvisc Smagorinsky closure:
!! opt-DC (hvisc_compute_fused, `do concurrent`) vs opt-CUDA (hvisc_opt_launch)
!! on the SAME device arrays via `!$acc host_data use_device`.
!!
!! This removes the two-harness caveat of dc_ab (DC binary) vs ab (CUDA binary):
!! both endpoints run in ONE binary, on ONE set of device allocations, timed
!! over identical reps with the same host clock. The kernel is stateless per rep
!! (reads u/v + geometry, writes du/dv), so between the two timed runs we only
!! zero the outputs -- no reseed.
!!
!! A: warm + timed opt-DC (fused DC) -> snapshot du0/dv0.
!! B: zero du/dv, warm + timed opt-CUDA -> du/dv.
!! Compare du/dv (opt-CUDA) vs du0/dv0 (opt-DC) field-relative; bar 1e-12.
!! Usage: ./cmp_acc [nx_phys] [ny_phys] [nz] [nreps] [nwarm]
program cmp_main
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use constants, only: wp
   use ocean_horizontal_viscosity, only: hvisc_compute_fused
   use hvisc_bridge, only: hvisc_opt_step
   implicit none

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30
   integer, parameter :: REPS_DEF = 200, WARM_DEF = 10, NGHOST = 3
   real(wp), parameter :: DX = 1000.0_wp, DY = 1000.0_wp
   real(wp), parameter :: C_SMAG = 0.2_wp, AH_BG = 1.0_wp, AH_MAX = 1.0e5_wp, NS = 0.0_wp

   real(wp), allocatable :: u_face(:, :, :), v_face(:, :, :)
   real(wp), allocatable :: du(:, :, :), dv(:, :, :), du0(:, :, :), dv0(:, :, :)
   real(wp), allocatable :: dxT(:, :), dyT(:, :), idxT(:, :), idyT(:, :)
   real(wp), allocatable :: dy_dxBu(:, :), dx_dyBu(:, :), idyCv(:, :), idxCu(:, :), wet_q(:, :)
   real(wp), allocatable :: dy_dxT(:, :), iareaCu(:, :), dx_dyT(:, :), iareaCv(:, :)
   real(wp) :: t0, t1, ms_dc, ms_cu, id, ia, s1, s2, s1c, s2c, dmax, rmax, sc, df
   integer :: i, j, k, rep, nx, ny, nz, nxp, nyp, n_reps, n_warm, nbad

   nxp = iarg(1, NXP_DEF); nyp = iarg(2, NYP_DEF); nz = iarg(3, NZ_DEF)
   n_reps = iarg(4, REPS_DEF); n_warm = iarg(5, WARM_DEF)
   nx = nxp + 2*NGHOST; ny = nyp + 2*NGHOST
   if (nx < 4 .or. ny < 4 .or. nz < 1) then
      write (output_unit, '(a)') 'ERROR: need nx_phys,ny_phys >= 1 and nz >= 1'; stop 1
   end if
   id = 1.0_wp/DX; ia = 1.0_wp/(DX*DY)

   allocate (u_face(nx + 1, ny, nz), v_face(nx, ny + 1, nz))
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
   du = 0.0_wp; dv = 0.0_wp; du0 = 0.0_wp; dv0 = 0.0_wp

   write (output_unit, '(a)') repeat('=', 70)
   write (output_unit, '(a)') '  HEAD-TO-HEAD: opt-CUDA vs opt-DC on ONE set of device arrays'
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
   DC_ENTER_CREATE(du)
   DC_ENTER_CREATE(dv)

   ! ---- A: opt-DC (fused single-pass do concurrent) ------------------------
   do rep = 1, n_warm
      call opt_dc()
   end do
   DC_WAIT
   t0 = wall()
   do rep = 1, n_reps
      call opt_dc()
   end do
   DC_WAIT
   t1 = wall()
   ms_dc = (t1 - t0)*1000.0_wp/real(n_reps, wp)
   ! snapshot opt-DC result to host (du -> du0, dv -> dv0). du0/dv0 are host-only;
   ! opt-CUDA overwrites every device du/dv cell (interior computed, walls -> 0),
   ! so no device reset is needed between the two timed runs.
   DC_UPDATE_SELF(du)
   DC_UPDATE_SELF(dv)
   du0 = du
   dv0 = dv
   s1 = sum(du0); s2 = sum(dv0)

   ! ---- B: opt-CUDA (fused single-launch, same device arrays) --------------
   do rep = 1, n_warm
      call opt_cuda()
   end do
   DC_WAIT
   t0 = wall()
   do rep = 1, n_reps
      call opt_cuda()
   end do
   DC_WAIT
   t1 = wall()
   ms_cu = (t1 - t0)*1000.0_wp/real(n_reps, wp)
   DC_UPDATE_SELF(du)
   DC_UPDATE_SELF(dv)
   s1c = sum(du); s2c = sum(dv)

   ! ---- compare du/dv (opt-CUDA) vs du0/dv0 (opt-DC) field-relative --------
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
   write (output_unit, '(a,es14.6,a,es14.6)') '  opt-DC   : sum du ', s1, '  sum dv ', s2
   write (output_unit, '(a,es14.6,a,es14.6)') '  opt-CUDA : sum du ', s1c, '  sum dv ', s2c
   if (s1 /= s1 .or. s1c /= s1c .or. &
       (s1 == 0.0_wp .and. s2 == 0.0_wp) .or. (s1c == 0.0_wp .and. s2c == 0.0_wp)) then
      write (output_unit, '(a)') '  sanity           : *** garbage / zero output ***'; stop 2
   else
      write (output_unit, '(a)') '  sanity           : OK (both finite, non-zero)'
   end if
   write (output_unit, '(a,es12.5,a,es12.5)') '  agreement        : max|diff| ', dmax, '  max rel ', rmax
   if (rmax < 1.0e-12_wp) then
      write (output_unit, '(a)') '  verdict          : OK (opt-CUDA == opt-DC, <1e-12 rel)'
   else
      write (output_unit, '(a,i0,a)') '  verdict          : *** ', nbad, ' cells >1e-12 rel -- layout/kernel mismatch ***'
   end if
   write (output_unit, '(a)') repeat('-', 70)
   write (output_unit, '(a,f10.4,a)') '  opt-DC           : ', ms_dc, ' ms/rep'
   write (output_unit, '(a,f10.4,a)') '  opt-CUDA         : ', ms_cu, ' ms/rep'
   if (ms_dc < ms_cu) then
      write (output_unit, '(a,f7.3,a)') '  ratio            : opt-DC faster by ', ms_cu/ms_dc, 'x'
   else if (ms_cu < ms_dc) then
      write (output_unit, '(a,f7.3,a)') '  ratio            : opt-CUDA faster by ', ms_dc/ms_cu, 'x'
   else
      write (output_unit, '(a)') '  ratio            : tie'
   end if
   write (output_unit, '(a)') repeat('=', 70)

contains

   subroutine opt_dc()
      call hvisc_compute_fused(u_face, v_face, du, dv, dxT, dyT, idxT, idyT, &
                               dy_dxBu, dx_dyBu, idyCv, idxCu, wet_q, dy_dxT, iareaCu, &
                               dx_dyT, iareaCv, C_SMAG, AH_BG, AH_MAX, NS, nx, ny, nz)
   end subroutine opt_dc

   subroutine opt_cuda()
      call hvisc_opt_step(u_face, v_face, du, dv, dxT, dyT, idxT, idyT, &
                          dy_dxBu, dx_dyBu, idyCv, idxCu, wet_q, dy_dxT, iareaCu, &
                          dx_dyT, iareaCv, C_SMAG, AH_BG, AH_MAX, NS, nx, ny, nz)
   end subroutine opt_cuda

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
end program cmp_main
