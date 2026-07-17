#include "directives.h"
!! DC A/B driver for the production CLOSED-BASIN barotropic substep.
!!
!! A = btstep_nonlinear_closed        (faithful: 11 do-concurrent loops/substep)
!! B = btstep_nonlinear_closed_fused  (portable best-practice: 5 loops/substep,
!!     the CUDA fusion win minus the non-portable cudaGraph)
!!
!! Both launchers evolve state, so unlike the stateless continuity A/B this
!! driver RESETS the device state to the identical seed between A and B
!! (re-init on host + DC_UPDATE_DEVICE). Each side runs n_warm warm-up calls
!! then n_reps timed calls (n_inner substeps each) from that seed; the fused
!! bt_eta must match the faithful bt_eta bit-for-bit (max rel < 1e-12).
!!
!! The device DATA layer is chosen ENTIRELY by directives.h at compile time:
!!   -DDC_DATA_ACC -> OpenACC (GPU, measured path)
!!   -DDC_DATA_OMP -> OpenMP target (GPU)
!!   (neither)     -> host: bare DC on the CPU (portability check). No CUDA.
!!
!! Usage: ./dc_ab [nx_phys] [ny_phys] [n_inner] [nreps] [nwarm]
program dc_ab
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use constants, only: wp
   use bt_state, only: hgrid_t, ocean_metrics_t, coriolis_t, bt_work_t
   use btstep, only: btstep_nonlinear_closed, btstep_nonlinear_closed_fused
   implicit none

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NINNER_DEF = 24
   integer, parameter :: REPS_DEF = 200, WARM_DEF = 10, NGH = 3
   real(wp), parameter :: DXM = 10000.0_wp, DYM = 10000.0_wp, DT_INNER = 12.0_wp

   type(hgrid_t) :: grid
   type(ocean_metrics_t) :: met
   type(coriolis_t) :: cor
   type(bt_work_t) :: w
   real(wp), allocatable :: force_u(:, :), force_v(:, :)
   real(wp), allocatable :: eta_ref(:, :)
   real(wp) :: t0, t1, ms_dc, ms_fu, dmax, rmax, sc
   integer :: i, j, rep, nx, ny, nxp, nyp, n_inner, n_reps, n_warm

   nxp = iarg(1, NXP_DEF); nyp = iarg(2, NYP_DEF); n_inner = iarg(3, NINNER_DEF)
   n_reps = iarg(4, REPS_DEF); n_warm = iarg(5, WARM_DEF)
   nx = nxp + 2*NGH; ny = nyp + 2*NGH
   if (nx < 6 .or. ny < 6 .or. n_inner < 1) then
      write (output_unit, '(a)') 'ERROR: need nx_phys,ny_phys >= 1 and n_inner >= 1'; stop 1
   end if
   grid%nx_total = nx; grid%ny_total = ny
   grid%nx_phys = nxp; grid%ny_phys = nyp; grid%nghost = NGH

   call alloc_state(w)
   allocate (met%dy_cu(nx+1, ny), met%dx_cv(nx, ny+1), met%iareaT(nx, ny))
   allocate (met%areaCu(nx+1, ny), met%areaCv(nx, ny+1))
   allocate (met%dxCu(nx+1, ny), met%dyCv(nx, ny+1))
   allocate (met%idxCu(nx+1, ny), met%idyCv(nx, ny+1), met%iareaBu(nx+1, ny+1))
   allocate (cor%f_corner(nx+1, ny+1))
   allocate (force_u(nx+1, ny), force_v(nx, ny+1))
   allocate (eta_ref(nx, ny))

   met%dy_cu = DYM; met%dx_cv = DXM
   met%iareaT = 1.0_wp/(DXM*DYM)
   met%areaCu = DXM*DYM; met%areaCv = DXM*DYM
   met%dxCu = DXM; met%dyCv = DYM
   met%idxCu = 1.0_wp/DXM; met%idyCv = 1.0_wp/DYM
   met%iareaBu = 1.0_wp/(DXM*DYM)
   do j = 1, ny + 1
      do i = 1, nx + 1
         cor%f_corner(i, j) = -1.0e-4_wp + 2.0e-11_wp*real(j - ny/2, wp)*DYM
      end do
   end do
   do j = 1, ny
      do i = 1, nx + 1
         force_u(i, j) = 1.0e-6_wp*sin(3.14159_wp*real(j, wp)/real(ny, wp))
      end do
   end do
   force_v = 0.0_wp

   call init_state(w)

   write (output_unit, '(a)') repeat('=', 72)
   write (output_unit, '(a,i0,a,i0,a,i0,a)') '  domain: ', nxp, ' x ', nyp, ' interior (+', NGH, ' ghosts) '
   write (output_unit, '(a,i0,a,i0,a,i0)') '  arrays: ', nx, ' x ', ny, ' = ', nx*ny
   write (output_unit, '(a,i0,a,i0,a,i0,a)') '  n_inner: ', n_inner, ' substeps;  ', n_reps, &
      ' timed reps, ', n_warm, ' warm-up'
   write (output_unit, '(3a,f0.2,a,f0.1,a)') '  DATA layer: ', DC_DATA_NAME, '   bebt = ', w%bebt, &
      ';  dt_inner = ', DT_INNER, ' s'
   write (output_unit, '(a)') repeat('=', 72)

   ! ---- map the working set (no-ops when the DATA layer is 'host') ----------
   DC_ENTER_IN(met)
   DC_ENTER_IN(cor)
   DC_ENTER_IN(w)
   DC_ENTER_IN(w%bt_eta)
   DC_ENTER_IN(w%bt_eta_new)
   DC_ENTER_IN(w%bt_H_ref)
   DC_ENTER_IN(w%bt_ubt)
   DC_ENTER_IN(w%bt_vbt)
   DC_ENTER_IN(w%bt_ubt_prev)
   DC_ENTER_IN(w%bt_vbt_prev)
   DC_ENTER_IN(w%bt_rem_u)
   DC_ENTER_IN(w%bt_rem_v)
   DC_ENTER_IN(w%bt_zeta_corner)
   DC_ENTER_IN(w%bt_ke_centre)
   DC_ENTER_IN(w%ubt_sum)
   DC_ENTER_IN(w%vbt_sum)
   DC_ENTER_IN(w%eta_sum)
   DC_ENTER_IN(w%uhbt_sum)
   DC_ENTER_IN(w%vhbt_sum)
   DC_ENTER_IN(met%dy_cu)
   DC_ENTER_IN(met%dx_cv)
   DC_ENTER_IN(met%iareaT)
   DC_ENTER_IN(met%areaCu)
   DC_ENTER_IN(met%areaCv)
   DC_ENTER_IN(met%dxCu)
   DC_ENTER_IN(met%dyCv)
   DC_ENTER_IN(met%idxCu)
   DC_ENTER_IN(met%idyCv)
   DC_ENTER_IN(met%iareaBu)
   DC_ENTER_IN(cor%f_corner)
   DC_ENTER_IN(force_u)
   DC_ENTER_IN(force_v)

   ! ---- A: faithful do concurrent (11 loops/substep) -----------------------
   do rep = 1, n_warm
      call btstep_nonlinear_closed(grid, met, w, cor, force_u, force_v, n_inner, DT_INNER)
   end do
   DC_WAIT
   t0 = wall()
   do rep = 1, n_reps
      call btstep_nonlinear_closed(grid, met, w, cor, force_u, force_v, n_inner, DT_INNER)
   end do
   DC_WAIT
   t1 = wall()
   ms_dc = (t1 - t0)*1000.0_wp/real(n_reps, wp)
   DC_UPDATE_SELF(w%bt_eta)
   eta_ref = w%bt_eta

   ! ---- reset the evolving state to the IDENTICAL seed, then push to device -
   call init_state(w)
   DC_UPDATE_DEVICE(w%bt_eta)
   DC_UPDATE_DEVICE(w%bt_eta_new)
   DC_UPDATE_DEVICE(w%bt_ubt)
   DC_UPDATE_DEVICE(w%bt_vbt)
   DC_UPDATE_DEVICE(w%bt_ubt_prev)
   DC_UPDATE_DEVICE(w%bt_vbt_prev)
   DC_UPDATE_DEVICE(w%bt_zeta_corner)
   DC_UPDATE_DEVICE(w%bt_ke_centre)
   DC_UPDATE_DEVICE(w%ubt_sum)
   DC_UPDATE_DEVICE(w%vbt_sum)
   DC_UPDATE_DEVICE(w%eta_sum)
   DC_UPDATE_DEVICE(w%uhbt_sum)
   DC_UPDATE_DEVICE(w%vhbt_sum)
   DC_WAIT

   ! ---- B: fused do concurrent (5 loops/substep) ---------------------------
   do rep = 1, n_warm
      call btstep_nonlinear_closed_fused(grid, met, w, cor, force_u, force_v, n_inner, DT_INNER)
   end do
   DC_WAIT
   t0 = wall()
   do rep = 1, n_reps
      call btstep_nonlinear_closed_fused(grid, met, w, cor, force_u, force_v, n_inner, DT_INNER)
   end do
   DC_WAIT
   t1 = wall()
   ms_fu = (t1 - t0)*1000.0_wp/real(n_reps, wp)
   DC_UPDATE_SELF(w%bt_eta)

   ! ---- compare bt_eta: fused vs faithful, both from the same seed ----------
   dmax = 0.0_wp; rmax = 0.0_wp
   do j = 1, ny
      do i = 1, nx
         dmax = max(dmax, abs(w%bt_eta(i, j) - eta_ref(i, j)))
         sc = max(abs(w%bt_eta(i, j)), abs(eta_ref(i, j)))
         if (sc > 1.0e-30_wp) rmax = max(rmax, abs(w%bt_eta(i, j) - eta_ref(i, j))/sc)
      end do
   end do

   write (output_unit, '(a,i0,a,i0,a,i0,a)') '  grid ', nxp, 'x', nyp, ' (', nx*ny, ' cells)'
   write (output_unit, '(3a)') '  data layer       : ', DC_DATA_NAME, ''
   write (output_unit, '(a,es14.6)') '  min bt_eta (fu)  : ', minval(w%bt_eta)
   write (output_unit, '(a,es14.6)') '  max bt_eta (fu)  : ', maxval(w%bt_eta)
   if (maxval(abs(w%bt_eta)) == 0.0_wp .or. w%bt_eta(nx/2, ny/2) /= w%bt_eta(nx/2, ny/2)) then
      write (output_unit, '(a)') '  sanity           : *** garbage (NaN or all-zero) ***'; stop 2
   end if
   write (output_unit, '(a,es12.5,a,es12.5)') '  correctness      : max|d eta| ', dmax, '  max rel ', rmax
   if (rmax < 1.0e-12_wp) then
      write (output_unit, '(a)') '  verdict          : OK (fused == faithful, <1e-12 rel)'
   else
      write (output_unit, '(a)') '  verdict          : *** DIFF -- fused kernel bug ***'
   end if
   write (output_unit, '(a,f10.4,a)') '  faithful DC      : ', ms_dc, ' ms/rep'
   write (output_unit, '(a,f10.4,a,f7.3,a)') '  fused DC         : ', ms_fu, ' ms/rep   -> ', ms_dc/ms_fu, 'x'
   write (output_unit, '(a)') repeat('=', 72)

contains

   subroutine alloc_state(s)
      type(bt_work_t), intent(inout) :: s
      allocate (s%bt_eta(nx, ny), s%bt_eta_new(nx, ny), s%bt_H_ref(nx, ny))
      allocate (s%bt_ubt(nx+1, ny), s%bt_vbt(nx, ny+1))
      allocate (s%bt_ubt_prev(nx+1, ny), s%bt_vbt_prev(nx, ny+1))
      allocate (s%bt_rem_u(nx+1, ny), s%bt_rem_v(nx, ny+1))
      allocate (s%bt_zeta_corner(nx+1, ny+1), s%bt_ke_centre(nx, ny))
      allocate (s%ubt_sum(nx+1, ny), s%vbt_sum(nx, ny+1), s%eta_sum(nx, ny))
      allocate (s%uhbt_sum(nx+1, ny), s%vhbt_sum(nx, ny+1))
   end subroutine

   subroutine init_state(s)
      type(bt_work_t), intent(inout) :: s
      integer :: ii, jj
      s%g_bt = 9.81_wp
      s%bebt = 0.2_wp
      do jj = 1, ny
         do ii = 1, nx
            s%bt_H_ref(ii, jj) = 4000.0_wp
            s%bt_eta(ii, jj) = 0.5_wp*exp(-((real(ii - nx/2, wp)/real(max(nx/8, 1), wp))**2 &
                                            + (real(jj - ny/2, wp)/real(max(ny/8, 1), wp))**2))
         end do
      end do
      s%bt_eta_new = 0.0_wp
      s%bt_ubt = 0.0_wp; s%bt_vbt = 0.0_wp
      s%bt_ubt_prev = 0.0_wp; s%bt_vbt_prev = 0.0_wp
      s%bt_rem_u = 1.0_wp; s%bt_rem_v = 1.0_wp
      s%bt_zeta_corner = 0.0_wp; s%bt_ke_centre = 0.0_wp
      s%ubt_sum = 0.0_wp; s%vbt_sum = 0.0_wp; s%eta_sum = 0.0_wp
      s%uhbt_sum = 0.0_wp; s%vhbt_sum = 0.0_wp
   end subroutine

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
