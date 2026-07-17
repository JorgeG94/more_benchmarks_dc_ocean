#include "directives.h"
!! SHARED single-binary head-to-head: opt-DC vs opt-CUDA on ONE device truth.
!!
!! This removes the two-harness caveat. dc_ab.F90 timed fused-DC in an nvfortran
!! binary and ab_main.cu timed fused-CUDA in a separate nvcc binary -- two
!! harnesses, two seeds, two devices-under-load. Here BOTH optimized endpoints
!!   A = btstep_nonlinear_closed_fused   (do concurrent, 5 loops/substep)
!!   B = btstep_opt_launch_flat(opt_mode=0)  (CUDA, 5 kernels/substep, fused)
!! run in the SAME binary on the SAME device arrays: the CUDA launcher is handed
!! the do-concurrent allocations verbatim via `!$acc host_data use_device`
!! (btstep_bridge.F90). No copies between them; one truth.
!!
!! Both endpoints EVOLVE state (eta/velocity), so -- like dc_ab -- the driver
!! resets the device state to the IDENTICAL seed between the two timings
!! (re-init on host + DC_UPDATE_DEVICE). Each side: n_warm untimed calls then
!! n_reps timed calls (n_inner substeps each) from that seed. Headline is
!! fused-vs-fused; opt_mode=1 (fused captured in a cudaGraph) is timed too and
!! reported on its own line -- a replayed graph is a CUDA-only lever, not a fair
!! DC comparison, so it never sets the headline ratio.
!!
!! The device DATA layer is OpenACC (this driver only makes sense with -acc;
!! host_data needs device pointers). Built with -DDC_DATA_ACC.
!!
!! Usage: ./cmp_acc [nx_phys] [ny_phys] [n_inner] [nreps] [nwarm]
program cmp_main
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use constants, only: wp
   use bt_state, only: hgrid_t, ocean_metrics_t, coriolis_t, bt_work_t
   use btstep, only: btstep_nonlinear_closed_fused
   use btstep_bridge, only: btstep_opt_cuda
   implicit none

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NINNER_DEF = 24
   integer, parameter :: REPS_DEF = 200, WARM_DEF = 10, NGH = 3
   real(wp), parameter :: DXM = 10000.0_wp, DYM = 10000.0_wp, DT_INNER = 12.0_wp

   type(hgrid_t) :: grid
   type(ocean_metrics_t) :: met
   type(coriolis_t) :: cor
   type(bt_work_t) :: w
   real(wp), allocatable :: force_u(:, :), force_v(:, :)
   real(wp), allocatable :: eta_dc(:, :), eta_cu(:, :)
   real(wp) :: t0, t1, ms_dc, ms_cu, ms_gr, dmax, rmax, sc
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
   allocate (eta_dc(nx, ny), eta_cu(nx, ny))

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
   write (output_unit, '(a)') '  SHARED head-to-head: opt-DC vs opt-CUDA on one device truth (host_data)'
   write (output_unit, '(a,i0,a,i0,a,i0,a)') '  domain: ', nxp, ' x ', nyp, ' interior (+', NGH, ' ghosts) '
   write (output_unit, '(a,i0,a,i0,a,i0)') '  arrays: ', nx, ' x ', ny, ' = ', nx*ny
   write (output_unit, '(a,i0,a,i0,a,i0,a)') '  n_inner: ', n_inner, ' substeps;  ', n_reps, &
      ' timed reps, ', n_warm, ' warm-up'
   write (output_unit, '(3a,f0.2,a,f0.1,a)') '  DATA layer: ', DC_DATA_NAME, '   bebt = ', w%bebt, &
      ';  dt_inner = ', DT_INNER, ' s'
   write (output_unit, '(a)') repeat('=', 72)

   ! ---- map the working set to the device (ONE truth for both endpoints) -----
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

   ! ================= A: opt-DC (fused do concurrent) =======================
   call reseed()
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
   ms_dc = (t1 - t0)*1000.0_wp/real(n_reps, wp)
   DC_UPDATE_SELF(w%bt_eta)
   eta_dc = w%bt_eta

   ! ================= B: opt-CUDA (fused, opt_mode=0) =======================
   call reseed()
   do rep = 1, n_warm
      call btstep_opt_cuda(grid, met, w, cor, force_u, force_v, n_inner, DT_INNER, 0)
   end do
   t0 = wall()
   do rep = 1, n_reps
      call btstep_opt_cuda(grid, met, w, cor, force_u, force_v, n_inner, DT_INNER, 0)
   end do
   t1 = wall()
   ms_cu = (t1 - t0)*1000.0_wp/real(n_reps, wp)
   DC_UPDATE_SELF(w%bt_eta)
   eta_cu = w%bt_eta

   ! ================= C: opt-CUDA (fused-in-cudaGraph, opt_mode=1) ===========
   ! CUDA-only lever -- reported separately, never the headline ratio. Graph is
   ! instantiated on the first (warm-up) call, outside the timed window.
   call reseed()
   do rep = 1, n_warm
      call btstep_opt_cuda(grid, met, w, cor, force_u, force_v, n_inner, DT_INNER, 1)
   end do
   t0 = wall()
   do rep = 1, n_reps
      call btstep_opt_cuda(grid, met, w, cor, force_u, force_v, n_inner, DT_INNER, 1)
   end do
   t1 = wall()
   ms_gr = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- agreement: opt-DC vs opt-CUDA(fused), both from the SAME seed --------
   ! FIELD-relative, the documented bar for bt_eta (divergence-driven; interior
   ! cells ~0, so a POINTWISE relative would divide the FMA-level abs diff by a
   ! near-zero cell and explode -- the same trap OPTIMIZATION.md / redi_bench /
   ! dc_ab call out). Judge max|d eta| against the field magnitude max|eta|.
   ! opt-DC is nvfortran codegen, opt-CUDA is nvcc: FMA contraction may differ
   ! ~1e-15 absolute. >1e-12 field-relative would be a real port/wiring bug.
   dmax = 0.0_wp
   do j = 1, ny
      do i = 1, nx
         dmax = max(dmax, abs(eta_cu(i, j) - eta_dc(i, j)))
      end do
   end do
   sc = max(maxval(abs(eta_dc)), 1.0e-30_wp)
   rmax = dmax/sc

   write (output_unit, '(a,es14.6,a,es14.6)') '  bt_eta range (dc): min ', minval(eta_dc), &
      '  max ', maxval(eta_dc)
   if (maxval(abs(eta_dc)) == 0.0_wp .or. maxval(abs(eta_cu)) == 0.0_wp .or. &
       eta_dc(nx/2, ny/2) /= eta_dc(nx/2, ny/2)) then
      write (output_unit, '(a)') '  sanity           : *** garbage (NaN or all-zero) ***'; stop 2
   end if
   write (output_unit, '(a,es12.5,a,es12.5)') '  agreement        : max|d eta| ', dmax, &
      '  field-rel ', rmax
   if (rmax < 1.0e-12_wp) then
      write (output_unit, '(a)') '  verdict          : OK (opt-DC == opt-CUDA, <1e-12 field-rel)'
   else
      write (output_unit, '(a)') '  verdict          : *** DIFF -- flat wrapper wired a field wrong? ***'
   end if
   write (output_unit, '(a)') repeat('-', 72)
   write (output_unit, '(a,f10.4,a)') '  opt-DC   (fused do concurrent) : ', ms_dc, ' ms/rep'
   write (output_unit, '(a,f10.4,a)') '  opt-CUDA (fused, 5 kern)       : ', ms_cu, ' ms/rep'
   write (output_unit, '(a,f10.4,a)') '  opt-CUDA (fused-in-cudaGraph)  : ', ms_gr, ' ms/rep  (CUDA-only)'
   write (output_unit, '(a)') repeat('-', 72)
   if (ms_cu < ms_dc) then
      write (output_unit, '(a,f7.3,a)') '  headline: opt-CUDA faster by ', ms_dc/ms_cu, 'x  (fused vs fused)'
   else
      write (output_unit, '(a,f7.3,a)') '  headline: opt-DC faster by ', ms_cu/ms_dc, 'x  (fused vs fused)'
   end if
   write (output_unit, '(a)') repeat('=', 72)

contains

   subroutine reseed()
      !! Reset the evolving state to the IDENTICAL seed on the host, then push it
      !! to the device. The static fields (met/cor/force/H_ref/rem_*) are not
      !! re-pushed -- they never change during a substep sweep.
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
   end subroutine reseed

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

end program cmp_main
