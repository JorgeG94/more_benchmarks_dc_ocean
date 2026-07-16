#include "directives.h"
!! DC-only driver for the production CLOSED-BASIN barotropic substep.
!!
!! COMPUTE is `btstep_nonlinear_closed` -- bare `do concurrent` loops wrapped in
!! the production `!$acc kernels async(1)` / `!$acc wait(1)` idiom (n_inner=24
!! substeps per timed call). The device DATA layer is chosen ENTIRELY by
!! directives.h at compile time:
!!   -DDC_DATA_ACC  -> OpenACC  (nvfortran -acc=gpu -stdpar=gpu)   GPU
!!   -DDC_DATA_OMP  -> OpenMP target                               GPU (AMD/Intel too)
!!   (neither)      -> host: bare DC on the CPU (-stdpar=multicore/serial)
!! There is NO CUDA and NO nvcc in this binary. The `!$acc kernels async(1)`
!! lines inside btstep.F90 are honoured only under -acc (DATA=acc, the measured
!! path); under -mp / host they are inert comments and the do-concurrent loops
!! run blocking. Correctness is identical either way -- all loops in a substep
!! sit on the same async queue, so async merely overlaps launches, it never
!! reorders dependent work.
!!
!! Cross-check (proves the macro'd data layer did not change the numbers):
!!   DC_DUMP=file  writes nx,ny + bt_eta  (a reference)
!!   DC_REF=file   reads that reference and reports max|d eta| vs this run
!! Run once with DC_DATA_ACC (GPU) dumping a ref, then again on the CPU / OpenMP
!! reading it: agreement to FMA level means the data-layer swap is inert.
!!
!! Usage: ./dc_main [nx_phys] [ny_phys] [n_inner] [nreps] [nwarm]
program dc_main
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use constants, only: wp
   use bt_state, only: hgrid_t, ocean_metrics_t, coriolis_t, bt_work_t
   use btstep, only: btstep_nonlinear_closed
   implicit none

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NINNER_DEF = 24
   integer, parameter :: REPS_DEF = 200, WARM_DEF = 10, NGH = 3
   real(wp), parameter :: DXM = 10000.0_wp, DYM = 10000.0_wp, DT_INNER = 12.0_wp

   type(hgrid_t) :: grid
   type(ocean_metrics_t) :: met
   type(coriolis_t) :: cor
   type(bt_work_t) :: w
   real(wp), allocatable :: force_u(:, :), force_v(:, :)
   real(wp) :: t0, t1, ms_dc, eta_min, eta_max, eta_sum
   integer :: i, j, rep, nx, ny, nxp, nyp, n_inner, n_reps, n_warm, iu, ios
   character(len=256) :: ref_path, dump_path

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

   ! uniform-Cartesian metrics (what the arithmetic assumes on a flat grid)
   met%dy_cu = DYM; met%dx_cv = DXM
   met%iareaT = 1.0_wp/(DXM*DYM)
   met%areaCu = DXM*DYM; met%areaCv = DXM*DYM
   met%dxCu = DXM; met%dyCv = DYM
   met%idxCu = 1.0_wp/DXM; met%idyCv = 1.0_wp/DYM
   met%iareaBu = 1.0_wp/(DXM*DYM)
   ! beta-plane f, and a wind-like forcing so the substep does real work
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
   ! DEEP COPY: parent derived type first, then each allocatable payload, so the
   ! components attach to the device struct. Mirrors btstep_bench.F90's manual
   ! deep copy (drops the wc/wp_/wf CUDA-comparison states -- DC-only here).
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

   ! ---- do concurrent, production verbatim (n_inner substeps per call) -------
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

   eta_min = minval(w%bt_eta)
   eta_max = maxval(w%bt_eta)
   eta_sum = sum(w%bt_eta)

   write (output_unit, '(3a,f10.4,a)') '  do concurrent (', DC_DATA_NAME, ')  : ', ms_dc, ' ms/call'
   write (output_unit, '(a,es14.6)') '  min bt_eta       : ', eta_min
   write (output_unit, '(a,es14.6)') '  max bt_eta       : ', eta_max
   write (output_unit, '(a,es14.6)') '  sum bt_eta       : ', eta_sum
   if (eta_min /= eta_min .or. (eta_min == 0.0_wp .and. eta_max == 0.0_wp)) then
      write (output_unit, '(a)') '  sanity           : *** garbage (NaN or all-zero) ***'; stop 2
   else
      write (output_unit, '(a)') '  sanity           : OK (finite, non-zero)'
   end if

   ! ---- optional reference dump / cross-check ------------------------------
   call get_environment_variable('DC_DUMP', dump_path, status=ios)
   if (ios == 0 .and. len_trim(dump_path) > 0) then
      open (newunit=iu, file=trim(dump_path), access='stream', form='unformatted', status='replace')
      write (iu) nx, ny
      write (iu) w%bt_eta
      close (iu)
      write (output_unit, '(3a)') '  wrote ref       : ', trim(dump_path), ' (nx,ny, bt_eta)'
   end if

   call get_environment_variable('DC_REF', ref_path, status=ios)
   if (ios == 0 .and. len_trim(ref_path) > 0) call compare_ref(trim(ref_path))

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
      s%bebt = 0.2_wp                      ! the gabight configs' value
      do jj = 1, ny
         do ii = 1, nx
            s%bt_H_ref(ii, jj) = 4000.0_wp
            ! a Gaussian SSH bump -> a geostrophic adjustment problem
            s%bt_eta(ii, jj) = 0.5_wp*exp(-((real(ii - nx/2, wp)/real(max(nx/8, 1), wp))**2 &
                                            + (real(jj - ny/2, wp)/real(max(ny/8, 1), wp))**2))
         end do
      end do
      s%bt_eta_new = 0.0_wp
      s%bt_ubt = 0.0_wp; s%bt_vbt = 0.0_wp
      s%bt_ubt_prev = 0.0_wp; s%bt_vbt_prev = 0.0_wp
      s%bt_rem_u = 1.0_wp; s%bt_rem_v = 1.0_wp    ! drag off -> 1.0, as production init
      s%bt_zeta_corner = 0.0_wp; s%bt_ke_centre = 0.0_wp
      s%ubt_sum = 0.0_wp; s%vbt_sum = 0.0_wp; s%eta_sum = 0.0_wp
      s%uhbt_sum = 0.0_wp; s%vhbt_sum = 0.0_wp
   end subroutine

   !! Cross-check bar = btstep_bench.F90's own: max|d eta| judged against the
   !! FIELD magnitude max|eta|, NOT pointwise rel. bt_eta is a solution to a
   !! divergence-driven update, so it has interior cells ~ 0 where any 1-ulp
   !! wobble is 100% "pointwise relative" -- the wrong bar. The GPU->CPU swap
   !! reorders FMAs/reductions, so max|d eta| grows to ~1e-15 over the substeps
   !! (bit-identity is not expected across -stdpar=gpu vs multicore); field-rel
   !! ~1e-15 << 1e-12 is the meaningful "numerically inert" verdict.
   subroutine compare_ref(path)
      character(len=*), intent(in) :: path
      real(wp), allocatable :: ref(:, :)
      real(wp) :: dmax, emax, field_rel
      integer :: rnx, rny, u, st
      open (newunit=u, file=path, access='stream', form='unformatted', status='old', iostat=st)
      if (st /= 0) then
         write (output_unit, '(3a)') '  cross-check     : ref ', path, ' not found -- skipped'; return
      end if
      read (u) rnx, rny
      if (rnx /= nx .or. rny /= ny) then
         write (output_unit, '(a)') '  cross-check     : ref has a different shape -- skipped'
         close (u); return
      end if
      allocate (ref(rnx, rny)); read (u) ref; close (u)
      dmax = 0.0_wp; emax = 0.0_wp
      do j = 1, ny
         do i = 1, nx
            dmax = max(dmax, abs(w%bt_eta(i, j) - ref(i, j)))
            emax = max(emax, abs(w%bt_eta(i, j)))
         end do
      end do
      field_rel = dmax/max(emax, 1.0e-30_wp)
      write (output_unit, '(a,es12.5,a,es12.5)') '  cross-check vs ref: max|d eta| ', dmax, '  |eta|~', emax
      write (output_unit, '(a,es12.5)') '                      field-rel |d eta|/|eta| : ', field_rel
      if (field_rel < 1.0e-12_wp) then
         write (output_unit, '(a)') '  cross-check     : OK (<1e-12 field-rel -> data layer is numerically inert)'
      else
         write (output_unit, '(a)') '  cross-check     : *** >1e-12 field-rel -- INVESTIGATE ***'
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
