#include "directives.h"
!! A/B driver for the REDI apply-flux `do concurrent` optimization.
!!
!! Mirrors continuity_layered/drivers/dc_ab.F90: it times the FAITHFUL public
!! path (redi_apply_flux, byte-identical to production) against the OPTIMIZED
!! path (redi_apply_flux_hoist, the precompute hoist) and checks they are
!! BIT-IDENTICAL (max rel diff < 1e-12) on the temperature tracer hTr.
!!
!! Only apply-flux differs between A and B, so this driver isolates apply-flux:
!! redi_calc_coeffs (Phase A) is run ONCE up front to populate the neutral-
!! position coefficient slots, then both apply-flux variants are timed against
!! those fixed coefficients. This is the DC analogue of the CUDA "ApplyFlux"
!! row in OPTIMIZATION.md (the kernel-level number, not the whole launch).
!!
!! apply-flux ACCUMULATES into hTr (it is not idempotent), so the tracer is
!! reset to its initial value before each measured block; the hoist drifts
!! bit-identically to the faithful path, so both timed loops do the same work.
!!
!! The device data layer is chosen ENTIRELY by directives.h at compile time
!! (DC_DATA_ACC / DC_DATA_OMP / host). There is NO CUDA in this binary.
!!
!! Usage: ./dc_ab [nx_phys] [ny_phys] [nz] [nreps] [nwarm]
program dc_ab
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use constants, only: wp, NZ_STACK_MAX
   use grid, only: hgrid_t
   use ocean_metrics, only: ocean_metrics_t
   use multilayer_cgrid_state, only: multilayer_cgrid_state_t
   use ocean_eos, only: ocean_eos_t
   use ocean_boundary_types, only: ocean_bc_state_t, OBC_WALL
   use ocean_redi, only: ocean_redi_t, redi_calc_coeffs, redi_apply_flux, redi_apply_flux_hoist
   implicit none

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30
   integer, parameter :: REPS_DEF = 50, WARM_DEF = 5, NGHOST = 3
   real(wp), parameter :: DT = 900.0_wp

   type(hgrid_t) :: grid
   type(ocean_metrics_t) :: metrics
   type(ocean_eos_t) :: eos
   type(ocean_bc_state_t) :: bc
   type(multilayer_cgrid_state_t) :: ms
   type(ocean_redi_t) :: redi
   real(wp), allocatable :: khtr_u_ext(:, :), khtr_v_ext(:, :)
   real(wp), allocatable :: hTr1_init(:, :, :), hTr2_init(:, :, :), hf1(:, :, :)
   real(wp) :: t0, t1, ms_faithful, ms_hoist
   real(wp) :: dmax, rmax, sc, f_min, f_max, f_sum
   integer :: i, j, k, rep, nx, ny, nz, nxp, nyp, n_reps, n_warm

   nxp = iarg(1, NXP_DEF); nyp = iarg(2, NYP_DEF); nz = iarg(3, NZ_DEF)
   n_reps = iarg(4, REPS_DEF); n_warm = iarg(5, WARM_DEF)
   nx = nxp + 2*NGHOST; ny = nyp + 2*NGHOST
   if (nx < 6 .or. ny < 6 .or. nz < 1) then
      write (output_unit, '(a)') 'ERROR: need nx_phys,ny_phys >= 1 and nz >= 1'; stop 1
   end if
   if (2*nz + 2 > NZ_STACK_MAX) then
      write (output_unit, '(a,i0,a,i0)') 'ERROR: 2*nz+2 = ', 2*nz + 2, ' exceeds NZ_STACK_MAX = ', NZ_STACK_MAX
      stop 1
   end if
   grid%nx_total = nx; grid%ny_total = ny
   grid%nx_phys = nxp; grid%ny_phys = nyp
   grid%nghost = NGHOST; grid%dx = 0.1_wp; grid%dy = 0.1_wp

   ! ---- BC: all four edges WALL (the case that does the MOST work) ----------
   bc%west%bc_type = OBC_WALL; bc%east%bc_type = OBC_WALL
   bc%south%bc_type = OBC_WALL; bc%north%bc_type = OBC_WALL

   ! ---- Metrics: uniform 0.1 deg spherical-ish, ALL WET --------------------
   allocate (metrics%dy_cu(nx + 1, ny), source=11100.0_wp)
   allocate (metrics%dx_cv(nx, ny + 1), source=8000.0_wp)
   allocate (metrics%idxCu(nx + 1, ny), source=1.0_wp/8000.0_wp)
   allocate (metrics%idyCv(nx, ny + 1), source=1.0_wp/11100.0_wp)
   allocate (metrics%areaT(nx, ny), source=8000.0_wp*11100.0_wp)
   allocate (metrics%wet_u(nx + 1, ny), source=1.0_wp)
   allocate (metrics%wet_v(nx, ny + 1), source=1.0_wp)

   allocate (khtr_u_ext(nx + 1, ny), source=100.0_wp)
   allocate (khtr_v_ext(nx, ny + 1), source=100.0_wp)

   call build_state(ms)
   call redi%init(grid, nz)
   redi%enable = .true.; redi%khtr = 100.0_wp

   ! Keep pristine copies of the initial tracers to reset between blocks.
   allocate (hTr1_init(nx, ny, nz), source=ms%tracers(1)%hTr)
   allocate (hTr2_init(nx, ny, nz), source=ms%tracers(2)%hTr)
   allocate (hf1(nx, ny, nz))

   write (output_unit, '(a)') repeat('=', 70)
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  domain: ', nxp, ' x ', nyp, ' x ', nz, &
      ' interior (+', NGHOST, ' ghosts/side)'
   write (output_unit, '(a,i0)') '  cells : ', nx*ny*nz
   write (output_unit, '(a,i0)') '  NZ_STACK_MAX : ', NZ_STACK_MAX
   write (output_unit, '(3a,i0,a,i0,a)') '  DATA layer: ', DC_DATA_NAME, &
      '   (reps ', n_reps, ', warm ', n_warm, ')'
   write (output_unit, '(a)') '  A/B: apply-flux only (coeffs fixed) -- faithful vs hoist'
   write (output_unit, '(a)') repeat('=', 70)

   ! ---- map the working set (no-ops when the DATA layer is 'host') ----------
   DC_ENTER_IN(ms%h_layer)
   DC_ENTER_IN(ms%tracers(1)%hTr)
   DC_ENTER_IN(ms%tracers(2)%hTr)
   DC_ENTER_IN(hTr1_init)
   DC_ENTER_IN(hTr2_init)
   DC_ENTER_IN(metrics%dy_cu)
   DC_ENTER_IN(metrics%dx_cv)
   DC_ENTER_IN(metrics%idxCu)
   DC_ENTER_IN(metrics%idyCv)
   DC_ENTER_IN(metrics%areaT)
   DC_ENTER_IN(metrics%wet_u)
   DC_ENTER_IN(metrics%wet_v)
   DC_ENTER_IN(khtr_u_ext)
   DC_ENTER_IN(khtr_v_ext)
   DC_ENTER_CREATE(redi%uPoL)
   DC_ENTER_CREATE(redi%uPoR)
   DC_ENTER_CREATE(redi%uKoL)
   DC_ENTER_CREATE(redi%uKoR)
   DC_ENTER_CREATE(redi%uhEff)
   DC_ENTER_CREATE(redi%vPoL)
   DC_ENTER_CREATE(redi%vPoR)
   DC_ENTER_CREATE(redi%vKoL)
   DC_ENTER_CREATE(redi%vKoR)
   DC_ENTER_CREATE(redi%vhEff)
   DC_ENTER_CREATE(redi%khtr_u)
   DC_ENTER_CREATE(redi%khtr_v)
   DC_ENTER_CREATE(redi%tr_snap)

   ! ---- Phase A once: populate the neutral-position coefficient slots -------
   call redi_calc_coeffs(grid, metrics, eos, redi, ms)
   DC_WAIT

   ! =====================================================================
   ! Correctness: one apply from the SAME initial tracer, faithful vs hoist.
   ! =====================================================================
   call reset_tracers()
   call redi_apply_flux(grid, metrics, redi, ms, DT, khtr_u_ext, khtr_v_ext, bc)
   DC_UPDATE_SELF(ms%tracers(1)%hTr)
   hf1 = ms%tracers(1)%hTr

   call reset_tracers()
   call redi_apply_flux_hoist(grid, metrics, redi, ms, DT, khtr_u_ext, khtr_v_ext, bc)
   DC_UPDATE_SELF(ms%tracers(1)%hTr)

   dmax = 0.0_wp; rmax = 0.0_wp
   f_min = hf1(1, 1, 1); f_max = hf1(1, 1, 1); f_sum = 0.0_wp
   do k = 1, nz
      do j = 1, ny
         do i = 1, nx
            dmax = max(dmax, abs(ms%tracers(1)%hTr(i, j, k) - hf1(i, j, k)))
            sc = max(abs(ms%tracers(1)%hTr(i, j, k)), abs(hf1(i, j, k)))
            if (sc > 1.0e-30_wp) rmax = max(rmax, abs(ms%tracers(1)%hTr(i, j, k) - hf1(i, j, k))/sc)
            f_min = min(f_min, hf1(i, j, k)); f_max = max(f_max, hf1(i, j, k))
            f_sum = f_sum + hf1(i, j, k)
         end do
      end do
   end do

   ! =====================================================================
   ! Timing A: faithful apply-flux
   ! =====================================================================
   call reset_tracers()
   do rep = 1, n_warm
      call redi_apply_flux(grid, metrics, redi, ms, DT, khtr_u_ext, khtr_v_ext, bc)
   end do
   DC_WAIT
   t0 = wall()
   do rep = 1, n_reps
      call redi_apply_flux(grid, metrics, redi, ms, DT, khtr_u_ext, khtr_v_ext, bc)
   end do
   DC_WAIT
   t1 = wall()
   ms_faithful = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! =====================================================================
   ! Timing B: hoisted apply-flux
   ! =====================================================================
   call reset_tracers()
   do rep = 1, n_warm
      call redi_apply_flux_hoist(grid, metrics, redi, ms, DT, khtr_u_ext, khtr_v_ext, bc)
   end do
   DC_WAIT
   t0 = wall()
   do rep = 1, n_reps
      call redi_apply_flux_hoist(grid, metrics, redi, ms, DT, khtr_u_ext, khtr_v_ext, bc)
   end do
   DC_WAIT
   t1 = wall()
   ms_hoist = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- report -------------------------------------------------------------
   write (output_unit, '(a,es14.6,a,es14.6)') '  faithful hTr     : min ', f_min, '  max ', f_max
   write (output_unit, '(a,es14.6)') '  faithful sum hTr : ', f_sum
   write (output_unit, '(a,es12.5,a,es12.5)') '  correctness      : max|diff| ', dmax, '  max rel ', rmax
   if (f_min == 0.0_wp .and. f_max == 0.0_wp) then
      write (output_unit, '(a)') '  sanity           : *** faithful output all-zero (verifier trivial) ***'; stop 2
   end if
   if (rmax < 1.0e-12_wp) then
      write (output_unit, '(a)') '  verdict          : OK (hoist == faithful, <1e-12 rel)'
   else
      write (output_unit, '(a)') '  verdict          : *** DIFF -- hoisted kernel bug ***'
   end if
   write (output_unit, '(a,f10.4,a)') '  faithful DC      : ', ms_faithful, ' ms/rep'
   write (output_unit, '(a,f10.4,a,f7.3,a)') '  hoisted  DC      : ', ms_hoist, ' ms/rep   -> ', &
      ms_faithful/ms_hoist, 'x'
   write (output_unit, '(a)') repeat('=', 70)

contains

   subroutine reset_tracers()
      !! Restore both tracers to their initial values on host AND device.
      ms%tracers(1)%hTr = hTr1_init
      ms%tracers(2)%hTr = hTr2_init
      DC_UPDATE_DEVICE(ms%tracers(1)%hTr)
      DC_UPDATE_DEVICE(ms%tracers(2)%hTr)
   end subroutine reset_tracers

   subroutine build_state(s)
      !! Realistic stratified column set with a horizontal T/S front (verbatim
      !! from the benchmark driver). Native indexing is BOTTOM-UP: k=1 bed,
      !! k=nz surface.
      type(multilayer_cgrid_state_t), intent(inout) :: s
      integer :: ii, jj, kk
      real(wp) :: depth, zz, tt, ss, hh, fx, fy

      s%nz_ml = nz
      s%idx_temperature = 1
      s%idx_salinity = 2
      allocate (s%h_layer(nx, ny, nz))
      allocate (s%tracers(2))
      allocate (s%tracers(1)%hTr(nx, ny, nz))
      allocate (s%tracers(2)%hTr(nx, ny, nz))

      depth = 4000.0_wp
      hh = depth/real(nz, wp)
      do kk = 1, nz
         zz = (real(nz - kk, wp) + 0.5_wp)*hh
         do jj = 1, ny
            fy = real(jj - 1, wp)/real(ny - 1, wp)
            do ii = 1, nx
               fx = real(ii - 1, wp)/real(nx - 1, wp)
               tt = 2.0_wp + 16.0_wp*exp(-zz/800.0_wp) + 3.0_wp*fx + 3.0_wp*fy
               ss = 34.5_wp + 0.7_wp*(1.0_wp - exp(-zz/1500.0_wp)) - 0.2_wp*fx
               s%h_layer(ii, jj, kk) = hh
               s%tracers(1)%hTr(ii, jj, kk) = tt*hh
               s%tracers(2)%hTr(ii, jj, kk) = ss*hh
            end do
         end do
      end do
   end subroutine build_state

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
