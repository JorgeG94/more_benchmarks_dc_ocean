#include "directives.h"
!! DC-only driver for the production REDI neutral-diffusion kernel.
!!
!! COMPUTE is the production do-concurrent pair (redi_calc_coeffs / Phase A +
!! redi_apply_flux / Phase B), verbatim from ocean_redi.F90. The device data
!! layer is chosen ENTIRELY by directives.h at compile time:
!!   -DDC_DATA_ACC  -> OpenACC  (nvfortran -acc=gpu -stdpar=gpu)   GPU
!!   -DDC_DATA_OMP  -> OpenMP target                               GPU (AMD/Intel too)
!!   (neither)      -> host: bare DC on the CPU (-stdpar=multicore/serial)
!! There is NO CUDA and NO nvcc in this binary. That is the point: it builds and
!! runs the do-concurrent implementation on the CPU / AMD / Intel unchanged.
!!
!! The *_cu / probe / precomp variants of the benchmark driver are DROPPED — they
!! existed only for the CUDA comparison. This driver times ONLY the production DC
!! path and cross-checks that the macro'd data layer is numerically inert.
!!
!! Cross-check (proves the macro'd data layer did not change the numbers):
!!   DC_DUMP=file  writes nx,ny,nz + the temperature tracer hTr  (a reference)
!!   DC_REF=file   reads that reference and reports max|diff| vs this run
!! Run once with DC_DATA_ACC (GPU) dumping a ref, then again on the CPU reading
!! it: agreement to FMA level means the OpenACC->host swap is numerically inert.
!!
!! Usage: ./dc_main [nx_phys] [ny_phys] [nz] [nreps] [nwarm]
program dc_main
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use constants, only: wp, NZ_STACK_MAX
   use grid, only: hgrid_t
   use ocean_metrics, only: ocean_metrics_t
   use multilayer_cgrid_state, only: multilayer_cgrid_state_t
   use ocean_eos, only: ocean_eos_t
   use ocean_boundary_types, only: ocean_bc_state_t, OBC_WALL
   use ocean_redi, only: ocean_redi_t, redi_calc_coeffs, redi_apply_flux
   implicit none

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30
   integer, parameter :: REPS_DEF = 8, WARM_DEF = 2, NGHOST = 3
   real(wp), parameter :: DT = 900.0_wp

   type(hgrid_t) :: grid
   type(ocean_metrics_t) :: metrics
   type(ocean_eos_t) :: eos
   type(ocean_bc_state_t) :: bc
   type(multilayer_cgrid_state_t) :: ms
   type(ocean_redi_t) :: redi
   real(wp), allocatable :: khtr_u_ext(:, :), khtr_v_ext(:, :)
   real(wp) :: t0, t1, ms_dc, tr_min, tr_max, tr_sum
   integer :: i, j, k, rep, nx, ny, nz, nxp, nyp, n_reps, n_warm, iu, ios
   character(len=256) :: ref_path, dump_path

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

   ! ---- VarMix per-face KhTr (MEKE khcoeff=1.0 => the `use_ext` path is live)
   allocate (khtr_u_ext(nx + 1, ny), source=100.0_wp)
   allocate (khtr_v_ext(nx, ny + 1), source=100.0_wp)

   call build_state(ms)
   call redi%init(grid, nz)
   redi%enable = .true.; redi%khtr = 100.0_wp

   write (output_unit, '(a)') repeat('=', 70)
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  domain: ', nxp, ' x ', nyp, ' x ', nz, &
      ' interior (+', NGHOST, ' ghosts/side)'
   write (output_unit, '(a,i0)') '  cells : ', nx*ny*nz
   write (output_unit, '(a,i0)') '  NZ_STACK_MAX : ', NZ_STACK_MAX
   write (output_unit, '(3a,i0,a,i0,a)') '  DATA layer: ', DC_DATA_NAME, &
      '   (reps ', n_reps, ', warm ', n_warm, ')'
   write (output_unit, '(a)') repeat('=', 70)

   ! ---- map the working set (no-ops when the DATA layer is 'host') ----------
   ! Inputs (copyin); the tracer hTr are inout -> update self at the end.
   DC_ENTER_IN(ms%h_layer)
   DC_ENTER_IN(ms%tracers(1)%hTr)
   DC_ENTER_IN(ms%tracers(2)%hTr)
   DC_ENTER_IN(metrics%dy_cu)
   DC_ENTER_IN(metrics%dx_cv)
   DC_ENTER_IN(metrics%idxCu)
   DC_ENTER_IN(metrics%idyCv)
   DC_ENTER_IN(metrics%areaT)
   DC_ENTER_IN(metrics%wet_u)
   DC_ENTER_IN(metrics%wet_v)
   DC_ENTER_IN(khtr_u_ext)
   DC_ENTER_IN(khtr_v_ext)
   ! Phase-A coefficient slots + snapshot + resolved KhTr: device-only scratch.
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

   ! ---- do concurrent, production verbatim (Phase A + Phase B) -------------
   do rep = 1, n_warm
      call redi_calc_coeffs(grid, metrics, eos, redi, ms)
      call redi_apply_flux(grid, metrics, redi, ms, DT, khtr_u_ext, khtr_v_ext, bc)
   end do
   DC_WAIT
   t0 = wall()
   do rep = 1, n_reps
      call redi_calc_coeffs(grid, metrics, eos, redi, ms)
      call redi_apply_flux(grid, metrics, redi, ms, DT, khtr_u_ext, khtr_v_ext, bc)
   end do
   DC_WAIT
   t1 = wall()
   ms_dc = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   DC_UPDATE_SELF(ms%tracers(1)%hTr)
   DC_UPDATE_SELF(ms%tracers(2)%hTr)

   tr_min = minval(ms%tracers(1)%hTr)
   tr_max = maxval(ms%tracers(1)%hTr)
   tr_sum = sum(ms%tracers(1)%hTr)

   write (output_unit, '(3a,f10.4,a)') '  do concurrent (', DC_DATA_NAME, ')  : ', ms_dc, ' ms/rep'
   write (output_unit, '(a,es14.6)') '  min hTr(temp)    : ', tr_min
   write (output_unit, '(a,es14.6)') '  max hTr(temp)    : ', tr_max
   write (output_unit, '(a,es14.6)') '  sum hTr(temp)    : ', tr_sum
   if (tr_min /= tr_min .or. (tr_min == 0.0_wp .and. tr_max == 0.0_wp)) then
      write (output_unit, '(a)') '  sanity           : *** garbage (NaN or all-zero) ***'; stop 2
   else
      write (output_unit, '(a)') '  sanity           : OK (finite, non-zero)'
   end if

   ! ---- optional reference dump / cross-check ------------------------------
   call get_environment_variable('DC_DUMP', dump_path, status=ios)
   if (ios == 0 .and. len_trim(dump_path) > 0) then
      open (newunit=iu, file=trim(dump_path), access='stream', form='unformatted', status='replace')
      write (iu) nx, ny, nz
      write (iu) ms%tracers(1)%hTr
      close (iu)
      write (output_unit, '(3a)') '  wrote ref       : ', trim(dump_path), ' (nx,ny,nz, hTr temperature)'
   end if

   call get_environment_variable('DC_REF', ref_path, status=ios)
   if (ios == 0 .and. len_trim(ref_path) > 0) call compare_ref(trim(ref_path))

   write (output_unit, '(a)') repeat('=', 70)

contains

   subroutine build_state(s)
      !! Realistic stratified column set with a horizontal T/S front (verbatim
      !! from the benchmark driver). A horizontally UNIFORM field would make
      !! dRho == 0 at every surface and send the sweep down its tie-break branch
      !! on every step -- i.e. it would not exercise the kernel that runs in
      !! production. Native indexing is BOTTOM-UP: k=1 bed, k=nz surface.
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
         write (output_unit, '(a)') '  cross-check     : ref has a different shape -- skipped'
         close (u); return
      end if
      allocate (ref(rnx, rny, rnz)); read (u) ref; close (u)
      dmax = 0.0_wp; rmax = 0.0_wp; nbad = 0
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               dmax = max(dmax, abs(ms%tracers(1)%hTr(i, j, k) - ref(i, j, k)))
               sc = max(abs(ms%tracers(1)%hTr(i, j, k)), abs(ref(i, j, k)))
               if (sc > 1.0e-30_wp) then
                  rmax = max(rmax, abs(ms%tracers(1)%hTr(i, j, k) - ref(i, j, k))/sc)
                  if (abs(ms%tracers(1)%hTr(i, j, k) - ref(i, j, k))/sc > 1.0e-12_wp) nbad = nbad + 1
               end if
            end do
         end do
      end do
      write (output_unit, '(a,es12.5,a,es12.5)') '  cross-check vs ref: max|diff| ', dmax, '  max rel ', rmax
      if (rmax < 1.0e-12_wp) then
         write (output_unit, '(a)') '  cross-check     : OK (<1e-12 rel -> data layer is numerically inert)'
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
