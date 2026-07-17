!! SHARED single-binary head-to-head: OPTIMIZED do-concurrent vs OPTIMIZED CUDA
!! for the Redi kernel, both on the SAME device arrays via `!$acc host_data
!! use_device`. Removes the two-harness caveat (OpenACC vs native cudaMalloc):
!! one binary, one truth, identical reps at production size.
!!
!! REGION TIMED (identical on both sides -- Phase A + Phase B):
!!   opt-DC   : redi_calc_coeffs (faithful Phase A)
!!            + redi_apply_flux_hoist (hoisted Phase B: face copy + per tracer
!!              snapshot + hoisted apply-flux)
!!   opt-CUDA : redi_opt_launch (opt_kernel.cu) -- the SAME sequence: calc-coeffs
!!              X/Y + face copy X/Y + per tracer (snapshot + hoisted apply-flux)
!!
!! Each side owns DISTINCT output state (ms/redi vs ms_cu/redi_cu) built from the
!! identical initial condition and run for the identical rep count, so both
!! accumulate the same number of apply-flux steps and their final states must
!! agree bit-for-bit (FMA-contraction level, < 1e-12 rel). apply-flux is not
!! idempotent -- separate states is how the legacy benchmark keeps the two
!! variants comparable without cross-contamination.
!!
!! GPU DISCIPLINE: NEVER run this binary directly. Every GPU run goes through
!!   ../tmp_local_artifacts/gpu_run.sh redi-cmp ./cmp_acc 473 297 30 200 10
!!
!! Usage: ./cmp_acc [nx_phys] [ny_phys] [nz] [nreps] [nwarm]
!!   defaults 473 297 30 200 10 -- the 0.1 deg Redi/MEKE config.
program cmp_main
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use constants, only: wp, NZ_STACK_MAX
   use grid, only: hgrid_t
   use ocean_metrics, only: ocean_metrics_t
   use multilayer_cgrid_state, only: multilayer_cgrid_state_t
   use ocean_eos, only: ocean_eos_t
   use ocean_boundary_types, only: ocean_bc_state_t, OBC_WALL
   use ocean_redi, only: ocean_redi_t, redi_calc_coeffs, redi_apply_flux_hoist
   use redi_bridge, only: redi_opt_step
   implicit none

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30
   integer, parameter :: REPS_DEF = 200, WARM_DEF = 10, NGHOST = 3
   real(wp), parameter :: DT = 900.0_wp

   type(hgrid_t) :: grid
   type(ocean_metrics_t) :: metrics
   type(ocean_eos_t) :: eos
   type(ocean_bc_state_t) :: bc
   ! Separate output state per side: sharing hTr means only the last writer is
   ! checked and the other inherits "agreement OK" silently.
   type(multilayer_cgrid_state_t) :: ms, ms_cu
   type(ocean_redi_t) :: redi, redi_cu
   real(wp), allocatable :: khtr_u_ext(:, :), khtr_v_ext(:, :)
   real(wp) :: t0, t1, ms_dc, ms_cuda
   real(wp) :: dmax, rmax, sc, o_min, o_max
   integer :: i, j, k, t, rep, nx, ny, nz, nxp, nyp, n_reps, n_warm

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

   ! ---- Metrics: uniform 0.1 deg spherical-ish, ALL WET ---------------------
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
   call build_state(ms_cu)
   call redi%init(grid, nz)
   call redi_cu%init(grid, nz)
   redi%enable = .true.; redi%khtr = 100.0_wp
   redi_cu%enable = .true.; redi_cu%khtr = 100.0_wp

   write (output_unit, '(a)') repeat('=', 70)
   write (output_unit, '(a)') ' Redi head-to-head: opt-DC vs opt-CUDA on ONE device truth (host_data)'
   write (output_unit, '(a)') repeat('=', 70)
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  grid  : ', nxp, ' x ', nyp, ' phys (+', NGHOST, &
      ' ghost) x ', nz, ' layers'
   write (output_unit, '(a,i0,a,i0,a,i0)') '  arrays: nx_total=', nx, ' ny_total=', ny, ' nsurf=', redi%nsurf
   write (output_unit, '(a,i0)') '  cells : ', nx*ny*nz
   write (output_unit, '(a,i0)') '  NZ_STACK_MAX : ', NZ_STACK_MAX
   write (output_unit, '(a,i0,a,i0)') '  reps  : ', n_reps, '   warmup: ', n_warm
   write (output_unit, '(a)') '  region: Phase A (redi_calc_coeffs) + Phase B (hoisted apply-flux)'
   write (output_unit, '(a)') repeat('=', 70)

   ! ---- Deep copy to the device: PARENT-OWNED SCOPE, each payload explicit --
   !$acc enter data copyin(metrics%dy_cu, metrics%dx_cv, metrics%idxCu, metrics%idyCv)
   !$acc enter data copyin(metrics%areaT, metrics%wet_u, metrics%wet_v)
   !$acc enter data copyin(khtr_u_ext, khtr_v_ext)
   call state_enter(ms)
   call state_enter(ms_cu)
   call redi%enter_data()
   call redi_cu%enter_data()

   ! ================= TIMED: opt-DC (Phase A + hoisted Phase B) ==============
   do rep = 1, n_warm
      call redi_calc_coeffs(grid, metrics, eos, redi, ms)
      call redi_apply_flux_hoist(grid, metrics, redi, ms, DT, khtr_u_ext, khtr_v_ext, bc)
   end do
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call redi_calc_coeffs(grid, metrics, eos, redi, ms)
      call redi_apply_flux_hoist(grid, metrics, redi, ms, DT, khtr_u_ext, khtr_v_ext, bc)
   end do
   !$acc wait
   t1 = wall()
   ms_dc = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ================= TIMED: opt-CUDA (redi_opt_launch) =====================
   ! Same device arrays (ms_cu/redi_cu are separate slots, identical initial),
   ! same rep count -> same accumulation, so the final states must agree.
   do rep = 1, n_warm
      call redi_opt_step(grid, metrics, eos, redi_cu, ms_cu, DT, khtr_u_ext, khtr_v_ext)
   end do
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call redi_opt_step(grid, metrics, eos, redi_cu, ms_cu, DT, khtr_u_ext, khtr_v_ext)
   end do
   !$acc wait
   t1 = wall()
   ms_cuda = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- Bring both results back to the host BEFORE checking them -----------
   ! Under -gpu=mem:separate the host hTr arrays are stale until state_exit's
   ! copyout runs; checking before that compares the (identical) inputs.
   call state_exit(ms)
   call state_exit(ms_cu)

   ! ---- Agreement (temperature + salinity tracers) -------------------------
   dmax = 0.0_wp; rmax = 0.0_wp
   o_min = ms_cu%tracers(1)%hTr(1, 1, 1); o_max = o_min
   do t = 1, 2
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               dmax = max(dmax, abs(ms%tracers(t)%hTr(i, j, k) - ms_cu%tracers(t)%hTr(i, j, k)))
               sc = max(abs(ms%tracers(t)%hTr(i, j, k)), abs(ms_cu%tracers(t)%hTr(i, j, k)))
               if (sc > 1.0e-30_wp) &
                  rmax = max(rmax, abs(ms%tracers(t)%hTr(i, j, k) - ms_cu%tracers(t)%hTr(i, j, k))/sc)
               o_min = min(o_min, ms_cu%tracers(t)%hTr(i, j, k))
               o_max = max(o_max, ms_cu%tracers(t)%hTr(i, j, k))
            end do
         end do
      end do
   end do

   ! ---- Report -------------------------------------------------------------
   write (output_unit, '(a,f10.4,a)') '  opt-DC       : ', ms_dc, ' ms/rep'
   write (output_unit, '(a,f10.4,a)') '  opt-CUDA     : ', ms_cuda, ' ms/rep'
   if (ms_cuda > 0.0_wp .and. ms_dc > 0.0_wp) then
      if (ms_dc <= ms_cuda) then
         write (output_unit, '(a,f7.3,a)') '  ratio        : opt-CUDA/opt-DC = ', ms_cuda/ms_dc, &
            'x   (opt-DC faster)'
      else
         write (output_unit, '(a,f7.3,a)') '  ratio        : opt-DC/opt-CUDA = ', ms_dc/ms_cuda, &
            'x   (opt-CUDA faster)'
      end if
   end if
   write (output_unit, '(a,es12.5,a,es12.5)') '  output range : min ', o_min, '  max ', o_max
   write (output_unit, '(a,es12.5,a,es12.5)') '  agreement    : max|diff| ', dmax, '  max rel ', rmax
   if (o_min == 0.0_wp .and. o_max == 0.0_wp) then
      write (output_unit, '(a)') '  sanity       : *** output all-zero (verifier trivial) ***'; stop 2
   end if
   if (rmax < 1.0e-12_wp) then
      write (output_unit, '(a)') '  verdict      : OK (opt-DC == opt-CUDA on shared arrays, <1e-12 rel)'
   else
      write (output_unit, '(a)') '  verdict      : *** DISAGREE -- layout/port bug (>1e-12 rel) ***'
   end if
   write (output_unit, '(a)') repeat('=', 70)

   call redi%exit_data(); call redi_cu%exit_data()

contains

   subroutine state_enter(s)
      type(multilayer_cgrid_state_t), intent(inout) :: s
      !$acc enter data copyin(s%h_layer)
      !$acc enter data copyin(s%tracers(1)%hTr)
      !$acc enter data copyin(s%tracers(2)%hTr)
   end subroutine state_enter

   subroutine state_exit(s)
      type(multilayer_cgrid_state_t), intent(inout) :: s
      !$acc exit data copyout(s%tracers(1)%hTr)
      !$acc exit data copyout(s%tracers(2)%hTr)
      !$acc exit data delete(s%h_layer)
   end subroutine state_exit

   subroutine build_state(s)
      !! Realistic stratified column set with a horizontal T/S front (verbatim
      !! from the benchmark drivers). Native indexing is BOTTOM-UP: k=1 bed,
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

   function wall() result(tw)
      real(wp) :: tw
      integer(int64) :: cnt, rate
      call system_clock(cnt, rate)
      tw = real(cnt, wp)/real(rate, wp)
   end function wall

end program cmp_main
