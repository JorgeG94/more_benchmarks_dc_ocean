!! Driver for the Redi neutral-diffusion benchmark: production `do concurrent`
!! vs a faithful hand-written CUDA C port.
!!
!! Times the production call pair at THERMO cadence, exactly as
!! ocean_step calls it:
!!     call redi_calc_coeffs(grid, metrics, eos, redi, ms)          ! Phase A
!!     call redi_apply_flux (grid, metrics, redi, ms, dt, ku, kv, bc)! Phase B
!! Phase B loops `size(ms%tracers)` = 2 (temperature, salinity), so one call
!! pair = 2 coeff kernels + 2 face copies + 2*(snapshot + flux) = 8 launches.
!!
!! ⚠ HARNESS RULES OBSERVED (RESUME_GPU_MRE.md §4 + the task gotchas):
!!   * Separate output state per variant: the DC and CUDA variants own DISTINCT
!!     hTr arrays (`ms` vs `ms_cu`) and DISTINCT coefficient slots (`redi` vs
!!     `redi_cu`). Sharing them means only the last writer is checked.
!!   * >= 10 untimed warm-up reps (default nwarm=10).
!!   * Deep copy is done HERE, in the scope that owns `ms`/`metrics` — never
!!     inside a helper taking them as dummies.
!!
!! Usage: ./redi_bench [nx_phys] [ny_phys] [nz] [nreps] [nwarm] [order] [dump]
!!   defaults 473 297 30 20 10 0 0  -- the 0.1 deg gabight config
!!   dump=1 writes redi_ref_init.bin / redi_ref_final.bin for redi_native.
program redi_bench
   use, intrinsic :: iso_fortran_env, only: int64
   use constants, only: wp, NZ_STACK_MAX
   use grid, only: hgrid_t
   use ocean_metrics, only: ocean_metrics_t
   use multilayer_cgrid_state, only: multilayer_cgrid_state_t
   use ocean_eos, only: ocean_eos_t
   use ocean_boundary_types, only: ocean_bc_state_t, OBC_WALL
   use ocean_redi, only: ocean_redi_t, redi_calc_coeffs, redi_apply_flux
   use ocean_redi, only: redi_snapshot_pub => redi_snapshot, redi_face_copy_pub => redi_face_copy
   use ocean_redi_pre, only: redi_tracer_precompute, redi_apply_flux_impl_pre
#ifdef WITH_CUDA
   use redi_cuda, only: redi_cuda_step, redi_cuda_probe
#endif
   implicit none

   type(hgrid_t) :: grid
   type(ocean_metrics_t) :: metrics
   type(ocean_eos_t) :: eos
   type(ocean_bc_state_t) :: bc
   ! Separate output state per variant (RESUME §4): sharing hTr means only the
   ! last writer is checked and the others inherit "agreement OK" silently.
   type(multilayer_cgrid_state_t) :: ms, ms_cu, ms_pre
   ! Pristine copy of the initial condition, never offloaded and never stepped.
   ! Only built when dump=1; lets redi_native verify its OWN init against the
   ! Fortran one, so an init mismatch cannot masquerade as a port bug.
   type(multilayer_cgrid_state_t) :: ms_ref_init
   type(ocean_redi_t) :: redi, redi_cu, redi_pre
   real(wp), allocatable :: Tlay(:,:,:), Tint(:,:,:), aLe(:,:,:), aRe(:,:,:)

   real(wp), allocatable :: khtr_u_ext(:, :), khtr_v_ext(:, :)
   integer :: nxp, nyp, nz, nreps, nwarm, nghost, order, dump
   integer :: nx, ny, i, j, k, r, it
   integer(int64) :: c0, c1, crate
   real(wp) :: t_dc, t_cu, t_pre, t_probe, dt, mx
   character(len=32) :: arg
   logical :: have_cuda

   ! ---- CLI ----------------------------------------------------------------
   nxp = 473; nyp = 297; nz = 30; nreps = 20; nwarm = 10; nghost = 3
   order = 0   ! 0 = DC timed first, 1 = CUDA timed first. See `order` note below.
   dump = 0    ! 1 = write reference state dumps for redi_native.
   if (command_argument_count() >= 1) then
      call get_command_argument(1, arg); read (arg, *) nxp
   end if
   if (command_argument_count() >= 2) then
      call get_command_argument(2, arg); read (arg, *) nyp
   end if
   if (command_argument_count() >= 3) then
      call get_command_argument(3, arg); read (arg, *) nz
   end if
   if (command_argument_count() >= 4) then
      call get_command_argument(4, arg); read (arg, *) nreps
   end if
   if (command_argument_count() >= 5) then
      call get_command_argument(5, arg); read (arg, *) nwarm
   end if
   if (command_argument_count() >= 6) then
      call get_command_argument(6, arg); read (arg, *) order
   end if
   if (command_argument_count() >= 7) then
      call get_command_argument(7, arg); read (arg, *) dump
   end if

   nx = nxp + 2*nghost
   ny = nyp + 2*nghost
   dt = 900.0_wp

   if (2*nz + 2 > NZ_STACK_MAX) then
      write (*, '(a,i0,a,i0,a)') "FATAL: 2*nz+2 = ", 2*nz + 2, " exceeds NZ_STACK_MAX = ", &
         NZ_STACK_MAX, " -- rebuild with a larger -DNZ_STACK_MAX"
      stop 1
   end if

   grid%nx_phys = nxp; grid%ny_phys = nyp; grid%nghost = nghost
   grid%nx_total = nx; grid%ny_total = ny
   grid%dx = 0.1_wp; grid%dy = 0.1_wp

   ! ---- EOS: LINEAR (no &eos_nml in any Redi-enabled config => default) -----
   ! eos%variant keeps its default EOS_VARIANT_LINEAR; rho0/T_ref/S_ref/
   ! alpha_T/beta_S keep the production defaults from ocean_eos.F90.

   ! ---- BC: all four edges WALL (gabight_sph_meke_v100 has open W/E/S, but
   ! the wall tags only remove faces from the loop; closed is the case that
   ! does the MOST work, so it is the conservative timing choice) ------------
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
   call build_state(ms_pre)
   if (dump == 1) call build_state(ms_ref_init)

   call redi%init(grid, nz)
   call redi_cu%init(grid, nz)
   call redi_pre%init(grid, nz)
   redi%enable = .true.; redi%khtr = 100.0_wp
   redi_cu%enable = .true.; redi_cu%khtr = 100.0_wp
   redi_pre%enable = .true.; redi_pre%khtr = 100.0_wp

   ! PRECOMP scratch: the price of the transform, ~139 MB at 0.1 deg.
   allocate (Tlay(nx, ny, nz), Tint(nx, ny, nz + 1), aLe(nx, ny, nz), aRe(nx, ny, nz))

   write (*, '(a)') "================================================================"
   write (*, '(a)') " Redi neutral diffusion: do concurrent vs hand-written CUDA C"
   write (*, '(a)') "================================================================"
   write (*, '(a,i0,a,i0,a,i0,a,i0,a)') " grid   : ", nxp, " x ", nyp, " phys (+", nghost, &
      " ghost) x ", nz, " layers"
   write (*, '(a,i0,a,i0,a,i0)') " arrays : nx_total=", nx, " ny_total=", ny, " nsurf=", redi%nsurf
   write (*, '(a,i0)') " cells  : ", nx*ny*nz
   write (*, '(a,i0)') " NZ_STACK_MAX : ", NZ_STACK_MAX
   write (*, '(a,i0,a,i0)') " reps   : ", nreps, "   warmup: ", nwarm
   write (*, '(a)') ""

   ! ---- Deep copy: PARENT-OWNED SCOPE. Each payload explicitly. ------------
   !$acc enter data copyin(metrics%dy_cu, metrics%dx_cv, metrics%idxCu, metrics%idyCv)
   !$acc enter data copyin(metrics%areaT, metrics%wet_u, metrics%wet_v)
   !$acc enter data copyin(khtr_u_ext, khtr_v_ext)
   call state_enter(ms)
   call state_enter(ms_cu)
   call state_enter(ms_pre)
   call redi%enter_data()
   call redi_cu%enter_data()
   call redi_pre%enter_data()
   !$acc enter data create(Tlay, Tint, aLe, aRe)

   ! ================= TIMED VARIANTS ========================================
   ! ⚠ ORDER IS LOAD-BEARING. These kernels run 45-165 ms back-to-back and the
   ! V100 drifts under sustained load (measured: +20% across ~5 consecutive
   ! runs). Whichever variant is timed SECOND is penalised. `order` swaps them;
   ! quote a number only when both orders agree. See README "Caveats".
   have_cuda = .false.
#ifdef WITH_CUDA
   have_cuda = .true.
#endif

   t_probe = 0.0_wp
   if (order == 0) then
      call time_dc()
      call time_cu()
      call time_pre()
   else
      call time_pre()
      call time_cu()
      call time_dc()
   end if
   call time_probe()

   ! ---- Report -------------------------------------------------------------
   write (*, '(a)') " variant                                   ms/call"
   write (*, '(a)') " ---------------------------------------------------"
   write (*, '(a,f12.4)') " DC  = PRODUCTION (verbatim extract)   ", t_dc
   if (have_cuda) write (*, '(a,f12.4)') " CUDA faithful port                    ", t_cu
   write (*, '(a,f12.4)') " DC + PRECOMP (hoist column rebuild)   ", t_pre
   write (*, '(a)') ""
   if (have_cuda) write (*, '(a,f8.3,a)') " dc/cuda        = ", t_dc/t_cu, " x"
   write (*, '(a,f8.3,a)') " dc/precomp     = ", t_dc/t_pre, " x   <- portable Fortran"
   if (have_cuda) write (*, '(a,f8.3,a)') " precomp/cuda   = ", t_pre/t_cu, " x"
   write (*, '(a)') ""

   ! ---- What the native C++/CUDA driver actually removes -------------------
   if (have_cuda) then
      write (*, '(a)') " OpenACC host_data use_device overhead (24 arrays, per call):"
      write (*, '(a,f12.6,a)') "   present-table lookup only        ", t_probe, " ms"
      write (*, '(a,f12.4,a)') "   CUDA variant total               ", t_cu, " ms"
      write (*, '(a,f11.3,a)') "   lookup as a share of the call    ", &
         100.0_wp*t_probe/max(t_cu, 1.0e-30_wp), " %"
      write (*, '(a)') "   ^ this is the ONLY thing redi_native removes. Host-side,"
      write (*, '(a)') "     so this figure survives GPU contention."
      write (*, '(a)') ""
   end if

   ! ---- Bring results back to the host BEFORE checking them ----------------
   ! ⚠ ORDER IS LOAD-BEARING AND WAS WRONG UNTIL 2026-07-16. `check` used to be
   ! called HERE, before state_exit -- i.e. before the copyout. Under
   ! `-gpu=mem:separate` the host hTr arrays are stale until state_exit runs,
   ! so all three variants still held their IDENTICAL build_state initial
   ! condition and every check reported max|d| = 0.0 unconditionally. It was
   ! comparing the input to itself. The old README's "DC vs CUDA is 0.0 EXACTLY"
   ! was that artifact; the true figure is ~1e-16 relative (FMA contraction),
   ! which still passes the >1e-12 port-bug bar.
   call state_exit(ms); call state_exit(ms_cu); call state_exit(ms_pre)

   ! ---- Agreement ----------------------------------------------------------
   if (have_cuda) then
      write (*, '(a)') " DC vs CUDA (FMA contraction may show ~1e-15 rel; >1e-12 = PORT BUG):"
      call check(ms, ms_cu)
   end if
   write (*, '(a)') " DC vs PRECOMP (a source transform: MUST be exactly 0.0):"
   call check(ms, ms_pre)

   ! ---- Reference dumps for the native C++/CUDA driver ---------------------
   ! `dump=1` writes the CUDA variant's initial and final state so
   ! redi_native.cu can verify itself against THIS run bit-for-bit.
   if (dump == 1) then
      call dump_state("redi_ref_init.bin", ms_ref_init)
      call dump_state("redi_ref_final.bin", ms_cu)
      write (*, '(a)') ""
      write (*, '(a)') " wrote redi_ref_init.bin + redi_ref_final.bin (for redi_native)"
   end if

   call redi%exit_data(); call redi_cu%exit_data(); call redi_pre%exit_data()

contains

   subroutine time_dc()
      integer :: rr
      do rr = 1, nwarm
         call redi_calc_coeffs(grid, metrics, eos, redi, ms)
         call redi_apply_flux(grid, metrics, redi, ms, dt, khtr_u_ext, khtr_v_ext, bc)
      end do
      call system_clock(c0, crate)
      do rr = 1, nreps
         call redi_calc_coeffs(grid, metrics, eos, redi, ms)
         call redi_apply_flux(grid, metrics, redi, ms, dt, khtr_u_ext, khtr_v_ext, bc)
      end do
      call system_clock(c1)
      t_dc = 1000.0_wp*real(c1 - c0, wp)/real(crate, wp)/real(nreps, wp)
   end subroutine time_dc

   subroutine time_cu()
#ifdef WITH_CUDA
      integer :: rr
      do rr = 1, nwarm
         call redi_cuda_step(grid, metrics, eos, redi_cu, ms_cu, dt, khtr_u_ext, khtr_v_ext)
      end do
      call system_clock(c0, crate)
      do rr = 1, nreps
         call redi_cuda_step(grid, metrics, eos, redi_cu, ms_cu, dt, khtr_u_ext, khtr_v_ext)
      end do
      call system_clock(c1)
      t_cu = 1000.0_wp*real(c1 - c0, wp)/real(crate, wp)/real(nreps, wp)
#endif
   end subroutine time_cu

   subroutine time_probe()
      !! Cost of the `host_data use_device` present-table lookup ALONE (24
      !! arrays), with the kernels replaced by a no-op. This is the overhead the
      !! native C++/CUDA driver removes; measuring it says whether removing it
      !! could possibly matter. Host-side, so contention-proof.
      !! MANY reps: one lookup is ~microseconds and system_clock is coarse.
      integer :: rr
      integer, parameter :: nprobe = 2000
#ifdef WITH_CUDA
      do rr = 1, 100
         call redi_cuda_probe(grid, metrics, eos, redi_cu, ms_cu, dt, khtr_u_ext, khtr_v_ext)
      end do
      call system_clock(c0, crate)
      do rr = 1, nprobe
         call redi_cuda_probe(grid, metrics, eos, redi_cu, ms_cu, dt, khtr_u_ext, khtr_v_ext)
      end do
      call system_clock(c1)
      t_probe = 1000.0_wp*real(c1 - c0, wp)/real(crate, wp)/real(nprobe, wp)
#endif
   end subroutine time_probe

   subroutine time_pre()
      !! PRECOMP mirrors redi_apply_flux's outer shim: face KhTr copy, then
      !! per tracer { snapshot -> precompute columns -> flux }. Phase A
      !! (redi_calc_coeffs) is UNCHANGED and shared -- the transform is Phase B
      !! only.
      integer :: rr, tt
      do rr = 1, nwarm + nreps
         if (rr == nwarm + 1) call system_clock(c0, crate)
         call redi_calc_coeffs(grid, metrics, eos, redi_pre, ms_pre)
         call redi_face_copy_pub(nx + 1, ny, khtr_u_ext, redi_pre%khtr_u)
         call redi_face_copy_pub(nx, ny + 1, khtr_v_ext, redi_pre%khtr_v)
         do tt = 1, 2
            call redi_snapshot_pub(nx, ny, nz, ms_pre%tracers(tt)%hTr, redi_pre%tr_snap)
            call redi_tracer_precompute(nx, ny, nz, ms_pre%h_layer, redi_pre%tr_snap, &
                                        Tlay, Tint, aLe, aRe)
            call redi_apply_flux_impl_pre(nx, ny, nz, redi_pre%nsurf, dt, &
                                          nghost, nxp, nyp, .true., .true., .true., .true., &
                                          redi_pre%khtr_u, redi_pre%khtr_v, &
                                          metrics%dy_cu, metrics%dx_cv, &
                                          metrics%idxCu, metrics%idyCv, metrics%areaT, &
                                          ms_pre%tracers(tt)%hTr, Tlay, Tint, aLe, aRe, &
                                          redi_pre%uPoL, redi_pre%uPoR, redi_pre%uKoL, &
                                          redi_pre%uKoR, redi_pre%uhEff, &
                                          redi_pre%vPoL, redi_pre%vPoR, redi_pre%vKoL, &
                                          redi_pre%vKoR, redi_pre%vhEff)
         end do
      end do
      call system_clock(c1)
      t_pre = 1000.0_wp*real(c1 - c0, wp)/real(crate, wp)/real(nreps, wp)
   end subroutine time_pre

   subroutine build_state(s)
      !! Realistic stratified column set with a horizontal T/S front. A
      !! horizontally UNIFORM field would make dRho == 0 at every surface and
      !! send the sweep down its tie-break branch on every step -- i.e. it
      !! would not exercise the kernel that runs in production.
      !! Native indexing is BOTTOM-UP: k=1 bed, k=nz surface.
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
         ! zz = depth below surface at layer centre; k=nz is the surface layer.
         zz = (real(nz - kk, wp) + 0.5_wp)*hh
         do jj = 1, ny
            fy = real(jj - 1, wp)/real(ny - 1, wp)
            do ii = 1, nx
               fx = real(ii - 1, wp)/real(nx - 1, wp)
               ! Exponential thermocline + a diagonal surface front (~6 degC
               ! across the basin) -- gives genuinely tilted neutral surfaces.
               tt = 2.0_wp + 16.0_wp*exp(-zz/800.0_wp) + 3.0_wp*fx + 3.0_wp*fy
               ss = 34.5_wp + 0.7_wp*(1.0_wp - exp(-zz/1500.0_wp)) - 0.2_wp*fx
               s%h_layer(ii, jj, kk) = hh
               s%tracers(1)%hTr(ii, jj, kk) = tt*hh
               s%tracers(2)%hTr(ii, jj, kk) = ss*hh
            end do
         end do
      end do
   end subroutine build_state

   subroutine state_enter(s)
      !! Deep copy of the state payloads. Called from the program scope that
      !! OWNS `s` -- doing this inside a routine that takes `s` as a dummy maps
      !! the dummy, the components never attach, and every region silently
      !! falls back to implicit copies (RESUME gotcha 3).
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

   subroutine dump_state(fname, s)
      !! Raw little-endian stream dump for redi_native.cu to read back:
      !!   int32 nx, ny, nz
      !!   float64 h_layer(nx,ny,nz), hTr_T(nx,ny,nz), hTr_S(nx,ny,nz)
      !! `access='stream'` -- NOT the default sequential, which would interleave
      !! 4-byte record-length markers and desynchronise the C reader.
      character(len=*), intent(in) :: fname
      type(multilayer_cgrid_state_t), intent(in) :: s
      integer :: u
      open (newunit=u, file=fname, form='unformatted', access='stream', status='replace')
      write (u) nx, ny, nz
      write (u) s%h_layer
      write (u) s%tracers(1)%hTr
      write (u) s%tracers(2)%hTr
      close (u)
   end subroutine dump_state

   subroutine check(a, b)
      !! Bit-identity between variants. Source-level fixes must be EXACTLY 0.0;
      !! DC vs CUDA may differ ~1e-15 relative from FMA contraction. Anything
      !! beyond ~1e-12 relative is a PORT BUG, not rounding.
      type(multilayer_cgrid_state_t), intent(in) :: a, b
      integer :: t
      real(wp) :: d, scl
      do t = 1, 2
         mx = 0.0_wp; scl = 0.0_wp
         do k = 1, nz
            do j = 1, ny
               do i = 1, nx
                  d = abs(a%tracers(t)%hTr(i, j, k) - b%tracers(t)%hTr(i, j, k))
                  if (d > mx) mx = d
                  if (abs(a%tracers(t)%hTr(i, j, k)) > scl) scl = abs(a%tracers(t)%hTr(i, j, k))
               end do
            end do
         end do
         write (*, '(a,i0,a,es12.4,a,es12.4)') " tracer ", t, ": max|dc-cuda| = ", mx, &
            "   relative = ", mx/max(scl, 1.0e-30_wp)
      end do
   end subroutine check

end program redi_bench
