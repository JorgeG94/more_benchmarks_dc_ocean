!! Driver for the ocean model's kappa-shear (JHL08) column kernel:
!! production `do concurrent` vs hand-written CUDA C.
!!
!! WHY THIS KERNEL: `ocean_vmix_compute` is 39-41% of production runtime -- the
!! single largest region in the profile
!! (<scratch>/<user>/tassie_runs/gabight_acc_120d.173705307.<sched>.log) --
!! and it wraps EPBL + kappa-shear. It had never been measured.
!!
!! WHY IT IS STRUCTURALLY DIFFERENT from every kernel measured before it: 1443
!! production lines carry only FOUR `do concurrent` loops. The parallelism is
!! one-thread-per-COLUMN with a fully sequential-in-k solve inside, and that
!! solve ITERATES to convergence (Picard inner x adaptive-substep outer). So the
!! questions are NOT codegen or launch overhead. They are divergence, per-thread
!! state, and occupancy.
!!
!! CADENCE: production calls kappa_shear_compute once per outer step, stage 1
!! only, with `dyn%therm_dt(dt)` (ocean_dyn.F90:851). `dt_therm_ratio`
!! defaults to 1 and no gabight namelist sets it, so therm_dt == dt ==
!! dt_fixed == 300.0 s. That is the dt used here -- and it matters: dt sets how
!! many adaptive substeps the outer loop needs.
!!
!! ============================================================================
!! THE STATE, AND WHY IT IS BUILT THIS WAY
!! ============================================================================
!! An iterative scheme is only as honest as its state. A trivially-stratified
!! column converges in one iteration where production takes many, so a "nice"
!! state would fake a tie. kappa-shear fires where the gradient Richardson
!! number Ri = N^2/S^2 drops below ri_crit = 0.25, so the state must actually
!! straddle Ri_c. This one does, by construction:
!!
!!   - z* vertical grid, geometric stretch r = 1.18 -> ~5 m surface layer at
!!     H = 4000 m, matching `&vcoord_nml zstar_h_surf_target = 5.0`.
!!   - A surface MIXED LAYER of depth MLD(i,j) in [40, 100] m: T and S uniform
!!     inside it, so N^2 ~ 0 there -- which is the regime EPBL actually
!!     maintains and the reason kappa-shear has anything to do at all.
!!   - A thermocline below (T_surf = 14 -> T_bot = 2 over ~600 m), from
!!     `&tracer_nml`.
!!   - A surface-intensified jet u = U_s(i,j)*0.5*(1 - tanh((z-MLD)/25)), so
!!     the SHEAR IS CONCENTRATED AT THE MIXED-LAYER BASE -- exactly where the
!!     stratification is too. That is the physically real competition.
!!   - U_s(i,j) sweeps 0.25 -> 0.85 m/s across the domain (ACC-like). At the ML
!!     base this drives Ri from ~0.5 (stable, no source, column exits early) to
!!     ~0.1 (strongly unstable, many iterations). SO THE DOMAIN CONTAINS BOTH,
!!     AND THE ITERATION COUNT VARIES PER COLUMN. That variation is the thing
!!     under test -- it is what makes a warp run at its slowest lane.
!!
!! EOS: linear, alpha_T = 0.2, beta_S = 7.6e-4, rho0 = 1035. Not a default
!! guess -- gabight carries no `&ocean_eos_nml`, so config.F90:264's
!! `eos = "linear"` stands, and ocean_state.F90:474-475 overwrites alpha_T
!! and rho0 from `&ocean_ic_nml` (alpha_T = 0.2, rho_0 = 1035.0).
!!
!! LAND: default all-wet (`land_pct = 0`), matching the sibling benchmarks. A
!! land column returns after the gather, which is a real and large divergence
!! source in a coastal domain -- but a trivially-understood one that would mask
!! the interesting effect (iteration-count spread among WET columns). `land_pct`
!! is a CLI knob; the README reports both.
!! ============================================================================
!!
!! MEMORY: -gpu=mem:separate + manual deep copy (parent struct, then payload).
!! Both toolchains read the SAME device allocation via host_data use_device, so
!! the only variable is who generated the code.
!!
!! Usage: ./ks_bench [nxp] [nyp] [nz] [nreps] [nwarm] [cuda_sync] [land_pct]
!!   ./ks_bench                 -> 473 297 30 (the 0.1 deg gabight config)
!!   ./ks_bench 945 594 30      -> the 0.05 deg config
!!   ./ks_bench 473 297 30 20 10 2 25   -> 25% land
program ks_bench
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr, c_loc, c_null_ptr
   use constants, only: wp, NZ_STACK_MAX
   use grid, only: hgrid_t
   use multilayer_cgrid_state, only: multilayer_cgrid_state_t
   use ocean_eos, only: EOS_VARIANT_LINEAR
   use ks, only: ocean_kappa_shear_t, kappa_shear_column_kernel
   implicit none

   !! Mirrors `struct KsPar` in ks_kernel.cu. bind(C) so the layout is the
   !! compiler's problem, not mine.
   type, bind(C) :: ks_par_t
      real(c_double) :: dt, ri_crit, shearmix_rate, fri_curvature
      real(c_double) :: c_n, c_s, lambda, lz_rescale
      real(c_double) :: kappa_0, kappa_seed, kappa_trunc, tke_bg
      real(c_double) :: tol_err, src_max_chg, vel_underflow, rho0
      integer(c_int) :: max_inner_it, max_substep_it
      integer(c_int) :: eos_variant
      real(c_double) :: eos_rho0, eos_alpha_T, eos_beta_S
   end type ks_par_t

   interface
      subroutine ks_cuda_launch(h, u, v, hT, hS, wet, fc, kd, tke, n_out, n_in, &
                                nx, ny, nz, p, sync) bind(C, name="ks_cuda_launch")
         import :: c_double, c_int, ks_par_t
         implicit none
         real(c_double), intent(in) :: h(*), u(*), v(*), hT(*), hS(*)
         real(c_double), intent(in) :: wet(*), fc(*)
         real(c_double), intent(inout) :: kd(*), tke(*)
         integer(c_int), intent(inout) :: n_out(*), n_in(*)
         integer(c_int), value :: nx, ny, nz, sync
         type(ks_par_t), intent(in) :: p
      end subroutine

      subroutine ks_cuda_attrs(nregs, lmem, smem, maxtpb) bind(C, name="ks_cuda_attrs")
         import :: c_int
         implicit none
         integer(c_int), intent(out) :: nregs, lmem, smem, maxtpb
      end subroutine

#ifdef HAVE_WARP
      subroutine ks_warp_launch(h, u, v, hT, hS, wet, fc, kd, tke, &
                                nx, ny, nz, p, sync) bind(C, name="ks_warp_launch")
         import :: c_double, c_int, ks_par_t
         implicit none
         real(c_double), intent(in) :: h(*), u(*), v(*), hT(*), hS(*)
         real(c_double), intent(in) :: wet(*), fc(*)
         real(c_double), intent(inout) :: kd(*), tke(*)
         integer(c_int), value :: nx, ny, nz, sync
         type(ks_par_t), intent(in) :: p
      end subroutine

      subroutine ks_warp_attrs(nregs, lmem, smem, maxtpb) bind(C, name="ks_warp_attrs")
         import :: c_int
         implicit none
         integer(c_int), intent(out) :: nregs, lmem, smem, maxtpb
      end subroutine
#endif
   end interface

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30
   integer, parameter :: REPS_DEF = 20, WARM_DEF = 10, NGHOST = 3
   real(wp), parameter :: DT_THERM = 300.0_wp   !! &time_nml dt_fixed, therm ratio 1

   type(hgrid_t) :: grid
   type(multilayer_cgrid_state_t) :: ms
   type(ocean_kappa_shear_t) :: ks
   type(ks_par_t), target :: par

   ! SEPARATE output state per variant. Sharing them means only the last writer
   ! is checked and the others silently inherit "agreement OK" (RESUME §4).
   real(wp), allocatable :: kd_cu(:, :, :), tke_cu(:, :, :)
   real(wp), allocatable :: kd_wp_(:, :, :), tke_wp_(:, :, :)
   real(wp), allocatable :: hT(:, :, :), hS(:, :, :)
   integer(c_int), allocatable :: n_out(:, :), n_in(:, :)

   real(wp) :: t0, t1, ms_dc, ms_cu, ms_wp
   real(wp) :: dmax, rmax, scale, gib, dmaxw, rmaxw
   real(wp) :: kd_min, kd_max, kd_sum
   integer :: i, j, k, rep, nx, ny, nz, nxp, nyp, n_reps, n_warm, cu_sync
   integer :: nbad, nbadw, land_pct, ncol, nwet, nactive
   integer :: nregs, lmem, smem, maxtpb
   logical :: have_warp

   nxp = iarg(1, NXP_DEF)
   nyp = iarg(2, NYP_DEF)
   nz = iarg(3, NZ_DEF)
   n_reps = iarg(4, REPS_DEF)
   n_warm = iarg(5, WARM_DEF)
   cu_sync = iarg(6, 1)
   land_pct = iarg(7, 0)

   nx = nxp + 2*NGHOST
   ny = nyp + 2*NGHOST
   if (nx < 1 .or. ny < 1 .or. nz < 2) then
      write (output_unit, '(a)') 'ERROR: need nxp,nyp >= 1 and nz >= 2'
      stop 1
   end if
   if (nz + 1 > NZ_STACK_MAX) then
      write (output_unit, '(a,i0,a,i0)') 'ERROR: nz+1 = ', nz + 1, &
         ' exceeds NZ_STACK_MAX = ', NZ_STACK_MAX
      stop 1
   end if
   grid%nx_total = nx; grid%ny_total = ny
   grid%nx_phys = nxp; grid%ny_phys = nyp
   grid%nghost = NGHOST
   grid%dx = 0.1_wp; grid%dy = 0.1_wp
   ncol = nx*ny

   allocate (ms%h_layer(nx, ny, nz))
   allocate (ms%u_face_x_layer(nx + 1, ny, nz), ms%v_face_y_layer(nx, ny + 1, nz))
   allocate (ms%wet_mask(nx, ny))
   allocate (hT(nx, ny, nz), hS(nx, ny, nz))
   allocate (ks%f_centre(nx, ny))
   allocate (ks%kd_int(nx, ny, nz + 1), ks%tke_int(nx, ny, nz + 1))
   allocate (kd_cu(nx, ny, nz + 1), tke_cu(nx, ny, nz + 1))
   allocate (kd_wp_(nx, ny, nz + 1), tke_wp_(nx, ny, nz + 1))
   allocate (n_out(nx, ny), n_in(nx, ny))
   ms%nz_ml = nz

   gib = 11.0_wp*real(nx, wp)*real(ny, wp)*real(nz, wp)*8.0_wp/(1024.0_wp**3)

   call build_state()

   ks%enable = .true.
   ks%eos%variant = EOS_VARIANT_LINEAR
   ks%eos%rho0 = 1035.0_wp
   ks%eos%alpha_T = 0.2_wp     ! &ocean_ic_nml alpha_T -> ocean_state.F90:474
   ks%eos%beta_S = 7.6e-4_wp   ! ocean_eos.F90 default (no nml override)
   ks%rho0 = 1035.0_wp
   ks%kd_int = 0.0_wp; ks%tke_int = 0.0_wp
   kd_cu = 0.0_wp; tke_cu = 0.0_wp
   kd_wp_ = 0.0_wp; tke_wp_ = 0.0_wp
   n_out = 0; n_in = 0

   par%dt = DT_THERM
   par%ri_crit = ks%ri_crit; par%shearmix_rate = ks%shearmix_rate
   par%fri_curvature = ks%fri_curvature
   par%c_n = ks%c_n; par%c_s = ks%c_s; par%lambda = ks%lambda
   par%lz_rescale = ks%lz_rescale
   par%kappa_0 = ks%kappa_0; par%kappa_seed = ks%kappa_seed
   par%kappa_trunc = ks%kappa_trunc; par%tke_bg = ks%tke_bg
   par%tol_err = ks%tol_err; par%src_max_chg = ks%src_max_chg
   par%vel_underflow = ks%vel_underflow; par%rho0 = ks%rho0
   par%max_inner_it = ks%max_inner_it; par%max_substep_it = ks%max_substep_it
   par%eos_variant = ks%eos%variant
   par%eos_rho0 = ks%eos%rho0; par%eos_alpha_T = ks%eos%alpha_T
   par%eos_beta_S = ks%eos%beta_S

   nwet = count(ms%wet_mask > 0.0_wp)
   write (output_unit, '(a)') repeat('=', 74)
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  domain: ', nxp, ' x ', nyp, &
      ' x ', nz, ' interior (+', NGHOST, ' ghosts/side)'
   write (output_unit, '(a,i0,a,i0,a,i0,a)') '  arrays: ', nx, ' x ', ny, &
      ' -> ', ncol, ' COLUMNS (the parallel width)'
   write (output_unit, '(a,i0,a,f5.1,a)') '  wet columns: ', nwet, '  (', &
      100.0_wp*real(nwet, wp)/real(ncol, wp), ' %)'
   write (output_unit, '(a,i0,a,i0,a)') '  NZ_STACK_MAX = ', NZ_STACK_MAX, &
      ' (production); nz = ', nz, ' -> per-thread arrays are 4x oversized'
   write (output_unit, '(a,f6.1,a)') '  dt (therm) = ', DT_THERM, ' s'
   write (output_unit, '(a,i0,a,i0,a)') '  reps:   ', n_reps, ' timed, ', n_warm, ' warm-up'
   write (output_unit, '(a,f0.2,a,i0)') '  global mem ~', gib, ' GiB;  cuda_sync = ', cu_sync
   write (output_unit, '(a)') repeat('=', 74)

   ! ---- Deep copy. MUST happen in the scope that OWNS the variables:
   ! parent struct first, then each payload (RESUME gotcha 3).
   !$acc enter data copyin(ms)
   !$acc enter data copyin(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, ms%wet_mask)
   !$acc enter data copyin(ks)
   !$acc enter data copyin(ks%f_centre)
   !$acc enter data create(ks%kd_int, ks%tke_int)
   !$acc enter data copyin(hT, hS)
   !$acc enter data create(kd_cu, tke_cu, kd_wp_, tke_wp_)
   !$acc enter data create(n_out, n_in)

   ! ---- variant A: do concurrent (production) ---------------------------
   do rep = 1, n_warm
      call kappa_shear_column_kernel(grid, ks, ms, hT, hS, DT_THERM)
   end do
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call kappa_shear_column_kernel(grid, ks, ms, hT, hS, DT_THERM)
   end do
   !$acc wait
   t1 = wall()
   ms_dc = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variant B: faithful CUDA C (one thread per column) ---------------
   do rep = 1, n_warm
      call launch_cu(1)
   end do
   t0 = wall()
   do rep = 1, n_reps
      call launch_cu(cu_sync)
   end do
   call launch_cu(1)
   t1 = wall()
   ms_cu = (t1 - t0)*1000.0_wp/real(n_reps + 1, wp)

   ! ---- variant C: warp-per-column CUDA C --------------------------------
   ! Built only with -DHAVE_WARP. This variant answers a question that only
   ! EXISTS if the iteration count diverges across a warp -- measure that first
   ! (`warp work-balance ratio` below), then decide. See README.
   have_warp = .false.
   ms_wp = 0.0_wp
   dmaxw = 0.0_wp; rmaxw = 0.0_wp; nbadw = 0
#ifdef HAVE_WARP
   have_warp = .true.
   do rep = 1, n_warm
      call launch_warp(1)
   end do
   t0 = wall()
   do rep = 1, n_reps
      call launch_warp(cu_sync)
   end do
   call launch_warp(1)
   t1 = wall()
   ms_wp = (t1 - t0)*1000.0_wp/real(n_reps + 1, wp)
#endif

   !$acc update self(ks%kd_int, kd_cu, kd_wp_, n_out, n_in)

   call compare(ks%kd_int, kd_cu, dmax, rmax, nbad)
   if (have_warp) call compare(ks%kd_int, kd_wp_, dmaxw, rmaxw, nbadw)

   call maybe_dump()

   kd_min = minval(ks%kd_int); kd_max = maxval(ks%kd_int); kd_sum = sum(ks%kd_int)
   nactive = count(n_in > 0)

   write (output_unit, '(a)') ''
   write (output_unit, '(a,f10.4,a)') '  kappa-shear (do concurrent)           : ', ms_dc, ' ms/rep'
   write (output_unit, '(a,f10.4,a)') '  CUDA C faithful (1 thread / column)   : ', ms_cu, ' ms/rep'
   if (have_warp) write (output_unit, '(a,f10.4,a)') '  CUDA C warp-per-column                : ', ms_wp, ' ms/rep'
   write (output_unit, '(a)') ''
   write (output_unit, '(a,f10.3,a)') '  ratio  dc / cuda_faithful             : ', ms_dc/ms_cu, ' x'
   if (have_warp) write (output_unit, '(a,f10.3,a)') '  ratio  dc / cuda_warp                 : ', ms_dc/ms_wp, ' x'

   write (output_unit, '(a)') ''
   write (output_unit, '(a)') '  --- kernel resources (cudaFuncGetAttributes, authoritative) ---'
   call ks_cuda_attrs(nregs, lmem, smem, maxtpb)
   write (output_unit, '(a,i5,a,i8,a,i7,a,i5)') '  faithful : regs=', nregs, &
      '  local=', lmem, ' B/thread  shared=', smem, ' B  maxTPB=', maxtpb
#ifdef HAVE_WARP
   call ks_warp_attrs(nregs, lmem, smem, maxtpb)
   write (output_unit, '(a,i5,a,i8,a,i7,a,i5)') '  warp     : regs=', nregs, &
      '  local=', lmem, ' B/thread  shared=', smem, ' B  maxTPB=', maxtpb
#endif

   call iter_report()

   write (output_unit, '(a)') ''
   write (output_unit, '(a,es12.5,a,es12.5)') '  DC vs CUDA-faithful  max|d| = ', dmax, &
      '   max rel = ', rmax
   call verdict(rmax, nbad)
   if (have_warp) then
      write (output_unit, '(a,es12.5,a,es12.5)') '  DC vs CUDA-warp      max|d| = ', dmaxw, &
         '   max rel = ', rmaxw
      call verdict(rmaxw, nbadw)
   end if

   write (output_unit, '(a)') ''
   write (output_unit, '(a,es14.6)') '  min kd_int : ', kd_min
   write (output_unit, '(a,es14.6)') '  max kd_int : ', kd_max
   write (output_unit, '(a,es14.6)') '  sum kd_int : ', kd_sum
   if (kd_min /= kd_min .or. kd_max /= kd_max) then
      write (output_unit, '(a)') '  sanity     : *** NaN -- results are garbage ***'
   else if (kd_min == 0.0_wp .and. kd_max == 0.0_wp) then
      write (output_unit, '(a)') '  sanity     : *** all-zero -- NO COLUMN MIXED. The state is'
      write (output_unit, '(a)') '               too stable; every column exits on the first'
      write (output_unit, '(a)') '               substep and this benchmark measures nothing.'
   else
      write (output_unit, '(a)') '  sanity     : OK (finite, non-zero)'
   end if
   write (output_unit, '(a)') repeat('=', 74)

contains

   subroutine launch_cu(sy)
      integer, intent(in) :: sy
      !$acc host_data use_device(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, &
      !$acc                      hT, hS, ms%wet_mask, ks%f_centre, kd_cu, tke_cu, n_out, n_in)
      call ks_cuda_launch(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, &
                          hT, hS, ms%wet_mask, ks%f_centre, kd_cu, tke_cu, &
                          n_out, n_in, nx, ny, nz, par, sy)
      !$acc end host_data
   end subroutine launch_cu

#ifdef HAVE_WARP
   subroutine launch_warp(sy)
      integer, intent(in) :: sy
      !$acc host_data use_device(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, &
      !$acc                      hT, hS, ms%wet_mask, ks%f_centre, kd_wp_, tke_wp_)
      call ks_warp_launch(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, &
                          hT, hS, ms%wet_mask, ks%f_centre, kd_wp_, tke_wp_, &
                          nx, ny, nz, par, sy)
      !$acc end host_data
   end subroutine launch_warp
#endif

   !! Reference dump for ks_native.cu (the no-Fortran, no-OpenACC driver).
   !!
   !! WHY: ks_native.cu re-implements build_state() in C++. If that mirror is
   !! even slightly wrong the native timing is measuring a DIFFERENT problem and
   !! the OpenACC-overhead question it exists to answer is unanswerable. So the
   !! native driver does not take the mirror on trust: it reads this file and
   !! compares its own INPUTS to these, element by element, before comparing any
   !! output. Inputs must be bit-identical; kd_cu must then be bit-identical too
   !! (same kernel, same data), and kd_dc reproduces the pre-existing ~2e-9 gap.
   !!
   !! Off unless KS_DUMP names a path -- it writes ~250 MB and default runs must
   !! stay untouched.
   subroutine maybe_dump()
      character(len=512) :: path
      integer :: st, un, ln
      call get_environment_variable('KS_DUMP', path, ln, st)
      if (st /= 0 .or. ln == 0) return
      open (newunit=un, file=trim(path), form='unformatted', access='stream', &
            status='replace', action='write')
      write (un) nx, ny, nz, land_pct
      write (un) ms%h_layer
      write (un) ms%u_face_x_layer
      write (un) ms%v_face_y_layer
      write (un) hT
      write (un) hS
      write (un) ms%wet_mask
      write (un) ks%f_centre
      write (un) ks%kd_int      ! do concurrent  output
      write (un) kd_cu          ! CUDA-via-Fortran output
      close (un)
      write (output_unit, '(a)') ''
      write (output_unit, '(a)') '  KS_DUMP: wrote inputs + kd_int(dc) + kd_cu to '//trim(path)
   end subroutine maybe_dump

   !! Build the physically realistic state described in the header.
   subroutine build_state()
      integer, parameter :: NZ_DEF_MAX = 512
      real(wp) :: depth, mld, us, vs, zt, zb, zm, hh, tt, ss, uu, vv
      real(wp) :: w(NZ_DEF_MAX), wsum, r, xr, yr, lat
      integer :: m, kg

      r = 1.18_wp   ! geometric stretch -> ~5 m surface layer at H = 4000 m
      wsum = 0.0_wp
      do m = 1, nz
         w(m) = r**(m - 1)
         wsum = wsum + w(m)
      end do
      do m = 1, nz
         w(m) = w(m)/wsum
      end do

      do j = 1, ny
         yr = real(j - NGHOST, wp)/real(max(nyp, 1), wp)      ! 0..1 south->north
         lat = -60.61_wp + 0.1_wp*real(j - NGHOST, wp)        ! gabight lat span
         do i = 1, nx
            xr = real(i - NGHOST, wp)/real(max(nxp, 1), wp)

            ! |f| from the planetary metric (coriolis_scheme = "planetary").
            ks%f_centre(i, j) = abs(2.0_wp*7.2921e-5_wp*sin(lat*3.141592653589793_wp/180.0_wp))

            ! Bathymetry: shelf -> abyssal, smooth, 200..4500 m.
            depth = 200.0_wp + 4300.0_wp* &
                    (0.5_wp*(1.0_wp + tanh(3.0_wp*(yr - 0.25_wp))))* &
                    (0.85_wp + 0.15_wp*sin(6.0_wp*xr))

            ! Land: a contiguous NW block, so the mask is spatially coherent
            ! (scattered land would understate warp divergence).
            if (land_pct > 0) then
               if (xr < 0.01_wp*real(land_pct, wp) .and. yr > 0.55_wp) then
                  ms%wet_mask(i, j) = 0.0_wp
               else
                  ms%wet_mask(i, j) = 1.0_wp
               end if
            else
               ms%wet_mask(i, j) = 1.0_wp
            end if

            ! Mixed-layer depth 40..100 m, and the jet strength that competes
            ! with the stratification at its base.
            mld = 40.0_wp + 60.0_wp*(0.5_wp*(1.0_wp + sin(5.0_wp*xr)*cos(4.0_wp*yr)))
            us = 0.25_wp + 0.60_wp*(0.5_wp*(1.0_wp + sin(7.0_wp*xr + 2.0_wp*yr)* &
                                            cos(3.0_wp*yr)))
            vs = 0.10_wp*sin(4.0_wp*xr)*cos(5.0_wp*yr)

            zt = 0.0_wp
            do m = 1, nz          ! m = local, surface-down
               hh = max(depth*w(m), 1.0e-3_wp)
               zb = zt + hh
               zm = 0.5_wp*(zt + zb)

               ! T: uniform in the ML, thermocline below (T_surf 14 -> T_bot 2).
               if (zm <= mld) then
                  tt = 14.0_wp
               else
                  tt = 2.0_wp + 12.0_wp*exp(-(zm - mld)/600.0_wp)
               end if
               ss = 35.0_wp - 0.5_wp*exp(-zm/300.0_wp)

               ! u: surface jet, shear concentrated at the ML base.
               uu = us*0.5_wp*(1.0_wp - tanh((zm - mld)/25.0_wp))
               vv = vs*0.5_wp*(1.0_wp - tanh((zm - mld)/25.0_wp))

               kg = nz + 1 - m    ! global bottom-up: kg = nz is the surface
               ms%h_layer(i, j, kg) = hh
               hT(i, j, kg) = tt*hh
               hS(i, j, kg) = ss*hh
               ms%u_face_x_layer(i, j, kg) = uu
               ms%v_face_y_layer(i, j, kg) = vv
               zt = zb
            end do
         end do
      end do
      ! Face arrays carry one extra row/column; fill the overhang by copy so
      ! the face average at i = nx / j = ny is not reading uninitialised memory.
      do kg = 1, nz
         do j = 1, ny
            ms%u_face_x_layer(nx + 1, j, kg) = ms%u_face_x_layer(nx, j, kg)
         end do
         do i = 1, nx
            ms%v_face_y_layer(i, ny + 1, kg) = ms%v_face_y_layer(i, ny, kg)
         end do
      end do
   end subroutine build_state

   !! Iteration-count distribution -- the divergence evidence. Counters come
   !! from the CUDA port (n_out = adaptive substeps, n_in = summed Picard
   !! iterations). The DC and CUDA kernels agree on results, so the CUDA
   !! distribution characterises both.
   subroutine iter_report()
      integer :: h_out(0:14), h_in(0:9), b, mx, mn, tot
      integer :: warp_max_sum, warp_mean_sum, nwarps, w0, lane, wmax, wsum
      real(wp) :: eff
      integer, allocatable :: flat(:)

      write (output_unit, '(a)') ''
      write (output_unit, '(a)') '  --- iteration-count distribution across columns (CUDA counters) ---'
      h_out = 0
      do j = 1, ny
         do i = 1, nx
            b = min(n_out(i, j), 14)
            h_out(b) = h_out(b) + 1
         end do
      end do
      write (output_unit, '(a)') '  outer adaptive substeps :'
      do b = 0, 14
         if (h_out(b) > 0) write (output_unit, '(a,i3,a,i9,a,f6.2,a)') '    n_out = ', b, &
            ' : ', h_out(b), '  (', 100.0_wp*real(h_out(b), wp)/real(ncol, wp), ' %)'
      end do

      h_in = 0
      mx = 0; mn = huge(1); tot = 0
      do j = 1, ny
         do i = 1, nx
            b = min(n_in(i, j)/10, 9)
            h_in(b) = h_in(b) + 1
            mx = max(mx, n_in(i, j)); mn = min(mn, n_in(i, j))
            tot = tot + n_in(i, j)
         end do
      end do
      write (output_unit, '(a)') '  total Picard iterations per column (binned by 10):'
      do b = 0, 9
         if (h_in(b) > 0) write (output_unit, '(a,i3,a,i3,a,i9,a,f6.2,a)') '    [', b*10, &
            ',', b*10 + 9, '] : ', h_in(b), '  (', &
            100.0_wp*real(h_in(b), wp)/real(ncol, wp), ' %)'
      end do
      write (output_unit, '(a,i0,a,i0,a,f8.2)') '    min = ', mn, '  max = ', mx, &
         '  mean = ', real(tot, wp)/real(ncol, wp)
      write (output_unit, '(a,i0,a,f6.2,a)') '    columns that mixed at all: ', nactive, &
         '  (', 100.0_wp*real(nactive, wp)/real(ncol, wp), ' %)'

      ! Warp-level work imbalance. A warp costs its SLOWEST lane, so the
      ! ceiling on what a perfectly-balanced scheme could save is
      ! sum(mean per warp) / sum(max per warp). This is a STATIC estimate from
      ! the iteration counts -- it is NOT a measured lane efficiency (see
      ! `make ncu` for that), and it assumes cost is proportional to Picard
      ! iterations, which is only approximately true.
      allocate (flat(ncol))
      flat = reshape(n_in, [ncol])
      nwarps = (ncol + 31)/32
      warp_max_sum = 0; warp_mean_sum = 0
      do w0 = 1, nwarps
         wmax = 0; wsum = 0
         do lane = (w0 - 1)*32 + 1, min(w0*32, ncol)
            wmax = max(wmax, flat(lane))
            wsum = wsum + flat(lane)
         end do
         warp_max_sum = warp_max_sum + wmax*32
         warp_mean_sum = warp_mean_sum + wsum
      end do
      eff = 0.0_wp
      if (warp_max_sum > 0) eff = real(warp_mean_sum, wp)/real(warp_max_sum, wp)
      write (output_unit, '(a,f6.3)') '    warp work-balance ratio (mean/max, static est.): ', eff
      write (output_unit, '(a)') '      1.0 = every lane in a warp iterates the same number of'
      write (output_unit, '(a)') '      times; lower = the warp waits on its slowest lane.'
      deallocate (flat)
   end subroutine iter_report

   subroutine compare(a, b, dmx, rmx, nb)
      real(wp), intent(in) :: a(:, :, :), b(:, :, :)
      real(wp), intent(out) :: dmx, rmx
      integer, intent(out) :: nb
      real(wp) :: sc
      integer :: ii, jj, kk
      dmx = 0.0_wp; rmx = 0.0_wp; nb = 0
      do kk = 1, nz + 1
         do jj = 1, ny
            do ii = 1, nx
               dmx = max(dmx, abs(a(ii, jj, kk) - b(ii, jj, kk)))
               sc = max(abs(a(ii, jj, kk)), abs(b(ii, jj, kk)))
               if (sc > 1.0e-30_wp) then
                  rmx = max(rmx, abs(a(ii, jj, kk) - b(ii, jj, kk))/sc)
                  if (abs(a(ii, jj, kk) - b(ii, jj, kk))/sc > 1.0e-12_wp) nb = nb + 1
               end if
            end do
         end do
      end do
   end subroutine compare

   !! ⚠ An ITERATIVE scheme is legitimately sensitive: a 1-ulp difference from
   !! FMA contraction can flip a convergence test, change an iteration count,
   !! and then diverge visibly. So a >1e-12 result here is NOT automatically a
   !! port bug -- but it is not automatically fine either. Diagnose it by
   !! comparing iteration counts, never by loosening the tolerance.
   subroutine verdict(r, nb)
      real(wp), intent(in) :: r
      integer, intent(in) :: nb
      if (r < 1.0e-12_wp) then
         write (output_unit, '(a)') '      agreement: OK (<1e-12 rel -> FMA contraction only)'
      else
         write (output_unit, '(a,i0,a)') '      agreement: *** ', nb, &
            ' cells >1e-12 rel -- PORT BUG or an iteration-count flip. DIAGNOSE.'
      end if
   end subroutine verdict

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

end program ks_bench
