#include "directives.h"
!! DC-only driver for the ocean model's kappa-shear (JHL08) column kernel.
!!
!! COMPUTE is a bare `do concurrent (j, i)` (kappa_shear_column_kernel), one
!! thread per COLUMN with a fully sequential, ITERATIVE (Picard x adaptive
!! substep) solve inside. The device data layer is chosen ENTIRELY by
!! directives.h at compile time:
!!   -DDC_DATA_ACC  -> OpenACC  (nvfortran -acc=gpu -stdpar=gpu)   GPU
!!   -DDC_DATA_OMP  -> OpenMP target                               GPU (AMD/Intel too)
!!   (neither)      -> host: bare DC on the CPU (-stdpar=multicore/serial)
!! There is NO CUDA and NO nvcc in this binary -- the comparison-only CUDA
!! variant, its counters (n_out/n_in), and the ks_par_t bridge are all dropped.
!!
!! Cross-check (proves the macro'd data layer did not change the numbers):
!!   DC_DUMP=file  writes nx,ny,nz + kd_int  (a reference)
!!   DC_REF=file   reads that reference and reports max|diff| vs this run
!! Run once with DC_DATA_ACC (GPU) dumping a ref, then again on the CPU reading
!! it: agreement to FMA level means the OpenACC->host swap is numerically inert.
!!
!! NOTE (the old bench's "*** PORT BUG ***" line does NOT apply here): the old
!! DC-vs-CUDA metric is a reduction that can hide sign-paired/permuted
!! differences (documented false alarm -- see the old README). This driver's
!! cross-check is DC-acc vs DC-none/omp -- the SAME do-concurrent kernel across
!! data layers -- which is clean by construction.
!!
!! Usage: ./dc_main [nxp] [nyp] [nz] [nreps] [nwarm] [land_pct]
!!
!! nreps <= 0 means AUTO: warm up, time ONE rep, then pick nreps so the timed
!! loop lasts about KS_TARGET_MS (env, default 2000 ms). An nz sweep spans
!! ~2 orders of magnitude in cost per rep, so a fixed rep count either wastes
!! an hour at the deep end or measures clock noise at the shallow end.
!!
!! Every run ends with ONE machine-readable line for the sweep harness:
!!   RESULT kernel=kappa_shear variant=... nz=... ms=... ns_per_col=... kd_sum=...
!! Parse that, not the pretty table.
program dc_main
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use constants, only: wp, NZ_STACK_MAX
   use grid, only: hgrid_t
   use multilayer_cgrid_state, only: multilayer_cgrid_state_t
   use ocean_eos, only: EOS_VARIANT_LINEAR
   use ks, only: ocean_kappa_shear_t, kappa_shear_column_kernel, &
                 KS_VARIANT, KS_LANES
#ifdef KS_WITH_CUDA
   !! `make cmp` links the hand-written CUDA kernels into THIS driver, so both
   !! toolchains run on the SAME device allocation and the same state. There is
   !! no C++ main() and no hand-mirrored build_state to drift.
   use ks_bridge, only: ks_par_t, ks_cuda_launch, ks_cuda_attrs, ks_cuda_sync, &
                        ks_opt_launch, ks_opt_attrs
#endif
   implicit none

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30
   integer, parameter :: REPS_DEF = 200, WARM_DEF = 10, NGHOST = 3
   integer, parameter :: NZ_BUILD_MAX = 512   !! build_state's stretch-weight bound
   real(wp), parameter :: DT_THERM = 300.0_wp   !! &time_nml dt_fixed, therm ratio 1

   !! NOT named `grid`/`ks`: this program USEs modules of those names, and a
   !! module name is a global identifier, so reusing it for a local entity in
   !! the same scope is non-conforming. nvfortran and gfortran accept it; ifx
   !! rejects it outright (error #6450) and the whole Intel lane fails to build.
   type(hgrid_t) :: hgrid
   type(multilayer_cgrid_state_t) :: ms
   type(ocean_kappa_shear_t) :: ksh
   real(wp), allocatable :: hT(:, :, :), hS(:, :, :)

   real(wp) :: t0, t1, ms_dc, gib, kd_min, kd_max, kd_sum
   real(wp) :: target_ms, ms_probe, ns_per_col
   integer :: max_reps
   integer :: i, j, k, rep, nx, ny, nz, nxp, nyp, n_reps, n_warm
   integer :: ncol, nwet, land_pct, iu, ios
   integer(int64) :: it_out_tot, it_in_tot
   character(len=256) :: ref_path, dump_path, envbuf

   !! Every timed kernel in this binary is a "variant" and gets its own RESULT
   !! line, so one run yields the whole head-to-head and the harness never has
   !! to correlate separate processes (or separate device allocations).
   integer, parameter :: MAXVAR = 3
   !! vimpl = WHICH KERNEL (dc | cuda_faithful | cuda_opt). Kept distinct from
   !! the `variant` field, which is the Fortran ARRANGEMENT (faithful | block) --
   !! conflating the two would make the CSV unable to express "blocked DC vs
   !! faithful CUDA".
   character(len=16) :: vimpl(MAXVAR)
   character(len=8)  :: vdata(MAXVAR)
   real(wp) :: ms_v(MAXVAR), kdsum_v(MAXVAR), kdmin_v(MAXVAR), kdmax_v(MAXVAR)
   integer :: reps_v(MAXVAR), regs_v(MAXVAR), lmem_v(MAXVAR)
   integer(int64) :: ito_v(MAXVAR), iti_v(MAXVAR)
   integer :: nvar, iv
#ifdef KS_WITH_CUDA
#ifndef KS_OPT_NZMAX
#define KS_OPT_NZMAX 48
#endif
   type(ks_par_t), target :: par
   !! SEPARATE output state per toolchain. Sharing it means only the last
   !! writer is ever checked and the others silently inherit "agreement OK".
   real(wp), allocatable :: kd_cu(:, :, :), tke_cu(:, :, :)
   integer, allocatable :: n_out_a(:), n_in_a(:)
   integer :: nregs, lmem, smem, maxtpb
   real(wp) :: dmax_cu, rmax_cu, sc_cu
   integer(int64) :: nbad_cu
#endif

   nxp = iarg(1, NXP_DEF); nyp = iarg(2, NYP_DEF); nz = iarg(3, NZ_DEF)
   n_reps = iarg(4, REPS_DEF); n_warm = iarg(5, WARM_DEF); land_pct = iarg(6, 0)

   target_ms = 1000.0_wp
   call get_environment_variable('KS_TARGET_MS', envbuf, status=ios)
   if (ios == 0 .and. len_trim(envbuf) > 0) then
      read (envbuf, *, iostat=ios) target_ms
      if (ios /= 0 .or. target_ms <= 0.0_wp) target_ms = 1000.0_wp
   end if
   ! Ceiling on the auto rep count. Rep-to-rep spread on this kernel is small,
   ! so past ~50-100 reps you are buying decimal places nobody reads and paying
   ! for them across a few hundred sweep configurations.
   max_reps = 100
   call get_environment_variable('KS_MAX_REPS', envbuf, status=ios)
   if (ios == 0 .and. len_trim(envbuf) > 0) then
      read (envbuf, *, iostat=ios) max_reps
      if (ios /= 0 .or. max_reps < 1) max_reps = 100
   end if

   nx = nxp + 2*NGHOST; ny = nyp + 2*NGHOST
   if (nx < 1 .or. ny < 1 .or. nz < 2) then
      write (output_unit, '(a)') 'ERROR: need nxp,nyp >= 1 and nz >= 2'; stop 1
   end if
   if (nz + 1 > NZ_STACK_MAX) then
      write (output_unit, '(a,i0,a,i0,a)') 'ERROR: nz+1 = ', nz + 1, &
         ' exceeds NZ_STACK_MAX = ', NZ_STACK_MAX, &
         ' -- rebuild with make NZSTACK=<at least nz+1>'
      stop 1
   end if
   if (nz > NZ_BUILD_MAX) then
      write (output_unit, '(a,i0,a,i0)') 'ERROR: nz = ', nz, &
         ' exceeds the driver state builder bound NZ_BUILD_MAX = ', NZ_BUILD_MAX
      stop 1
   end if
   hgrid%nx_total = nx; hgrid%ny_total = ny
   hgrid%nx_phys = nxp; hgrid%ny_phys = nyp
   hgrid%nghost = NGHOST; hgrid%dx = 0.1_wp; hgrid%dy = 0.1_wp
   ncol = nx*ny

   allocate (ms%h_layer(nx, ny, nz))
   allocate (ms%u_face_x_layer(nx + 1, ny, nz), ms%v_face_y_layer(nx, ny + 1, nz))
   allocate (ms%wet_mask(nx, ny))
   allocate (hT(nx, ny, nz), hS(nx, ny, nz))
   allocate (ksh%f_centre(nx, ny))
   allocate (ksh%kd_int(nx, ny, nz + 1), ksh%tke_int(nx, ny, nz + 1))
#ifdef KS_COUNTERS
   allocate (ksh%it_outer(nx, ny), ksh%it_inner(nx, ny))
   ksh%it_outer = 0; ksh%it_inner = 0
#endif
#ifdef KS_WITH_CUDA
   allocate (kd_cu(nx, ny, nz + 1), tke_cu(nx, ny, nz + 1))
   allocate (n_out_a(nx*ny), n_in_a(nx*ny))
   kd_cu = 0.0_wp; tke_cu = 0.0_wp; n_out_a = 0; n_in_a = 0
#endif
   ms%nz_ml = nz

   gib = 8.0_wp*real(nx, wp)*real(ny, wp)*real(nz, wp)*8.0_wp/(1024.0_wp**3)

   call build_state()

   ksh%enable = .true.
   ksh%eos%variant = EOS_VARIANT_LINEAR
   ksh%eos%rho0 = 1035.0_wp
   ksh%eos%alpha_T = 0.2_wp     ! &ocean_ic_nml alpha_T -> ocean_state.F90:474
   ksh%eos%beta_S = 7.6e-4_wp   ! ocean_eos.F90 default (no nml override)
   ksh%rho0 = 1035.0_wp
   ksh%kd_int = 0.0_wp; ksh%tke_int = 0.0_wp

   nwet = count(ms%wet_mask > 0.0_wp)
   write (output_unit, '(a)') repeat('=', 74)
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  domain: ', nxp, ' x ', nyp, &
      ' x ', nz, ' interior (+', NGHOST, ' ghosts/side)'
   write (output_unit, '(a,i0,a,i0,a,i0,a)') '  arrays: ', nx, ' x ', ny, &
      ' -> ', ncol, ' COLUMNS (the parallel width)'
   write (output_unit, '(a,i0,a,f5.1,a)') '  wet columns: ', nwet, '  (', &
      100.0_wp*real(nwet, wp)/real(ncol, wp), ' %)'
   write (output_unit, '(a,i0,a,i0,a)') '  NZ_STACK_MAX = ', NZ_STACK_MAX, &
      ' (production); nz = ', nz
   if (n_reps > 0) then
      write (output_unit, '(3a,i0,a,i0,a)') '  DATA layer: ', DC_DATA_NAME, &
         '   (reps ', n_reps, ', warm ', n_warm, ')'
   else
      write (output_unit, '(3a,i0,a)') '  DATA layer: ', DC_DATA_NAME, &
         '   (reps auto, warm ', n_warm, ')'
   end if
   write (output_unit, '(a)') repeat('=', 74)

   ! ---- map the DC working set (no-ops when the DATA layer is 'host'). The
   ! deep copy MUST happen where the variables are owned: parent struct first,
   ! then each allocatable payload. The comparison-only CUDA arrays (kd_cu,
   ! tke_cu, kd_wp_, tke_wp_, n_out, n_in) are gone with the CUDA variant.
   DC_ENTER_IN(ms)
   DC_ENTER_IN(ms%h_layer)
   DC_ENTER_IN(ms%u_face_x_layer)
   DC_ENTER_IN(ms%v_face_y_layer)
   DC_ENTER_IN(ms%wet_mask)
   DC_ENTER_IN(ksh)
   DC_ENTER_IN(ksh%f_centre)
   DC_ENTER_CREATE(ksh%kd_int)
   DC_ENTER_CREATE(ksh%tke_int)
   DC_ENTER_IN(hT)
   DC_ENTER_IN(hS)
#ifdef KS_COUNTERS
   DC_ENTER_CREATE(ksh%it_outer)
   DC_ENTER_CREATE(ksh%it_inner)
#endif
#ifdef KS_WITH_CUDA
   ! DC_ENTER_IN (not CREATE): these arrive on the device already zeroed, which
   ! the iteration counters rely on.
   DC_ENTER_IN(kd_cu)
   DC_ENTER_IN(tke_cu)
   DC_ENTER_IN(n_out_a)
   DC_ENTER_IN(n_in_a)

   ! Knobs across the bind(C) boundary. Field ORDER is the contract -- it must
   ! match ks_par_t in ks_bridge.F90 and struct KsPar in both .cu files, or the
   ! kernel gets garbage knobs and still runs happily.
   par%dt = DT_THERM
   par%ri_crit = ksh%ri_crit
   par%shearmix_rate = ksh%shearmix_rate
   par%fri_curvature = ksh%fri_curvature
   par%c_n = ksh%c_n
   par%c_s = ksh%c_s
   par%lambda = ksh%lambda
   par%lz_rescale = ksh%lz_rescale
   par%kappa_0 = ksh%kappa_0
   par%kappa_seed = ksh%kappa_seed
   par%kappa_trunc = ksh%kappa_trunc
   par%tke_bg = ksh%tke_bg
   par%tol_err = ksh%tol_err
   par%src_max_chg = ksh%src_max_chg
   par%vel_underflow = ksh%vel_underflow
   par%rho0 = ksh%rho0
   par%max_inner_it = ksh%max_inner_it
   par%max_substep_it = ksh%max_substep_it
   par%eos_variant = ksh%eos%variant
   par%eos_rho0 = ksh%eos%rho0
   par%eos_alpha_T = ksh%eos%alpha_T
   par%eos_beta_S = ksh%eos%beta_S
#endif

   ! ---- which kernels this binary can time --------------------------------
   nvar = 1
   vimpl(1) = 'dc'
   vdata(1) = DC_DATA_NAME
#ifdef KS_WITH_CUDA
   nvar = 2
   vimpl(2) = 'cuda_faithful'
   vdata(2) = 'cuda'
   ! The optimised kernel sizes its frame to KS_OPT_NZMAX and REFUSES to launch
   ! past it rather than corrupt results, so it is offered only when it fits.
   if (nz + 1 <= KS_OPT_NZMAX) then
      nvar = 3
      vimpl(3) = 'cuda_opt'
      vdata(3) = 'cuda'
   else
      write (output_unit, '(a,i0,a,i0,a)') '  NOTE: cuda_opt skipped -- nz+1 = ', &
         nz + 1, ' > KS_OPT_NZMAX = ', KS_OPT_NZMAX, ' (rebuild with OPT_NZMAX >= nz+1)'
   end if
#endif

   do iv = 1, nvar
      call time_variant(iv)
   end do

   ! Variant 1 (do concurrent) is the reference for the dump / cross-check.
   ms_dc = ms_v(1)
   kd_min = kdmin_v(1); kd_max = kdmax_v(1); kd_sum = kdsum_v(1)
   it_out_tot = ito_v(1); it_in_tot = iti_v(1)

   ! ---- per-variant report -------------------------------------------------
   do iv = 1, nvar
      write (output_unit, '(2a,f12.4,a)') '  ', adjustl(vimpl(iv)//'              '), &
         ms_v(iv), ' ms/rep'
   end do
   if (nvar > 1) then
      write (output_unit, '(a)') ''
      do iv = 2, nvar
         write (output_unit, '(3a,f10.3,a)') '  ratio  dc / ', trim(vimpl(iv)), &
            '  : ', ms_v(1)/max(ms_v(iv), 1.0e-30_wp), ' x'
      end do
      write (output_unit, '(a)') ''
      do iv = 2, nvar
         write (output_unit, '(3a,i5,a,i9,a)') '  ', trim(vimpl(iv)), &
            ' : regs=', regs_v(iv), '  local=', lmem_v(iv), ' B/thread'
      end do
   end if

   write (output_unit, '(a)') ''
   write (output_unit, '(a,es14.6)') '  min kd_int : ', kd_min
   write (output_unit, '(a,es14.6)') '  max kd_int : ', kd_max
   write (output_unit, '(a,es14.6)') '  sum kd_int : ', kd_sum
   if (kd_min /= kd_min .or. kd_max /= kd_max) then
      write (output_unit, '(a)') '  sanity     : *** NaN -- results are garbage ***'; stop 2
   else if (kd_min == 0.0_wp .and. kd_max == 0.0_wp) then
      write (output_unit, '(a)') '  sanity     : *** all-zero -- NO COLUMN MIXED (state too stable) ***'; stop 2
   else
      write (output_unit, '(a)') '  sanity     : OK (finite, non-zero)'
   end if
#ifdef KS_COUNTERS
   write (output_unit, '(a,i0,a,f8.3,a)') '  substeps   : ', it_out_tot, &
      '  (', real(it_out_tot, wp)/real(max(nwet, 1), wp), ' per wet column)'
   write (output_unit, '(a,i0,a,f8.3,a)') '  Picard its : ', it_in_tot, &
      '  (', real(it_in_tot, wp)/real(max(nwet, 1), wp), ' per wet column)'
#endif
#ifdef KS_WITH_CUDA
   ! Agreement is checked on the FULL field, not on the sum: a sum is a
   ! reduction and reductions hide sign-paired and permuted differences.
   ! Iteration totals are compared too -- equal totals mean the two kernels did
   ! the same WORK, which is what a timing ratio needs in order to mean anything.
   dmax_cu = 0.0_wp; rmax_cu = 0.0_wp; nbad_cu = 0_int64
   do k = 1, nz + 1
      do j = 1, ny
         do i = 1, nx
            dmax_cu = max(dmax_cu, abs(ksh%kd_int(i, j, k) - kd_cu(i, j, k)))
            sc_cu = max(abs(ksh%kd_int(i, j, k)), abs(kd_cu(i, j, k)))
            if (sc_cu > 1.0e-30_wp) then
               rmax_cu = max(rmax_cu, abs(ksh%kd_int(i, j, k) - kd_cu(i, j, k))/sc_cu)
               if (abs(ksh%kd_int(i, j, k) - kd_cu(i, j, k))/sc_cu > 1.0e-12_wp) &
                  nbad_cu = nbad_cu + 1
            end if
         end do
      end do
   end do
   write (output_unit, '(a)') ''
   write (output_unit, '(a,es12.5,a,es12.5,a,i0,a)') '  DC vs CUDA: max|d| ', &
      dmax_cu, '  max rel ', rmax_cu, '  (', nbad_cu, ' cells > 1e-12 rel)'
   write (output_unit, '(a,i0,a,i0)') '  iterations: DC substeps ', it_out_tot, &
      '  CUDA substeps ', ito_v(nvar)
   write (output_unit, '(a,i0,a,i0)') '             DC Picard   ', it_in_tot, &
      '  CUDA Picard   ', iti_v(nvar)
#endif

   ! ---- optional reference dump / cross-check ------------------------------
   call get_environment_variable('DC_DUMP', dump_path, status=ios)
   if (ios == 0 .and. len_trim(dump_path) > 0) then
      open (newunit=iu, file=trim(dump_path), access='stream', form='unformatted', status='replace')
      write (iu) nx, ny, nz
      write (iu) ksh%kd_int
      close (iu)
      write (output_unit, '(3a)') '  wrote ref       : ', trim(dump_path), ' (nx,ny,nz, kd_int)'
   end if

   call get_environment_variable('DC_REF', ref_path, status=ios)
   if (ios == 0 .and. len_trim(ref_path) > 0) call compare_ref(trim(ref_path))

   write (output_unit, '(a)') repeat('=', 74)

   ! ---- ONE machine-readable line PER VARIANT, which tools/ks_sweep.sh parses.
   ! Every field the sweep needs to identify and normalise the measurement,
   ! including kd_sum/min/max as a free correctness fingerprint: any variant,
   ! compiler or arrangement that quietly changes the answers shows up as a
   ! changed fingerprint in the CSV without a separate verification run.
   do iv = 1, nvar
      call emit_result(iv)
   end do

contains

   !! Warm up, size the rep count, then time one variant. Same window for both
   !! toolchains: N launches followed by ONE device synchronise.
   subroutine time_variant(iv)
      integer, intent(in) :: iv
      integer :: r, nr
      real(wp) :: a, b, pms
      do r = 1, n_warm
         call run_variant(iv)
      end do
      call wait_variant(iv)
      nr = n_reps
      if (nr <= 0) then
         a = wall()
         call run_variant(iv)
         call wait_variant(iv)
         b = wall()
         pms = (b - a)*1000.0_wp
         if (pms <= 0.0_wp) then
            nr = max_reps
         else
            nr = max(3, min(max_reps, int(target_ms/pms) + 1))
         end if
         write (output_unit, '(3a,f10.4,a,i0,a,f8.1,a,i0,a)') '  auto-reps [', &
            trim(vimpl(iv)), ']: probe ', pms, ' ms/rep -> reps = ', nr, &
            ' (target ', target_ms, ' ms, cap ', max_reps, ')'
      end if
      a = wall()
      do r = 1, nr
         call run_variant(iv)
      end do
      call wait_variant(iv)
      b = wall()
      ms_v(iv) = (b - a)*1000.0_wp/real(nr, wp)
      reps_v(iv) = nr
      call collect_variant(iv)
   end subroutine time_variant

   subroutine run_variant(iv)
      integer, intent(in) :: iv
      if (iv == 1) then
         call kappa_shear_column_kernel(hgrid, ksh, ms, hT, hS, DT_THERM)
#ifdef KS_WITH_CUDA
      else if (iv == 2) then
         call launch_cu(0)
      else
         call launch_opt(0)
#endif
      end if
   end subroutine run_variant

   subroutine wait_variant(iv)
      integer, intent(in) :: iv
      if (iv == 1) then
         DC_WAIT
#ifdef KS_WITH_CUDA
      else
         call ks_cuda_sync()
#endif
      end if
   end subroutine wait_variant

   !! Pull this variant's outputs back and record its stats. The CUDA kernels
   !! count iterations unconditionally; the Fortran side needs -DKS_COUNTERS.
   subroutine collect_variant(iv)
      integer, intent(in) :: iv
      ito_v(iv) = 0_int64; iti_v(iv) = 0_int64
      regs_v(iv) = 0; lmem_v(iv) = 0
      if (iv == 1) then
         DC_UPDATE_SELF(ksh%kd_int)
         kdmin_v(iv) = minval(ksh%kd_int)
         kdmax_v(iv) = maxval(ksh%kd_int)
         kdsum_v(iv) = sum(ksh%kd_int)
#ifdef KS_COUNTERS
         DC_UPDATE_SELF(ksh%it_outer)
         DC_UPDATE_SELF(ksh%it_inner)
         ito_v(iv) = sum(int(ksh%it_outer, int64))
         iti_v(iv) = sum(int(ksh%it_inner, int64))
#endif
#ifdef KS_WITH_CUDA
      else
         DC_UPDATE_SELF(kd_cu)
         DC_UPDATE_SELF(n_out_a)
         DC_UPDATE_SELF(n_in_a)
         kdmin_v(iv) = minval(kd_cu)
         kdmax_v(iv) = maxval(kd_cu)
         kdsum_v(iv) = sum(kd_cu)
         ito_v(iv) = sum(int(n_out_a, int64))
         iti_v(iv) = sum(int(n_in_a, int64))
         if (iv == 2) then
            call ks_cuda_attrs(nregs, lmem, smem, maxtpb)
         else
            call ks_opt_attrs(nregs, lmem, smem, maxtpb)
         end if
         regs_v(iv) = nregs
         lmem_v(iv) = lmem
#endif
      end if
   end subroutine collect_variant

#ifdef KS_WITH_CUDA
   !! host_data use_device hands the CUDA launcher the SAME device pointers the
   !! `do concurrent` kernel just used. No cudaMalloc, no second allocation, no
   !! host<->device copy in the timed window.
   subroutine launch_cu(sy)
      integer, intent(in) :: sy
      !$acc host_data use_device(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, &
      !$acc                      hT, hS, ms%wet_mask, ksh%f_centre, kd_cu, tke_cu, &
      !$acc                      n_out_a, n_in_a)
      call ks_cuda_launch(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, &
                          hT, hS, ms%wet_mask, ksh%f_centre, kd_cu, tke_cu, &
                          n_out_a, n_in_a, nx, ny, nz, par, sy)
      !$acc end host_data
   end subroutine launch_cu

   subroutine launch_opt(sy)
      integer, intent(in) :: sy
      !$acc host_data use_device(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, &
      !$acc                      hT, hS, ms%wet_mask, ksh%f_centre, kd_cu, tke_cu, &
      !$acc                      n_out_a, n_in_a)
      call ks_opt_launch(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, &
                         hT, hS, ms%wet_mask, ksh%f_centre, kd_cu, tke_cu, &
                         n_out_a, n_in_a, nx, ny, nz, par, sy)
      !$acc end host_data
   end subroutine launch_opt
#endif

   subroutine emit_result(iv)
      integer, intent(in) :: iv
      real(wp) :: nspc
      nspc = ms_v(iv)*1.0e6_wp/real(max(ncol, 1), wp)
      write (output_unit, '(a)', advance='no') 'RESULT kernel=kappa_shear'
      write (output_unit, '(2a)', advance='no') ' impl=', trim(vimpl(iv))
      if (iv == 1) then
         write (output_unit, '(2a)', advance='no') ' variant=', trim(KS_VARIANT)
         write (output_unit, '(a,i0)', advance='no') ' lanes=', KS_LANES
      else
         write (output_unit, '(a)', advance='no') ' variant=cuda lanes=1'
      end if
      write (output_unit, '(2a)', advance='no') ' data=', trim(vdata(iv))
      write (output_unit, '(a,i0)', advance='no') ' nzstack=', NZ_STACK_MAX
#ifdef KS_BOUNDED_COPY
      write (output_unit, '(a)', advance='no') ' bcopy=1'
#else
      write (output_unit, '(a)', advance='no') ' bcopy=0'
#endif
      if (iv == 1) then
#ifdef KS_COUNTERS
         write (output_unit, '(a)', advance='no') ' counters=1'
#else
         write (output_unit, '(a)', advance='no') ' counters=0'
#endif
      else
         write (output_unit, '(a)', advance='no') ' counters=1'
      end if
      write (output_unit, '(a,i0)', advance='no') ' nxp=', nxp
      write (output_unit, '(a,i0)', advance='no') ' nyp=', nyp
      write (output_unit, '(a,i0)', advance='no') ' nx=', nx
      write (output_unit, '(a,i0)', advance='no') ' ny=', ny
      write (output_unit, '(a,i0)', advance='no') ' ncols=', ncol
      write (output_unit, '(a,i0)', advance='no') ' nwet=', nwet
      write (output_unit, '(a,i0)', advance='no') ' nz=', nz
      write (output_unit, '(a,i0)', advance='no') ' reps=', reps_v(iv)
      write (output_unit, '(a,i0)', advance='no') ' warm=', n_warm
      write (output_unit, '(a,i0)', advance='no') ' land_pct=', land_pct
      write (output_unit, '(2a)', advance='no') ' ms=', trim(rs(ms_v(iv)))
      write (output_unit, '(2a)', advance='no') ' ns_per_col=', trim(rs(nspc))
      write (output_unit, '(2a)', advance='no') ' kd_sum=', trim(rs(kdsum_v(iv)))
      write (output_unit, '(2a)', advance='no') ' kd_min=', trim(rs(kdmin_v(iv)))
      write (output_unit, '(2a)', advance='no') ' kd_max=', trim(rs(kdmax_v(iv)))
      write (output_unit, '(a,i0)', advance='no') ' it_outer=', ito_v(iv)
      write (output_unit, '(a,i0)', advance='no') ' it_inner=', iti_v(iv)
      write (output_unit, '(a,i0)', advance='no') ' regs=', regs_v(iv)
      write (output_unit, '(a,i0)') ' local_b=', lmem_v(iv)
   end subroutine emit_result

   !! real -> a compact, whitespace-free token for the RESULT line. The line is
   !! parsed by splitting on blanks, so no field may contain one -- which rules
   !! out plain `esNN.N` output with its leading pad.
   function rs(v) result(s)
      real(wp), intent(in) :: v
      character(len=26) :: s
      write (s, '(es17.10)') v
      s = adjustl(s)
   end function rs

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
      allocate (ref(rnx, rny, rnz + 1)); read (u) ref; close (u)
      dmax = 0.0_wp; rmax = 0.0_wp; nbad = 0
      do k = 1, nz + 1
         do j = 1, ny
            do i = 1, nx
               dmax = max(dmax, abs(ksh%kd_int(i, j, k) - ref(i, j, k)))
               sc = max(abs(ksh%kd_int(i, j, k)), abs(ref(i, j, k)))
               if (sc > 1.0e-30_wp) then
                  rmax = max(rmax, abs(ksh%kd_int(i, j, k) - ref(i, j, k))/sc)
                  if (abs(ksh%kd_int(i, j, k) - ref(i, j, k))/sc > 1.0e-12_wp) nbad = nbad + 1
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

   !! Build the physically realistic state described in the old bench header:
   !! z* hgrid, a surface mixed layer straddling Ri_crit, a surface-intensified
   !! jet, and a per-column iteration count that actually varies. VERBATIM from
   !! ks_bench.F90's build_state.
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
            ksh%f_centre(i, j) = abs(2.0_wp*7.2921e-5_wp*sin(lat*3.141592653589793_wp/180.0_wp))

            ! Bathymetry: shelf -> abyssal, smooth, 200..4500 m.
            depth = 200.0_wp + 4300.0_wp* &
                    (0.5_wp*(1.0_wp + tanh(3.0_wp*(yr - 0.25_wp))))* &
                    (0.85_wp + 0.15_wp*sin(6.0_wp*xr))

            ! Land: a contiguous NW block, so the mask is spatially coherent.
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
