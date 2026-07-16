!! MEKE step: `do concurrent` vs hand-written CUDA C.
!!
!! THE QUESTION: MEKE is structurally the barotropic substep all over again --
!! MANY SMALL 2-D LOOPS on a ~145k-cell domain, i.e. launch-bound, the regime
!! where every other benchmark in this repo shows `do concurrent` losing to
!! CUDA C. 16 `do concurrent` loops per `meke_step` at the one config that
!! turns MEKE on (`~/analysis_gebco/gabight_sph_meke_v100.nml`).
!!
!! The btstep benchmark found 1.42x lying free in portable Fortran from two
!! source changes. This benchmark tests both HYPOTHESES here:
!!
!!   H1 "SIGNATURE FIX is worth ~1.3x": ALREADY FALSIFIED BY INSPECTION --
!!      production MEKE already passes plain explicit-shape dummies bounded by
!!      plain integer dummies. There is no fix to apply. So instead this
!!      benchmark measures the COUNTERFACTUAL: variant DT re-writes the same
!!      kernels in the derived-type-component idiom btstep used
!!      (`m%meke(i,j)`, `metrics%areaT(i,j)`) to measure what MEKE's existing
!!      signature is WORTH. Prediction from btstep: small, because the cost
!!      scales with the number of distinct arrays per loop and MEKE's loops
!!      touch 4-11, not 19.
!!
!!   H2 "FUSION is worth ~1.09x": 16 loops -> 6 (meke_fused.F90). Tested.
!!
!! FIVE VARIANTS:
!!   DC            : production's kernels, verbatim. 16 loops.
!!   DC + DT       : same body, derived-type-component dummies. 16 loops.
!!                   The counterfactual for H1 -- SLOWER is the expected sign.
!!   DC + FUSED    : same body, 6 loops. Must be bit-identical.
!!   CUDA faithful : 16 kernels, one per DC loop -> measures CODEGEN.
!!   CUDA fused    : 6 kernels, the same fusion -> measures fusion in C.
!!   CUDA graph    : the 6 fused kernels in a cudaGraph -> the launch residual
!!                   `do concurrent` cannot reach.
!!
!! Usage: ./meke_bench [nx] [ny] [nz] [nreps] [nwarm]
!!   nx/ny are POINT COUNTS from the nml; the arrays are (nx+2*nghost).
!!   ./meke_bench            -> 473 297 30 (gabight_sph_meke_v100.nml)
!!   ./meke_bench 945 594 30 -> the 0.05 deg config
program meke_bench
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr, c_loc
   use constants, only: wp
   use meke_state
   use meke, only: meke_step_ext
   use meke_dt, only: meke_step_dt
   use meke_fused, only: meke_step_fused
   use meke_acc, only: meke_step_acc
   use meke_fused_acc, only: meke_step_fused_acc
   implicit none

   !! Mirrors `struct MekeArgs` in meke_kernel.cu -- FIELD ORDER IS LOAD-BEARING.
   type, bind(C) :: meke_args_t
      type(c_ptr) :: meke, kh_diff, le, ku, i_mass, depth_tot, bottom_fac2
      type(c_ptr) :: barotr_fac2, src, uflux, vflux, mass_ws, rd_ws
      type(c_ptr) :: sn_u_ws, sn_v_ws, ke_diss_ws
      type(c_ptr) :: u_bbl2, f_centre, gm_src, ke_diss_ext
      type(c_ptr) :: h_layer, rho_layer
      type(c_ptr) :: areaT, iareaT, idxT, idyT, dy_cu, dx_cv, idxCu, idyCv
      integer(c_int) :: nx, ny, nz
      real(c_double) :: dt, dtscale, cd_scale, cb, ct, min_gamma2, cdrag, uscale
      real(c_double) :: a_deform, a_rhines, a_eady, a_frict, a_grid
      real(c_double) :: bgsrc, gmcoeff, frcoeff, damping
      real(c_double) :: kh_bg, k4, khmeke_fac, khcoeff, visc_coeff_ku, rho0
      integer(c_int) :: backscatter
   end type meke_args_t

   interface
      subroutine meke_cuda_launch(a, mode) bind(C, name="meke_cuda_launch")
         import :: meke_args_t, c_int
         type(meke_args_t), intent(in) :: a
         integer(c_int), value :: mode
      end subroutine
      subroutine meke_cuda_graph_reset() bind(C, name="meke_cuda_graph_reset")
      end subroutine
   end interface

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30
   integer, parameter :: REPS_DEF = 200, WARM_DEF = 20, NGH = 3
   real(wp), parameter :: DXM = 8000.0_wp, DYM = 8000.0_wp, DT_THERMO = 1800.0_wp

   type(ocean_metrics_t) :: met
   type(multilayer_cgrid_state_t) :: ms
   type(ocean_gm_t) :: gm
   ! ⚠ SEPARATE OUTPUT STATE PER VARIANT. Sharing `meke` would mean only the
   ! last writer is checked and every other variant silently inherits
   ! "agreement OK" -- a trap that has already produced a phantom result in
   ! this repo. Each variant gets its own ocean_meke_t.
   type(ocean_meke_t) :: m_dc, m_dt, m_fu, m_cu, m_ac, m_fa
   type(meke_args_t) :: args
   real(wp), allocatable, target :: ke_diss_ext(:, :)
   !! Reference dumps for meke_native.cu (the native C++/CUDA driver), which has
   !! no Fortran and so cannot share this init. Written only when MEKE_DUMP is
   !! set, so the default run is byte-for-byte the benchmark it always was.
   real(wp), allocatable :: final_cu(:, :)
   character(len=8) :: dump_env
   integer :: dump_st
   real(wp) :: ms_dc, ms_dt, ms_fus, ms_cf, ms_cu, ms_cg, ms_acc, ms_fac
   real(wp) :: t0, t1, d_dt, d_fus, d_cuda, r_meke, d_acc, d_fac
   integer :: i, j, k, rep, nx, ny, nz, nxp, nyp, n_reps, n_warm

   nxp = iarg(1, NXP_DEF); nyp = iarg(2, NYP_DEF); nz = iarg(3, NZ_DEF)
   n_reps = iarg(4, REPS_DEF); n_warm = iarg(5, WARM_DEF)
   nx = nxp + 2*NGH; ny = nyp + 2*NGH

   allocate (met%areaT(nx, ny), met%iareaT(nx, ny), met%idxT(nx, ny), met%idyT(nx, ny))
   allocate (met%dy_cu(nx + 1, ny), met%dx_cv(nx, ny + 1))
   allocate (met%idxCu(nx + 1, ny), met%idyCv(nx, ny + 1))
   allocate (ms%h_layer(nx, ny, nz), ms%rho_layer(nx, ny, nz))
   allocate (gm%gm_src(nx, ny), ke_diss_ext(nx, ny))
   ms%nz_ml = nz; gm%is_init = .true.

   ! uniform-Cartesian metrics (the gabight spherical configs are close to this)
   met%areaT = DXM*DYM; met%iareaT = 1.0_wp/(DXM*DYM)
   met%idxT = 1.0_wp/DXM; met%idyT = 1.0_wp/DYM
   met%dy_cu = DYM; met%dx_cv = DXM
   met%idxCu = 1.0_wp/DXM; met%idyCv = 1.0_wp/DYM
   ! a stratified column + a GM PE-release source that varies in space, so the
   ! drag/closure branches see a real spread of E rather than one value
   do k = 1, nz
      do j = 1, ny
         do i = 1, nx
            ms%h_layer(i, j, k) = 4000.0_wp/real(nz, wp)
            ms%rho_layer(i, j, k) = 1025.0_wp + 5.0_wp*real(nz - k, wp)/real(nz, wp)
         end do
      end do
   end do
   do j = 1, ny
      do i = 1, nx
         gm%gm_src(i, j) = 1.0e-2_wp*exp(-((real(i - nx/2, wp)/real(max(nx/6, 1), wp))**2 &
                                           + (real(j - ny/2, wp)/real(max(ny/6, 1), wp))**2))
         ke_diss_ext(i, j) = -1.0e-3_wp   ! hvisc KE dissipation (<=0); frcoeff<0 ignores it
      end do
   end do

   call alloc_meke(m_dc); call alloc_meke(m_dt); call alloc_meke(m_fu); call alloc_meke(m_cu)
   call alloc_meke(m_ac); call alloc_meke(m_fa)
   call init_meke(m_dc); call init_meke(m_dt); call init_meke(m_fu); call init_meke(m_cu)
   call init_meke(m_ac); call init_meke(m_fa)

   write (output_unit, '(a)') repeat('=', 74)
   write (output_unit, '(a,i0,a,i0,a,i0,a)') '  config: ', nxp, ' x ', nyp, ' points (+', NGH, ' ghosts), gabight_sph_meke_v100.nml'
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0)') '  arrays: ', nx, ' x ', ny, ' = ', nx*ny, ' cells;  nz = ', nz
   write (output_unit, '(a,i0,a,i0,a)') '  ', n_reps, ' timed reps, ', n_warm, ' warm-up'
   write (output_unit, '(a,f0.1,a)') '  dt_thermo = ', DT_THERMO, ' s;  kh=100  gmcoeff=0.9  khcoeff=1.0  k4=-1 (biharm off)'
   write (output_unit, '(a)') repeat('=', 74)

   ! ⚠ DEEP COPY MUST BE IN THIS SCOPE (the one that OWNS the variables).
   ! Doing it in a helper maps the DUMMY; components never attach and every
   ! region falls back to implicit copies -- you then time PCIe, not the kernel.
   ! Parent struct first, then each payload.
   !$acc enter data copyin(met, ms, gm)
   !$acc enter data copyin(met%areaT, met%iareaT, met%idxT, met%idyT)
   !$acc enter data copyin(met%dy_cu, met%dx_cv, met%idxCu, met%idyCv)
   !$acc enter data copyin(ms%h_layer, ms%rho_layer)
   !$acc enter data copyin(gm%gm_src)
   !$acc enter data copyin(ke_diss_ext)
   !$acc enter data copyin(m_dc)
   call enter_meke_payload_dc()
   !$acc enter data copyin(m_dt)
   call enter_meke_payload_dt()
   !$acc enter data copyin(m_fu)
   call enter_meke_payload_fu()
   !$acc enter data copyin(m_cu)
   call enter_meke_payload_cu()
   !$acc enter data copyin(m_ac)
   call enter_meke_payload_ac()
   !$acc enter data copyin(m_fa)
   call enter_meke_payload_fa()

   ! ---- variant A: DC, production's kernels verbatim (16 loops) ----
   do rep = 1, n_warm
      call step_dc()
   end do
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call step_dc()
   end do
   !$acc wait
   t1 = wall()
   ms_dc = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variant B: DC + derived-type dummies (the H1 counterfactual) ----
   do rep = 1, n_warm
      call meke_step_dt(nx, ny, nz, m_dt, met, gm, ms, DT_THERMO, ke_diss_ext)
   end do
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call meke_step_dt(nx, ny, nz, m_dt, met, gm, ms, DT_THERMO, ke_diss_ext)
   end do
   !$acc wait
   t1 = wall()
   ms_dt = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variant C: DC + FUSED (16 loops -> 6) ----
   do rep = 1, n_warm
      call step_fused()
   end do
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call step_fused()
   end do
   !$acc wait
   t1 = wall()
   ms_fus = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variant C2: DC + `!$acc kernels async(1)` per loop, ONE drain ----
   ! The idiom production already uses on the barotropic substep and does NOT
   ! use on MEKE. If the gap is per-loop host sync, this removes it.
   do rep = 1, n_warm
      call step_acc()
   end do
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call step_acc()
   end do
   !$acc wait
   t1 = wall()
   ms_acc = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variant C3: FUSED + async. Do the two wins stack? ----
   do rep = 1, n_warm
      call step_fused_acc()
   end do
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call step_fused_acc()
   end do
   !$acc wait
   t1 = wall()
   ms_fac = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variants D/E/F: CUDA faithful / fused / graph ----
   ! Re-seed m_cu before each mode so all three start from the same state and
   ! the last one to run is the one we compare against DC.
   call fill_args(args, m_cu)
   call reseed_cu(); ms_cf = time_cuda(0, n_reps, n_warm, args)
   call reseed_cu(); ms_cu = time_cuda(1, n_reps, n_warm, args)
   call reseed_cu(); ms_cg = time_cuda(2, n_reps, n_warm, args)

   !$acc update self(m_dc%meke, m_dt%meke, m_fu%meke, m_cu%meke, m_ac%meke, m_fa%meke)

   ! Snapshot the CUDA variant's final field BEFORE the comparisons below, so the
   ! dump at the end cannot be affected by anything they do.
   allocate (final_cu(nx, ny)); final_cu = m_cu%meke

   d_dt = 0.0_wp; d_fus = 0.0_wp; d_cuda = 0.0_wp; r_meke = 0.0_wp
   d_acc = 0.0_wp; d_fac = 0.0_wp
   do j = 1, ny
      do i = 1, nx
         d_dt = max(d_dt, abs(m_dc%meke(i, j) - m_dt%meke(i, j)))
         d_fus = max(d_fus, abs(m_dc%meke(i, j) - m_fu%meke(i, j)))
         d_cuda = max(d_cuda, abs(m_dc%meke(i, j) - m_cu%meke(i, j)))
         d_acc = max(d_acc, abs(m_dc%meke(i, j) - m_ac%meke(i, j)))
         d_fac = max(d_fac, abs(m_dc%meke(i, j) - m_fa%meke(i, j)))
         r_meke = max(r_meke, abs(m_dc%meke(i, j)))
      end do
   end do

   write (output_unit, '(a,f10.4,a)') '  DC   (production, 16 loops)          : ', ms_dc, ' ms/step'
   write (output_unit, '(a,f10.4,a)') '  DC + derived-type dummies (16 loops) : ', ms_dt, ' ms/step'
   write (output_unit, '(a,f10.4,a)') '  DC + FUSED (6 loops)                 : ', ms_fus, ' ms/step'
   write (output_unit, '(a,f10.4,a)') '  DC + ACC async (16 loops, 1 drain)   : ', ms_acc, ' ms/step'
   write (output_unit, '(a,f10.4,a)') '  DC + FUSED + ACC async (6 loops)     : ', ms_fac, ' ms/step'
   write (output_unit, '(a,f10.4,a)') '  CUDA faithful (16 kernels)           : ', ms_cf, ' ms/step'
   write (output_unit, '(a,f10.4,a)') '  CUDA fused    ( 6 kernels)           : ', ms_cu, ' ms/step'
   write (output_unit, '(a,f10.4,a)') '  CUDA graph    ( 6 kern, 1 graph)     : ', ms_cg, ' ms/step'
   write (output_unit, '(a)') ''
   write (output_unit, '(a,f10.3,a)') '  dc-DT / dc  (what MEKEs EXISTING signature is worth) : ', ms_dt/ms_dc, ' x'
   write (output_unit, '(a,f10.3,a)') '  dc / cuda-faithful     (CODEGEN, equal launches)     : ', ms_dc/ms_cf, ' x'
   write (output_unit, '(a,f10.3,a)') '  dc / dc-fused          (FUSION, free in Fortran)     : ', ms_dc/ms_fus, ' x'
   write (output_unit, '(a,f10.3,a)') '  cuda-faithful / cuda-fused (FUSION in C)             : ', ms_cf/ms_cu, ' x'
   write (output_unit, '(a,f10.3,a)') '  dc-fused / cuda-fused  (Fortran vs C, equal fusion)  : ', ms_fus/ms_cu, ' x'
   write (output_unit, '(a,f10.3,a)') '  dc-fused / cuda-graph  <-- ALL a CUDA rewrite buys   : ', ms_fus/ms_cg, ' x'
   write (output_unit, '(a,f10.3,a)') '  dc / dc-acc            (ASYNC WRAPPER, free)         : ', ms_dc/ms_acc, ' x'
   write (output_unit, '(a,f10.3,a)') '  dc / dc-fused-acc      (BEST FORTRAN, free)          : ', ms_dc/ms_fac, ' x'
   write (output_unit, '(a,f10.3,a)') '  dc-fused-acc / cuda-fused (Fortran vs C, equal work) : ', ms_fac/ms_cu, ' x'
   write (output_unit, '(a,f10.3,a)') '  dc-fused-acc / cuda-graph <-- ALL a CUDA rewrite buys: ', ms_fac/ms_cg, ' x'
   write (output_unit, '(a,f10.3,a)') '  dc / cuda-graph        (TOTAL headroom)              : ', ms_dc/ms_cg, ' x'
   write (output_unit, '(a)') ''
   write (output_unit, '(a,es12.5,a,es12.5)') '  DC vs CUDA     max|d meke| : ', d_cuda, '   |meke|~', r_meke
   write (output_unit, '(a,es12.5,a)') '  DC vs DC-DT    max|d meke| : ', d_dt, '  (expect 0 -- same body)'
   ! The fused merges are single-assignment and every fold is pointwise within
   ! a thread, so fusion must be BIT-identical, not merely close. Anything
   ! non-zero here means a merge changed the answer and the speedup is void.
   write (output_unit, '(a,es12.5,a)') '  DC vs DC-FUSED max|d meke| : ', d_fus, '  (expect 0 -- merges are exact)'
   if (d_fus /= 0.0_wp) write (output_unit, '(a)') '  *** FUSION CHANGED THE ANSWER -- speedup is invalid ***'
   write (output_unit, '(a,es12.5,a)') '  DC vs DC-ACC   max|d meke| : ', d_acc, '  (expect 0 -- same loops, async)'
   write (output_unit, '(a,es12.5,a)') '  DC vs FUSED+ACC max|d meke|: ', d_fac, '  (expect 0)'
   if (d_dt /= 0.0_wp) write (output_unit, '(a)') '  *** DT VARIANT CHANGED THE ANSWER -- port bug ***'
   if (d_acc /= 0.0_wp) write (output_unit, '(a)') '  *** ACC WRAPPER CHANGED THE ANSWER -- ordering bug ***'
   if (d_fac /= 0.0_wp) write (output_unit, '(a)') '  *** FUSED+ACC CHANGED THE ANSWER ***'
   if (d_cuda <= 1.0e-12_wp*max(r_meke, 1.0e-30_wp)) then
      write (output_unit, '(a)') '  DC vs CUDA     agreement   : OK (FMA-contraction level)'
   else
      write (output_unit, '(a)') '  DC vs CUDA     agreement   : *** SUSPECT -- port bug? ***'
   end if
   write (output_unit, '(a)') repeat('=', 74)

   ! ---- reference dumps for the native C++/CUDA driver (MEKE_DUMP=1) --------
   ! Deliberately AFTER every comparison and print above: `init_meke(m_cu)` below
   ! clobbers m_cu, which is dead by this point. Emitting both the INIT and the
   ! FINAL fields is the point -- it lets meke_native.cu separate a host-side
   ! init difference (nvfortran's exp() vs glibc's) from a real difference on
   ! the kernel path. Checking only the final field could not tell them apart.
   call get_environment_variable('MEKE_DUMP', dump_env, status=dump_st)
   if (dump_st == 0 .and. len_trim(dump_env) > 0) then
      call dump2d('meke_ref_final.bin', final_cu)
      call init_meke(m_cu)     ! m_cu is dead: restore the pristine host init state
      call dump2d('meke_ref_init.bin', m_cu%meke)
      call dump2d('meke_ref_fcentre.bin', m_cu%f_centre)
      call dump2d('meke_ref_gmsrc.bin', gm%gm_src)
      write (output_unit, '(a)') '  wrote meke_ref_{final,init,fcentre,gmsrc}.bin for meke_native'
   end if

   call meke_cuda_graph_reset()

contains

   !! Raw little-endian float64, Fortran storage order -- meke_native.cu reads
   !! it back with fread and compares bit-for-bit.
   subroutine dump2d(fname, a)
      character(len=*), intent(in) :: fname
      real(wp), intent(in) :: a(:, :)
      integer :: u
      open (newunit=u, file=fname, form='unformatted', access='stream', status='replace')
      write (u) a
      close (u)
   end subroutine

   subroutine step_dc()
      call meke_step_ext(nx, ny, nz, m_dc, met, gm, ms, DT_THERMO, ke_diss_ext)
   end subroutine

   subroutine step_acc()
      call meke_step_acc(nx, ny, nz, m_ac, met, gm, ms, DT_THERMO, ke_diss_ext)
   end subroutine

   subroutine step_fused()
      call meke_step_fused(nx, ny, nz, DT_THERMO, m_fu%dtscale, &
                           m_fu%cd_scale, m_fu%cb, m_fu%ct, m_fu%min_gamma2, m_fu%cdrag, &
                           m_fu%uscale, m_fu%alpha_deform, m_fu%alpha_rhines, m_fu%alpha_eady, &
                           m_fu%alpha_frict, m_fu%alpha_grid, m_fu%bgsrc, m_fu%gmcoeff, &
                           m_fu%frcoeff, m_fu%damping, m_fu%kh, m_fu%k4, m_fu%khmeke_fac, &
                           m_fu%khcoeff, m_fu%backscatter, m_fu%visc_coeff_ku, gm%rho0, &
                           met%areaT, met%idxT, met%idyT, met%dy_cu, met%dx_cv, &
                           met%idxCu, met%idyCv, met%iareaT, &
                           ms%h_layer, ms%rho_layer, gm%gm_src, ke_diss_ext, m_fu%f_centre, &
                           m_fu%u_bbl2, m_fu%rd_ws, m_fu%sn_u_ws, m_fu%sn_v_ws, m_fu%ke_diss_ws, &
                           m_fu%i_mass, m_fu%depth_tot, m_fu%mass_ws, m_fu%bottom_fac2, &
                           m_fu%barotr_fac2, m_fu%src, m_fu%le, m_fu%uflux, m_fu%vflux, &
                           m_fu%kh_diff, m_fu%ku, m_fu%meke)
   end subroutine

   subroutine alloc_meke(s)
      type(ocean_meke_t), intent(inout) :: s
      allocate (s%meke(nx, ny), s%kh_diff(nx, ny), s%le(nx, ny), s%ku(nx, ny))
      allocate (s%i_mass(nx, ny), s%depth_tot(nx, ny))
      allocate (s%bottom_fac2(nx, ny), s%barotr_fac2(nx, ny), s%src(nx, ny))
      allocate (s%uflux(nx + 1, ny), s%vflux(nx, ny + 1), s%del2(nx, ny))
      allocate (s%mass_ws(nx, ny), s%u_bbl2(nx, ny), s%ke_diss_ws(nx, ny))
      allocate (s%rd_ws(nx, ny), s%f_centre(nx, ny))
      allocate (s%sn_u_ws(nx + 1, ny), s%sn_v_ws(nx, ny + 1))
      allocate (s%baro_hu(nx + 1, ny), s%baro_hv(nx, ny + 1))
      s%nx_total = nx; s%ny_total = ny; s%nz_ml = nz; s%is_init = .true.
   end subroutine

   !! Knobs from ~/analysis_gebco/gabight_sph_meke_v100.nml; everything else
   !! keeps ocean_meke_t's production default.
   subroutine init_meke(s)
      type(ocean_meke_t), intent(inout) :: s
      integer :: ii, jj
      s%enable = .true.
      s%gmcoeff = 0.9_wp
      s%khcoeff = 1.0_wp
      s%cdrag = 2.5e-3_wp
      s%kh = 100.0_wp
      do jj = 1, ny
         do ii = 1, nx
            ! a seeded eddy-energy field so sqrt/drag see a real spread
            s%meke(ii, jj) = 1.0e-3_wp + 1.0e-2_wp* &
               exp(-((real(ii - nx/3, wp)/real(max(nx/8, 1), wp))**2 &
                     + (real(jj - ny/3, wp)/real(max(ny/8, 1), wp))**2))
            ! |f| at cell centres, beta-plane -- as set_f_centre would fill it
            s%f_centre(ii, jj) = abs(-1.0e-4_wp + 2.0e-11_wp*real(jj - ny/2, wp)*DYM)
         end do
      end do
      s%kh_diff = 0.0_wp; s%le = 0.0_wp; s%ku = 0.0_wp
      s%i_mass = 0.0_wp; s%depth_tot = 0.0_wp
      s%bottom_fac2 = 0.0_wp; s%barotr_fac2 = 0.0_wp; s%src = 0.0_wp
      s%uflux = 0.0_wp; s%vflux = 0.0_wp; s%del2 = 0.0_wp
      s%mass_ws = 0.0_wp
      s%u_bbl2 = 0.0_wp          ! use_bbl_drag off -> stays 0, as production
      s%ke_diss_ws = 0.0_wp; s%rd_ws = 0.0_wp
      s%sn_u_ws = 0.0_wp; s%sn_v_ws = 0.0_wp
      s%baro_hu = 0.0_wp; s%baro_hv = 0.0_wp
   end subroutine

   subroutine reseed_cu()
      call init_meke(m_cu)
      !$acc update device(m_cu%meke, m_cu%kh_diff, m_cu%le, m_cu%ku, m_cu%i_mass)
      !$acc update device(m_cu%depth_tot, m_cu%bottom_fac2, m_cu%barotr_fac2, m_cu%src)
      !$acc update device(m_cu%uflux, m_cu%vflux, m_cu%mass_ws, m_cu%u_bbl2)
      !$acc update device(m_cu%ke_diss_ws, m_cu%rd_ws, m_cu%f_centre)
      !$acc update device(m_cu%sn_u_ws, m_cu%sn_v_ws)
   end subroutine

   subroutine enter_meke_payload_dc()
      !$acc enter data copyin(m_dc%meke, m_dc%kh_diff, m_dc%le, m_dc%ku, m_dc%i_mass)
      !$acc enter data copyin(m_dc%depth_tot, m_dc%bottom_fac2, m_dc%barotr_fac2, m_dc%src)
      !$acc enter data copyin(m_dc%uflux, m_dc%vflux, m_dc%del2, m_dc%mass_ws, m_dc%u_bbl2)
      !$acc enter data copyin(m_dc%ke_diss_ws, m_dc%rd_ws, m_dc%f_centre)
      !$acc enter data copyin(m_dc%sn_u_ws, m_dc%sn_v_ws, m_dc%baro_hu, m_dc%baro_hv)
   end subroutine
   subroutine enter_meke_payload_dt()
      !$acc enter data copyin(m_dt%meke, m_dt%kh_diff, m_dt%le, m_dt%ku, m_dt%i_mass)
      !$acc enter data copyin(m_dt%depth_tot, m_dt%bottom_fac2, m_dt%barotr_fac2, m_dt%src)
      !$acc enter data copyin(m_dt%uflux, m_dt%vflux, m_dt%del2, m_dt%mass_ws, m_dt%u_bbl2)
      !$acc enter data copyin(m_dt%ke_diss_ws, m_dt%rd_ws, m_dt%f_centre)
      !$acc enter data copyin(m_dt%sn_u_ws, m_dt%sn_v_ws, m_dt%baro_hu, m_dt%baro_hv)
   end subroutine
   subroutine enter_meke_payload_fu()
      !$acc enter data copyin(m_fu%meke, m_fu%kh_diff, m_fu%le, m_fu%ku, m_fu%i_mass)
      !$acc enter data copyin(m_fu%depth_tot, m_fu%bottom_fac2, m_fu%barotr_fac2, m_fu%src)
      !$acc enter data copyin(m_fu%uflux, m_fu%vflux, m_fu%del2, m_fu%mass_ws, m_fu%u_bbl2)
      !$acc enter data copyin(m_fu%ke_diss_ws, m_fu%rd_ws, m_fu%f_centre)
      !$acc enter data copyin(m_fu%sn_u_ws, m_fu%sn_v_ws, m_fu%baro_hu, m_fu%baro_hv)
   end subroutine
   subroutine enter_meke_payload_cu()
      !$acc enter data copyin(m_cu%meke, m_cu%kh_diff, m_cu%le, m_cu%ku, m_cu%i_mass)
      !$acc enter data copyin(m_cu%depth_tot, m_cu%bottom_fac2, m_cu%barotr_fac2, m_cu%src)
      !$acc enter data copyin(m_cu%uflux, m_cu%vflux, m_cu%del2, m_cu%mass_ws, m_cu%u_bbl2)
      !$acc enter data copyin(m_cu%ke_diss_ws, m_cu%rd_ws, m_cu%f_centre)
      !$acc enter data copyin(m_cu%sn_u_ws, m_cu%sn_v_ws, m_cu%baro_hu, m_cu%baro_hv)
   end subroutine

   subroutine enter_meke_payload_ac()
      !$acc enter data copyin(m_ac%meke, m_ac%kh_diff, m_ac%le, m_ac%ku, m_ac%i_mass)
      !$acc enter data copyin(m_ac%depth_tot, m_ac%bottom_fac2, m_ac%barotr_fac2, m_ac%src)
      !$acc enter data copyin(m_ac%uflux, m_ac%vflux, m_ac%del2, m_ac%mass_ws, m_ac%u_bbl2)
      !$acc enter data copyin(m_ac%ke_diss_ws, m_ac%rd_ws, m_ac%f_centre)
      !$acc enter data copyin(m_ac%sn_u_ws, m_ac%sn_v_ws, m_ac%baro_hu, m_ac%baro_hv)
   end subroutine
   subroutine enter_meke_payload_fa()
      !$acc enter data copyin(m_fa%meke, m_fa%kh_diff, m_fa%le, m_fa%ku, m_fa%i_mass)
      !$acc enter data copyin(m_fa%depth_tot, m_fa%bottom_fac2, m_fa%barotr_fac2, m_fa%src)
      !$acc enter data copyin(m_fa%uflux, m_fa%vflux, m_fa%del2, m_fa%mass_ws, m_fa%u_bbl2)
      !$acc enter data copyin(m_fa%ke_diss_ws, m_fa%rd_ws, m_fa%f_centre)
      !$acc enter data copyin(m_fa%sn_u_ws, m_fa%sn_v_ws, m_fa%baro_hu, m_fa%baro_hv)
   end subroutine

   subroutine step_fused_acc()
      call meke_step_fused_acc(nx, ny, nz, DT_THERMO, m_fa%dtscale, &
                           m_fa%cd_scale, m_fa%cb, m_fa%ct, m_fa%min_gamma2, m_fa%cdrag, &
                           m_fa%uscale, m_fa%alpha_deform, m_fa%alpha_rhines, m_fa%alpha_eady, &
                           m_fa%alpha_frict, m_fa%alpha_grid, m_fa%bgsrc, m_fa%gmcoeff, &
                           m_fa%frcoeff, m_fa%damping, m_fa%kh, m_fa%k4, m_fa%khmeke_fac, &
                           m_fa%khcoeff, m_fa%backscatter, m_fa%visc_coeff_ku, gm%rho0, &
                           met%areaT, met%idxT, met%idyT, met%dy_cu, met%dx_cv, &
                           met%idxCu, met%idyCv, met%iareaT, &
                           ms%h_layer, ms%rho_layer, gm%gm_src, ke_diss_ext, m_fa%f_centre, &
                           m_fa%u_bbl2, m_fa%rd_ws, m_fa%sn_u_ws, m_fa%sn_v_ws, m_fa%ke_diss_ws, &
                           m_fa%i_mass, m_fa%depth_tot, m_fa%mass_ws, m_fa%bottom_fac2, &
                           m_fa%barotr_fac2, m_fa%src, m_fa%le, m_fa%uflux, m_fa%vflux, &
                           m_fa%kh_diff, m_fa%ku, m_fa%meke)
   end subroutine

   !! Fill the C struct with DEVICE pointers. `host_data use_device` is the whole
   !! bridge: both toolchains then read the SAME device allocation.
   subroutine fill_args(a, s)
      type(meke_args_t), intent(out) :: a
      type(ocean_meke_t), intent(inout), target :: s
      !$acc host_data use_device(s%meke, s%kh_diff, s%le, s%ku, s%i_mass, s%depth_tot, &
      !$acc                      s%bottom_fac2, s%barotr_fac2, s%src, s%uflux, s%vflux, &
      !$acc                      s%mass_ws, s%rd_ws, s%sn_u_ws, s%sn_v_ws, s%ke_diss_ws, &
      !$acc                      s%u_bbl2, s%f_centre, gm%gm_src, ke_diss_ext, &
      !$acc                      ms%h_layer, ms%rho_layer, met%areaT, met%iareaT, &
      !$acc                      met%idxT, met%idyT, met%dy_cu, met%dx_cv, met%idxCu, met%idyCv)
      a%meke = c_loc(s%meke); a%kh_diff = c_loc(s%kh_diff); a%le = c_loc(s%le)
      a%ku = c_loc(s%ku); a%i_mass = c_loc(s%i_mass); a%depth_tot = c_loc(s%depth_tot)
      a%bottom_fac2 = c_loc(s%bottom_fac2); a%barotr_fac2 = c_loc(s%barotr_fac2)
      a%src = c_loc(s%src); a%uflux = c_loc(s%uflux); a%vflux = c_loc(s%vflux)
      a%mass_ws = c_loc(s%mass_ws); a%rd_ws = c_loc(s%rd_ws)
      a%sn_u_ws = c_loc(s%sn_u_ws); a%sn_v_ws = c_loc(s%sn_v_ws)
      a%ke_diss_ws = c_loc(s%ke_diss_ws); a%u_bbl2 = c_loc(s%u_bbl2)
      a%f_centre = c_loc(s%f_centre); a%gm_src = c_loc(gm%gm_src)
      a%ke_diss_ext = c_loc(ke_diss_ext)
      a%h_layer = c_loc(ms%h_layer); a%rho_layer = c_loc(ms%rho_layer)
      a%areaT = c_loc(met%areaT); a%iareaT = c_loc(met%iareaT)
      a%idxT = c_loc(met%idxT); a%idyT = c_loc(met%idyT)
      a%dy_cu = c_loc(met%dy_cu); a%dx_cv = c_loc(met%dx_cv)
      a%idxCu = c_loc(met%idxCu); a%idyCv = c_loc(met%idyCv)
      !$acc end host_data
      a%nx = nx; a%ny = ny; a%nz = nz
      a%dt = DT_THERMO; a%dtscale = s%dtscale
      a%cd_scale = s%cd_scale; a%cb = s%cb; a%ct = s%ct
      a%min_gamma2 = s%min_gamma2; a%cdrag = s%cdrag; a%uscale = s%uscale
      a%a_deform = s%alpha_deform; a%a_rhines = s%alpha_rhines; a%a_eady = s%alpha_eady
      a%a_frict = s%alpha_frict; a%a_grid = s%alpha_grid
      a%bgsrc = s%bgsrc; a%gmcoeff = s%gmcoeff; a%frcoeff = s%frcoeff; a%damping = s%damping
      a%kh_bg = s%kh; a%k4 = s%k4; a%khmeke_fac = s%khmeke_fac; a%khcoeff = s%khcoeff
      a%visc_coeff_ku = s%visc_coeff_ku; a%rho0 = gm%rho0
      a%backscatter = 0
      if (s%backscatter) a%backscatter = 1
   end subroutine

   real(wp) function time_cuda(mode, nr, nw, a) result(msout)
      integer, intent(in) :: mode, nr, nw
      type(meke_args_t), intent(in) :: a
      real(wp) :: ta, tb
      integer :: r
      do r = 1, nw
         call meke_cuda_launch(a, mode)
      end do
      ta = wall()
      do r = 1, nr
         call meke_cuda_launch(a, mode)
      end do
      tb = wall()
      msout = (tb - ta)*1000.0_wp/real(nr, wp)
   end function

   integer function iarg(kk, dflt)
      integer, intent(in) :: kk, dflt
      character(len=32) :: buf
      integer :: ln, st
      iarg = dflt
      if (command_argument_count() < kk) return
      call get_command_argument(kk, buf, ln, st)
      if (st /= 0 .or. ln == 0) return
      read (buf, *, iostat=st) iarg
      if (st /= 0) iarg = dflt
   end function

   function wall() result(t)
      real(wp) :: t
      integer(int64) :: cnt, rate
      call system_clock(cnt, rate)
      t = real(cnt, wp)/real(rate, wp)
   end function

end program meke_bench
