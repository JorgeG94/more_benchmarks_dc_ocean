!! Driver for the ocean model's PRODUCTION continuity-PPM barotropic flux kernel:
!! `do concurrent` vs a hand-inlined control vs a faithful CUDA C port.
!!
!! WHY: the HLL flux benchmark next door found `do concurrent` 1.4x slower than
!! CUDA C from two independent compiler defects (RESUME §1, §5a, §5b). This asks
!! whether the ocean continuity path pays the same tax. Static analysis predicts
!! NO, for two reasons:
!!
!!   bug 1 (lost auto-collapse): needs a grid%-bounded explicit-shape DUMMY.
!!     This kernel's arrays are derived-type COMPONENTS (bs%h) -> not triggered;
!!     -Minfo confirms all five 2-D loops auto-collapse.
!!
!!   bug 2 (lost CSE across the inline boundary): needs the CALLEE to do its own
!!     array indexing. This kernel HOISTS every read into a scalar first
!!     (h0 = bs%h(i,j)) and hands the PPM helpers SCALARS -> not triggered.
!!
!! Three variants, so the two questions separate:
!!   DC   : production, verbatim.
!!   FLAT : helpers hand-inlined. Tests "do the CALLS cost anything?"
!!   CUDA : faithful C port. Tests "does nvfortran's CODEGEN cost anything?"
!! FLAT alone cannot answer the second -- that is why the CUDA port exists.
!!
!! ⚠ SCOPE: two kernels is not a law. Nothing here predicts the barotropic
!! subset, ePBL, or tracer advection -- they differ in register pressure,
!! occupancy, call idiom and stencil width, which is exactly what decides this.
!! What transfers is the METHOD, not the verdict. See README.md.
!!
!! STATE: all-wet (wet_T = 1), so ppm_mirror_h and the slope-flatten multiply by
!! 1 and the land branches never fire -- same choice as flux_bench.F90, and it
!! measures the kernel rather than the coast. use_ppm_limit_pos = .false.
!! (production default), so ppm_limit_pos never executes.
!!
!! MEMORY: -gpu=mem:separate + MANUAL DEEP COPY. The state is nested allocatable
!! derived types, so each payload needs its own `enter data` after its parent's
!! (`copyin(cont%h_face_left_x)` then `create(cont%h_face_left_x%data)`).
!! Without those, -Minfo's "Generating implicit copyin(...)" fires on EVERY call
!! and the benchmark would time PCIe, not the kernel. Both toolchains then read
!! the SAME device allocation (via host_data use_device), so the only variable
!! is who generated the code.
program continuity_bench
   use, intrinsic :: iso_fortran_env, only: real64, int64, output_unit
   use, intrinsic :: iso_c_binding, only: c_double, c_int
   use constants, only: wp
   use grid, only: hgrid_t
   use ocean_metrics, only: ocean_metrics_t
   use barotropic_cgrid_state, only: barotropic_cgrid_state_t
   use continuity, only: continuity_t, continuity_compute_fluxes_barotropic
   use continuity_flat, only: continuity_compute_fluxes_barotropic_flat
   use continuity_acc, only: continuity_compute_fluxes_barotropic_acc
   implicit none

   interface
      subroutine continuity_cuda_launch(h, u_face_x, v_face_y, wet_T, dy_cu, dx_cv, &
                                        iareaT, hfl_x, hfr_x, hfl_y, hfr_y, &
                                        mass_flux_x, mass_flux_y, flux_h, &
                                        nx, ny, do_pos, h_min_pos, sync) &
         bind(C, name="continuity_cuda_launch")
         import :: c_double, c_int
         implicit none
         real(c_double), intent(in) :: h(*), u_face_x(*), v_face_y(*), wet_T(*)
         real(c_double), intent(in) :: dy_cu(*), dx_cv(*), iareaT(*)
         real(c_double), intent(inout) :: hfl_x(*), hfr_x(*), hfl_y(*), hfr_y(*)
         real(c_double), intent(inout) :: mass_flux_x(*), mass_flux_y(*), flux_h(*)
         integer(c_int), value :: nx, ny, do_pos, sync
         real(c_double), value :: h_min_pos
      end subroutine continuity_cuda_launch
   end interface

   ! Usage: ./continuity_bench [nx] [ny] [nreps] [nwarm] [cuda_sync]
   !   ./continuity_bench                 -> 4096 4096 1000 10, cuda_sync=2
   !   ./continuity_bench 1024 1024       -> 1024^2, 1000 reps
   !   ./continuity_bench 8192 8192 200   -> fewer reps for a big domain
   !   ./continuity_bench 1024 1024 1000 10 0  -> CUDA fully async (pipelined)
   ! cuda_sync: 0 = async, 1 = sync per rep, 2 = sync per kernel (DEFAULT --
   ! matches what `do concurrent` does, so the ratio reflects CODEGEN, not the
   ! launch model). See continuity_kernel.cu's launcher comment.
   integer, parameter :: N_DEF = 4096, REPS_DEF = 1000, WARM_DEF = 10
   real(wp), parameter :: DX = 10.0_wp, DY = 10.0_wp
   integer :: n_reps, n_warm, cu_sync

   type(hgrid_t) :: grid
   type(ocean_metrics_t) :: metrics
   ! SEPARATE output state per variant. The flux bench's harness trap (RESUME §4)
   ! was two variants sharing an output array: only the last writer got checked
   ! and three variants silently inherited "agreement OK". Inputs (h, u, v,
   ! metrics) are read-only, so those ARE shared -- deliberately: all three
   ! variants must see bit-identical input for the comparison to mean anything.
   type(barotropic_cgrid_state_t) :: bs, bs_fl, bs_ac
   type(continuity_t) :: cont, cont_fl, cont_ac
   ! CUDA's outputs: plain arrays, since the C port takes raw pointers.
   real(wp), allocatable :: hfl_x_cu(:, :, :), hfr_x_cu(:, :, :)
   real(wp), allocatable :: hfl_y_cu(:, :, :), hfr_y_cu(:, :, :)
   real(wp), allocatable :: mfx_cu(:, :), mfy_cu(:, :), fh_cu(:, :)
   real(wp) :: t0, t1, ms, ms_fl, ms_cu, ms_ac
   real(wp) :: fh_min, fh_max, fh_sum, dmax_fl, dmax_cu, rmax_cu, dmax_ac, scale, gib
   integer :: i, j, rep, nx, ny, nbad_fl, nbad_cu, nbad_ac, do_pos_i

   nx = iarg(1, N_DEF)
   ny = iarg(2, nx)             ! square by default
   n_reps = iarg(3, REPS_DEF)
   n_warm = iarg(4, WARM_DEF)
   cu_sync = iarg(5, 2)
   ! The PPM loops run i=3:nx-2 / j=3:ny-2, and the boundary loops index
   ! nx-1, ny-1 -- below 6 those overlap and the kernel is meaningless.
   if (nx < 6 .or. ny < 6) then
      write (output_unit, '(a)') 'ERROR: nx and ny must be >= 6 (5-point PPM stencil + boundary rows)'
      stop 1
   end if
   grid%nx_total = nx
   grid%ny_total = ny
   grid%dx = DX
   grid%dy = DY

   ! 31 arrays of ~nx*ny doubles across the three variants' state. A domain
   ! that overruns the device fails inside `enter data` with a CUDA OOM whose
   ! message does not mention the domain size, so say it up front.
   gib = 31.0_wp*real(nx, wp)*real(ny, wp)*8.0_wp/(1024.0_wp**3)

   ! ---- allocate exactly the fields the kernel touches -------------------
   ! Face arrays carry one extra element in their own direction: the kernel
   ! writes mass_flux_x(nx+1, j) and reads mass_flux_x(i+1, j) at i = nx.
   allocate (bs%h(nx, ny))
   allocate (bs%u_face_x(nx + 1, ny), bs%v_face_y(nx, ny + 1))
   allocate (bs%flux_h(nx, ny))
   allocate (bs%mass_flux_x(nx + 1, ny), bs%mass_flux_y(nx, ny + 1))
   allocate (metrics%dy_cu(nx + 1, ny), metrics%dx_cv(nx, ny + 1))
   allocate (metrics%iareaT(nx, ny), metrics%wet_T(nx, ny))
   allocate (cont%h_face_left_x%data(nx + 1, ny, 1))
   allocate (cont%h_face_right_x%data(nx + 1, ny, 1))
   allocate (cont%h_face_left_y%data(nx, ny + 1, 1))
   allocate (cont%h_face_right_y%data(nx, ny + 1, 1))
   ! FLAT's own state
   allocate (bs_fl%h(nx, ny))
   allocate (bs_fl%u_face_x(nx + 1, ny), bs_fl%v_face_y(nx, ny + 1))
   allocate (bs_fl%flux_h(nx, ny))
   allocate (bs_fl%mass_flux_x(nx + 1, ny), bs_fl%mass_flux_y(nx, ny + 1))
   allocate (cont_fl%h_face_left_x%data(nx + 1, ny, 1))
   allocate (cont_fl%h_face_right_x%data(nx + 1, ny, 1))
   allocate (cont_fl%h_face_left_y%data(nx, ny + 1, 1))
   allocate (cont_fl%h_face_right_y%data(nx, ny + 1, 1))
   ! ACC async variant's own state
   allocate (bs_ac%h(nx, ny))
   allocate (bs_ac%u_face_x(nx + 1, ny), bs_ac%v_face_y(nx, ny + 1))
   allocate (bs_ac%flux_h(nx, ny))
   allocate (bs_ac%mass_flux_x(nx + 1, ny), bs_ac%mass_flux_y(nx, ny + 1))
   allocate (cont_ac%h_face_left_x%data(nx + 1, ny, 1))
   allocate (cont_ac%h_face_right_x%data(nx + 1, ny, 1))
   allocate (cont_ac%h_face_left_y%data(nx, ny + 1, 1))
   allocate (cont_ac%h_face_right_y%data(nx, ny + 1, 1))
   ! CUDA's own outputs
   allocate (hfl_x_cu(nx + 1, ny, 1), hfr_x_cu(nx + 1, ny, 1))
   allocate (hfl_y_cu(nx, ny + 1, 1), hfr_y_cu(nx, ny + 1, 1))
   allocate (mfx_cu(nx + 1, ny), mfy_cu(nx, ny + 1), fh_cu(nx, ny))

   ! ---- fill: a wet, gently-varying ocean with a Gaussian bump ----------
   ! The bump is what makes the PPM limiter earn its keep: a flat h would let
   ! every limited slope collapse to the same trivial branch.
   do j = 1, ny
      do i = 1, nx
         bs%h(i, j) = 1000.0_wp &
                      + 50.0_wp*exp(-((real(i - nx/2, wp)/real(nx/8, wp))**2 &
                                      + (real(j - ny/2, wp)/real(ny/8, wp))**2)) &
                      + 2.0_wp*sin(0.01_wp*real(i, wp))*cos(0.013_wp*real(j, wp))
         metrics%wet_T(i, j) = 1.0_wp          ! all-wet
         metrics%iareaT(i, j) = 1.0_wp/(DX*DY)
      end do
   end do
   ! Both signs of u/v, so the upwind branch in the transport loops goes both ways.
   do j = 1, ny
      do i = 1, nx + 1
         bs%u_face_x(i, j) = 0.5_wp*sin(0.02_wp*real(i, wp))
         metrics%dy_cu(i, j) = DY
      end do
   end do
   do j = 1, ny + 1
      do i = 1, nx
         bs%v_face_y(i, j) = 0.3_wp*cos(0.017_wp*real(j, wp))
         metrics%dx_cv(i, j) = DX
      end do
   end do
   bs%flux_h = 0.0_wp; bs%mass_flux_x = 0.0_wp; bs%mass_flux_y = 0.0_wp
   cont%h_face_left_x%data = 0.0_wp; cont%h_face_right_x%data = 0.0_wp
   cont%h_face_left_y%data = 0.0_wp; cont%h_face_right_y%data = 0.0_wp
   cont%h_min = 1.0e-6_wp
   cont%use_ppm_limit_pos = .false.
#ifdef PPM_POS
   cont%use_ppm_limit_pos = .true.
#endif
   ! FLAT gets an identical copy of the same input
   bs_fl%h = bs%h; bs_fl%u_face_x = bs%u_face_x; bs_fl%v_face_y = bs%v_face_y
   bs_fl%flux_h = 0.0_wp; bs_fl%mass_flux_x = 0.0_wp; bs_fl%mass_flux_y = 0.0_wp
   cont_fl%h_face_left_x%data = 0.0_wp; cont_fl%h_face_right_x%data = 0.0_wp
   cont_fl%h_face_left_y%data = 0.0_wp; cont_fl%h_face_right_y%data = 0.0_wp
   cont_fl%h_min = cont%h_min
   cont_fl%use_ppm_limit_pos = cont%use_ppm_limit_pos
   bs_ac%h = bs%h; bs_ac%u_face_x = bs%u_face_x; bs_ac%v_face_y = bs%v_face_y
   bs_ac%flux_h = 0.0_wp; bs_ac%mass_flux_x = 0.0_wp; bs_ac%mass_flux_y = 0.0_wp
   cont_ac%h_face_left_x%data = 0.0_wp; cont_ac%h_face_right_x%data = 0.0_wp
   cont_ac%h_face_left_y%data = 0.0_wp; cont_ac%h_face_right_y%data = 0.0_wp
   cont_ac%h_min = cont%h_min
   cont_ac%use_ppm_limit_pos = cont%use_ppm_limit_pos
   hfl_x_cu = 0.0_wp; hfr_x_cu = 0.0_wp; hfl_y_cu = 0.0_wp; hfr_y_cu = 0.0_wp
   mfx_cu = 0.0_wp; mfy_cu = 0.0_wp; fh_cu = 0.0_wp
   do_pos_i = merge(1, 0, cont%use_ppm_limit_pos)

   write (output_unit, '(a)') repeat('=', 66)
   write (output_unit, '(a,i0,a,i0,a,i0,a)') '  domain: ', nx, ' x ', ny, ' cells (', &
      nx*ny, ' total)'
   write (output_unit, '(a,i0,a,i0,a)') '  reps:   ', n_reps, ' timed, ', n_warm, ' warm-up (untimed)'
   write (output_unit, '(a,f0.2,a)') '  device memory: ~', gib, ' GiB across all 3 variants'
   write (output_unit, '(a,l1)') '  state:  all-wet, Gaussian bump; use_ppm_limit_pos = ', &
      cont%use_ppm_limit_pos
   write (output_unit, '(a,i0,a)') '  cuda_sync: ', cu_sync, &
      merge('  (per-kernel -- matches do concurrent)   ', &
            merge('  (per-rep)                              ', &
                  '  (async -- CUDA pipelines, dc cannot)   ', cu_sync == 1), cu_sync == 2)
   write (output_unit, '(a)') repeat('=', 66)

   ! ---- manual deep copy: parent struct THEN payload, in that order ------
   !$acc enter data copyin(bs)
   !$acc enter data copyin(bs%h, bs%u_face_x, bs%v_face_y)
   !$acc enter data create(bs%flux_h, bs%mass_flux_x, bs%mass_flux_y)
   !$acc enter data copyin(metrics)
   !$acc enter data copyin(metrics%dy_cu, metrics%dx_cv, metrics%iareaT, metrics%wet_T)
   !$acc enter data copyin(cont)
   !$acc enter data copyin(cont%h_face_left_x, cont%h_face_right_x)
   !$acc enter data copyin(cont%h_face_left_y, cont%h_face_right_y)
   !$acc enter data create(cont%h_face_left_x%data, cont%h_face_right_x%data)
   !$acc enter data create(cont%h_face_left_y%data, cont%h_face_right_y%data)
   !$acc enter data copyin(bs_fl)
   !$acc enter data copyin(bs_fl%h, bs_fl%u_face_x, bs_fl%v_face_y)
   !$acc enter data create(bs_fl%flux_h, bs_fl%mass_flux_x, bs_fl%mass_flux_y)
   !$acc enter data copyin(cont_fl)
   !$acc enter data copyin(cont_fl%h_face_left_x, cont_fl%h_face_right_x)
   !$acc enter data copyin(cont_fl%h_face_left_y, cont_fl%h_face_right_y)
   !$acc enter data create(cont_fl%h_face_left_x%data, cont_fl%h_face_right_x%data)
   !$acc enter data create(cont_fl%h_face_left_y%data, cont_fl%h_face_right_y%data)
   !$acc enter data copyin(bs_ac)
   !$acc enter data copyin(bs_ac%h, bs_ac%u_face_x, bs_ac%v_face_y)
   !$acc enter data create(bs_ac%flux_h, bs_ac%mass_flux_x, bs_ac%mass_flux_y)
   !$acc enter data copyin(cont_ac)
   !$acc enter data copyin(cont_ac%h_face_left_x, cont_ac%h_face_right_x)
   !$acc enter data copyin(cont_ac%h_face_left_y, cont_ac%h_face_right_y)
   !$acc enter data create(cont_ac%h_face_left_x%data, cont_ac%h_face_right_x%data)
   !$acc enter data create(cont_ac%h_face_left_y%data, cont_ac%h_face_right_y%data)
   !$acc enter data create(hfl_x_cu, hfr_x_cu, hfl_y_cu, hfr_y_cu)
   !$acc enter data create(mfx_cu, mfy_cu, fh_cu)

   ! ---- variant A: do concurrent (the production kernel, verbatim) -------
   ! n_warm untimed launches first: the first call JITs and faults in every
   ! page, and a single warm-up left a visible cold-run bias (one 4.15 ms
   ! outlier against a 3.83 ms steady state, i.e. it inverted the dc/cuda
   ! verdict). Warm every variant, then time.
   do rep = 1, n_warm
      call continuity_compute_fluxes_barotropic(grid, metrics, cont, bs)
   end do
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call continuity_compute_fluxes_barotropic(grid, metrics, cont, bs)
   end do
   !$acc wait
   t1 = wall()
   ms = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variant B: FLAT control — same body, helper calls hand-inlined ---
   do rep = 1, n_warm
      call continuity_compute_fluxes_barotropic_flat(grid, metrics, cont_fl, bs_fl)
   end do
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call continuity_compute_fluxes_barotropic_flat(grid, metrics, cont_fl, bs_fl)
   end do
   !$acc wait
   t1 = wall()
   ms_fl = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variant B2: same body + !$acc kernels async(1) -------------------
   ! The DC bodies are untouched -- only the launch queue changes. 9 host syncs
   ! per call become 1 (the wait(1) inside the routine).
   do rep = 1, n_warm
      call continuity_compute_fluxes_barotropic_acc(grid, metrics, cont_ac, bs_ac)
   end do
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call continuity_compute_fluxes_barotropic_acc(grid, metrics, cont_ac, bs_ac)
   end do
   !$acc wait
   t1 = wall()
   ms_ac = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variant C: faithful CUDA C port of the same kernel ---------------
   ! Reads the SAME device input allocation as A and B (bs%h etc.), writes its
   ! own outputs. 9 kernels, one per Fortran loop, same order.
   do rep = 1, n_warm
      !$acc host_data use_device(bs%h, bs%u_face_x, bs%v_face_y, metrics%wet_T, &
      !$acc                      metrics%dy_cu, metrics%dx_cv, metrics%iareaT, &
      !$acc                      hfl_x_cu, hfr_x_cu, hfl_y_cu, hfr_y_cu, &
      !$acc                      mfx_cu, mfy_cu, fh_cu)
      call continuity_cuda_launch(bs%h, bs%u_face_x, bs%v_face_y, metrics%wet_T, &
                                  metrics%dy_cu, metrics%dx_cv, metrics%iareaT, &
                                  hfl_x_cu, hfr_x_cu, hfl_y_cu, hfr_y_cu, &
                                  mfx_cu, mfy_cu, fh_cu, nx, ny, do_pos_i, cont%h_min, cu_sync)
      !$acc end host_data
   end do
   t0 = wall()
   do rep = 1, n_reps
      !$acc host_data use_device(bs%h, bs%u_face_x, bs%v_face_y, metrics%wet_T, &
      !$acc                      metrics%dy_cu, metrics%dx_cv, metrics%iareaT, &
      !$acc                      hfl_x_cu, hfr_x_cu, hfl_y_cu, hfr_y_cu, &
      !$acc                      mfx_cu, mfy_cu, fh_cu)
      call continuity_cuda_launch(bs%h, bs%u_face_x, bs%v_face_y, metrics%wet_T, &
                                  metrics%dy_cu, metrics%dx_cv, metrics%iareaT, &
                                  hfl_x_cu, hfr_x_cu, hfl_y_cu, hfr_y_cu, &
                                  mfx_cu, mfy_cu, fh_cu, nx, ny, do_pos_i, cont%h_min, cu_sync)
      !$acc end host_data
   end do
   !$acc host_data use_device(bs%h, bs%u_face_x, bs%v_face_y, metrics%wet_T, &
   !$acc                      metrics%dy_cu, metrics%dx_cv, metrics%iareaT, &
   !$acc                      hfl_x_cu, hfr_x_cu, hfl_y_cu, hfr_y_cu, &
   !$acc                      mfx_cu, mfy_cu, fh_cu)
   call continuity_cuda_launch(bs%h, bs%u_face_x, bs%v_face_y, metrics%wet_T, &
                               metrics%dy_cu, metrics%dx_cv, metrics%iareaT, &
                               hfl_x_cu, hfr_x_cu, hfl_y_cu, hfr_y_cu, &
                               mfx_cu, mfy_cu, fh_cu, nx, ny, do_pos_i, cont%h_min, 1)
   !$acc end host_data
   t1 = wall()
   ms_cu = (t1 - t0)*1000.0_wp/real(n_reps + 1, wp)

   !$acc update self(bs%flux_h)
   !$acc update self(bs_fl%flux_h)
   !$acc update self(bs_ac%flux_h)
   !$acc update self(fh_cu)

   ! ---- do the three agree? ---------------------------------------------
   ! DC vs FLAT: hand-inlining reassociates nothing -> must be BIT-identical.
   ! DC vs CUDA: not bit-exact -- FMA contraction differs between nvfortran and
   ! nvcc, so the last ~1-2 ulp legitimately move. Beyond ~1e-12 relative is a
   ! PORT BUG, which would mean the timing compares two different algorithms.
   dmax_fl = 0.0_wp; nbad_fl = 0
   dmax_ac = 0.0_wp; nbad_ac = 0
   dmax_cu = 0.0_wp; rmax_cu = 0.0_wp; nbad_cu = 0
   do j = 1, ny
      do i = 1, nx
         dmax_fl = max(dmax_fl, abs(bs%flux_h(i, j) - bs_fl%flux_h(i, j)))
         if (bs%flux_h(i, j) /= bs_fl%flux_h(i, j)) nbad_fl = nbad_fl + 1
         dmax_ac = max(dmax_ac, abs(bs%flux_h(i, j) - bs_ac%flux_h(i, j)))
         if (bs%flux_h(i, j) /= bs_ac%flux_h(i, j)) nbad_ac = nbad_ac + 1
         dmax_cu = max(dmax_cu, abs(bs%flux_h(i, j) - fh_cu(i, j)))
         scale = max(abs(bs%flux_h(i, j)), abs(fh_cu(i, j)))
         if (scale > 1.0e-30_wp) then
            rmax_cu = max(rmax_cu, abs(bs%flux_h(i, j) - fh_cu(i, j))/scale)
            if (abs(bs%flux_h(i, j) - fh_cu(i, j))/scale > 1.0e-12_wp) nbad_cu = nbad_cu + 1
         end if
      end do
   end do

   ! ---- sanity: the kernel must have DONE something --------------------
   ! Catches a kernel that silently no-op'd (all-zero) or went unstable (NaN),
   ! either of which would make the timing meaningless.
   fh_min = minval(bs%flux_h)
   fh_max = maxval(bs%flux_h)
   fh_sum = sum(bs%flux_h)

   write (output_unit, '(a,f10.4,a)') '  continuity PPM (do concurrent)        : ', ms, ' ms/rep'
   write (output_unit, '(a,f10.4,a)') '  FLAT control (helpers hand-inlined)   : ', ms_fl, ' ms/rep'
   write (output_unit, '(a,f10.4,a)') '  ACC kernels async(1) (same DC body)   : ', ms_ac, ' ms/rep'
   write (output_unit, '(a,f10.4,a)') '  CUDA C (faithful port, nvcc)          : ', ms_cu, ' ms/rep'
   write (output_unit, '(a)') ''
   write (output_unit, '(a,f10.3,a)') '  ratio  dc / cuda                      : ', ms/ms_cu, ' x'
   write (output_unit, '(a,f10.3,a)') '  ratio  dc / flat                      : ', ms/ms_fl, ' x'
   write (output_unit, '(a,f10.3,a)') '  ratio  dc / acc-async                 : ', ms/ms_ac, ' x'
   write (output_unit, '(a,f10.3,a)') '  ratio  acc-async / cuda               : ', ms_ac/ms_cu, ' x'
   write (output_unit, '(a)') ''
   write (output_unit, '(a,i0,a,es10.3)') '  DC vs FLAT : ', nbad_fl, ' cells differ, max |diff| = ', dmax_fl
   write (output_unit, '(a,i0,a,es10.3)') '  DC vs ACC  : ', nbad_ac, ' cells differ, max |diff| = ', dmax_ac
   if (nbad_fl /= 0) write (output_unit, '(a)') '               *** INLINE NOT FAITHFUL ***'
   write (output_unit, '(a,es12.5)') '  DC vs CUDA : max |diff|  : ', dmax_cu
   write (output_unit, '(a,es12.5)') '               max rel diff: ', rmax_cu
   if (rmax_cu < 1.0e-12_wp) then
      write (output_unit, '(a)') '               agreement   : OK (<1e-12 rel -> FMA contraction only)'
   else
      write (output_unit, '(a,i0,a)') '               agreement   : *** SUSPECT *** ', nbad_cu, &
         ' cells >1e-12 rel -- likely a PORT BUG, not rounding'
   end if
   write (output_unit, '(a)') ''
   write (output_unit, '(a,es14.6)') '  min flux_h             : ', fh_min
   write (output_unit, '(a,es14.6)') '  max flux_h             : ', fh_max
   write (output_unit, '(a,es14.6)') '  sum flux_h             : ', fh_sum
   if (fh_min /= fh_min .or. fh_max /= fh_max) then
      write (output_unit, '(a)') '  sanity                 : *** NaN -- results are garbage ***'
   else if (fh_min == 0.0_wp .and. fh_max == 0.0_wp) then
      write (output_unit, '(a)') '  sanity                 : *** all-zero -- kernel did nothing ***'
   else
      write (output_unit, '(a)') '  sanity                 : OK (finite, non-zero)'
   end if
   write (output_unit, '(a)') repeat('=', 66)

contains

   !! Positional integer argument `k`, or `dflt` if absent/unparseable.
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

end program continuity_bench
