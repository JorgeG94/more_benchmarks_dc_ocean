!! Driver for the ALE-remap microbenchmark.
!!
!!   ./ale_bench [nx] [ny] [nz] [reps] [warmup] [pert_pct] [dump]
!!
!! `dump` (arg 7, default 0) is OPT-IN and off for `make run`. With dump=1 the
!! driver writes `ale_ref.bin`: the PRISTINE input state plus the results of
!! variants 1 (DC as shipped), 6 (CUDA faithful) and 7 (CUDA fused). It exists
!! solely so `ale_native.cu` -- the Fortran-free C++/CUDA driver -- can prove
!! that (a) its independently written init reproduces this one and (b) its
!! output matches this run's. Nothing else reads the file; see ale_native.cu.
!!
!! nx/ny are POINT COUNTS as they appear in ~/analysis_gebco/*.nml; the driver
!! adds nghost=3 on each side to get nx_total/ny_total, which is what production
!! passes as `grid%nx_total`. Default 473 297 30 = the 0.1 deg gabight config
!! that produced the profile (ocean_ale_remap = 11.2% of runtime).
!!
!! Method notes:
!!  * ALL device data is mapped HERE, in the scope that OWNS the variables
!!    (parent struct first, then each payload) -- mapping inside a helper maps
!!    the dummy and the components never attach.
!!  * Every variant gets its OWN result slot, and the working state is restored
!!    from a pristine device copy before every rep. Sharing outputs means only
!!    the last writer is checked and the others silently inherit "agreement OK".
!!  * The restore is MANDATORY for a meaningful measurement, not just for
!!    correctness: the remap has a FIXED POINT. After one call h_old == target_h,
!!    so every subsequent call finds each new layer exactly aligned with an old
!!    layer and the PPM overlap sweep collapses to one iteration per layer.
!!    Timing reps without restoring would measure a kernel doing strictly less
!!    work than production's.
!!  * min-of-reps; the restore is outside the timed section.
program ale_bench
   use constants, only: wp, REMAP_PPM, VCOORD_ZSTAR
   use remap_state, only: hgrid_t, ocean_vcoord_t, multilayer_cgrid_state_t
   use ale_remap, only: step_dc => ale_remap_step
   use ale_remap_acc, only: step_dc_acc => ale_remap_step
   use ale_remap_fix, only: step_dc_fix => ale_remap_step
   use ale_remap_fused, only: step_fused => ale_remap_step
   use ale_remap_fused_acc, only: step_fused_acc => ale_remap_step
   use iso_c_binding, only: c_int, c_ptr, c_loc
   implicit none

   interface
      subroutine ale_remap_cuda(nx, ny, nz, method, fused, &
                                h_layer, h_old, target_h, total_h, h_ref, &
                                hTr_t, hTr_s, u_face, v_face, &
                                mass_b, heat_b, salt_b, bt_eta, bt_H_ref, dsig) &
         bind(C, name="ale_remap_cuda")
         import :: c_int, c_ptr
         integer(c_int), value :: nx, ny, nz, method, fused
         type(c_ptr), value :: h_layer, h_old, target_h, total_h, h_ref
         type(c_ptr), value :: hTr_t, hTr_s, u_face, v_face
         type(c_ptr), value :: mass_b, heat_b, salt_b, bt_eta, bt_H_ref, dsig
      end subroutine ale_remap_cuda
      subroutine cuda_sync() bind(C, name="cuda_sync")
      end subroutine cuda_sync
   end interface

   integer, parameter :: NGHOST = 3
   integer, parameter :: NVAR = 7
   character(len=26), parameter :: vname(NVAR) = [ character(len=26) :: &
      "DC (as shipped)", "DC + acc async(1)", "DC + fixes", &
      "DC + fixes + fused", "DC + fixes + fused + async", &
      "CUDA faithful", "CUDA fused" ]

   type(hgrid_t) :: grid
   type(ocean_vcoord_t), target :: vc
   type(multilayer_cgrid_state_t), target :: ms

   integer :: nx, ny, nz, nxt, nyt, reps, warmup
   integer :: pert_pct, dump
   integer :: i, j, k, r, v
   real(wp) :: t0, t1, best(NVAR), dt

   real(wp), allocatable, target :: bt_eta(:, :), bt_H_ref(:, :)
   real(wp), allocatable :: p_h_layer(:, :, :), p_hTr_t(:, :, :), p_hTr_s(:, :, :)
   real(wp), allocatable :: p_u(:, :, :), p_v(:, :, :), p_eta(:, :)
   real(wp), allocatable :: p_mass(:, :, :), p_heat(:, :, :), p_salt(:, :, :)
   real(wp), allocatable :: res_h(:, :, :, :), res_t(:, :, :, :), res_u(:, :, :, :)
   real(wp), allocatable :: res_eta(:, :, :), res_b(:, :, :, :)
   real(wp) :: hbed, sig, pert, cT, cS, csum

   nx = 473; ny = 297; nz = 30; reps = 20; warmup = 10
   ! Amplitude (%) of the layer-thickness drift the remap has to undo. This is
   ! the ONE state knob the cost is strongly sensitive to: it sets how many old
   ! layers each new layer overlaps, i.e. the PPM sweep's inner trip count.
   ! Default 25 is deliberately pessimistic; see README "calibration".
   pert_pct = 25
   dump = 0
   call getarg_i(1, nx); call getarg_i(2, ny); call getarg_i(3, nz)
   call getarg_i(4, reps); call getarg_i(5, warmup); call getarg_i(6, pert_pct)
   call getarg_i(7, dump)

   nxt = nx + 2*NGHOST
   nyt = ny + 2*NGHOST

   grid%nx_total = nxt; grid%ny_total = nyt
   vc%nx_total = nxt; vc%ny_total = nyt; vc%nz_ml = nz
   vc%coord_type = VCOORD_ZSTAR; vc%is_init = .true.; vc%remap_method = REMAP_PPM
   vc%regrid_time_scale = 0.0_wp; vc%remap_vel_conserve_ke = .false.
   ms%nz_ml = nz; ms%idx_temperature = 1; ms%idx_salinity = 2

   allocate (vc%dsig(nz), vc%target_h(nxt, nyt, nz), vc%remap_h_old(nxt, nyt, nz))
   allocate (vc%remap_total_h(nxt, nyt), vc%remap_h_ref(nxt, nyt))
   allocate (ms%h_layer(nxt, nyt, nz), ms%mass_budget_remap(nxt, nyt, nz))
   allocate (ms%heat_budget_remap(nxt, nyt, nz), ms%salt_budget_remap(nxt, nyt, nz))
   allocate (ms%u_face_x_layer(nxt + 1, nyt, nz), ms%v_face_y_layer(nxt, nyt + 1, nz))
   allocate (ms%tracers(2))
   allocate (ms%tracers(1)%hTr(nxt, nyt, nz), ms%tracers(2)%hTr(nxt, nyt, nz))
   allocate (bt_eta(nxt, nyt), bt_H_ref(nxt, nyt))
   allocate (p_h_layer(nxt, nyt, nz), p_hTr_t(nxt, nyt, nz), p_hTr_s(nxt, nyt, nz))
   allocate (p_u(nxt + 1, nyt, nz), p_v(nxt, nyt + 1, nz), p_eta(nxt, nyt))
   allocate (p_mass(nxt, nyt, nz), p_heat(nxt, nyt, nz), p_salt(nxt, nyt, nz))
   allocate (res_h(nxt, nyt, nz, NVAR), res_t(nxt, nyt, nz, NVAR))
   allocate (res_u(nxt + 1, nyt, nz, NVAR), res_eta(nxt, nyt, NVAR))
   allocate (res_b(nxt, nyt, nz, NVAR))

   do k = 1, nz
      vc%dsig(k) = 1.0_wp + 3.0_wp*real(k - 1, wp)/real(max(1, nz - 1), wp)
   end do
   vc%dsig = vc%dsig/sum(vc%dsig)

   ! Branch coverage matters more than physics here: the PPM limiter, the
   ! vanishing-layer guard and the overlap sweep must all fire on a realistic
   ! MIX of columns, or the warp divergence is fake.
   do j = 1, nyt
      do i = 1, nxt
         hbed = 15.0_wp + 4485.0_wp*abs(sin(0.013_wp*real(i, wp))*cos(0.017_wp*real(j, wp)))
         if (mod(i + j, 97) == 0) hbed = 3.0_wp     ! very shallow => vanished layers
         bt_H_ref(i, j) = hbed
         bt_eta(i, j) = 0.4_wp*sin(0.05_wp*real(i, wp) + 0.03_wp*real(j, wp))
         do k = 1, nz
            sig = vc%dsig(k)
            ! h_layer perturbed AWAY from the zstar target: this is the state
            ! continuity/dynamics hands the remap.
            pert = (0.01_wp*real(pert_pct, wp))*sin(3.0_wp*real(k, wp) + 0.02_wp*real(i, wp) + 0.03_wp*real(j, wp))
            ms%h_layer(i, j, k) = (hbed + bt_eta(i, j))*sig*(1.0_wp + pert)
         end do
         ! renormalise so sum_k h_layer == hbed + eta exactly => the remap is a
         ! genuine conservative redistribution, as production's is.
         csum = sum(ms%h_layer(i, j, 1:nz))
         ms%h_layer(i, j, 1:nz) = ms%h_layer(i, j, 1:nz)*(hbed + bt_eta(i, j))/csum
         do k = 1, nz
            cT = 2.0_wp + 12.0_wp*exp(-real(nz - k, wp)/6.0_wp) + 0.5_wp*sin(0.02_wp*real(i, wp))
            cS = 34.5_wp + 0.8_wp*real(k, wp)/real(nz, wp)
            ms%tracers(1)%hTr(i, j, k) = ms%h_layer(i, j, k)*cT
            ms%tracers(2)%hTr(i, j, k) = ms%h_layer(i, j, k)*cS
         end do
      end do
   end do
   ms%mass_budget_remap = 0.0_wp
   ms%heat_budget_remap = 0.0_wp
   ms%salt_budget_remap = 0.0_wp
   do k = 1, nz
      do j = 1, nyt
         do i = 1, nxt + 1
            ms%u_face_x_layer(i, j, k) = 0.3_wp*sin(0.01_wp*real(i, wp))* &
                                         cos(0.02_wp*real(j, wp))*(1.0_wp + 0.1_wp*real(k, wp))
         end do
      end do
      do j = 1, nyt + 1
         do i = 1, nxt
            ms%v_face_y_layer(i, j, k) = 0.2_wp*cos(0.015_wp*real(i, wp))* &
                                         sin(0.018_wp*real(j, wp))*(1.0_wp - 0.05_wp*real(k, wp))
         end do
      end do
   end do

   p_h_layer = ms%h_layer; p_hTr_t = ms%tracers(1)%hTr; p_hTr_s = ms%tracers(2)%hTr
   p_u = ms%u_face_x_layer; p_v = ms%v_face_y_layer; p_eta = bt_eta
   p_mass = 0.0_wp; p_heat = 0.0_wp; p_salt = 0.0_wp

   write (*, '(a)') "================================================================"
   write (*, '(a,i0,a,i0,a,i0,a,i0)') " ALE remap: ", nx, "x", ny, " pts (+", NGHOST, &
      " ghost each side) x nz=", nz
   write (*, '(a,i0,a,i0,a,i0,a)') "   nx_total=", nxt, " ny_total=", nyt, " => ", &
      nxt*nyt*nz, " cells"
   write (*, '(a,i0,a,i0,a,i0,a)') "   reps=", reps, "  warmup=", warmup, &
      "  h-drift=", pert_pct, "%" 
   write (*, '(a)') "================================================================"

   !$acc enter data copyin(grid, vc, ms)
   !$acc enter data copyin(vc%dsig, vc%target_h, vc%remap_h_old, vc%remap_total_h, vc%remap_h_ref)
   !$acc enter data copyin(ms%h_layer, ms%mass_budget_remap, ms%heat_budget_remap, &
   !$acc                   ms%salt_budget_remap, ms%u_face_x_layer, ms%v_face_y_layer)
   !$acc enter data copyin(ms%tracers)
   !$acc enter data copyin(ms%tracers(1)%hTr)
   !$acc enter data copyin(ms%tracers(2)%hTr)
   !$acc enter data copyin(bt_eta, bt_H_ref)
   !$acc enter data copyin(p_h_layer, p_hTr_t, p_hTr_s, p_u, p_v, p_eta, p_mass, p_heat, p_salt)

   best = huge(1.0_wp)
   do v = 1, NVAR
      do r = 1, warmup
         call restore()
         call run(v)
      end do
      do r = 1, reps
         call restore()
         !$acc wait
         call cuda_sync()
         t0 = wtime()
         call run(v)
         !$acc wait
         call cuda_sync()
         t1 = wtime()
         dt = t1 - t0
         if (dt < best(v)) best(v) = dt
      end do
      !$acc update self(ms%h_layer, ms%tracers(1)%hTr, ms%u_face_x_layer, bt_eta, &
      !$acc             ms%heat_budget_remap)
      res_h(:, :, :, v) = ms%h_layer
      res_t(:, :, :, v) = ms%tracers(1)%hTr
      res_u(:, :, :, v) = ms%u_face_x_layer
      res_eta(:, :, v) = bt_eta
      res_b(:, :, :, v) = ms%heat_budget_remap
   end do

   write (*, '(a)') " variant                          ms     vs DC"
   write (*, '(a)') " ------------------------------------------------"
   do v = 1, NVAR
      write (*, '(a,a26,f9.3,f8.3,a)') "  ", vname(v), best(v)*1.0e3_wp, best(1)/best(v), "x"
   end do

   write (*, '(a)') ""
   write (*, '(a)') " max|diff| vs DC (as shipped):  h_layer     hTr_T    u_face_x    bt_eta   heat_budget"
   write (*, '(a)') " ---------------------------------------------------------------------------------"
   do v = 2, NVAR
      write (*, '(a,a26,5(1x,es10.3))') "  ", vname(v), &
         maxval(abs(res_h(:, :, :, v) - res_h(:, :, :, 1))), &
         maxval(abs(res_t(:, :, :, v) - res_t(:, :, :, 1))), &
         maxval(abs(res_u(:, :, :, v) - res_u(:, :, :, 1))), &
         maxval(abs(res_eta(:, :, v) - res_eta(:, :, 1))), &
         maxval(abs(res_b(:, :, :, v) - res_b(:, :, :, 1)))
   end do

   ! ---- Does the check DISCRIMINATE? Perturb ONE cell by 1 ulp and confirm the
   ! comparison trips. A check that cannot fail proves nothing.
   block
      real(wp) :: save_val, trip
      save_val = res_h(nxt/2, nyt/2, max(1, nz/2), 1)
      res_h(nxt/2, nyt/2, max(1, nz/2), 1) = nearest(save_val, 1.0_wp)
      trip = maxval(abs(res_h(:, :, :, 4) - res_h(:, :, :, 1)))
      write (*, '(a)') ""
      write (*, '(a,es10.3,a)') " check-discriminates probe: perturb 1 cell of DC by 1 ulp -> "// &
         "max|diff| vs fused = ", trip, "   (MUST be > 0)"
      res_h(nxt/2, nyt/2, max(1, nz/2), 1) = save_val
   end block

   ! ---- OPT-IN state/reference dump for the native C++/CUDA driver ---------
   ! Raw stream, host byte order, plain doubles in Fortran storage order. The
   ! native driver recomputes the init itself and diffs against the PRISTINE
   ! block here; the per-variant blocks let it diff its own kernel output
   ! against this run's. Written only when arg 7 /= 0.
   if (dump /= 0) then
      block
         integer :: iu
         integer, parameter :: DUMP_VARS(3) = [1, 6, 7]
         open (newunit=iu, file="ale_ref.bin", access="stream", &
               form="unformatted", status="replace", action="write")
         write (iu) 1094992177             ! magic 'ALE1'
         write (iu) 1                      ! format version
         write (iu) nxt, nyt, nz, pert_pct
         write (iu) vc%dsig                ! pristine inputs
         write (iu) bt_H_ref
         write (iu) p_eta
         write (iu) p_h_layer
         write (iu) p_hTr_t
         write (iu) p_hTr_s
         write (iu) p_u
         write (iu) p_v
         write (iu) size(DUMP_VARS)
         do v = 1, size(DUMP_VARS)
            write (iu) DUMP_VARS(v)
            write (iu) res_h(:, :, :, DUMP_VARS(v))
            write (iu) res_t(:, :, :, DUMP_VARS(v))
            write (iu) res_u(:, :, :, DUMP_VARS(v))
            write (iu) res_eta(:, :, DUMP_VARS(v))
            write (iu) res_b(:, :, :, DUMP_VARS(v))
         end do
         close (iu)
         write (*, '(a)') ""
         write (*, '(a)') " dump: wrote ale_ref.bin (pristine state + variants 1, 6, 7)"
      end block
   end if

contains

   subroutine getarg_i(n, val)
      integer, intent(in) :: n
      integer, intent(inout) :: val
      character(len=32) :: s
      integer :: st
      if (command_argument_count() >= n) then
         call get_command_argument(n, s)
         read (s, *, iostat=st) val
      end if
   end subroutine getarg_i

   real(wp) function wtime()
      integer(8) :: c, cr
      call system_clock(c, cr)
      wtime = real(c, wp)/real(cr, wp)
   end function wtime

   subroutine restore()
      integer :: i, j, k
      do concurrent(k=1:nz, j=1:nyt, i=1:nxt)
         ms%h_layer(i, j, k) = p_h_layer(i, j, k)
         ms%tracers(1)%hTr(i, j, k) = p_hTr_t(i, j, k)
         ms%tracers(2)%hTr(i, j, k) = p_hTr_s(i, j, k)
         ms%mass_budget_remap(i, j, k) = p_mass(i, j, k)
         ms%heat_budget_remap(i, j, k) = p_heat(i, j, k)
         ms%salt_budget_remap(i, j, k) = p_salt(i, j, k)
      end do
      do concurrent(k=1:nz, j=1:nyt, i=1:nxt + 1)
         ms%u_face_x_layer(i, j, k) = p_u(i, j, k)
      end do
      do concurrent(k=1:nz, j=1:nyt + 1, i=1:nxt)
         ms%v_face_y_layer(i, j, k) = p_v(i, j, k)
      end do
      do concurrent(j=1:nyt, i=1:nxt)
         bt_eta(i, j) = p_eta(i, j)
      end do
   end subroutine restore

   subroutine run(v)
      integer, intent(in) :: v
      select case (v)
      case (1); call step_dc(grid, vc, ms, bt_eta, bt_H_ref)
      case (2); call step_dc_acc(grid, vc, ms, bt_eta, bt_H_ref)
      case (3); call step_dc_fix(grid, vc, ms, bt_eta, bt_H_ref)
      case (4); call step_fused(grid, vc, ms, bt_eta, bt_H_ref)
      case (5); call step_fused_acc(grid, vc, ms, bt_eta, bt_H_ref)
      case (6); call run_cuda(0)
      case (7); call run_cuda(1)
      end select
   end subroutine run

   subroutine run_cuda(fused)
      integer, intent(in) :: fused
      type(c_ptr) :: c_hl, c_ho, c_th, c_tot, c_hr, c_tt, c_ts
      type(c_ptr) :: c_u, c_v, c_mb, c_hb, c_sb, c_eta, c_href, c_ds
      !$acc host_data use_device(ms%h_layer, vc%remap_h_old, vc%target_h, &
      !$acc                     vc%remap_total_h, vc%remap_h_ref, &
      !$acc                     ms%tracers(1)%hTr, ms%tracers(2)%hTr, &
      !$acc                     ms%u_face_x_layer, ms%v_face_y_layer, &
      !$acc                     ms%mass_budget_remap, ms%heat_budget_remap, &
      !$acc                     ms%salt_budget_remap, bt_eta, bt_H_ref, vc%dsig)
      c_hl = c_loc(ms%h_layer); c_ho = c_loc(vc%remap_h_old); c_th = c_loc(vc%target_h)
      c_tot = c_loc(vc%remap_total_h); c_hr = c_loc(vc%remap_h_ref)
      c_tt = c_loc(ms%tracers(1)%hTr); c_ts = c_loc(ms%tracers(2)%hTr)
      c_u = c_loc(ms%u_face_x_layer); c_v = c_loc(ms%v_face_y_layer)
      c_mb = c_loc(ms%mass_budget_remap); c_hb = c_loc(ms%heat_budget_remap)
      c_sb = c_loc(ms%salt_budget_remap)
      c_eta = c_loc(bt_eta); c_href = c_loc(bt_H_ref); c_ds = c_loc(vc%dsig)
      !$acc end host_data
      call ale_remap_cuda(nxt, nyt, nz, vc%remap_method, fused, &
                          c_hl, c_ho, c_th, c_tot, c_hr, c_tt, c_ts, &
                          c_u, c_v, c_mb, c_hb, c_sb, c_eta, c_href, c_ds)
   end subroutine run_cuda

end program ale_bench
