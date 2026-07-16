!! Barotropic substep: `do concurrent` vs CUDA C (faithful / fused / graph).
!!
!! THE QUESTION: `ocean_barotropic_solver` is 8.6% of runtime with EPBL on and
!! ~16% with it off (profile: tassie_runs/gabight_acc_120d.*.log). It runs
!! n_inner=24 substeps x 3 calls per outer step on a ~140k-cell 2-D domain --
!! the size where the other benchmarks in this repo show `do concurrent` losing
!! 1.5-1.6x to CUDA C. But production already wraps the whole substep in
!! `!$acc kernels async(1)` with a single drain, which removes the per-loop host
!! sync that causes that loss. So: is there anything left, and if so, what kind?
!!
!! FOUR VARIANTS, isolating three different things:
!!   DC              : the closed-basin substep, `!$acc kernels async(1)`, as production.
!!   CUDA faithful   : 11 kernels/substep, one per DC loop -> measures CODEGEN.
!!   CUDA fused      : 5 kernels/substep (walls + accumulators folded into their
!!                     producers as single-assignment merges) -> measures what
!!                     FEWER LAUNCHES buy. Should stay bit-identical.
!!   CUDA graph      : the fused kernels captured in a cudaGraph -> measures the
!!                     residual LAUNCH cost. `do concurrent` cannot do this.
!!
!! If fused >> faithful, the fix is to fuse in Fortran -- no CUDA needed.
!! If graph >> fused, the remaining cost is launch overhead and only CUDA (or a
!! compiler that emits graphs) can take it.
!! If they all tie, 3.35 ms/step is a hard floor and this path is done.
!!
!! ⚠ SCOPE: this is a TRANSCRIPTION of the live closed-basin path, not a
!! verbatim extract -- see btstep.F90's header for exactly what was dropped
!! (wetdry, BT_cont, upstream-h_face, Flather OBC, periodic wraps). It models
!! the substep's SHAPE faithfully; it is not bit-comparable to a production run.
!!
!! Usage: ./btstep_bench [nx_phys] [ny_phys] [n_inner] [nreps] [nwarm]
!!   ./btstep_bench                -> 473 297 24 (the 0.1 deg config), 200 reps
!!   ./btstep_bench 945 594 24     -> the 0.05 deg hi-res config
program btstep_bench
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr, c_loc, c_f_pointer
   use constants, only: wp
   use bt_state
   use btstep, only: btstep_nonlinear_closed
   use btstep_plain_mod, only: btstep_plain
   use btstep_fused_mod, only: btstep_fused
   implicit none

   !! Mirrors `struct BtArgs` in btstep_kernel.cu -- field order is load-bearing.
   type, bind(C) :: bt_args_t
      type(c_ptr) :: eta, eta_new, H_ref, ubt, vbt, ubt_prev, vbt_prev
      type(c_ptr) :: rem_u, rem_v, zeta, ke, ubt_sum, vbt_sum, eta_sum
      type(c_ptr) :: uhbt_sum, vhbt_sum
      type(c_ptr) :: dy_cu, dx_cv, iareaT, areaCu, areaCv, dxCu, dyCv
      type(c_ptr) :: idxCu, idyCv, iareaBu, f_corner, force_u, force_v
      integer(c_int) :: nx, ny, nghost, nx_phys, ny_phys
      real(c_double) :: G, bebt, dt
   end type bt_args_t

   interface
      subroutine btstep_cuda_launch(a, n_steps, mode) bind(C, name="btstep_cuda_launch")
         import :: bt_args_t, c_int
         type(bt_args_t), intent(in) :: a
         integer(c_int), value :: n_steps, mode
      end subroutine
   end interface

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NINNER_DEF = 24
   integer, parameter :: REPS_DEF = 200, WARM_DEF = 5, NGH = 3
   real(wp), parameter :: DXM = 10000.0_wp, DYM = 10000.0_wp, DT_INNER = 12.0_wp

   type(hgrid_t) :: grid
   type(ocean_metrics_t) :: met
   type(coriolis_t) :: cor
   type(bt_work_t) :: w, wc, wp_, wf
   type(bt_args_t) :: args
   real(wp), allocatable, target :: force_u(:, :), force_v(:, :)
   real(wp), allocatable :: eta_ini(:, :)   !! init_state's eta, for dump_ref
   real(wp) :: t0, t1, ms_dc, ms_pl, ms_fus, ms_f, ms_fu, ms_g, d_eta, d_u, r_eta, d_pl, d_fus
   integer :: i, j, rep, nx, ny, nxp, nyp, n_inner, n_reps, n_warm

   nxp = iarg(1, NXP_DEF); nyp = iarg(2, NYP_DEF); n_inner = iarg(3, NINNER_DEF)
   n_reps = iarg(4, REPS_DEF); n_warm = iarg(5, WARM_DEF)
   nx = nxp + 2*NGH; ny = nyp + 2*NGH
   grid%nx_total = nx; grid%ny_total = ny
   grid%nx_phys = nxp; grid%ny_phys = nyp; grid%nghost = NGH

   call alloc_state(w); call alloc_state(wc); call alloc_state(wp_); call alloc_state(wf)
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

   call init_state(w); call init_state(wc); call init_state(wp_); call init_state(wf)

   write (output_unit, '(a)') repeat('=', 72)
   write (output_unit, '(a,i0,a,i0,a,i0,a)') '  domain: ', nxp, ' x ', nyp, ' interior (+', NGH, ' ghosts) '
   write (output_unit, '(a,i0,a,i0,a,i0)') '  arrays: ', nx, ' x ', ny, ' = ', nx*ny
   write (output_unit, '(a,i0,a,i0,a,i0,a)') '  n_inner: ', n_inner, ' substeps;  ', n_reps, &
      ' timed reps, ', n_warm, ' warm-up'
   write (output_unit, '(a,f0.2,a,f0.1,a)') '  bebt = ', w%bebt, ';  dt_inner = ', DT_INNER, ' s'
   write (output_unit, '(a)') repeat('=', 72)

   ! ⚠ DEEP COPY MUST BE IN THIS SCOPE. Doing it inside a helper
   ! (`enter_state(s)`) maps the DUMMY; the components then never attach to the
   ! device struct, every `!$acc kernels` region falls back to
   ! "Generating implicit copy(bt_work)", and the benchmark times PCIe. That
   ! bug read as a 2.5x "codegen" gap until nsys showed 17429 cuMemcpyDtoHAsync
   ! calls (42.9% of API time). Parent struct first, then each payload.
   !$acc enter data copyin(met, cor)
   !$acc enter data copyin(w)
   !$acc enter data copyin(w%bt_eta, w%bt_eta_new, w%bt_H_ref, w%bt_ubt, w%bt_vbt)
   !$acc enter data copyin(w%bt_ubt_prev, w%bt_vbt_prev, w%bt_rem_u, w%bt_rem_v)
   !$acc enter data copyin(w%bt_zeta_corner, w%bt_ke_centre)
   !$acc enter data copyin(w%ubt_sum, w%vbt_sum, w%eta_sum, w%uhbt_sum, w%vhbt_sum)
   !$acc enter data copyin(wc)
   !$acc enter data copyin(wc%bt_eta, wc%bt_eta_new, wc%bt_H_ref, wc%bt_ubt, wc%bt_vbt)
   !$acc enter data copyin(wc%bt_ubt_prev, wc%bt_vbt_prev, wc%bt_rem_u, wc%bt_rem_v)
   !$acc enter data copyin(wc%bt_zeta_corner, wc%bt_ke_centre)
   !$acc enter data copyin(wc%ubt_sum, wc%vbt_sum, wc%eta_sum, wc%uhbt_sum, wc%vhbt_sum)
   !$acc enter data copyin(wp_)
   !$acc enter data copyin(wp_%bt_eta, wp_%bt_eta_new, wp_%bt_H_ref, wp_%bt_ubt, wp_%bt_vbt)
   !$acc enter data copyin(wp_%bt_ubt_prev, wp_%bt_vbt_prev, wp_%bt_rem_u, wp_%bt_rem_v)
   !$acc enter data copyin(wp_%bt_zeta_corner, wp_%bt_ke_centre)
   !$acc enter data copyin(wp_%ubt_sum, wp_%vbt_sum, wp_%eta_sum, wp_%uhbt_sum, wp_%vhbt_sum)
   !$acc enter data copyin(wf%bt_eta, wf%bt_eta_new, wf%bt_H_ref, wf%bt_ubt, wf%bt_vbt)
   !$acc enter data copyin(wf%bt_ubt_prev, wf%bt_vbt_prev, wf%bt_rem_u, wf%bt_rem_v)
   !$acc enter data copyin(wf%bt_zeta_corner, wf%bt_ke_centre)
   !$acc enter data copyin(wf%ubt_sum, wf%vbt_sum, wf%eta_sum, wf%uhbt_sum, wf%vhbt_sum)
   !$acc enter data copyin(met%dy_cu, met%dx_cv, met%iareaT, met%areaCu, met%areaCv, &
   !$acc                   met%dxCu, met%dyCv, met%idxCu, met%idyCv, met%iareaBu)
   !$acc enter data copyin(cor%f_corner)
   !$acc enter data copyin(force_u, force_v)

   ! ---- variant A: do concurrent (production idiom: acc kernels async(1)) ----
   do rep = 1, n_warm
      call btstep_nonlinear_closed(grid, met, w, cor, force_u, force_v, n_inner, DT_INNER)
   end do
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call btstep_nonlinear_closed(grid, met, w, cor, force_u, force_v, n_inner, DT_INNER)
   end do
   !$acc wait
   t1 = wall()
   ms_dc = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variant A2: SIGNATURE FIX — same body, plain array dummies ----
   ! The caller still holds state in a derived type; only the KERNEL's dummies
   ! are plain. That is the whole change.
   do rep = 1, n_warm
      call call_plain(wp_)
   end do
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call call_plain(wp_)
   end do
   !$acc wait
   t1 = wall()
   ms_pl = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variant A3: FUSED Fortran — same body, 11 loops -> 5 ----
   do rep = 1, n_warm
      call call_fused()
   end do
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call call_fused()
   end do
   !$acc wait
   t1 = wall()
   ms_fus = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variants B/C/D: CUDA faithful / fused / graph ----
   call fill_args(args, wc)
   call init_state(wc)
   !$acc update device(wc%bt_eta, wc%bt_ubt, wc%bt_vbt, wc%bt_ubt_prev, wc%bt_vbt_prev)
   !$acc update device(wc%bt_zeta_corner, wc%bt_ke_centre)
   ms_f  = time_cuda(0, n_inner, n_reps, n_warm, args)
   call init_state(wc)
   !$acc update device(wc%bt_eta, wc%bt_ubt, wc%bt_vbt, wc%bt_ubt_prev, wc%bt_vbt_prev)
   !$acc update device(wc%bt_zeta_corner, wc%bt_ke_centre)
   ms_fu = time_cuda(1, n_inner, n_reps, n_warm, args)
   call init_state(wc)
   eta_ini = wc%bt_eta        ! the seed the graph run starts from -> dump_ref
   !$acc update device(wc%bt_eta, wc%bt_ubt, wc%bt_vbt, wc%bt_ubt_prev, wc%bt_vbt_prev)
   !$acc update device(wc%bt_zeta_corner, wc%bt_ke_centre)
   ms_g  = time_cuda(2, n_inner, n_reps, n_warm, args)

   !$acc update self(w%bt_eta, w%bt_ubt)
   !$acc update self(wp_%bt_eta)
   !$acc update self(wf%bt_eta)
   !$acc update self(wc%bt_eta, wc%bt_ubt)

   ! Optional: dump the FORTRAN-DRIVEN CUDA result so the NATIVE C++ driver
   ! (btstep_native.cu -- cudaMalloc + main, no OpenACC) can check itself
   ! against it bit-for-bit. Off unless BTSTEP_REF_OUT is set, and written
   ! after every timed loop, so it cannot perturb a number. `wc` holds the
   ! mode-2 (graph) state, which is the mode btstep_native compares.
   call dump_ref(wc)

   ! DC vs CUDA: the CUDA modes ran from the same initial state for the same
   ! n_inner substeps, so eta/ubt should agree to FMA-contraction level.
   d_eta = 0.0_wp; d_u = 0.0_wp; r_eta = 0.0_wp; d_pl = 0.0_wp; d_fus = 0.0_wp
   do j = 1, ny
      do i = 1, nx
         d_eta = max(d_eta, abs(w%bt_eta(i, j) - wc%bt_eta(i, j)))
         d_pl = max(d_pl, abs(w%bt_eta(i, j) - wp_%bt_eta(i, j)))
         d_fus = max(d_fus, abs(w%bt_eta(i, j) - wf%bt_eta(i, j)))
         r_eta = max(r_eta, abs(w%bt_eta(i, j)))
      end do
   end do
   do j = 1, ny
      do i = 1, nx + 1
         d_u = max(d_u, abs(w%bt_ubt(i, j) - wc%bt_ubt(i, j)))
      end do
   end do

   write (output_unit, '(a,f10.4,a)') '  DC   (do concurrent, 11 kern/substep) : ', ms_dc, ' ms/call'
   write (output_unit, '(a,f10.4,a)') '  DC + SIGNATURE FIX (plain dummies)    : ', ms_pl, ' ms/call'
   write (output_unit, '(a,f10.4,a)') '  DC + SIG FIX + FUSED (5 kern/substep) : ', ms_fus, ' ms/call'
   write (output_unit, '(a,f10.4,a)') '  CUDA faithful (11 kern/substep)       : ', ms_f, ' ms/call'
   write (output_unit, '(a,f10.4,a)') '  CUDA fused    ( 5 kern/substep)       : ', ms_fu, ' ms/call'
   write (output_unit, '(a,f10.4,a)') '  CUDA graph    ( 5 kern, 1 graph)      : ', ms_g, ' ms/call'
   write (output_unit, '(a)') ''
   write (output_unit, '(a,f10.3,a)') '  dc / dc-plain       (SIGNATURE cost)  : ', ms_dc/ms_pl, ' x'
   write (output_unit, '(a,f10.3,a)') '  dc-plain / cuda-faithful (CODEGEN)    : ', ms_pl/ms_f, ' x'
   write (output_unit, '(a,f10.3,a)') '  dc-plain / cuda-graph (residual)      : ', ms_pl/ms_g, ' x'
   write (output_unit, '(a,f10.3,a)') '  dc / cuda-faithful  (was CODEGEN)     : ', ms_dc/ms_f, ' x'
   write (output_unit, '(a,f10.3,a)') '  faithful / fused    (FUSION win)      : ', ms_f/ms_fu, ' x'
   write (output_unit, '(a,f10.3,a)') '  fused / graph       (LAUNCH residual) : ', ms_fu/ms_g, ' x'
   write (output_unit, '(a,f10.3,a)') '  dc / cuda-graph     (TOTAL headroom)  : ', ms_dc/ms_g, ' x'
   write (output_unit, '(a)') ''
   write (output_unit, '(a,f10.3,a)') '  dc-plain / dc-fused (Fortran FUSION)   : ', ms_pl/ms_fus, ' x'
   write (output_unit, '(a,f10.3,a)') '  dc-fused / cuda-fused                 : ', ms_fus/ms_fu, ' x'
   write (output_unit, '(a,f10.3,a)') '  dc-fused / cuda-graph  <-- THE ANSWER : ', ms_fus/ms_g, ' x'
   write (output_unit, '(a,f10.3,a)') '  dc(prod) / dc-fused (free, in Fortran) : ', ms_dc/ms_fus, ' x'
   write (output_unit, '(a)') ''
   write (output_unit, '(a,es12.5,a,es12.5)') '  DC vs CUDA  max|d eta| : ', d_eta, '   |eta|~', r_eta
   write (output_unit, '(a,es12.5)') '              max|d ubt| : ', d_u
   write (output_unit, '(a,es12.5,a)') '  DC vs DC-plain max|d eta| : ', d_pl, '  (expect 0 -- same body)'
   ! The fused merges are single-assignment (`ubt = wall ? 0 : computed`) and
   ! eta_sum moves to a point where eta is already final -- so fusion must be
   ! BIT-identical, not merely close. Anything non-zero here means a merge
   ! changed the answer and the speedup is void.
   write (output_unit, '(a,es12.5,a)') '  DC vs DC-FUSED max|d eta| : ', d_fus, '  (expect 0 -- merges are exact)'
   if (d_fus /= 0.0_wp) then
      write (output_unit, '(a)') '  *** FUSION CHANGED THE ANSWER -- speedup is invalid ***'
   end if
   if (d_eta <= 1.0e-12_wp*max(r_eta, 1.0e-30_wp)) then
      write (output_unit, '(a)') '              agreement  : OK'
   else
      write (output_unit, '(a)') '              agreement  : *** SUSPECT -- port bug? ***'
   end if
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

   !! Fill the C struct with DEVICE pointers for every array.
   subroutine fill_args(a, s)
      type(bt_args_t), intent(out) :: a
      type(bt_work_t), intent(inout), target :: s
      !$acc host_data use_device(s%bt_eta, s%bt_eta_new, s%bt_H_ref, s%bt_ubt, s%bt_vbt, &
      !$acc                      s%bt_ubt_prev, s%bt_vbt_prev, s%bt_rem_u, s%bt_rem_v, &
      !$acc                      s%bt_zeta_corner, s%bt_ke_centre, s%ubt_sum, s%vbt_sum, &
      !$acc                      s%eta_sum, s%uhbt_sum, s%vhbt_sum, &
      !$acc                      met%dy_cu, met%dx_cv, met%iareaT, met%areaCu, met%areaCv, &
      !$acc                      met%dxCu, met%dyCv, met%idxCu, met%idyCv, met%iareaBu, &
      !$acc                      cor%f_corner, force_u, force_v)
      a%eta = c_loc(s%bt_eta); a%eta_new = c_loc(s%bt_eta_new); a%H_ref = c_loc(s%bt_H_ref)
      a%ubt = c_loc(s%bt_ubt); a%vbt = c_loc(s%bt_vbt)
      a%ubt_prev = c_loc(s%bt_ubt_prev); a%vbt_prev = c_loc(s%bt_vbt_prev)
      a%rem_u = c_loc(s%bt_rem_u); a%rem_v = c_loc(s%bt_rem_v)
      a%zeta = c_loc(s%bt_zeta_corner); a%ke = c_loc(s%bt_ke_centre)
      a%ubt_sum = c_loc(s%ubt_sum); a%vbt_sum = c_loc(s%vbt_sum); a%eta_sum = c_loc(s%eta_sum)
      a%uhbt_sum = c_loc(s%uhbt_sum); a%vhbt_sum = c_loc(s%vhbt_sum)
      a%dy_cu = c_loc(met%dy_cu); a%dx_cv = c_loc(met%dx_cv); a%iareaT = c_loc(met%iareaT)
      a%areaCu = c_loc(met%areaCu); a%areaCv = c_loc(met%areaCv)
      a%dxCu = c_loc(met%dxCu); a%dyCv = c_loc(met%dyCv)
      a%idxCu = c_loc(met%idxCu); a%idyCv = c_loc(met%idyCv); a%iareaBu = c_loc(met%iareaBu)
      a%f_corner = c_loc(cor%f_corner); a%force_u = c_loc(force_u); a%force_v = c_loc(force_v)
      !$acc end host_data
      a%nx = nx; a%ny = ny; a%nghost = NGH; a%nx_phys = nxp; a%ny_phys = nyp
      a%G = s%g_bt; a%bebt = s%bebt; a%dt = DT_INNER
   end subroutine

   !! Re-seed wc, then time `mode` for n_reps calls.
   real(wp) function time_cuda(mode, n_inner, n_reps, n_warm, a) result(msout)
      integer, intent(in) :: mode, n_inner, n_reps, n_warm
      type(bt_args_t), intent(in) :: a
      real(wp) :: ta, tb
      integer :: r
      do r = 1, n_warm
         call btstep_cuda_launch(a, n_inner, mode)
      end do
      ta = wall()
      do r = 1, n_reps
         call btstep_cuda_launch(a, n_inner, mode)
      end do
      tb = wall()
      msout = (tb - ta)*1000.0_wp/real(n_reps, wp)
   end function

   subroutine call_plain(s)
      type(bt_work_t), intent(inout) :: s
      call btstep_plain(nx, ny, NGH, nxp, nyp, s%g_bt, s%bebt, DT_INNER, n_inner, &
                        s%bt_eta, s%bt_eta_new, s%bt_H_ref, s%bt_ubt, s%bt_vbt, &
                        s%bt_ubt_prev, s%bt_vbt_prev, s%bt_rem_u, s%bt_rem_v, &
                        s%bt_zeta_corner, s%bt_ke_centre, s%ubt_sum, s%vbt_sum, s%eta_sum, &
                        s%uhbt_sum, s%vhbt_sum, met%dy_cu, met%dx_cv, met%iareaT, &
                        met%areaCu, met%areaCv, met%dxCu, met%dyCv, met%idxCu, met%idyCv, &
                        met%iareaBu, cor%f_corner, force_u, force_v)
   end subroutine

   subroutine call_fused()
      call btstep_fused(nx, ny, NGH, nxp, nyp, wf%g_bt, wf%bebt, DT_INNER, n_inner, &
                        wf%bt_eta, wf%bt_eta_new, wf%bt_H_ref, wf%bt_ubt, wf%bt_vbt, &
                        wf%bt_ubt_prev, wf%bt_vbt_prev, wf%bt_rem_u, wf%bt_rem_v, &
                        wf%bt_zeta_corner, wf%bt_ke_centre, wf%ubt_sum, wf%vbt_sum, wf%eta_sum, &
                        wf%uhbt_sum, wf%vhbt_sum, met%dy_cu, met%dx_cv, met%iareaT, &
                        met%areaCu, met%areaCv, met%dxCu, met%dyCv, met%idxCu, met%idyCv, &
                        met%iareaBu, cor%f_corner, force_u, force_v)
   end subroutine

   !! Raw dump for btstep_native.cu. Stream, column-major real64, no record
   !! marks -- the allocations byte for byte:
   !!    nx, ny            (2 default integers)
   !!    INPUTS : eta_ini (nx,ny), force_u (nx+1,ny), f_corner (nx+1,ny+1)
   !!    OUTPUTS: bt_eta  (nx,ny), bt_ubt  (nx+1,ny)   -- the mode-2 result
   !!
   !! The three INPUTS are dumped because they are the only initial arrays not
   !! exactly representable: eta_ini goes through exp(), force_u through sin(),
   !! and f_corner through a multiply-add nvfortran may contract to an FMA.
   !! nvfortran's libm and glibc's libm disagree by ~1 ulp on exp/sin (measured:
   !! 29301/145137 of eta_ini, 46/303 of force_u), so a native driver that
   !! RECOMPUTES them cannot bit-match -- over (nwarm+nreps)*n_inner substeps
   !! that 1 ulp amplifies into the 1e-15 range. Handing the native driver these
   !! three arrays isolates the question that matters -- do the two drivers run
   !! the same kernels on the same numbers? -- from a libm rounding difference
   !! that is not about CUDA at all. Everything else (metrics, H_ref, rem_u/v,
   !! the zeroed arrays) is exact in both languages and needs no dump.
   !!
   !! No-op unless BTSTEP_REF_OUT names a file.
   subroutine dump_ref(s)
      type(bt_work_t), intent(in) :: s
      character(len=512) :: path
      integer :: u, st, ln
      call get_environment_variable('BTSTEP_REF_OUT', path, ln, st)
      if (st /= 0 .or. ln == 0) return
      open (newunit=u, file=path(1:ln), access='stream', form='unformatted', &
            status='replace', iostat=st)
      if (st /= 0) then
         write (output_unit, '(a,a)') '  *** could not write ref dump: ', path(1:ln)
         return
      end if
      write (u) nx, ny
      write (u) eta_ini
      write (u) force_u
      write (u) cor%f_corner
      write (u) s%bt_eta
      write (u) s%bt_ubt
      close (u)
      write (output_unit, '(a,a)') '  ref dump written : ', path(1:ln)
   end subroutine

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
   end function

   function wall() result(t)
      real(wp) :: t
      integer(int64) :: cnt, rate
      call system_clock(cnt, rate)
      t = real(cnt, wp)/real(rate, wp)
   end function

end program btstep_bench
