!! Flux-kernel bench: `do concurrent` vs a faithful CUDA C port of the SAME kernel.
!!
!! WHY THIS EXISTS: the daxpy MRE next door found do-concurrent == CUDA C == 810
!! GB/s, but that proved almost nothing — a bandwidth-bound elementwise op cannot
!! distinguish compilers. This runs the ocean model's PRODUCTION HLL flux kernel
!! (kernel_flux.F90, byte-identical to the shipped file), which is where they
!! could actually diverge:
!!
!!   * a cross-procedure `call flux_cell(...)` INSIDE do concurrent — CLAUDE.md
!!     warns NVHPC does not inline `pure` helpers across module boundaries
!!   * a 7-scalar `local()` locality clause
!!   * ~40 live doubles per cell -> real register pressure, occupancy limits
!!   * minmod + dry-tolerance + HLL wave-speed branches
!!   * a 5-point stencil (i-2..i+2) -> reuse the cache, or don't
!!
!! Both kernels read the SAME device arrays (OpenACC-owned, handed to CUDA via
!! host_data use_device), so the only variable is who generated the code.
!!
!! STATE: 2-D dam break on a wet bed. All-wet by construction, so the wet/dry
!! branches do NOT fire and warps do not diverge on them — this measures the
!! kernel, not the dry path. The discontinuity still exercises minmod and the
!! Riemann solver genuinely.
program flux_bench
   use, intrinsic :: iso_fortran_env, only: real64, int64, output_unit
   use, intrinsic :: iso_c_binding, only: c_double, c_int
   use constants, only: wp
   use grid, only: hgrid_t
   use kernel_flux, only: compute_flux_hll
   use kernel_flux_acc, only: compute_flux_hll_acc
   use kernel_flux_nest, only: compute_flux_hll_nest
   use kernel_flux_flat, only: compute_flux_hll_flat
   use kernel_flux_flatdc, only: compute_flux_hll_flatdc
   use kernel_flux_dims, only: compute_flux_hll_dims
   implicit none

   interface
      subroutine flux_hll_cuda_launch(h, hu, hv, b, flux_h, flux_hu, flux_hv, &
                                      mass_flux_x, mass_flux_y, &
                                      nx, ny, nghost, dx, dy, sync) &
         bind(C, name="flux_hll_cuda_launch")
         import :: c_double, c_int
         implicit none
         real(c_double), intent(in) :: h(*), hu(*), hv(*), b(*)
         real(c_double), intent(inout) :: flux_h(*), flux_hu(*), flux_hv(*)
         real(c_double), intent(inout) :: mass_flux_x(*), mass_flux_y(*)
         integer(c_int), value :: nx, ny, nghost, sync
         real(c_double), value :: dx, dy
      end subroutine flux_hll_cuda_launch
   end interface

   integer, parameter :: NPHYS_DEF = 4096  ! interior cells per side
   integer, parameter :: NGH = 2           ! stencil reads i-2..i+2
   integer, parameter :: N_REPS_DEF = 20

   ! Overridable via FLUX_REPS. 20 reps is fine at 4096^2 (5 ms each) and far too
   ! few at small sizes, where a rep is microseconds and the whole loop lands
   ! inside the noise. Small grids need thousands of reps to say anything.
   integer :: n_reps

   ! Interior size, overridable via FLUX_NPHYS_X / FLUX_NPHYS_Y.
   !
   ! WHY THIS IS A VARIABLE NOW: 4096^2 is a stress size -- nothing runs there.
   ! Production is ~473 x 297 (see the model's configs), where the kernel is ~100x
   ! cheaper and per-CALL host-side costs stop being hidden behind it. Comparing
   ! the Fortran driver against flux_native at 4096^2 alone can only ever measure
   ! the regime where host overhead is invisible by construction.
   !
   ! NOT a parameter -> the arrays are already allocatable and grid% is already
   ! runtime, so nothing downstream had to change.
   integer :: nphys_x, nphys_y

   type(hgrid_t) :: grid
   real(wp), allocatable :: h(:, :), hu(:, :), hv(:, :), b(:, :)
   real(wp), allocatable :: fh_dc(:, :), fhu_dc(:, :), fhv_dc(:, :)
   real(wp), allocatable :: mx_dc(:, :), my_dc(:, :)
   real(wp), allocatable :: fh_fl(:, :), fhu_fl(:, :), fhv_fl(:, :)
   real(wp), allocatable :: mx_fl(:, :), my_fl(:, :)
   ! flatdc gets its OWN outputs. It used to share fh_fl with flat and dims, and
   ! dims runs LAST -- so the line printed as "FLAT vs CUDA" was really comparing
   ! DIMS vs CUDA, and flatdc (the headline "portable Fortran beats CUDA" variant)
   ! was never checked against anything. It inherited dims' pass. This is exactly
   ! the shared-output trap RESUME 4 documents; it had simply grown a third victim.
   real(wp), allocatable :: fh_fd(:, :), fhu_fd(:, :), fhv_fd(:, :)
   real(wp), allocatable :: mx_fd(:, :), my_fd(:, :)
   real(wp), allocatable :: fh_dm(:, :), fhu_dm(:, :), fhv_dm(:, :)
   real(wp), allocatable :: mx_dm(:, :), my_dm(:, :)
   real(wp), allocatable :: fh_cu(:, :), fhu_cu(:, :), fhv_cu(:, :)
   real(wp), allocatable :: mx_cu(:, :), my_cu(:, :)
   real(wp) :: t0, t1, dc_ms, cu_ms, ac_ms, nz_ms, fl_ms, fd_ms, dm_ms
   real(wp) :: mx_abs, mx_rel, ref
   integer :: nx, ny, i, j, rep, nbad
   character(len=32) :: dump_env
   integer :: dump_len, dump_stat

   nphys_x = env_int('FLUX_NPHYS_X', NPHYS_DEF)
   nphys_y = env_int('FLUX_NPHYS_Y', NPHYS_DEF)
   n_reps = env_int('FLUX_REPS', N_REPS_DEF)

   grid%nghost = NGH
   grid%nx_total = nphys_x + 2*NGH
   grid%ny_total = nphys_y + 2*NGH
   grid%dx = 10.0_wp
   grid%dy = 10.0_wp
   nx = grid%nx_total
   ny = grid%ny_total

   allocate (h(nx, ny), hu(nx, ny), hv(nx, ny), b(nx, ny))
   allocate (fh_dc(nx, ny), fhu_dc(nx, ny), fhv_dc(nx, ny), mx_dc(nx, ny), my_dc(nx, ny))
   allocate (fh_cu(nx, ny), fhu_cu(nx, ny), fhv_cu(nx, ny), mx_cu(nx, ny), my_cu(nx, ny))
   allocate (fh_fl(nx, ny), fhu_fl(nx, ny), fhv_fl(nx, ny), mx_fl(nx, ny), my_fl(nx, ny))
   allocate (fh_fd(nx, ny), fhu_fd(nx, ny), fhv_fd(nx, ny), mx_fd(nx, ny), my_fd(nx, ny))
   allocate (fh_dm(nx, ny), fhu_dm(nx, ny), fhv_dm(nx, ny), mx_dm(nx, ny), my_dm(nx, ny))

   ! ---- 2-D dam break, wet bed ------------------------------------------
   ! Radial dam: deep inside, shallow outside, both well above
   ! THIN_LAYER_THRESHOLD so no cell ever takes the dry branch. Flat bed keeps
   ! the well-balanced correction exercised but simple.
   do concurrent(j=1:ny, i=1:nx)
      b(i, j) = 0.0_wp
      if ((real(i - nx/2, wp)**2 + real(j - ny/2, wp)**2) < real(nx/4, wp)**2) then
         h(i, j) = 10.0_wp
      else
         h(i, j) = 2.0_wp
      end if
      hu(i, j) = 0.05_wp*real(i, wp)/real(nx, wp)
      hv(i, j) = -0.03_wp*real(j, wp)/real(ny, wp)
   end do
   fh_dc = 0.0_wp; fhu_dc = 0.0_wp; fhv_dc = 0.0_wp; mx_dc = 0.0_wp; my_dc = 0.0_wp
   fh_cu = 0.0_wp; fhu_cu = 0.0_wp; fhv_cu = 0.0_wp; mx_cu = 0.0_wp; my_cu = 0.0_wp
   fh_fl = 0.0_wp; fhu_fl = 0.0_wp; fhv_fl = 0.0_wp; mx_fl = 0.0_wp; my_fl = 0.0_wp
   fh_fd = 0.0_wp; fhu_fd = 0.0_wp; fhv_fd = 0.0_wp; mx_fd = 0.0_wp; my_fd = 0.0_wp
   fh_dm = 0.0_wp; fhu_dm = 0.0_wp; fhv_dm = 0.0_wp; mx_dm = 0.0_wp; my_dm = 0.0_wp

   write (output_unit, '(a)') repeat('=', 66)
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  grid: ', nx, ' x ', ny, ' total (', &
      nphys_x, ' x ', nphys_y, ' interior), nghost=2'
   write (output_unit, '(a,i0)') '  reps: ', n_reps
   write (output_unit, '(a)') '  state: 2-D dam break, wet bed (no dry cells -> no branch divergence)'
   write (output_unit, '(a)') repeat('=', 66)

   !$acc enter data copyin(h, hu, hv, b) &
   !$acc            create(fh_dc, fhu_dc, fhv_dc, mx_dc, my_dc) &
   !$acc            create(fh_cu, fhu_cu, fhv_cu, mx_cu, my_cu) &
   !$acc            create(fh_fl, fhu_fl, fhv_fl, mx_fl, my_fl) &
   !$acc            create(fh_fd, fhu_fd, fhv_fd, mx_fd, my_fd) &
   !$acc            create(fh_dm, fhu_dm, fhv_dm, mx_dm, my_dm)

   ! ---- variant A: do concurrent (the production kernel, verbatim) -------
   call compute_flux_hll(h, hu, hv, b, fh_dc, fhu_dc, fhv_dc, mx_dc, my_dc, grid)  ! warm-up
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call compute_flux_hll(h, hu, hv, b, fh_dc, fhu_dc, fhv_dc, mx_dc, my_dc, grid)
   end do
   !$acc wait
   t1 = wall()
   dc_ms = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variant A2: explicit OpenACC, collapse(2) -> ONE THREAD PER CELL --
   ! Same file, same helpers, same everything -- ONLY the loop directive differs
   ! from variant A. do concurrent gave no handle on geometry; this does.
   call compute_flux_hll_acc(h, hu, hv, b, fh_dc, fhu_dc, fhv_dc, mx_dc, my_dc, grid)
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call compute_flux_hll_acc(h, hu, hv, b, fh_dc, fhu_dc, fhv_dc, mx_dc, my_dc, grid)
   end do
   !$acc wait
   t1 = wall()
   ac_ms = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variant A3: NESTED do concurrent(j) / do concurrent(i) ----------
   call compute_flux_hll_nest(h, hu, hv, b, fh_dc, fhu_dc, fhv_dc, mx_dc, my_dc, grid)
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call compute_flux_hll_nest(h, hu, hv, b, fh_dc, fhu_dc, fhv_dc, mx_dc, my_dc, grid)
   end do
   !$acc wait
   t1 = wall()
   nz_ms = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variant A4: LITERALLY FLAT — no procedures at all ---------------
   call compute_flux_hll_flat(h, hu, hv, b, fh_fl, fhu_fl, fhv_fl, mx_fl, my_fl, grid)
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call compute_flux_hll_flat(h, hu, hv, b, fh_fl, fhu_fl, fhv_fl, mx_fl, my_fl, grid)
   end do
   !$acc wait
   t1 = wall()
   fl_ms = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variant A5: FLAT body + PLAIN do concurrent (no OpenACC!) --------
   call compute_flux_hll_flatdc(h, hu, hv, b, fh_fd, fhu_fd, fhv_fd, mx_fd, my_fd, grid)
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call compute_flux_hll_flatdc(h, hu, hv, b, fh_fd, fhu_fd, fhv_fd, mx_fd, my_fd, grid)
   end do
   !$acc wait
   t1 = wall()
   fd_ms = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variant A6: SIGNATURE FIX — plain nx/ny dummies, body unchanged -----
   call compute_flux_hll_dims(h, hu, hv, b, fh_dm, fhu_dm, fhv_dm, mx_dm, my_dm, &
                              nx, ny, NGH, grid%dx, grid%dy)
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call compute_flux_hll_dims(h, hu, hv, b, fh_dm, fhu_dm, fhv_dm, mx_dm, my_dm, &
                                 nx, ny, NGH, grid%dx, grid%dy)
   end do
   !$acc wait
   t1 = wall()
   dm_ms = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variant B: faithful CUDA C port of the same kernel ---------------
   !$acc host_data use_device(h, hu, hv, b, fh_cu, fhu_cu, fhv_cu, mx_cu, my_cu)
   call flux_hll_cuda_launch(h, hu, hv, b, fh_cu, fhu_cu, fhv_cu, mx_cu, my_cu, &
                             nx, ny, NGH, grid%dx, grid%dy, 1)   ! warm-up
   !$acc end host_data
   t0 = wall()
   do rep = 1, n_reps
      !$acc host_data use_device(h, hu, hv, b, fh_cu, fhu_cu, fhv_cu, mx_cu, my_cu)
      call flux_hll_cuda_launch(h, hu, hv, b, fh_cu, fhu_cu, fhv_cu, mx_cu, my_cu, &
                                nx, ny, NGH, grid%dx, grid%dy, 0)
      !$acc end host_data
   end do
   !$acc host_data use_device(h, hu, hv, b, fh_cu, fhu_cu, fhv_cu, mx_cu, my_cu)
   call flux_hll_cuda_launch(h, hu, hv, b, fh_cu, fhu_cu, fhv_cu, mx_cu, my_cu, &
                             nx, ny, NGH, grid%dx, grid%dy, 1)   ! drain the queue
   !$acc end host_data
   t1 = wall()
   cu_ms = (t1 - t0)*1000.0_wp/real(n_reps + 1, wp)

   !$acc update self(fh_dc, fhu_dc, fhv_dc, mx_dc, my_dc)
   !$acc update self(fh_cu, fhu_cu, fhv_cu, mx_cu, my_cu)
   !$acc update self(fh_fl, fhu_fl, fhv_fl, mx_fl, my_fl)
   !$acc update self(fh_fd, fhu_fd, fhv_fd, mx_fd, my_fd)
   !$acc update self(fh_dm, fhu_dm, fhv_dm, mx_dm, my_dm)
   !$acc exit data delete(h, hu, hv, b, fh_dc, fhu_dc, fhv_dc, mx_dc, my_dc, &
   !$acc                  fh_cu, fhu_cu, fhv_cu, mx_cu, my_cu)

   ! ---- do the two kernels AGREE? ---------------------------------------
   ! Not bit-exact: FMA contraction differs between nvfortran and nvcc, so the
   ! last ~1-2 ulp legitimately move. Anything beyond ~1e-12 relative is a PORT
   ! BUG, not rounding — and would mean the timing compares two different
   ! algorithms.
   mx_abs = 0.0_wp
   mx_rel = 0.0_wp
   nbad = 0
   do j = NGH + 1, ny - NGH
      do i = NGH + 1, nx - NGH
         call track(fh_dc(i, j), fh_cu(i, j), mx_abs, mx_rel, nbad)
         call track(fhu_dc(i, j), fhu_cu(i, j), mx_abs, mx_rel, nbad)
         call track(fhv_dc(i, j), fhv_cu(i, j), mx_abs, mx_rel, nbad)
      end do
   end do

   ! ---- EVERY variant vs CUDA, each against ITS OWN output ---------------
   ! Each variant now owns its arrays, so each line below is evidence about the
   ! variant it names. Previously flat/flatdc/dims all wrote fh_fl and only the
   ! last writer (dims) was ever compared -- the single line printed as
   ! "FLAT vs CUDA" was really "dims vs CUDA", and flat and flatdc rode along on
   ! its pass. flatdc is the variant the README calls the winner, so it was the
   ! worst possible one to leave unchecked.
   write (output_unit, '(a)') '  agreement vs CUDA C (max rel diff over the interior):'
   call report('do concurrent (shipped)', fh_dc, fhu_dc, fhv_dc)
   call report('FLAT + collapse(2)     ', fh_fl, fhu_fl, fhv_fl)
   call report('FLAT + PLAIN do conc.  ', fh_fd, fhu_fd, fhv_fd)
   call report('SIGNATURE FIX (dims)   ', fh_dm, fhu_dm, fhv_dm)
   write (output_unit, '(a)') ''
   write (output_unit, '(a,f10.4,a)') '  do concurrent (nvfortran -stdpar=gpu) : ', dc_ms, ' ms/rep'
   write (output_unit, '(a,f10.4,a)') '  do concurrent NESTED dc(j)/dc(i)      : ', nz_ms, ' ms/rep'
   write (output_unit, '(a,f10.4,a)') '  OpenACC collapse(2) vector(VLEN)      : ', ac_ms, ' ms/rep'
   write (output_unit, '(a,f10.4,a)') '  FLAT (no procedures) + collapse(2)    : ', fl_ms, ' ms/rep'
   write (output_unit, '(a,f10.4,a)') '  FLAT + PLAIN do concurrent (portable) : ', fd_ms, ' ms/rep'
   write (output_unit, '(a,f10.4,a)') '  SIGNATURE FIX (plain nx/ny dummies)   : ', dm_ms, ' ms/rep'
   write (output_unit, '(a,f10.4,a)') '  CUDA C  (faithful port, nvcc)         : ', cu_ms, ' ms/rep'
   write (output_unit, '(a,f10.3,a)') '  ratio  dc / cuda                      : ', dc_ms/cu_ms, ' x'
   write (output_unit, '(a)') ''
   write (output_unit, '(a,es12.5)') '  max |dc - cuda|        : ', mx_abs
   write (output_unit, '(a,es12.5)') '  max relative diff      : ', mx_rel
   if (mx_rel < 1.0e-12_wp) then
      write (output_unit, '(a)') '  agreement              : OK (<1e-12 rel -> FMA contraction only)'
   else
      write (output_unit, '(a,i0,a)') '  agreement              : *** SUSPECT *** ', nbad, &
         ' cells >1e-12 rel -- likely a PORT BUG, not rounding'
   end if
   write (output_unit, '(a)') repeat('=', 66)

   ! ---- reference dump for the NATIVE C++/CUDA driver (flux_native.cu) ----
   ! Off unless FLUX_DUMP_REF is set, so the default run is untouched.
   !
   ! Dumps the CUDA-via-Fortran variant's arrays. Those five (fh_cu, fhu_cu,
   ! fhv_cu, mx_cu, my_cu) are written by NO other variant -- which matters,
   ! because the shared-array trap above (see the FLAT check) means a shared
   ! array's contents belong to whoever ran last, and dumping one would hand
   ! flux_native a reference from the wrong kernel.
   !
   ! The inputs go in first. flux_native re-derives the same state from scratch
   ! in C, so shipping h/hu/hv/b lets it prove its init matches BEFORE it blames
   ! any output difference on the port.
   call get_environment_variable('FLUX_DUMP_REF', dump_env, dump_len, dump_stat)
   if (dump_stat == 0 .and. dump_len > 0) then
      call dump_ref()
   end if

   deallocate (h, hu, hv, b, fh_dc, fhu_dc, fhv_dc, mx_dc, my_dc)
   deallocate (fh_cu, fhu_cu, fhv_cu, mx_cu, my_cu)
   deallocate (fh_fl, fhu_fl, fhv_fl, mx_fl, my_fl, fh_fd, fhu_fd, fhv_fd, mx_fd, my_fd)
   deallocate (fh_dm, fhu_dm, fhv_dm, mx_dm, my_dm)

contains

   ! Integer from the environment, or `fallback` if unset/blank/unparseable.
   ! Silent fallback is deliberate: a typo'd size must not change what is
   ! measured without saying so, and the grid line is printed either way.
   integer function env_int(name, fallback) result(v)
      character(len=*), intent(in) :: name
      integer, intent(in) :: fallback
      character(len=32) :: buf
      integer :: ln, st, ios
      v = fallback
      call get_environment_variable(name, buf, ln, st)
      if (st /= 0 .or. ln <= 0) return
      read (buf(1:ln), *, iostat=ios) v
      if (ios /= 0 .or. v <= 0) v = fallback
   end function env_int

   ! Unformatted STREAM: no record markers, so C can fread() it directly. A
   ! sequential-access file would interleave 4-byte lengths between the arrays
   ! and flux_native would read garbage one word out of step.
   ! Order must stay in lockstep with flux_native.cu's `names[9]`.
   subroutine dump_ref()
      character(len=64) :: fname
      integer :: u
      write (fname, '(a,i0,a,i0,a,i0,a)') 'flux_ref_', nx, 'x', ny, 'x', NGH, '.bin'
      open (newunit=u, file=trim(fname), form='unformatted', access='stream', status='replace')
      write (u) h
      write (u) hu
      write (u) hv
      write (u) b
      write (u) fh_cu
      write (u) fhu_cu
      write (u) fhv_cu
      write (u) mx_cu
      write (u) my_cu
      ! Appended AFTER the 9 flux_native reads, so flux_native is unaffected.
      ! Present so two nvfortran builds (e.g. default vs -Kieee) can be compared
      ! against each other with cmp(1): if flatdc's bits are identical across
      ! builds, a flag that changed its SPEED did not change its ARITHMETIC.
      write (u) fh_fd
      write (u) fhu_fd
      write (u) fhv_fd
      close (u)
      write (output_unit, '(a,a,a,f6.2,a)') '  reference dumped: ', trim(fname), '  (', &
         real(9_int64*int(nx, int64)*int(ny, int64)*8_int64, wp)/1.0e9_wp, ' GB) -> ./flux_native'
   end subroutine dump_ref

   ! One variant vs the CUDA port, over the interior, on all three fluxes.
   subroutine report(label, fh, fhu, fhv)
      character(len=*), intent(in) :: label
      real(wp), intent(in) :: fh(:, :), fhu(:, :), fhv(:, :)
      real(wp) :: amax, rmax
      integer :: nb, ii, jj
      amax = 0.0_wp; rmax = 0.0_wp; nb = 0
      do jj = NGH + 1, ny - NGH
         do ii = NGH + 1, nx - NGH
            call track(fh(ii, jj), fh_cu(ii, jj), amax, rmax, nb)
            call track(fhu(ii, jj), fhu_cu(ii, jj), amax, rmax, nb)
            call track(fhv(ii, jj), fhv_cu(ii, jj), amax, rmax, nb)
         end do
      end do
      if (rmax == 0.0_wp) then
         write (output_unit, '(a,a,a)') '    ', label, ' : BIT-IDENTICAL'
      else if (rmax < 1.0e-12_wp) then
         write (output_unit, '(a,a,a,es10.3,a)') '    ', label, ' : OK  max rel ', rmax, &
            ' (FMA contraction only)'
      else
         write (output_unit, '(a,a,a,es10.3,a,i0,a)') '    ', label, ' : *** SUSPECT *** max rel ', &
            rmax, '  (', nb, ' values >1e-12 -- likely a PORT BUG, not rounding)'
      end if
   end subroutine report

   subroutine track(a, c, amax, rmax, nb)
      real(wp), intent(in) :: a, c
      real(wp), intent(inout) :: amax, rmax
      integer, intent(inout) :: nb
      real(wp) :: d, r, scale
      d = abs(a - c)
      if (d > amax) amax = d
      scale = max(abs(a), abs(c))
      if (scale > 1.0e-30_wp) then
         r = d/scale
         if (r > rmax) rmax = r
         if (r > 1.0e-12_wp) nb = nb + 1
      end if
   end subroutine track

   function wall() result(t)
      real(wp) :: t
      integer(int64) :: cnt, rate
      call system_clock(cnt, rate)
      t = real(cnt, wp)/real(rate, wp)
   end function wall

end program flux_bench
