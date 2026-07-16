!! VARIANT 1 of 2 — OpenACC owns the device memory, `do concurrent` computes.
!!
!!   c(i,j,k) = alpha*a(i,j,k) + b(i,j,k)
!!
!! Data model: `-gpu=mem:separate` (NO managed/unified memory). There are no
!! implicit host<->device copies. Every array a kernel touches must have been
!! explicitly mapped or the kernel silently reads stale host memory — no crash,
!! just wrong numbers. So `!$acc enter data` here is doing real work, not
!! decoration.
!!
!! The pairing with daxpy_cuda.F90 is the point: identical allocation, identical
!! mapping, identical timing harness. ONLY the compute differs.
program daxpy_dc
   use, intrinsic :: iso_fortran_env, only: output_unit
   use daxpy_common, only: wp, nx, ny, nz, alpha, n_reps, &
                           fill_inputs, verify, report, wall_seconds
   implicit none

   real(wp), allocatable :: a(:, :, :), b(:, :, :), c(:, :, :)
   real(wp) :: t0, t1, max_err
   integer :: i, j, k, rep, n_bad

   allocate (a(nx, ny, nz), b(nx, ny, nz), c(nx, ny, nz))
   call fill_inputs(a, b)
   c = 0.0_wp

   ! ---- OpenACC allocates + populates device memory --------------------
   ! copyin(a,b): host values are needed on the device.
   ! create(c)  : device-written before it is ever read, so copying the host's
   !              zeros over would be pure waste. (Getting this backwards is
   !              free correctness and lost bandwidth; getting `copyin` wrong
   !              on `a`/`b` is silent garbage.)
   !$acc enter data copyin(a, b) create(c)

   ! ---- warm-up: pay JIT + first-launch cost outside the timer ----------
   do concurrent(k=1:nz, j=1:ny, i=1:nx)
      c(i, j, k) = alpha*a(i, j, k) + b(i, j, k)
   end do
   !$acc wait

   ! ---- timed region ---------------------------------------------------
   t0 = wall_seconds()
   do rep = 1, n_reps
      ! THE KERNEL. nvfortran -stdpar=gpu lowers this to a GPU kernel; the
      ! arrays are already device-resident thanks to the `enter data` above,
      ! so no transfer happens here.
      do concurrent(k=1:nz, j=1:ny, i=1:nx)
         c(i, j, k) = alpha*a(i, j, k) + b(i, j, k)
      end do
   end do
   !$acc wait
   t1 = wall_seconds()

   ! ---- pull the answer back and check it ------------------------------
   !$acc update self(c)
   !$acc exit data delete(a, b, c)

   call verify(c, max_err, n_bad)
   call report('do concurrent (-stdpar=gpu)', (t1 - t0)/real(n_reps, wp), max_err, n_bad)

   deallocate (a, b, c)
   if (n_bad /= 0) error stop 1
end program daxpy_dc
