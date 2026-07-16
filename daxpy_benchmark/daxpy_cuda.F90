!! VARIANT 2 of 2 — OpenACC owns the device memory, PURE CUDA C computes.
!!
!!   c(i,j,k) = alpha*a(i,j,k) + b(i,j,k)
!!
!! This is the interop the MRE exists to demonstrate. Allocation, mapping and
!! lifetime are 100% OpenACC. The kernel is 100% CUDA C (daxpy_kernel.cu) and
!! knows nothing about OpenACC. The bridge is exactly one directive:
!!
!!     !$acc host_data use_device(a, b, c)
!!
!! Inside that region, the names a/b/c evaluate to their DEVICE addresses
!! instead of their host addresses. So a plain `bind(C)` call — which passes
!! Fortran arrays by reference, i.e. passes an address — hands CUDA C a device
!! pointer. No cudaMalloc, no cudaMemcpy, no CUDA Fortran, no `device` attribute.
!!
!! Get this wrong (drop the host_data, or reference the array outside the
!! region) and you pass a HOST pointer to a CUDA kernel. Under mem:separate
!! that is usually an "illegal address" abort, but it can also just corrupt
!! memory — hence `daxpy3d_is_device_ptr` below, which asserts the contract
!! rather than trusting it.
program daxpy_cuda
   use, intrinsic :: iso_fortran_env, only: output_unit
   use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr, c_loc
   use daxpy_common, only: wp, nx, ny, nz, alpha, n_reps, &
                           fill_inputs, verify, report, wall_seconds
   implicit none

   ! ---- the CUDA C interface -------------------------------------------
   ! c/a/b are Fortran arrays passed BY REFERENCE (the bind(C) default for a
   ! dummy without `value`), so C receives `double*`. alpha and the extents
   ! carry `value`, so C receives them by value. That is the entire ABI.
   interface
      subroutine daxpy3d_cuda_launch(c, a, b, alpha, nx, ny, nz, sync) &
         bind(C, name="daxpy3d_cuda_launch")
         import :: c_double, c_int
         implicit none
         real(c_double), intent(inout) :: c(*)
         real(c_double), intent(in) :: a(*), b(*)
         real(c_double), value :: alpha
         integer(c_int), value :: nx, ny, nz, sync
      end subroutine daxpy3d_cuda_launch

      integer(c_int) function daxpy3d_is_device_ptr(p) &
         bind(C, name="daxpy3d_is_device_ptr")
         import :: c_ptr, c_int
         implicit none
         type(c_ptr), value :: p
      end function daxpy3d_is_device_ptr
   end interface

   real(wp), allocatable, target :: a(:, :, :), b(:, :, :), c(:, :, :)
   real(wp) :: t0, t1, max_err
   integer :: rep, n_bad, is_dev, is_dev_host

   allocate (a(nx, ny, nz), b(nx, ny, nz), c(nx, ny, nz))
   call fill_inputs(a, b)
   c = 0.0_wp

   ! ---- OpenACC allocates + populates device memory --------------------
   ! IDENTICAL to daxpy_dc.F90. CUDA C never allocates anything.
   !$acc enter data copyin(a, b) create(c)

   ! ---- prove the bridge is load-bearing, both directions ---------------
   ! Diagnostic, not required for the daxpy — but it turns "I think
   ! use_device sorts this out" into a checked fact, and it is the first thing
   ! to reach for when an interop MRE misbehaves.
   !
   ! Checking BOTH sides is the point. `a` is one array, asked about twice in
   ! consecutive statements; the ONLY difference is the directive. Outside, it
   ! is the host allocation. Inside, it is the device allocation. That contrast
   ! *is* the mechanism, and it is why passing `a` to CUDA C without the
   ! directive hands the kernel a host pointer.
   is_dev_host = daxpy3d_is_device_ptr(c_loc(a))

   !$acc host_data use_device(a)
   is_dev = daxpy3d_is_device_ptr(c_loc(a))
   !$acc end host_data

   write (output_unit, '(a,i0,a)') '  ptr check      : OUTSIDE host_data, a is device ptr = ', &
      is_dev_host, '  (expect 0 -> host address)'
   write (output_unit, '(a,i0,a)') '  ptr check      : INSIDE  host_data, a is device ptr = ', &
      is_dev, '  (expect 1 -> device address)'
   if (is_dev /= 1) then
      write (output_unit, '(a)') '  *** host_data use_device did NOT yield a device pointer ***'
      error stop 2
   end if
   if (is_dev_host /= 0) then
      ! Would mean managed memory is in play and the host pointer is also
      ! device-addressable — in which case this MRE is not demonstrating what
      ! it claims, because a missing host_data would appear to "work".
      write (output_unit, '(a)') '  *** host pointer is device-addressable: mem:separate not in effect? ***'
      error stop 3
   end if

   ! ---- warm-up: JIT + first-launch cost outside the timer --------------
   !$acc host_data use_device(a, b, c)
   call daxpy3d_cuda_launch(c, a, b, alpha, nx, ny, nz, 1)
   !$acc end host_data

   ! ---- timed region ---------------------------------------------------
   t0 = wall_seconds()
   do rep = 1, n_reps
      ! THE KERNEL. Inside host_data, a/b/c are device addresses; the call is
      ! an ordinary bind(C) call. sync=0 -> async launches, so the loop mirrors
      ! how a real code would batch; one sync after the loop pays the bill.
      !$acc host_data use_device(a, b, c)
      call daxpy3d_cuda_launch(c, a, b, alpha, nx, ny, nz, 0)
      !$acc end host_data
   end do
   ! One blocking launch to drain the queue, so t1 measures real work and not
   ! how fast we can enqueue.
   !$acc host_data use_device(a, b, c)
   call daxpy3d_cuda_launch(c, a, b, alpha, nx, ny, nz, 1)
   !$acc end host_data
   t1 = wall_seconds()

   ! ---- pull the answer back and check it ------------------------------
   ! `update self(c)` copies device->host. c was written ONLY by CUDA C, and
   ! OpenACC has no idea that happened — but it does not need to: both sides
   ! are pointing at the same device allocation, which is the whole trick.
   !$acc update self(c)
   !$acc exit data delete(a, b, c)

   call verify(c, max_err, n_bad)
   ! n_reps+1 launches were timed (the loop plus the draining one).
   call report('pure CUDA C via host_data use_device', &
               (t1 - t0)/real(n_reps + 1, wp), max_err, n_bad)

   deallocate (a, b, c)
   if (n_bad /= 0) error stop 1
end program daxpy_cuda
