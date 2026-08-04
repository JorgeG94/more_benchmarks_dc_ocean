! MRE 2/2 -- flang `-fdo-concurrent-to-openmp=device`: compiler CRASH
!            (std::bad_function_call in genMapInfoOpForLiveIn)
!
! A `do concurrent` whose live-in is a derived type having an ALLOCATABLE ARRAY
! OF DERIVED TYPE component crashes the DoConcurrent->OpenMP conversion pass.
! This is the harder sibling of mre_dc_device_nested_scalar_record.f90: where
! flang 22 emits a clean "not yet implemented" for a scalar derived component,
! flang 23 aborts here with an unhandled C++ exception.
!
! REPRODUCE
!   amdflang -O3 -fopenmp --offload-arch=gfx90a \
!            -fdo-concurrent-to-openmp=device mre_dc_device_alloc_array_of_record.f90
!
! ACTUAL (AMD flang 23.0.0git, ROCm 7.13.0, MI250X)
!   warning: Mapping `do concurrent` to OpenMP is still experimental.
!   terminate called after throwing an instance of 'std::bad_function_call'
!     what():  bad_function_call
!   PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/
!   Stack dump:
!    #12 DoConcurrentConversion::genMapInfoOpForLiveIn(fir::FirOpBuilder&, mlir::Value)
!    #13 DoConcurrentConversion::matchAndRewrite(fir::DoConcurrentOp, ...)
!    #22 DoConcurrentConversionPass::runOnOperation()
!   flang-23: error: unable to execute command: Aborted (core dumped)
!
!   i.e. the pass cannot build a map-info op for this live-in's type and
!   dereferences an empty std::function rather than diagnosing it.
!
! EXPECTED
!   builds, and prints:  36.000000000000000
!
! WORKS (so this is specific to the device conversion pass, not the language)
!   amdflang -O3 -fopenmp -fdo-concurrent-to-openmp=host  <this file>
!   nvfortran -O3 -stdpar=gpu -acc=gpu                    <this file>
!   gfortran -O2                                          <this file>   [verified]
!
! ------------------------------------------------------------------------------
! PLEASE DO NOT "SIMPLIFY" state_t BY REMOVING slots(:)
!
! The derived-type component is the trigger, and it is load-bearing even when
! the loop body never dereferences it. Replacing the loop body above with a
! reference to the FLAT component only --
!
!     do concurrent(i=1:n)
!        s%h(i) = real(i, real64)          ! never touches s%slots
!     end do
!
! -- still fails, because the pass maps the whole captured container. That is
! how this was first hit in real code: the affected kernels contain zero
! `%a%b` references anywhere in the file; they merely pass a state type that
! happens to have one derived-type member among a dozen plain arrays.
! ------------------------------------------------------------------------------

module alloc_array_of_record_m
   use iso_fortran_env, only: real64
   implicit none

   type :: slot_t
      real(real64), allocatable :: data(:)
   end type slot_t

   type :: state_t
      type(slot_t), allocatable :: slots(:)   ! <-- allocatable ARRAY of DT
      real(real64), allocatable :: h(:)       !     plain component, for the note above
   end type state_t

contains

   subroutine fill(s, n)
      type(state_t), intent(inout) :: s
      integer, intent(in) :: n
      integer :: i

      do concurrent(i=1:n)
         s%slots(1)%data(i) = real(i, real64)
      end do
   end subroutine fill

end module alloc_array_of_record_m

program mre_dc_device_alloc_array_of_record
   use alloc_array_of_record_m
   use iso_fortran_env, only: real64
   implicit none
   integer, parameter :: n = 8
   type(state_t) :: s

   allocate (s%slots(1))
   allocate (s%slots(1)%data(n), source=0.0_real64)
   allocate (s%h(n), source=0.0_real64)

   call fill(s, n)
   print *, sum(s%slots(1)%data)              ! 36.0
end program mre_dc_device_alloc_array_of_record
