! MRE 1/2 -- flang `-fdo-concurrent-to-openmp=device`: "Nested record types"
!
! A `do concurrent` whose live-in is a derived type having a SCALAR
! derived-type component fails to convert to OpenMP target.
!
! REPRODUCE
!   amdflang -O3 -fopenmp --offload-arch=gfx90a \
!            -fdo-concurrent-to-openmp=device mre_dc_device_nested_scalar_record.f90
!
! ACTUAL (AMD AFAR drop #7.0, flang 22.0.0git, MI250X/gfx90a)
!   warning: Mapping `do concurrent` to OpenMP is still experimental.
!   error: loc("...":25:39): .../DoConcurrentConversion.cpp:603:
!          not yet implemented: Nested record types are not supported yet.
!   LLVM ERROR: aborting
!
!   Note the reported location is the DECLARATION of the captured variable
!   (`:: o`), not the `o%buf%data(i)` reference.
!
! EXPECTED
!   builds, and prints:  36.000000000000000
!
! WORKS (so this is specific to the device conversion pass, not the language)
!   amdflang -O3 -fopenmp -fdo-concurrent-to-openmp=host  <this file>
!   nvfortran -O3 -stdpar=gpu -acc=gpu                    <this file>
!   gfortran -O2                                          <this file>   [verified]
!
! Companion: mre_dc_device_alloc_array_of_record.f90 reproduces a HARD CRASH
! (std::bad_function_call) on flang 23 with an allocatable array of derived type.

module nested_scalar_m
   use iso_fortran_env, only: real64
   implicit none

   type :: buffer_t
      real(real64), allocatable :: data(:)
   end type buffer_t

   type :: solver_t
      type(buffer_t) :: buf              ! <-- scalar derived-type component
   end type solver_t

contains

   subroutine fill(o, n)
      type(solver_t), intent(inout) :: o
      integer, intent(in) :: n
      integer :: i

      do concurrent(i=1:n)
         o%buf%data(i) = real(i, real64)
      end do
   end subroutine fill

end module nested_scalar_m

program mre_dc_device_nested_scalar_record
   use nested_scalar_m
   use iso_fortran_env, only: real64
   implicit none
   integer, parameter :: n = 8
   type(solver_t) :: o

   allocate (o%buf%data(n), source=0.0_real64)
   call fill(o, n)
   print *, sum(o%buf%data)               ! 36.0
end program mre_dc_device_nested_scalar_record
