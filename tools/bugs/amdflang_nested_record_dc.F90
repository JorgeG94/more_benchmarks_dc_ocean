!! amdflang `-fdo-concurrent-to-openmp=device` vs. derived types that have
!! DERIVED-TYPE COMPONENTS -- reproducer + candidate workarounds.
!!
!! Build one variant per compile with -DV=<n>; every variant is semantically
!! identical and must print   sum = 1049600.0   expected = 1049600.0
!! Drive all of them with ./probe_amdflang_dc.sh (prints a PASS/FAIL table).
!!
!!   V=1  NESTED_SCALAR  container has `type(buffer_t) :: buf`
!!                       -> the continuity_layered / epbl / kappa_shear shape
!!                          (`this%h_face_left_x%data(i,j,k)`, a scalar
!!                           `type(scratch_3d_buffer_t)` member of a solver type)
!!   V=2  NESTED_ARRAY   container has `type(buffer_t), allocatable :: slots(:)`
!!                       -> the ale_remap shape (`ms%tracers(t)%hTr(i,j,k)`,
!!                          an allocatable ARRAY of derived type)
!!   V=3  ASSOCIATE      V=2's type, but the loop body reaches the arrays via
!!                       `associate` so the live-in is an array, not the record
!!                       -> CANDIDATE WORKAROUND (zero data-layout change)
!!   V=4  DUMMY_ARG      V=2's type, but the loop lives in a subroutine taking
!!                       explicit-shape array dummies
!!                       -> CANDIDATE WORKAROUND (the idiom ale_remap already
!!                          uses for `ocean_remap_tracer_pair`)
!!   V=5  FLAT           container holds the allocatable directly (no nesting)
!!                       -> CONTROL: isolates nesting as the trigger
!!
!! OBSERVED (2026-07-29, MI250X):
!!   AMD AFAR drop #7.0, flang 22.0.0git  (Pawsey Setonix)
!!     V=1 -> error: DoConcurrentConversion.cpp:603: not yet implemented:
!!            Nested record types are not supported yet.   LLVM ERROR: aborting
!!            (location points at the DECLARATION of the captured variable,
!!             not at any `%a%b` reference)
!!   AMD flang 23.0.0git, ROCm 7.13.0    (OLCF Frontier)
!!     V=2 -> terminate called after throwing an instance of
!!            'std::bad_function_call'  -- a hard compiler crash in
!!            DoConcurrentConversion::genMapInfoOpForLiveIn, i.e. the pass
!!            cannot build a map-info op for the live-in's type. Not a clean
!!            NYI diagnostic; flang-23 aborts with a stack dump.
!!   V=3/V=4/V=5 status: UNKNOWN -- this file exists to determine it.
!!
!! Unaffected, i.e. this is a device-pass limitation and not a language issue:
!!   amdflang -fopenmp -fdo-concurrent-to-openmp=host      (all variants)
!!   nvfortran -stdpar=gpu -acc=gpu                        (all variants)
!!   gfortran -O2 (serial)                                 (all variants, verified)
!!
!! WHY NOT JUST FLATTEN THE TYPES: these kernels are verbatim extractions of a
!! production GPU-native ocean model; the nesting IS the shipped data layout
!! (`ms%tracers(t)%hTr` is a real pointer chase the benchmark exists to measure).
!! Flattening would benchmark a kernel that is not the one that ships. V=3/V=4
!! are therefore the only acceptable fixes -- both keep the layout and change
!! only how the loop body names the arrays.

#ifndef V
#define V 1
#endif

module nested_record_m
   implicit none
   public

   integer, parameter :: wp = kind(1.0d0)
   integer, parameter :: nslots = 3, islot = 2   ! mirrors ms%idx_temperature

   type :: buffer_t
      real(wp), allocatable :: data(:)
   end type buffer_t

   type :: container_t
#if V == 1
      type(buffer_t) :: buf                        ! scalar derived component
#elif V == 5
      real(wp), allocatable :: data(:)             ! control: no nesting
#else
      type(buffer_t), allocatable :: slots(:)      ! allocatable array of DT
#endif
      integer :: n = 0
   end type container_t

contains

   subroutine run(c, n)
      type(container_t), intent(inout) :: c
      integer, intent(in) :: n
      integer :: i

#if V == 1
      do concurrent(i=1:n)
         c%buf%data(i) = 2.0_wp * real(i, wp)
      end do

#elif V == 2
      do concurrent(i=1:n)
         c%slots(islot)%data(i) = 2.0_wp * real(i, wp)
      end do

#elif V == 3
      !! WORKAROUND A: hoist the component to an associate-name, so the
      !! do-concurrent live-in is a plain array rather than the record.
      associate (d => c%slots(islot)%data)
         do concurrent(i=1:n)
            d(i) = 2.0_wp * real(i, wp)
         end do
      end associate

#elif V == 4
      !! WORKAROUND B: the loop sees only explicit-shape array dummies.
      call fill(c%slots(islot)%data, n)

#elif V == 5
      do concurrent(i=1:n)
         c%data(i) = 2.0_wp * real(i, wp)
      end do
#endif
   end subroutine run

#if V == 4
   subroutine fill(d, n)
      integer, intent(in) :: n
      real(wp), intent(inout) :: d(n)
      integer :: i
      do concurrent(i=1:n)
         d(i) = 2.0_wp * real(i, wp)
      end do
   end subroutine fill
#endif

   !! allocate + zero, whatever the variant's layout is
   subroutine setup(c, n)
      type(container_t), intent(inout) :: c
      integer, intent(in) :: n
      c%n = n
#if V == 1
      allocate (c%buf%data(n), source=0.0_wp)
#elif V == 5
      allocate (c%data(n), source=0.0_wp)
#else
      block
         integer :: s
         allocate (c%slots(nslots))
         do s = 1, nslots
            allocate (c%slots(s)%data(n), source=0.0_wp)
         end do
      end block
#endif
   end subroutine setup

   pure function total(c) result(t)
      type(container_t), intent(in) :: c
      real(wp) :: t
#if V == 1
      t = sum(c%buf%data)
#elif V == 5
      t = sum(c%data)
#else
      t = sum(c%slots(islot)%data)
#endif
   end function total

end module nested_record_m

program amdflang_nested_record_dc
   use nested_record_m
   implicit none
   integer, parameter :: n = 1024
   type(container_t) :: c

   call setup(c, n)
   call run(c, n)
   print '(a,i0,a,f12.1,a,f12.1)', 'V=', V, '   sum =', total(c), &
      '   expected =', real(n, wp) * real(n + 1, wp)
end program amdflang_nested_record_dc
