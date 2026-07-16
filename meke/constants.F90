!! MRE stub of the ocean model's constants — ONLY the symbols ocean_meke.F90 uses.
!!
!! Values copied verbatim from <model>/src/core/constants.F90. They must stay
!! in sync: the point of this MRE is that the kernel body is byte-identical to
!! the production file, so a drifted constant here would silently make the
!! benchmark measure a different kernel from the one that ships.
module constants
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
   public

   integer, parameter :: wp = real64

   real(wp), parameter :: GRAVITY = 9.80665_wp
      !! src/core/constants.F90:36
   real(wp), parameter :: H_VANISHED = 1.5e-4_wp
      !! src/core/constants.F90:62
   real(wp), parameter :: H_DIV_EPS = 1.0e-20_wp
      !! src/core/constants.F90:67
   integer, parameter :: NZ_STACK_MAX = 128
      !! production builds with -DNZ_STACK_MAX=128 (flags.make)

end module constants
