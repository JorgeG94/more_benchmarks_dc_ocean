!! MRE stub of the ocean model's constants — ONLY the symbols kernel_flux.F90 uses.
!!
!! Values copied verbatim from src/core/constants.F90 (<model>-sea-ice @
!! feat/sea-ice). They must stay in sync: the whole point of this MRE is that
!! `kernel_flux.F90` is byte-identical to the production file, so a drifted
!! constant here would silently make the benchmark measure a different kernel
!! from the one that ships.
module constants
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
   public

   integer, parameter :: wp = real64

   real(wp), parameter :: GRAVITY = 9.80665_wp
      !! src/core/constants.F90:36
   real(wp), parameter :: DRY_TOLERANCE = 1.0e-6_wp
      !! src/core/constants.F90:51
   real(wp), parameter :: THIN_LAYER_THRESHOLD = 1.0e-3_wp
      !! src/core/constants.F90:57

end module constants
