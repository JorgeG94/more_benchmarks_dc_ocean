!! MRE stub of the ocean model's constants -- ONLY the symbols ocean_epbl.F90 uses.
!!
!! Values copied verbatim from <model>/src/core/constants.F90. They must stay
!! in sync: the whole point of this MRE is that the EPBL kernel is a verbatim
!! extract of the production file, so a drifted constant here would silently make
!! the benchmark measure a different kernel from the one that ships.
module constants
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
   public

   integer, parameter :: wp = real64

   real(wp), parameter :: GRAVITY = 9.80665_wp
      !! src/core/constants.F90
   real(wp), parameter :: H_DIV_EPS = 1.0e-20_wp
      !! src/core/constants.F90 -- EPBL aliases this as H_NEGLECT.
   real(wp), parameter :: SEAWATER_CP = 3992.0_wp
      !! src/parameterizations/vertical/structured/ocean_surface_flux.F90:51
      !! -- ocean_surface_flux_t%cp default.

end module constants
