!! Minimal stub: just the working precision (double), matching <model>.
module constants
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
   public
   integer, parameter :: wp = real64
end module constants
