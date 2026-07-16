!! MRE stub of the ocean model's constants — ONLY the symbols the kappa-shear
!! column kernel uses. Values copied verbatim from
!! <model>/src/core/constants.F90.
!!
!! ⚠ NZ_STACK_MAX IS THE WHOLE BALLGAME FOR THIS KERNEL.
!! Production compiles with `-DNZ_STACK_MAX=128`
!! (<model>/build/CMakeFiles/core_model_objs.dir/flags.make). Every per-thread
!! column array in ks_solve_column is dimensioned NZL/NZLI = 128/129 REGARDLESS
!! of the actual nz (=30 in every production config). The frame is therefore
!! sized for 128 layers and only the first 31 entries of each array are ever
!! touched. Keep this at 128 unless you are deliberately measuring the
!! sensitivity — `make NZSTACK=32` does that.
module constants
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
   public

   integer, parameter :: wp = real64
      !! Production builds with -DMODEL_DOUBLE_PRECISION (flags.make), so
      !! wp = dp = real64.

   real(wp), parameter :: GRAVITY = 9.80665_wp
      !! src/core/constants.F90:36

   real(wp), parameter :: H_VANISHED = 1.5e-4_wp
      !! src/core/constants.F90:62 — the gather floor.

   real(wp), parameter :: H_DIV_EPS = 1.0e-20_wp
      !! src/core/constants.F90:67 — division-safety epsilon.

#ifndef MODEL_NZ_STACK_MAX
#define MODEL_NZ_STACK_MAX 128
#endif
   integer, parameter :: NZ_STACK_MAX = MODEL_NZ_STACK_MAX
      !! src/core/constants.F90:92. Production: 128 (see flags.make).

end module constants
