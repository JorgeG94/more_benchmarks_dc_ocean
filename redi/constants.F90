!! MRE stub of the ocean model's constants — ONLY the symbols ocean_redi.F90 and
!! the extracted EOS use. Values copied VERBATIM from
!! <model>/src/core/constants.F90. They must stay in sync: the point of this
!! MRE is that ocean_redi.F90 is byte-identical to production, so a drifted
!! constant here would silently measure a different kernel.
!!
!! ⚠ NZ_STACK_MAX IS THE LOAD-BEARING CONSTANT OF THIS BENCHMARK.
!! Production compiles with `-DNZ_STACK_MAX=128` (see
!! <model>/build/CMakeFiles/core_model_objs.dir/flags.make), NOT the 256 default
!! in the header. Redi's per-thread stack frame scales linearly with it, so the
!! benchmark's default is 128 to match production. `make STACK=64` sweeps it.
module constants
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
   public

   integer, parameter :: wp = real64

   real(wp), parameter :: GRAVITY = 9.80665_wp
      !! src/core/constants.F90:36
   real(wp), parameter :: H_DIV_EPS = 1.0e-20_wp
      !! src/core/constants.F90:67 — floors a thickness before dividing;
      !! only prevents a 1/0, never changes a resolved value.
   real(wp), parameter :: H_VANISHED = 1.0e-10_wp
      !! src/core/constants.F90 — pulled in by the EOS extract's `use`.

#ifndef MODEL_NZ_STACK_MAX
#define MODEL_NZ_STACK_MAX 128
#endif
   integer, parameter :: NZ_STACK_MAX = MODEL_NZ_STACK_MAX
      !! Verbatim mechanism from src/core/constants.F90:89-92, but the
      !! FALLBACK here is 128 (production's -D value) rather than the 256 the
      !! production header defaults to. Redi's init hard-fails unless
      !! 2*nz_ml+2 <= NZ_STACK_MAX, so nz=30 needs >= 62.
      !!
      !! The production header claims "NZ_STACK_MAX * 8 bytes * ~6 arrays =
      !! ~5 KB per column, well within GPU thread stack limits". That comment
      !! predates Redi and is WRONG for it: redi_calc_coeffs_x's frame is ~28 KB
      !! per thread at NZ_STACK_MAX=128 (see README).

end module constants
