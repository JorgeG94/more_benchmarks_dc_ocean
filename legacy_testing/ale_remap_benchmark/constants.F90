!! Minimal stub of <model>'s `constants`, carrying ONLY the parameters the
!! ALE remap path reads. Values copied verbatim from
!! <model-root>/src/core/constants.F90
!!
!! NZ_STACK_MAX defaults to 128 here to match PRODUCTION's actual build:
!!   <model>/build/CMakeFiles/core_model_objs.dir/flags.make
!!   -DNZ_STACK_MAX=128
!! (the constants.F90 in-source #define default is 256, but the real build
!! overrides it to 128 via CMake -- benchmark the value that ships.)
module constants
   implicit none
   public

   integer, parameter :: wp = kind(1.0d0)   ! -DMODEL_DOUBLE_PRECISION

#ifndef MODEL_NZ_STACK_MAX
#define MODEL_NZ_STACK_MAX 128
#endif
   integer, parameter :: NZ_STACK_MAX = MODEL_NZ_STACK_MAX

   ! ---- Thin-layer constants (verbatim) ----
   real(wp), parameter :: H_VANISHED = 1.5e-4_wp

   ! ---- Vertical remapping method constants (verbatim) ----
   integer, parameter :: REMAP_PCM = 1
   integer, parameter :: REMAP_PLM = 2
   integer, parameter :: REMAP_PPM = 3
   integer, parameter :: REMAP_PPM_H4 = 4
   integer, parameter :: REMAP_PQM = 5

   ! ---- vcoord type tags (verbatim from ocean_vcoord.F90) ----
   integer, parameter :: VCOORD_EULERIAN_Z = 1
   integer, parameter :: VCOORD_SIGMA = 2
   integer, parameter :: VCOORD_ZSTAR = 3
   integer, parameter :: VCOORD_LAGRANGIAN = 4
   integer, parameter :: VCOORD_RHO = 5
   integer, parameter :: VCOORD_HYCOM = 6

end module constants
