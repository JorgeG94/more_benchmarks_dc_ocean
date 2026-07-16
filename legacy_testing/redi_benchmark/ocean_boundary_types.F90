!! MRE stub of the ocean model's ocean_boundary_types — ONLY what
!! ocean_redi.F90 reads: the four per-edge tags and OBC_WALL.
!! Verbatim from <model>/src/core/ocean/boundary/ocean_boundary_types.F90
!! (OBC_WALL = 1 at line 39; ocean_bc_face_tag_t at 78; ocean_bc_state_t at 113).
!!
!! Redi only ever reads `bc%<edge>%bc_type == OBC_WALL`. The sponge/reservoir
!! payload of the production type is dropped — it is not referenced by Redi.
module ocean_boundary_types
   implicit none
   private

   public :: ocean_bc_state_t, ocean_bc_face_tag_t, OBC_WALL

   integer, parameter, public :: OBC_WALL = 1

   type :: ocean_bc_face_tag_t
      integer :: bc_type = OBC_WALL
   end type ocean_bc_face_tag_t

   type :: ocean_bc_state_t
      type(ocean_bc_face_tag_t) :: west, east, south, north
   end type ocean_bc_state_t

end module ocean_boundary_types
