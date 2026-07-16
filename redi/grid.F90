!! MRE stub of the ocean model's grid — ONLY the hgrid_t fields ocean_redi.F90
!! reads. Field names + defaults verbatim from
!! <model>/src/core/structured/grid.F90:12-30 (the `init` TBP is dropped;
!! the driver sets the fields directly).
!!
!! Redi is ghost-aware: redi_apply_flux_impl places the no-normal-flow WALL
!! faces at the PHYSICAL domain edges (`nghost+1`, `nghost+nx_phys+1`), so
!! nx_total = nx_phys + 2*nghost must actually hold or the walls land in the
!! wrong place.
module grid
   use constants, only: wp
   implicit none
   private

   public :: hgrid_t

   type :: hgrid_t
      integer  :: nx_total = 0   !! total cells in x incl. ghosts (nx_phys + 2*nghost)
      integer  :: ny_total = 0   !! total cells in y incl. ghosts
      integer  :: nx_phys = 0    !! physical (interior) cells in x
      integer  :: ny_phys = 0    !! physical (interior) cells in y
      integer  :: nghost = 2     !! ghost cell width on each side
      real(wp) :: dx = 0.0_wp    !! cell size in x (m)
      real(wp) :: dy = 0.0_wp    !! cell size in y (m)
   end type hgrid_t

end module grid
