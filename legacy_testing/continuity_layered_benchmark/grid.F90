!! MRE stub of the ocean model's grid — ONLY the hgrid_t fields the LAYERED
!! continuity kernel reads. Field names + defaults verbatim from
!! <model>/src/core/structured/grid.F90:12-25.
!!
!! NOTE vs the barotropic benchmark's stub: this kernel also reads nx_phys /
!! ny_phys. It is ghost-aware — the closed-wall BC is applied at the PHYSICAL
!! domain edges (`nghost+1`, `nghost+nx_phys+1`), not the array edges the
!! barotropic kernel uses. So nx_total = nx_phys + 2*nghost must actually hold
!! here, or the walls land in the wrong place.
module grid
   use constants, only: wp
   implicit none
   private

   public :: hgrid_t

   type :: hgrid_t
      integer  :: nx_total = 0   !! cells in x incl. ghosts (nx_phys + 2*nghost)
      integer  :: ny_total = 0   !! cells in y incl. ghosts
      integer  :: nx_phys = 0    !! interior cells in x
      integer  :: ny_phys = 0    !! interior cells in y
      integer  :: nghost = 2     !! the PPM stencil reads i-2..i+2, so >= 2
      real(wp) :: dx = 0.0_wp
      real(wp) :: dy = 0.0_wp
   end type hgrid_t

end module grid
