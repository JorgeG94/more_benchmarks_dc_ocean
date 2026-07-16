!! MRE stub of the ocean model's grid — ONLY the hgrid_t fields kernel_flux.F90
!! reads (nx_total, ny_total, nghost, dx, dy). The production type carries more;
!! the flux kernel does not touch them.
module grid
   use constants, only: wp
   implicit none
   private

   public :: hgrid_t

   type :: hgrid_t
      integer  :: nx_total = 0   !! cells in x incl. ghosts (nx_phys + 2*nghost)
      integer  :: ny_total = 0   !! cells in y incl. ghosts
      integer  :: nghost = 2     !! the flux stencil reads i-2..i+2, so >= 2
      real(wp) :: dx = 0.0_wp
      real(wp) :: dy = 0.0_wp
   end type hgrid_t

end module grid
