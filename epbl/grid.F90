!! MRE stub of the ocean model's grid -- ONLY the hgrid_t fields the EPBL column
!! kernel reads. Field names + defaults verbatim from
!! <model>/src/core/structured/grid.F90.
!!
!! The EPBL kernel reads nx_total / ny_total only (it sweeps the FULL array
!! including ghosts -- `do concurrent (j=1:ny, i=1:nx)` with nx = grid%nx_total).
!! nx_phys / ny_phys / nghost / dx / dy are carried so the driver can build a
!! realistic ghosted domain and so `set_f_centre` (which reads nghost + dy)
!! compiles unchanged.
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
      integer  :: nghost = 2
      real(wp) :: dx = 0.0_wp
      real(wp) :: dy = 0.0_wp
   end type hgrid_t

end module grid
