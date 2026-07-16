!! Stubbed derived types for the ALE-remap MRE.
!!
!! These mirror the SHAPE of production's `hgrid_t`, `ocean_vcoord_t` and
!! `multilayer_cgrid_state_t` for exactly the components the remap path
!! touches -- nothing else. Component NAMES and the array-of-derived-type
!! `tracers(t)%hTr` indirection are preserved verbatim, because reproducing
!! the derived-type-component access pattern is one of the things this
!! benchmark measures (the "signature fix" hypothesis).
!!
!! Production sources:
!!   src/core/grid.F90                    (hgrid_t)
!!   src/core/ocean/vcoord/ocean_vcoord.F90   (ocean_vcoord_t)
!!   src/core/structured/multilayer_cgrid_state.F90 (multilayer_cgrid_state_t)
module remap_state
   use constants, only: wp
   implicit none
   public

   type :: hgrid_t
      integer :: nx_total = 0
      integer :: ny_total = 0
   end type hgrid_t

   !! One registered tracer slot. Production's `tracers` is an array of these,
   !! so `ms%tracers(t)%hTr(i,j,k)` is a pointer chase through an
   !! array-of-derived-types -- reproduced exactly.
   type :: tracer_slot_t
      real(wp), allocatable :: hTr(:, :, :)
   end type tracer_slot_t

   type :: ocean_vcoord_t
      integer :: coord_type = 3           ! VCOORD_ZSTAR (every production nml)
      logical :: is_init = .true.
      integer :: nz_ml = 0
      integer :: nx_total = 0
      integer :: ny_total = 0
      integer :: remap_method = 3         ! REMAP_PPM (default)
      real(wp) :: regrid_time_scale = 0.0_wp    ! default => time-filter OFF
      logical :: remap_vel_conserve_ke = .false. ! default => KE rescale OFF
      real(wp), allocatable :: dsig(:)
      real(wp), allocatable :: target_h(:, :, :)
      real(wp), allocatable :: remap_h_old(:, :, :)
      real(wp), allocatable :: remap_total_h(:, :)
      real(wp), allocatable :: remap_h_ref(:, :)
   end type ocean_vcoord_t

   type :: multilayer_cgrid_state_t
      integer :: nz_ml = 0
      integer :: idx_temperature = 1
      integer :: idx_salinity = 2
      type(tracer_slot_t), allocatable :: tracers(:)
      real(wp), allocatable :: h_layer(:, :, :)
      real(wp), allocatable :: u_face_x_layer(:, :, :)
      real(wp), allocatable :: v_face_y_layer(:, :, :)
      real(wp), allocatable :: mass_budget_remap(:, :, :)
      real(wp), allocatable :: heat_budget_remap(:, :, :)
      real(wp), allocatable :: salt_budget_remap(:, :, :)
   end type multilayer_cgrid_state_t

end module remap_state
