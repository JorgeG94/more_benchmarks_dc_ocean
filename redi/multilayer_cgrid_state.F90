!! MRE stub of the ocean model's multilayer_cgrid_state — ONLY the fields
!! ocean_redi.F90 reads. Names + shapes verbatim from
!! <model>/src/core/structured/multilayer_cgrid_state.F90.
!!
!! Redi reads: nz_ml, h_layer, tracers(:)%hTr, idx_temperature, idx_salinity.
!! The tracer registry is an ARRAY OF DERIVED TYPE with an allocatable
!! component — production's redi_calc_coeffs / redi_apply_flux dereference
!! `ms%tracers(it)%hTr` ON THE HOST and pass the flat allocatable down to the
!! `_impl` kernels (the outer-shim + flat-impl pattern), precisely to avoid the
!! NVHPC stdpar deep-deref issue. This stub preserves that shape.
module multilayer_cgrid_state
   use constants, only: wp
   implicit none
   private

   public :: multilayer_cgrid_state_t, tracer_t

   type :: tracer_t
      real(wp), allocatable :: hTr(:, :, :)  !! thickness-weighted tracer, (nx, ny, nz_ml)
   end type tracer_t

   type :: multilayer_cgrid_state_t
      integer :: nz_ml = 0            !! multilayer levels (k=1 bed, k=nz_ml surface)
      integer :: idx_temperature = 0  !! index into tracers(:); 0 = not registered
      integer :: idx_salinity = 0     !! index into tracers(:); 0 = not registered
      real(wp), allocatable :: h_layer(:, :, :)  !! layer thickness (m), (nx, ny, nz_ml)
      type(tracer_t), allocatable :: tracers(:)  !! registered prognostic tracers
   end type multilayer_cgrid_state_t

end module multilayer_cgrid_state
