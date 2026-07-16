!! MRE stub of the ocean model's ocean_metrics — ONLY the seven fields
!! ocean_redi.F90 reads. Names + shapes verbatim from
!! <model>/src/core/ocean/state/ocean_metrics.F90.
!!
!! Redi reads: wet_u, wet_v (Phase A face masks); dy_cu, dx_cv, idxCu, idyCv,
!! areaT (Phase B flux coefficients + divergence).
module ocean_metrics
   use constants, only: wp
   implicit none
   private

   public :: ocean_metrics_t

   type :: ocean_metrics_t
      real(wp), allocatable :: dy_cu(:, :)   !! u-face length (m),      (nx+1, ny)
      real(wp), allocatable :: dx_cv(:, :)   !! v-face length (m),      (nx, ny+1)
      real(wp), allocatable :: idxCu(:, :)   !! 1/dx at u-face (1/m),   (nx+1, ny)
      real(wp), allocatable :: idyCv(:, :)   !! 1/dy at v-face (1/m),   (nx, ny+1)
      real(wp), allocatable :: areaT(:, :)   !! cell area (m^2),        (nx, ny)
      real(wp), allocatable :: wet_u(:, :)   !! u-face open mask,       (nx+1, ny)
      real(wp), allocatable :: wet_v(:, :)   !! v-face open mask,       (nx, ny+1)
   end type ocean_metrics_t

end module ocean_metrics
