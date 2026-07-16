#include "directives.h"
!! MRE stubs + the ocean model's PRODUCTION LAYERED continuity-PPM kernel.
!!
!! VERBATIM from <model> @ HEAD,
!! src/core/ocean/kernels/structured/continuity_ppm/continuity.F90:
!!   continuity_compute_fluxes            612-775   <- the LAYERED kernel
!!   ppm_mirror_h / ppm_limited_slope / ppm_cell_limiter / ppm_limit_pos
!!
!! WHY THIS AND NOT THE BAROTROPIC ONE (../continuity_ppm_benchmark):
!! the production profiler reports `ocean_continuity` at 11% of runtime with
!! EPBL+kappa-shear on, and ~20% (co-dominant with ale_remap) when they are
!! off — and `ocean_continuity` wraps THIS routine, not the barotropic one.
!! 2.16 ms/call in production vs the barotropic kernel's 0.16 ms. The
!! barotropic benchmark next door measured the wrong kernel for that regime.
!!
!! SHAPE: this is the 3-D one. Loops are `do concurrent(k=1:nz, j=1:ny, i=...)`
!! — 11 of them — over (nx, ny, nz) state, while the metrics stay 2-D
!! (`wet_T(i,j)`, `dy_cu(i,j)`). Whether the 3-D iteration space collapses,
!! and how the 2-D-metric/3-D-state mix behaves, is exactly what is unknown:
!! NOTHING measured on the 2-D barotropic kernel predicts it.
!!
!! Only the fields THIS subroutine touches are stubbed (verified by grep):
!!   grid    : nx_total, ny_total
!!   metrics : dy_cu, dx_cv, iareaT, wet_T          (all 2-D)
!!   ms      : nz_ml, h_layer, u_face_x_layer, v_face_y_layer,
!!             mass_flux_x_layer, mass_flux_y_layer, flux_h_layer
!!   this    : h_face_{left,right}_{x,y}%data, h_min, use_ppm_limit_pos
!! Re-extract if the production file moves -- a drifted copy would benchmark a
!! kernel that is not the one that ships.

module ocean_metrics
   use constants, only: wp
   implicit none
   public
   type :: ocean_metrics_t
      real(wp), allocatable :: dy_cu(:, :)    !! east-face width  [m]
      real(wp), allocatable :: dx_cv(:, :)    !! north-face width [m]
      real(wp), allocatable :: iareaT(:, :)   !! 1/cell area   [1/m^2]
      real(wp), allocatable :: wet_T(:, :)    !! 1 wet / 0 land
   end type ocean_metrics_t
end module ocean_metrics

module scratch_3d
   use constants, only: wp
   implicit none
   public
   type :: scratch_3d_buffer_t
      logical :: is_init = .false.
      integer :: n1 = 0, n2 = 0, n3 = 0
      real(wp), allocatable :: data(:, :, :)
   end type scratch_3d_buffer_t
end module scratch_3d

!! Stub of multilayer_cgrid_state -- ONLY the fields the layered
!! continuity kernel reads. Shapes verified against the production
!! allocate()s (multilayer_cgrid_state.F90:216-229): the face arrays
!! carry one extra element in their own direction.
module multilayer_cgrid_state
   use constants, only: wp
   implicit none
   public
   type :: multilayer_cgrid_state_t
      integer :: nz_ml = 0
      real(wp), allocatable :: h_layer(:, :, :)            !! (nx,   ny,   nz)
      real(wp), allocatable :: u_face_x_layer(:, :, :)     !! (nx+1, ny,   nz)
      real(wp), allocatable :: v_face_y_layer(:, :, :)     !! (nx,   ny+1, nz)
      real(wp), allocatable :: mass_flux_x_layer(:, :, :)  !! (nx+1, ny,   nz)
      real(wp), allocatable :: mass_flux_y_layer(:, :, :)  !! (nx,   ny+1, nz)
      real(wp), allocatable :: flux_h_layer(:, :, :)       !! (nx,   ny,   nz)
   end type multilayer_cgrid_state_t
end module multilayer_cgrid_state

module continuity_layered
   use constants, only: wp
   use grid, only: hgrid_t
   use ocean_metrics, only: ocean_metrics_t
   use scratch_3d, only: scratch_3d_buffer_t
   use multilayer_cgrid_state, only: multilayer_cgrid_state_t
   implicit none
   private
   public :: continuity_t, continuity_compute_fluxes
   ! MRE-only: production keeps the helpers private. Exposed so the FLAT/ACC
   ! controls can reuse them verbatim. The bodies below are the verbatim extract.
   public :: ppm_mirror_h, ppm_limited_slope, ppm_cell_limiter, ppm_limit_pos

   type :: continuity_t
      type(scratch_3d_buffer_t) :: h_face_left_x, h_face_right_x
      type(scratch_3d_buffer_t) :: h_face_left_y, h_face_right_y
      real(wp) :: h_min = 1.0e-6_wp
      logical  :: use_ppm_limit_pos = .false.
   end type continuity_t

contains

   pure subroutine continuity_compute_fluxes(grid, metrics, this, ms)
      !! **Test-only** (no production caller): the unsplit reference path,
      !! kept as the oracle the split production path is checked against.
      !! Multilayer counterpart to
      !! `continuity_compute_fluxes_barotropic`: identical PPM
      !! reconstruction + upwind face pick + flux divergence, lifted
      !! per-layer.  Each k-slice is independent (the PPM stencil
      !! reads only the same k), so the do-concurrent kernels
      !! parallelize over (k, j, i) simultaneously for GPU
      !! occupancy.
      !!
      !! Workspaces (`this%h_face_*_x/y`) must have been initialised
      !! with `nz_ml` matching `ms%nz_ml` — handled by passing the
      !! optional `nz_ml` to `continuity_init`.
      type(hgrid_t), intent(in) :: grid
      type(ocean_metrics_t), intent(in) :: metrics
      type(continuity_t), intent(inout) :: this
      type(multilayer_cgrid_state_t), intent(inout) :: ms

      integer :: i, j, k, nx, ny, nz
      real(wp) :: dh_m1, dh_0, dh_p1, h_left, h_right, u, v, h_face
      real(wp) :: hm2, hm1, h0, hp1, hp2
      logical :: do_pos
      real(wp) :: h_min_pos

      nx = grid%nx_total
      ny = grid%ny_total
      nz = ms%nz_ml
      do_pos = this%use_ppm_limit_pos
      h_min_pos = this%h_min

      ! ============================================================
      ! X-DIRECTION reconstruction (5-point stencil per cell)
      ! ============================================================
      do concurrent(k=1:nz, j=1:ny, i=3:nx - 2) &
         local(dh_m1, dh_0, dh_p1, h_left, h_right, hm2, hm1, h0, hp1, hp2)
         ! Mirror-h at land neighbours (C2); bit-identical for all-wet.
         h0 = ms%h_layer(i, j, k)
         hm1 = ppm_mirror_h(ms%h_layer(i - 1, j, k), h0, metrics%wet_T(i - 1, j))
         hp1 = ppm_mirror_h(ms%h_layer(i + 1, j, k), h0, metrics%wet_T(i + 1, j))
         hm2 = ppm_mirror_h(ms%h_layer(i - 2, j, k), hm1, metrics%wet_T(i - 2, j))
         hp2 = ppm_mirror_h(ms%h_layer(i + 2, j, k), hp1, metrics%wet_T(i + 2, j))
         call ppm_limited_slope(hm2, hm1, h0, dh_m1)
         call ppm_limited_slope(hm1, h0, hp1, dh_0)
         call ppm_limited_slope(h0, hp1, hp2, dh_p1)
         dh_0 = dh_0*metrics%wet_T(i - 1, j)*metrics%wet_T(i, j)*metrics%wet_T(i + 1, j)
         h_left = 0.5_wp*(hm1 + h0) - (dh_0 - dh_m1)/6.0_wp
         h_right = 0.5_wp*(h0 + hp1) - (dh_p1 - dh_0)/6.0_wp
         call ppm_cell_limiter(h0, h_left, h_right)
         if (do_pos) call ppm_limit_pos(h0, h_left, h_right, h_min_pos)
         this%h_face_right_x%data(i, j, k) = h_left
         this%h_face_left_x%data(i + 1, j, k) = h_right
      end do
      ! Boundary cells: 1st-order fallback
      do concurrent(k=1:nz, j=1:ny)
         this%h_face_left_x%data(1, j, k) = ms%h_layer(1, j, k)
         this%h_face_right_x%data(1, j, k) = ms%h_layer(1, j, k)
         this%h_face_left_x%data(2, j, k) = ms%h_layer(1, j, k)
         this%h_face_right_x%data(2, j, k) = ms%h_layer(2, j, k)
         ! Face 3: cell 2's right edge falls back to 1st order (its
         ! 5-point stencil needs cell 0); cell 3's left edge is set
         ! by the interior loop above.
         this%h_face_left_x%data(3, j, k) = ms%h_layer(2, j, k)
         this%h_face_right_x%data(3, j, k) = ms%h_layer(2, j, k)
         ! Face nx-1: mirror of face 3.  Cell nx-1's left edge falls
         ! back to 1st order; cell nx-2's right edge came from the
         ! interior loop.
         this%h_face_right_x%data(nx - 1, j, k) = ms%h_layer(nx - 1, j, k)
         this%h_face_left_x%data(nx, j, k) = ms%h_layer(nx - 1, j, k)
         this%h_face_right_x%data(nx, j, k) = ms%h_layer(nx, j, k)
         this%h_face_left_x%data(nx + 1, j, k) = ms%h_layer(nx, j, k)
         this%h_face_right_x%data(nx + 1, j, k) = ms%h_layer(nx, j, k)
      end do

      ! X-direction face transport (upwind pick) — width-weighted dy_cu
      do concurrent(k=1:nz, j=1:ny, i=2:nx) local(u, h_face)
         u = ms%u_face_x_layer(i, j, k)
         if (u >= 0.0_wp) then
            h_face = this%h_face_left_x%data(i, j, k)
         else
            h_face = this%h_face_right_x%data(i, j, k)
         end if
         ms%mass_flux_x_layer(i, j, k) = u*h_face*metrics%dy_cu(i, j)
      end do
      do concurrent(k=1:nz, j=1:ny)
         ms%mass_flux_x_layer(1, j, k) = 0.0_wp
         ms%mass_flux_x_layer(nx + 1, j, k) = 0.0_wp
      end do
      ! Physical wall zeroing — see `continuity_zonal_flux` for the
      ! full rationale.  Must mirror the split form's wall closure or
      ! `test_split_zonal_only_matches_unsplit` breaks.
      do concurrent(k=1:nz, j=1:ny)
         ms%mass_flux_x_layer(grid%nghost + 1, j, k) = 0.0_wp
         ms%mass_flux_x_layer(grid%nghost + grid%nx_phys + 1, j, k) = 0.0_wp
      end do

      ! ============================================================
      ! Y-DIRECTION reconstruction
      ! ============================================================
      do concurrent(k=1:nz, j=3:ny - 2, i=1:nx) &
         local(dh_m1, dh_0, dh_p1, h_left, h_right, hm2, hm1, h0, hp1, hp2)
         h0 = ms%h_layer(i, j, k)
         hm1 = ppm_mirror_h(ms%h_layer(i, j - 1, k), h0, metrics%wet_T(i, j - 1))
         hp1 = ppm_mirror_h(ms%h_layer(i, j + 1, k), h0, metrics%wet_T(i, j + 1))
         hm2 = ppm_mirror_h(ms%h_layer(i, j - 2, k), hm1, metrics%wet_T(i, j - 2))
         hp2 = ppm_mirror_h(ms%h_layer(i, j + 2, k), hp1, metrics%wet_T(i, j + 2))
         call ppm_limited_slope(hm2, hm1, h0, dh_m1)
         call ppm_limited_slope(hm1, h0, hp1, dh_0)
         call ppm_limited_slope(h0, hp1, hp2, dh_p1)
         dh_0 = dh_0*metrics%wet_T(i, j - 1)*metrics%wet_T(i, j)*metrics%wet_T(i, j + 1)
         h_left = 0.5_wp*(hm1 + h0) - (dh_0 - dh_m1)/6.0_wp
         h_right = 0.5_wp*(h0 + hp1) - (dh_p1 - dh_0)/6.0_wp
         call ppm_cell_limiter(h0, h_left, h_right)
         if (do_pos) call ppm_limit_pos(h0, h_left, h_right, h_min_pos)
         this%h_face_right_y%data(i, j, k) = h_left
         this%h_face_left_y%data(i, j + 1, k) = h_right
      end do
      do concurrent(k=1:nz, i=1:nx)
         this%h_face_left_y%data(i, 1, k) = ms%h_layer(i, 1, k)
         this%h_face_right_y%data(i, 1, k) = ms%h_layer(i, 1, k)
         this%h_face_left_y%data(i, 2, k) = ms%h_layer(i, 1, k)
         this%h_face_right_y%data(i, 2, k) = ms%h_layer(i, 2, k)
         this%h_face_left_y%data(i, 3, k) = ms%h_layer(i, 2, k)
         this%h_face_right_y%data(i, 3, k) = ms%h_layer(i, 2, k)
         this%h_face_right_y%data(i, ny - 1, k) = ms%h_layer(i, ny - 1, k)
         this%h_face_left_y%data(i, ny, k) = ms%h_layer(i, ny - 1, k)
         this%h_face_right_y%data(i, ny, k) = ms%h_layer(i, ny, k)
         this%h_face_left_y%data(i, ny + 1, k) = ms%h_layer(i, ny, k)
         this%h_face_right_y%data(i, ny + 1, k) = ms%h_layer(i, ny, k)
      end do

      do concurrent(k=1:nz, j=2:ny, i=1:nx) local(v, h_face)
         v = ms%v_face_y_layer(i, j, k)
         if (v >= 0.0_wp) then
            h_face = this%h_face_left_y%data(i, j, k)
         else
            h_face = this%h_face_right_y%data(i, j, k)
         end if
         ms%mass_flux_y_layer(i, j, k) = v*h_face*metrics%dx_cv(i, j)
      end do
      do concurrent(k=1:nz, i=1:nx)
         ms%mass_flux_y_layer(i, 1, k) = 0.0_wp
         ms%mass_flux_y_layer(i, ny + 1, k) = 0.0_wp
      end do
      ! Physical wall zeroing — mirrors split form's
      ! `continuity_meridional_flux`.
      do concurrent(k=1:nz, i=1:nx)
         ms%mass_flux_y_layer(i, grid%nghost + 1, k) = 0.0_wp
         ms%mass_flux_y_layer(i, grid%nghost + grid%ny_phys + 1, k) = 0.0_wp
      end do

      ! ============================================================
      ! Per-layer flux divergence (transport divergence · iareaT)
      ! ============================================================
      do concurrent(k=1:nz, j=1:ny, i=1:nx)
         ms%flux_h_layer(i, j, k) = &
            ((ms%mass_flux_x_layer(i + 1, j, k) - &
              ms%mass_flux_x_layer(i, j, k)) + &
             (ms%mass_flux_y_layer(i, j + 1, k) - &
              ms%mass_flux_y_layer(i, j, k)))*metrics%iareaT(i, j)
      end do
   end subroutine continuity_compute_fluxes

   pure elemental function ppm_mirror_h(h_nbr, h_loc, w_nbr) result(h_out)
      DC_ROUTINE_SEQ
      !! Mirror-h at a land neighbour (spec §14 C2 / MOM6
      !! `MOM_continuity_PPM.F90:2764`): substitute the LOCAL cell's
      !! thickness (or tracer) for a LAND neighbour's held floor value
      !! so the PPM parabola sees a flat, reflected coast and the
      !! wet-side face value is not biased by the dry column.  Branchless:
      !!   `h_out = w_nbr·h_nbr + (1-w_nbr)·h_loc`.
      !! Wet neighbour (`w_nbr=1`) ⇒ `h_out = h_nbr` (literal no-op);
      !! land neighbour (`w_nbr=0`) ⇒ `h_out = h_loc`.
      real(wp), intent(in) :: h_nbr  !! neighbour value
      real(wp), intent(in) :: h_loc  !! local-cell value (the mirror target)
      real(wp), intent(in) :: w_nbr  !! neighbour wet mask (0/1)
      real(wp) :: h_out
      h_out = w_nbr*h_nbr + (1.0_wp - w_nbr)*h_loc
   end function ppm_mirror_h

   pure subroutine ppm_limited_slope(h_im1, h_i, h_ip1, dh)
      DC_ROUTINE_SEQ
      !! Van Leer monotonized centred slope for cell i.  Returns 0
      !! at local extrema (sign change between left and right
      !! differences) and the slope-limited centred derivative
      !! otherwise.  Standard PPM convention; see Colella-Woodward
      !! 1984.
      real(wp), intent(in) :: h_im1, h_i, h_ip1
      real(wp), intent(out) :: dh
      real(wp) :: dh_centered, dh_left, dh_right

      dh_left = h_i - h_im1
      dh_right = h_ip1 - h_i
      dh_centered = 0.5_wp*(dh_left + dh_right)
      if (dh_left*dh_right > 0.0_wp) then
         dh = sign(min(abs(dh_centered), &
                       2.0_wp*abs(dh_left), &
                       2.0_wp*abs(dh_right)), &
                   dh_centered)
      else
         dh = 0.0_wp
      end if
   end subroutine ppm_limited_slope

   pure subroutine ppm_cell_limiter(h_centre, h_left, h_right)
      DC_ROUTINE_SEQ
      !! Colella-Woodward 1984 eq 1.10 monotonic limiter on the
      !! parabolic profile in a single cell.  Three branches:
      !!
      !!   1. Local extremum in cell (h_centre lies outside
      !!      [min(h_left,h_right), max(h_left,h_right)]): flatten
      !!      the parabola — h_left = h_right = h_centre.
      !!   2. "Overshoot" at the left edge — reset h_left so the
      !!      parabola's minimum/maximum lies at the right edge.
      !!   3. "Overshoot" at the right edge — symmetric.
      real(wp), intent(in) :: h_centre
      real(wp), intent(inout) :: h_left, h_right
      real(wp) :: dh_lr, h_six

      dh_lr = h_right - h_left
      h_six = 6.0_wp*(h_centre - 0.5_wp*(h_left + h_right))
      if ((h_right - h_centre)*(h_centre - h_left) <= 0.0_wp) then
         h_left = h_centre
         h_right = h_centre
      else if (dh_lr*h_six > dh_lr*dh_lr) then
         h_left = 3.0_wp*h_centre - 2.0_wp*h_right
      else if (dh_lr*h_six < -dh_lr*dh_lr) then
         h_right = 3.0_wp*h_centre - 2.0_wp*h_left
      end if
   end subroutine ppm_cell_limiter

   pure subroutine ppm_limit_pos(h_centre, h_left, h_right, h_min)
      DC_ROUTINE_SEQ
      !! Positivity-preserving limiter on the PPM reconstruction.
      !! Mirrors MOM6's `PPM_limit_pos` (`MOM_continuity_PPM.F90`):
      !! when the parabolic fit predicts a minimum interior to the
      !! cell that dips below `h_min`, shrink h_left / h_right toward
      !! h_centre so the minimum sits at exactly `h_min`.  Pure
      !! scalar form per cell; runs after `ppm_cell_limiter` so the
      !! monotonic-limited reconstruction is the input.
      !!
      !! Algorithm:
      !!   curv = 3·(h_L + h_R − 2·h_in)         ! +ve ⇒ interior min
      !!   if curv > 0 and |dh| < curv:           ! min inside cell
      !!     if h_in ≤ h_min: flatten (h_L = h_R = h_in)
      !!     elif 12·curv·(h_in − h_min) < curv² + 3·dh²:
      !!        loc_scale = 12·curv·(h_in − h_min) / (curv² + 3·dh²) ∈ (0,1)
      !!        h_L = h_in + loc_scale·(h_L − h_in)
      !!        h_R = h_in + loc_scale·(h_R − h_in)
      !!
      !! `h_min = 0` ⇒ pure positivity (parabola can't go negative
      !! inside the cell).  Larger `h_min` ⇒ harder floor; matches
      !! MOM6's `GV%Angstrom_H` for the reduced-gravity setup.
      !!
      !! No-op when curv ≤ 0 (maximum interior, or linear / monotone
      !! profile) or |dh| ≥ curv (minimum outside the cell, edges
      !! already control).
      real(wp), intent(in) :: h_centre, h_min
      real(wp), intent(inout) :: h_left, h_right
      real(wp) :: curv, dh, loc_scale

      curv = 3.0_wp*((h_left + h_right) - 2.0_wp*h_centre)
      if (curv > 0.0_wp) then
         dh = h_right - h_left
         if (abs(dh) < curv) then
            if (h_centre <= h_min) then
               h_left = h_centre
               h_right = h_centre
            else if (12.0_wp*curv*(h_centre - h_min) < (curv*curv + 3.0_wp*dh*dh)) then
               loc_scale = 12.0_wp*curv*(h_centre - h_min)/(curv*curv + 3.0_wp*dh*dh)
               h_left = h_centre + loc_scale*(h_left - h_centre)
               h_right = h_centre + loc_scale*(h_right - h_centre)
            end if
         end if
      end if
   end subroutine ppm_limit_pos
end module continuity_layered
