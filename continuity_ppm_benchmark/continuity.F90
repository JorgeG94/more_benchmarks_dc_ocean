!! MRE stubs + the ocean model's PRODUCTION continuity-PPM barotropic flux kernel.
!!
!! VERBATIM from <model>-sea-ice @ feat/sea-ice,
!! src/core/ocean/kernels/structured/continuity_ppm/continuity.F90:
!!   continuity_compute_fluxes_barotropic  428-609
!!   ppm_mirror_h                         2311-2330
!!   ppm_limited_slope                    2364-2389
!!   ppm_cell_limiter                     2391-2419
!!   ppm_limit_pos                        2421-2471
!!
!! The real module is 3586 lines and pulls in six state types. Only the fields
!! THIS subroutine touches are stubbed (verified by grep against the source):
!!   grid    : nx_total, ny_total
!!   metrics : dy_cu, dx_cv, iareaT, wet_T
!!   bs      : h, u_face_x, v_face_y, flux_h, mass_flux_x, mass_flux_y
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

module barotropic_cgrid_state
   use constants, only: wp
   implicit none
   public
   type :: barotropic_cgrid_state_t
      real(wp), allocatable :: h(:, :)
      real(wp), allocatable :: u_face_x(:, :)
      real(wp), allocatable :: v_face_y(:, :)
      real(wp), allocatable :: flux_h(:, :)
      real(wp), allocatable :: mass_flux_x(:, :)
      real(wp), allocatable :: mass_flux_y(:, :)
   end type barotropic_cgrid_state_t
end module barotropic_cgrid_state

module continuity
   use constants, only: wp
   use grid, only: hgrid_t
   use ocean_metrics, only: ocean_metrics_t
   use scratch_3d, only: scratch_3d_buffer_t
   use barotropic_cgrid_state, only: barotropic_cgrid_state_t
   implicit none
   private
   public :: continuity_t, continuity_compute_fluxes_barotropic
   ! MRE-only: production keeps these private. The FLAT and ACC variants reuse
   ! them verbatim rather than duplicating ~120 lines. Scaffolding change --
   ! the subroutine bodies below are still the verbatim extract.
   public :: ppm_limit_pos, ppm_mirror_h, ppm_limited_slope, ppm_cell_limiter

   type :: continuity_t
      type(scratch_3d_buffer_t) :: h_face_left_x, h_face_right_x
      type(scratch_3d_buffer_t) :: h_face_left_y, h_face_right_y
      real(wp) :: h_min = 1.0e-6_wp
      logical  :: use_ppm_limit_pos = .false.
   end type continuity_t

contains

   pure subroutine continuity_compute_fluxes_barotropic(grid, metrics, this, bs)
      !! PPM face reconstruction + per-face mass flux + cell-centred
      !! flux divergence for the barotropic C-grid state.
      !!
      !! Curvilinear (design §2): the per-face TRANSPORT is
      !! `uh = u·h_face·dy_cu` [m³/s] (east) / `vh = v·h_face·dx_cv`
      !! [m³/s] (north), and the divergence is `Δuh·iareaT`.  On
      !! uniform Cartesian `dy_cu = dy`, `dx_cv = dx`, `iareaT =
      !! 1/(dx·dy)`, so `Δ(u·h·dy)/(dx·dy) = Δ(u·h)/dx` bitwise — the
      !! old `inv_dx`/`inv_dy` form.  The PPM reconstruction is a
      !! dimensionless h-difference (no dx enters the slope), so it is
      !! untouched.  `mass_flux_x/y` now carry the m³/s transport, and
      !! every downstream consumer (tracer advect, uhbt renormalise)
      !! reads the same width-weighted flux for tracer consistency.
      !!
      !! Algorithm (per direction, x shown; y mirrors):
      !!
      !!   1. Pass 1: for each cell i with valid 5-point stencil
      !!      (i.e. cells [i-2, i+2] all exist), compute the limited
      !!      slopes δh_{i-1}, δh_i, δh_{i+1} and from them the
      !!      face-left value h_L(i) (= h at i-1/2) and face-right
      !!      value h_R(i) (= h at i+1/2) via CW eq 1.6, then apply
      !!      eq 1.10 monotonic limiter.  Store h_L at
      !!      h_face_right_x(i) (the right state at face i = left
      !!      edge of cell i) and h_R at h_face_left_x(i+1) (left
      !!      state at face i+1 = right edge of cell i).
      !!
      !!   2. Pass 2: for each east face i, pick upwind based on
      !!      u_face_x sign and emit mass_flux_x = u * h_face.
      !!      Domain-wall faces (i=1, nx+1) force mass_flux_x = 0
      !!      (closed-wall BC).
      !!
      !!   3. Pass 3: cell-centred flux divergence flux_h.
      !!
      !! Cells closer than 2 to the boundary use first-order
      !! (h_L = h_R = h_centre) — the stencil is short and the
      !! flux there gets gated by the closed-wall BC anyway.
      !!
      !! Loop order: j-then-i for NVHPC GPU coalescing (CLAUDE.md
      !! memory feedback_do_concurrent_order).
      type(hgrid_t), intent(in) :: grid
      type(ocean_metrics_t), intent(in) :: metrics
      type(continuity_t), intent(inout) :: this
      type(barotropic_cgrid_state_t), intent(inout) :: bs

      integer :: i, j, nx, ny
      real(wp) :: dh_m1, dh_0, dh_p1, h_left, h_right, u, v, h_face
      real(wp) :: hm2, hm1, h0, hp1, hp2
      logical :: do_pos
      real(wp) :: h_min_pos

      nx = grid%nx_total
      ny = grid%ny_total
      do_pos = this%use_ppm_limit_pos
      h_min_pos = this%h_min

      ! ============================================================
      ! X-DIRECTION reconstruction (5-point stencil per cell)
      ! ============================================================
      ! Interior cells i = 3..nx-2 — full 5-point PPM
      do concurrent(j=1:ny, i=3:nx - 2) &
         local(dh_m1, dh_0, dh_p1, h_left, h_right, &
               hm2, hm1, h0, hp1, hp2)
         ! Mirror-h: replace a LAND neighbour's held floor-h with the
         ! local cell's so the parabola sees a reflected coast (C2).
         h0 = bs%h(i, j)
         hm1 = ppm_mirror_h(bs%h(i - 1, j), h0, metrics%wet_T(i - 1, j))
         hp1 = ppm_mirror_h(bs%h(i + 1, j), h0, metrics%wet_T(i + 1, j))
         hm2 = ppm_mirror_h(bs%h(i - 2, j), hm1, metrics%wet_T(i - 2, j))
         hp2 = ppm_mirror_h(bs%h(i + 2, j), hp1, metrics%wet_T(i + 2, j))
         call ppm_limited_slope(hm2, hm1, h0, dh_m1)
         call ppm_limited_slope(hm1, h0, hp1, dh_0)
         call ppm_limited_slope(h0, hp1, hp2, dh_p1)
         ! Slope-flatten: zero the centre slope if the 3-cell stencil
         ! touches land (MOM6 :2782); bit-identical for all-wet (×1).
         dh_0 = dh_0*metrics%wet_T(i - 1, j)*metrics%wet_T(i, j)*metrics%wet_T(i + 1, j)
         h_left = 0.5_wp*(hm1 + h0) - (dh_0 - dh_m1)/6.0_wp
         h_right = 0.5_wp*(h0 + hp1) - (dh_p1 - dh_0)/6.0_wp
         call ppm_cell_limiter(h0, h_left, h_right)
         if (do_pos) call ppm_limit_pos(h0, h_left, h_right, h_min_pos)
         this%h_face_right_x%data(i, j, 1) = h_left
         this%h_face_left_x%data(i + 1, j, 1) = h_right
      end do
      ! Boundary cells (i = 1, 2, nx-1, nx): 1st-order — h_face_*
      ! at the four border faces (1, 2, nx, nx+1) just take the
      ! abutting cell's centre value.  Face 1 and face nx+1 are
      ! walls (mass_flux forced to 0); face 2 and face nx use a
      ! one-sided downwind reconstruction equivalent to 1st-order.
      do concurrent(j=1:ny)
         this%h_face_left_x%data(1, j, 1) = bs%h(1, j)
         this%h_face_right_x%data(1, j, 1) = bs%h(1, j)
         this%h_face_left_x%data(2, j, 1) = bs%h(1, j)
         this%h_face_right_x%data(2, j, 1) = bs%h(2, j)
         ! Face 3: cell 2's right edge (h_face_left at face 3) falls
         ! back to 1st order — its 5-point PPM stencil needs cell 0.
         ! Same for cell 2's left-edge contribution at face 3
         ! (h_face_right at face 3).
         this%h_face_left_x%data(3, j, 1) = bs%h(2, j)
         this%h_face_right_x%data(3, j, 1) = bs%h(2, j)
         ! Face nx-1: mirror of face 3.
         this%h_face_right_x%data(nx - 1, j, 1) = bs%h(nx - 1, j)
         this%h_face_left_x%data(nx, j, 1) = bs%h(nx - 1, j)
         this%h_face_right_x%data(nx, j, 1) = bs%h(nx, j)
         this%h_face_left_x%data(nx + 1, j, 1) = bs%h(nx, j)
         this%h_face_right_x%data(nx + 1, j, 1) = bs%h(nx, j)
      end do

      ! X-direction face transport (upwind pick from h_face_left/right_x).
      ! uh = u·h_face·dy_cu(i,j) [m³/s] — face-width-weighted (design §2).
      do concurrent(j=1:ny, i=2:nx) local(u, h_face)
         u = bs%u_face_x(i, j)
         if (u >= 0.0_wp) then
            h_face = this%h_face_left_x%data(i, j, 1)
         else
            h_face = this%h_face_right_x%data(i, j, 1)
         end if
         bs%mass_flux_x(i, j) = u*h_face*metrics%dy_cu(i, j)
      end do
      do concurrent(j=1:ny)
         bs%mass_flux_x(1, j) = 0.0_wp
         bs%mass_flux_x(nx + 1, j) = 0.0_wp
      end do

      ! ============================================================
      ! Y-DIRECTION reconstruction (mirror of X)
      ! ============================================================
      do concurrent(j=3:ny - 2, i=1:nx) &
         local(dh_m1, dh_0, dh_p1, h_left, h_right, &
               hm2, hm1, h0, hp1, hp2)
         h0 = bs%h(i, j)
         hm1 = ppm_mirror_h(bs%h(i, j - 1), h0, metrics%wet_T(i, j - 1))
         hp1 = ppm_mirror_h(bs%h(i, j + 1), h0, metrics%wet_T(i, j + 1))
         hm2 = ppm_mirror_h(bs%h(i, j - 2), hm1, metrics%wet_T(i, j - 2))
         hp2 = ppm_mirror_h(bs%h(i, j + 2), hp1, metrics%wet_T(i, j + 2))
         call ppm_limited_slope(hm2, hm1, h0, dh_m1)
         call ppm_limited_slope(hm1, h0, hp1, dh_0)
         call ppm_limited_slope(h0, hp1, hp2, dh_p1)
         dh_0 = dh_0*metrics%wet_T(i, j - 1)*metrics%wet_T(i, j)*metrics%wet_T(i, j + 1)
         h_left = 0.5_wp*(hm1 + h0) - (dh_0 - dh_m1)/6.0_wp
         h_right = 0.5_wp*(h0 + hp1) - (dh_p1 - dh_0)/6.0_wp
         call ppm_cell_limiter(h0, h_left, h_right)
         if (do_pos) call ppm_limit_pos(h0, h_left, h_right, h_min_pos)
         this%h_face_right_y%data(i, j, 1) = h_left
         this%h_face_left_y%data(i, j + 1, 1) = h_right
      end do
      do concurrent(i=1:nx)
         this%h_face_left_y%data(i, 1, 1) = bs%h(i, 1)
         this%h_face_right_y%data(i, 1, 1) = bs%h(i, 1)
         this%h_face_left_y%data(i, 2, 1) = bs%h(i, 1)
         this%h_face_right_y%data(i, 2, 1) = bs%h(i, 2)
         this%h_face_left_y%data(i, 3, 1) = bs%h(i, 2)
         this%h_face_right_y%data(i, 3, 1) = bs%h(i, 2)
         this%h_face_right_y%data(i, ny - 1, 1) = bs%h(i, ny - 1)
         this%h_face_left_y%data(i, ny, 1) = bs%h(i, ny - 1)
         this%h_face_right_y%data(i, ny, 1) = bs%h(i, ny)
         this%h_face_left_y%data(i, ny + 1, 1) = bs%h(i, ny)
         this%h_face_right_y%data(i, ny + 1, 1) = bs%h(i, ny)
      end do

      do concurrent(j=2:ny, i=1:nx) local(v, h_face)
         v = bs%v_face_y(i, j)
         if (v >= 0.0_wp) then
            h_face = this%h_face_left_y%data(i, j, 1)
         else
            h_face = this%h_face_right_y%data(i, j, 1)
         end if
         bs%mass_flux_y(i, j) = v*h_face*metrics%dx_cv(i, j)
      end do
      do concurrent(i=1:nx)
         bs%mass_flux_y(i, 1) = 0.0_wp
         bs%mass_flux_y(i, ny + 1) = 0.0_wp
      end do

      ! ============================================================
      ! Cell-centred flux divergence (transport divergence · iareaT)
      ! ============================================================
      do concurrent(j=1:ny, i=1:nx)
         bs%flux_h(i, j) = &
            ((bs%mass_flux_x(i + 1, j) - bs%mass_flux_x(i, j)) + &
             (bs%mass_flux_y(i, j + 1) - bs%mass_flux_y(i, j)))*metrics%iareaT(i, j)
      end do
   end subroutine continuity_compute_fluxes_barotropic

   pure elemental function ppm_mirror_h(h_nbr, h_loc, w_nbr) result(h_out)
      !$acc routine seq
      !! Mirror-h at a land neighbour (spec §14 C2 / MOM6
      !! `MOM_continuity_PPM.F90:2764`): substitute the LOCAL cell's
      !! thickness (or tracer) for a LAND neighbour's held floor value
      !! so the PPM parabola sees a flat, reflected coast and the
      !! wet-side face value is not biased by the dry column.  Branchless:
      !!   `h_out = w_nbr·h_nbr + (1-w_nbr)·h_loc`.
      !! Wet neighbour (`w_nbr=1`) ⇒ `h_out = h_nbr` (literal no-op);
      !! land neighbour (`w_nbr=0`) ⇒ `h_out = h_loc`.
      !!
      !! Consumer: `ice_transport` (sea-ice PR 4b) reuses this
      !! verbatim for the category-summed ice/snow PPM parabola —
      !! SIS2's mask2dT substitution (`SIS_continuity.F90:1478-1479`).
      real(wp), intent(in) :: h_nbr  !! neighbour value
      real(wp), intent(in) :: h_loc  !! local-cell value (the mirror target)
      real(wp), intent(in) :: w_nbr  !! neighbour wet mask (0/1)
      real(wp) :: h_out
      h_out = w_nbr*h_nbr + (1.0_wp - w_nbr)*h_loc
   end function ppm_mirror_h

   pure subroutine ppm_limited_slope(h_im1, h_i, h_ip1, dh)
      !$acc routine seq
      !! Van Leer monotonized centred slope for cell i.  Returns 0
      !! at local extrema (sign change between left and right
      !! differences) and the slope-limited centred derivative
      !! otherwise.  Standard PPM convention; see Colella-Woodward
      !! 1984.
      !!
      !! Consumer: `ice_transport` (sea-ice PR 4b), same role as here
      !! (SIS2 Lin-94 slope + limit, `SIS_continuity.F90:1464-1468`).
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
      !$acc routine seq
      !! Colella-Woodward 1984 eq 1.10 monotonic limiter on the
      !! parabolic profile in a single cell.  Three branches:
      !!
      !!   1. Local extremum in cell (h_centre lies outside
      !!      [min(h_left,h_right), max(h_left,h_right)]): flatten
      !!      the parabola — h_left = h_right = h_centre.
      !!   2. "Overshoot" at the left edge — reset h_left so the
      !!      parabola's minimum/maximum lies at the right edge.
      !!   3. "Overshoot" at the right edge — symmetric.
      !!
      !! Consumer: `ice_transport` (sea-ice PR 4b) — SIS2's
      !! `PPM_limit_CW84`.
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
      !$acc routine seq
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
      !!
      !! Consumer: `ice_transport` (sea-ice PR 4b) — SIS2 runs this
      !! unconditionally on the PD continuity scheme
      !! (`SIS_continuity.F90:1585`, `PPM_limit_pos`), so the ice-mass
      !! reconstruction always applies it (not optional there, unlike
      !! the ocean's `use_ppm_limit_pos` knob).
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

end module continuity
