!! ASYNC variant: production body VERBATIM, wrapped in `!$acc kernels async(1)`.
!!
!! WHY: at small domains `do concurrent` loses badly to CUDA C -- 3.55x at 256^2,
!! falling to 1.07x at 4096^2. But the KERNELS are within ~12% (at 256^2 the DC
!! x-PPM kernel is 5.9 us vs the CUDA port's 5.2 us). The gap is ~10 us of dead
!! time between each of the 9 loops: plain `do concurrent` under -stdpar=gpu
!! blocks the host after every loop, while the CUDA port queues all 9 and
!! returns. It is a LAUNCH-MODEL difference, not codegen.
!!
!! This variant tests the fix: keep the `do concurrent` bodies exactly as they
!! ship and only change how they are queued. One `!$acc kernels async(1)` region
!! covers all 9 loops -- same queue means they still execute in order, so the
!! loop-to-loop dependencies (x-PPM writes h_face_left_x, x-transport reads it)
!! are preserved -- then ONE `!$acc wait(1)` drains at the end. 9 host syncs
!! per call become 1.
!!
!! IDIOM: copied from production, which already does exactly this --
!! <model>/src/core/ocean/kernels/structured/barotropic/barotropic_substep.F90
!! (`!$acc kernels async(1)` per group, `!$acc wait(1)` at :1369, "single drain:
!! the whole substep ran on async(1)"). Like that routine, this one is NOT
!! `pure` -- the shipped continuity kernel is, so that is the one deviation
!! from the verbatim body, and it is the same deviation production accepted.
module continuity_acc
   use constants, only: wp
   use grid, only: hgrid_t
   use ocean_metrics, only: ocean_metrics_t
   use barotropic_cgrid_state, only: barotropic_cgrid_state_t
   use continuity, only: continuity_t, ppm_mirror_h, ppm_limited_slope, &
                             ppm_cell_limiter, ppm_limit_pos
   implicit none
   private
   public :: continuity_compute_fluxes_barotropic_acc

contains

   subroutine continuity_compute_fluxes_barotropic_acc(grid, metrics, this, bs)
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

      !$acc kernels async(1)
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
      !$acc end kernels
      !$acc wait(1)   ! single drain: the whole kernel ran on async(1)
   end subroutine continuity_compute_fluxes_barotropic_acc

end module continuity_acc
