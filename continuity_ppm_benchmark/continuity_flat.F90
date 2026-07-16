!! CONTROL variant of continuity_compute_fluxes_barotropic: the PPM helper
!! calls hand-inlined into the loop body. Everything else is verbatim from
!! continuity.F90.
!!
!! WHY: this is the experiment, not a proposed rewrite. The HLL flux kernel
!! loses 22% because nvfortran fails to CSE across an inlined call boundary
!! (RESUME §5a). That defect needs the CALLEE to do its own array indexing
!! (§5b) -- and this kernel's helpers take SCALARS, because the caller hoists
!! every read (h0 = bs%h(i,j)) before calling. So the prediction is:
!!
!!     FLAT == production, to the instruction.
!!
!! If that holds, the ocean path is clean and the scalar-hoisting idiom is
!! what protects it. If FLAT wins, the §5b mechanism is incomplete and the
!! bug report needs revisiting.
!!
!! WHAT IS INLINED: the four calls that actually execute --
!!   ppm_mirror_h    x4  (elemental, branchless)
!!   ppm_limited_slope x3
!!   ppm_cell_limiter  x1
!! `if (do_pos) call ppm_limit_pos(...)` is left as a call in BOTH variants:
!! use_ppm_limit_pos is .false. by default so it never executes, and keeping
!! it identical on both sides keeps the comparison to the executing path.
module continuity_flat
   use constants, only: wp
   use grid, only: hgrid_t
   use ocean_metrics, only: ocean_metrics_t
   use barotropic_cgrid_state, only: barotropic_cgrid_state_t
   use continuity, only: continuity_t, ppm_limit_pos
   implicit none
   private
   public :: continuity_compute_fluxes_barotropic_flat

contains

   pure subroutine continuity_compute_fluxes_barotropic_flat(grid, metrics, this, bs)
      type(hgrid_t), intent(in) :: grid
      type(ocean_metrics_t), intent(in) :: metrics
      type(continuity_t), intent(inout) :: this
      type(barotropic_cgrid_state_t), intent(inout) :: bs

      integer :: i, j, nx, ny
      real(wp) :: dh_m1, dh_0, dh_p1, h_left, h_right, u, v, h_face
      real(wp) :: hm2, hm1, h0, hp1, hp2
      real(wp) :: dl, dr, dc, dh_lr, h_six
      real(wp) :: wm1, wp1, wm2, wp2
      logical :: do_pos
      real(wp) :: h_min_pos

      nx = grid%nx_total
      ny = grid%ny_total
      do_pos = this%use_ppm_limit_pos
      h_min_pos = this%h_min

      ! ============================================================
      ! X-DIRECTION reconstruction -- helpers inlined
      ! ============================================================
      do concurrent(j=1:ny, i=3:nx - 2) &
         local(dh_m1, dh_0, dh_p1, h_left, h_right, &
               hm2, hm1, h0, hp1, hp2, dl, dr, dc, dh_lr, h_six, &
               wm1, wp1, wm2, wp2)
         h0 = bs%h(i, j)
         ! ppm_mirror_h inlined: h_out = w*h_nbr + (1-w)*h_loc
         wm1 = metrics%wet_T(i - 1, j)
         wp1 = metrics%wet_T(i + 1, j)
         wm2 = metrics%wet_T(i - 2, j)
         wp2 = metrics%wet_T(i + 2, j)
         hm1 = wm1*bs%h(i - 1, j) + (1.0_wp - wm1)*h0
         hp1 = wp1*bs%h(i + 1, j) + (1.0_wp - wp1)*h0
         hm2 = wm2*bs%h(i - 2, j) + (1.0_wp - wm2)*hm1
         hp2 = wp2*bs%h(i + 2, j) + (1.0_wp - wp2)*hp1
         ! ppm_limited_slope(hm2, hm1, h0, dh_m1) inlined
         dl = hm1 - hm2
         dr = h0 - hm1
         dc = 0.5_wp*(dl + dr)
         if (dl*dr > 0.0_wp) then
            dh_m1 = sign(min(abs(dc), 2.0_wp*abs(dl), 2.0_wp*abs(dr)), dc)
         else
            dh_m1 = 0.0_wp
         end if
         ! ppm_limited_slope(hm1, h0, hp1, dh_0) inlined
         dl = h0 - hm1
         dr = hp1 - h0
         dc = 0.5_wp*(dl + dr)
         if (dl*dr > 0.0_wp) then
            dh_0 = sign(min(abs(dc), 2.0_wp*abs(dl), 2.0_wp*abs(dr)), dc)
         else
            dh_0 = 0.0_wp
         end if
         ! ppm_limited_slope(h0, hp1, hp2, dh_p1) inlined
         dl = hp1 - h0
         dr = hp2 - hp1
         dc = 0.5_wp*(dl + dr)
         if (dl*dr > 0.0_wp) then
            dh_p1 = sign(min(abs(dc), 2.0_wp*abs(dl), 2.0_wp*abs(dr)), dc)
         else
            dh_p1 = 0.0_wp
         end if
         dh_0 = dh_0*wm1*metrics%wet_T(i, j)*wp1
         h_left = 0.5_wp*(hm1 + h0) - (dh_0 - dh_m1)/6.0_wp
         h_right = 0.5_wp*(h0 + hp1) - (dh_p1 - dh_0)/6.0_wp
         ! ppm_cell_limiter(h0, h_left, h_right) inlined
         dh_lr = h_right - h_left
         h_six = 6.0_wp*(h0 - 0.5_wp*(h_left + h_right))
         if ((h_right - h0)*(h0 - h_left) <= 0.0_wp) then
            h_left = h0
            h_right = h0
         else if (dh_lr*h_six > dh_lr*dh_lr) then
            h_left = 3.0_wp*h0 - 2.0_wp*h_right
         else if (dh_lr*h_six < -dh_lr*dh_lr) then
            h_right = 3.0_wp*h0 - 2.0_wp*h_left
         end if
         if (do_pos) call ppm_limit_pos(h0, h_left, h_right, h_min_pos)
         this%h_face_right_x%data(i, j, 1) = h_left
         this%h_face_left_x%data(i + 1, j, 1) = h_right
      end do

      ! ---- boundary + transport + divergence: verbatim from production ----
      do concurrent(j=1:ny)
         this%h_face_left_x%data(1, j, 1) = bs%h(1, j)
         this%h_face_right_x%data(1, j, 1) = bs%h(1, j)
         this%h_face_left_x%data(2, j, 1) = bs%h(1, j)
         this%h_face_right_x%data(2, j, 1) = bs%h(2, j)
         this%h_face_left_x%data(3, j, 1) = bs%h(2, j)
         this%h_face_right_x%data(3, j, 1) = bs%h(2, j)
         this%h_face_right_x%data(nx - 1, j, 1) = bs%h(nx - 1, j)
         this%h_face_left_x%data(nx, j, 1) = bs%h(nx - 1, j)
         this%h_face_right_x%data(nx, j, 1) = bs%h(nx, j)
         this%h_face_left_x%data(nx + 1, j, 1) = bs%h(nx, j)
         this%h_face_right_x%data(nx + 1, j, 1) = bs%h(nx, j)
      end do

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
      ! Y-DIRECTION reconstruction -- helpers inlined
      ! ============================================================
      do concurrent(j=3:ny - 2, i=1:nx) &
         local(dh_m1, dh_0, dh_p1, h_left, h_right, &
               hm2, hm1, h0, hp1, hp2, dl, dr, dc, dh_lr, h_six, &
               wm1, wp1, wm2, wp2)
         h0 = bs%h(i, j)
         wm1 = metrics%wet_T(i, j - 1)
         wp1 = metrics%wet_T(i, j + 1)
         wm2 = metrics%wet_T(i, j - 2)
         wp2 = metrics%wet_T(i, j + 2)
         hm1 = wm1*bs%h(i, j - 1) + (1.0_wp - wm1)*h0
         hp1 = wp1*bs%h(i, j + 1) + (1.0_wp - wp1)*h0
         hm2 = wm2*bs%h(i, j - 2) + (1.0_wp - wm2)*hm1
         hp2 = wp2*bs%h(i, j + 2) + (1.0_wp - wp2)*hp1
         dl = hm1 - hm2
         dr = h0 - hm1
         dc = 0.5_wp*(dl + dr)
         if (dl*dr > 0.0_wp) then
            dh_m1 = sign(min(abs(dc), 2.0_wp*abs(dl), 2.0_wp*abs(dr)), dc)
         else
            dh_m1 = 0.0_wp
         end if
         dl = h0 - hm1
         dr = hp1 - h0
         dc = 0.5_wp*(dl + dr)
         if (dl*dr > 0.0_wp) then
            dh_0 = sign(min(abs(dc), 2.0_wp*abs(dl), 2.0_wp*abs(dr)), dc)
         else
            dh_0 = 0.0_wp
         end if
         dl = hp1 - h0
         dr = hp2 - hp1
         dc = 0.5_wp*(dl + dr)
         if (dl*dr > 0.0_wp) then
            dh_p1 = sign(min(abs(dc), 2.0_wp*abs(dl), 2.0_wp*abs(dr)), dc)
         else
            dh_p1 = 0.0_wp
         end if
         dh_0 = dh_0*wm1*metrics%wet_T(i, j)*wp1
         h_left = 0.5_wp*(hm1 + h0) - (dh_0 - dh_m1)/6.0_wp
         h_right = 0.5_wp*(h0 + hp1) - (dh_p1 - dh_0)/6.0_wp
         dh_lr = h_right - h_left
         h_six = 6.0_wp*(h0 - 0.5_wp*(h_left + h_right))
         if ((h_right - h0)*(h0 - h_left) <= 0.0_wp) then
            h_left = h0
            h_right = h0
         else if (dh_lr*h_six > dh_lr*dh_lr) then
            h_left = 3.0_wp*h0 - 2.0_wp*h_right
         else if (dh_lr*h_six < -dh_lr*dh_lr) then
            h_right = 3.0_wp*h0 - 2.0_wp*h_left
         end if
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

      do concurrent(j=1:ny, i=1:nx)
         bs%flux_h(i, j) = &
            ((bs%mass_flux_x(i + 1, j) - bs%mass_flux_x(i, j)) + &
             (bs%mass_flux_y(i, j + 1) - bs%mass_flux_y(i, j)))*metrics%iareaT(i, j)
      end do
   end subroutine continuity_compute_fluxes_barotropic_flat

end module continuity_flat
