!! SIGNATURE-FIX control for the barotropic substep.
!!
!! Identical body to btstep.F90's `btstep_nonlinear_closed` -- same loops,
!! same arithmetic, same `!$acc kernels async(1) default(present)` structure. The ONLY difference
!! is the SIGNATURE: every array arrives as a plain explicit-shape dummy bounded
!! by plain integers, instead of through derived-type components.
!!
!! WHY: the DC substep issues 103 global loads in Pass 1 against the CUDA port's
!! 47, with ZERO spills on either side. Pass 1 touches ~19 distinct arrays, all
!! reached through `bt_work%` / `metrics%` / `cor%`; the CUDA port gets raw
!! pointers in a by-value struct. Hypothesis: each component access reloads a
!! descriptor base pointer that is never cached in a register, and the cost
!! scales with the NUMBER of distinct arrays -- which would explain why the
!! continuity kernel (6 arrays, heavy PPM math) loses only 1.1x while this one
!! (19 arrays, light arithmetic) loses 2.5x.
!!
!! This file tests that hypothesis. If LDG drops toward 47 and the wall-clock
!! closes on the CUDA port, the fix for the ocean model is a SIGNATURE CHANGE, not a CUDA
!! rewrite. If it does not, the hypothesis is dead -- as four other plausible
!! mechanisms in this investigation already are (RESUME §1).
module btstep_noacc_mod
   use constants, only: wp
   implicit none
   private
   public :: btstep_noacc

contains

   subroutine btstep_noacc(nx, ny, nghost, nx_phys, ny_phys, G, bebt, dt_inner, n_steps, &
                           eta, eta_new, H_ref, ubt, vbt, ubt_prev, vbt_prev, &
                           rem_u, rem_v, zeta, ke, ubt_sum, vbt_sum, eta_sum, &
                           uhbt_sum, vhbt_sum, dy_cu, dx_cv, iareaT, areaCu, areaCv, &
                           dxCu, dyCv, idxCu, idyCv, iareaBu, f_corner, force_u, force_v)
      !! Identical body to btstep_nonlinear_closed -- ONLY the signature differs:
      !! every array arrives as a PLAIN explicit-shape dummy bounded by plain
      !! integers, instead of through derived-type components.
      integer, intent(in) :: nx, ny, nghost, nx_phys, ny_phys, n_steps
      real(wp), intent(in) :: G, bebt, dt_inner
      real(wp), intent(inout) :: eta(nx, ny), eta_new(nx, ny), ke(nx, ny), eta_sum(nx, ny)
      real(wp), intent(in) :: H_ref(nx, ny)
      real(wp), intent(inout) :: ubt(nx+1, ny), ubt_prev(nx+1, ny), ubt_sum(nx+1, ny), uhbt_sum(nx+1, ny)
      real(wp), intent(inout) :: vbt(nx, ny+1), vbt_prev(nx, ny+1), vbt_sum(nx, ny+1), vhbt_sum(nx, ny+1)
      real(wp), intent(in) :: rem_u(nx+1, ny), rem_v(nx, ny+1)
      real(wp), intent(inout) :: zeta(nx+1, ny+1)
      real(wp), intent(in) :: dy_cu(nx+1, ny), dx_cv(nx, ny+1), iareaT(nx, ny)
      real(wp), intent(in) :: areaCu(nx+1, ny), areaCv(nx, ny+1)
      real(wp), intent(in) :: dxCu(nx+1, ny), dyCv(nx, ny+1)
      real(wp), intent(in) :: idxCu(nx+1, ny), idyCv(nx, ny+1), iareaBu(nx+1, ny+1)
      real(wp), intent(in) :: f_corner(nx+1, ny+1)
      real(wp), intent(in) :: force_u(nx+1, ny), force_v(nx, ny+1)

      integer :: i, j, n

      real(wp) :: h_face_E, h_face_W, h_face_N, h_face_S
      real(wp) :: ubt_R, ubt_L, vbt_N, vbt_S
      real(wp) :: flux_x_R, flux_x_L, flux_y_N, flux_y_S, div_h_u
      real(wp) :: zeta_at_u, f_at_u, v_at_u, ke_grad_x, d_eta
      real(wp) :: zeta_at_v, f_at_v, u_at_v, ke_grad_y

      do concurrent(j=1:ny, i=1:nx)
         eta_sum(i, j) = 0.0_wp
      end do
      do concurrent(j=1:ny, i=1:nx + 1)
         ubt_sum(i, j) = 0.0_wp
         uhbt_sum(i, j) = 0.0_wp
      end do
      do concurrent(j=1:ny + 1, i=1:nx)
         vbt_sum(i, j) = 0.0_wp
         vhbt_sum(i, j) = 0.0_wp
      end do

      do n = 1, n_steps
         ! Drained by the !$acc wait(1) after the n_steps loop.

         ! ---- prev-save for the BEBT projection (bebt > 0) ----
         if (bebt > 0.0_wp) then
            do concurrent(j=1:ny, i=1:nx + 1)
               ubt_prev(i, j) = ubt(i, j)
            end do
            do concurrent(j=1:ny + 1, i=1:nx)
               vbt_prev(i, j) = vbt(i, j)
            end do
         end if

         ! ---- Pass 1: eta update with (H_ref + eta) face thickness ----
         ! KE (centre) and interior zeta (NE corner) ride along this sweep:
         ! they read only pre-update ubt/vbt and write disjoint arrays.
         do concurrent(j=1:ny, i=1:nx) &
            local(h_face_E, h_face_W, h_face_N, h_face_S, &
                  ubt_R, ubt_L, vbt_N, vbt_S, &
                  flux_x_R, flux_x_L, flux_y_N, flux_y_S, div_h_u)
            if (i < nx) then
               h_face_E = 0.5_wp*((H_ref(i, j) + eta(i, j)) + &
                                  (H_ref(i + 1, j) + eta(i + 1, j)))
            else
               h_face_E = H_ref(i, j) + eta(i, j)
            end if
            if (i > 1) then
               h_face_W = 0.5_wp*((H_ref(i - 1, j) + eta(i - 1, j)) + &
                                  (H_ref(i, j) + eta(i, j)))
            else
               h_face_W = H_ref(i, j) + eta(i, j)
            end if
            if (j < ny) then
               h_face_N = 0.5_wp*((H_ref(i, j) + eta(i, j)) + &
                                  (H_ref(i, j + 1) + eta(i, j + 1)))
            else
               h_face_N = H_ref(i, j) + eta(i, j)
            end if
            if (j > 1) then
               h_face_S = 0.5_wp*((H_ref(i, j - 1) + eta(i, j - 1)) + &
                                  (H_ref(i, j) + eta(i, j)))
            else
               h_face_S = H_ref(i, j) + eta(i, j)
            end if
            ubt_R = (1.0_wp + bebt)*ubt(i + 1, j) - bebt*ubt_prev(i + 1, j)
            ubt_L = (1.0_wp + bebt)*ubt(i, j) - bebt*ubt_prev(i, j)
            vbt_N = (1.0_wp + bebt)*vbt(i, j + 1) - bebt*vbt_prev(i, j + 1)
            vbt_S = (1.0_wp + bebt)*vbt(i, j) - bebt*vbt_prev(i, j)
            flux_x_R = h_face_E*ubt_R*dy_cu(i + 1, j)
            flux_x_L = h_face_W*ubt_L*dy_cu(i, j)
            flux_y_N = h_face_N*vbt_N*dx_cv(i, j + 1)
            flux_y_S = h_face_S*vbt_S*dx_cv(i, j)
            div_h_u = ((flux_x_R - flux_x_L) + (flux_y_N - flux_y_S))*iareaT(i, j)
            eta_new(i, j) = eta(i, j) - dt_inner*div_h_u
            ! Cell (i,j) owns its east face (i+1,j) and north face (i,j+1)
            ! for the transport accumulator, so the per-face sums are race-free.
            uhbt_sum(i + 1, j) = uhbt_sum(i + 1, j) + flux_x_R
            vhbt_sum(i, j + 1) = vhbt_sum(i, j + 1) + flux_y_N
            ke(i, j) = 0.25_wp*iareaT(i, j)*( &
                                         areaCu(i, j)*ubt(i, j)**2 + &
                                         areaCu(i + 1, j)*ubt(i + 1, j)**2 + &
                                         areaCv(i, j)*vbt(i, j)**2 + &
                                         areaCv(i, j + 1)*vbt(i, j + 1)**2)
            if (i >= 2 .and. j >= 2) then
               zeta(i, j) = &
                  ((vbt(i, j)*dyCv(i, j) - vbt(i - 1, j)*dyCv(i - 1, j)) - &
                   (ubt(i, j)*dxCu(i, j) - ubt(i, j - 1)*dxCu(i, j - 1)))* &
                  iareaBu(i, j)
            end if
         end do

         ! ---- eta-swap + zeta wall closure — one barrier sweep ----
         ! Both must finish before Pass 2b reads eta/zeta neighbours; they
         ! write disjoint arrays (eta centres vs zeta corner ring).
         ! CLOSED BASIN: zeta = 0 on the outer ring AND at the physical walls.
         do concurrent(j=1:ny + 1, i=1:nx + 1)
            if (i <= nx .and. j <= ny) eta(i, j) = eta_new(i, j)
            if (i == 1 .or. i == nx + 1) zeta(i, j) = 0.0_wp
            if (j == 1 .or. j == ny + 1) zeta(i, j) = 0.0_wp
            if (i == nghost + 1) zeta(i, j) = 0.0_wp
            if (i == nghost + nx_phys + 1) zeta(i, j) = 0.0_wp
            if (j == nghost + 1) zeta(i, j) = 0.0_wp
            if (j == nghost + ny_phys + 1) zeta(i, j) = 0.0_wp
         end do

         ! ---- Pass 2b: u_bt update at interior east faces ----
         do concurrent(j=1:ny, i=2:nx) &
            local(zeta_at_u, f_at_u, v_at_u, ke_grad_x, d_eta)
            zeta_at_u = 0.5_wp*(zeta(i, j) + zeta(i, j + 1))
            f_at_u = 0.5_wp*(f_corner(i, j) + f_corner(i, j + 1))
            if (j > 1 .and. j < ny) then
               v_at_u = 0.25_wp*(vbt(i - 1, j) + vbt(i - 1, j + 1) + &
                                 vbt(i, j) + vbt(i, j + 1))
            else if (j == 1) then
               v_at_u = 0.5_wp*(vbt(i - 1, j + 1) + vbt(i, j + 1))
            else
               v_at_u = 0.5_wp*(vbt(i - 1, j) + vbt(i, j))
            end if
            ke_grad_x = (ke(i, j) - ke(i - 1, j))*idxCu(i, j)
            d_eta = eta(i, j) - eta(i - 1, j)
            ubt(i, j) = rem_u(i, j)*( &
                                   ubt(i, j) + dt_inner*( &
                                   (zeta_at_u + f_at_u)*v_at_u &
                                   - G*d_eta*idxCu(i, j) &
                                   - ke_grad_x &
                                   + force_u(i, j)))
         end do
         ! CLOSED BASIN: hard zero at the array edges and the physical walls.
         do concurrent(j=1:ny)
            ubt(1, j) = 0.0_wp
            ubt(nx + 1, j) = 0.0_wp
            ubt(nghost + 1, j) = 0.0_wp
            ubt(nghost + nx_phys + 1, j) = 0.0_wp
         end do

         ! ---- Pass 2c: v_bt update at interior north faces ----
         do concurrent(j=2:ny, i=1:nx) &
            local(zeta_at_v, f_at_v, u_at_v, ke_grad_y, d_eta)
            zeta_at_v = 0.5_wp*(zeta(i, j) + zeta(i + 1, j))
            f_at_v = 0.5_wp*(f_corner(i, j) + f_corner(i + 1, j))
            if (i > 1 .and. i < nx) then
               u_at_v = 0.25_wp*(ubt(i, j - 1) + ubt(i + 1, j - 1) + &
                                 ubt(i, j) + ubt(i + 1, j))
            else if (i == 1) then
               u_at_v = 0.5_wp*(ubt(i + 1, j - 1) + ubt(i + 1, j))
            else
               u_at_v = 0.5_wp*(ubt(i, j - 1) + ubt(i, j))
            end if
            ke_grad_y = (ke(i, j) - ke(i, j - 1))*idyCv(i, j)
            d_eta = eta(i, j) - eta(i, j - 1)
            vbt(i, j) = rem_v(i, j)*( &
                                   vbt(i, j) + dt_inner*( &
                                   -(zeta_at_v + f_at_v)*u_at_v &
                                   - G*d_eta*idyCv(i, j) &
                                   - ke_grad_y &
                                   + force_v(i, j)))
         end do
         do concurrent(i=1:nx)
            vbt(i, 1) = 0.0_wp
            vbt(i, ny + 1) = 0.0_wp
            vbt(i, nghost + 1) = 0.0_wp
            vbt(i, nghost + ny_phys + 1) = 0.0_wp
         end do

         ! ---- time-mean accumulators ----
         do concurrent(j=1:ny, i=1:nx)
            eta_sum(i, j) = eta_sum(i, j) + eta(i, j)
         end do
         do concurrent(j=1:ny, i=1:nx + 1)
            ubt_sum(i, j) = ubt_sum(i, j) + ubt(i, j)
         end do
         do concurrent(j=1:ny + 1, i=1:nx)
            vbt_sum(i, j) = vbt_sum(i, j) + vbt(i, j)
         end do
      end do

   end subroutine btstep_noacc

end module btstep_noacc_mod
