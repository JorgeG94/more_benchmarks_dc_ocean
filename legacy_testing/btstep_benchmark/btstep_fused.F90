!! FUSED variant — same body as btstep_plain.F90, 11 loops -> 5 per substep.
!!
!! WHY: the CUDA port showed fusion (11 kern -> 5) is worth 1.134x, while CUDA
!! graphs on top are worth only 1.022x. Fusion is NOT a CUDA feature -- it is a
!! source transform, expressible in `do concurrent` exactly as the .cu does it.
!! This file tests whether Fortran gets the same win, which decides whether the
!! CUDA case collapses to "graphs, 2%".
!!
!! WHAT IS FUSED (mirroring btstep_kernel.cu's FUSE_WALL path):
!!   * eta_sum   -> folded into the eta-swap sweep. eta is final there (Pass 2
!!     never writes it), so accumulating early is identical.
!!   * u walls + ubt_sum -> folded into Pass 2b, which now runs the FULL face
!!     range 1:nx+1 with a single-assignment merge:
!!         ubt = (wall ? 0 : computed);  ubt_sum += ubt
!!     Same values, same order -> bit-identical, not an approximation.
!!   * v walls + vbt_sum -> same, into Pass 2c.
!!
!! WHAT IS NOT FUSED, and why (checked, not assumed):
!!   * Pass 2b -> Pass 2c is SEQUENTIAL: 2c's `u_at_v` reads the `ubt` 2b just
!!     wrote (Gauss-Seidel, not Jacobi). Fusing changes the answer.
!!   * Pass 1 -> swap -> Pass 2b are separated by real barriers: Pass 2b reads
!!     eta/zeta NEIGHBOURS, so every thread's producer must have landed first.
!!   * prev-save stays 2 loops: disjoint arrays, different shapes, and it only
!!     runs when bebt > 0.
!! 5 loops per substep is that dependency structure, not an implementation limit.
module btstep_fused_mod
   use constants, only: wp
   implicit none
   private
   public :: btstep_fused

contains

   subroutine btstep_fused(nx, ny, nghost, nx_phys, ny_phys, G, bebt, dt_inner, n_steps, &
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

      !$acc kernels async(1)
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
      !$acc end kernels

      do n = 1, n_steps
         ! Drained by the !$acc wait(1) after the n_steps loop.
         !$acc kernels async(1)

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
            ! FUSED: eta is final for the substep here (Pass 2 never writes it),
            ! so accumulating now is identical to accumulating at the end.
            if (i <= nx .and. j <= ny) then
               eta(i, j) = eta_new(i, j)
               eta_sum(i, j) = eta_sum(i, j) + eta(i, j)
            end if
            if (i == 1 .or. i == nx + 1) zeta(i, j) = 0.0_wp
            if (j == 1 .or. j == ny + 1) zeta(i, j) = 0.0_wp
            if (i == nghost + 1) zeta(i, j) = 0.0_wp
            if (i == nghost + nx_phys + 1) zeta(i, j) = 0.0_wp
            if (j == nghost + 1) zeta(i, j) = 0.0_wp
            if (j == nghost + ny_phys + 1) zeta(i, j) = 0.0_wp
         end do

         ! ---- Pass 2b: u_bt update at interior east faces ----
         ! FUSED: walls + ubt_sum folded in as a single-assignment merge over
         ! the FULL face range 1:nx+1. Same values, same order -> bit-identical.
         do concurrent(j=1:ny, i=1:nx + 1) &
            local(zeta_at_u, f_at_u, v_at_u, ke_grad_x, d_eta)
            if (i == 1 .or. i == nx + 1 .or. i == nghost + 1 &
                .or. i == nghost + nx_phys + 1) then
               ubt(i, j) = 0.0_wp
            else
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
            end if
            ubt_sum(i, j) = ubt_sum(i, j) + ubt(i, j)
         end do

         ! ---- Pass 2c: v_bt update at interior north faces ----
         do concurrent(j=1:ny + 1, i=1:nx) &
            local(zeta_at_v, f_at_v, u_at_v, ke_grad_y, d_eta)
            if (j == 1 .or. j == ny + 1 .or. j == nghost + 1 &
                .or. j == nghost + ny_phys + 1) then
               vbt(i, j) = 0.0_wp
            else
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
            end if
            vbt_sum(i, j) = vbt_sum(i, j) + vbt(i, j)
         end do

         ! (accumulators fused into the swap / Pass 2b / Pass 2c sweeps above)
         !$acc end kernels
      end do
      !$acc wait(1)   ! single drain: the whole substep ran on async(1)

   end subroutine btstep_fused

end module btstep_fused_mod
