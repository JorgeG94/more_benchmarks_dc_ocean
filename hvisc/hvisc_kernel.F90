!! Ocean horizontal viscosity -- the Smagorinsky/Laplacian APPLICATION kernel.
!!
!! VERBATIM extract (anonymised) of <model>'s
!! src/parameterizations/lateral/structured ocean horizontal-viscosity module,
!! subroutine hvisc_compute_face_impl: the per-face curvilinear finite-volume
!! Laplacian of the C-grid face velocities, scaled by a spatially-varying
!! per-face viscosity `ah_face_{x,y}`. For the Smagorinsky closure that
!! viscosity field is A_h = (Cs*delta)^2 * |D| filled upstream from the
!! strain-rate magnitude; here it is an input, exactly as in production.
!!
!! This is the compute the `ocean_hvisc` profile region (~4.8% of runtime)
!! actually spends its time in. Plain-integer (explicit-shape) dummies, no
!! procedure calls in the loop body -> auto-collapses, no compiler-bug risk.
module ocean_horizontal_viscosity
   use constants, only: wp
   implicit none
   private
   public :: hvisc_compute_smag, hvisc_compute_face, hvisc_compute_fused

contains

   !! Smagorinsky harmonic viscosity: A_h(face) = clamp((Cs*sqrt(dxT*dyT))^2*|D|)
   !! with |D| = sqrt(D_T^2 + D_S^2); tension D_T cell-centred, shear D_S at the
   !! corner (slip-masked by wet_q / ns), each averaged onto the face. Fills
   !! ah_face_{x,y} consumed by hvisc_compute_face. Verbatim (anonymised); the
   !! optional VarMix resolution scaling is omitted (do_resoln=.false. path).
   pure subroutine hvisc_compute_smag(u_face, v_face, ah_face_x, ah_face_y, &
                                      dxT, dyT, idxT, idyT, dy_dxBu, dx_dyBu, &
                                      idyCv, idxCu, wet_q, c_smag, ah_bg, ah_max, ns, &
                                      nx, ny, nz)
      integer, intent(in) :: nx, ny, nz
      real(wp), intent(in)  :: u_face(nx + 1, ny, nz), v_face(nx, ny + 1, nz)
      real(wp), intent(out) :: ah_face_x(nx + 1, ny, nz), ah_face_y(nx, ny + 1, nz)
      real(wp), intent(in)  :: dxT(nx, ny), dyT(nx, ny), idxT(nx, ny), idyT(nx, ny)
      real(wp), intent(in)  :: dy_dxBu(nx + 1, ny + 1), dx_dyBu(nx + 1, ny + 1)
      real(wp), intent(in)  :: idyCv(nx, ny + 1), idxCu(nx + 1, ny), wet_q(nx + 1, ny + 1)
      real(wp), intent(in)  :: c_smag, ah_bg, ah_max, ns
      integer :: i, j, k
      real(wp) :: D_T_W, D_T_E, D_T_S, D_T_N, D_T_face
      real(wp) :: D_S_S, D_S_N, D_S_W, D_S_E, D_S_face
      real(wp) :: strain_mag, A_raw, smag_scale

      ! ---- u-face viscosity (i-1/2, j) ----
      do concurrent(k=1:nz, j=2:ny - 1, i=2:nx) &
         local(D_T_W, D_T_E, D_S_S, D_S_N, D_T_face, D_S_face, strain_mag, A_raw, smag_scale)
         smag_scale = (c_smag*sqrt(dxT(i, j)*dyT(i, j)))**2
         D_T_W = (u_face(i, j, k) - u_face(i - 1, j, k))*idxT(i - 1, j) &
                 - (v_face(i - 1, j + 1, k) - v_face(i - 1, j, k))*idyT(i - 1, j)
         D_T_E = (u_face(i + 1, j, k) - u_face(i, j, k))*idxT(i, j) &
                 - (v_face(i, j + 1, k) - v_face(i, j, k))*idyT(i, j)
         D_T_face = 0.5_wp*(D_T_W + D_T_E)
         D_S_S = ((1.0_wp - 2.0_wp*ns)*wet_q(i, j) + 2.0_wp*ns)* &
                 (dy_dxBu(i, j)*(v_face(i, j, k)*idyCv(i, j) - v_face(i - 1, j, k)*idyCv(i - 1, j)) + &
                  dx_dyBu(i, j)*(u_face(i, j, k)*idxCu(i, j) - u_face(i, j - 1, k)*idxCu(i, j - 1)))
         D_S_N = ((1.0_wp - 2.0_wp*ns)*wet_q(i, j + 1) + 2.0_wp*ns)* &
                 (dy_dxBu(i, j + 1)*(v_face(i, j + 1, k)*idyCv(i, j + 1) - v_face(i - 1, j + 1, k)*idyCv(i - 1, j + 1)) + &
                  dx_dyBu(i, j + 1)*(u_face(i, j + 1, k)*idxCu(i, j + 1) - u_face(i, j, k)*idxCu(i, j)))
         D_S_face = 0.5_wp*(D_S_S + D_S_N)
         strain_mag = sqrt(D_T_face*D_T_face + D_S_face*D_S_face)
         A_raw = smag_scale*strain_mag
         ah_face_x(i, j, k) = min(ah_max, max(ah_bg, A_raw))
      end do
      do concurrent(k=1:nz, i=2:nx)
         ah_face_x(i, 1, k) = ah_face_x(i, 2, k)
         ah_face_x(i, ny, k) = ah_face_x(i, ny - 1, k)
      end do
      do concurrent(k=1:nz, j=1:ny)
         ah_face_x(1, j, k) = ah_bg
         ah_face_x(nx + 1, j, k) = ah_bg
      end do

      ! ---- v-face viscosity (i, j-1/2) ----
      do concurrent(k=1:nz, j=2:ny, i=2:nx - 1) &
         local(D_T_S, D_T_N, D_S_W, D_S_E, D_T_face, D_S_face, strain_mag, A_raw, smag_scale)
         smag_scale = (c_smag*sqrt(dxT(i, j)*dyT(i, j)))**2
         D_T_S = (u_face(i + 1, j - 1, k) - u_face(i, j - 1, k))*idxT(i, j - 1) &
                 - (v_face(i, j, k) - v_face(i, j - 1, k))*idyT(i, j - 1)
         D_T_N = (u_face(i + 1, j, k) - u_face(i, j, k))*idxT(i, j) &
                 - (v_face(i, j + 1, k) - v_face(i, j, k))*idyT(i, j)
         D_T_face = 0.5_wp*(D_T_S + D_T_N)
         D_S_W = ((1.0_wp - 2.0_wp*ns)*wet_q(i, j) + 2.0_wp*ns)* &
                 (dy_dxBu(i, j)*(v_face(i, j, k)*idyCv(i, j) - v_face(i - 1, j, k)*idyCv(i - 1, j)) + &
                  dx_dyBu(i, j)*(u_face(i, j, k)*idxCu(i, j) - u_face(i, j - 1, k)*idxCu(i, j - 1)))
         D_S_E = ((1.0_wp - 2.0_wp*ns)*wet_q(i + 1, j) + 2.0_wp*ns)* &
                 (dy_dxBu(i + 1, j)*(v_face(i + 1, j, k)*idyCv(i + 1, j) - v_face(i, j, k)*idyCv(i, j)) + &
                  dx_dyBu(i + 1, j)*(u_face(i + 1, j, k)*idxCu(i + 1, j) - u_face(i + 1, j - 1, k)*idxCu(i + 1, j - 1)))
         D_S_face = 0.5_wp*(D_S_W + D_S_E)
         strain_mag = sqrt(D_T_face*D_T_face + D_S_face*D_S_face)
         A_raw = smag_scale*strain_mag
         ah_face_y(i, j, k) = min(ah_max, max(ah_bg, A_raw))
      end do
      do concurrent(k=1:nz, j=2:ny)
         ah_face_y(1, j, k) = ah_face_y(2, j, k)
         ah_face_y(nx, j, k) = ah_face_y(nx - 1, j, k)
      end do
      do concurrent(k=1:nz, i=1:nx)
         ah_face_y(i, 1, k) = ah_bg
         ah_face_y(i, ny + 1, k) = ah_bg
      end do
   end subroutine hvisc_compute_smag

   !! Per-face metric Laplacian x spatially-varying viscosity.
   !! u-faces live at (nx+1, ny), v-faces at (nx, ny+1); metrics are 2-D.
   !! Closed/free-slip walls get zero tendency.
   pure subroutine hvisc_compute_face(u_face, v_face, ah_face_x, ah_face_y, &
                                      du_visc, dv_visc, &
                                      dy_dxT, dx_dyBu, iareaCu, &
                                      dx_dyT, dy_dxBu, iareaCv, &
                                      nx, ny, nz)
      integer, intent(in) :: nx, ny, nz
      real(wp), intent(in)    :: u_face(nx + 1, ny, nz), v_face(nx, ny + 1, nz)
      real(wp), intent(in)    :: ah_face_x(nx + 1, ny, nz), ah_face_y(nx, ny + 1, nz)
      real(wp), intent(inout) :: du_visc(nx + 1, ny, nz), dv_visc(nx, ny + 1, nz)
      real(wp), intent(in)    :: dy_dxT(nx, ny), dx_dyBu(nx + 1, ny + 1), iareaCu(nx + 1, ny)
      real(wp), intent(in)    :: dx_dyT(nx, ny), dy_dxBu(nx + 1, ny + 1), iareaCv(nx, ny + 1)
      integer :: i, j, k
      real(wp) :: lap_u, lap_v

      ! u-face Laplacian interior + zero boundaries (curvilinear FV form)
      do concurrent(k=1:nz, j=2:ny - 1, i=2:nx) local(lap_u)
         lap_u = iareaCu(i, j)*( &
                 (dy_dxT(i, j)*(u_face(i + 1, j, k) - u_face(i, j, k)) - &
                  dy_dxT(i - 1, j)*(u_face(i, j, k) - u_face(i - 1, j, k))) + &
                 (dx_dyBu(i, j + 1)*(u_face(i, j + 1, k) - u_face(i, j, k)) - &
                  dx_dyBu(i, j)*(u_face(i, j, k) - u_face(i, j - 1, k))))
         du_visc(i, j, k) = ah_face_x(i, j, k)*lap_u
      end do
      do concurrent(k=1:nz, j=1:ny)
         du_visc(1, j, k) = 0.0_wp
         du_visc(nx + 1, j, k) = 0.0_wp
      end do
      do concurrent(k=1:nz, i=1:nx + 1)
         du_visc(i, 1, k) = 0.0_wp
         du_visc(i, ny, k) = 0.0_wp
      end do

      ! v-face Laplacian interior + zero boundaries (curvilinear FV form)
      do concurrent(k=1:nz, j=2:ny, i=2:nx - 1) local(lap_v)
         lap_v = iareaCv(i, j)*( &
                 (dx_dyT(i, j)*(v_face(i, j + 1, k) - v_face(i, j, k)) - &
                  dx_dyT(i, j - 1)*(v_face(i, j, k) - v_face(i, j - 1, k))) + &
                 (dy_dxBu(i + 1, j)*(v_face(i + 1, j, k) - v_face(i, j, k)) - &
                  dy_dxBu(i, j)*(v_face(i, j, k) - v_face(i - 1, j, k))))
         dv_visc(i, j, k) = ah_face_y(i, j, k)*lap_v
      end do
      do concurrent(k=1:nz, i=1:nx)
         dv_visc(i, 1, k) = 0.0_wp
         dv_visc(i, ny + 1, k) = 0.0_wp
      end do
      do concurrent(k=1:nz, j=1:ny + 1)
         dv_visc(1, j, k) = 0.0_wp
         dv_visc(nx, j, k) = 0.0_wp
      end do
   end subroutine hvisc_compute_face

   !! FUSED closure (best-CUDA-practice port back to do concurrent):
   !! smag(strain) -> ah_face and face(ah_face * Laplacian) -> du/dv are fused
   !! into ONE pass per face direction. The ah_face_{x,y} intermediates NEVER
   !! touch memory: each interior face thread computes its own ah in a register
   !! and feeds it straight into the Laplacian.
   !!
   !! Why this is bit-identical (no neighbour recompute needed): the faithful
   !! `hvisc_compute_face` reads ah_face_x(i,j,k) at EXACTLY the (i,j,k) it
   !! writes du_visc, and the du-interior range (j=2:ny-1, i=2:nx) is IDENTICAL
   !! to the smag u-face interior write range. So every consumed ah value is the
   !! one produced at the same cell -- the ah halo/reuse writes are dead code for
   !! the du/dv output and are simply dropped. Each per-cell expression is
   !! reproduced in the same operand order/grouping as the two-pass path, so the
   !! compiler contracts FMAs identically -> max|diff| == 0. Boundary faces write
   !! 0.0 inline, folding the wall-zeroing loops into the same pass.
   !!
   !! 12 loops (6 smag + 6 face) -> 2; two face-sized arrays of round-trip DRAM
   !! traffic removed. Inlined, plain-integer indexing -> auto-collapses, builds
   !! and runs DATA=none (CPU) unchanged.
   pure subroutine hvisc_compute_fused(u_face, v_face, du_visc, dv_visc, &
                                       dxT, dyT, idxT, idyT, dy_dxBu, dx_dyBu, &
                                       idyCv, idxCu, wet_q, dy_dxT, iareaCu, &
                                       dx_dyT, iareaCv, c_smag, ah_bg, ah_max, ns, &
                                       nx, ny, nz)
      integer, intent(in) :: nx, ny, nz
      real(wp), intent(in)    :: u_face(nx + 1, ny, nz), v_face(nx, ny + 1, nz)
      real(wp), intent(inout) :: du_visc(nx + 1, ny, nz), dv_visc(nx, ny + 1, nz)
      real(wp), intent(in)  :: dxT(nx, ny), dyT(nx, ny), idxT(nx, ny), idyT(nx, ny)
      real(wp), intent(in)  :: dy_dxBu(nx + 1, ny + 1), dx_dyBu(nx + 1, ny + 1)
      real(wp), intent(in)  :: idyCv(nx, ny + 1), idxCu(nx + 1, ny), wet_q(nx + 1, ny + 1)
      real(wp), intent(in)  :: dy_dxT(nx, ny), iareaCu(nx + 1, ny)
      real(wp), intent(in)  :: dx_dyT(nx, ny), iareaCv(nx, ny + 1)
      real(wp), intent(in)  :: c_smag, ah_bg, ah_max, ns
      integer :: i, j, k
      real(wp) :: D_T_W, D_T_E, D_T_S, D_T_N, D_T_face
      real(wp) :: D_S_S, D_S_N, D_S_W, D_S_E, D_S_face
      real(wp) :: strain_mag, A_raw, smag_scale, ah, lap_u, lap_v

      ! ---- u-face: fused ah_face_x (strain) + u-Laplacian -> du_visc ----------
      do concurrent(k=1:nz, j=1:ny, i=1:nx + 1) &
         local(D_T_W, D_T_E, D_T_face, D_S_S, D_S_N, D_S_face, strain_mag, A_raw, &
               smag_scale, ah, lap_u)
         if (i >= 2 .and. i <= nx .and. j >= 2 .and. j <= ny - 1) then
            smag_scale = (c_smag*sqrt(dxT(i, j)*dyT(i, j)))**2
            D_T_W = (u_face(i, j, k) - u_face(i - 1, j, k))*idxT(i - 1, j) &
                    - (v_face(i - 1, j + 1, k) - v_face(i - 1, j, k))*idyT(i - 1, j)
            D_T_E = (u_face(i + 1, j, k) - u_face(i, j, k))*idxT(i, j) &
                    - (v_face(i, j + 1, k) - v_face(i, j, k))*idyT(i, j)
            D_T_face = 0.5_wp*(D_T_W + D_T_E)
            D_S_S = ((1.0_wp - 2.0_wp*ns)*wet_q(i, j) + 2.0_wp*ns)* &
                    (dy_dxBu(i, j)*(v_face(i, j, k)*idyCv(i, j) - v_face(i - 1, j, k)*idyCv(i - 1, j)) + &
                     dx_dyBu(i, j)*(u_face(i, j, k)*idxCu(i, j) - u_face(i, j - 1, k)*idxCu(i, j - 1)))
            D_S_N = ((1.0_wp - 2.0_wp*ns)*wet_q(i, j + 1) + 2.0_wp*ns)* &
                    (dy_dxBu(i, j + 1)*(v_face(i, j + 1, k)*idyCv(i, j + 1) - v_face(i - 1, j + 1, k)*idyCv(i - 1, j + 1)) + &
                     dx_dyBu(i, j + 1)*(u_face(i, j + 1, k)*idxCu(i, j + 1) - u_face(i, j, k)*idxCu(i, j)))
            D_S_face = 0.5_wp*(D_S_S + D_S_N)
            strain_mag = sqrt(D_T_face*D_T_face + D_S_face*D_S_face)
            A_raw = smag_scale*strain_mag
            ah = min(ah_max, max(ah_bg, A_raw))
            lap_u = iareaCu(i, j)*( &
                    (dy_dxT(i, j)*(u_face(i + 1, j, k) - u_face(i, j, k)) - &
                     dy_dxT(i - 1, j)*(u_face(i, j, k) - u_face(i - 1, j, k))) + &
                    (dx_dyBu(i, j + 1)*(u_face(i, j + 1, k) - u_face(i, j, k)) - &
                     dx_dyBu(i, j)*(u_face(i, j, k) - u_face(i, j - 1, k))))
            du_visc(i, j, k) = ah*lap_u
         else
            du_visc(i, j, k) = 0.0_wp
         end if
      end do

      ! ---- v-face: fused ah_face_y (strain) + v-Laplacian -> dv_visc ----------
      do concurrent(k=1:nz, j=1:ny + 1, i=1:nx) &
         local(D_T_S, D_T_N, D_T_face, D_S_W, D_S_E, D_S_face, strain_mag, A_raw, &
               smag_scale, ah, lap_v)
         if (i >= 2 .and. i <= nx - 1 .and. j >= 2 .and. j <= ny) then
            smag_scale = (c_smag*sqrt(dxT(i, j)*dyT(i, j)))**2
            D_T_S = (u_face(i + 1, j - 1, k) - u_face(i, j - 1, k))*idxT(i, j - 1) &
                    - (v_face(i, j, k) - v_face(i, j - 1, k))*idyT(i, j - 1)
            D_T_N = (u_face(i + 1, j, k) - u_face(i, j, k))*idxT(i, j) &
                    - (v_face(i, j + 1, k) - v_face(i, j, k))*idyT(i, j)
            D_T_face = 0.5_wp*(D_T_S + D_T_N)
            D_S_W = ((1.0_wp - 2.0_wp*ns)*wet_q(i, j) + 2.0_wp*ns)* &
                    (dy_dxBu(i, j)*(v_face(i, j, k)*idyCv(i, j) - v_face(i - 1, j, k)*idyCv(i - 1, j)) + &
                     dx_dyBu(i, j)*(u_face(i, j, k)*idxCu(i, j) - u_face(i, j - 1, k)*idxCu(i, j - 1)))
            D_S_E = ((1.0_wp - 2.0_wp*ns)*wet_q(i + 1, j) + 2.0_wp*ns)* &
                    (dy_dxBu(i + 1, j)*(v_face(i + 1, j, k)*idyCv(i + 1, j) - v_face(i, j, k)*idyCv(i, j)) + &
                     dx_dyBu(i + 1, j)*(u_face(i + 1, j, k)*idxCu(i + 1, j) - u_face(i + 1, j - 1, k)*idxCu(i + 1, j - 1)))
            D_S_face = 0.5_wp*(D_S_W + D_S_E)
            strain_mag = sqrt(D_T_face*D_T_face + D_S_face*D_S_face)
            A_raw = smag_scale*strain_mag
            ah = min(ah_max, max(ah_bg, A_raw))
            lap_v = iareaCv(i, j)*( &
                    (dx_dyT(i, j)*(v_face(i, j + 1, k) - v_face(i, j, k)) - &
                     dx_dyT(i, j - 1)*(v_face(i, j, k) - v_face(i, j - 1, k))) + &
                    (dy_dxBu(i + 1, j)*(v_face(i + 1, j, k) - v_face(i, j, k)) - &
                     dy_dxBu(i, j)*(v_face(i, j, k) - v_face(i - 1, j, k))))
            dv_visc(i, j, k) = ah*lap_v
         else
            dv_visc(i, j, k) = 0.0_wp
         end if
      end do
   end subroutine hvisc_compute_fused

end module ocean_horizontal_viscosity
