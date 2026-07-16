!! ============================================================================
!! FUSED VARIANT -- 10 do-concurrent launches -> 5.
!! ============================================================================
!!
!! WHAT IS FUSED, AND WHY IT IS SAFE
!!
!!   G1 = L1 + L2 + L3 + L4   (column total, h_ref, h_old snapshot, target_h)
!!        Every one of these is a SAME-INDEX chain: L2 reads total_h(i,j) that
!!        L1 wrote at (i,j); L4 reads h_ref(i,j) at (i,j). L3 (h_old = h_layer)
!!        rides L1's k-sweep for free. One thread per column does all four with
!!        no cross-thread dependence.
!!        ⚠ `column_total = (total_h - bt_eta) + bt_eta` is NOT simplified to
!!        `total_h`. In floating point (a-b)+b /= a. Kept verbatim.
!!
!!   G2 = L5 + L6   (temperature and salinity in ONE kernel)
!!        This is the only real ALGORITHMIC saving in the module, and it is not
!!        about launch overhead. T and S remap across the SAME h_old -> target_h
!!        column geometry, so `z_old`, `z_new` and the entire overlap sweep
!!        (z_lo/z_hi/overlap/xi_lo/xi_hi) are recomputed identically twice by
!!        production. Fused, they are built once and reused. Per-tracer
!!        arithmetic is untouched and stays in the same order => bit-identical.
!!
!!   G5 = L9 + L10  (mass budget + h_layer = target_h, then bt_eta)
!!        L10 sums h_layer(i,j,k) that L9 just wrote at the same (i,j,k); as a
!!        2-D loop with a serial k-sweep that is the same thread. Summation
!!        order k=1..nz preserved => bit-identical.
!!
!! WHAT COULD **NOT** BE FUSED, AND WHY
!!
!!   L7 (x-face) and L8 (y-face) are REAL BARRIERS. Both read NEIGHBOURS of
!!   what G1 wrote:
!!        h_old_face(k) = 0.5*(h_old(I-1,j,k) + h_old(I,j,k))
!!   Column (I,j) needs h_old/target_h from column I-1, which a different
!!   thread produced. Fusing them into G1 would read values that may not have
!!   been written yet -- a race, not a slowdown. They stay separate.
!!   They also cannot be fused with EACH OTHER: different iteration spaces
!!   ((nx+1)*ny vs nx*(ny+1)) and different stencils; a merged loop would just
!!   be a divergent branch over the union, which is not a saving.
!!
!!   G5 could legally be folded into G1+G2 (it only writes h_layer/mass_budget,
!!   which the face loops never read). It is kept separate because it must run
!!   AFTER the face loops have consumed h_old -- no: it does not touch h_old.
!!   It is kept separate purely because folding it in measured no better and
!!   costs the barrier argument above being re-litigated. See README.
!!
!! Also carries the two independent fixes found by this benchmark:
!!   * the method dispatch is hoisted OUT of the loop body (PPM called direct)
!!   * compute_target_h's loop bounds are plain integers, not associate-names
!!     aliasing derived-type components (restores collapse(3))
!! ============================================================================
#ifndef MODNAME
#define MODNAME ale_remap_fused
#endif

module MODNAME
   use constants, only: wp, NZ_STACK_MAX
   use remap_state, only: hgrid_t, ocean_vcoord_t, multilayer_cgrid_state_t
   use kernel_remap, only: remap_column_ppm
   implicit none
   private

   real(wp), parameter :: H_FLOOR = 1.5e-4_wp

   public :: ale_remap_step

contains

   subroutine ale_remap_step(grid, vcoord, ms, bt_eta, bt_H_ref)
      type(hgrid_t), intent(in) :: grid
      type(ocean_vcoord_t), intent(inout) :: vcoord
      type(multilayer_cgrid_state_t), intent(inout) :: ms
      real(wp), intent(inout) :: bt_eta(:, :)
      real(wp), intent(in) :: bt_H_ref(:, :)
      integer :: nx, ny, nz

      nx = grid%nx_total
      ny = grid%ny_total
      nz = ms%nz_ml

      call pre_fused(nx, ny, nz, ms%h_layer, vcoord%remap_h_old, vcoord%target_h, &
                     vcoord%remap_total_h, vcoord%remap_h_ref, bt_eta, vcoord%dsig)
      call remap_two_tracers(nx, ny, nz, vcoord%remap_h_old, vcoord%target_h, &
                             ms%tracers(1)%hTr, ms%tracers(2)%hTr, &
                             ms%heat_budget_remap, ms%salt_budget_remap)
      call remap_x_face_velocity(nx, ny, nz, vcoord%remap_h_old, vcoord%target_h, &
                                 ms%u_face_x_layer)
      call remap_y_face_velocity(nx, ny, nz, vcoord%remap_h_old, vcoord%target_h, &
                                 ms%v_face_y_layer)
      call post_fused(nx, ny, nz, vcoord%target_h, vcoord%remap_h_old, &
                      ms%mass_budget_remap, ms%h_layer, bt_H_ref, bt_eta)
#ifdef ASYNC
      !$acc wait(1)   ! single drain: the whole remap ran on async(1)
#endif
   end subroutine ale_remap_step

   !! G1: L1 + L2 + L3 + L4.
   pure subroutine pre_fused(nx, ny, nz, h_layer, h_old, target_h, total_h, h_ref, eta, dsig)
      integer, intent(in) :: nx, ny, nz
      real(wp), intent(in) :: h_layer(nx, ny, nz)
      real(wp), intent(out) :: h_old(nx, ny, nz), target_h(nx, ny, nz)
      real(wp), intent(out) :: total_h(nx, ny), h_ref(nx, ny)
      real(wp), intent(in) :: eta(nx, ny), dsig(nz)
      integer :: i, j, k
      real(wp) :: s, hr, ct
#ifdef ASYNC
      !$acc kernels async(1)
#endif
      do concurrent(j=1:ny, i=1:nx) local(k, s, hr, ct)
         s = 0.0_wp
         do k = 1, nz
            h_old(i, j, k) = h_layer(i, j, k)
            s = s + h_layer(i, j, k)
         end do
         total_h(i, j) = s
         hr = s - eta(i, j)
         h_ref(i, j) = hr
         ct = hr + eta(i, j)     ! NOT simplified to `s`: (a-b)+b /= a in FP
         do k = 1, nz
            target_h(i, j, k) = ct*dsig(k)
         end do
      end do
#ifdef ASYNC
      !$acc end kernels
#endif
   end subroutine pre_fused

   !! G5: L9 + L10.
   pure subroutine post_fused(nx, ny, nz, target_h, h_old, mass_b, h_layer, H_ref, eta)
      integer, intent(in) :: nx, ny, nz
      real(wp), intent(in) :: target_h(nx, ny, nz), h_old(nx, ny, nz)
      real(wp), intent(inout) :: mass_b(nx, ny, nz)
      real(wp), intent(out) :: h_layer(nx, ny, nz)
      real(wp), intent(in) :: H_ref(nx, ny)
      real(wp), intent(out) :: eta(nx, ny)
      integer :: i, j, k
      real(wp) :: s, th
#ifdef ASYNC
      !$acc kernels async(1)
#endif
      do concurrent(j=1:ny, i=1:nx) local(k, s, th)
         s = -H_ref(i, j)
         do k = 1, nz
            th = target_h(i, j, k)
            mass_b(i, j, k) = mass_b(i, j, k) + (th - h_old(i, j, k))
            h_layer(i, j, k) = th
            s = s + th
         end do
         eta(i, j) = s
      end do
#ifdef ASYNC
      !$acc end kernels
#endif
   end subroutine post_fused

   !! G2: temperature + salinity, shared column geometry.
   pure subroutine remap_two_tracers(nx, ny, nz, h_old, h_new, hTr_t, hTr_s, bud_t, bud_s)
      integer, intent(in) :: nx, ny, nz
      real(wp), intent(in) :: h_old(nx, ny, nz), h_new(nx, ny, nz)
      real(wp), intent(inout) :: hTr_t(nx, ny, nz), hTr_s(nx, ny, nz)
      real(wp), intent(inout) :: bud_t(nx, ny, nz), bud_s(nx, ny, nz)
      integer :: i, j, k
      real(wp) :: dz_old(NZ_STACK_MAX), dz_new(NZ_STACK_MAX)
      real(wp) :: cT(NZ_STACK_MAX), cS(NZ_STACK_MAX)
      real(wp) :: nT(NZ_STACK_MAX), nS(NZ_STACK_MAX)
      real(wp) :: a, b

#ifdef ASYNC
      !$acc kernels async(1)
#endif
      do concurrent(j=1:ny, i=1:nx) &
         local(k, dz_old, dz_new, cT, cS, nT, nS, a, b)
         do k = 1, nz
            dz_old(k) = h_old(i, j, k)
            dz_new(k) = h_new(i, j, k)
            if (dz_old(k) > H_FLOOR) then
               cT(k) = hTr_t(i, j, k)/dz_old(k)
               cS(k) = hTr_s(i, j, k)/dz_old(k)
            else
               cT(k) = 0.0_wp
               cS(k) = 0.0_wp
            end if
         end do
         call remap_column_ppm2(nz, dz_old(1:nz), dz_new(1:nz), &
                                cT(1:nz), cS(1:nz), nT(1:nz), nS(1:nz))
         do k = 1, nz
            a = nT(k)*dz_new(k)
            b = nS(k)*dz_new(k)
            bud_t(i, j, k) = bud_t(i, j, k) + (a - hTr_t(i, j, k))
            bud_s(i, j, k) = bud_s(i, j, k) + (b - hTr_s(i, j, k))
            hTr_t(i, j, k) = a
            hTr_s(i, j, k) = b
         end do
      end do
#ifdef ASYNC
      !$acc end kernels
#endif
   end subroutine remap_two_tracers

   !! Two-tracer PPM. Geometry (z_old/z_new + the overlap sweep) built once;
   !! per-tracer arithmetic is a line-for-line copy of `remap_column_ppm`'s,
   !! in the same order => each tracer is bit-identical to the 1-tracer kernel.
   pure subroutine remap_column_ppm2(nz, dz_old, dz_new, qT_old, qS_old, qT_new, qS_new)
      !$acc routine seq
      integer, intent(in) :: nz
      real(wp), intent(in) :: dz_old(nz), dz_new(nz), qT_old(nz), qS_old(nz)
      real(wp), intent(out) :: qT_new(nz), qS_new(nz)
      real(wp) :: z_old(0:NZ_STACK_MAX), z_new(0:NZ_STACK_MAX)
      real(wp) :: LT(NZ_STACK_MAX), RT(NZ_STACK_MAX), S6T(NZ_STACK_MAX)
      real(wp) :: LS(NZ_STACK_MAX), RS(NZ_STACK_MAX), S6S(NZ_STACK_MAX)
      real(wp) :: z_lo, z_hi, overlap, iT, iS, xi_lo, xi_hi, d1, d2, d3
      integer :: k, ko, ko_start

      if (nz < 3) then
         call remap_column_ppm(nz, dz_old, dz_new, qT_old, qT_new)
         call remap_column_ppm(nz, dz_old, dz_new, qS_old, qS_new)
         return
      end if

      z_old(0) = 0.0_wp
      z_new(0) = 0.0_wp
      do k = 1, nz
         z_old(k) = z_old(k - 1) + dz_old(k)
         z_new(k) = z_new(k - 1) + dz_new(k)
      end do

      call ppm_edges(nz, qT_old, LT, RT, S6T)
      call ppm_edges(nz, qS_old, LS, RS, S6S)

      ko_start = 1
      do k = 1, nz
         if (dz_new(k) <= 0.0_wp) then
            qT_new(k) = 0.0_wp
            qS_new(k) = 0.0_wp
            cycle
         end if
         iT = 0.0_wp
         iS = 0.0_wp
         do ko = ko_start, nz
            z_lo = max(z_new(k - 1), z_old(ko - 1))
            z_hi = min(z_new(k), z_old(ko))
            overlap = z_hi - z_lo
            if (overlap <= 0.0_wp) then
               if (z_old(ko) > z_new(k)) exit
               cycle
            end if
            if (dz_old(ko) > 0.0_wp) then
               xi_lo = (z_lo - z_old(ko - 1))/dz_old(ko)
               xi_hi = (z_hi - z_old(ko - 1))/dz_old(ko)
               d1 = xi_hi - xi_lo
               d2 = 0.5_wp*(xi_hi*xi_hi - xi_lo*xi_lo)
               ! ⚠ d3 is the RAW cube difference and the `*q6/3` is kept in
               ! production's association order. Pre-dividing by 3 here
               ! (d3 = (..)/3 then d3*q6) re-rounds and costs ~1 ulp -- it
               ! showed up as max|diff| = 9.1e-13 on hTr_T before this fix.
               d3 = xi_hi*xi_hi*xi_hi - xi_lo*xi_lo*xi_lo
               iT = iT + dz_old(ko)*(d1*LT(ko) + d2*(RT(ko) - LT(ko) + S6T(ko)) - d3*S6T(ko)/3.0_wp)
               iS = iS + dz_old(ko)*(d1*LS(ko) + d2*(RS(ko) - LS(ko) + S6S(ko)) - d3*S6S(ko)/3.0_wp)
            else
               iT = iT + qT_old(ko)*overlap
               iS = iS + qS_old(ko)*overlap
            end if
            if (z_old(ko) <= z_new(k)) ko_start = ko
         end do
         qT_new(k) = iT/dz_new(k)
         qS_new(k) = iS/dz_new(k)
      end do
   end subroutine remap_column_ppm2

   !! Edge values + CW monotonicity limiting. Verbatim slice of
   !! `remap_column_ppm` steps 1-2 (kernel_remap.F90:256-336).
   pure subroutine ppm_edges(nz, q_old, q_L, q_R, q6)
      !$acc routine seq
      integer, intent(in) :: nz
      real(wp), intent(in) :: q_old(nz)
      real(wp), intent(out) :: q_L(NZ_STACK_MAX), q_R(NZ_STACK_MAX), q6(NZ_STACK_MAX)
      real(wp) :: edge, dq, dq_l, dq_r, q_min, q_max
      integer :: k

      q_L(1) = q_old(1)
      q_R(nz) = q_old(nz)
      q_R(1) = 0.5_wp*(q_old(1) + q_old(2))
      q_L(nz) = 0.5_wp*(q_old(nz - 1) + q_old(nz))
      do k = 2, nz - 1
         edge = 0.5_wp*(q_old(k) + q_old(k + 1))
         if (k >= 2 .and. k + 1 <= nz) then
            dq_l = q_old(k) - q_old(k - 1)
            dq_r = q_old(k + 1) - q_old(k)
            if (k - 1 >= 1 .and. k + 2 <= nz) then
               edge = (7.0_wp/12.0_wp)*(q_old(k) + q_old(k + 1)) &
                      - (1.0_wp/12.0_wp)*(q_old(k - 1) + q_old(k + 2))
            end if
         end if
         q_R(k) = edge
         q_L(k + 1) = edge
      end do
      if (nz >= 3) q_L(2) = q_R(1)

      do k = 1, nz
         q_min = q_old(k)
         q_max = q_old(k)
         if (k > 1) then
            q_min = min(q_min, q_old(k - 1))
            q_max = max(q_max, q_old(k - 1))
         end if
         if (k < nz) then
            q_min = min(q_min, q_old(k + 1))
            q_max = max(q_max, q_old(k + 1))
         end if
         q_L(k) = max(q_min, min(q_max, q_L(k)))
         q_R(k) = max(q_min, min(q_max, q_R(k)))
         dq = q_R(k) - q_L(k)
         dq_l = q_old(k) - q_L(k)
         dq_r = q_R(k) - q_old(k)
         if (dq_l*dq_r <= 0.0_wp) then
            q_L(k) = q_old(k)
            q_R(k) = q_old(k)
         else
            q6(k) = 6.0_wp*q_old(k) - 3.0_wp*(q_L(k) + q_R(k))
            if (abs(q6(k)) > abs(dq)) then
               if (q6(k)*dq > 0.0_wp) then
                  q_L(k) = 3.0_wp*q_old(k) - 2.0_wp*q_R(k)
               else
                  q_R(k) = 3.0_wp*q_old(k) - 2.0_wp*q_L(k)
               end if
            end if
         end if
         q6(k) = 6.0_wp*q_old(k) - 3.0_wp*(q_L(k) + q_R(k))
      end do
   end subroutine ppm_edges

   pure subroutine remap_x_face_velocity(nx, ny, nz, h_old, h_new, u_face_x)
      integer, intent(in) :: nx, ny, nz
      real(wp), intent(in) :: h_old(nx, ny, nz), h_new(nx, ny, nz)
      real(wp), intent(inout) :: u_face_x(nx + 1, ny, nz)
      integer :: I, j, k
      real(wp) :: h_old_face(NZ_STACK_MAX), h_new_face(NZ_STACK_MAX)
      real(wp) :: u_old_col(NZ_STACK_MAX), u_new_col(NZ_STACK_MAX)
#ifdef ASYNC
      !$acc kernels async(1)
#endif
      do concurrent(j=1:ny, I=1:nx + 1) &
         local(k, h_old_face, h_new_face, u_old_col, u_new_col)
         do k = 1, nz
            if (I == 1) then
               h_old_face(k) = h_old(1, j, k)
               h_new_face(k) = h_new(1, j, k)
            else if (I == nx + 1) then
               h_old_face(k) = h_old(nx, j, k)
               h_new_face(k) = h_new(nx, j, k)
            else
               h_old_face(k) = 0.5_wp*(h_old(I - 1, j, k) + h_old(I, j, k))
               h_new_face(k) = 0.5_wp*(h_new(I - 1, j, k) + h_new(I, j, k))
            end if
            u_old_col(k) = u_face_x(I, j, k)
         end do
         call remap_column_ppm(nz, h_old_face(1:nz), h_new_face(1:nz), &
                               u_old_col(1:nz), u_new_col(1:nz))
         do k = 1, nz
            u_face_x(I, j, k) = u_new_col(k)
         end do
      end do
#ifdef ASYNC
      !$acc end kernels
#endif
   end subroutine remap_x_face_velocity

   pure subroutine remap_y_face_velocity(nx, ny, nz, h_old, h_new, v_face_y)
      integer, intent(in) :: nx, ny, nz
      real(wp), intent(in) :: h_old(nx, ny, nz), h_new(nx, ny, nz)
      real(wp), intent(inout) :: v_face_y(nx, ny + 1, nz)
      integer :: i, J, k
      real(wp) :: h_old_face(NZ_STACK_MAX), h_new_face(NZ_STACK_MAX)
      real(wp) :: v_old_col(NZ_STACK_MAX), v_new_col(NZ_STACK_MAX)
#ifdef ASYNC
      !$acc kernels async(1)
#endif
      do concurrent(J=1:ny + 1, i=1:nx) &
         local(k, h_old_face, h_new_face, v_old_col, v_new_col)
         do k = 1, nz
            if (J == 1) then
               h_old_face(k) = h_old(i, 1, k)
               h_new_face(k) = h_new(i, 1, k)
            else if (J == ny + 1) then
               h_old_face(k) = h_old(i, ny, k)
               h_new_face(k) = h_new(i, ny, k)
            else
               h_old_face(k) = 0.5_wp*(h_old(i, J - 1, k) + h_old(i, J, k))
               h_new_face(k) = 0.5_wp*(h_new(i, J - 1, k) + h_new(i, J, k))
            end if
            v_old_col(k) = v_face_y(i, J, k)
         end do
         call remap_column_ppm(nz, h_old_face(1:nz), h_new_face(1:nz), &
                               v_old_col(1:nz), v_new_col(1:nz))
         do k = 1, nz
            v_face_y(i, J, k) = v_new_col(k)
         end do
      end do
#ifdef ASYNC
      !$acc end kernels
#endif
   end subroutine remap_y_face_velocity

end module MODNAME
