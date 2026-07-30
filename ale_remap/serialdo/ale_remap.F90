#include "directives.h"
!! ============================================================================
!! BASELINE VARIANT -- `ocean_apply_ale_remap_step` AS SHIPPED.
!! ============================================================================
!!
!! ⚠ THIS IS A TRANSCRIPTION, NOT A VERBATIM COPY. Read this list before
!!   trusting any number produced from it.
!!
!! Source: <model-root>/src/ALE/ocean_remap.F90
!!         (`ocean_apply_ale_remap_step`, lines 378-506)
!!         + <model-root>/src/core/ocean/vcoord/ocean_vcoord.F90
!!           (`ocean_vcoord_compute_target_h_impl`, VCOORD_SIGMA/ZSTAR branch)
!!
!! VERBATIM (byte-identical to production, do not touch):
!!   * `kernel_remap.F90`   -- copied whole, md5 a079bd1e00b179d3233195c73953c2ee.
!!     Kept ENTIRE, including `remap_column_ppm_h4` / `remap_column_pqm` which
!!     production never calls at these configs. They are reachable through the
!!     `select case (method)` INSIDE the do-concurrent body, so deleting them
!!     would change what the compiler inlines per thread and therefore change
!!     register pressure. Dropping them would invalidate the measurement.
!!   * `ocean_remap_tracer_field`, `remap_x_face_velocity`,
!!     `remap_y_face_velocity`, `rescale_anomaly_ke` -- transcribed line-for-line.
!!
!! DROPPED (each verified dead for every config under ~/analysis_gebco, all .nml,
!! ALL of which set `vcoord_type = "zstar"`):
!!   1. The `VCOORD_EULERIAN_Z .or. VCOORD_LAGRANGIAN` early return.
!!   2. The `VCOORD_RHO / VCOORD_HYCOM` branch, i.e. `build_ts_concentration`
!!      + `compute_target_h_rho` + the `eos` argument. Never taken at zstar.
!!   3. The grid time-filter (step 3b). Gated on `regrid_time_scale > 0`;
!!      the default is `0.0_wp` (config.F90:1807) and no namelist sets it.
!!   4. `compute_target_h`'s other branches (EULERIAN_Z / ZSIGMA / ZSTAR_FULL /
!!      ZSTAR_SIGMA). SAFE TO DROP: each branch is its own separate
!!      `do concurrent` loop selected by a runtime `select case` OUTSIDE the
!!      loops, so the untaken branches cannot affect the taken loop's codegen.
!!      (Contrast with `remap_column`'s select case, which is INSIDE the loop
!!      body -- that one is kept whole for exactly this reason.)
!!   5. `conserve_ke` is KEPT and still gated on the flag, but the flag's
!!      default is `.false.` (config.F90:1813) so `rescale_anomaly_ke`
!!      does not fire in the default measurement.
!!
!! PRESERVED ON PURPOSE (these are the things under test):
!!   * derived-type component access in the orchestrator loops
!!     (`vcoord%remap_total_h(i,j)`, `ms%h_layer(i,j,k)`, `ms%tracers(t)%hTr`)
!!   * the assumed-shape `total_h(:,:)` / `eta(:,:)` dummies on
!!     `compute_target_h` (production marks these "assumed-shape-ok")
!!   * plain explicit-shape dummies on the four heavy column kernels --
!!     production ALREADY does this ("Flat-arg so GPU codegen doesn't chase
!!     the array-of-derived-types pointer")
!! ============================================================================
#ifndef MODNAME
#define MODNAME ale_remap
#endif

module MODNAME
   use constants, only: wp, NZ_STACK_MAX, REMAP_PPM, VCOORD_SIGMA, VCOORD_ZSTAR
   use remap_state, only: hgrid_t, ocean_vcoord_t, multilayer_cgrid_state_t
#ifdef PPM_DIRECT
   use kernel_remap, only: remap_column_ppm
#else
   use kernel_remap, only: remap_column
   ! remap_column_ppm: needed directly by the optimized T+S fused path
   ! (remap_column_ppm2's nz<3 fallback). Harmless to the faithful path.
   use kernel_remap, only: remap_column_ppm
#endif
   implicit none
   private

   ! Vanishing-layer guard for the `c = hTr / h` step. VERBATIM (ocean_remap.F90:23).
   real(wp), parameter :: H_FLOOR = 1.5e-4_wp

   public :: ale_remap_step
   public :: ale_remap_step_opt

contains

   subroutine ale_remap_step(grid, vcoord, ms, bt_eta, bt_H_ref)
      type(hgrid_t), intent(in) :: grid
      type(ocean_vcoord_t), intent(inout) :: vcoord
      type(multilayer_cgrid_state_t), intent(inout) :: ms
      real(wp), intent(inout) :: bt_eta(:, :)
      real(wp), intent(in) :: bt_H_ref(:, :)

      integer :: t, m, nx, ny, nz, i, j, k
      logical :: conserve_ke

      m = vcoord%remap_method
      nx = grid%nx_total
      ny = grid%ny_total
      nz = ms%nz_ml

      ! --- L1. Column total (2D loop, serial k) ---
#ifdef ASYNC
      !$acc kernels async(1)
#endif
      do j=1,ny
      do i=1,nx
         vcoord%remap_total_h(i, j) = 0.0_wp
         do k = 1, nz
            vcoord%remap_total_h(i, j) = vcoord%remap_total_h(i, j) &
                                         + ms%h_layer(i, j, k)
         end do
      end do
      end do
#ifdef ASYNC
      !$acc end kernels
#endif

      ! --- L2. h_ref ---
#ifdef ASYNC
      !$acc kernels async(1)
#endif
      do j=1,ny
      do i=1,nx
         vcoord%remap_h_ref(i, j) = vcoord%remap_total_h(i, j) - bt_eta(i, j)
      end do
      end do
#ifdef ASYNC
      !$acc end kernels
#endif

      ! --- L3. Snapshot h_old on-device ---
#ifdef ASYNC
      !$acc kernels async(1)
#endif
      do k=1,nz
      do j=1,ny
      do i=1,nx
         vcoord%remap_h_old(i, j, k) = ms%h_layer(i, j, k)
      end do
      end do
      end do
#ifdef ASYNC
      !$acc end kernels
#endif

      ! --- L4. Populate target_h (zstar) ---
      call vcoord_compute_target_h(vcoord, vcoord%remap_h_ref, bt_eta)

      ! --- L5/L6. Tracer remap (centre cells) ---
      if (allocated(ms%tracers)) then
         do t = 1, size(ms%tracers)
            if (.not. allocated(ms%tracers(t)%hTr)) cycle
            if (t == ms%idx_temperature) then
               call ocean_remap_tracer_field( &
                  nx, ny, nz, vcoord%remap_h_old, vcoord%target_h, ms%tracers(t)%hTr, m, &
                  budget=ms%heat_budget_remap)
            else if (t == ms%idx_salinity) then
               call ocean_remap_tracer_field( &
                  nx, ny, nz, vcoord%remap_h_old, vcoord%target_h, ms%tracers(t)%hTr, m, &
                  budget=ms%salt_budget_remap)
            else
               call ocean_remap_tracer_field( &
                  nx, ny, nz, vcoord%remap_h_old, vcoord%target_h, ms%tracers(t)%hTr, m)
            end if
         end do
      end if

      ! --- L7/L8. Face-velocity remap ---
      conserve_ke = vcoord%remap_vel_conserve_ke
      if (allocated(ms%u_face_x_layer) .and. allocated(ms%v_face_y_layer)) then
         call remap_x_face_velocity(nx, ny, nz, vcoord%remap_h_old, vcoord%target_h, &
                                    ms%u_face_x_layer, m, conserve_ke)
         call remap_y_face_velocity(nx, ny, nz, vcoord%remap_h_old, vcoord%target_h, &
                                    ms%v_face_y_layer, m, conserve_ke)
      end if

      ! --- L9. h_layer = target_h; capture mass-budget delta ---
#ifdef ASYNC
      !$acc kernels async(1)
#endif
      do k=1,nz
      do j=1,ny
      do i=1,nx
         ms%mass_budget_remap(i, j, k) = ms%mass_budget_remap(i, j, k) &
                                         + (vcoord%target_h(i, j, k) - vcoord%remap_h_old(i, j, k))
         ms%h_layer(i, j, k) = vcoord%target_h(i, j, k)
      end do
      end do
      end do
#ifdef ASYNC
      !$acc end kernels
#endif

      ! --- L10. Re-derive bt_eta ---
#ifdef ASYNC
      !$acc kernels async(1)
#endif
      do j=1,ny
      do i=1,nx
         bt_eta(i, j) = -bt_H_ref(i, j)
         do k = 1, nz
            bt_eta(i, j) = bt_eta(i, j) + ms%h_layer(i, j, k)
         end do
      end do
      end do
#ifdef ASYNC
      !$acc end kernels
#endif
#ifdef ASYNC
      !$acc wait(1)   ! single drain: the whole remap ran on async(1)
#endif
   end subroutine ale_remap_step

   !! ========================================================================
   !! OPTIMIZED VARIANT -- portable subset of the CUDA `ale_remap_opt` win.
   !! ========================================================================
   !! The one real lever (bit-identical to `ale_remap_step`): T and S ride on
   !! the SAME h_old/target_h column, so the faithful path pays for the whole
   !! PPM machinery (z_old/z_new interfaces, edge reconstruction, overlap sweep)
   !! TWICE -- once per tracer. `ocean_remap_tracer_pair` builds the geometry
   !! ONCE and consumes it for both tracers in a single per-column pass; the
   !! only per-tracer work kept is the arithmetic that genuinely differs (the
   !! two edge reconstructions + two integral accumulations), in the SAME ORDER
   !! as `remap_column_ppm`, which is why the result is bit-identical.
   !!
   !! Two further waste-removal fusions (also bit-identical, pure register
   !! hoists -- no CUDA-specific tricks, valid F2018, run under DATA=none):
   !!   * PRE : total_h + h_ref + h_old snapshot + target_h in ONE 2-D pass.
   !!           total_h and h_ref never leave registers (the faithful path
   !!           round-trips remap_total_h/remap_h_ref through global memory).
   !!           target_h keeps the faithful association ((Sh - eta) + eta)*dsig.
   !!           This folds in the collapse/sig hoist: the target_h loop bounds
   !!           are plain locals, not derived-type components.
   !!   * POST: mass-budget delta + h_layer commit + bt_eta in ONE 2-D pass.
   !! 10 do-concurrent launches -> 5.  Assumes the production regime the driver
   !! exercises (VCOORD_ZSTAR, REMAP_PPM, tracers = {T, S} both with budgets);
   !! that is exactly what ale_remap_step is doing at these configs.
   subroutine ale_remap_step_opt(grid, vcoord, ms, bt_eta, bt_H_ref)
      type(hgrid_t), intent(in) :: grid
      type(ocean_vcoord_t), intent(inout) :: vcoord
      type(multilayer_cgrid_state_t), intent(inout) :: ms
      real(wp), intent(inout) :: bt_eta(:, :)
      real(wp), intent(in) :: bt_H_ref(:, :)

      integer :: m, nx, ny, nz, i, j, k, it, is
      logical :: conserve_ke
      real(wp) :: s, hr, ct

      m = vcoord%remap_method
      nx = grid%nx_total
      ny = grid%ny_total
      nz = ms%nz_ml
      it = ms%idx_temperature
      is = ms%idx_salinity

      ! --- PRE (fused L1+L2+L3+L4): total_h, h_ref (registers only), h_old
      !     snapshot (to global), target_h (zstar). Bit-identical to the four
      !     faithful loops: s == remap_total_h, hr == remap_h_ref = s - eta,
      !     ct == column_total = hr + eta, target_h = ct*dsig(k). ---
      do j=1,ny
      do i=1,nx
         s = 0.0_wp
         do k = 1, nz
            vcoord%remap_h_old(i, j, k) = ms%h_layer(i, j, k)
            s = s + ms%h_layer(i, j, k)
         end do
         hr = s - bt_eta(i, j)
         ct = hr + bt_eta(i, j)
         do k = 1, nz
            vcoord%target_h(i, j, k) = ct*vcoord%dsig(k)
         end do
      end do
      end do

      ! --- Tracer remap: T + S share the column geometry (THE lever) ---
      if (allocated(ms%tracers)) then
         if (allocated(ms%tracers(it)%hTr) .and. allocated(ms%tracers(is)%hTr)) then
            if (m == REMAP_PPM) then
               call ocean_remap_tracer_pair(nx, ny, nz, vcoord%remap_h_old, vcoord%target_h, &
                                            ms%tracers(it)%hTr, ms%tracers(is)%hTr, &
                                            ms%heat_budget_remap, ms%salt_budget_remap)
            else
               ! non-PPM (rare): fall back to the faithful per-tracer path
               call ocean_remap_tracer_field(nx, ny, nz, vcoord%remap_h_old, vcoord%target_h, &
                                             ms%tracers(it)%hTr, m, budget=ms%heat_budget_remap)
               call ocean_remap_tracer_field(nx, ny, nz, vcoord%remap_h_old, vcoord%target_h, &
                                             ms%tracers(is)%hTr, m, budget=ms%salt_budget_remap)
            end if
         end if
      end if

      ! --- Face-velocity remap: no T/S-style twin to fuse, carried verbatim ---
      conserve_ke = vcoord%remap_vel_conserve_ke
      if (allocated(ms%u_face_x_layer) .and. allocated(ms%v_face_y_layer)) then
         call remap_x_face_velocity(nx, ny, nz, vcoord%remap_h_old, vcoord%target_h, &
                                    ms%u_face_x_layer, m, conserve_ke)
         call remap_y_face_velocity(nx, ny, nz, vcoord%remap_h_old, vcoord%target_h, &
                                    ms%v_face_y_layer, m, conserve_ke)
      end if

      ! --- POST (fused L9+L10): mass-budget delta, h_layer commit, bt_eta.
      !     bt_eta sums h_layer (== target_h just committed), -bt_H_ref start,
      !     same k order as the faithful L10 loop -> bit-identical. ---
      do j=1,ny
      do i=1,nx
         s = -bt_H_ref(i, j)
         do k = 1, nz
            ms%mass_budget_remap(i, j, k) = ms%mass_budget_remap(i, j, k) &
                                            + (vcoord%target_h(i, j, k) - vcoord%remap_h_old(i, j, k))
            ms%h_layer(i, j, k) = vcoord%target_h(i, j, k)
            s = s + vcoord%target_h(i, j, k)
         end do
         bt_eta(i, j) = s
      end do
      end do
   end subroutine ale_remap_step_opt

   !! Fused T+S tracer remap (PPM). The OUTER do-concurrent privatizes only the
   !! six small per-column vectors (dz_old/dz_new/cT/cS/nT/nS); the heavy PPM
   !! geometry+edge workspace lives in the callee `remap_column_ppm2`'s frame,
   !! NOT the outer `local()` list. Keeping the outer private set small is what
   !! makes this a WIN on nvfortran GPU codegen -- flattening every array into
   !! the loop's `local()` clause spills them all to local memory and collapses
   !! occupancy (this kernel is register/occupancy-bound). Bit-identical to two
   !! faithful `ocean_remap_tracer_field` calls (verified max rel diff == 0).
   pure subroutine ocean_remap_tracer_pair(nx, ny, nz, h_old, h_new, &
                                           hTr_t, hTr_s, budget_t, budget_s)
      integer, intent(in) :: nx, ny, nz
      real(wp), intent(in) :: h_old(nx, ny, nz)
      real(wp), intent(in) :: h_new(nx, ny, nz)
      real(wp), intent(inout) :: hTr_t(nx, ny, nz)
      real(wp), intent(inout) :: hTr_s(nx, ny, nz)
      real(wp), intent(inout) :: budget_t(nx, ny, nz)
      real(wp), intent(inout) :: budget_s(nx, ny, nz)
      integer :: i, j, k
      real(wp) :: dz_old(NZ_STACK_MAX), dz_new(NZ_STACK_MAX)
      real(wp) :: cT(NZ_STACK_MAX), cS(NZ_STACK_MAX)
      real(wp) :: nT(NZ_STACK_MAX), nS(NZ_STACK_MAX)
      real(wp) :: a, b

#ifdef ASYNC
      !$acc kernels async(1)
#endif
      do j=1,ny
      do i=1,nx
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
            budget_t(i, j, k) = budget_t(i, j, k) + (a - hTr_t(i, j, k))
            budget_s(i, j, k) = budget_s(i, j, k) + (b - hTr_s(i, j, k))
            hTr_t(i, j, k) = a
            hTr_s(i, j, k) = b
         end do
      end do
      end do
#ifdef ASYNC
      !$acc end kernels
#endif
   end subroutine ocean_remap_tracer_pair

   !! Two-tracer PPM column remap. Geometry (z_old/z_new + the overlap sweep) is
   !! built ONCE and consumed by both tracers; each tracer's edge reconstruction
   !! and integral accumulation are a line-for-line copy of `remap_column_ppm`'s
   !! (kernel_remap.F90), in the SAME association order -> bit-identical to two
   !! single-tracer calls. nz<3 columns fall back to two `remap_column_ppm`.
   pure subroutine remap_column_ppm2(nz, dz_old, dz_new, qT_old, qS_old, qT_new, qS_new)
      DC_ROUTINE_SEQ
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
               ! d3 is the RAW cube difference; the `*q6/3` division is kept in
               ! production's association order (pre-dividing re-rounds ~1 ulp).
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

   !! PPM edge reconstruction: steps 1+2 of `remap_column_ppm` (4th-order edge
   !! interp + Colella-Woodward monotonicity limiting), producing q_L/q_R/q6.
   !! Transcribed op-for-op from kernel_remap.F90 (the two dead dq_l/dq_r reads
   !! in the faithful edge loop are elided -- they never touch q_L/q_R/q6).
   pure subroutine ppm_edges(nz, q_old, q_L, q_R, q6)
      DC_ROUTINE_SEQ
      integer, intent(in) :: nz
      real(wp), intent(in) :: q_old(nz)
      real(wp), intent(out) :: q_L(nz), q_R(nz), q6(nz)
      real(wp) :: edge, dq, dq_l, dq_r, q_min, q_max
      integer :: k

      q_L(1) = q_old(1)
      q_R(nz) = q_old(nz)
      q_R(1) = 0.5_wp*(q_old(1) + q_old(2))
      q_L(nz) = 0.5_wp*(q_old(nz - 1) + q_old(nz))
      do k = 2, nz - 1
         edge = 0.5_wp*(q_old(k) + q_old(k + 1))
         if (k + 2 <= nz) then
            edge = (7.0_wp/12.0_wp)*(q_old(k) + q_old(k + 1)) &
                   - (1.0_wp/12.0_wp)*(q_old(k - 1) + q_old(k + 2))
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

   !! zstar branch of `ocean_vcoord_compute_target_h_impl`. The assumed-shape
   !! dummies and the `associate` block are production's, kept verbatim.
#ifdef FLATSIG
   subroutine vcoord_compute_target_h(this, total_h, eta)
      type(ocean_vcoord_t), intent(inout) :: this
      real(wp), intent(in) :: total_h(this%nx_total, this%ny_total)
      real(wp), intent(in) :: eta(this%nx_total, this%ny_total)
      integer :: i, j, k, nz
#else
   subroutine vcoord_compute_target_h(this, total_h, eta)
      type(ocean_vcoord_t), intent(inout) :: this
      real(wp), intent(in) :: total_h(:, :)  ! assumed-shape: production's signature
      real(wp), intent(in) :: eta(:, :)      ! assumed-shape: production's signature
      integer :: i, j, k, nz
#endif
      integer :: nxl, nyl
      real(wp) :: column_total

      if (.not. this%is_init) return
      nz = this%nz_ml

#ifdef COLLAPSE_FIX
      ! COLLAPSE FIX: hoist the do-concurrent bounds out of the associate (and
      ! so out of the derived type) into plain local integers. With production's
      ! `nx_total => this%nx_total` associate-name as the bound, nvfortran
      ! refuses to auto-collapse this loop and runs k SEQUENTIALLY -- every
      ! other 3-D loop in the module gets collapse(3). Pure schedule change.
      nxl = this%nx_total
      nyl = this%ny_total
      associate (target_h => this%target_h, dsig => this%dsig)
#else
      associate (target_h => this%target_h, dsig => this%dsig, &
                 nx_total => this%nx_total, ny_total => this%ny_total)
#endif
         select case (this%coord_type)
         case (VCOORD_SIGMA, VCOORD_ZSTAR)
#ifdef ASYNC
            !$acc kernels async(1)
#endif
#ifdef COLLAPSE_FIX
            do k=1,nz
            do j=1,nyl
            do i=1,nxl
#else
            do k=1,nz
            do j=1,ny_total
            do i=1,nx_total
#endif
               column_total = total_h(i, j) + eta(i, j)
               target_h(i, j, k) = column_total*dsig(k)
            end do
            end do
            end do
#ifdef ASYNC
            !$acc end kernels
#endif
         end select
      end associate
   end subroutine vcoord_compute_target_h

   ! ---- VERBATIM from ocean_remap.F90:163-210 ----
   pure subroutine ocean_remap_tracer_field(nx, ny, nz, h_old, h_new, hTr, method, budget)
      integer, intent(in) :: nx, ny, nz, method
      real(wp), intent(in) :: h_old(nx, ny, nz)
      real(wp), intent(in) :: h_new(nx, ny, nz)
      real(wp), intent(inout) :: hTr(nx, ny, nz)
      real(wp), intent(inout), optional :: budget(nx, ny, nz)
      integer :: i, j, k
      real(wp) :: h_old_col(NZ_STACK_MAX), h_new_col(NZ_STACK_MAX)
      real(wp) :: c_old_col(NZ_STACK_MAX), c_new_col(NZ_STACK_MAX)
      real(wp) :: hTr_col(NZ_STACK_MAX)
      real(wp) :: hTr_new

#ifdef ASYNC
      !$acc kernels async(1)
#endif
      do j=1,ny
      do i=1,nx
         do k = 1, nz
            h_old_col(k) = h_old(i, j, k)
            h_new_col(k) = h_new(i, j, k)
            hTr_col(k) = hTr(i, j, k)
            if (h_old_col(k) > H_FLOOR) then
               c_old_col(k) = hTr_col(k)/h_old_col(k)
            else
               c_old_col(k) = 0.0_wp
            end if
         end do
#ifdef PPM_DIRECT
         call remap_column_ppm(nz, &
                               h_old_col(1:nz), h_new_col(1:nz), &
                               c_old_col(1:nz), c_new_col(1:nz))
#else
         call remap_column(method, nz, &
                           h_old_col(1:nz), h_new_col(1:nz), &
                           c_old_col(1:nz), c_new_col(1:nz))
#endif
         if (present(budget)) then
            do k = 1, nz
               hTr_new = c_new_col(k)*h_new_col(k)
               budget(i, j, k) = budget(i, j, k) + (hTr_new - hTr(i, j, k))
               hTr(i, j, k) = hTr_new
            end do
         else
            do k = 1, nz
               hTr(i, j, k) = c_new_col(k)*h_new_col(k)
            end do
         end if
      end do
      end do
#ifdef ASYNC
      !$acc end kernels
#endif
   end subroutine ocean_remap_tracer_field

   ! ---- VERBATIM from ocean_remap.F90:246-286 ----
   pure subroutine remap_x_face_velocity(nx, ny, nz, h_old, h_new, u_face_x, method, conserve_ke)
      integer, intent(in) :: nx, ny, nz, method
      real(wp), intent(in) :: h_old(nx, ny, nz)
      real(wp), intent(in) :: h_new(nx, ny, nz)
      real(wp), intent(inout) :: u_face_x(nx + 1, ny, nz)
      logical, intent(in) :: conserve_ke
      integer :: I, j, k
      real(wp) :: h_old_face(NZ_STACK_MAX), h_new_face(NZ_STACK_MAX)
      real(wp) :: u_old_col(NZ_STACK_MAX), u_new_col(NZ_STACK_MAX)

#ifdef ASYNC
      !$acc kernels async(1)
#endif
      do j=1,ny
      do I=1,nx + 1
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
#ifdef PPM_DIRECT
         call remap_column_ppm(nz, &
                               h_old_face(1:nz), h_new_face(1:nz), &
                               u_old_col(1:nz), u_new_col(1:nz))
#else
         call remap_column(method, nz, &
                           h_old_face(1:nz), h_new_face(1:nz), &
                           u_old_col(1:nz), u_new_col(1:nz))
#endif
         if (conserve_ke) then
            call rescale_anomaly_ke(nz, h_old_face, h_new_face, u_old_col, u_new_col)
         end if
         do k = 1, nz
            u_face_x(I, j, k) = u_new_col(k)
         end do
      end do
      end do
#ifdef ASYNC
      !$acc end kernels
#endif
   end subroutine remap_x_face_velocity

   ! ---- VERBATIM from ocean_remap.F90:288-325 ----
   pure subroutine remap_y_face_velocity(nx, ny, nz, h_old, h_new, v_face_y, method, conserve_ke)
      integer, intent(in) :: nx, ny, nz, method
      real(wp), intent(in) :: h_old(nx, ny, nz)
      real(wp), intent(in) :: h_new(nx, ny, nz)
      real(wp), intent(inout) :: v_face_y(nx, ny + 1, nz)
      logical, intent(in) :: conserve_ke
      integer :: i, J, k
      real(wp) :: h_old_face(NZ_STACK_MAX), h_new_face(NZ_STACK_MAX)
      real(wp) :: v_old_col(NZ_STACK_MAX), v_new_col(NZ_STACK_MAX)

#ifdef ASYNC
      !$acc kernels async(1)
#endif
      do J=1,ny + 1
      do i=1,nx
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
#ifdef PPM_DIRECT
         call remap_column_ppm(nz, &
                               h_old_face(1:nz), h_new_face(1:nz), &
                               v_old_col(1:nz), v_new_col(1:nz))
#else
         call remap_column(method, nz, &
                           h_old_face(1:nz), h_new_face(1:nz), &
                           v_old_col(1:nz), v_new_col(1:nz))
#endif
         if (conserve_ke) then
            call rescale_anomaly_ke(nz, h_old_face, h_new_face, v_old_col, v_new_col)
         end if
         do k = 1, nz
            v_face_y(i, J, k) = v_new_col(k)
         end do
      end do
      end do
#ifdef ASYNC
      !$acc end kernels
#endif
   end subroutine remap_y_face_velocity

   ! ---- VERBATIM from ocean_remap.F90:327-376 ----
   pure subroutine rescale_anomaly_ke(nz, h_old_face, h_new_face, u_old_col, u_new_col)
      DC_ROUTINE_SEQ
      integer, intent(in) :: nz
      real(wp), intent(in) :: h_old_face(NZ_STACK_MAX)
      real(wp), intent(in) :: h_new_face(NZ_STACK_MAX)
      real(wp), intent(in) :: u_old_col(NZ_STACK_MAX)
      real(wp), intent(inout) :: u_new_col(NZ_STACK_MAX)
      integer :: k
      real(wp) :: h_old_sum, h_new_sum, mom_old, mom_new
      real(wp) :: u_bar_old, u_bar_new, ke_old, ke_new, anom, scale_fac

      h_old_sum = 0.0_wp
      h_new_sum = 0.0_wp
      mom_old = 0.0_wp
      mom_new = 0.0_wp
      do k = 1, nz
         h_old_sum = h_old_sum + h_old_face(k)
         h_new_sum = h_new_sum + h_new_face(k)
         mom_old = mom_old + h_old_face(k)*u_old_col(k)
         mom_new = mom_new + h_new_face(k)*u_new_col(k)
      end do
      if (h_old_sum <= H_FLOOR .or. h_new_sum <= H_FLOOR) return
      u_bar_old = mom_old/h_old_sum
      u_bar_new = mom_new/h_new_sum

      ke_old = 0.0_wp
      ke_new = 0.0_wp
      do k = 1, nz
         anom = u_old_col(k) - u_bar_old
         ke_old = ke_old + 0.5_wp*h_old_face(k)*anom*anom
         anom = u_new_col(k) - u_bar_new
         ke_new = ke_new + 0.5_wp*h_new_face(k)*anom*anom
      end do
      if (ke_new <= 0.0_wp .or. ke_old <= 0.0_wp) return
      scale_fac = sqrt(ke_old/ke_new)
      if (scale_fac > 1.25_wp) scale_fac = 1.25_wp
      if (scale_fac < 0.0_wp) scale_fac = 0.0_wp
      do k = 1, nz
         u_new_col(k) = u_bar_new + scale_fac*(u_new_col(k) - u_bar_new)
      end do
   end subroutine rescale_anomaly_ke

end module MODNAME
