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
!! DROPPED (each verified dead for every config in ~/analysis_gebco/*.nml,
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
#endif
   implicit none
   private

   ! Vanishing-layer guard for the `c = hTr / h` step. VERBATIM (ocean_remap.F90:23).
   real(wp), parameter :: H_FLOOR = 1.5e-4_wp

   public :: ale_remap_step

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
      do concurrent(j=1:ny, i=1:nx) local(k)
         vcoord%remap_total_h(i, j) = 0.0_wp
         do k = 1, nz
            vcoord%remap_total_h(i, j) = vcoord%remap_total_h(i, j) &
                                         + ms%h_layer(i, j, k)
         end do
      end do
#ifdef ASYNC
      !$acc end kernels
#endif

      ! --- L2. h_ref ---
#ifdef ASYNC
      !$acc kernels async(1)
#endif
      do concurrent(j=1:ny, i=1:nx)
         vcoord%remap_h_ref(i, j) = vcoord%remap_total_h(i, j) - bt_eta(i, j)
      end do
#ifdef ASYNC
      !$acc end kernels
#endif

      ! --- L3. Snapshot h_old on-device ---
#ifdef ASYNC
      !$acc kernels async(1)
#endif
      do concurrent(k=1:nz, j=1:ny, i=1:nx)
         vcoord%remap_h_old(i, j, k) = ms%h_layer(i, j, k)
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
      do concurrent(k=1:nz, j=1:ny, i=1:nx)
         ms%mass_budget_remap(i, j, k) = ms%mass_budget_remap(i, j, k) &
                                         + (vcoord%target_h(i, j, k) - vcoord%remap_h_old(i, j, k))
         ms%h_layer(i, j, k) = vcoord%target_h(i, j, k)
      end do
#ifdef ASYNC
      !$acc end kernels
#endif

      ! --- L10. Re-derive bt_eta ---
#ifdef ASYNC
      !$acc kernels async(1)
#endif
      do concurrent(j=1:ny, i=1:nx) local(k)
         bt_eta(i, j) = -bt_H_ref(i, j)
         do k = 1, nz
            bt_eta(i, j) = bt_eta(i, j) + ms%h_layer(i, j, k)
         end do
      end do
#ifdef ASYNC
      !$acc end kernels
#endif
#ifdef ASYNC
      !$acc wait(1)   ! single drain: the whole remap ran on async(1)
#endif
   end subroutine ale_remap_step

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
            do concurrent(k=1:nz, j=1:nyl, i=1:nxl)
#else
            do concurrent(k=1:nz, j=1:ny_total, i=1:nx_total)
#endif
               column_total = total_h(i, j) + eta(i, j)
               target_h(i, j, k) = column_total*dsig(k)
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
      do concurrent(j=1:ny, i=1:nx) &
         local(k, h_old_col, h_new_col, c_old_col, c_new_col, hTr_col, hTr_new)
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
