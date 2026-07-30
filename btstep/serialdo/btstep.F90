#include "directives.h"
!! CLOSED-BASIN barotropic substep — the live path of the ocean model's
!! `barotropic_substep_nonlinear`, transcribed from
!! <model>/src/core/ocean/kernels/structured/barotropic/barotropic_substep.F90:259-1404.
!!
!! WHY THIS IS A TRANSCRIPTION AND NOT A VERBATIM EXTRACT (unlike the other
!! benchmarks in this repo): the production routine is 1146 lines with 72
!! branches, and ~207 of those lines plus most of the branches are DEAD in the
!! gabight configs (wetdry off, use_bt_cont_type off, use_upstream_h_face off).
!! Carrying them would benchmark code that never runs. This keeps the LIVE
!! closed-basin path — arithmetic, operation order and loop structure preserved
!! line-for-line — and drops:
!!   * the `wd_on` wet/dry branch          (no &ocean_wetdry_nml in the configs)
!!   * BT_cont / upstream-h_face branches  (knobs off)
!!   * Flather OBC + periodic ghost-wraps  (CLOSED BASIN, by request)
!!   * `bebt > 0` prev-save                (kept: bebt = 0.2 in the configs)
!! ⚠ So this is NOT bit-comparable to a production run. It is a faithful model
!! of the substep's SHAPE — many small 2-D kernels in a substep loop — which is
!! what decides the launch-bound behaviour under test.
!!
!! PRODUCTION FUSION IS PRESERVED. The real routine already fuses aggressively
!! and the comments say so ("KE (centre) and interior zeta ... ride along Pass
!! 1's sweep -- bit-identical, one fewer launch each"; "eta-swap + zeta wall
!! closure -- one barrier sweep"). Undoing that would invent headroom that
!! production has already taken. Live kernels per substep here: 7.
!!
!! `!$acc kernels async(1)` wraps the substep loop with a single `wait(1)`
!! drain, exactly as production does (:1369, "single drain: the whole substep
!! ran on async(1)"). That idiom is worth ~9x on this path -- at the ~11 us/loop
!! a plain blocking `do concurrent` pays, these launches would cost ~29 ms/step
!! against the measured 3.35 ms.
module btstep
   use constants, only: wp
   use bt_state, only: hgrid_t, ocean_metrics_t, coriolis_t, bt_work_t
   implicit none
   private
   public :: btstep_nonlinear_closed
   public :: btstep_nonlinear_closed_fused

contains

   subroutine btstep_nonlinear_closed(grid, metrics, bt_work, cor, force_u, force_v, &
                                      n_steps, dt_inner)
      type(hgrid_t), intent(in) :: grid
      type(ocean_metrics_t), intent(in) :: metrics
      type(bt_work_t), intent(inout) :: bt_work
      type(coriolis_t), intent(in) :: cor
      ! EXPLICIT-SHAPE, exactly as production declares them
      ! (barotropic_substep.F90:296-297). Assumed-shape `force_u(:,:)` here
      ! made -Minfo emit `implicit copyin(force_v(:nx,2:ny))` and the benchmark
      ! timed PCIe -- see RESUME §2's assumed-shape warning.
      real(wp), intent(in) :: force_u(grid%nx_total + 1, grid%ny_total)
      real(wp), intent(in) :: force_v(grid%nx_total, grid%ny_total + 1)
      integer, intent(in) :: n_steps
      real(wp), intent(in) :: dt_inner

      integer :: i, j, n, nx, ny
      real(wp) :: G, bebt
      real(wp) :: h_face_E, h_face_W, h_face_N, h_face_S
      real(wp) :: ubt_R, ubt_L, vbt_N, vbt_S
      real(wp) :: flux_x_R, flux_x_L, flux_y_N, flux_y_S, div_h_u
      real(wp) :: zeta_at_u, f_at_u, v_at_u, ke_grad_x, d_eta
      real(wp) :: zeta_at_v, f_at_v, u_at_v, ke_grad_y

      nx = grid%nx_total
      ny = grid%ny_total
      G = bt_work%g_bt
      bebt = bt_work%bebt

      !$acc kernels async(1)
      do j=1,ny
      do i=1,nx
         bt_work%eta_sum(i, j) = 0.0_wp
      end do
      end do
      do j=1,ny
      do i=1,nx + 1
         bt_work%ubt_sum(i, j) = 0.0_wp
         bt_work%uhbt_sum(i, j) = 0.0_wp
      end do
      end do
      do j=1,ny + 1
      do i=1,nx
         bt_work%vbt_sum(i, j) = 0.0_wp
         bt_work%vhbt_sum(i, j) = 0.0_wp
      end do
      end do
      !$acc end kernels

      do n = 1, n_steps
         ! Drained by the !$acc wait(1) after the n_steps loop.
         !$acc kernels async(1)

         ! ---- prev-save for the BEBT projection (bebt > 0) ----
         if (bebt > 0.0_wp) then
            do j=1,ny
            do i=1,nx + 1
               bt_work%bt_ubt_prev(i, j) = bt_work%bt_ubt(i, j)
            end do
            end do
            do j=1,ny + 1
            do i=1,nx
               bt_work%bt_vbt_prev(i, j) = bt_work%bt_vbt(i, j)
            end do
            end do
         end if

         ! ---- Pass 1: eta update with (H_ref + eta) face thickness ----
         ! KE (centre) and interior zeta (NE corner) ride along this sweep:
         ! they read only pre-update ubt/vbt and write disjoint arrays.
         do j=1,ny
         do i=1,nx
            if (i < nx) then
               h_face_E = 0.5_wp*((bt_work%bt_H_ref(i, j) + bt_work%bt_eta(i, j)) + &
                                  (bt_work%bt_H_ref(i + 1, j) + bt_work%bt_eta(i + 1, j)))
            else
               h_face_E = bt_work%bt_H_ref(i, j) + bt_work%bt_eta(i, j)
            end if
            if (i > 1) then
               h_face_W = 0.5_wp*((bt_work%bt_H_ref(i - 1, j) + bt_work%bt_eta(i - 1, j)) + &
                                  (bt_work%bt_H_ref(i, j) + bt_work%bt_eta(i, j)))
            else
               h_face_W = bt_work%bt_H_ref(i, j) + bt_work%bt_eta(i, j)
            end if
            if (j < ny) then
               h_face_N = 0.5_wp*((bt_work%bt_H_ref(i, j) + bt_work%bt_eta(i, j)) + &
                                  (bt_work%bt_H_ref(i, j + 1) + bt_work%bt_eta(i, j + 1)))
            else
               h_face_N = bt_work%bt_H_ref(i, j) + bt_work%bt_eta(i, j)
            end if
            if (j > 1) then
               h_face_S = 0.5_wp*((bt_work%bt_H_ref(i, j - 1) + bt_work%bt_eta(i, j - 1)) + &
                                  (bt_work%bt_H_ref(i, j) + bt_work%bt_eta(i, j)))
            else
               h_face_S = bt_work%bt_H_ref(i, j) + bt_work%bt_eta(i, j)
            end if
            ubt_R = (1.0_wp + bebt)*bt_work%bt_ubt(i + 1, j) - bebt*bt_work%bt_ubt_prev(i + 1, j)
            ubt_L = (1.0_wp + bebt)*bt_work%bt_ubt(i, j) - bebt*bt_work%bt_ubt_prev(i, j)
            vbt_N = (1.0_wp + bebt)*bt_work%bt_vbt(i, j + 1) - bebt*bt_work%bt_vbt_prev(i, j + 1)
            vbt_S = (1.0_wp + bebt)*bt_work%bt_vbt(i, j) - bebt*bt_work%bt_vbt_prev(i, j)
            flux_x_R = h_face_E*ubt_R*metrics%dy_cu(i + 1, j)
            flux_x_L = h_face_W*ubt_L*metrics%dy_cu(i, j)
            flux_y_N = h_face_N*vbt_N*metrics%dx_cv(i, j + 1)
            flux_y_S = h_face_S*vbt_S*metrics%dx_cv(i, j)
            div_h_u = ((flux_x_R - flux_x_L) + (flux_y_N - flux_y_S))*metrics%iareaT(i, j)
            bt_work%bt_eta_new(i, j) = bt_work%bt_eta(i, j) - dt_inner*div_h_u
            ! Cell (i,j) owns its east face (i+1,j) and north face (i,j+1)
            ! for the transport accumulator, so the per-face sums are race-free.
            bt_work%uhbt_sum(i + 1, j) = bt_work%uhbt_sum(i + 1, j) + flux_x_R
            bt_work%vhbt_sum(i, j + 1) = bt_work%vhbt_sum(i, j + 1) + flux_y_N
            bt_work%bt_ke_centre(i, j) = 0.25_wp*metrics%iareaT(i, j)*( &
                                         metrics%areaCu(i, j)*bt_work%bt_ubt(i, j)**2 + &
                                         metrics%areaCu(i + 1, j)*bt_work%bt_ubt(i + 1, j)**2 + &
                                         metrics%areaCv(i, j)*bt_work%bt_vbt(i, j)**2 + &
                                         metrics%areaCv(i, j + 1)*bt_work%bt_vbt(i, j + 1)**2)
            if (i >= 2 .and. j >= 2) then
               bt_work%bt_zeta_corner(i, j) = &
                  ((bt_work%bt_vbt(i, j)*metrics%dyCv(i, j) - bt_work%bt_vbt(i - 1, j)*metrics%dyCv(i - 1, j)) - &
                   (bt_work%bt_ubt(i, j)*metrics%dxCu(i, j) - bt_work%bt_ubt(i, j - 1)*metrics%dxCu(i, j - 1)))* &
                  metrics%iareaBu(i, j)
            end if
         end do
         end do

         ! ---- eta-swap + zeta wall closure — one barrier sweep ----
         ! Both must finish before Pass 2b reads eta/zeta neighbours; they
         ! write disjoint arrays (eta centres vs zeta corner ring).
         ! CLOSED BASIN: zeta = 0 on the outer ring AND at the physical walls.
         do j=1,ny + 1
         do i=1,nx + 1
            if (i <= nx .and. j <= ny) bt_work%bt_eta(i, j) = bt_work%bt_eta_new(i, j)
            if (i == 1 .or. i == nx + 1) bt_work%bt_zeta_corner(i, j) = 0.0_wp
            if (j == 1 .or. j == ny + 1) bt_work%bt_zeta_corner(i, j) = 0.0_wp
            if (i == grid%nghost + 1) bt_work%bt_zeta_corner(i, j) = 0.0_wp
            if (i == grid%nghost + grid%nx_phys + 1) bt_work%bt_zeta_corner(i, j) = 0.0_wp
            if (j == grid%nghost + 1) bt_work%bt_zeta_corner(i, j) = 0.0_wp
            if (j == grid%nghost + grid%ny_phys + 1) bt_work%bt_zeta_corner(i, j) = 0.0_wp
         end do
         end do

         ! ---- Pass 2b: u_bt update at interior east faces ----
         do j=1,ny
         do i=2,nx
            zeta_at_u = 0.5_wp*(bt_work%bt_zeta_corner(i, j) + bt_work%bt_zeta_corner(i, j + 1))
            f_at_u = 0.5_wp*(cor%f_corner(i, j) + cor%f_corner(i, j + 1))
            if (j > 1 .and. j < ny) then
               v_at_u = 0.25_wp*(bt_work%bt_vbt(i - 1, j) + bt_work%bt_vbt(i - 1, j + 1) + &
                                 bt_work%bt_vbt(i, j) + bt_work%bt_vbt(i, j + 1))
            else if (j == 1) then
               v_at_u = 0.5_wp*(bt_work%bt_vbt(i - 1, j + 1) + bt_work%bt_vbt(i, j + 1))
            else
               v_at_u = 0.5_wp*(bt_work%bt_vbt(i - 1, j) + bt_work%bt_vbt(i, j))
            end if
            ke_grad_x = (bt_work%bt_ke_centre(i, j) - bt_work%bt_ke_centre(i - 1, j))*metrics%idxCu(i, j)
            d_eta = bt_work%bt_eta(i, j) - bt_work%bt_eta(i - 1, j)
            bt_work%bt_ubt(i, j) = bt_work%bt_rem_u(i, j)*( &
                                   bt_work%bt_ubt(i, j) + dt_inner*( &
                                   (zeta_at_u + f_at_u)*v_at_u &
                                   - G*d_eta*metrics%idxCu(i, j) &
                                   - ke_grad_x &
                                   + force_u(i, j)))
         end do
         end do
         ! CLOSED BASIN: hard zero at the array edges and the physical walls.
         do j=1,ny
            bt_work%bt_ubt(1, j) = 0.0_wp
            bt_work%bt_ubt(nx + 1, j) = 0.0_wp
            bt_work%bt_ubt(grid%nghost + 1, j) = 0.0_wp
            bt_work%bt_ubt(grid%nghost + grid%nx_phys + 1, j) = 0.0_wp
         end do

         ! ---- Pass 2c: v_bt update at interior north faces ----
         do j=2,ny
         do i=1,nx
            zeta_at_v = 0.5_wp*(bt_work%bt_zeta_corner(i, j) + bt_work%bt_zeta_corner(i + 1, j))
            f_at_v = 0.5_wp*(cor%f_corner(i, j) + cor%f_corner(i + 1, j))
            if (i > 1 .and. i < nx) then
               u_at_v = 0.25_wp*(bt_work%bt_ubt(i, j - 1) + bt_work%bt_ubt(i + 1, j - 1) + &
                                 bt_work%bt_ubt(i, j) + bt_work%bt_ubt(i + 1, j))
            else if (i == 1) then
               u_at_v = 0.5_wp*(bt_work%bt_ubt(i + 1, j - 1) + bt_work%bt_ubt(i + 1, j))
            else
               u_at_v = 0.5_wp*(bt_work%bt_ubt(i, j - 1) + bt_work%bt_ubt(i, j))
            end if
            ke_grad_y = (bt_work%bt_ke_centre(i, j) - bt_work%bt_ke_centre(i, j - 1))*metrics%idyCv(i, j)
            d_eta = bt_work%bt_eta(i, j) - bt_work%bt_eta(i, j - 1)
            bt_work%bt_vbt(i, j) = bt_work%bt_rem_v(i, j)*( &
                                   bt_work%bt_vbt(i, j) + dt_inner*( &
                                   -(zeta_at_v + f_at_v)*u_at_v &
                                   - G*d_eta*metrics%idyCv(i, j) &
                                   - ke_grad_y &
                                   + force_v(i, j)))
         end do
         end do
         do i=1,nx
            bt_work%bt_vbt(i, 1) = 0.0_wp
            bt_work%bt_vbt(i, ny + 1) = 0.0_wp
            bt_work%bt_vbt(i, grid%nghost + 1) = 0.0_wp
            bt_work%bt_vbt(i, grid%nghost + grid%ny_phys + 1) = 0.0_wp
         end do

         ! ---- time-mean accumulators ----
         do j=1,ny
         do i=1,nx
            bt_work%eta_sum(i, j) = bt_work%eta_sum(i, j) + bt_work%bt_eta(i, j)
         end do
         end do
         do j=1,ny
         do i=1,nx + 1
            bt_work%ubt_sum(i, j) = bt_work%ubt_sum(i, j) + bt_work%bt_ubt(i, j)
         end do
         end do
         do j=1,ny + 1
         do i=1,nx
            bt_work%vbt_sum(i, j) = bt_work%vbt_sum(i, j) + bt_work%bt_vbt(i, j)
         end do
         end do
         !$acc end kernels
      end do
      !$acc wait(1)   ! single drain: the whole substep ran on async(1)

   end subroutine btstep_nonlinear_closed

   !! FUSED variant — the PORTABLE subset of the best-CUDA-practice pass
   !! (btstep/opt_kernel.cu). Loop/kernel fusion only; the cudaGraph is NOT
   !! portable and is deliberately omitted. Per substep the 11 faithful loops
   !! collapse to 5, each fusion a single-assignment merge that folds a wall
   !! closure and/or a time-mean accumulator into its producer:
   !!   * prev-save u + prev-save v  -> ONE guard-merged copy loop
   !!   * eta-swap + zeta walls + eta_sum   (eta_sum rides the swap)
   !!   * Pass 2b + u-wall + ubt_sum        (wall = branch; ubt = wall?0:computed)
   !!   * Pass 2c + v-wall + vbt_sum        (same)
   !! Pass 1 (with its ride-along KE + interior zeta) is untouched. Every merge
   !! reads only the value written in the SAME cell/iteration, so the per-cell
   !! arithmetic order is preserved => bit-identical to the faithful routine.
   !! Valid F2018 do concurrent … local(…); runs under DATA=none (CPU) too.
   subroutine btstep_nonlinear_closed_fused(grid, metrics, bt_work, cor, force_u, force_v, &
                                            n_steps, dt_inner)
      type(hgrid_t), intent(in) :: grid
      type(ocean_metrics_t), intent(in) :: metrics
      type(bt_work_t), intent(inout) :: bt_work
      type(coriolis_t), intent(in) :: cor
      real(wp), intent(in) :: force_u(grid%nx_total + 1, grid%ny_total)
      real(wp), intent(in) :: force_v(grid%nx_total, grid%ny_total + 1)
      integer, intent(in) :: n_steps
      real(wp), intent(in) :: dt_inner

      integer :: i, j, n, nx, ny, ng, nxp, nyp
      real(wp) :: G, bebt
      real(wp) :: h_face_E, h_face_W, h_face_N, h_face_S
      real(wp) :: ubt_R, ubt_L, vbt_N, vbt_S
      real(wp) :: flux_x_R, flux_x_L, flux_y_N, flux_y_S, div_h_u
      real(wp) :: zeta_at_u, f_at_u, v_at_u, ke_grad_x, d_eta
      real(wp) :: zeta_at_v, f_at_v, u_at_v, ke_grad_y

      nx = grid%nx_total
      ny = grid%ny_total
      ng = grid%nghost
      nxp = grid%nx_phys
      nyp = grid%ny_phys
      G = bt_work%g_bt
      bebt = bt_work%bebt

      !$acc kernels async(1)
      do j=1,ny
      do i=1,nx
         bt_work%eta_sum(i, j) = 0.0_wp
      end do
      end do
      do j=1,ny
      do i=1,nx + 1
         bt_work%ubt_sum(i, j) = 0.0_wp
         bt_work%uhbt_sum(i, j) = 0.0_wp
      end do
      end do
      do j=1,ny + 1
      do i=1,nx
         bt_work%vbt_sum(i, j) = 0.0_wp
         bt_work%vhbt_sum(i, j) = 0.0_wp
      end do
      end do
      !$acc end kernels

      do n = 1, n_steps
         !$acc kernels async(1)

         ! ---- prev-save for the BEBT projection (bebt > 0) ----
         ! FUSED: the u-face copy (1:ny,1:nx+1) and v-face copy (1:ny+1,1:nx)
         ! ride one guard-merged sweep over the (1:ny+1,1:nx+1) superset. Pure
         ! independent copies into disjoint arrays -> bit-identical.
         if (bebt > 0.0_wp) then
            do j=1,ny + 1
            do i=1,nx + 1
               if (j <= ny) bt_work%bt_ubt_prev(i, j) = bt_work%bt_ubt(i, j)
               if (i <= nx) bt_work%bt_vbt_prev(i, j) = bt_work%bt_vbt(i, j)
            end do
            end do
         end if

         ! ---- Pass 1: eta update (UNCHANGED from the faithful routine) ----
         do j=1,ny
         do i=1,nx
            if (i < nx) then
               h_face_E = 0.5_wp*((bt_work%bt_H_ref(i, j) + bt_work%bt_eta(i, j)) + &
                                  (bt_work%bt_H_ref(i + 1, j) + bt_work%bt_eta(i + 1, j)))
            else
               h_face_E = bt_work%bt_H_ref(i, j) + bt_work%bt_eta(i, j)
            end if
            if (i > 1) then
               h_face_W = 0.5_wp*((bt_work%bt_H_ref(i - 1, j) + bt_work%bt_eta(i - 1, j)) + &
                                  (bt_work%bt_H_ref(i, j) + bt_work%bt_eta(i, j)))
            else
               h_face_W = bt_work%bt_H_ref(i, j) + bt_work%bt_eta(i, j)
            end if
            if (j < ny) then
               h_face_N = 0.5_wp*((bt_work%bt_H_ref(i, j) + bt_work%bt_eta(i, j)) + &
                                  (bt_work%bt_H_ref(i, j + 1) + bt_work%bt_eta(i, j + 1)))
            else
               h_face_N = bt_work%bt_H_ref(i, j) + bt_work%bt_eta(i, j)
            end if
            if (j > 1) then
               h_face_S = 0.5_wp*((bt_work%bt_H_ref(i, j - 1) + bt_work%bt_eta(i, j - 1)) + &
                                  (bt_work%bt_H_ref(i, j) + bt_work%bt_eta(i, j)))
            else
               h_face_S = bt_work%bt_H_ref(i, j) + bt_work%bt_eta(i, j)
            end if
            ubt_R = (1.0_wp + bebt)*bt_work%bt_ubt(i + 1, j) - bebt*bt_work%bt_ubt_prev(i + 1, j)
            ubt_L = (1.0_wp + bebt)*bt_work%bt_ubt(i, j) - bebt*bt_work%bt_ubt_prev(i, j)
            vbt_N = (1.0_wp + bebt)*bt_work%bt_vbt(i, j + 1) - bebt*bt_work%bt_vbt_prev(i, j + 1)
            vbt_S = (1.0_wp + bebt)*bt_work%bt_vbt(i, j) - bebt*bt_work%bt_vbt_prev(i, j)
            flux_x_R = h_face_E*ubt_R*metrics%dy_cu(i + 1, j)
            flux_x_L = h_face_W*ubt_L*metrics%dy_cu(i, j)
            flux_y_N = h_face_N*vbt_N*metrics%dx_cv(i, j + 1)
            flux_y_S = h_face_S*vbt_S*metrics%dx_cv(i, j)
            div_h_u = ((flux_x_R - flux_x_L) + (flux_y_N - flux_y_S))*metrics%iareaT(i, j)
            bt_work%bt_eta_new(i, j) = bt_work%bt_eta(i, j) - dt_inner*div_h_u
            bt_work%uhbt_sum(i + 1, j) = bt_work%uhbt_sum(i + 1, j) + flux_x_R
            bt_work%vhbt_sum(i, j + 1) = bt_work%vhbt_sum(i, j + 1) + flux_y_N
            bt_work%bt_ke_centre(i, j) = 0.25_wp*metrics%iareaT(i, j)*( &
                                         metrics%areaCu(i, j)*bt_work%bt_ubt(i, j)**2 + &
                                         metrics%areaCu(i + 1, j)*bt_work%bt_ubt(i + 1, j)**2 + &
                                         metrics%areaCv(i, j)*bt_work%bt_vbt(i, j)**2 + &
                                         metrics%areaCv(i, j + 1)*bt_work%bt_vbt(i, j + 1)**2)
            if (i >= 2 .and. j >= 2) then
               bt_work%bt_zeta_corner(i, j) = &
                  ((bt_work%bt_vbt(i, j)*metrics%dyCv(i, j) - bt_work%bt_vbt(i - 1, j)*metrics%dyCv(i - 1, j)) - &
                   (bt_work%bt_ubt(i, j)*metrics%dxCu(i, j) - bt_work%bt_ubt(i, j - 1)*metrics%dxCu(i, j - 1)))* &
                  metrics%iareaBu(i, j)
            end if
         end do
         end do

         ! ---- eta-swap + zeta walls + eta_sum accumulate (FUSED) ----
         ! eta_sum rides the swap: it reads bt_eta(i,j) in the same cell/
         ! iteration the swap just wrote, and bt_eta is not touched again this
         ! substep, so the accumulated value is identical to the trailing loop.
         ! zeta closure is unchanged (disjoint corner-ring writes).
         do j=1,ny + 1
         do i=1,nx + 1
            if (i <= nx .and. j <= ny) then
               bt_work%bt_eta(i, j) = bt_work%bt_eta_new(i, j)
               bt_work%eta_sum(i, j) = bt_work%eta_sum(i, j) + bt_work%bt_eta(i, j)
            end if
            if (i == 1 .or. i == nx + 1) bt_work%bt_zeta_corner(i, j) = 0.0_wp
            if (j == 1 .or. j == ny + 1) bt_work%bt_zeta_corner(i, j) = 0.0_wp
            if (i == ng + 1) bt_work%bt_zeta_corner(i, j) = 0.0_wp
            if (i == ng + nxp + 1) bt_work%bt_zeta_corner(i, j) = 0.0_wp
            if (j == ng + 1) bt_work%bt_zeta_corner(i, j) = 0.0_wp
            if (j == ng + nyp + 1) bt_work%bt_zeta_corner(i, j) = 0.0_wp
         end do
         end do

         ! ---- Pass 2b + u-wall closure + ubt_sum accumulate (FUSED) ----
         ! One sweep over ALL east faces (1:nx+1): a wall face is set to 0
         ! (was a trailing wall loop), an interior face gets the faithful
         ! Pass-2b update, then ubt_sum folds in the just-written bt_ubt(i,j).
         ! Gauss-Seidel: Pass 2c below reads this bt_ubt, so this stays its own
         ! loop (a real barrier), exactly as the faithful routine.
         do j=1,ny
         do i=1,nx + 1
            if (i == 1 .or. i == nx + 1 .or. i == ng + 1 .or. i == ng + nxp + 1) then
               bt_work%bt_ubt(i, j) = 0.0_wp
            else
               zeta_at_u = 0.5_wp*(bt_work%bt_zeta_corner(i, j) + bt_work%bt_zeta_corner(i, j + 1))
               f_at_u = 0.5_wp*(cor%f_corner(i, j) + cor%f_corner(i, j + 1))
               if (j > 1 .and. j < ny) then
                  v_at_u = 0.25_wp*(bt_work%bt_vbt(i - 1, j) + bt_work%bt_vbt(i - 1, j + 1) + &
                                    bt_work%bt_vbt(i, j) + bt_work%bt_vbt(i, j + 1))
               else if (j == 1) then
                  v_at_u = 0.5_wp*(bt_work%bt_vbt(i - 1, j + 1) + bt_work%bt_vbt(i, j + 1))
               else
                  v_at_u = 0.5_wp*(bt_work%bt_vbt(i - 1, j) + bt_work%bt_vbt(i, j))
               end if
               ke_grad_x = (bt_work%bt_ke_centre(i, j) - bt_work%bt_ke_centre(i - 1, j))*metrics%idxCu(i, j)
               d_eta = bt_work%bt_eta(i, j) - bt_work%bt_eta(i - 1, j)
               bt_work%bt_ubt(i, j) = bt_work%bt_rem_u(i, j)*( &
                                      bt_work%bt_ubt(i, j) + dt_inner*( &
                                      (zeta_at_u + f_at_u)*v_at_u &
                                      - G*d_eta*metrics%idxCu(i, j) &
                                      - ke_grad_x &
                                      + force_u(i, j)))
            end if
            bt_work%ubt_sum(i, j) = bt_work%ubt_sum(i, j) + bt_work%bt_ubt(i, j)
         end do
         end do

         ! ---- Pass 2c + v-wall closure + vbt_sum accumulate (FUSED) ----
         do j=1,ny + 1
         do i=1,nx
            if (j == 1 .or. j == ny + 1 .or. j == ng + 1 .or. j == ng + nyp + 1) then
               bt_work%bt_vbt(i, j) = 0.0_wp
            else
               zeta_at_v = 0.5_wp*(bt_work%bt_zeta_corner(i, j) + bt_work%bt_zeta_corner(i + 1, j))
               f_at_v = 0.5_wp*(cor%f_corner(i, j) + cor%f_corner(i + 1, j))
               if (i > 1 .and. i < nx) then
                  u_at_v = 0.25_wp*(bt_work%bt_ubt(i, j - 1) + bt_work%bt_ubt(i + 1, j - 1) + &
                                    bt_work%bt_ubt(i, j) + bt_work%bt_ubt(i + 1, j))
               else if (i == 1) then
                  u_at_v = 0.5_wp*(bt_work%bt_ubt(i + 1, j - 1) + bt_work%bt_ubt(i + 1, j))
               else
                  u_at_v = 0.5_wp*(bt_work%bt_ubt(i, j - 1) + bt_work%bt_ubt(i, j))
               end if
               ke_grad_y = (bt_work%bt_ke_centre(i, j) - bt_work%bt_ke_centre(i, j - 1))*metrics%idyCv(i, j)
               d_eta = bt_work%bt_eta(i, j) - bt_work%bt_eta(i, j - 1)
               bt_work%bt_vbt(i, j) = bt_work%bt_rem_v(i, j)*( &
                                      bt_work%bt_vbt(i, j) + dt_inner*( &
                                      -(zeta_at_v + f_at_v)*u_at_v &
                                      - G*d_eta*metrics%idyCv(i, j) &
                                      - ke_grad_y &
                                      + force_v(i, j)))
            end if
            bt_work%vbt_sum(i, j) = bt_work%vbt_sum(i, j) + bt_work%bt_vbt(i, j)
         end do
         end do
         !$acc end kernels
      end do
      !$acc wait(1)

   end subroutine btstep_nonlinear_closed_fused

end module btstep
