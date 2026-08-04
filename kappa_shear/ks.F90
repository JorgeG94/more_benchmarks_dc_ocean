#include "directives.h"
!! the ocean model's kappa-shear (JHL08) column kernel, extracted for the
!! `do concurrent` vs hand-written-CUDA-C benchmark.
!!
!! SOURCE: <model>/src/parameterizations/vertical/structured/ocean_kappa_shear.F90
!! (1443 lines). The six column-solve helpers —
!!   ks_src_func         (:474-490)
!!   ks_precompute       (:492-553)
!!   ks_projected_state  (:555-692)
!!   ks_find_kappa_tke   (:694-934)
!!   ks_adaptive_dt      (:936-1064)
!!   ks_solve_column     (:1066-1430)
!! — are VERBATIM. Not one arithmetic line was changed.
!!
!! ============================================================================
!! ⚠ WHAT WAS DROPPED, AND WHY — READ THIS BEFORE QUOTING ANY NUMBER
!! ============================================================================
!! This is a TRANSCRIPTION of `kappa_shear_column_kernel` (:296-466), not a
!! verbatim copy. Exactly three things were removed. Each is dead in the
!! production configs (~/analysis_gebco/gabight_sph_acc_v100.nml and siblings):
!!
!! 1. THE D4 MASSLESS-MERGE PATH (production :336-341, :378-386, :390-420).
!!    Gated on `this%massless_merge`, whose default is `.false.`
!!    (ocean_kappa_shear.F90:109) and which NO production namelist sets —
!!    `&ocean_kappa_shear_nml` in every gabight config contains `enable =
!!    .true.` and nothing else. So `merge_on = .false.`, `do_merge` is
!!    unconditionally `.false.`, and the entire `if (do_merge)` branch plus its
!!    nine per-column scratch arrays (hc/uc/vc/tc/sc/kappa_c/tke_c/kc/kf) and
!!    the three `massless_*` calls are unreachable. Dropping it also drops the
!!    `massless` dependency.
!!    ⚠ COST OF THIS CHOICE: those nine arrays are ~9 KB of the per-thread
!!    frame at NZ_STACK_MAX=128. Their absence makes THIS benchmark's frame
!!    SMALLER than production's. Measured frame sizes are therefore a LOWER
!!    BOUND on production's — see README "Caveats".
!!
!! 2. THE OUTER SHIM `kappa_shear_compute` (:276-294). It host-dereferences
!!    `ms%tracers(ms%idx_temperature)%hTr` and forwards; the benchmark's driver
!!    passes hT/hS directly, which is what the shim's callee sees anyway.
!!
!! 3. THE LIFECYCLE / MERGE-INTO-KV-KT PLUMBING (:153-270, :1432-1441): init,
!!    destroy, enter_data, set_f_centre, bytes, kappa_shear_merge_into_kv_kt.
!!    Not part of the column solve. `kappa_shear_merge_into_kv_kt` is a
!!    separate trivially-bandwidth-bound 3-D loop that runs every RK2 stage;
!!    it is NOT what `ocean_vmix_compute` spends its time in and is out of
!!    scope here.
!!
!! Nothing else changed. In particular the `do concurrent(j,i) local(...)`
!! clause, the gather/scatter index flip, the `wet_mask` gate, and the
!! `!$acc routine seq` markers on all six helpers are as shipped.
!!
!! 4. (LATER) -DKS_COUNTERS adds per-column iteration counters -- the WORK
!!    metric the nz sweep needs to separate "more layers" from "more
!!    iterations". They are integer counts only: NO arithmetic line changed,
!!    and they are compiled out entirely by default. NEVER time a KS_COUNTERS
!!    build -- this kernel is register-bound (see OPTIMIZATION.md), so the
!!    extra live integers are not free. tools/ks_sweep.sh runs them separately.
!! ============================================================================
module ks
   use constants, only: wp, GRAVITY, NZ_STACK_MAX, H_VANISHED
   use grid, only: hgrid_t
   use multilayer_cgrid_state, only: multilayer_cgrid_state_t
   use ocean_eos, only: ocean_eos_t, eos_specvol_derivs
   implicit none
   private

   public :: ocean_kappa_shear_t
   public :: kappa_shear_column_kernel
   public :: KS_VARIANT, KS_LANES

   integer, parameter :: NZL = NZ_STACK_MAX
   integer, parameter :: NZLI = NZ_STACK_MAX + 1

   !! Build identity, so the driver can stamp every result row with the
   !! arrangement it actually measured instead of the harness guessing.
   character(len=*), parameter :: KS_VARIANT = 'faithful'
   integer, parameter :: KS_LANES = 1

   !! Knob defaults VERBATIM from ocean_kappa_shear.F90:64-145 (the JHL08 /
   !! OM4 production values). `massless_merge` is gone with the merge path.
   type :: ocean_kappa_shear_t
      logical  :: enable = .false.
      real(wp) :: ri_crit = 0.25_wp
      real(wp) :: shearmix_rate = 0.089_wp
      real(wp) :: fri_curvature = -0.97_wp
      real(wp) :: c_n = 0.24_wp
      real(wp) :: c_s = 0.14_wp
      real(wp) :: lambda = 0.82_wp
      real(wp) :: lz_rescale = 1.0_wp
      real(wp) :: kappa_0 = 1.0e-7_wp
      real(wp) :: kappa_seed = 1.0_wp
      real(wp) :: kappa_trunc = 1.0e-9_wp
      real(wp) :: tke_bg = 0.0_wp
      real(wp) :: tol_err = 0.1_wp
      integer  :: max_inner_it = 50
      integer  :: max_substep_it = 13
      real(wp) :: src_max_chg = 10.0_wp
      real(wp) :: prandtl_turb = 1.0_wp
      real(wp) :: vel_underflow = 0.0_wp
      type(ocean_eos_t) :: eos
      real(wp) :: rho0 = 1035.0_wp
      real(wp), allocatable :: f_centre(:, :)
      real(wp), allocatable :: kd_int(:, :, :)
      real(wp), allocatable :: tke_int(:, :, :)
#ifdef KS_COUNTERS
      !! WORK METRIC (KS_COUNTERS builds only). Per column: substeps actually
      !! executed, and total Picard iterations summed over every
      !! ks_find_kappa_tke call in the column. Written, never read, by the
      !! kernel -- so no reduction and no atomics; the driver sums on the host.
      integer, allocatable :: it_outer(:, :)
      integer, allocatable :: it_inner(:, :)
#endif
   end type ocean_kappa_shear_t

#ifdef KS_COUNTERS
#define KS_CNT_LOCALS , n_out_l, n_in_l
#else
#define KS_CNT_LOCALS
#endif

#ifdef DC_DATA_OMP
   !! ⚠ ifx NEEDS THIS AT MODULE SCOPE, and it is a WARNING not an error.
   !! The per-procedure `DC_ROUTINE_SEQ` (-> `!$omp declare target` inside each
   !! helper) is accepted by nvfortran but NOT honoured by ifx: it still reports
   !!   warning #5476: The DO CONCURRENT construct will not be offloaded; it
   !!   contains a call to a procedure that has not been specified in a DECLARE
   !!   TARGET directive
   !! and then COMPILES AND RUNS ON THE HOST ANYWAY. A `dc_gpu_vendor` lane
   !! would happily record host timings as Intel GPU results. This module-scope
   !! list is what ifx actually accepts; it is additive, so the per-procedure
   !! markers stay for the compilers that use them.
   !$omp declare target(ks_src_func, ks_precompute, ks_projected_state)
   !$omp declare target(ks_find_kappa_tke, ks_adaptive_dt, ks_solve_column)
#endif

contains

   pure subroutine kappa_shear_column_kernel(grid, this, ms, hT, hS, dt)
      !! Per-column JHL08 solve. One `do concurrent (j, i)` over owned cells;
      !! every column is solved serially in surface-down order. TRANSCRIBED
      !! from production :296-466 — see the module header for the three
      !! dropped pieces.
      type(hgrid_t), intent(in) :: grid
      type(ocean_kappa_shear_t), intent(inout) :: this
      type(multilayer_cgrid_state_t), intent(in) :: ms
      real(wp), intent(in) :: hT(:, :, :)
      real(wp), intent(in) :: hS(:, :, :)
      real(wp), intent(in) :: dt

      call ks_columns_dc(grid%nx_total, grid%ny_total, ms%nz_ml, dt, &
                         ms, hT, hS, this%f_centre, this%lz_rescale, &
                         this%rho0, this%ri_crit, this%shearmix_rate, &
                         this%fri_curvature, this%c_n, this%c_s, &
                         this%lambda, this%kappa_0, this%kappa_seed, &
                         this%kappa_trunc, this%tke_bg, this%tol_err, &
                         this%max_inner_it, this%max_substep_it, &
                         this%src_max_chg, this%vel_underflow, this%eos, &
                         this%kd_int, this%tke_int &
#ifdef KS_COUNTERS
                         , this%it_outer, this%it_inner &
#endif
                         )
   end subroutine kappa_shear_column_kernel

   pure subroutine ks_columns_dc(nx, ny, nz, dt, ms, hT, hS, f_centre, &
                                 lz_rescale, rho0, ri_crit, shearmix_rate, &
                                 fri_curvature, c_n, c_s, lambda, kappa_0, &
                                 kappa_seed, kappa_trunc, tke_bg, tol_err, &
                                 max_inner_it, max_substep_it, src_max_chg, &
                                 vel_underflow, eos, kd_int, tke_int &
#ifdef KS_COUNTERS
                                 , it_outer, it_inner &
#endif
                                 )
      !! ⚠ THIS SPLIT EXISTS FOR ONE COMPILER, AND IS NUMERICALLY INERT.
      !! The `do concurrent` body used to live directly in
      !! `kappa_shear_column_kernel` and reference `this%…` throughout. AMD
      !! flang's `-fdo-concurrent-to-openmp=device` pass cannot map a derived
      !! type that has a derived-type COMPONENT:
      !!   DoConcurrentConversion.cpp:603: not yet implemented:
      !!   Nested record types are not supported yet.
      !! `ocean_kappa_shear_t` has exactly one — `type(ocean_eos_t) :: eos` —
      !! and that alone is fatal even for a loop that never reads it (the pass
      !! rejects on the TYPE of a referenced variable, not on the reference).
      !! So the outer routine is now a host-side shim that dereferences `this`,
      !! and the loop sees only plain scalars, plain arrays, and the FLAT
      !! `ocean_eos_t` — which the pass accepts. This mirrors what production's
      !! own `kappa_shear_compute` shim (:276-294) already does for the tracer
      !! registry. Not one arithmetic line changed; `make verify` is the gate.
      integer, intent(in) :: nx, ny, nz
      real(wp), intent(in) :: dt
      type(multilayer_cgrid_state_t), intent(in) :: ms
      real(wp), intent(in) :: hT(:, :, :)
      real(wp), intent(in) :: hS(:, :, :)
      real(wp), intent(in) :: f_centre(:, :)
      real(wp), intent(in) :: lz_rescale, rho0, ri_crit, shearmix_rate
      real(wp), intent(in) :: fri_curvature, c_n, c_s, lambda
      real(wp), intent(in) :: kappa_0, kappa_seed, kappa_trunc, tke_bg, tol_err
      integer, intent(in) :: max_inner_it, max_substep_it
      real(wp), intent(in) :: src_max_chg, vel_underflow
      type(ocean_eos_t), intent(in) :: eos
      real(wp), intent(inout) :: kd_int(:, :, :), tke_int(:, :, :)
#ifdef KS_COUNTERS
      integer, intent(inout) :: it_outer(:, :), it_inner(:, :)
#endif

      integer :: i, j, k, kg
      real(wp) :: f2_val, hk, inv_h
      real(wp) :: h_sd(NZL), u_sd(NZL), v_sd(NZL), t_sd(NZL), s_sd(NZL)
      real(wp) :: idz_s(NZL), idz_int_s(NZLI), hint_s(NZLI), il2_s(NZLI)
      real(wp) :: kappa_avg_sd(NZLI), tke_avg_sd(NZLI)
#ifdef KS_COUNTERS
      integer :: n_out_l, n_in_l
#endif

      do concurrent(j=1:ny, i=1:nx) &
         local(k, kg, f2_val, hk, inv_h KS_CNT_LOCALS, &
               h_sd, u_sd, v_sd, t_sd, s_sd, &
               idz_s, idz_int_s, hint_s, il2_s, &
               kappa_avg_sd, tke_avg_sd)

         ! ---- Default outputs (overwritten for wet columns) ----
         do k = 1, nz + 1
            kappa_avg_sd(k) = 0.0_wp
            tke_avg_sd(k) = 0.0_wp
         end do
#ifdef KS_COUNTERS
         n_out_l = 0
         n_in_l = 0
#endif

         if (ms%wet_mask(i, j) > 0.0_wp) then
            ! ---- Gather with the index flip (local k=1 = surface) ----
            do k = 1, nz
               kg = nz + 1 - k
               h_sd(k) = ms%h_layer(i, j, kg)
               u_sd(k) = 0.5_wp*(ms%u_face_x_layer(i, j, kg) + &
                                 ms%u_face_x_layer(i + 1, j, kg))
               v_sd(k) = 0.5_wp*(ms%v_face_y_layer(i, j, kg) + &
                                 ms%v_face_y_layer(i, j + 1, kg))
            end do

            f2_val = f_centre(i, j)*f_centre(i, j)

            ! ---- Gather floor at H_VANISHED, solve on nz. (Production's
            !      `else` branch — the live one; `do_merge` is always false.)
            do k = 1, nz
               kg = nz + 1 - k
               hk = max(h_sd(k), H_VANISHED)
               inv_h = 1.0_wp/hk
               h_sd(k) = hk
               t_sd(k) = hT(i, j, kg)*inv_h
               s_sd(k) = hS(i, j, kg)*inv_h
            end do

            call ks_precompute(nz, lz_rescale, h_sd, &
                               idz_s, idz_int_s, hint_s, il2_s)

            call ks_solve_column(nz, dt, f2_val, rho0, &
                                 ri_crit, shearmix_rate, &
                                 fri_curvature, c_n, c_s, &
                                 lambda, kappa_0, kappa_seed, &
                                 kappa_trunc, tke_bg, tol_err, &
                                 max_inner_it, max_substep_it, &
                                 src_max_chg, vel_underflow, &
                                 eos, &
                                 h_sd, u_sd, v_sd, t_sd, s_sd, &
                                 idz_s, idz_int_s, hint_s, il2_s, &
                                 kappa_avg_sd, tke_avg_sd &
#ifdef KS_COUNTERS
                                 , n_out_l, n_in_l &
#endif
                                 )
         end if

         ! ---- Scatter with the flip; force exact 0 at bed + surface ----
         do k = 1, nz + 1
            kg = nz + 2 - k
            kd_int(i, j, kg) = kappa_avg_sd(k)
            tke_int(i, j, kg) = tke_avg_sd(k)
         end do
         kd_int(i, j, 1) = 0.0_wp
         kd_int(i, j, nz + 1) = 0.0_wp
         tke_int(i, j, 1) = 0.0_wp
         tke_int(i, j, nz + 1) = 0.0_wp
#ifdef KS_COUNTERS
         it_outer(i, j) = n_out_l
         it_inner(i, j) = n_in_l
#endif
      end do
   end subroutine ks_columns_dc

   ! =================================================================
   ! Column solve helpers — VERBATIM from production :474-1430.
   ! =================================================================

   pure function ks_src_func(ri_crit, shearmix_rate, fri_curvature, n2, s2) &
      result(ksrc)
      DC_ROUTINE_SEQ
      real(wp), intent(in) :: ri_crit, shearmix_rate, fri_curvature, n2, s2
      real(wp) :: ksrc
      real(wp) :: dnom

      ksrc = 0.0_wp
      if (n2 < ri_crit*s2) then
         dnom = ri_crit*s2 + fri_curvature*n2
         if (dnom /= 0.0_wp .and. s2 > 0.0_wp) then
            ksrc = 2.0_wp*shearmix_rate*sqrt(s2)*(ri_crit*s2 - n2)/dnom
         end if
      end if
   end function ks_src_func

   pure subroutine ks_precompute(nz, lz_rescale, h_sd, &
                                 idz_o, idz_int_o, hint_o, il2_o)
      DC_ROUTINE_SEQ
      integer, intent(in) :: nz
      real(wp), intent(in) :: lz_rescale
      real(wp), intent(in) :: h_sd(NZL)
      real(wp), intent(out) :: idz_o(NZL), idz_int_o(NZLI)
      real(wp), intent(out) :: hint_o(NZLI), il2_o(NZLI)

      integer :: k
      real(wp) :: hk, hkm1, hkp1, norm_l, wt_a, wt_b, i_lz2, dtop
      real(wp) :: dbot(NZLI)

      i_lz2 = 1.0_wp/(lz_rescale*lz_rescale)

      do k = 1, nz
         idz_o(k) = 1.0_wp/h_sd(k)
      end do

      idz_int_o(1) = 2.0_wp/h_sd(1)
      do k = 2, nz
         idz_int_o(k) = 2.0_wp/(h_sd(k - 1) + h_sd(k))
      end do
      idz_int_o(nz + 1) = 2.0_wp/h_sd(nz)

      hint_o(1) = 0.0_wp
      if (nz >= 2) then
         hint_o(2) = h_sd(1)
         do k = 2, nz - 1
            hk = h_sd(k)
            hkm1 = h_sd(k - 1)
            hkp1 = h_sd(k + 1)
            norm_l = 1.0_wp/(hk*(hkm1 + hkp1) + 2.0_wp*hkm1*hkp1)
            wt_a = (hk + hkp1)*hkm1*norm_l
            wt_b = (hkm1 + hk)*hkp1*norm_l
            hint_o(k) = hint_o(k) + hk*wt_a
            hint_o(k + 1) = hk*wt_b
         end do
         hint_o(nz) = hint_o(nz) + h_sd(nz)
      end if
      hint_o(nz + 1) = 0.0_wp

      dbot(nz + 1) = 0.0_wp
      do k = nz, 1, -1
         dbot(k) = dbot(k + 1) + h_sd(k)
      end do
      il2_o(1) = 0.0_wp
      il2_o(nz + 1) = 0.0_wp
      dtop = 0.0_wp
      do k = 2, nz
         dtop = dtop + h_sd(k - 1)
         if (dtop > 0.0_wp .and. dbot(k) > 0.0_wp) then
            il2_o(k) = i_lz2*(dtop + dbot(k))**2/((dtop*dbot(k))**2)
         else
            il2_o(k) = 0.0_wp
         end if
      end do
   end subroutine ks_precompute

   pure subroutine ks_projected_state(nz, dt_now, ks, ke, vel_underflow, &
                                      dbuoy_t, dbuoy_s, h_sd, idz_int_s, &
                                      u0, v0, t0, s0, kappa_ps, &
                                      u_o, v_o, t_o, s_o, c1_o, n2_o, s2_o)
      DC_ROUTINE_SEQ
      integer, intent(in) :: nz, ks, ke
      real(wp), intent(in) :: dt_now, vel_underflow
      real(wp), intent(in) :: dbuoy_t(NZLI), dbuoy_s(NZLI)
      real(wp), intent(in) :: h_sd(NZL), idz_int_s(NZLI)
      real(wp), intent(in) :: u0(NZL), v0(NZL), t0(NZL), s0(NZL)
      real(wp), intent(in) :: kappa_ps(NZLI)
      real(wp), intent(out) :: u_o(NZL), v_o(NZL), t_o(NZL), s_o(NZL)
      real(wp), intent(out) :: c1_o(NZLI), n2_o(NZLI), s2_o(NZLI)

      integer :: k, kk
      real(wp) :: a_b, a_a, b1, b1nz, d1, bd1
      real(wp) :: ua, ub, va, vb, ta, tb, sa, sb, n2v

      do k = 1, nz
         u_o(k) = u0(k)
         v_o(k) = v0(k)
         t_o(k) = t0(k)
         s_o(k) = s0(k)
      end do
      do k = 1, nz + 1
         c1_o(k) = 0.0_wp
      end do

      if (ks <= ke .and. dt_now > 0.0_wp) then
         ! Forward sweep (top layer ks).
         a_b = dt_now*kappa_ps(ks + 1)*idz_int_s(ks + 1)
         b1 = 1.0_wp/(h_sd(ks) + a_b)
         c1_o(ks + 1) = a_b*b1
         d1 = h_sd(ks)*b1
         u_o(ks) = b1*h_sd(ks)*u0(ks)
         v_o(ks) = b1*h_sd(ks)*v0(ks)
         t_o(ks) = b1*h_sd(ks)*t0(ks)
         s_o(ks) = b1*h_sd(ks)*s0(ks)

         do k = ks + 1, ke - 1
            a_a = a_b
            a_b = dt_now*kappa_ps(k + 1)*idz_int_s(k + 1)
            bd1 = h_sd(k) + d1*a_a
            b1 = 1.0_wp/(bd1 + a_b)
            c1_o(k + 1) = a_b*b1
            d1 = bd1*b1
            u_o(k) = b1*(h_sd(k)*u0(k) + a_a*u_o(k - 1))
            v_o(k) = b1*(h_sd(k)*v0(k) + a_a*v_o(k - 1))
            t_o(k) = b1*(h_sd(k)*t0(k) + a_a*t_o(k - 1))
            s_o(k) = b1*(h_sd(k)*s0(k) + a_a*s_o(k - 1))
         end do

         ! Bottom layer of the band (ke).
         a_a = a_b
         if (ke > ks) then
            b1 = 1.0_wp/(h_sd(ke) + d1*a_a)
            t_o(ke) = b1*(h_sd(ke)*t0(ke) + a_a*t_o(ke - 1))
            s_o(ke) = b1*(h_sd(ke)*s0(ke) + a_a*s_o(ke - 1))
            if (ke == nz) then
               b1nz = 1.0_wp/((h_sd(ke) + d1*a_a) + &
                              dt_now*kappa_ps(nz + 1)*idz_int_s(nz + 1))
            else
               b1nz = b1
            end if
            u_o(ke) = b1nz*(h_sd(ke)*u0(ke) + a_a*u_o(ke - 1))
            v_o(ke) = b1nz*(h_sd(ke)*v0(ke) + a_a*v_o(ke - 1))
         else
            b1 = 1.0_wp/(h_sd(ke) + a_a)
            t_o(ke) = b1*h_sd(ke)*t0(ke)
            s_o(ke) = b1*h_sd(ke)*s0(ke)
            if (ke == nz) then
               b1nz = 1.0_wp/(h_sd(ke) + a_a + &
                              dt_now*kappa_ps(nz + 1)*idz_int_s(nz + 1))
            else
               b1nz = b1
            end if
            u_o(ke) = b1nz*h_sd(ke)*u0(ke)
            v_o(ke) = b1nz*h_sd(ke)*v0(ke)
         end if
         if (abs(u_o(ke)) < vel_underflow) u_o(ke) = 0.0_wp
         if (abs(v_o(ke)) < vel_underflow) v_o(ke) = 0.0_wp

         ! Back-substitution upward.
         do k = ke - 1, ks, -1
            u_o(k) = u_o(k) + c1_o(k + 1)*u_o(k + 1)
            v_o(k) = v_o(k) + c1_o(k + 1)*v_o(k + 1)
            t_o(k) = t_o(k) + c1_o(k + 1)*t_o(k + 1)
            s_o(k) = s_o(k) + c1_o(k + 1)*s_o(k + 1)
            if (abs(u_o(k)) < vel_underflow) u_o(k) = 0.0_wp
            if (abs(v_o(k)) < vel_underflow) v_o(k) = 0.0_wp
         end do
      else
         do k = 1, nz
            if (abs(u_o(k)) < vel_underflow) u_o(k) = 0.0_wp
            if (abs(v_o(k)) < vel_underflow) v_o(k) = 0.0_wp
         end do
      end if

      ! N^2, S^2 — mixed inside the band, original values outside.
      n2_o(1) = 0.0_wp
      n2_o(nz + 1) = 0.0_wp
      s2_o(1) = 0.0_wp
      s2_o(nz + 1) = 0.0_wp
      do kk = 2, nz
         if (kk - 1 >= ks .and. kk - 1 <= ke) then
            ua = u_o(kk - 1)
            va = v_o(kk - 1)
            ta = t_o(kk - 1)
            sa = s_o(kk - 1)
         else
            ua = u0(kk - 1)
            va = v0(kk - 1)
            ta = t0(kk - 1)
            sa = s0(kk - 1)
         end if
         if (kk >= ks .and. kk <= ke) then
            ub = u_o(kk)
            vb = v_o(kk)
            tb = t_o(kk)
            sb = s_o(kk)
         else
            ub = u0(kk)
            vb = v0(kk)
            tb = t0(kk)
            sb = s0(kk)
         end if
         n2v = idz_int_s(kk)*(dbuoy_t(kk)*(ta - tb) + dbuoy_s(kk)*(sa - sb))
         if (n2v < 0.0_wp) n2v = 0.0_wp
         n2_o(kk) = n2v
         s2_o(kk) = ((ua - ub)**2 + (va - vb)**2)*idz_int_s(kk)**2
      end do
   end subroutine ks_projected_state

   pure subroutine ks_find_kappa_tke(nz, tke_min, f2_val, &
                                     ri_crit, shearmix_rate, fri_curvature, &
                                     c_n2, c_s2, ilambda2, kappa_0, &
                                     kappa_trunc, tke_bg, tol_err, max_inner_it, &
                                     n2_in, s2_in, kappa_seed, k_q_io, &
                                     idz_s, hint_s, il2_s, e1_s, &
                                     tke_o, kappa_o, &
                                     ksrc_sc, tkedec_sc, aq_sc, dq_sc, cq_sc, &
                                     dk_sc, ck_sc, ild2_sc &
#ifdef KS_COUNTERS
                                     , n_it &
#endif
                                     )
      DC_ROUTINE_SEQ
      integer, intent(in) :: nz, max_inner_it
      real(wp), intent(in) :: tke_min, f2_val
      real(wp), intent(in) :: ri_crit, shearmix_rate, fri_curvature
      real(wp), intent(in) :: c_n2, c_s2, ilambda2, kappa_0
      real(wp), intent(in) :: kappa_trunc, tke_bg, tol_err
      real(wp), intent(in) :: n2_in(NZLI), s2_in(NZLI)
      real(wp), intent(in) :: kappa_seed(NZLI)
      real(wp), intent(inout) :: k_q_io(NZLI)
      real(wp), intent(in) :: idz_s(NZL), hint_s(NZLI), il2_s(NZLI), e1_s(NZLI)
      real(wp), intent(out) :: tke_o(NZLI), kappa_o(NZLI)
      real(wp), intent(inout) :: ksrc_sc(NZLI), tkedec_sc(NZLI)
      real(wp), intent(inout) :: aq_sc(NZL), dq_sc(NZLI), cq_sc(NZLI)
      real(wp), intent(inout) :: dk_sc(NZLI), ck_sc(NZLI), ild2_sc(NZLI)
#ifdef KS_COUNTERS
      integer, intent(out) :: n_it
#endif

      integer :: kk, k, k2, it
      integer :: ks_src, ke_src, ks_kap, ke_kap, ks_kp, ke_kp, ke_tke
      integer :: ks_new, ke_new, k_lo, k_hi
      real(wp) :: cqc_l, ckc_l, bqd1_l, bq_l, tsrc_l, raw_l, dnom_l
      real(wp) :: bkd1_l, bk_l, trv, tr2v, lhs_l, rhs_l
      logical :: conv

#ifdef KS_COUNTERS
      n_it = 0
#endif
      ks_src = nz + 2
      ke_src = 0
      ksrc_sc(1) = 0.0_wp
      ksrc_sc(nz + 1) = 0.0_wp
      do kk = 2, nz
         ksrc_sc(kk) = ks_src_func(ri_crit, shearmix_rate, fri_curvature, &
                                   n2_in(kk), s2_in(kk))
         if (ksrc_sc(kk) > 0.0_wp) then
            if (ks_src > kk) ks_src = kk
            ke_src = kk
         end if
      end do

      if (ks_src > ke_src) then
         do kk = 1, nz + 1
            tke_o(kk) = tke_min
            kappa_o(kk) = 0.0_wp
            k_q_io(kk) = 0.0_wp
         end do
         return
      end if

      do kk = 2, nz
         tkedec_sc(kk) = sqrt(c_n2*n2_in(kk) + c_s2*s2_in(kk))
      end do

      tke_o(1) = tke_bg
      do kk = 2, nz
         if (kappa_seed(kk) > 0.0_wp .and. k_q_io(kk) > 0.0_wp) then
            tke_o(kk) = kappa_seed(kk)/k_q_io(kk)
         else
            tke_o(kk) = tke_min
         end if
      end do
      tke_o(nz + 1) = tke_min

      do kk = 1, nz + 1
         kappa_o(kk) = kappa_seed(kk)
      end do
      kappa_o(1) = 0.0_wp
      kappa_o(nz + 1) = 0.0_wp

      ks_kap = 2
      ke_kap = nz
      ks_kp = 2
      ke_kp = nz

      do it = 1, max_inner_it
#ifdef KS_COUNTERS
         n_it = n_it + 1
#endif
         ! (a) TKE tridiagonal sweep.
         ke_tke = min(max(ke_kap, ke_kp) + 1, nz + 1)
         do k = 1, min(ke_tke, nz)
            aq_sc(k) = (0.5_wp*(kappa_o(k) + kappa_o(k + 1)) + kappa_0)*idz_s(k)
         end do

         dq_sc(1) = -tke_o(1)
         tke_o(1) = tke_bg
         cq_sc(2) = 0.0_wp
         cqc_l = 1.0_wp
         do kk = 2, ke_tke - 1
            dq_sc(kk) = -tke_o(kk)
            tsrc_l = (kappa_o(kk) + kappa_0)*s2_in(kk) + tke_bg*tkedec_sc(kk)
            bqd1_l = hint_s(kk)*(tkedec_sc(kk) + n2_in(kk)*k_q_io(kk)) + &
                     cqc_l*aq_sc(kk - 1)
            bq_l = 1.0_wp/(bqd1_l + aq_sc(kk))
            tke_o(kk) = bq_l*(hint_s(kk)*tsrc_l + aq_sc(kk - 1)*tke_o(kk - 1))
            cq_sc(kk + 1) = aq_sc(kk)*bq_l
            cqc_l = bqd1_l*bq_l
         end do

         if (ke_tke == nz + 1) then
            tke_o(nz + 1) = tke_min
            dq_sc(nz + 1) = 0.0_wp
         else
            kk = ke_tke
            tsrc_l = kappa_0*s2_in(kk) + tke_bg*tkedec_sc(kk)
            bq_l = 1.0_wp/(hint_s(kk)*tkedec_sc(kk) + cqc_l*aq_sc(kk - 1) + &
                           aq_sc(kk))
            cq_sc(kk + 1) = aq_sc(kk)*bq_l
            dq_sc(kk) = -tke_o(kk)
            raw_l = bq_l*(hint_s(kk)*tsrc_l + aq_sc(kk - 1)*tke_o(kk - 1))
            dnom_l = 1.0_wp - cq_sc(kk + 1)*e1_s(kk + 1)
            if (abs(dnom_l) > 1.0e-30_wp) then
               tke_o(kk) = max((raw_l + cq_sc(kk + 1)* &
                                (tke_o(kk + 1) - e1_s(kk + 1)*tke_o(kk)))/dnom_l, &
                               tke_min)
            else
               tke_o(kk) = max(raw_l, tke_min)
            end if
            dq_sc(kk) = tke_o(kk) + dq_sc(kk)
            do k2 = ke_tke + 1, nz + 1
               dq_sc(k2) = e1_s(k2)*dq_sc(k2 - 1)
               tke_o(k2) = max(tke_o(k2) + dq_sc(k2), tke_min)
               if (abs(dq_sc(k2)) < 1.0e-16_wp*tke_o(k2)) exit
            end do
         end if

         do kk = ke_tke - 1, 1, -1
            tke_o(kk) = max(tke_o(kk) + cq_sc(kk + 1)*tke_o(kk + 1), tke_min)
            dq_sc(kk) = tke_o(kk) + dq_sc(kk)
         end do

         ! (b) kappa tridiagonal sweep with truncation ramp + range track.
         ks_kp = ks_kap
         ke_kp = ke_kap
         do kk = 2, nz
            if (tke_o(kk) > 0.0_wp) then
               ild2_sc(kk) = (n2_in(kk)*ilambda2 + f2_val)/tke_o(kk) + il2_s(kk)
            else
               ild2_sc(kk) = 1.0e30_wp
            end if
         end do

         dk_sc(1) = 0.0_wp
         ck_sc(2) = 0.0_wp
         ckc_l = 1.0_wp
         ke_new = 0
         ks_new = nz
         do kk = 2, nz
            dk_sc(kk) = -kappa_o(kk)
            bkd1_l = hint_s(kk)*ild2_sc(kk) + ckc_l*idz_s(kk - 1)
            bk_l = 1.0_wp/(bkd1_l + idz_s(kk))
            kappa_o(kk) = bk_l*(idz_s(kk - 1)*kappa_o(kk - 1) + &
                                hint_s(kk)*ksrc_sc(kk))
            ck_sc(kk + 1) = idz_s(kk)*bk_l
            ckc_l = bkd1_l*bk_l

            trv = ckc_l*kappa_trunc
            tr2v = 2.0_wp*trv
            if (kappa_o(kk) < trv) then
               kappa_o(kk) = 0.0_wp
               if (kk > ke_src) then
                  ke_kap = kk - 1
                  k_q_io(kk) = 0.0_wp
                  exit
               end if
            else if (kappa_o(kk) < tr2v) then
               kappa_o(kk) = 2.0_wp*(kappa_o(kk) - trv)
            end if
            ke_new = kk
         end do
         if (ke_new > 0) ke_kap = ke_new

         if (ke_kap >= 1 .and. tke_o(ke_kap) > 0.0_wp) then
            k_q_io(ke_kap) = kappa_o(ke_kap)/tke_o(ke_kap)
         end if
         dk_sc(ke_kap) = dk_sc(ke_kap) + kappa_o(ke_kap)

         do kk = ke_kap + 2, ke_kp + 1
            if (kk > nz) exit
            dk_sc(kk) = -kappa_o(kk)
            kappa_o(kk) = 0.0_wp
            k_q_io(kk) = 0.0_wp
         end do

         ks_new = 2
         do kk = ke_kap - 1, 2, -1
            kappa_o(kk) = kappa_o(kk) + ck_sc(kk + 1)*kappa_o(kk + 1)
            if (kappa_o(kk) <= kappa_trunc) then
               kappa_o(kk) = 0.0_wp
               if (kk < ks_src) then
                  ks_kap = kk + 1
                  k_q_io(kk) = 0.0_wp
                  exit
               end if
            else if (kappa_o(kk) < 2.0_wp*kappa_trunc) then
               kappa_o(kk) = 2.0_wp*(kappa_o(kk) - kappa_trunc)
            end if
            dk_sc(kk) = dk_sc(kk) + kappa_o(kk)
            if (tke_o(kk) > 0.0_wp) then
               k_q_io(kk) = kappa_o(kk)/tke_o(kk)
            else
               k_q_io(kk) = 0.0_wp
            end if
            ks_new = kk
         end do
         ks_kap = max(ks_new, 2)

         do kk = ks_kp, ks_kap - 2
            if (kk < 2) cycle
            kappa_o(kk) = 0.0_wp
            k_q_io(kk) = 0.0_wp
         end do

         ! (c) Picard convergence test.
         k_lo = min(ks_kap, ks_kp)
         k_hi = max(ke_kap, ke_kp)
         conv = .true.
         do kk = k_lo, k_hi
            lhs_l = abs(dk_sc(kk))
            rhs_l = tol_err*(kappa_0 + kappa_o(kk) - 0.5_wp*dk_sc(kk))
            if (lhs_l > rhs_l) then
               conv = .false.
               exit
            end if
         end do
         if (conv) exit
      end do

      kappa_o(1) = 0.0_wp
      kappa_o(nz + 1) = 0.0_wp
   end subroutine ks_find_kappa_tke

   pure function ks_adaptive_dt(nz, dt_rem, itt_outer, max_substep_it, &
                                ri_crit, shearmix_rate, fri_curvature, &
                                src_max_chg, tol_err, vel_underflow, &
                                dbuoy_t, dbuoy_s, &
                                h_s, u_cur, v_cur, t_cur, s_cur, &
                                kappa_out_s, kappa_src_s, local_src_s, &
                                local_src_avg_s, ks_kap, ke_kap, &
                                idz_int_s) result(dt_now_r)
      DC_ROUTINE_SEQ
      integer, intent(in) :: nz, itt_outer, max_substep_it, ks_kap, ke_kap
      real(wp), intent(in) :: dt_rem, ri_crit, shearmix_rate, fri_curvature
      real(wp), intent(in) :: src_max_chg, tol_err, vel_underflow
      real(wp), intent(in) :: dbuoy_t(NZLI), dbuoy_s(NZLI)
      real(wp), intent(in) :: h_s(NZL), u_cur(NZL), v_cur(NZL)
      real(wp), intent(in) :: t_cur(NZL), s_cur(NZL)
      real(wp), intent(in) :: kappa_out_s(NZLI), kappa_src_s(NZLI)
      real(wp), intent(in) :: local_src_s(NZLI), local_src_avg_s(NZLI)
      real(wp), intent(in) :: idz_int_s(NZLI)
      real(wp) :: dt_now_r

      real(wp) :: tol_max(NZLI), tol_min_a(NZLI), tol_chg_a(NZLI)
      real(wp) :: u_pr(NZL), v_pr(NZL), t_pr(NZL), s_pr(NZL)
      real(wp) :: c1_pr(NZLI), n2_pr(NZLI), s2_pr(NZLI)

      integer :: kk, k_lo, k_hi, ih, ir, max_halvings, ks_lyr, ke_lyr
      real(wp) :: dt_tst, dt_inc, dt_try, idtt, tol_dksrc_low
      real(wp) :: ksrc_tst, upper_t, lower_t, tol2_l
      logical :: valid_dt, valid_try

      if (src_max_chg == 10.0_wp) then
         tol_dksrc_low = 0.95_wp
      else
         tol_dksrc_low = (src_max_chg - 0.5_wp)/src_max_chg
      end if
      tol2_l = 2.0_wp*tol_err

      do kk = 1, nz + 1
         tol_max(kk) = kappa_src_s(kk) + src_max_chg*local_src_s(kk)
         tol_min_a(kk) = kappa_src_s(kk) - tol_dksrc_low*local_src_s(kk)
         tol_chg_a(kk) = tol2_l*local_src_avg_s(kk)
      end do

      k_lo = max(ks_kap - 1, 2)
      k_hi = min(ke_kap + 1, nz)
      ks_lyr = max(ks_kap - 1, 1)
      ke_lyr = min(ke_kap, nz)

      dt_tst = dt_rem
      valid_dt = .false.
      max_halvings = (max_substep_it + 1 - itt_outer)/2

      ! Halving pass.
      do ih = 1, max(max_halvings, 1)
         call ks_projected_state(nz, 0.5_wp*dt_tst, ks_lyr, ke_lyr, &
                                 vel_underflow, dbuoy_t, dbuoy_s, h_s, &
                                 idz_int_s, u_cur, v_cur, t_cur, s_cur, &
                                 kappa_out_s, u_pr, v_pr, t_pr, s_pr, &
                                 c1_pr, n2_pr, s2_pr)
         idtt = 0.0_wp
         if (dt_tst > 0.0_wp) idtt = 1.0_wp/dt_tst
         valid_dt = .true.
         do kk = k_lo, k_hi
            if (n2_pr(kk) < ri_crit*s2_pr(kk)) then
               ksrc_tst = ks_src_func(ri_crit, shearmix_rate, fri_curvature, &
                                      n2_pr(kk), s2_pr(kk))
               upper_t = max(tol_max(kk), kappa_src_s(kk) + idtt*tol_chg_a(kk))
               lower_t = min(tol_min_a(kk), kappa_src_s(kk) - idtt*tol_chg_a(kk))
               if (ksrc_tst > upper_t .or. ksrc_tst < lower_t) then
                  valid_dt = .false.
                  exit
               end if
            else
               lower_t = min(tol_min_a(kk), kappa_src_s(kk) - idtt*tol_chg_a(kk))
               if (0.0_wp < lower_t) then
                  valid_dt = .false.
                  exit
               end if
            end if
         end do
         if (valid_dt) exit
         dt_tst = 0.5_wp*dt_tst
      end do

      ! Refinement pass.
      dt_inc = 0.0_wp
      if (dt_tst < dt_rem .and. valid_dt) then
         dt_inc = 0.5_wp*dt_tst
         do ir = 1, 5
            dt_try = dt_tst + dt_inc
            call ks_projected_state(nz, 0.5_wp*dt_try, ks_lyr, ke_lyr, &
                                    vel_underflow, dbuoy_t, dbuoy_s, h_s, &
                                    idz_int_s, u_cur, v_cur, t_cur, s_cur, &
                                    kappa_out_s, u_pr, v_pr, t_pr, s_pr, &
                                    c1_pr, n2_pr, s2_pr)
            idtt = 0.0_wp
            if (dt_try > 0.0_wp) idtt = 1.0_wp/dt_try
            valid_try = .true.
            do kk = k_lo, k_hi
               if (n2_pr(kk) < ri_crit*s2_pr(kk)) then
                  ksrc_tst = ks_src_func(ri_crit, shearmix_rate, &
                                         fri_curvature, n2_pr(kk), s2_pr(kk))
                  upper_t = max(tol_max(kk), kappa_src_s(kk) + idtt*tol_chg_a(kk))
                  lower_t = min(tol_min_a(kk), &
                                kappa_src_s(kk) - idtt*tol_chg_a(kk))
                  if (ksrc_tst > upper_t .or. ksrc_tst < lower_t) then
                     valid_try = .false.
                     exit
                  end if
               else
                  lower_t = min(tol_min_a(kk), &
                                kappa_src_s(kk) - idtt*tol_chg_a(kk))
                  if (0.0_wp < lower_t) then
                     valid_try = .false.
                     exit
                  end if
               end if
            end do
            if (valid_try) dt_tst = dt_try
            dt_inc = 0.5_wp*dt_inc
         end do
      end if

      dt_now_r = min(dt_tst*(1.0_wp + tol_err) + dt_inc, dt_rem)
   end function ks_adaptive_dt

   pure subroutine ks_solve_column(nz, dt, f2_val, rho0, &
                                   ri_crit, shearmix_rate, fri_curvature, &
                                   c_n, c_s, lambda, kappa_0, kappa_seed_in, &
                                   kappa_trunc, tke_bg, tol_err, max_inner_it, &
                                   max_substep_it, src_max_chg, vel_underflow, &
                                   eos, &
                                   h_sd, u_sd, v_sd, t_sd, s_sd, &
                                   idz_s, idz_int_s, hint_s, il2_s, &
                                   kappa_avg_sd, tke_avg_sd &
#ifdef KS_COUNTERS
                                   , n_out, n_in &
#endif
                                   )
      DC_ROUTINE_SEQ
      type(ocean_eos_t), intent(in) :: eos
      integer, intent(in) :: nz, max_inner_it, max_substep_it
      real(wp), intent(in) :: dt, f2_val, rho0
      real(wp), intent(in) :: ri_crit, shearmix_rate, fri_curvature
      real(wp), intent(in) :: c_n, c_s, lambda, kappa_0, kappa_seed_in
      real(wp), intent(in) :: kappa_trunc, tke_bg, tol_err, src_max_chg
      real(wp), intent(in) :: vel_underflow
      real(wp), intent(in) :: h_sd(NZL), u_sd(NZL), v_sd(NZL)
      real(wp), intent(in) :: t_sd(NZL), s_sd(NZL)
      real(wp), intent(in) :: idz_s(NZL), idz_int_s(NZLI)
      real(wp), intent(in) :: hint_s(NZLI), il2_s(NZLI)
      real(wp), intent(out) :: kappa_avg_sd(NZLI), tke_avg_sd(NZLI)
#ifdef KS_COUNTERS
      integer, intent(out) :: n_out, n_in
      integer :: n_it_l
#endif

      ! Per-thread work arrays (fixed size, surface-down).
      real(wp) :: u_c(NZL), v_c(NZL), t_c(NZL), s_c(NZL)
      real(wp) :: dbuoy_t(NZLI), dbuoy_s(NZLI)
      real(wp) :: e1(NZLI)
      real(wp) :: kappa(NZLI), k_q(NZLI), kappa_avg(NZLI), tke_avg(NZLI)
      real(wp) :: n2(NZLI), s2(NZLI), tke(NZLI)
      real(wp) :: ksrc(NZLI), tkedec(NZLI)
      real(wp) :: aq(NZL), dq(NZLI), cq(NZLI)
      real(wp) :: dk(NZLI), ck(NZLI), ild2(NZLI)
      real(wp) :: cqsav(NZLI)
      real(wp) :: u_ps(NZL), v_ps(NZL), t_ps(NZL), s_ps(NZL), c1_ps(NZLI)
      real(wp) :: n2p(NZLI), s2p(NZLI), n2c(NZLI), s2c(NZLI)
      real(wp) :: kappa_out(NZLI), kq_tmp(NZLI)
      real(wp) :: tke_pred(NZLI), kappa_pred(NZLI), kappa_mid(NZLI)
      real(wp) :: tke_fin(NZLI), kappa_pred2(NZLI)
      real(wp) :: local_src_avg(NZLI), kappa_src(NZLI), local_src(NZLI)

      integer :: k, kk, io
      real(wp) :: k0dt, tke_min, c_n2, c_s2, ilambda2
      real(wp) :: ome_l, eden1_l, eden2_l, i_eden_l
      real(wp) :: a1n, a1c, b1_l, d1_l, bd1_l, base_l, b1ns, b1in
      integer :: ks_src, ke_src, ks_kap, ke_kap
      real(wp) :: dsv_dt_k, dsv_ds_k, t_int, s_int, p_int, dpres
      real(wp) :: n2v, dt_rem, dt_now, dt_wt
      integer :: ks_ps, ke_ps, ks_mid, ke_mid
      logical :: no_mixing

#ifdef KS_COUNTERS
      n_out = 0
      n_in = 0
      n_it_l = 0
#endif
      k0dt = dt*kappa_0
      tke_min = max(tke_bg, 1.0e-20_wp)
      c_n2 = c_n*c_n
      c_s2 = c_s*c_s
      ilambda2 = 1.0_wp/(lambda*lambda)

      ! ---- Background-diffusion pre-step (kappa_0) ----
      if (nz == 1) then
         b1_l = 1.0_wp/(h_sd(1) + k0dt*idz_int_s(2))
         u_c(1) = b1_l*h_sd(1)*u_sd(1)
         v_c(1) = b1_l*h_sd(1)*v_sd(1)
         t_c(1) = t_sd(1)
         s_c(1) = s_sd(1)
      else
         a1n = k0dt*idz_int_s(2)
         b1_l = 1.0_wp/(h_sd(1) + a1n)
         cqsav(2) = a1n*b1_l
         d1_l = h_sd(1)*b1_l
         u_c(1) = b1_l*h_sd(1)*u_sd(1)
         v_c(1) = b1_l*h_sd(1)*v_sd(1)
         t_c(1) = b1_l*h_sd(1)*t_sd(1)
         s_c(1) = b1_l*h_sd(1)*s_sd(1)
         do k = 2, nz - 1
            a1c = a1n
            a1n = k0dt*idz_int_s(k + 1)
            bd1_l = h_sd(k) + d1_l*a1c
            b1_l = 1.0_wp/(bd1_l + a1n)
            u_c(k) = b1_l*(h_sd(k)*u_sd(k) + a1c*u_c(k - 1))
            v_c(k) = b1_l*(h_sd(k)*v_sd(k) + a1c*v_c(k - 1))
            t_c(k) = b1_l*(h_sd(k)*t_sd(k) + a1c*t_c(k - 1))
            s_c(k) = b1_l*(h_sd(k)*s_sd(k) + a1c*s_c(k - 1))
            cqsav(k + 1) = a1n*b1_l
            d1_l = bd1_l*b1_l
         end do
         a1c = a1n
         base_l = h_sd(nz) + d1_l*a1c
         b1ns = 1.0_wp/(base_l + k0dt*idz_int_s(nz + 1))
         b1in = 1.0_wp/base_l
         u_c(nz) = b1ns*(h_sd(nz)*u_sd(nz) + a1c*u_c(nz - 1))
         v_c(nz) = b1ns*(h_sd(nz)*v_sd(nz) + a1c*v_c(nz - 1))
         t_c(nz) = b1in*(h_sd(nz)*t_sd(nz) + a1c*t_c(nz - 1))
         s_c(nz) = b1in*(h_sd(nz)*s_sd(nz) + a1c*s_c(nz - 1))
         cqsav(nz + 1) = 0.0_wp
         do k = nz - 1, 1, -1
            u_c(k) = u_c(k) + cqsav(k + 1)*u_c(k + 1)
            v_c(k) = v_c(k) + cqsav(k + 1)*v_c(k + 1)
            t_c(k) = t_c(k) + cqsav(k + 1)*t_c(k + 1)
            s_c(k) = s_c(k) + cqsav(k + 1)*s_c(k + 1)
         end do
      end if

      ! ---- Frozen interface buoyancy derivatives ----
      dbuoy_t(1) = 0.0_wp
      dbuoy_s(1) = 0.0_wp
      dbuoy_t(nz + 1) = 0.0_wp
      dbuoy_s(nz + 1) = 0.0_wp
      p_int = 0.0_wp
      do kk = 2, nz
         dpres = GRAVITY*rho0*h_sd(kk - 1)
         p_int = p_int + dpres
         t_int = 0.5_wp*(t_c(kk - 1) + t_c(kk))
         s_int = 0.5_wp*(s_c(kk - 1) + s_c(kk))
         call eos_specvol_derivs(eos, t_int, s_int, p_int, &
                                 dsv_dt_k, dsv_ds_k)
         dbuoy_t(kk) = GRAVITY*rho0*dsv_dt_k
         dbuoy_s(kk) = GRAVITY*rho0*dsv_ds_k
      end do

      ! ---- Initial N^2, S^2 ----
      n2(1) = 0.0_wp
      n2(nz + 1) = 0.0_wp
      s2(1) = 0.0_wp
      s2(nz + 1) = 0.0_wp
      do kk = 2, nz
         n2v = idz_int_s(kk)*(dbuoy_t(kk)*(t_c(kk - 1) - t_c(kk)) + &
                              dbuoy_s(kk)*(s_c(kk - 1) - s_c(kk)))
         if (n2v < 0.0_wp) n2v = 0.0_wp
         n2(kk) = n2v
         s2(kk) = ((u_c(kk - 1) - u_c(kk))**2 + (v_c(kk - 1) - v_c(kk))**2)* &
                  idz_int_s(kk)**2
      end do

      ! ---- e1 tail recursion ----
      e1(nz + 1) = 0.0_wp
      ome_l = 1.0_wp
      eden2_l = kappa_0*idz_s(nz)
      do kk = nz, 2, -1
         eden1_l = hint_s(kk)*sqrt(c_n2*n2(kk) + c_s2*s2(kk)) + ome_l*eden2_l
         eden2_l = kappa_0*idz_s(kk - 1)
         i_eden_l = 1.0_wp/(eden2_l + eden1_l)
         e1(kk) = eden2_l*i_eden_l
         ome_l = eden1_l*i_eden_l
      end do
      e1(1) = 0.0_wp

      ! ---- Outer-loop init ----
      do kk = 1, nz + 1
         k_q(kk) = 0.0_wp
         kappa_avg(kk) = 0.0_wp
         tke_avg(kk) = 0.0_wp
         kappa(kk) = kappa_seed_in
      end do
      kappa(1) = 0.0_wp
      kappa(nz + 1) = 0.0_wp
      dt_rem = dt
      local_src_avg(1) = 0.0_wp
      local_src_avg(nz + 1) = 0.0_wp
      do kk = 2, nz
         if (hint_s(kk) > 0.0_wp) then
            local_src_avg(kk) = 0.1_wp*k0dt*idz_int_s(kk)/hint_s(kk)
         else
            local_src_avg(kk) = 0.0_wp
         end if
      end do

      ! ---- Adaptive outer substepping ----
      do io = 1, max_substep_it
#ifdef KS_COUNTERS
         n_out = n_out + 1
#endif
         ! Step 1: K_src + seed TKE from previous kappa/K_Q.
         ks_src = nz + 2
         ke_src = 0
         ksrc(1) = 0.0_wp
         ksrc(nz + 1) = 0.0_wp
         do kk = 2, nz
            ksrc(kk) = ks_src_func(ri_crit, shearmix_rate, fri_curvature, &
                                   n2(kk), s2(kk))
            if (ksrc(kk) > 0.0_wp) then
               if (ks_src > kk) ks_src = kk
               ke_src = kk
            end if
         end do
         do kk = 1, nz + 1
            kappa_src(kk) = ksrc(kk)
         end do

         do kk = 2, nz
            tkedec(kk) = sqrt(c_n2*n2(kk) + c_s2*s2(kk))
         end do
         tke(1) = tke_bg
         do kk = 2, nz
            if (kappa(kk) > 0.0_wp .and. k_q(kk) > 0.0_wp) then
               tke(kk) = kappa(kk)/k_q(kk)
            else
               tke(kk) = tke_min
            end if
         end do
         tke(nz + 1) = tke_min

         ! ⚠ WHOLE-ARRAY ASSIGNMENT OVER THE DECLARED EXTENT. `kappa_out` and
         ! `kappa` are declared (NZLI) = (NZ_STACK_MAX+1), so this copies 129
         ! doubles at the production NZ_STACK_MAX=128 while only 1..nz+1 = 31
         ! are ever read. This is the ONLY place the routine touches the full
         ! declared extent -- and therefore the entire reason wall time depends
         ! on NZ_STACK_MAX at all (measured: 2.3% over 32->128). Everything else
         ! is bounded by the runtime nz, which is why the Redi agent's
         ! NZ_STACK_MAX null result reproduces here.
#ifdef KS_BOUNDED_COPY
         do kk = 1, nz + 1
            kappa_out(kk) = kappa(kk)
         end do
#else
         kappa_out = kappa
#endif

         if (ks_src > ke_src) then
            do kk = 1, nz + 1
               kappa_out(kk) = 0.0_wp
            end do
            do kk = 2, nz
               ild2(kk) = 0.0_wp
            end do
         else
#ifdef KS_BOUNDED_COPY
            do kk = 1, nz + 1
               kq_tmp(kk) = k_q(kk)
            end do
#else
            kq_tmp = k_q
#endif
            call ks_find_kappa_tke(nz, tke_min, f2_val, ri_crit, &
                                   shearmix_rate, fri_curvature, c_n2, c_s2, &
                                   ilambda2, kappa_0, kappa_trunc, tke_bg, &
                                   tol_err, max_inner_it, n2, s2, kappa, &
                                   kq_tmp, idz_s, hint_s, il2_s, e1, tke, &
                                   kappa_out, ksrc, tkedec, aq, dq, cq, dk, &
                                   ck, ild2 &
#ifdef KS_COUNTERS
                                   , n_it_l &
#endif
                                   )
#ifdef KS_COUNTERS
            n_in = n_in + n_it_l
#endif
            do kk = 1, nz + 1
               k_q(kk) = kq_tmp(kk)
            end do
         end if

         ! local_src for the adaptive-dt bands.
         local_src(1) = 0.0_wp
         local_src(nz + 1) = 0.0_wp
         do kk = 2, nz
            if (hint_s(kk) > 0.0_wp) then
               local_src(kk) = ksrc(kk) + kappa_0* &
                               ((idz_s(kk - 1) + idz_s(kk))/ &
                                max(hint_s(kk), 1.0e-30_wp) + ild2(kk))
               n2v = idz_s(kk - 1)*(kappa_out(kk - 1) - kappa_out(kk)) + &
                     idz_s(kk)*(kappa_out(kk + 1) - kappa_out(kk))
               if (n2v > 0.0_wp) then
                  local_src(kk) = local_src(kk) + n2v/max(hint_s(kk), 1.0e-30_wp)
               end if
            else
               local_src(kk) = ksrc(kk)
            end if
         end do

         ! Step 2: active range of kappa_out.
         ks_kap = nz + 2
         ke_kap = 0
         do kk = 2, nz
            if (kappa_out(kk) > 0.0_wp) then
               if (ks_kap > kk) ks_kap = kk
               ke_kap = kk
            end if
         end do
         if (ke_kap == nz) kappa_out(nz + 1) = 0.0_wp
         no_mixing = (ke_kap < ks_kap)

         ! Step 3: choose dt_now.
         if (no_mixing .or. io == max_substep_it) then
            dt_now = dt_rem
         else
            dt_now = ks_adaptive_dt(nz, dt_rem, io, max_substep_it, ri_crit, &
                                    shearmix_rate, fri_curvature, src_max_chg, &
                                    tol_err, vel_underflow, dbuoy_t, dbuoy_s, &
                                    h_sd, u_c, v_c, t_c, s_c, kappa_out, &
                                    kappa_src, local_src, local_src_avg, &
                                    ks_kap, ke_kap, idz_int_s)
         end if
         do kk = 2, nz
            local_src_avg(kk) = local_src_avg(kk) + dt_now*local_src(kk)
         end do
         dt_wt = dt_now/dt

         if (no_mixing) then
            do kk = 1, nz + 1
               tke_avg(kk) = tke_avg(kk) + dt_wt*tke(kk)
            end do
            dt_rem = 0.0_wp
         else
            ! Predictor.
            ks_ps = max(ks_kap - 1, 1)
            ke_ps = min(ke_kap, nz)
            call ks_projected_state(nz, dt_now, ks_ps, ke_ps, vel_underflow, &
                                    dbuoy_t, dbuoy_s, h_sd, idz_int_s, u_c, &
                                    v_c, t_c, s_c, kappa_out, u_ps, v_ps, &
                                    t_ps, s_ps, c1_ps, n2p, s2p)
#ifdef KS_BOUNDED_COPY
            do kk = 1, nz + 1
               kq_tmp(kk) = k_q(kk)
            end do
#else
            kq_tmp = k_q
#endif
            call ks_find_kappa_tke(nz, tke_min, f2_val, ri_crit, &
                                   shearmix_rate, fri_curvature, c_n2, c_s2, &
                                   ilambda2, kappa_0, kappa_trunc, tke_bg, &
                                   tol_err, max_inner_it, n2p, s2p, kappa_out, &
                                   kq_tmp, idz_s, hint_s, il2_s, e1, tke_pred, &
                                   kappa_pred, ksrc, tkedec, aq, dq, cq, dk, &
                                   ck, ild2 &
#ifdef KS_COUNTERS
                                   , n_it_l &
#endif
                                   )
#ifdef KS_COUNTERS
            n_in = n_in + n_it_l
#endif
            do kk = 1, nz + 1
               kappa_mid(kk) = 0.5_wp*(kappa_out(kk) + kappa_pred(kk))
            end do
            ks_mid = nz + 2
            ke_mid = 0
            do kk = 1, nz + 1
               if (kappa_mid(kk) > 0.0_wp) then
                  if (ks_mid > kk) ks_mid = kk
                  ke_mid = kk
               end if
            end do
            ks_ps = max(ks_mid - 1, 1)
            ke_ps = min(ke_mid, nz)

            ! Corrector (real K_Q now).
            call ks_projected_state(nz, dt_now, ks_ps, ke_ps, vel_underflow, &
                                    dbuoy_t, dbuoy_s, h_sd, idz_int_s, u_c, &
                                    v_c, t_c, s_c, kappa_mid, u_ps, v_ps, &
                                    t_ps, s_ps, c1_ps, n2c, s2c)
            call ks_find_kappa_tke(nz, tke_min, f2_val, ri_crit, &
                                   shearmix_rate, fri_curvature, c_n2, c_s2, &
                                   ilambda2, kappa_0, kappa_trunc, tke_bg, &
                                   tol_err, max_inner_it, n2c, s2c, kappa_out, &
                                   k_q, idz_s, hint_s, il2_s, e1, tke_fin, &
                                   kappa_pred2, ksrc, tkedec, aq, dq, cq, dk, &
                                   ck, ild2 &
#ifdef KS_COUNTERS
                                   , n_it_l &
#endif
                                   )
#ifdef KS_COUNTERS
            n_in = n_in + n_it_l
#endif
            dt_rem = dt_rem - dt_now
            do kk = 1, nz + 1
               kappa_avg(kk) = kappa_avg(kk) + &
                               dt_wt*0.5_wp*(kappa_out(kk) + kappa_pred2(kk))
               tke_avg(kk) = tke_avg(kk) + &
                             dt_wt*0.5_wp*(tke_pred(kk) + tke_fin(kk))
               kappa(kk) = kappa_pred2(kk)
            end do
            kappa(1) = 0.0_wp
            kappa(nz + 1) = 0.0_wp

            ! Step 5: full-column real-state advance for the next substep.
            if (dt_rem > 0.0_wp) then
               do kk = 1, nz + 1
                  kappa_mid(kk) = 0.5_wp*(kappa_out(kk) + kappa_pred2(kk))
               end do
               call ks_projected_state(nz, dt_now, 1, nz, vel_underflow, &
                                       dbuoy_t, dbuoy_s, h_sd, idz_int_s, u_c, &
                                       v_c, t_c, s_c, kappa_mid, u_ps, v_ps, &
                                       t_ps, s_ps, c1_ps, n2, s2)
               do k = 1, nz
                  u_c(k) = u_ps(k)
                  v_c(k) = v_ps(k)
                  t_c(k) = t_ps(k)
                  s_c(k) = s_ps(k)
               end do
            end if
         end if

         if (dt_rem <= 0.0_wp) exit
      end do

      do kk = 1, nz + 1
         kappa_avg_sd(kk) = kappa_avg(kk)
         tke_avg_sd(kk) = tke_avg(kk)
      end do
      kappa_avg_sd(1) = 0.0_wp
      kappa_avg_sd(nz + 1) = 0.0_wp
   end subroutine ks_solve_column

end module ks
