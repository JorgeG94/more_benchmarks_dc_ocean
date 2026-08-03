#include "directives.h"
!! COLUMN-BLOCKED ("VLEN") arrangement of the kappa-shear (JHL08) column solve.
!!
!! Same physics, same arithmetic, same iteration counts as ks.F90 -- a different
!! ARRANGEMENT of the parallelism. ks.F90 is the shipped one:
!!
!!     do concurrent (j, i)                  ! one worker per COLUMN
!!        real :: h_sd(NZL)                  ! private, k-indexed
!!        <k-sequential Picard/substep solve>
!!
!! and this file is:
!!
!!     do concurrent (j, ib)                 ! one worker per BLOCK of VL columns
!!        real :: h_sd(VL, NZL)              ! private, LANE-MAJOR
!!        do k ...                           ! k stays sequential (it must)
!!           do l = 1, VL                    ! ... and THIS is the vector loop
!!
!! WHY. The k recursion is genuinely sequential -- every layer depends on the
!! one above -- so it cannot be parallelised or vectorised. In the shipped
!! arrangement that leaves NOTHING to vectorise: the innermost loops are over k,
!! which is the serial axis, and their trip count is nz. On a CPU that throws
!! away the whole vector unit. Blocking moves the vector axis onto the COLUMN
!! index, which is embarrassingly parallel, and leaves k where it has to be.
!! `h_sd(VL, NZL)` (lane-major, NOT `(NZL, VL)`) is what makes `h_sd(l, k)`
!! stride-1 across the vector loop.
!!
!! VL IS THE PORTABILITY KNOB, and its optimum is target-dependent:
!!   VL = 1      identical work to ks.F90, and REQUIRED to be bit-identical to
!!               it -- that is the gate this file is validated by.
!!   VL = 4/8    AVX2 / AVX-512 double-precision width.
!!   VL = 16     two AVX-512 registers; more ILP, more register pressure.
!! ⚠ ON A GPU, VL > 1 IS EXPECTED TO BE BAD AND THAT IS THE POINT. `local()`
!! privatises per DC ITERATION, so a GPU thread would own VL columns' worth of
!! frame: ~55 arrays x (NZL+1) x VL x 8 B, i.e. 225 KB/thread at VL=16,
!! NZSTACK=31 and past CUDA's per-thread local limit at VL=32, NZSTACK=128. The
!! sweep still BUILDS those points so the cost is measured rather than asserted.
!! Use the `fit` stack policy for GPU VLEN runs or they will not launch.
!!
!! ============================================================================
!! VERIFICATION STATUS + FIRST MEASURED RESULT
!! ============================================================================
!! CORRECTNESS (`make verify-variant DATA=none VLEN=<n>`), nvfortran 26.5:
!!   -O0                 max rel = 0.0 at VL = 1, 2, 4, 8   <- EXACT
!!   -O1 -Kieee -Mnofma  max rel = 0.0 at VL = 1, 2, 4, 8   <- EXACT
!!   -O2 / -O3 -fast     max rel ~ 1e-10, iteration counts IDENTICAL
!! The transformation is therefore provably right: with contraction off it
!! reproduces ks.F90 bit-for-bit at every lane width. The -O2+ residual is
!! codegen, not arrangement -- it tracks the optimiser (it even appears at VL=1
!! when the FAITHFUL reference is rebuilt at `-O2 -Mnovect`), and the INTEGER
!! Picard/substep counts, which rounding cannot touch, match at every width.
!! ~1e-10 is the same amplification README.md already reports for DATA=none vs
!! DATA=acc: a property of an iterative solver with convergence tests.
!!
!! PERFORMANCE -- BLOCKING NEVER BEATS THE SHIPPED ARRANGEMENT ON AVX2, BUT
!! THE VECTORISATION ITSELF WORKS. Xeon E5-2698 v4 (Broadwell, AVX2 = 4 doubles,
!! 20 cores / 40 threads), 64x64 = 4900 columns, -O3 -fast, `fit` stack policy.
!! Ratios are vs the faithful per-column kernel measured in the SAME run:
!!
!!   lane            nz   VL=1   VL=2   VL=4   VL=8   VL=16
!!   serial_do       30   0.61   0.78   0.78   0.71   0.63
!!   serial_do       75   0.66   0.93   0.98   0.80   0.75
!!   dc_serial       30   0.58   0.74   0.74   0.71   0.61
!!   dc_serial       75   0.63   0.88   0.92   0.79   0.70
!!   dc_multicore    30   0.56   0.61   0.59   0.57   0.53
!!   dc_multicore    75   0.59   0.65   0.64   0.56   0.46
!!
!! Read it as three separate facts, because they point different ways:
!!
!! 1. THE TRANSFORMATION COSTS ~1.7x, AND THAT IS THE FLOOR. VL=1 does identical
!!    arithmetic to ks.F90 with trip-count-1 lane loops, and lands at 0.56-0.66.
!!    That is rule 2/3 of the header: masked full-range loops instead of
!!    per-column early exits, so the block does max-over-lanes work.
!! 2. VECTORISATION IS REAL AND PEAKS AT THE HARDWARE WIDTH. VL=4 -- exactly
!!    AVX2's four doubles -- is the optimum in every serial lane, and VL=8/16
!!    regress. `-Minfo=vect` agrees: 103 vectorised loops at VL=8 vs 47 in the
!!    faithful build. It recovers much of (1) but never all of it.
!! 3. DEPTH HELPS, THREADS HURT. 0.78 -> 0.98 going nz=30 -> 75 (more k-work per
!!    column amortises the masking). But dc_multicore tops out at 0.65 where the
!!    single-threaded lanes reach 0.98 -- the blocked frame is VL times larger,
!!    and 40 threads competing for cache pay for that in a way one thread does
!!    not.
!!
!! ⚠ WHAT THIS PREDICTS FOR AVX-512 / SAPPHIRE RAPIDS, and why it is a
!! prediction and not a result: if the optimum tracks vector width, VL=8 should
!! win there. But VL=8 ALREADY REGRESSES here, which says the binding constraint
!! is the masking overhead and cache footprint of (1) and (3), not vector width.
!! So the honest expectation is that a 104-core SPR helps the single-threaded
!! lane and hurts the fully-threaded one. `tools/ks_sweep.sh vlen` settles it.
!!
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

#ifndef KS_VLEN
#define KS_VLEN 1
#endif
   integer, parameter :: VL = KS_VLEN
      !! lanes (columns) per block -- the vector width of the inner loops
   integer, parameter :: NZL = NZ_STACK_MAX
   integer, parameter :: NZLI = NZ_STACK_MAX + 1

   character(len=*), parameter :: KS_VARIANT = 'block'
   integer, parameter :: KS_LANES = VL

   !! Benign filler for lanes that are not a wet in-range column (rule 1).
   !! Zero shear and uniform T/S => N^2 = S^2 = 0 => K_src = 0 => no-mixing
   !! path on the first substep. Finite, cheap, and its output is discarded.
   real(wp), parameter :: FILL_H = 1.0_wp
   real(wp), parameter :: FILL_T = 10.0_wp
   real(wp), parameter :: FILL_S = 35.0_wp

   !! Knob defaults VERBATIM from ocean_kappa_shear.F90:64-145 -- identical to
   !! ks.F90's type, so the driver and the reference dumps are interchangeable.
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
      !! Per-BLOCK JHL08 solve. One `do concurrent (j, ib)` over blocks of VL
      !! columns; k stays sequential inside, lanes are the vector axis.
      type(hgrid_t), intent(in) :: grid
      type(ocean_kappa_shear_t), intent(inout) :: this
      type(multilayer_cgrid_state_t), intent(in) :: ms
      real(wp), intent(in) :: hT(:, :, :)
      real(wp), intent(in) :: hS(:, :, :)
      real(wp), intent(in) :: dt

      integer :: i, j, k, l, kg, ib, nblk, nx, ny, nz
      real(wp) :: f2_val(VL), hk, inv_h
      logical :: act(VL)
      real(wp) :: h_sd(VL, NZL), u_sd(VL, NZL), v_sd(VL, NZL)
      real(wp) :: t_sd(VL, NZL), s_sd(VL, NZL)
      real(wp) :: idz_s(VL, NZL), idz_int_s(VL, NZLI)
      real(wp) :: hint_s(VL, NZLI), il2_s(VL, NZLI)
      real(wp) :: kappa_avg_sd(VL, NZLI), tke_avg_sd(VL, NZLI)
#ifdef KS_COUNTERS
      integer :: n_out_l(VL), n_in_l(VL)
#endif

      nx = grid%nx_total
      ny = grid%ny_total
      nz = ms%nz_ml
      nblk = (nx + VL - 1)/VL

      do j=1,ny
      do ib=1,nblk

         ! ---- which lanes are real, in-range, wet columns ------------------
         do l = 1, VL
            i = (ib - 1)*VL + l
            act(l) = .false.
            if (i <= nx) then
               if (ms%wet_mask(i, j) > 0.0_wp) act(l) = .true.
            end if
         end do

         do k = 1, nz + 1
            do l = 1, VL
               kappa_avg_sd(l, k) = 0.0_wp
               tke_avg_sd(l, k) = 0.0_wp
            end do
         end do
#ifdef KS_COUNTERS
         do l = 1, VL
            n_out_l(l) = 0
            n_in_l(l) = 0
         end do
#endif

         ! ---- Gather with the index flip (local k=1 = surface). Inactive
         !      lanes take the benign filler (rule 1) so every expression
         !      below is unconditional and finite.
         do k = 1, nz
            do l = 1, VL
               i = (ib - 1)*VL + l
               kg = nz + 1 - k
               if (act(l)) then
                  h_sd(l, k) = ms%h_layer(i, j, kg)
                  u_sd(l, k) = 0.5_wp*(ms%u_face_x_layer(i, j, kg) + &
                                       ms%u_face_x_layer(i + 1, j, kg))
                  v_sd(l, k) = 0.5_wp*(ms%v_face_y_layer(i, j, kg) + &
                                       ms%v_face_y_layer(i, j + 1, kg))
               else
                  h_sd(l, k) = FILL_H
                  u_sd(l, k) = 0.0_wp
                  v_sd(l, k) = 0.0_wp
               end if
            end do
         end do

         do l = 1, VL
            i = (ib - 1)*VL + l
            if (act(l)) then
               f2_val(l) = this%f_centre(i, j)*this%f_centre(i, j)
            else
               f2_val(l) = 0.0_wp
            end if
         end do

         ! ---- Gather floor at H_VANISHED, solve on nz -----------------------
         do k = 1, nz
            do l = 1, VL
               i = (ib - 1)*VL + l
               kg = nz + 1 - k
               hk = max(h_sd(l, k), H_VANISHED)
               inv_h = 1.0_wp/hk
               h_sd(l, k) = hk
               if (act(l)) then
                  t_sd(l, k) = hT(i, j, kg)*inv_h
                  s_sd(l, k) = hS(i, j, kg)*inv_h
               else
                  t_sd(l, k) = FILL_T
                  s_sd(l, k) = FILL_S
               end if
            end do
         end do

         call ks_precompute(nz, this%lz_rescale, h_sd, &
                            idz_s, idz_int_s, hint_s, il2_s)

         call ks_solve_column(nz, dt, f2_val, this%rho0, &
                              this%ri_crit, this%shearmix_rate, &
                              this%fri_curvature, this%c_n, this%c_s, &
                              this%lambda, this%kappa_0, this%kappa_seed, &
                              this%kappa_trunc, this%tke_bg, this%tol_err, &
                              this%max_inner_it, this%max_substep_it, &
                              this%src_max_chg, this%vel_underflow, &
                              this%eos, &
                              h_sd, u_sd, v_sd, t_sd, s_sd, &
                              idz_s, idz_int_s, hint_s, il2_s, &
                              kappa_avg_sd, tke_avg_sd &
#ifdef KS_COUNTERS
                              , n_out_l, n_in_l &
#endif
                              )

         ! ---- Scatter with the flip; force exact 0 at bed + surface. THIS is
         !      the only masked output: inactive lanes never reach memory, and
         !      dry columns get the zeros ks.F90 would have left them.
         do k = 1, nz + 1
            do l = 1, VL
               i = (ib - 1)*VL + l
               kg = nz + 2 - k
               if (i <= nx) then
                  if (act(l)) then
                     this%kd_int(i, j, kg) = kappa_avg_sd(l, k)
                     this%tke_int(i, j, kg) = tke_avg_sd(l, k)
                  else
                     this%kd_int(i, j, kg) = 0.0_wp
                     this%tke_int(i, j, kg) = 0.0_wp
                  end if
               end if
            end do
         end do
         do l = 1, VL
            i = (ib - 1)*VL + l
            if (i <= nx) then
               this%kd_int(i, j, 1) = 0.0_wp
               this%kd_int(i, j, nz + 1) = 0.0_wp
               this%tke_int(i, j, 1) = 0.0_wp
               this%tke_int(i, j, nz + 1) = 0.0_wp
#ifdef KS_COUNTERS
               this%it_outer(i, j) = n_out_l(l)
               this%it_inner(i, j) = n_in_l(l)
#endif
            end if
         end do
      end do
      end do
   end subroutine kappa_shear_column_kernel

   ! =================================================================
   ! Column solve helpers -- arithmetic VERBATIM from ks.F90, control
   ! flow transformed by the four rules in the file header.
   ! =================================================================

   pure function ks_src_func(ri_crit, shearmix_rate, fri_curvature, n2, s2) &
      result(ksrc)
      !! Left scalar on purpose: it is tiny and branchy, and at -O3 both
      !! nvfortran and ifx inline it into the lane loop and turn the branch
      !! into a select. If a compiler ever refuses to, that shows up as a
      !! vectorisation report miss, not as wrong numbers.
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
      !! All bounds depend only on nz, which is uniform across the block, so
      !! this one is a pure loop interchange -- no masking anywhere.
      DC_ROUTINE_SEQ
      integer, intent(in) :: nz
      real(wp), intent(in) :: lz_rescale
      real(wp), intent(in) :: h_sd(VL, NZL)
      real(wp), intent(out) :: idz_o(VL, NZL), idz_int_o(VL, NZLI)
      real(wp), intent(out) :: hint_o(VL, NZLI), il2_o(VL, NZLI)

      integer :: k, l
      real(wp) :: hk, hkm1, hkp1, norm_l, wt_a, wt_b, i_lz2
      real(wp) :: dtop(VL), dbot(VL, NZLI)

      i_lz2 = 1.0_wp/(lz_rescale*lz_rescale)

      do k = 1, nz
         do l = 1, VL
            idz_o(l, k) = 1.0_wp/h_sd(l, k)
         end do
      end do

      do l = 1, VL
         idz_int_o(l, 1) = 2.0_wp/h_sd(l, 1)
      end do
      do k = 2, nz
         do l = 1, VL
            idz_int_o(l, k) = 2.0_wp/(h_sd(l, k - 1) + h_sd(l, k))
         end do
      end do
      do l = 1, VL
         idz_int_o(l, nz + 1) = 2.0_wp/h_sd(l, nz)
      end do

      do l = 1, VL
         hint_o(l, 1) = 0.0_wp
      end do
      if (nz >= 2) then
         do l = 1, VL
            hint_o(l, 2) = h_sd(l, 1)
         end do
         do k = 2, nz - 1
            do l = 1, VL
               hk = h_sd(l, k)
               hkm1 = h_sd(l, k - 1)
               hkp1 = h_sd(l, k + 1)
               norm_l = 1.0_wp/(hk*(hkm1 + hkp1) + 2.0_wp*hkm1*hkp1)
               wt_a = (hk + hkp1)*hkm1*norm_l
               wt_b = (hkm1 + hk)*hkp1*norm_l
               hint_o(l, k) = hint_o(l, k) + hk*wt_a
               hint_o(l, k + 1) = hk*wt_b
            end do
         end do
         do l = 1, VL
            hint_o(l, nz) = hint_o(l, nz) + h_sd(l, nz)
         end do
      end if
      do l = 1, VL
         hint_o(l, nz + 1) = 0.0_wp
      end do

      do l = 1, VL
         dbot(l, nz + 1) = 0.0_wp
      end do
      do k = nz, 1, -1
         do l = 1, VL
            dbot(l, k) = dbot(l, k + 1) + h_sd(l, k)
         end do
      end do
      do l = 1, VL
         il2_o(l, 1) = 0.0_wp
         il2_o(l, nz + 1) = 0.0_wp
         dtop(l) = 0.0_wp
      end do
      do k = 2, nz
         do l = 1, VL
            dtop(l) = dtop(l) + h_sd(l, k - 1)
            if (dtop(l) > 0.0_wp .and. dbot(l, k) > 0.0_wp) then
               il2_o(l, k) = i_lz2*(dtop(l) + dbot(l, k))**2/ &
                             ((dtop(l)*dbot(l, k))**2)
            else
               il2_o(l, k) = 0.0_wp
            end if
         end do
      end do
   end subroutine ks_precompute

   pure subroutine ks_projected_state(nz, dt_now, ks_a, ke_a, vel_underflow, &
                                      dbuoy_t, dbuoy_s, h_sd, idz_int_s, &
                                      u0, v0, t0, s0, kappa_ps, &
                                      u_o, v_o, t_o, s_o, c1_o, n2_o, s2_o)
      !! ks/ke and dt_now are PER LANE here (each column runs its own
      !! substepper). The tridiagonal sweep is therefore split into
      !!   phase 1  top layer,      indexed AT ks(l)          -- per-lane gather
      !!   phase 2  interior,       k over the block union    -- range-masked
      !!   phase 3  bottom layer,   indexed AT ke(l)          -- per-lane gather
      !!   phase 4  back-substitution, descending, range-masked
      !! The gathers are O(1) per lane, so only phases 2 and 4 need to vectorise.
      DC_ROUTINE_SEQ
      integer, intent(in) :: nz
      integer, intent(in) :: ks_a(VL), ke_a(VL)
      real(wp), intent(in) :: dt_now(VL)
      real(wp), intent(in) :: vel_underflow
      real(wp), intent(in) :: dbuoy_t(VL, NZLI), dbuoy_s(VL, NZLI)
      real(wp), intent(in) :: h_sd(VL, NZL), idz_int_s(VL, NZLI)
      real(wp), intent(in) :: u0(VL, NZL), v0(VL, NZL), t0(VL, NZL), s0(VL, NZL)
      real(wp), intent(in) :: kappa_ps(VL, NZLI)
      real(wp), intent(out) :: u_o(VL, NZL), v_o(VL, NZL)
      real(wp), intent(out) :: t_o(VL, NZL), s_o(VL, NZL)
      real(wp), intent(out) :: c1_o(VL, NZLI), n2_o(VL, NZLI), s2_o(VL, NZLI)

      integer :: k, kk, l, ksl, kel, klo, khi
      real(wp) :: a_b(VL), a_a(VL), b1(VL), d1(VL)
      real(wp) :: bd1, b1nz, n2v
      real(wp) :: ua, ub, va, vb, ta, tb, sa, sb
      logical :: doit(VL)

      do k = 1, nz
         do l = 1, VL
            u_o(l, k) = u0(l, k)
            v_o(l, k) = v0(l, k)
            t_o(l, k) = t0(l, k)
            s_o(l, k) = s0(l, k)
         end do
      end do
      do k = 1, nz + 1
         do l = 1, VL
            c1_o(l, k) = 0.0_wp
         end do
      end do

      do l = 1, VL
         doit(l) = (ks_a(l) <= ke_a(l) .and. dt_now(l) > 0.0_wp)
         a_b(l) = 0.0_wp
         a_a(l) = 0.0_wp
         b1(l) = 0.0_wp
         d1(l) = 0.0_wp
      end do

      ! ---- phase 1: top layer of each lane's band (k = ks(l)) --------------
      do l = 1, VL
         if (doit(l)) then
            ksl = ks_a(l)
            a_b(l) = dt_now(l)*kappa_ps(l, ksl + 1)*idz_int_s(l, ksl + 1)
            b1(l) = 1.0_wp/(h_sd(l, ksl) + a_b(l))
            c1_o(l, ksl + 1) = a_b(l)*b1(l)
            d1(l) = h_sd(l, ksl)*b1(l)
            u_o(l, ksl) = b1(l)*h_sd(l, ksl)*u0(l, ksl)
            v_o(l, ksl) = b1(l)*h_sd(l, ksl)*v0(l, ksl)
            t_o(l, ksl) = b1(l)*h_sd(l, ksl)*t0(l, ksl)
            s_o(l, ksl) = b1(l)*h_sd(l, ksl)*s0(l, ksl)
         end if
      end do

      ! ---- phase 2: interior, ks(l)+1 .. ke(l)-1 --------------------------
      klo = nz + 2
      khi = 0
      do l = 1, VL
         if (doit(l)) then
            klo = min(klo, ks_a(l) + 1)
            khi = max(khi, ke_a(l) - 1)
         end if
      end do
      do k = klo, khi
         do l = 1, VL
            if (doit(l) .and. k >= ks_a(l) + 1 .and. k <= ke_a(l) - 1) then
               a_a(l) = a_b(l)
               a_b(l) = dt_now(l)*kappa_ps(l, k + 1)*idz_int_s(l, k + 1)
               bd1 = h_sd(l, k) + d1(l)*a_a(l)
               b1(l) = 1.0_wp/(bd1 + a_b(l))
               c1_o(l, k + 1) = a_b(l)*b1(l)
               d1(l) = bd1*b1(l)
               u_o(l, k) = b1(l)*(h_sd(l, k)*u0(l, k) + a_a(l)*u_o(l, k - 1))
               v_o(l, k) = b1(l)*(h_sd(l, k)*v0(l, k) + a_a(l)*v_o(l, k - 1))
               t_o(l, k) = b1(l)*(h_sd(l, k)*t0(l, k) + a_a(l)*t_o(l, k - 1))
               s_o(l, k) = b1(l)*(h_sd(l, k)*s0(l, k) + a_a(l)*s_o(l, k - 1))
            end if
         end do
      end do

      ! ---- phase 3: bottom layer of the band (k = ke(l)) ------------------
      do l = 1, VL
         if (doit(l)) then
            ksl = ks_a(l)
            kel = ke_a(l)
            a_a(l) = a_b(l)
            if (kel > ksl) then
               b1(l) = 1.0_wp/(h_sd(l, kel) + d1(l)*a_a(l))
               t_o(l, kel) = b1(l)*(h_sd(l, kel)*t0(l, kel) + a_a(l)*t_o(l, kel - 1))
               s_o(l, kel) = b1(l)*(h_sd(l, kel)*s0(l, kel) + a_a(l)*s_o(l, kel - 1))
               if (kel == nz) then
                  b1nz = 1.0_wp/((h_sd(l, kel) + d1(l)*a_a(l)) + &
                                 dt_now(l)*kappa_ps(l, nz + 1)*idz_int_s(l, nz + 1))
               else
                  b1nz = b1(l)
               end if
               u_o(l, kel) = b1nz*(h_sd(l, kel)*u0(l, kel) + a_a(l)*u_o(l, kel - 1))
               v_o(l, kel) = b1nz*(h_sd(l, kel)*v0(l, kel) + a_a(l)*v_o(l, kel - 1))
            else
               b1(l) = 1.0_wp/(h_sd(l, kel) + a_a(l))
               t_o(l, kel) = b1(l)*h_sd(l, kel)*t0(l, kel)
               s_o(l, kel) = b1(l)*h_sd(l, kel)*s0(l, kel)
               if (kel == nz) then
                  b1nz = 1.0_wp/(h_sd(l, kel) + a_a(l) + &
                                 dt_now(l)*kappa_ps(l, nz + 1)*idz_int_s(l, nz + 1))
               else
                  b1nz = b1(l)
               end if
               u_o(l, kel) = b1nz*h_sd(l, kel)*u0(l, kel)
               v_o(l, kel) = b1nz*h_sd(l, kel)*v0(l, kel)
            end if
            if (abs(u_o(l, kel)) < vel_underflow) u_o(l, kel) = 0.0_wp
            if (abs(v_o(l, kel)) < vel_underflow) v_o(l, kel) = 0.0_wp
         end if
      end do

      ! ---- phase 4: back-substitution, ke(l)-1 down to ks(l) --------------
      klo = nz + 2
      khi = 0
      do l = 1, VL
         if (doit(l)) then
            klo = min(klo, ks_a(l))
            khi = max(khi, ke_a(l) - 1)
         end if
      end do
      do k = khi, klo, -1
         do l = 1, VL
            if (doit(l) .and. k >= ks_a(l) .and. k <= ke_a(l) - 1) then
               u_o(l, k) = u_o(l, k) + c1_o(l, k + 1)*u_o(l, k + 1)
               v_o(l, k) = v_o(l, k) + c1_o(l, k + 1)*v_o(l, k + 1)
               t_o(l, k) = t_o(l, k) + c1_o(l, k + 1)*t_o(l, k + 1)
               s_o(l, k) = s_o(l, k) + c1_o(l, k + 1)*s_o(l, k + 1)
               if (abs(u_o(l, k)) < vel_underflow) u_o(l, k) = 0.0_wp
               if (abs(v_o(l, k)) < vel_underflow) v_o(l, k) = 0.0_wp
            end if
         end do
      end do

      ! ---- the `else` arm of ks.F90's guard: lanes that did no sweep ------
      do k = 1, nz
         do l = 1, VL
            if (.not. doit(l)) then
               if (abs(u_o(l, k)) < vel_underflow) u_o(l, k) = 0.0_wp
               if (abs(v_o(l, k)) < vel_underflow) v_o(l, k) = 0.0_wp
            end if
         end do
      end do

      ! ---- N^2, S^2 -- mixed inside the band, original values outside -----
      do l = 1, VL
         n2_o(l, 1) = 0.0_wp
         n2_o(l, nz + 1) = 0.0_wp
         s2_o(l, 1) = 0.0_wp
         s2_o(l, nz + 1) = 0.0_wp
      end do
      do kk = 2, nz
         do l = 1, VL
            if (kk - 1 >= ks_a(l) .and. kk - 1 <= ke_a(l)) then
               ua = u_o(l, kk - 1)
               va = v_o(l, kk - 1)
               ta = t_o(l, kk - 1)
               sa = s_o(l, kk - 1)
            else
               ua = u0(l, kk - 1)
               va = v0(l, kk - 1)
               ta = t0(l, kk - 1)
               sa = s0(l, kk - 1)
            end if
            if (kk >= ks_a(l) .and. kk <= ke_a(l)) then
               ub = u_o(l, kk)
               vb = v_o(l, kk)
               tb = t_o(l, kk)
               sb = s_o(l, kk)
            else
               ub = u0(l, kk)
               vb = v0(l, kk)
               tb = t0(l, kk)
               sb = s0(l, kk)
            end if
            n2v = idz_int_s(l, kk)*(dbuoy_t(l, kk)*(ta - tb) + &
                                    dbuoy_s(l, kk)*(sa - sb))
            if (n2v < 0.0_wp) n2v = 0.0_wp
            n2_o(l, kk) = n2v
            s2_o(l, kk) = ((ua - ub)**2 + (va - vb)**2)*idz_int_s(l, kk)**2
         end do
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
                                     dk_sc, ck_sc, ild2_sc, skip &
#ifdef KS_COUNTERS
                                     , n_it &
#endif
                                     )
      !! The Picard solver. THE hard one: ks.F90 has a convergence `exit`, two
      !! truncation `exit`s inside k-loops, an early `return` for no-mixing
      !! columns, and four per-lane k ranges. Transformed by rules 2-4:
      !!   nomix(l)    lanes with no shear source at all -- outputs set, then
      !!               excluded from the iteration by pdone(l)
      !!   pdone(l)    converged; its state is held constant thereafter
      !!   stop_k(l)   frozen past its truncation exit inside a k-loop
      !! The loop runs max-over-lanes iterations; every lane's RESULT is what it
      !! would have got on its own.
      DC_ROUTINE_SEQ
      integer, intent(in) :: nz, max_inner_it
      real(wp), intent(in) :: tke_min
      real(wp), intent(in) :: f2_val(VL)
      real(wp), intent(in) :: ri_crit, shearmix_rate, fri_curvature
      real(wp), intent(in) :: c_n2, c_s2, ilambda2, kappa_0
      real(wp), intent(in) :: kappa_trunc, tke_bg, tol_err
      real(wp), intent(in) :: n2_in(VL, NZLI), s2_in(VL, NZLI)
      real(wp), intent(in) :: kappa_seed(VL, NZLI)
      real(wp), intent(inout) :: k_q_io(VL, NZLI)
      real(wp), intent(in) :: idz_s(VL, NZL), hint_s(VL, NZLI)
      real(wp), intent(in) :: il2_s(VL, NZLI), e1_s(VL, NZLI)
      !! INOUT, not OUT: a skipped lane must keep the value its caller had,
      !! because ks.F90 expresses "skip" by not making the call at all.
      real(wp), intent(inout) :: tke_o(VL, NZLI), kappa_o(VL, NZLI)
      real(wp), intent(inout) :: ksrc_sc(VL, NZLI), tkedec_sc(VL, NZLI)
      real(wp), intent(inout) :: aq_sc(VL, NZL), dq_sc(VL, NZLI), cq_sc(VL, NZLI)
      real(wp), intent(inout) :: dk_sc(VL, NZLI), ck_sc(VL, NZLI), ild2_sc(VL, NZLI)
      !! Lanes this call must not touch AT ALL. ks.F90 only calls this routine
      !! for mixing columns; a no-mixing column keeps its seeded `tke` and its
      !! `k_q`, and the early-return branch below would clobber both. So the
      !! caller passes the lanes it did not want called.
      logical, intent(in) :: skip(VL)
#ifdef KS_COUNTERS
      integer, intent(out) :: n_it(VL)
#endif

      integer :: kk, k, k2, it, l, kel, khi_l
      integer :: ks_src(VL), ke_src(VL)
      integer :: ks_kap(VL), ke_kap(VL), ks_kp(VL), ke_kp(VL), ke_tke(VL)
      integer :: ks_new(VL), ke_new(VL), k_lo(VL), k_hi(VL)
      integer :: kmax
      real(wp) :: cqc_l(VL), ckc_l(VL), tsrc_l, raw_l, dnom_l
      real(wp) :: bqd1_l, bq_l, bkd1_l, bk_l, trv, tr2v, lhs_l, rhs_l
      real(wp) :: knew
      logical :: nomix(VL), pdone(VL), stop_k(VL), conv(VL), allref
      logical :: tail_done(VL)

      do l = 1, VL
#ifdef KS_COUNTERS
         n_it(l) = 0
#endif
         ks_src(l) = nz + 2
         ke_src(l) = 0
         if (.not. skip(l)) then
            ksrc_sc(l, 1) = 0.0_wp
            ksrc_sc(l, nz + 1) = 0.0_wp
         end if
      end do
      do kk = 2, nz
         do l = 1, VL
            if (skip(l)) cycle
            ksrc_sc(l, kk) = ks_src_func(ri_crit, shearmix_rate, fri_curvature, &
                                         n2_in(l, kk), s2_in(l, kk))
            if (ksrc_sc(l, kk) > 0.0_wp) then
               if (ks_src(l) > kk) ks_src(l) = kk
               ke_src(l) = kk
            end if
         end do
      end do

      ! ks.F90 RETURNS here for a column with no shear source. Per lane that
      ! becomes: set the outputs, and never enter the iteration.
      do l = 1, VL
         nomix(l) = (.not. skip(l)) .and. (ks_src(l) > ke_src(l))
      end do
      do kk = 1, nz + 1
         do l = 1, VL
            if (nomix(l)) then
               tke_o(l, kk) = tke_min
               kappa_o(l, kk) = 0.0_wp
               k_q_io(l, kk) = 0.0_wp
            end if
         end do
      end do

      do kk = 2, nz
         do l = 1, VL
            if (.not. skip(l)) then
               tkedec_sc(l, kk) = sqrt(c_n2*n2_in(l, kk) + c_s2*s2_in(l, kk))
            end if
         end do
      end do

      do l = 1, VL
         if (.not. nomix(l) .and. .not. skip(l)) tke_o(l, 1) = tke_bg
      end do
      do kk = 2, nz
         do l = 1, VL
            if (.not. nomix(l) .and. .not. skip(l)) then
               if (kappa_seed(l, kk) > 0.0_wp .and. k_q_io(l, kk) > 0.0_wp) then
                  tke_o(l, kk) = kappa_seed(l, kk)/k_q_io(l, kk)
               else
                  tke_o(l, kk) = tke_min
               end if
            end if
         end do
      end do
      do l = 1, VL
         if (.not. nomix(l) .and. .not. skip(l)) tke_o(l, nz + 1) = tke_min
      end do

      do kk = 1, nz + 1
         do l = 1, VL
            if (.not. nomix(l) .and. .not. skip(l)) kappa_o(l, kk) = kappa_seed(l, kk)
         end do
      end do
      do l = 1, VL
         if (.not. nomix(l) .and. .not. skip(l)) then
            kappa_o(l, 1) = 0.0_wp
            kappa_o(l, nz + 1) = 0.0_wp
         end if
         ks_kap(l) = 2
         ke_kap(l) = nz
         ks_kp(l) = 2
         ke_kp(l) = nz
         pdone(l) = nomix(l) .or. skip(l)
      end do

      do it = 1, max_inner_it
         allref = .true.
         do l = 1, VL
            if (.not. pdone(l)) allref = .false.
         end do
         if (allref) exit
#ifdef KS_COUNTERS
         do l = 1, VL
            if (.not. pdone(l)) n_it(l) = n_it(l) + 1
         end do
#endif

         ! ---- (a) TKE tridiagonal sweep -------------------------------------
         do l = 1, VL
            ke_tke(l) = min(max(ke_kap(l), ke_kp(l)) + 1, nz + 1)
         end do
         kmax = 0
         do l = 1, VL
            if (.not. pdone(l)) kmax = max(kmax, min(ke_tke(l), nz))
         end do
         do k = 1, kmax
            do l = 1, VL
               if (.not. pdone(l) .and. k <= min(ke_tke(l), nz)) then
                  aq_sc(l, k) = (0.5_wp*(kappa_o(l, k) + kappa_o(l, k + 1)) + &
                                 kappa_0)*idz_s(l, k)
               end if
            end do
         end do

         do l = 1, VL
            if (.not. pdone(l)) then
               dq_sc(l, 1) = -tke_o(l, 1)
               tke_o(l, 1) = tke_bg
               cq_sc(l, 2) = 0.0_wp
               cqc_l(l) = 1.0_wp
            end if
         end do
         kmax = 0
         do l = 1, VL
            if (.not. pdone(l)) kmax = max(kmax, ke_tke(l) - 1)
         end do
         do kk = 2, kmax
            do l = 1, VL
               if (.not. pdone(l) .and. kk <= ke_tke(l) - 1) then
                  dq_sc(l, kk) = -tke_o(l, kk)
                  tsrc_l = (kappa_o(l, kk) + kappa_0)*s2_in(l, kk) + &
                           tke_bg*tkedec_sc(l, kk)
                  bqd1_l = hint_s(l, kk)*(tkedec_sc(l, kk) + &
                                          n2_in(l, kk)*k_q_io(l, kk)) + &
                           cqc_l(l)*aq_sc(l, kk - 1)
                  bq_l = 1.0_wp/(bqd1_l + aq_sc(l, kk))
                  tke_o(l, kk) = bq_l*(hint_s(l, kk)*tsrc_l + &
                                       aq_sc(l, kk - 1)*tke_o(l, kk - 1))
                  cq_sc(l, kk + 1) = aq_sc(l, kk)*bq_l
                  cqc_l(l) = bqd1_l*bq_l
               end if
            end do
         end do

         ! the `ke_tke == nz+1` / else split, per lane
         do l = 1, VL
            if (.not. pdone(l)) then
               if (ke_tke(l) == nz + 1) then
                  tke_o(l, nz + 1) = tke_min
                  dq_sc(l, nz + 1) = 0.0_wp
                  tail_done(l) = .true.
               else
                  kel = ke_tke(l)
                  tsrc_l = kappa_0*s2_in(l, kel) + tke_bg*tkedec_sc(l, kel)
                  bq_l = 1.0_wp/(hint_s(l, kel)*tkedec_sc(l, kel) + &
                                 cqc_l(l)*aq_sc(l, kel - 1) + aq_sc(l, kel))
                  cq_sc(l, kel + 1) = aq_sc(l, kel)*bq_l
                  dq_sc(l, kel) = -tke_o(l, kel)
                  raw_l = bq_l*(hint_s(l, kel)*tsrc_l + &
                                aq_sc(l, kel - 1)*tke_o(l, kel - 1))
                  dnom_l = 1.0_wp - cq_sc(l, kel + 1)*e1_s(l, kel + 1)
                  if (abs(dnom_l) > 1.0e-30_wp) then
                     tke_o(l, kel) = max((raw_l + cq_sc(l, kel + 1)* &
                                          (tke_o(l, kel + 1) - &
                                           e1_s(l, kel + 1)*tke_o(l, kel)))/dnom_l, &
                                         tke_min)
                  else
                     tke_o(l, kel) = max(raw_l, tke_min)
                  end if
                  dq_sc(l, kel) = tke_o(l, kel) + dq_sc(l, kel)
                  tail_done(l) = .false.
               end if
            else
               tail_done(l) = .true.
            end if
         end do
         ! the `do k2 = ke_tke+1, nz+1 ... exit` tail (rule 2)
         do k2 = 2, nz + 1
            do l = 1, VL
               if (.not. tail_done(l) .and. k2 >= ke_tke(l) + 1) then
                  dq_sc(l, k2) = e1_s(l, k2)*dq_sc(l, k2 - 1)
                  tke_o(l, k2) = max(tke_o(l, k2) + dq_sc(l, k2), tke_min)
                  if (abs(dq_sc(l, k2)) < 1.0e-16_wp*tke_o(l, k2)) tail_done(l) = .true.
               end if
            end do
         end do

         kmax = 0
         do l = 1, VL
            if (.not. pdone(l)) kmax = max(kmax, ke_tke(l) - 1)
         end do
         do kk = kmax, 1, -1
            do l = 1, VL
               if (.not. pdone(l) .and. kk <= ke_tke(l) - 1) then
                  tke_o(l, kk) = max(tke_o(l, kk) + cq_sc(l, kk + 1)*tke_o(l, kk + 1), &
                                     tke_min)
                  dq_sc(l, kk) = tke_o(l, kk) + dq_sc(l, kk)
               end if
            end do
         end do

         ! ---- (b) kappa sweep with truncation ramp + range track ------------
         do l = 1, VL
            if (.not. pdone(l)) then
               ks_kp(l) = ks_kap(l)
               ke_kp(l) = ke_kap(l)
            end if
         end do
         do kk = 2, nz
            do l = 1, VL
               if (.not. pdone(l)) then
                  if (tke_o(l, kk) > 0.0_wp) then
                     ild2_sc(l, kk) = (n2_in(l, kk)*ilambda2 + f2_val(l))/ &
                                      tke_o(l, kk) + il2_s(l, kk)
                  else
                     ild2_sc(l, kk) = 1.0e30_wp
                  end if
               end if
            end do
         end do

         do l = 1, VL
            if (.not. pdone(l)) then
               dk_sc(l, 1) = 0.0_wp
               ck_sc(l, 2) = 0.0_wp
               ckc_l(l) = 1.0_wp
               ke_new(l) = 0
               ks_new(l) = nz
               stop_k(l) = .false.
            else
               stop_k(l) = .true.
            end if
         end do
         do kk = 2, nz
            do l = 1, VL
               if (.not. stop_k(l)) then
                  dk_sc(l, kk) = -kappa_o(l, kk)
                  bkd1_l = hint_s(l, kk)*ild2_sc(l, kk) + ckc_l(l)*idz_s(l, kk - 1)
                  bk_l = 1.0_wp/(bkd1_l + idz_s(l, kk))
                  kappa_o(l, kk) = bk_l*(idz_s(l, kk - 1)*kappa_o(l, kk - 1) + &
                                         hint_s(l, kk)*ksrc_sc(l, kk))
                  ck_sc(l, kk + 1) = idz_s(l, kk)*bk_l
                  ckc_l(l) = bkd1_l*bk_l

                  trv = ckc_l(l)*kappa_trunc
                  tr2v = 2.0_wp*trv
                  if (kappa_o(l, kk) < trv) then
                     kappa_o(l, kk) = 0.0_wp
                     if (kk > ke_src(l)) then
                        ke_kap(l) = kk - 1
                        k_q_io(l, kk) = 0.0_wp
                        stop_k(l) = .true.      ! rule 2: freeze AFTER the body
                     end if
                  else if (kappa_o(l, kk) < tr2v) then
                     kappa_o(l, kk) = 2.0_wp*(kappa_o(l, kk) - trv)
                  end if
                  if (.not. stop_k(l)) ke_new(l) = kk
               end if
            end do
         end do
         do l = 1, VL
            if (.not. pdone(l)) then
               if (ke_new(l) > 0) ke_kap(l) = ke_new(l)
               kel = ke_kap(l)
               if (kel >= 1 .and. tke_o(l, kel) > 0.0_wp) then
                  k_q_io(l, kel) = kappa_o(l, kel)/tke_o(l, kel)
               end if
               dk_sc(l, kel) = dk_sc(l, kel) + kappa_o(l, kel)
            end if
         end do

         do kk = 2, nz
            do l = 1, VL
               if (.not. pdone(l) .and. kk >= ke_kap(l) + 2 .and. &
                   kk <= ke_kp(l) + 1) then
                  dk_sc(l, kk) = -kappa_o(l, kk)
                  kappa_o(l, kk) = 0.0_wp
                  k_q_io(l, kk) = 0.0_wp
               end if
            end do
         end do

         ! back-substitution with its own truncation `exit` (rule 2)
         do l = 1, VL
            if (.not. pdone(l)) then
               ks_new(l) = 2
               stop_k(l) = .false.
            else
               stop_k(l) = .true.
            end if
         end do
         khi_l = 0
         do l = 1, VL
            if (.not. pdone(l)) khi_l = max(khi_l, ke_kap(l) - 1)
         end do
         do kk = khi_l, 2, -1
            do l = 1, VL
               if (.not. stop_k(l) .and. kk <= ke_kap(l) - 1) then
                  kappa_o(l, kk) = kappa_o(l, kk) + ck_sc(l, kk + 1)*kappa_o(l, kk + 1)
                  if (kappa_o(l, kk) <= kappa_trunc) then
                     kappa_o(l, kk) = 0.0_wp
                     if (kk < ks_src(l)) then
                        ks_kap(l) = kk + 1
                        k_q_io(l, kk) = 0.0_wp
                        stop_k(l) = .true.
                     end if
                  else if (kappa_o(l, kk) < 2.0_wp*kappa_trunc) then
                     kappa_o(l, kk) = 2.0_wp*(kappa_o(l, kk) - kappa_trunc)
                  end if
                  if (.not. stop_k(l)) then
                     dk_sc(l, kk) = dk_sc(l, kk) + kappa_o(l, kk)
                     if (tke_o(l, kk) > 0.0_wp) then
                        k_q_io(l, kk) = kappa_o(l, kk)/tke_o(l, kk)
                     else
                        k_q_io(l, kk) = 0.0_wp
                     end if
                     ks_new(l) = kk
                  end if
               end if
            end do
         end do
         do l = 1, VL
            if (.not. pdone(l)) ks_kap(l) = max(ks_new(l), 2)
         end do

         do kk = 2, nz
            do l = 1, VL
               if (.not. pdone(l) .and. kk >= ks_kp(l) .and. &
                   kk <= ks_kap(l) - 2) then
                  kappa_o(l, kk) = 0.0_wp
                  k_q_io(l, kk) = 0.0_wp
               end if
            end do
         end do

         ! ---- (c) Picard convergence test -----------------------------------
         ! ks.F90 `exit`s on the first violation; ANDing over the whole range
         ! gives the same logical and has no side effects.
         do l = 1, VL
            k_lo(l) = min(ks_kap(l), ks_kp(l))
            k_hi(l) = max(ke_kap(l), ke_kp(l))
            conv(l) = .true.
         end do
         do kk = 2, nz + 1
            do l = 1, VL
               if (.not. pdone(l) .and. kk >= k_lo(l) .and. kk <= k_hi(l)) then
                  lhs_l = abs(dk_sc(l, kk))
                  rhs_l = tol_err*(kappa_0 + kappa_o(l, kk) - 0.5_wp*dk_sc(l, kk))
                  if (lhs_l > rhs_l) conv(l) = .false.
               end if
            end do
         end do
         do l = 1, VL
            if (.not. pdone(l) .and. conv(l)) pdone(l) = .true.
         end do
      end do

      do l = 1, VL
         if (.not. nomix(l) .and. .not. skip(l)) then
            kappa_o(l, 1) = 0.0_wp
            kappa_o(l, nz + 1) = 0.0_wp
         end if
      end do
      ! keep `knew` referenced so an unused-variable warning does not mask a
      ! real one during future edits
      knew = 0.0_wp
   end subroutine ks_find_kappa_tke

   pure subroutine ks_adaptive_dt(nz, dt_rem, itt_outer, max_substep_it, &
                                  ri_crit, shearmix_rate, fri_curvature, &
                                  src_max_chg, tol_err, vel_underflow, &
                                  dbuoy_t, dbuoy_s, &
                                  h_s, u_cur, v_cur, t_cur, s_cur, &
                                  kappa_out_s, kappa_src_s, local_src_s, &
                                  local_src_avg_s, ks_kap, ke_kap, &
                                  idz_int_s, active, dt_now_r)
      !! A function in ks.F90; a subroutine here because it returns a per-lane
      !! vector. `max_halvings` is per lane (it depends on the lane's substep
      !! index), so the halving loop runs the block-wide maximum and each lane
      !! stops at its own bound. `active` excludes lanes that took the
      !! no-mixing shortcut or have already finished their substepping.
      DC_ROUTINE_SEQ
      integer, intent(in) :: nz, itt_outer, max_substep_it
      integer, intent(in) :: ks_kap(VL), ke_kap(VL)
      real(wp), intent(in) :: dt_rem(VL)
      real(wp), intent(in) :: ri_crit, shearmix_rate, fri_curvature
      real(wp), intent(in) :: src_max_chg, tol_err, vel_underflow
      real(wp), intent(in) :: dbuoy_t(VL, NZLI), dbuoy_s(VL, NZLI)
      real(wp), intent(in) :: h_s(VL, NZL), u_cur(VL, NZL), v_cur(VL, NZL)
      real(wp), intent(in) :: t_cur(VL, NZL), s_cur(VL, NZL)
      real(wp), intent(in) :: kappa_out_s(VL, NZLI), kappa_src_s(VL, NZLI)
      real(wp), intent(in) :: local_src_s(VL, NZLI), local_src_avg_s(VL, NZLI)
      real(wp), intent(in) :: idz_int_s(VL, NZLI)
      logical, intent(in) :: active(VL)
      real(wp), intent(out) :: dt_now_r(VL)

      real(wp) :: tol_max(VL, NZLI), tol_min_a(VL, NZLI), tol_chg_a(VL, NZLI)
      real(wp) :: u_pr(VL, NZL), v_pr(VL, NZL), t_pr(VL, NZL), s_pr(VL, NZL)
      real(wp) :: c1_pr(VL, NZLI), n2_pr(VL, NZLI), s2_pr(VL, NZLI)
      real(wp) :: dt_half(VL)

      integer :: kk, l, ih, ir, nh_max
      integer :: k_lo(VL), k_hi(VL), ks_lyr(VL), ke_lyr(VL), max_halvings(VL)
      real(wp) :: dt_tst(VL), dt_inc(VL), dt_try(VL), idtt
      real(wp) :: ksrc_tst, upper_t, lower_t, tol2_l, tol_dksrc_low
      logical :: valid_dt(VL), valid_try(VL), hdone(VL)

      if (src_max_chg == 10.0_wp) then
         tol_dksrc_low = 0.95_wp
      else
         tol_dksrc_low = (src_max_chg - 0.5_wp)/src_max_chg
      end if
      tol2_l = 2.0_wp*tol_err

      do kk = 1, nz + 1
         do l = 1, VL
            tol_max(l, kk) = kappa_src_s(l, kk) + src_max_chg*local_src_s(l, kk)
            tol_min_a(l, kk) = kappa_src_s(l, kk) - tol_dksrc_low*local_src_s(l, kk)
            tol_chg_a(l, kk) = tol2_l*local_src_avg_s(l, kk)
         end do
      end do

      nh_max = 1
      do l = 1, VL
         k_lo(l) = max(ks_kap(l) - 1, 2)
         k_hi(l) = min(ke_kap(l) + 1, nz)
         ks_lyr(l) = max(ks_kap(l) - 1, 1)
         ke_lyr(l) = min(ke_kap(l), nz)
         dt_tst(l) = dt_rem(l)
         valid_dt(l) = .false.
         max_halvings(l) = (max_substep_it + 1 - itt_outer)/2
         hdone(l) = .not. active(l)
         if (active(l)) nh_max = max(nh_max, max(max_halvings(l), 1))
         dt_now_r(l) = dt_rem(l)
      end do

      ! ---- halving pass ---------------------------------------------------
      do ih = 1, nh_max
         do l = 1, VL
            dt_half(l) = 0.5_wp*dt_tst(l)
         end do
         call ks_projected_state(nz, dt_half, ks_lyr, ke_lyr, &
                                 vel_underflow, dbuoy_t, dbuoy_s, h_s, &
                                 idz_int_s, u_cur, v_cur, t_cur, s_cur, &
                                 kappa_out_s, u_pr, v_pr, t_pr, s_pr, &
                                 c1_pr, n2_pr, s2_pr)
         do l = 1, VL
            if (.not. hdone(l) .and. ih <= max(max_halvings(l), 1)) valid_dt(l) = .true.
         end do
         do kk = 2, nz
            do l = 1, VL
               if (.not. hdone(l) .and. ih <= max(max_halvings(l), 1) .and. &
                   kk >= k_lo(l) .and. kk <= k_hi(l) .and. valid_dt(l)) then
                  idtt = 0.0_wp
                  if (dt_tst(l) > 0.0_wp) idtt = 1.0_wp/dt_tst(l)
                  if (n2_pr(l, kk) < ri_crit*s2_pr(l, kk)) then
                     ksrc_tst = ks_src_func(ri_crit, shearmix_rate, fri_curvature, &
                                            n2_pr(l, kk), s2_pr(l, kk))
                     upper_t = max(tol_max(l, kk), kappa_src_s(l, kk) + idtt*tol_chg_a(l, kk))
                     lower_t = min(tol_min_a(l, kk), kappa_src_s(l, kk) - idtt*tol_chg_a(l, kk))
                     if (ksrc_tst > upper_t .or. ksrc_tst < lower_t) valid_dt(l) = .false.
                  else
                     lower_t = min(tol_min_a(l, kk), kappa_src_s(l, kk) - idtt*tol_chg_a(l, kk))
                     if (0.0_wp < lower_t) valid_dt(l) = .false.
                  end if
               end if
            end do
         end do
         do l = 1, VL
            if (.not. hdone(l) .and. ih <= max(max_halvings(l), 1)) then
               if (valid_dt(l)) then
                  hdone(l) = .true.
               else
                  dt_tst(l) = 0.5_wp*dt_tst(l)
               end if
            end if
         end do
      end do

      ! ---- refinement pass (always 5 iterations in ks.F90) ----------------
      do l = 1, VL
         dt_inc(l) = 0.0_wp
         if (active(l) .and. dt_tst(l) < dt_rem(l) .and. valid_dt(l)) then
            dt_inc(l) = 0.5_wp*dt_tst(l)
         end if
      end do
      do ir = 1, 5
         do l = 1, VL
            dt_try(l) = dt_tst(l) + dt_inc(l)
            dt_half(l) = 0.5_wp*dt_try(l)
         end do
         call ks_projected_state(nz, dt_half, ks_lyr, ke_lyr, &
                                 vel_underflow, dbuoy_t, dbuoy_s, h_s, &
                                 idz_int_s, u_cur, v_cur, t_cur, s_cur, &
                                 kappa_out_s, u_pr, v_pr, t_pr, s_pr, &
                                 c1_pr, n2_pr, s2_pr)
         do l = 1, VL
            valid_try(l) = (dt_inc(l) > 0.0_wp)
         end do
         do kk = 2, nz
            do l = 1, VL
               if (valid_try(l) .and. kk >= k_lo(l) .and. kk <= k_hi(l)) then
                  idtt = 0.0_wp
                  if (dt_try(l) > 0.0_wp) idtt = 1.0_wp/dt_try(l)
                  if (n2_pr(l, kk) < ri_crit*s2_pr(l, kk)) then
                     ksrc_tst = ks_src_func(ri_crit, shearmix_rate, fri_curvature, &
                                            n2_pr(l, kk), s2_pr(l, kk))
                     upper_t = max(tol_max(l, kk), kappa_src_s(l, kk) + idtt*tol_chg_a(l, kk))
                     lower_t = min(tol_min_a(l, kk), kappa_src_s(l, kk) - idtt*tol_chg_a(l, kk))
                     if (ksrc_tst > upper_t .or. ksrc_tst < lower_t) valid_try(l) = .false.
                  else
                     lower_t = min(tol_min_a(l, kk), kappa_src_s(l, kk) - idtt*tol_chg_a(l, kk))
                     if (0.0_wp < lower_t) valid_try(l) = .false.
                  end if
               end if
            end do
         end do
         do l = 1, VL
            if (dt_inc(l) > 0.0_wp) then
               if (valid_try(l)) dt_tst(l) = dt_try(l)
               dt_inc(l) = 0.5_wp*dt_inc(l)
            end if
         end do
      end do

      do l = 1, VL
         if (active(l)) then
            dt_now_r(l) = min(dt_tst(l)*(1.0_wp + tol_err) + dt_inc(l), dt_rem(l))
         end if
      end do
   end subroutine ks_adaptive_dt

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
      !! The outer adaptive substepper. `dt_rem` is per lane, so ks.F90's
      !! `if (dt_rem <= 0) exit` becomes a per-lane `sdone(l)` and the loop runs
      !! until every lane is finished (rule 3). Note the frame: every array is
      !! (VL, NZL/NZLI), so the private frame is VL times ks.F90's -- the reason
      !! VL > 1 is a CPU arrangement, not a GPU one.
      DC_ROUTINE_SEQ
      type(ocean_eos_t), intent(in) :: eos
      integer, intent(in) :: nz, max_inner_it, max_substep_it
      real(wp), intent(in) :: dt, rho0
      real(wp), intent(in) :: f2_val(VL)
      real(wp), intent(in) :: ri_crit, shearmix_rate, fri_curvature
      real(wp), intent(in) :: c_n, c_s, lambda, kappa_0, kappa_seed_in
      real(wp), intent(in) :: kappa_trunc, tke_bg, tol_err, src_max_chg
      real(wp), intent(in) :: vel_underflow
      real(wp), intent(in) :: h_sd(VL, NZL), u_sd(VL, NZL), v_sd(VL, NZL)
      real(wp), intent(in) :: t_sd(VL, NZL), s_sd(VL, NZL)
      real(wp), intent(in) :: idz_s(VL, NZL), idz_int_s(VL, NZLI)
      real(wp), intent(in) :: hint_s(VL, NZLI), il2_s(VL, NZLI)
      real(wp), intent(out) :: kappa_avg_sd(VL, NZLI), tke_avg_sd(VL, NZLI)
#ifdef KS_COUNTERS
      integer, intent(out) :: n_out(VL), n_in(VL)
      integer :: n_it_l(VL)
#endif

      real(wp) :: u_c(VL, NZL), v_c(VL, NZL), t_c(VL, NZL), s_c(VL, NZL)
      real(wp) :: dbuoy_t(VL, NZLI), dbuoy_s(VL, NZLI)
      real(wp) :: e1(VL, NZLI)
      real(wp) :: kappa(VL, NZLI), k_q(VL, NZLI)
      real(wp) :: kappa_avg(VL, NZLI), tke_avg(VL, NZLI)
      real(wp) :: n2(VL, NZLI), s2(VL, NZLI), tke(VL, NZLI)
      real(wp) :: ksrc(VL, NZLI), tkedec(VL, NZLI)
      real(wp) :: aq(VL, NZL), dq(VL, NZLI), cq(VL, NZLI)
      real(wp) :: dk(VL, NZLI), ck(VL, NZLI), ild2(VL, NZLI)
      real(wp) :: cqsav(VL, NZLI)
      real(wp) :: u_ps(VL, NZL), v_ps(VL, NZL), t_ps(VL, NZL), s_ps(VL, NZL)
      real(wp) :: c1_ps(VL, NZLI)
      real(wp) :: n2p(VL, NZLI), s2p(VL, NZLI), n2c(VL, NZLI), s2c(VL, NZLI)
      real(wp) :: kappa_out(VL, NZLI), kq_tmp(VL, NZLI)
      real(wp) :: tke_pred(VL, NZLI), kappa_pred(VL, NZLI), kappa_mid(VL, NZLI)
      real(wp) :: tke_fin(VL, NZLI), kappa_pred2(VL, NZLI)
      real(wp) :: local_src_avg(VL, NZLI), kappa_src(VL, NZLI), local_src(VL, NZLI)
      real(wp) :: dt_rem(VL), dt_now(VL), dt_wt(VL)
      integer :: ks_ps(VL), ke_ps(VL), ks_lyr(VL), ke_lyr(VL)

      integer :: k, kk, l, io, ctop
      real(wp) :: k0dt, tke_min, c_n2, c_s2, ilambda2
      real(wp) :: ome_l(VL), eden1_l, eden2_l(VL), i_eden_l
      real(wp) :: a1n(VL), a1c(VL), b1_l(VL), d1_l(VL), bd1_l, base_l
      real(wp) :: b1ns, b1in
      integer :: ks_src(VL), ke_src(VL), ks_kap(VL), ke_kap(VL)
      integer :: ks_mid(VL), ke_mid(VL)
      real(wp) :: dsv_dt_k, dsv_ds_k, t_int, s_int, p_int(VL), dpres
      real(wp) :: n2v
      logical :: no_mixing(VL), sdone(VL), allref, mixing_step(VL)
      logical :: skip_step1(VL), skip_mix(VL)

#ifdef KS_COUNTERS
      do l = 1, VL
         n_out(l) = 0
         n_in(l) = 0
         n_it_l(l) = 0
      end do
#endif
      k0dt = dt*kappa_0
      tke_min = max(tke_bg, 1.0e-20_wp)
      c_n2 = c_n*c_n
      c_s2 = c_s*c_s
      ilambda2 = 1.0_wp/(lambda*lambda)

      ! ---- Background-diffusion pre-step (kappa_0). nz is uniform, so the
      !      nz == 1 special case is a plain branch, not a mask.
      if (nz == 1) then
         do l = 1, VL
            b1_l(l) = 1.0_wp/(h_sd(l, 1) + k0dt*idz_int_s(l, 2))
            u_c(l, 1) = b1_l(l)*h_sd(l, 1)*u_sd(l, 1)
            v_c(l, 1) = b1_l(l)*h_sd(l, 1)*v_sd(l, 1)
            t_c(l, 1) = t_sd(l, 1)
            s_c(l, 1) = s_sd(l, 1)
         end do
      else
         do l = 1, VL
            a1n(l) = k0dt*idz_int_s(l, 2)
            b1_l(l) = 1.0_wp/(h_sd(l, 1) + a1n(l))
            cqsav(l, 2) = a1n(l)*b1_l(l)
            d1_l(l) = h_sd(l, 1)*b1_l(l)
            u_c(l, 1) = b1_l(l)*h_sd(l, 1)*u_sd(l, 1)
            v_c(l, 1) = b1_l(l)*h_sd(l, 1)*v_sd(l, 1)
            t_c(l, 1) = b1_l(l)*h_sd(l, 1)*t_sd(l, 1)
            s_c(l, 1) = b1_l(l)*h_sd(l, 1)*s_sd(l, 1)
         end do
         do k = 2, nz - 1
            do l = 1, VL
               a1c(l) = a1n(l)
               a1n(l) = k0dt*idz_int_s(l, k + 1)
               bd1_l = h_sd(l, k) + d1_l(l)*a1c(l)
               b1_l(l) = 1.0_wp/(bd1_l + a1n(l))
               u_c(l, k) = b1_l(l)*(h_sd(l, k)*u_sd(l, k) + a1c(l)*u_c(l, k - 1))
               v_c(l, k) = b1_l(l)*(h_sd(l, k)*v_sd(l, k) + a1c(l)*v_c(l, k - 1))
               t_c(l, k) = b1_l(l)*(h_sd(l, k)*t_sd(l, k) + a1c(l)*t_c(l, k - 1))
               s_c(l, k) = b1_l(l)*(h_sd(l, k)*s_sd(l, k) + a1c(l)*s_c(l, k - 1))
               cqsav(l, k + 1) = a1n(l)*b1_l(l)
               d1_l(l) = bd1_l*b1_l(l)
            end do
         end do
         do l = 1, VL
            a1c(l) = a1n(l)
            base_l = h_sd(l, nz) + d1_l(l)*a1c(l)
            b1ns = 1.0_wp/(base_l + k0dt*idz_int_s(l, nz + 1))
            b1in = 1.0_wp/base_l
            u_c(l, nz) = b1ns*(h_sd(l, nz)*u_sd(l, nz) + a1c(l)*u_c(l, nz - 1))
            v_c(l, nz) = b1ns*(h_sd(l, nz)*v_sd(l, nz) + a1c(l)*v_c(l, nz - 1))
            t_c(l, nz) = b1in*(h_sd(l, nz)*t_sd(l, nz) + a1c(l)*t_c(l, nz - 1))
            s_c(l, nz) = b1in*(h_sd(l, nz)*s_sd(l, nz) + a1c(l)*s_c(l, nz - 1))
            cqsav(l, nz + 1) = 0.0_wp
         end do
         do k = nz - 1, 1, -1
            do l = 1, VL
               u_c(l, k) = u_c(l, k) + cqsav(l, k + 1)*u_c(l, k + 1)
               v_c(l, k) = v_c(l, k) + cqsav(l, k + 1)*v_c(l, k + 1)
               t_c(l, k) = t_c(l, k) + cqsav(l, k + 1)*t_c(l, k + 1)
               s_c(l, k) = s_c(l, k) + cqsav(l, k + 1)*s_c(l, k + 1)
            end do
         end do
      end if

      ! ---- Frozen interface buoyancy derivatives --------------------------
      do l = 1, VL
         dbuoy_t(l, 1) = 0.0_wp
         dbuoy_s(l, 1) = 0.0_wp
         dbuoy_t(l, nz + 1) = 0.0_wp
         dbuoy_s(l, nz + 1) = 0.0_wp
         p_int(l) = 0.0_wp
      end do
      do kk = 2, nz
         do l = 1, VL
            dpres = GRAVITY*rho0*h_sd(l, kk - 1)
            p_int(l) = p_int(l) + dpres
            t_int = 0.5_wp*(t_c(l, kk - 1) + t_c(l, kk))
            s_int = 0.5_wp*(s_c(l, kk - 1) + s_c(l, kk))
            call eos_specvol_derivs(eos, t_int, s_int, p_int(l), &
                                    dsv_dt_k, dsv_ds_k)
            dbuoy_t(l, kk) = GRAVITY*rho0*dsv_dt_k
            dbuoy_s(l, kk) = GRAVITY*rho0*dsv_ds_k
         end do
      end do

      ! ---- Initial N^2, S^2 ----------------------------------------------
      do l = 1, VL
         n2(l, 1) = 0.0_wp
         n2(l, nz + 1) = 0.0_wp
         s2(l, 1) = 0.0_wp
         s2(l, nz + 1) = 0.0_wp
      end do
      do kk = 2, nz
         do l = 1, VL
            n2v = idz_int_s(l, kk)*(dbuoy_t(l, kk)*(t_c(l, kk - 1) - t_c(l, kk)) + &
                                    dbuoy_s(l, kk)*(s_c(l, kk - 1) - s_c(l, kk)))
            if (n2v < 0.0_wp) n2v = 0.0_wp
            n2(l, kk) = n2v
            s2(l, kk) = ((u_c(l, kk - 1) - u_c(l, kk))**2 + &
                         (v_c(l, kk - 1) - v_c(l, kk))**2)*idz_int_s(l, kk)**2
         end do
      end do

      ! ---- e1 tail recursion ---------------------------------------------
      do l = 1, VL
         e1(l, nz + 1) = 0.0_wp
         ome_l(l) = 1.0_wp
         eden2_l(l) = kappa_0*idz_s(l, nz)
      end do
      do kk = nz, 2, -1
         do l = 1, VL
            eden1_l = hint_s(l, kk)*sqrt(c_n2*n2(l, kk) + c_s2*s2(l, kk)) + &
                      ome_l(l)*eden2_l(l)
            eden2_l(l) = kappa_0*idz_s(l, kk - 1)
            i_eden_l = 1.0_wp/(eden2_l(l) + eden1_l)
            e1(l, kk) = eden2_l(l)*i_eden_l
            ome_l(l) = eden1_l*i_eden_l
         end do
      end do
      do l = 1, VL
         e1(l, 1) = 0.0_wp
      end do

      ! ---- Outer-loop init -----------------------------------------------
      do kk = 1, nz + 1
         do l = 1, VL
            k_q(l, kk) = 0.0_wp
            kappa_avg(l, kk) = 0.0_wp
            tke_avg(l, kk) = 0.0_wp
            kappa(l, kk) = kappa_seed_in
         end do
      end do
      do l = 1, VL
         kappa(l, 1) = 0.0_wp
         kappa(l, nz + 1) = 0.0_wp
         dt_rem(l) = dt
         sdone(l) = .false.
         local_src_avg(l, 1) = 0.0_wp
         local_src_avg(l, nz + 1) = 0.0_wp
      end do
      do kk = 2, nz
         do l = 1, VL
            if (hint_s(l, kk) > 0.0_wp) then
               local_src_avg(l, kk) = 0.1_wp*k0dt*idz_int_s(l, kk)/hint_s(l, kk)
            else
               local_src_avg(l, kk) = 0.0_wp
            end if
         end do
      end do

      ! ---- Adaptive outer substepping ------------------------------------
      do io = 1, max_substep_it
         allref = .true.
         do l = 1, VL
            if (.not. sdone(l)) allref = .false.
         end do
         if (allref) exit
#ifdef KS_COUNTERS
         do l = 1, VL
            if (.not. sdone(l)) n_out(l) = n_out(l) + 1
         end do
#endif

         ! Step 1: K_src + seed TKE from previous kappa/K_Q.
         do l = 1, VL
            ks_src(l) = nz + 2
            ke_src(l) = 0
            ksrc(l, 1) = 0.0_wp
            ksrc(l, nz + 1) = 0.0_wp
         end do
         do kk = 2, nz
            do l = 1, VL
               ksrc(l, kk) = ks_src_func(ri_crit, shearmix_rate, fri_curvature, &
                                         n2(l, kk), s2(l, kk))
               if (ksrc(l, kk) > 0.0_wp) then
                  if (ks_src(l) > kk) ks_src(l) = kk
                  ke_src(l) = kk
               end if
            end do
         end do
         do kk = 1, nz + 1
            do l = 1, VL
               kappa_src(l, kk) = ksrc(l, kk)
            end do
         end do

         do kk = 2, nz
            do l = 1, VL
               tkedec(l, kk) = sqrt(c_n2*n2(l, kk) + c_s2*s2(l, kk))
            end do
         end do
         do l = 1, VL
            tke(l, 1) = tke_bg
         end do
         do kk = 2, nz
            do l = 1, VL
               if (kappa(l, kk) > 0.0_wp .and. k_q(l, kk) > 0.0_wp) then
                  tke(l, kk) = kappa(l, kk)/k_q(l, kk)
               else
                  tke(l, kk) = tke_min
               end if
            end do
         end do
         do l = 1, VL
            tke(l, nz + 1) = tke_min
         end do

         ! ⚠ THE WHOLE-ARRAY COPY, and the reason wall time depends on
         ! NZ_STACK_MAX at all -- see ks.F90:949. KS_BOUNDED_COPY bounds it to
         ! 1..nz+1; the default copies the DECLARED extent, as Fortran
         ! whole-array assignment does. Kept byte-for-byte equivalent to
         ! ks.F90 so BCOPY means the same thing in both variants.
#ifdef KS_BOUNDED_COPY
         ctop = nz + 1
#else
         ctop = NZLI
#endif
         do kk = 1, ctop
            do l = 1, VL
               kappa_out(l, kk) = kappa(l, kk)
            end do
         end do

         do l = 1, VL
            no_mixing(l) = (ks_src(l) > ke_src(l))
         end do
         do kk = 1, nz + 1
            do l = 1, VL
               if (no_mixing(l) .and. .not. sdone(l)) kappa_out(l, kk) = 0.0_wp
            end do
         end do
         do kk = 2, nz
            do l = 1, VL
               if (no_mixing(l) .and. .not. sdone(l)) ild2(l, kk) = 0.0_wp
            end do
         end do
         do l = 1, VL
            skip_step1(l) = sdone(l) .or. no_mixing(l)
         end do

         ! ks.F90 calls ks_find_kappa_tke only for mixing columns; the blocked
         ! call is block-wide, and the routine's own `nomix` mask reproduces
         ! the skip per lane.
         do kk = 1, ctop
            do l = 1, VL
               kq_tmp(l, kk) = k_q(l, kk)
            end do
         end do
         call ks_find_kappa_tke(nz, tke_min, f2_val, ri_crit, &
                                shearmix_rate, fri_curvature, c_n2, c_s2, &
                                ilambda2, kappa_0, kappa_trunc, tke_bg, &
                                tol_err, max_inner_it, n2, s2, kappa, &
                                kq_tmp, idz_s, hint_s, il2_s, e1, tke, &
                                kappa_out, ksrc, tkedec, aq, dq, cq, dk, &
                                ck, ild2, skip_step1 &
#ifdef KS_COUNTERS
                                , n_it_l &
#endif
                                )
#ifdef KS_COUNTERS
         do l = 1, VL
            if (.not. sdone(l) .and. .not. no_mixing(l)) n_in(l) = n_in(l) + n_it_l(l)
         end do
#endif
         do kk = 1, nz + 1
            do l = 1, VL
               if (.not. no_mixing(l) .and. .not. sdone(l)) k_q(l, kk) = kq_tmp(l, kk)
            end do
         end do

         ! local_src for the adaptive-dt bands.
         do l = 1, VL
            local_src(l, 1) = 0.0_wp
            local_src(l, nz + 1) = 0.0_wp
         end do
         do kk = 2, nz
            do l = 1, VL
               if (hint_s(l, kk) > 0.0_wp) then
                  local_src(l, kk) = ksrc(l, kk) + kappa_0* &
                                     ((idz_s(l, kk - 1) + idz_s(l, kk))/ &
                                      max(hint_s(l, kk), 1.0e-30_wp) + ild2(l, kk))
                  n2v = idz_s(l, kk - 1)*(kappa_out(l, kk - 1) - kappa_out(l, kk)) + &
                        idz_s(l, kk)*(kappa_out(l, kk + 1) - kappa_out(l, kk))
                  if (n2v > 0.0_wp) then
                     local_src(l, kk) = local_src(l, kk) + &
                                        n2v/max(hint_s(l, kk), 1.0e-30_wp)
                  end if
               else
                  local_src(l, kk) = ksrc(l, kk)
               end if
            end do
         end do

         ! Step 2: active range of kappa_out.
         do l = 1, VL
            ks_kap(l) = nz + 2
            ke_kap(l) = 0
         end do
         do kk = 2, nz
            do l = 1, VL
               if (kappa_out(l, kk) > 0.0_wp) then
                  if (ks_kap(l) > kk) ks_kap(l) = kk
                  ke_kap(l) = kk
               end if
            end do
         end do
         do l = 1, VL
            if (ke_kap(l) == nz) kappa_out(l, nz + 1) = 0.0_wp
            no_mixing(l) = (ke_kap(l) < ks_kap(l))
            mixing_step(l) = (.not. sdone(l)) .and. (.not. no_mixing(l))
            skip_mix(l) = .not. mixing_step(l)
         end do

         ! Step 3: choose dt_now.
         do l = 1, VL
            ks_lyr(l) = max(ks_kap(l) - 1, 1)
            ke_lyr(l) = min(ke_kap(l), nz)
            dt_now(l) = dt_rem(l)
         end do
         if (io /= max_substep_it) then
            call ks_adaptive_dt(nz, dt_rem, io, max_substep_it, ri_crit, &
                                shearmix_rate, fri_curvature, src_max_chg, &
                                tol_err, vel_underflow, dbuoy_t, dbuoy_s, &
                                h_sd, u_c, v_c, t_c, s_c, kappa_out, &
                                kappa_src, local_src, local_src_avg, &
                                ks_kap, ke_kap, idz_int_s, mixing_step, dt_now)
         end if
         do l = 1, VL
            if (sdone(l) .or. no_mixing(l) .or. io == max_substep_it) dt_now(l) = dt_rem(l)
         end do
         do kk = 2, nz
            do l = 1, VL
               if (.not. sdone(l)) then
                  local_src_avg(l, kk) = local_src_avg(l, kk) + dt_now(l)*local_src(l, kk)
               end if
            end do
         end do
         do l = 1, VL
            dt_wt(l) = dt_now(l)/dt
         end do

         ! ---- the no_mixing arm ------------------------------------------
         do kk = 1, nz + 1
            do l = 1, VL
               if (.not. sdone(l) .and. no_mixing(l)) then
                  tke_avg(l, kk) = tke_avg(l, kk) + dt_wt(l)*tke(l, kk)
               end if
            end do
         end do
         do l = 1, VL
            if (.not. sdone(l) .and. no_mixing(l)) dt_rem(l) = 0.0_wp
         end do

         ! ---- the mixing arm: predictor ----------------------------------
         do l = 1, VL
            if (mixing_step(l)) then
               ks_ps(l) = max(ks_kap(l) - 1, 1)
               ke_ps(l) = min(ke_kap(l), nz)
            else
               ks_ps(l) = nz + 2        ! empty band -> ks_projected_state
               ke_ps(l) = 0             ! leaves this lane untouched
            end if
         end do
         call ks_projected_state(nz, dt_now, ks_ps, ke_ps, vel_underflow, &
                                 dbuoy_t, dbuoy_s, h_sd, idz_int_s, u_c, &
                                 v_c, t_c, s_c, kappa_out, u_ps, v_ps, &
                                 t_ps, s_ps, c1_ps, n2p, s2p)
         do kk = 1, ctop
            do l = 1, VL
               kq_tmp(l, kk) = k_q(l, kk)
            end do
         end do
         call ks_find_kappa_tke(nz, tke_min, f2_val, ri_crit, &
                                shearmix_rate, fri_curvature, c_n2, c_s2, &
                                ilambda2, kappa_0, kappa_trunc, tke_bg, &
                                tol_err, max_inner_it, n2p, s2p, kappa_out, &
                                kq_tmp, idz_s, hint_s, il2_s, e1, tke_pred, &
                                kappa_pred, ksrc, tkedec, aq, dq, cq, dk, &
                                ck, ild2, skip_mix &
#ifdef KS_COUNTERS
                                , n_it_l &
#endif
                                )
#ifdef KS_COUNTERS
         do l = 1, VL
            if (mixing_step(l)) n_in(l) = n_in(l) + n_it_l(l)
         end do
#endif
         do kk = 1, nz + 1
            do l = 1, VL
               if (mixing_step(l)) kappa_mid(l, kk) = 0.5_wp*(kappa_out(l, kk) + &
                                                              kappa_pred(l, kk))
            end do
         end do
         do l = 1, VL
            ks_mid(l) = nz + 2
            ke_mid(l) = 0
         end do
         do kk = 1, nz + 1
            do l = 1, VL
               if (mixing_step(l) .and. kappa_mid(l, kk) > 0.0_wp) then
                  if (ks_mid(l) > kk) ks_mid(l) = kk
                  ke_mid(l) = kk
               end if
            end do
         end do
         do l = 1, VL
            if (mixing_step(l)) then
               ks_ps(l) = max(ks_mid(l) - 1, 1)
               ke_ps(l) = min(ke_mid(l), nz)
            else
               ks_ps(l) = nz + 2
               ke_ps(l) = 0
            end if
         end do

         ! ---- corrector (real K_Q now) -----------------------------------
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
                                ck, ild2, skip_mix &
#ifdef KS_COUNTERS
                                , n_it_l &
#endif
                                )
#ifdef KS_COUNTERS
         do l = 1, VL
            if (mixing_step(l)) n_in(l) = n_in(l) + n_it_l(l)
         end do
#endif
         do l = 1, VL
            if (mixing_step(l)) dt_rem(l) = dt_rem(l) - dt_now(l)
         end do
         do kk = 1, nz + 1
            do l = 1, VL
               if (mixing_step(l)) then
                  kappa_avg(l, kk) = kappa_avg(l, kk) + &
                                     dt_wt(l)*0.5_wp*(kappa_out(l, kk) + kappa_pred2(l, kk))
                  tke_avg(l, kk) = tke_avg(l, kk) + &
                                   dt_wt(l)*0.5_wp*(tke_pred(l, kk) + tke_fin(l, kk))
                  kappa(l, kk) = kappa_pred2(l, kk)
               end if
            end do
         end do
         do l = 1, VL
            if (mixing_step(l)) then
               kappa(l, 1) = 0.0_wp
               kappa(l, nz + 1) = 0.0_wp
            end if
         end do

         ! ---- Step 5: full-column advance for the next substep -----------
         do kk = 1, nz + 1
            do l = 1, VL
               if (mixing_step(l) .and. dt_rem(l) > 0.0_wp) then
                  kappa_mid(l, kk) = 0.5_wp*(kappa_out(l, kk) + kappa_pred2(l, kk))
               end if
            end do
         end do
         do l = 1, VL
            if (mixing_step(l) .and. dt_rem(l) > 0.0_wp) then
               ks_ps(l) = 1
               ke_ps(l) = nz
            else
               ks_ps(l) = nz + 2
               ke_ps(l) = 0
            end if
         end do
         call ks_projected_state(nz, dt_now, ks_ps, ke_ps, vel_underflow, &
                                 dbuoy_t, dbuoy_s, h_sd, idz_int_s, u_c, &
                                 v_c, t_c, s_c, kappa_mid, u_ps, v_ps, &
                                 t_ps, s_ps, c1_ps, n2p, s2p)
         do kk = 1, nz + 1
            do l = 1, VL
               if (mixing_step(l) .and. dt_rem(l) > 0.0_wp) then
                  n2(l, kk) = n2p(l, kk)
                  s2(l, kk) = s2p(l, kk)
               end if
            end do
         end do
         do k = 1, nz
            do l = 1, VL
               if (mixing_step(l) .and. dt_rem(l) > 0.0_wp) then
                  u_c(l, k) = u_ps(l, k)
                  v_c(l, k) = v_ps(l, k)
                  t_c(l, k) = t_ps(l, k)
                  s_c(l, k) = s_ps(l, k)
               end if
            end do
         end do

         do l = 1, VL
            if (.not. sdone(l) .and. dt_rem(l) <= 0.0_wp) sdone(l) = .true.
         end do
      end do

      do kk = 1, nz + 1
         do l = 1, VL
            kappa_avg_sd(l, kk) = kappa_avg(l, kk)
            tke_avg_sd(l, kk) = tke_avg(l, kk)
         end do
      end do
      do l = 1, VL
         kappa_avg_sd(l, 1) = 0.0_wp
         kappa_avg_sd(l, nz + 1) = 0.0_wp
      end do
   end subroutine ks_solve_column

end module ks
