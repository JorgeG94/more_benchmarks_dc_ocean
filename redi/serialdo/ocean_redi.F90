!! ============================================================================
!! MRE EXTRACT — VERBATIM from <model>/src/parameterizations/lateral/structured/
!! ocean_redi.F90 (md5 c941f60101124f003f27453365930d2d, 1170 lines).
!!
!! Lines 1-1148 are BYTE-IDENTICAL to production. Every `do concurrent` loop,
!! every `!$acc routine seq` helper, and the whole neutral-surface sweep are
!! unmodified. Nothing was transcribed; nothing was simplified.
!!
!! THE ONLY CHANGES (both scaffolding, neither reachable from a kernel):
!!   1. `use mem_report, only: arr_bytes` dropped, and with it the
!!      `ocean_redi_bytes` function (production lines 1149-1168) and its
!!      `bytes` type-bound procedure. It is a host-side memory accountant that
!!      sums arr_bytes() over the 13 allocatables; it is never called on the
!!      device and pulling it in would drag the whole reporting module.
!!   2. `use, intrinsic :: iso_fortran_env, only: int64` dropped — it existed
!!      solely for ocean_bytes' return type.
!! The `use constants / grid / ocean_metrics /
!! multilayer_cgrid_state / ocean_eos / ocean_boundary_types` lines
!! are untouched; those modules are MRE stubs in this directory carrying the
!! exact fields Redi reads (see each stub's header).
!!
!! CONFIG COVERAGE — what the gabight namelists actually exercise:
!!   * `enable = .true.`, `khtr = 100.0` (gabight_sph_meke_v100.nml:125-128).
!!   * `continuous = .true.` — the only variant shipped; `.false.` is rejected
!!     at configure, so the discontinuous path does not exist yet.
!!   * `ref_pres = -1` (default, locally-referenced). NOTE: `ref_pres` is
!!     passed into redi_calc_coeffs_x/_y and is then NEVER READ — the interface
!!     pressure is always built hydrostatically in redi_build_column. It is a
!!     live dummy argument of a dead knob. Left in place (verbatim).
!!   * EOS: no `&eos_nml` in any Redi config => EOS_VARIANT_LINEAR.
!!   * MEKE supplies khtr_u_ext/khtr_v_ext (khcoeff=1.0), so the `use_ext`
!!     branch + redi_face_copy is the live path, not redi_face_const.
!!     The driver benchmarks the `use_ext` path for that reason.
!! ============================================================================
#include "directives.h"
module ocean_redi
   use constants, only: wp, NZ_STACK_MAX, H_DIV_EPS, GRAVITY
   use grid, only: hgrid_t
   use ocean_metrics, only: ocean_metrics_t
   use multilayer_cgrid_state, only: multilayer_cgrid_state_t
   use ocean_eos, only: ocean_eos_t, eos_specvol_derivs, eos_density_point
   use ocean_boundary_types, only: ocean_bc_state_t, OBC_WALL
   implicit none
   private

   public :: redi_interface_scalar
   public :: redi_interpolate_position
   public :: redi_neutral_positions_continuous
   ! SCAFFOLDING (3rd deviation from verbatim, host-visible only): these two
   ! `!$acc routine seq` helpers are PRIVATE in production. Exported so the
   ! PRECOMP control variant can reuse the IDENTICAL source rather than
   ! duplicate it -- duplication would put bit-identity in question. Same
   ! precedent as continuity_layered_benchmark. No device code changed.
   public :: redi_signum1, redi_ppm_ave
   public :: redi_snapshot, redi_face_copy

   ! R2: device flux kernel + slot.
   public :: ocean_redi_t
   public :: redi_calc_coeffs
   public :: redi_apply_flux
   ! OPTIMIZED A/B variant (portable-Fortran analogue of the CUDA `kOptApplyFlux`
   ! precompute hoist): bit-identical to redi_apply_flux, reconstructs each
   ! cell's centre tracer column ONCE instead of 4x. Added ALONGSIDE the faithful
   ! path (which is byte-identical to production) for the dc_ab driver.
   public :: redi_apply_flux_hoist

   ! =====================================================================
   ! R2 — neutral-diffusion slot + two-phase GPU flux kernel
   ! =====================================================================

   type :: ocean_redi_t
      !! Redi continuous neutral-diffusion state.  Defaults inert
      !! (`enable=.false.`) ⇒ bit-identical.  Reads the prognostic T/S + EOS
      !! directly (recomputes its own interface dR/dT, dR/dS — does NOT
      !! consume GM slopes).  Augments `ocean_hdiff_tracer`.
      logical :: is_init = .false.
         !! True between init/destroy; gate on this, not on `allocated`.
      logical :: enable = .false.
         !! Master switch.  Off => `redi_calc_coeffs`/`redi_apply_flux`
         !! no-op => bit-identity preserved.
      logical :: continuous = .true.
         !! Continuous variant (the only one shipped).  Discontinuous
         !! deferred (R4); a `.false.` value is rejected at configure.
      real(wp) :: khtr = 0.0_wp
         !! Scalar Redi neutral diffusivity (m^2/s); the FALLBACK used when
         !! VarMix is off (0 => no flux).  When VarMix is enabled its
         !! spatially-varying `khtr_u`/`khtr_v` override this per face.
      real(wp) :: ref_pres = -1.0_wp
         !! Reference pressure (Pa); -1 => locally-referenced (interface
         !! hydrostatic pressure).  Held for forward compatibility.

      ! ---- Cached extents ----
      integer :: nx_total = 0
      integer :: ny_total = 0
      integer :: nz_ml = 0
      integer :: nsurf = 0
         !! 2*nz_ml + 2 (the neutral-surface count of the continuous sweep).

      ! ---- Phase-A device-resident coefficient arrays (per face) ----
      ! u-faces (nx+1,ny,nsurf); v-faces (nx,ny+1,nsurf).  Fractional
      ! positions PoL/PoR in [0,1], native-layer indices KoL/KoR (left/right
      ! column), and uhEff the harmonic-mean effective thickness of each of
      ! the nsurf-1 neutral sublayers.  KoL/KoR are stored TOP-DOWN (k=1
      ! surface); the only k-flip is the Phase-B scatter (native k = nz+1-Ko).
      real(wp), allocatable :: uPoL(:, :, :), uPoR(:, :, :)
      integer, allocatable :: uKoL(:, :, :), uKoR(:, :, :)
      real(wp), allocatable :: uhEff(:, :, :)
      real(wp), allocatable :: vPoL(:, :, :), vPoR(:, :, :)
      integer, allocatable :: vKoL(:, :, :), vKoR(:, :, :)
      real(wp), allocatable :: vhEff(:, :, :)

      ! ---- Per-face neutral diffusivity KhTr (m^2/s) ----
      ! Filled each apply step from VarMix's `khtr_u`/`khtr_v` when enabled,
      ! else broadcast from the scalar `khtr` (VarMix-less ⇒ bit-identical).
      real(wp), allocatable :: khtr_u(:, :)
      real(wp), allocatable :: khtr_v(:, :)

      ! ---- Read-only tracer snapshot for the flux gather ----
      ! The double-visit flux reads NEIGHBOUR hTr AND writes its OWN — a
      ! do-concurrent read-write race against the live array.  `tr_snap` holds
      ! the pre-step hTr so the gather is order-independent on every backend.
      real(wp), allocatable :: tr_snap(:, :, :)
   contains
      procedure, non_overridable :: init => ocean_redi_init
      procedure, non_overridable :: destroy => ocean_redi_destroy
      procedure, non_overridable :: enter_data => ocean_redi_enter_data
      procedure, non_overridable :: exit_data => ocean_redi_exit_data
   end type ocean_redi_t

contains

   !> Second-order centred finite-volume slope of a layer scalar (MOM6
   !! `fv_diff`; Colella & Woodward 1984).  Returns the cell-centred difference
   !! across layer k given the three layer thicknesses and values.
   pure function redi_fv_diff(hkm1, hk, hkp1, skm1, sk, skp1) result(d)
      DC_ROUTINE_SEQ
      real(wp), intent(in) :: hkm1, hk, hkp1  !! layer thicknesses (above/centre/below)
      real(wp), intent(in) :: skm1, sk, skp1  !! layer scalar values
      real(wp) :: d
      real(wp) :: h_sum, hp, hm

      h_sum = (hkm1 + hkp1) + hk
      if (h_sum /= 0.0_wp) h_sum = 1.0_wp/h_sum
      hm = hkm1 + hk
      if (hm /= 0.0_wp) hm = 1.0_wp/hm
      hp = hkp1 + hk
      if (hp /= 0.0_wp) hp = 1.0_wp/hp
      d = (hk*h_sum)*((2.0_wp*hkm1 + hk)*hp*(skp1 - sk) &
                      + (2.0_wp*hkp1 + hk)*hm*(sk - skm1))
   end function redi_fv_diff

   !> A true signum: -|a| if x<0, +|a| if x>0, 0 if x==0 (MOM6 `signum`).
   pure function redi_signum(a, x) result(s)
      DC_ROUTINE_SEQ
      real(wp), intent(in) :: a, x
      real(wp) :: s

      s = sign(a, x)
      if (x == 0.0_wp) s = 0.0_wp
   end function redi_signum

   !> PLM van-Leer-limited layer-difference array (MOM6 `PLM_diff` with
   !! c_method=2 finite-volume slope, b_method=1 PCM ends) — the slope input to
   !! the PPM edge interpolation.
   pure subroutine redi_plm_diff(nk, h, s, diff)
      DC_ROUTINE_SEQ
      integer, intent(in) :: nk
      real(wp), intent(in) :: h(nk)     !! layer thicknesses
      real(wp), intent(in) :: s(nk)     !! layer scalar values
      real(wp), intent(out) :: diff(nk)  !! limited layer difference (PCM ends)
      integer :: k
      real(wp) :: diff_l, diff_r, diff_c

      do k = 2, nk - 1
         if ((h(k + 1) + h(k))*(h(k - 1) + h(k)) > 0.0_wp) then
            diff_c = redi_fv_diff(h(k - 1), h(k), h(k + 1), s(k - 1), s(k), s(k + 1))
            diff_l = 2.0_wp*(s(k) - s(k - 1))
            diff_r = 2.0_wp*(s(k + 1) - s(k))
            if (redi_signum(1.0_wp, diff_l)*redi_signum(1.0_wp, diff_r) <= 0.0_wp) then
               diff(k) = 0.0_wp   ! PCM for local extrema
            else
               diff(k) = sign(min(abs(diff_l), abs(diff_c), abs(diff_r)), diff_c)
            end if
         else
            diff(k) = 0.0_wp      ! PCM next to vanished layers
         end if
      end do
      diff(1) = 0.0_wp            ! b_method=1: PCM top
      diff(nk) = 0.0_wp          ! b_method=1: PCM bottom
   end subroutine redi_plm_diff

   !> PPM quasi-fourth-order edge value at interface k+1/2 (MOM6 `ppm_edge`;
   !! Colella & Woodward 1984 eq. 1.6).
   pure function redi_ppm_edge(hkm1, hk, hkp1, hkp2, ak, akp1, pk, pkp1, h_neglect) result(e)
      DC_ROUTINE_SEQ
      real(wp), intent(in) :: hkm1, hk, hkp1, hkp2  !! widths of cells k-1..k+2
      real(wp), intent(in) :: ak, akp1              !! cell averages k, k+1
      real(wp), intent(in) :: pk, pkp1              !! PLM slopes k, k+1
      real(wp), intent(in) :: h_neglect             !! negligible thickness floor
      real(wp) :: e
      real(wp) :: r_hk_hkp1, r_2hk_hkp1, r_hk_2hkp1
      real(wp) :: f1, f2, f3, f4

      r_hk_hkp1 = hk + hkp1
      if (r_hk_hkp1 <= 0.0_wp) then
         e = 0.5_wp*(ak + akp1)
         return
      end if
      r_hk_hkp1 = 1.0_wp/r_hk_hkp1
      if (hk < hkp1) then
         e = ak + (hk*r_hk_hkp1)*(akp1 - ak)
      else
         e = akp1 + (hkp1*r_hk_hkp1)*(ak - akp1)
      end if

      r_2hk_hkp1 = 1.0_wp/((2.0_wp*hk + hkp1) + h_neglect)
      r_hk_2hkp1 = 1.0_wp/((hk + 2.0_wp*hkp1) + h_neglect)
      f1 = 1.0_wp/((hk + hkp1) + (hkm1 + hkp2))
      f2 = 2.0_wp*(hkp1*hk)*r_hk_hkp1* &
           ((hkm1 + hk)*r_2hk_hkp1 - (hkp2 + hkp1)*r_hk_2hkp1)
      f3 = hk*(hkm1 + hk)*r_2hk_hkp1
      f4 = hkp1*(hkp1 + hkp2)*r_hk_2hkp1

      e = e + f1*(f2*(akp1 - ak) - (f3*pkp1 - f4*pk))
   end function redi_ppm_edge

   !> PPM continuous edge reconstruction of a layer scalar to interfaces
   !! (MOM6 `interface_scalar` with i_method=2).  `edge(1)`=surface,
   !! `edge(nk+1)`=bed in the MOM6 top-down sense (see module header on the
   !! deferred k-flip).
   pure subroutine redi_interface_scalar(nk, h, tr, edge, h_neglect)
      DC_ROUTINE_SEQ
      integer, intent(in) :: nk
      real(wp), intent(in) :: h(nk)        !! layer thicknesses
      real(wp), intent(in) :: tr(nk)       !! layer scalar (e.g. T)
      real(wp), intent(out) :: edge(nk + 1)  !! interface scalar
      real(wp), intent(in), optional :: h_neglect  !! negligible thickness (default 1e-30)
      real(wp) :: diff(NZ_STACK_MAX)  !! fixed-size device stack (dummy-sized automatic crashes on -stdpar=gpu)
      real(wp) :: hn
      integer :: k, km2, kp1

      hn = 1.0e-30_wp
      if (present(h_neglect)) hn = h_neglect

      call redi_plm_diff(nk, h, tr, diff)
      edge(1) = tr(1) - 0.5_wp*diff(1)
      do k = 2, nk
         km2 = max(1, k - 2)
         kp1 = min(nk, k + 1)
         edge(k) = redi_ppm_edge(h(km2), h(k - 1), h(k), h(kp1), &
                                 tr(k - 1), tr(k), diff(k - 1), diff(k), hn)
      end do
      edge(nk + 1) = tr(nk) + 0.5_wp*diff(nk)
   end subroutine redi_interface_scalar

   !> Non-dimensional position in [0,1] where the interpolated density
   !! difference is zero.  Guards the vanished/inverted (Ppos==Pneg) and
   !! degenerate (dRhoPos==dRhoNeg) cases device-safely (clamped values, no
   !! host I/O).
   pure function redi_interpolate_position(dRhoNeg, Pneg, dRhoPos, Ppos) result(pos)
      DC_ROUTINE_SEQ
      real(wp), intent(in) :: dRhoNeg  !! negative density difference
      real(wp), intent(in) :: Pneg     !! position of the negative difference
      real(wp), intent(in) :: dRhoPos  !! positive density difference
      real(wp), intent(in) :: Ppos     !! position of the positive difference
      real(wp) :: pos

      if ((Ppos > Pneg) .and. (dRhoPos - dRhoNeg >= 0.0_wp)) then
         if (dRhoPos - dRhoNeg > 0.0_wp) then
            pos = min(1.0_wp, max(0.0_wp, -dRhoNeg/(dRhoPos - dRhoNeg)))
         else  ! dRhoPos - dRhoNeg == 0
            if (dRhoNeg > 0.0_wp) then
               pos = 0.0_wp
            else if (dRhoNeg < 0.0_wp) then
               pos = 1.0_wp
            else
               pos = 0.5_wp
            end if
         end if
      else if (Ppos == Pneg) then  ! vanished or inverted layers
         pos = 0.5_wp
      else  ! (Ppos < Pneg) .or. (dRhoNeg > dRhoPos): MOM6 errors here; device-safe clamp
         pos = 0.5_wp
      end if
   end function redi_interpolate_position

   !> Absolute (pressure) position of neutral surface `ks` in a column
   !! (MOM6 `absolute_position`): Pint(K) + frac*(Pint(K+1)-Pint(K)).
   pure function redi_absolute_position(nk, Pint, Karr, frac) result(p)
      DC_ROUTINE_SEQ
      integer, intent(in) :: nk
      real(wp), intent(in) :: Pint(nk + 1)  !! interface pressures
      integer, intent(in) :: Karr         !! layer index for this surface
      real(wp), intent(in) :: frac        !! fractional position within the layer
      real(wp) :: p

      p = Pint(Karr) + frac*(Pint(Karr + 1) - Pint(Karr))
   end function redi_absolute_position

   !> The continuous neutral-surface sweep over a column pair.  A single
   !! deterministic top→bottom sweep of `2*nk+2` surfaces walking two interface
   !! pointers; closed-form linear crossing per step (no inner iteration).
   !! Inputs are interface T/S/P + interface dR/dT, dR/dS (nk+1 each).  Outputs
   !! `PoL/PoR` (fractional position within layer `KoL/KoR`) and `hEff`
   !! (harmonic-mean effective thickness between consecutive neutral surfaces;
   !! outcrops get hEff=0, not skipped).  TOP-DOWN indexing (see module header).
   pure subroutine redi_neutral_positions_continuous(nk, Pl, Tl, Sl, dRdTl, dRdSl, &
                                                     Pr, Tr, Sr, dRdTr, dRdSr, &
                                                     PoL, PoR, KoL, KoR, hEff)
      DC_ROUTINE_SEQ
      integer, intent(in) :: nk
      real(wp), intent(in) :: Pl(nk + 1), Tl(nk + 1), Sl(nk + 1)     !! left interface P, T, S
      real(wp), intent(in) :: dRdTl(nk + 1), dRdSl(nk + 1)         !! left interface dR/dT, dR/dS
      real(wp), intent(in) :: Pr(nk + 1), Tr(nk + 1), Sr(nk + 1)     !! right interface P, T, S
      real(wp), intent(in) :: dRdTr(nk + 1), dRdSr(nk + 1)         !! right interface dR/dT, dR/dS
      real(wp), intent(out) :: PoL(2*nk + 2), PoR(2*nk + 2)        !! fractional positions
      integer, intent(out) :: KoL(2*nk + 2), KoR(2*nk + 2)         !! layer indices
      real(wp), intent(out) :: hEff(2*nk + 1)                    !! effective thicknesses

      integer :: ns, k_surface, kl, kr, krm1, klm1
      integer :: lastK_left, lastK_right
      real(wp) :: lastP_left, lastP_right
      real(wp) :: dRho, dRhoTop, dRhoBot, hL, hR
      logical :: searching_left_column, searching_right_column, reached_bottom

      ns = 2*nk + 2

      kr = 1
      kl = 1
      lastP_right = 0.0_wp
      lastP_left = 0.0_wp
      lastK_right = 1
      lastK_left = 1
      reached_bottom = .false.
      searching_left_column = .false.
      searching_right_column = .false.

      do k_surface = 1, ns
         klm1 = max(kl - 1, 1)
         krm1 = max(kr - 1, 1)

         ! Cross-column dR/dT, dR/dS-weighted secant density difference:
         ! rho(kr) - rho(kl) (NOT a raw density difference).
         dRho = 0.5_wp*((dRdTr(kr) + dRdTl(kl))*(Tr(kr) - Tl(kl)) &
                        + (dRdSr(kr) + dRdSl(kl))*(Sr(kr) - Sl(kl)))

         if (.not. reached_bottom) then
            if (dRho < 0.0_wp) then
               searching_left_column = .true.
               searching_right_column = .false.
            else if (dRho > 0.0_wp) then
               searching_right_column = .true.
               searching_left_column = .false.
            else  ! dRho == 0: tie-break
               if (kl + kr == 2) then  ! still at surface
                  searching_left_column = .true.
                  searching_right_column = .false.
               else  ! not the surface — change direction
                  searching_left_column = .not. searching_left_column
                  searching_right_column = .not. searching_right_column
               end if
            end if
         end if

         if (searching_left_column) then
            ! rho(kl-1) - rho(kr) (should be negative)
            dRhoTop = 0.5_wp*((dRdTl(klm1) + dRdTr(kr))*(Tl(klm1) - Tr(kr)) &
                              + (dRdSl(klm1) + dRdSr(kr))*(Sl(klm1) - Sr(kr)))
            ! rho(kl) - rho(kr) (will be positive)
            dRhoBot = 0.5_wp*((dRdTl(klm1 + 1) + dRdTr(kr))*(Tl(klm1 + 1) - Tr(kr)) &
                              + (dRdSl(klm1 + 1) + dRdSr(kr))*(Sl(klm1 + 1) - Sr(kr)))

            if (dRhoTop > 0.0_wp .or. kr + kl == 2) then
               PoL(k_surface) = 0.0_wp
            else if (dRhoTop >= dRhoBot) then  ! unstratified left layer
               PoL(k_surface) = 1.0_wp
            else
               PoL(k_surface) = redi_interpolate_position(dRhoTop, Pl(klm1), dRhoBot, Pl(klm1 + 1))
            end if
            if (PoL(k_surface) >= 1.0_wp .and. klm1 < nk) then  ! carry to next layer
               klm1 = klm1 + 1
               PoL(k_surface) = PoL(k_surface) - 1.0_wp
            end if
            ! Monotonic-position backstop
            if (real(klm1 - lastK_left, wp) + (PoL(k_surface) - lastP_left) < 0.0_wp) then
               PoL(k_surface) = lastP_left
               klm1 = lastK_left
            end if
            KoL(k_surface) = klm1
            if (kr <= nk) then
               PoR(k_surface) = 0.0_wp
               KoR(k_surface) = kr
            else
               PoR(k_surface) = 1.0_wp
               KoR(k_surface) = nk
            end if
            if (kr <= nk) then
               kr = kr + 1
            else  ! column exhausted: direction flip
               reached_bottom = .true.
               searching_right_column = .true.
               searching_left_column = .false.
            end if
         else if (searching_right_column) then
            ! rho(kr-1) - rho(kl) (should be negative)
            dRhoTop = 0.5_wp*((dRdTr(krm1) + dRdTl(kl))*(Tr(krm1) - Tl(kl)) &
                              + (dRdSr(krm1) + dRdSl(kl))*(Sr(krm1) - Sl(kl)))
            ! rho(kr) - rho(kl) (will be positive)
            dRhoBot = 0.5_wp*((dRdTr(krm1 + 1) + dRdTl(kl))*(Tr(krm1 + 1) - Tl(kl)) &
                              + (dRdSr(krm1 + 1) + dRdSl(kl))*(Sr(krm1 + 1) - Sl(kl)))

            if (dRhoTop >= 0.0_wp .or. kr + kl == 2) then
               PoR(k_surface) = 0.0_wp
            else if (dRhoTop >= dRhoBot) then  ! unstratified right layer
               PoR(k_surface) = 1.0_wp
            else
               PoR(k_surface) = redi_interpolate_position(dRhoTop, Pr(krm1), dRhoBot, Pr(krm1 + 1))
            end if
            if (PoR(k_surface) >= 1.0_wp .and. krm1 < nk) then  ! carry to next layer
               krm1 = krm1 + 1
               PoR(k_surface) = PoR(k_surface) - 1.0_wp
            end if
            ! Monotonic-position backstop
            if (real(krm1 - lastK_right, wp) + (PoR(k_surface) - lastP_right) < 0.0_wp) then
               PoR(k_surface) = lastP_right
               krm1 = lastK_right
            end if
            KoR(k_surface) = krm1
            if (kl <= nk) then
               PoL(k_surface) = 0.0_wp
               KoL(k_surface) = kl
            else
               PoL(k_surface) = 1.0_wp
               KoL(k_surface) = nk
            end if
            if (kl <= nk) then
               kl = kl + 1
            else  ! column exhausted: direction flip
               reached_bottom = .true.
               searching_right_column = .false.
               searching_left_column = .true.
            end if
         end if

         lastK_left = KoL(k_surface)
         lastP_left = PoL(k_surface)
         lastK_right = KoR(k_surface)
         lastP_right = PoR(k_surface)

         ! Effective thickness between consecutive neutral surfaces (harmonic
         ! mean of the left/right pressure thicknesses; outcrops => hEff=0).
         if (k_surface > 1) then
            hL = redi_absolute_position(nk, Pl, KoL(k_surface), PoL(k_surface)) &
                 - redi_absolute_position(nk, Pl, KoL(k_surface - 1), PoL(k_surface - 1))
            hR = redi_absolute_position(nk, Pr, KoR(k_surface), PoR(k_surface)) &
                 - redi_absolute_position(nk, Pr, KoR(k_surface - 1), PoR(k_surface - 1))
            if (hL + hR > 0.0_wp) then
               hEff(k_surface - 1) = 2.0_wp*hL*hR/(hL + hR)
            else
               hEff(k_surface - 1) = 0.0_wp
            end if
         end if
      end do
   end subroutine redi_neutral_positions_continuous

   subroutine ocean_redi_init(this, grid, nz_ml)
      !! Allocate the Phase-A coefficient arrays.  Always allocates
      !! (configure runs after init); off-state footprint is the six
      !! face-shaped (nsurf) coefficient arrays.  Plain host allocation
      !! (no `do concurrent` before enter_data).
      class(ocean_redi_t), intent(inout) :: this
      type(hgrid_t), intent(in) :: grid
      integer, intent(in), optional :: nz_ml
      integer :: nx, ny, nz, ns

      nx = grid%nx_total
      ny = grid%ny_total
      nz = 1
      if (present(nz_ml)) nz = nz_ml
      if (nz < 1) nz = 1
      ! Fail loud: the sweep uses NZ_STACK_MAX-sized column-pair + nsurf
      ! locals.  nsurf = 2*nz+2 must fit; NZ_STACK_MAX bounds nz here.
      if (2*nz + 2 > NZ_STACK_MAX) then
         error stop "ocean_redi_init: 2*nz_ml+2 (the neutral-surface count) "// &
            "exceeds NZ_STACK_MAX; raise -DMODEL_NZ_STACK_MAX or disable Redi"
      end if
      this%nx_total = nx
      this%ny_total = ny
      this%nz_ml = nz
      ns = 2*nz + 2
      this%nsurf = ns

      allocate (this%uPoL(nx + 1, ny, ns), source=0.0_wp)
      allocate (this%uPoR(nx + 1, ny, ns), source=0.0_wp)
      allocate (this%uKoL(nx + 1, ny, ns), source=1)
      allocate (this%uKoR(nx + 1, ny, ns), source=1)
      allocate (this%uhEff(nx + 1, ny, ns - 1), source=0.0_wp)
      allocate (this%vPoL(nx, ny + 1, ns), source=0.0_wp)
      allocate (this%vPoR(nx, ny + 1, ns), source=0.0_wp)
      allocate (this%vKoL(nx, ny + 1, ns), source=1)
      allocate (this%vKoR(nx, ny + 1, ns), source=1)
      allocate (this%vhEff(nx, ny + 1, ns - 1), source=0.0_wp)
      allocate (this%khtr_u(nx + 1, ny), source=0.0_wp)
      allocate (this%khtr_v(nx, ny + 1), source=0.0_wp)
      allocate (this%tr_snap(nx, ny, this%nz_ml), source=0.0_wp)
      this%is_init = .true.
   end subroutine ocean_redi_init

   subroutine ocean_redi_destroy(this)
      class(ocean_redi_t), intent(inout) :: this
      this%is_init = .false.
      if (allocated(this%uPoL)) deallocate (this%uPoL)
      if (allocated(this%uPoR)) deallocate (this%uPoR)
      if (allocated(this%uKoL)) deallocate (this%uKoL)
      if (allocated(this%uKoR)) deallocate (this%uKoR)
      if (allocated(this%uhEff)) deallocate (this%uhEff)
      if (allocated(this%vPoL)) deallocate (this%vPoL)
      if (allocated(this%vPoR)) deallocate (this%vPoR)
      if (allocated(this%vKoL)) deallocate (this%vKoL)
      if (allocated(this%vKoR)) deallocate (this%vKoR)
      if (allocated(this%vhEff)) deallocate (this%vhEff)
      if (allocated(this%khtr_u)) deallocate (this%khtr_u)
      if (allocated(this%khtr_v)) deallocate (this%khtr_v)
      if (allocated(this%tr_snap)) deallocate (this%tr_snap)
      this%nx_total = 0
      this%ny_total = 0
      this%nz_ml = 0
      this%nsurf = 0
   end subroutine ocean_redi_destroy

   subroutine ocean_redi_enter_data(this)
      ! Poly TBP delegating to a `type(...)`-arg `_impl` (AMD-crash rule).
      class(ocean_redi_t), intent(inout) :: this
      select type (this)
      type is (ocean_redi_t)
         call ocean_redi_enter_data_impl(this)
      end select
   end subroutine ocean_redi_enter_data

   subroutine ocean_redi_enter_data_impl(this)
      type(ocean_redi_t), intent(inout) :: this
      if (.not. this%is_init) return
      !$acc enter data copyin(this%uPoL, this%uPoR, this%uKoL, this%uKoR, this%uhEff)
      !$acc enter data copyin(this%vPoL, this%vPoR, this%vKoL, this%vKoR, this%vhEff)
      !$acc enter data copyin(this%khtr_u, this%khtr_v)
      !$acc enter data copyin(this%tr_snap)
   end subroutine ocean_redi_enter_data_impl

   subroutine ocean_redi_exit_data(this)
      class(ocean_redi_t), intent(inout) :: this
      select type (this)
      type is (ocean_redi_t)
         call ocean_redi_exit_data_impl(this)
      end select
   end subroutine ocean_redi_exit_data

   subroutine ocean_redi_exit_data_impl(this)
      type(ocean_redi_t), intent(inout) :: this
      if (.not. this%is_init) return
      !$acc exit data delete(this%tr_snap)
      !$acc exit data delete(this%khtr_u, this%khtr_v)
      !$acc exit data delete(this%vPoL, this%vPoR, this%vKoL, this%vKoR, this%vhEff)
      !$acc exit data delete(this%uPoL, this%uPoR, this%uKoL, this%uKoR, this%uhEff)
   end subroutine ocean_redi_exit_data_impl

   ! =====================================================================
   ! Phase A — calc_coeffs (tracer-independent).  Per face, build the two
   ! adjacent columns' TOP-DOWN interface T/S/P + dR/dT, dR/dS, run the
   ! continuous sweep, store PoL/PoR/KoL/KoR/hEff.
   ! =====================================================================

   subroutine redi_calc_coeffs(grid, metrics, eos, this, ms)
      !! Public entry: fill the Phase-A coefficient arrays.  No-op if
      !! absent / uninitialised / disabled.  Run once per outer step at
      !! THERMO cadence (a slow, tracer-independent geometry).  Outer-shim:
      !! dereference the tracer-registry hTr arrays on the host, pass the
      !! flat top-level allocatables to the flat-impl kernels.
      type(hgrid_t), intent(in) :: grid
      type(ocean_metrics_t), intent(in) :: metrics
      type(ocean_eos_t), intent(in) :: eos
      type(ocean_redi_t), intent(inout) :: this
      type(multilayer_cgrid_state_t), intent(in) :: ms
      integer :: nx, ny, nz

      if (.not. this%is_init) return
      if (.not. this%enable) return
      if (ms%idx_temperature <= 0 .or. ms%idx_salinity <= 0) return
      if (.not. allocated(ms%h_layer)) return

      nx = grid%nx_total
      ny = grid%ny_total
      nz = ms%nz_ml
      if (this%nz_ml /= nz) return

      call redi_calc_coeffs_x(nx, ny, nz, this%nsurf, eos, this%ref_pres, &
                              ms%h_layer, &
                              ms%tracers(ms%idx_temperature)%hTr, &
                              ms%tracers(ms%idx_salinity)%hTr, &
                              metrics%wet_u, &
                              this%uPoL, this%uPoR, this%uKoL, this%uKoR, this%uhEff)
      call redi_calc_coeffs_y(nx, ny, nz, this%nsurf, eos, this%ref_pres, &
                              ms%h_layer, &
                              ms%tracers(ms%idx_temperature)%hTr, &
                              ms%tracers(ms%idx_salinity)%hTr, &
                              metrics%wet_v, &
                              this%vPoL, this%vPoR, this%vKoL, this%vKoR, this%vhEff)
   end subroutine redi_calc_coeffs

   pure subroutine redi_build_column(nz, h_col, thtr_col, shtr_col, eos, &
                                     Pint, Tint, Sint, dRdT, dRdS)
      !! Build one column's TOP-DOWN interface P/T/S + density derivs from the
      !! BOTTOM-UP native column.  Layer T/S = hTr/h (floored); interface T/S =
      !! PPM edge reconstruction on the flipped arrays; interface P =
      !! surface-relative hydrostatic; dR/dT, dR/dS = -rho² dSV/dX at each
      !! interface.
      DC_ROUTINE_SEQ
      integer, intent(in) :: nz
      real(wp), intent(in) :: h_col(nz), thtr_col(nz), shtr_col(nz)
      type(ocean_eos_t), intent(in) :: eos
      real(wp), intent(out) :: Pint(nz + 1), Tint(nz + 1), Sint(nz + 1)
      real(wp), intent(out) :: dRdT(nz + 1), dRdS(nz + 1)
      real(wp) :: htd(NZ_STACK_MAX), ttd(NZ_STACK_MAX), std(NZ_STACK_MAX)
      real(wp) :: rho_i, dsv_dt, dsv_ds, he
      integer :: k, kf

      ! Flip native bottom-up -> top-down layer arrays.  Native layer k
      ! maps to top-down layer (nz+1-k).  Layer T/S = hTr/h (floored).
      do k = 1, nz
         kf = nz + 1 - k
         he = max(h_col(k), H_DIV_EPS)
         htd(kf) = h_col(k)
         ttd(kf) = thtr_col(k)/he
         std(kf) = shtr_col(k)/he
      end do

      ! PPM interface edge values (top-down) for T and S.
      call redi_interface_scalar(nz, htd, ttd, Tint)
      call redi_interface_scalar(nz, htd, std, Sint)

      ! Top-down interface pressure: K=1 surface (p=0), accumulate
      ! g*rho0*h downward.  rho0 from the EOS reference.
      Pint(1) = 0.0_wp
      do k = 1, nz
         Pint(k + 1) = Pint(k) + GRAVITY*eos%rho0*htd(k)
      end do

      ! Interface density derivs (locally referenced at the interface P).
      do k = 1, nz + 1
         rho_i = eos_density_point(eos, Tint(k), Sint(k), Pint(k))
         call eos_specvol_derivs(eos, Tint(k), Sint(k), Pint(k), dsv_dt, dsv_ds)
         dRdT(k) = -(rho_i*rho_i)*dsv_dt
         dRdS(k) = -(rho_i*rho_i)*dsv_ds
      end do
   end subroutine redi_build_column

   pure subroutine redi_face_coeffs(nz, ns, hL, tL, sL, hR, tR, sR, eos, &
                                    PoLo, PoRo, KoLo, KoRo, hEffo)
      !! The per-face Phase-A core: build both columns TOP-DOWN, run the
      !! sweep, and store PoL/PoR/KoL/KoR/hEff verbatim in the top-down frame.
      !! The k-flip is confined to the Phase-B scatter; keeping Po/Ko top-down
      !! here lets the flux re-use the sweep's interface-edge convention with
      !! no position arithmetic on the flipped frame.
      DC_ROUTINE_SEQ
      integer, intent(in) :: nz, ns
      real(wp), intent(in) :: hL(nz), tL(nz), sL(nz), hR(nz), tR(nz), sR(nz)
      type(ocean_eos_t), intent(in) :: eos
      real(wp), intent(out) :: PoLo(ns), PoRo(ns), hEffo(ns - 1)
      integer, intent(out) :: KoLo(ns), KoRo(ns)
      real(wp) :: Pl(NZ_STACK_MAX + 1), Tli(NZ_STACK_MAX + 1), Sli(NZ_STACK_MAX + 1)
      real(wp) :: dRdTl(NZ_STACK_MAX + 1), dRdSl(NZ_STACK_MAX + 1)
      real(wp) :: Pr(NZ_STACK_MAX + 1), Tri(NZ_STACK_MAX + 1), Sri(NZ_STACK_MAX + 1)
      real(wp) :: dRdTr(NZ_STACK_MAX + 1), dRdSr(NZ_STACK_MAX + 1)
      real(wp) :: pa_to_h
      integer :: ks

      call redi_build_column(nz, hL, tL, sL, eos, Pl, Tli, Sli, dRdTl, dRdSl)
      call redi_build_column(nz, hR, tR, sR, eos, Pr, Tri, Sri, dRdTr, dRdSr)

      call redi_neutral_positions_continuous(nz, Pl, Tli, Sli, dRdTl, dRdSl, &
                                             Pr, Tri, Sri, dRdTr, dRdSr, &
                                             PoLo, PoRo, KoLo, KoRo, hEffo)
      ! The continuous sweep computes hEff from interface-PRESSURE differences
      ! (Pa, since the position coordinate is the hydrostatic pressure
      ! g*rho0*h), but the flux divergence and Coef consume hEff in THICKNESS
      ! units (m).  Convert Pa -> m by dividing by H_to_pa = g*rho0 (MOM6
      ! neutral_diffusion lines ~588: uhEff /= H_to_pa).  Po/Ko stay TOP-DOWN;
      ! the only k-flip is the Phase-B scatter.
      pa_to_h = 1.0_wp/(GRAVITY*eos%rho0)
      do ks = 1, ns - 1
         hEffo(ks) = hEffo(ks)*pa_to_h
      end do
   end subroutine redi_face_coeffs

   subroutine redi_calc_coeffs_x(nx, ny, nz, ns, eos, ref_pres, h_layer, &
                                 t_htr, s_htr, wet_u, uPoL, uPoR, uKoL, uKoR, uhEff)
      !! Flat-impl Phase-A u-face kernel.  Parallel over (i,j) interior
      !! u-faces (i=2..nx); the 2*nz+2 sweep runs serially inside each
      !! thread over a column pair (iw=i-1 west, i east).  Wall faces and
      !! land faces leave the inert (zero/identity) coefficients.
      integer, intent(in) :: nx, ny, nz, ns
      type(ocean_eos_t), intent(in) :: eos
      real(wp), intent(in) :: ref_pres
      real(wp), intent(in) :: h_layer(nx, ny, nz)
      real(wp), intent(in) :: t_htr(nx, ny, nz)
      real(wp), intent(in) :: s_htr(nx, ny, nz)
      real(wp), intent(in) :: wet_u(nx + 1, ny)
      real(wp), intent(out) :: uPoL(nx + 1, ny, ns)
      real(wp), intent(out) :: uPoR(nx + 1, ny, ns)
      integer, intent(out) :: uKoL(nx + 1, ny, ns)
      integer, intent(out) :: uKoR(nx + 1, ny, ns)
      real(wp), intent(out) :: uhEff(nx + 1, ny, ns - 1)

      integer :: i, j, k, s
      real(wp) :: hL(NZ_STACK_MAX), tcL(NZ_STACK_MAX), scL(NZ_STACK_MAX)
      real(wp) :: hR(NZ_STACK_MAX), tcR(NZ_STACK_MAX), scR(NZ_STACK_MAX)
      real(wp) :: PoLc(2*NZ_STACK_MAX + 2), PoRc(2*NZ_STACK_MAX + 2)
      integer :: KoLc(2*NZ_STACK_MAX + 2), KoRc(2*NZ_STACK_MAX + 2)
      real(wp) :: hEc(2*NZ_STACK_MAX + 1)

      do j=1,ny
      do i=2,nx
         do s = 1, ns
            uPoL(i, j, s) = 0.0_wp
            uPoR(i, j, s) = 0.0_wp
            uKoL(i, j, s) = 1
            uKoR(i, j, s) = 1
         end do
         do s = 1, ns - 1
            uhEff(i, j, s) = 0.0_wp
         end do
         if (wet_u(i, j) > 0.0_wp) then
            do k = 1, nz
               hL(k) = h_layer(i - 1, j, k)
               tcL(k) = t_htr(i - 1, j, k)
               scL(k) = s_htr(i - 1, j, k)
               hR(k) = h_layer(i, j, k)
               tcR(k) = t_htr(i, j, k)
               scR(k) = s_htr(i, j, k)
            end do
            call redi_face_coeffs(nz, ns, hL, tcL, scL, hR, tcR, scR, eos, &
                                  PoLc, PoRc, KoLc, KoRc, hEc)
            do s = 1, ns
               uPoL(i, j, s) = PoLc(s)
               uPoR(i, j, s) = PoRc(s)
               uKoL(i, j, s) = KoLc(s)
               uKoR(i, j, s) = KoRc(s)
            end do
            do s = 1, ns - 1
               uhEff(i, j, s) = hEc(s)
            end do
         end if
      end do
      end do
   end subroutine redi_calc_coeffs_x

   subroutine redi_calc_coeffs_y(nx, ny, nz, ns, eos, ref_pres, h_layer, &
                                 t_htr, s_htr, wet_v, vPoL, vPoR, vKoL, vKoR, vhEff)
      !! Flat-impl Phase-A v-face kernel — mirror of `_x` with v-stagger
      !! (js=j-1 south, j north), loop j=2..ny.
      integer, intent(in) :: nx, ny, nz, ns
      type(ocean_eos_t), intent(in) :: eos
      real(wp), intent(in) :: ref_pres
      real(wp), intent(in) :: h_layer(nx, ny, nz)
      real(wp), intent(in) :: t_htr(nx, ny, nz)
      real(wp), intent(in) :: s_htr(nx, ny, nz)
      real(wp), intent(in) :: wet_v(nx, ny + 1)
      real(wp), intent(out) :: vPoL(nx, ny + 1, ns)
      real(wp), intent(out) :: vPoR(nx, ny + 1, ns)
      integer, intent(out) :: vKoL(nx, ny + 1, ns)
      integer, intent(out) :: vKoR(nx, ny + 1, ns)
      real(wp), intent(out) :: vhEff(nx, ny + 1, ns - 1)

      integer :: i, j, k, s
      real(wp) :: hL(NZ_STACK_MAX), tcL(NZ_STACK_MAX), scL(NZ_STACK_MAX)
      real(wp) :: hR(NZ_STACK_MAX), tcR(NZ_STACK_MAX), scR(NZ_STACK_MAX)
      real(wp) :: PoLc(2*NZ_STACK_MAX + 2), PoRc(2*NZ_STACK_MAX + 2)
      integer :: KoLc(2*NZ_STACK_MAX + 2), KoRc(2*NZ_STACK_MAX + 2)
      real(wp) :: hEc(2*NZ_STACK_MAX + 1)

      do j=2,ny
      do i=1,nx
         do s = 1, ns
            vPoL(i, j, s) = 0.0_wp
            vPoR(i, j, s) = 0.0_wp
            vKoL(i, j, s) = 1
            vKoR(i, j, s) = 1
         end do
         do s = 1, ns - 1
            vhEff(i, j, s) = 0.0_wp
         end do
         if (wet_v(i, j) > 0.0_wp) then
            do k = 1, nz
               hL(k) = h_layer(i, j - 1, k)
               tcL(k) = t_htr(i, j - 1, k)
               scL(k) = s_htr(i, j - 1, k)
               hR(k) = h_layer(i, j, k)
               tcR(k) = t_htr(i, j, k)
               scR(k) = s_htr(i, j, k)
            end do
            call redi_face_coeffs(nz, ns, hL, tcL, scL, hR, tcR, scR, eos, &
                                  PoLc, PoRc, KoLc, KoRc, hEc)
            do s = 1, ns
               vPoL(i, j, s) = PoLc(s)
               vPoR(i, j, s) = PoRc(s)
               vKoL(i, j, s) = KoLc(s)
               vKoR(i, j, s) = KoRc(s)
            end do
            do s = 1, ns - 1
               vhEff(i, j, s) = hEc(s)
            end do
         end if
      end do
      end do
   end subroutine redi_calc_coeffs_y

   ! =====================================================================
   ! Phase B — neutral_surface_flux (per tracer).  Reads the Phase-A
   ! coefficients (top-down frame), evaluates the along-neutral tracer
   ! difference via PPM at matched neutral positions, applies the
   ! down-gradient sign guard, scatters the triad flux to native layers (the
   ! single k-flip), and updates the tracer with the divergence.  Slot arrays
   ! are read by scalar index; only fixed-size NZ_STACK_MAX column copies pass
   ! to the `!$acc routine seq` helpers (no per-call descriptor temporaries).
   ! =====================================================================

   pure function redi_signum1(x) result(s)
      DC_ROUTINE_SEQ
      !! signum(1.,x): -1 if x<0, +1 if x>0, 0 if x==0 (MOM6 sign guard).
      real(wp), intent(in) :: x
      real(wp) :: s
      s = sign(1.0_wp, x)
      if (x == 0.0_wp) s = 0.0_wp
   end function redi_signum1

   pure function redi_ppm_ave(xL, xR, aL, aR, aMean) result(av)
      DC_ROUTINE_SEQ
      !! Mean of a PPM parabola between fractional positions xL,xR in [0,1]
      !! (MOM6 `ppm_ave`).  Device-safe: dx<0 / dx>1 FATALs collapse to the
      !! dx==0 branch value (no host I/O on device).
      real(wp), intent(in) :: xL, xR, aL, aR, aMean
      real(wp) :: av
      real(wp) :: dx, xave, a6, a6o3
      dx = xR - xL
      xave = 0.5_wp*(xR + xL)
      a6o3 = 2.0_wp*aMean - (aL + aR)
      a6 = 3.0_wp*a6o3
      if (dx > 0.0_wp .and. dx <= 1.0_wp) then
         av = (aL + xave*((aR - aL) + a6)) - a6o3*(xR*xR + xR*xL + xL*xL)
      else
         av = aL + (aR - aL)*xR + a6*xR*(1.0_wp - xR)
      end if
   end function redi_ppm_ave

   pure subroutine redi_tracer_column(nz, h_col, htr_col, Tlay, Tint, aLe, aRe)
      DC_ROUTINE_SEQ
      !! Build one column's TOP-DOWN layer-average tracer `Tlay`, PPM
      !! interface edges `Tint`, and per-layer limited PPM left/right edges
      !! `aLe/aRe` from the bottom-up native (h, hTr) column (fixed-size
      !! NZ_STACK_MAX copies).  Tlay = hTr/h floored.  Mirrors MOM6
      !! interface_scalar + ppm_left_right_edge_values.
      integer, intent(in) :: nz
      real(wp), intent(in) :: h_col(NZ_STACK_MAX), htr_col(NZ_STACK_MAX)
      real(wp), intent(out) :: Tlay(NZ_STACK_MAX), Tint(NZ_STACK_MAX + 1)
      real(wp), intent(out) :: aLe(NZ_STACK_MAX), aRe(NZ_STACK_MAX)
      real(wp) :: htd(NZ_STACK_MAX), tedge(NZ_STACK_MAX + 1)
      real(wp) :: he, alk, ark, tlk
      integer :: k, kf
      do k = 1, nz
         kf = nz + 1 - k
         he = max(h_col(k), H_DIV_EPS)
         htd(kf) = h_col(k)
         Tlay(kf) = htr_col(k)/he
      end do
      call redi_interface_scalar(nz, htd, Tlay, tedge)
      do k = 1, nz + 1
         Tint(k) = tedge(k)
      end do
      do k = 1, nz
         alk = Tint(k)
         ark = Tint(k + 1)
         tlk = Tlay(k)
         if (redi_signum1(ark - tlk)*redi_signum1(tlk - alk) <= 0.0_wp) then
            alk = tlk
            ark = tlk
         else if (sign(3.0_wp, ark - alk)*((tlk - alk) + (tlk - ark)) > abs(ark - alk)) then
            alk = tlk + 2.0_wp*(tlk - ark)
         else if (sign(3.0_wp, ark - alk)*((tlk - alk) + (tlk - ark)) < -abs(ark - alk)) then
            ark = tlk + 2.0_wp*(tlk - alk)
         end if
         aLe(k) = alk
         aRe(k) = ark
      end do
   end subroutine redi_tracer_column

   pure function redi_sublayer_dT(nz, klt, klb, krt, krb, &
                                  PoLt, PoLb, PoRt, PoRb, &
                                  TlL, TiL, aLL, aRL, TlR, TiR, aLR, aRR) result(dT)
      DC_ROUTINE_SEQ
      !! Along-neutral tracer difference for one sublayer (MOM6
      !! neutral_surface_flux continuous branch).  TOP-DOWN layer indices
      !! (klt/klb = KoL at the surface/bed bound of the sublayer; krt/krb
      !! mirror) and fractional positions.  Returns dT_layer when the
      !! top/bottom/ave/layer triad is sign-consistent, else 0 (the
      !! down-gradient guard that prevents up-gradient transport).
      integer, intent(in) :: nz, klt, klb, krt, krb
      real(wp), intent(in) :: PoLt, PoLb, PoRt, PoRb
      real(wp), intent(in) :: TlL(NZ_STACK_MAX), TiL(NZ_STACK_MAX + 1), aLL(NZ_STACK_MAX), aRL(NZ_STACK_MAX)
      real(wp), intent(in) :: TlR(NZ_STACK_MAX), TiR(NZ_STACK_MAX + 1), aLR(NZ_STACK_MAX), aRR(NZ_STACK_MAX)
      real(wp) :: dT
      real(wp) :: tlt, tlb, trt, trb, tlay, trlay, dT_top, dT_bot, dT_ave, dT_layer
      tlt = (1.0_wp - PoLt)*TiL(klt) + PoLt*TiL(klt + 1)
      tlb = (1.0_wp - PoLb)*TiL(klb) + PoLb*TiL(klb + 1)
      trt = (1.0_wp - PoRt)*TiR(krt) + PoRt*TiR(krt + 1)
      trb = (1.0_wp - PoRb)*TiR(krb) + PoRb*TiR(krb + 1)
      tlay = redi_ppm_ave(PoLt, PoLb + real(klb - klt, wp), aLL(klt), aRL(klt), TlL(klt))
      trlay = redi_ppm_ave(PoRt, PoRb + real(krb - krt, wp), aLR(krt), aRR(krt), TlR(krt))
      dT_top = trt - tlt
      dT_bot = trb - tlb
      dT_ave = 0.5_wp*(dT_top + dT_bot)
      dT_layer = trlay - tlay
      if (redi_signum1(dT_top)*redi_signum1(dT_bot) <= 0.0_wp .or. &
          redi_signum1(dT_ave)*redi_signum1(dT_layer) <= 0.0_wp) then
         dT = 0.0_wp
      else
         dT = dT_layer
      end if
   end function redi_sublayer_dT

   subroutine redi_apply_flux(grid, metrics, this, ms, dt, khtr_u_ext, khtr_v_ext, bc)
      !! Public Phase-B entry: apply the neutral-diffusion tracer update for
      !! every registered tracer.  No-op if absent / uninit / disabled / zero
      !! diffusivity.  Run at thermo cadence after `redi_calc_coeffs` and the
      !! along-coordinate `tracer_hdiff` (Redi augments it).  Per-face KhTr
      !! comes from `khtr_u_ext`/`khtr_v_ext` (VarMix) when supplied, else the
      !! scalar `this%khtr` broadcast onto every face.
      type(hgrid_t), intent(in) :: grid
      type(ocean_metrics_t), intent(in) :: metrics
      type(ocean_redi_t), intent(inout) :: this
      type(multilayer_cgrid_state_t), intent(inout) :: ms
      real(wp), intent(in) :: dt
      real(wp), intent(in), optional :: khtr_u_ext(:, :)
         !! VarMix per-face KhTr at u-faces `(nx+1, ny)` (m^2/s).
      real(wp), intent(in), optional :: khtr_v_ext(:, :)
         !! VarMix per-face KhTr at v-faces `(nx, ny+1)` (m^2/s).
      type(ocean_bc_state_t), intent(in), optional :: bc
         !! Per-edge OBC tags.  No along-isopycnal flux crosses a
         !! no-normal-flow (`OBC_WALL`) physical-domain boundary face — else
         !! Redi bleeds tracer into the ghost halo.  Absent ⇒ all edges WALL.
      integer :: nx, ny, nz, it
      integer :: nghost, nxp, nyp
      logical :: use_ext, wall_w, wall_e, wall_s, wall_n

      if (.not. this%is_init) return
      if (.not. this%enable) return
      if (.not. allocated(ms%h_layer)) return
      if (.not. allocated(ms%tracers)) return

      nx = grid%nx_total
      ny = grid%ny_total
      nz = ms%nz_ml
      if (this%nz_ml /= nz) return
      nghost = grid%nghost
      nxp = grid%nx_phys
      nyp = grid%ny_phys
      ! Which physical-domain edges are no-normal-flow walls (default closed).
      wall_w = .true.
      wall_e = .true.
      wall_s = .true.
      wall_n = .true.
      if (present(bc)) then
         wall_w = (bc%west%bc_type == OBC_WALL)
         wall_e = (bc%east%bc_type == OBC_WALL)
         wall_s = (bc%south%bc_type == OBC_WALL)
         wall_n = (bc%north%bc_type == OBC_WALL)
      end if

      use_ext = present(khtr_u_ext) .and. present(khtr_v_ext)
      ! No diffusivity anywhere => nothing to do (scalar path only; the
      ! VarMix field may be non-zero even when the scalar floor is 0).
      if (.not. use_ext .and. this%khtr <= 0.0_wp) return

      ! Resolve the per-face KhTr into the slot fields (device-resident).
      if (use_ext) then
         call redi_face_copy(nx + 1, ny, khtr_u_ext, this%khtr_u)
         call redi_face_copy(nx, ny + 1, khtr_v_ext, this%khtr_v)
      else
         call redi_face_const(nx + 1, ny, this%khtr, this%khtr_u)
         call redi_face_const(nx, ny + 1, this%khtr, this%khtr_v)
      end if

      do it = 1, size(ms%tracers)
         ! Snapshot this tracer's hTr so the double-visit gather reads the
         ! pre-step field (order-independent — see `tr_snap`).
         call redi_snapshot(nx, ny, nz, ms%tracers(it)%hTr, this%tr_snap)
         call redi_apply_flux_impl(nx, ny, nz, this%nsurf, dt, &
                                   nghost, nxp, nyp, wall_w, wall_e, wall_s, wall_n, &
                                   this%khtr_u, this%khtr_v, &
                                   metrics%dy_cu, metrics%dx_cv, &
                                   metrics%idxCu, metrics%idyCv, metrics%areaT, &
                                   ms%h_layer, this%tr_snap, ms%tracers(it)%hTr, &
                                   this%uPoL, this%uPoR, this%uKoL, this%uKoR, this%uhEff, &
                                   this%vPoL, this%vPoR, this%vKoL, this%vKoR, this%vhEff)
      end do
   end subroutine redi_apply_flux

   pure subroutine redi_snapshot(nx, ny, nz, src, dst)
      !! Device-side copy `dst = src` of a `(nx,ny,nz)` tracer field.
      integer, intent(in) :: nx, ny, nz
      real(wp), intent(in) :: src(nx, ny, nz)
      real(wp), intent(inout) :: dst(nx, ny, nz)
      integer :: i, j, k
      do k=1,nz
      do j=1,ny
      do i=1,nx
         dst(i, j, k) = src(i, j, k)
      end do
      end do
      end do
   end subroutine redi_snapshot

   pure subroutine redi_face_copy(n1, n2, src, dst)
      !! Copy a face KhTr field `src -> dst` (both `(n1, n2)`), device-side.
      integer, intent(in) :: n1, n2
      real(wp), intent(in) :: src(n1, n2)
      real(wp), intent(inout) :: dst(n1, n2)
      integer :: i, j
      do j=1,n2
      do i=1,n1
         dst(i, j) = src(i, j)
      end do
      end do
   end subroutine redi_face_copy

   pure subroutine redi_face_const(n1, n2, val, dst)
      !! Broadcast the scalar KhTr `val` onto every face of `dst` `(n1, n2)`.
      integer, intent(in) :: n1, n2
      real(wp), intent(in) :: val
      real(wp), intent(inout) :: dst(n1, n2)
      integer :: i, j
      do j=1,n2
      do i=1,n1
         dst(i, j) = val
      end do
      end do
   end subroutine redi_face_const

   pure subroutine redi_face_flux(nz, ns, nxc, nyc, nfa, nfb, h_layer, hTr_in, &
                                  iL, jL, iR, jR, fa, fb, &
                                  PoL, PoR, KoL, KoR, hEff, coef, is_left, dTr)
      DC_ROUTINE_SEQ
      !! Accumulate ONE C-grid face's neutral-surface tracer flux into the
      !! owning cell's `dTr`.  Builds the left/right tracer columns from the
      !! READ-ONLY snapshot `hTr_in` (the live `hTr` is also written by the
      !! loop — reading it would be a do-concurrent read-write race) and loops
      !! the `ns-1` neutral sublayers.  The per-face column locals live in this
      !! frame to keep the caller's loop-body footprint small.
      !! `is_left`: this cell is the LEFT (west/south) column ⇒ `+flx` into
      !! native layer `nz+1-KoL`; else the RIGHT column ⇒ `-flx` into
      !! `nz+1-KoR`.  `(iL,jL)`/`(iR,jR)` index the columns; `(fa,fb)` the faces.
      integer, intent(in) :: nz, ns, nxc, nyc, nfa, nfb
      integer, intent(in) :: iL, jL, iR, jR, fa, fb
      real(wp), intent(in) :: h_layer(nxc, nyc, nz), hTr_in(nxc, nyc, nz)
      real(wp), intent(in) :: PoL(nfa, nfb, ns), PoR(nfa, nfb, ns)
      integer, intent(in) :: KoL(nfa, nfb, ns), KoR(nfa, nfb, ns)
      real(wp), intent(in) :: hEff(nfa, nfb, ns - 1)
      real(wp), intent(in) :: coef
      logical, intent(in) :: is_left
      real(wp), intent(inout) :: dTr(nz)

      integer :: k, ks, knat
      real(wp) :: hcL(NZ_STACK_MAX), trcL(NZ_STACK_MAX)
      real(wp) :: hcR(NZ_STACK_MAX), trcR(NZ_STACK_MAX)
      real(wp) :: TlL(NZ_STACK_MAX), TiL(NZ_STACK_MAX + 1), aLL(NZ_STACK_MAX), aRL(NZ_STACK_MAX)
      real(wp) :: TlR(NZ_STACK_MAX), TiR(NZ_STACK_MAX + 1), aLR(NZ_STACK_MAX), aRR(NZ_STACK_MAX)
      real(wp) :: dtdiff, flx

      do k = 1, nz
         hcL(k) = h_layer(iL, jL, k)
         trcL(k) = hTr_in(iL, jL, k)
         hcR(k) = h_layer(iR, jR, k)
         trcR(k) = hTr_in(iR, jR, k)
      end do
      call redi_tracer_column(nz, hcL, trcL, TlL, TiL, aLL, aRL)
      call redi_tracer_column(nz, hcR, trcR, TlR, TiR, aLR, aRR)
      do ks = 1, ns - 1
         if (hEff(fa, fb, ks) /= 0.0_wp) then
            dtdiff = redi_sublayer_dT(nz, KoL(fa, fb, ks), KoL(fa, fb, ks + 1), &
                                      KoR(fa, fb, ks), KoR(fa, fb, ks + 1), &
                                      PoL(fa, fb, ks), PoL(fa, fb, ks + 1), &
                                      PoR(fa, fb, ks), PoR(fa, fb, ks + 1), &
                                      TlL, TiL, aLL, aRL, TlR, TiR, aLR, aRR)
            flx = dtdiff*hEff(fa, fb, ks)*coef
            if (is_left) then
               knat = nz + 1 - KoL(fa, fb, ks)
               dTr(knat) = dTr(knat) + flx
            else
               knat = nz + 1 - KoR(fa, fb, ks)
               dTr(knat) = dTr(knat) - flx
            end if
         end if
      end do
   end subroutine redi_face_flux

   subroutine redi_apply_flux_impl(nx, ny, nz, ns, dt, &
                                   nghost, nxp, nyp, wall_w, wall_e, wall_s, wall_n, &
                                   khtr_u, khtr_v, dy_cu, dx_cv, &
                                   idxCu, idyCv, areaT, h_layer, hTr_in, hTr, &
                                   uPoL, uPoR, uKoL, uKoR, uhEff, &
                                   vPoL, vPoR, vKoL, vKoR, vhEff)
      !! Flat-impl Phase-B kernel for ONE tracer.  Cell-centric double-visit
      !! (no-scatter rule on the C-grid): cell (i,j) recomputes the
      !! along-neutral flux on each of its four bounding faces and accumulates
      !! ONLY into its own `dTr`; interior-face fluxes are computed twice but
      !! no thread writes a neighbour ⇒ race-free.
      !!   Face sign: the LEFT (west/south) cell of a face gets +Flx into
      !!   native layer nz+1-KoL; the RIGHT (east/north) cell gets -Flx into
      !!   nz+1-KoR.  The top-down→native flip k=nz+1-Ko is the only k-flip.
      !!   Flux = dT_layer * hEff * Coef, Coef_u = dt*khtr_u*dy_cu*idxCu;
      !!   divergence hTr(k) += dTr(k)/areaT (conservative).
      integer, intent(in) :: nx, ny, nz, ns
      real(wp), intent(in) :: dt
      integer, intent(in) :: nghost, nxp, nyp
      logical, intent(in) :: wall_w, wall_e, wall_s, wall_n
      real(wp), intent(in) :: khtr_u(nx + 1, ny)
      real(wp), intent(in) :: khtr_v(nx, ny + 1)
      real(wp), intent(in) :: dy_cu(nx + 1, ny)
      real(wp), intent(in) :: dx_cv(nx, ny + 1)
      real(wp), intent(in) :: idxCu(nx + 1, ny)
      real(wp), intent(in) :: idyCv(nx, ny + 1)
      real(wp), intent(in) :: areaT(nx, ny)
      real(wp), intent(in) :: h_layer(nx, ny, nz)
      real(wp), intent(in) :: hTr_in(nx, ny, nz)
      real(wp), intent(inout) :: hTr(nx, ny, nz)
      real(wp), intent(in) :: uPoL(nx + 1, ny, ns), uPoR(nx + 1, ny, ns)
      integer, intent(in) :: uKoL(nx + 1, ny, ns), uKoR(nx + 1, ny, ns)
      real(wp), intent(in) :: uhEff(nx + 1, ny, ns - 1)
      real(wp), intent(in) :: vPoL(nx, ny + 1, ns), vPoR(nx, ny + 1, ns)
      integer, intent(in) :: vKoL(nx, ny + 1, ns), vKoR(nx, ny + 1, ns)
      real(wp), intent(in) :: vhEff(nx, ny + 1, ns - 1)

      integer :: i, j, k
      integer :: wuf_w, wuf_e, wvf_s, wvf_n
      real(wp) :: dTr(NZ_STACK_MAX)
      real(wp) :: iaij

      ! No-normal-flow WALL faces of the physical domain.  The Redi flux must
      ! NOT cross these — else it bleeds tracer into the ghost halo.  A wall
      ! face is skipped by both adjacent cells, so the +flx/-flx pair never forms.
      wuf_w = nghost + 1
      wuf_e = nghost + nxp + 1
      wvf_s = nghost + 1
      wvf_n = nghost + nyp + 1

      ! Per-cell double-visit: each interior face is computed by both adjacent
      ! cells (race-free — every cell writes only its own dTr).  The heavy
      ! column locals live inside `redi_face_flux`, keeping this loop body's
      ! footprint to `dTr`/`iaij` (avoids the gfortran-15.1 dc-local corruption).
      do j=1,ny
      do i=1,nx
         do k = 1, nz
            dTr(k) = 0.0_wp
         end do
         ! WEST u-face (face i): this cell is the RIGHT column.
         if (i >= 2 .and. .not. ((wall_w .and. i == wuf_w) .or. (wall_e .and. i == wuf_e))) then
            call redi_face_flux(nz, ns, nx, ny, nx + 1, ny, h_layer, hTr_in, &
                                i - 1, j, i, j, i, j, uPoL, uPoR, uKoL, uKoR, uhEff, &
                                dt*khtr_u(i, j)*dy_cu(i, j)*idxCu(i, j), .false., dTr)
         end if
         ! EAST u-face (face i+1): this cell is the LEFT column.
         if (i <= nx - 1 .and. .not. ((wall_w .and. i + 1 == wuf_w) .or. (wall_e .and. i + 1 == wuf_e))) then
            call redi_face_flux(nz, ns, nx, ny, nx + 1, ny, h_layer, hTr_in, &
                                i, j, i + 1, j, i + 1, j, uPoL, uPoR, uKoL, uKoR, uhEff, &
                                dt*khtr_u(i + 1, j)*dy_cu(i + 1, j)*idxCu(i + 1, j), .true., dTr)
         end if
         ! SOUTH v-face (face j): this cell is the RIGHT (north) column.
         if (j >= 2 .and. .not. ((wall_s .and. j == wvf_s) .or. (wall_n .and. j == wvf_n))) then
            call redi_face_flux(nz, ns, nx, ny, nx, ny + 1, h_layer, hTr_in, &
                                i, j - 1, i, j, i, j, vPoL, vPoR, vKoL, vKoR, vhEff, &
                                dt*khtr_v(i, j)*dx_cv(i, j)*idyCv(i, j), .false., dTr)
         end if
         ! NORTH v-face (face j+1): this cell is the LEFT (south) column.
         if (j <= ny - 1 .and. .not. ((wall_s .and. j + 1 == wvf_s) .or. (wall_n .and. j + 1 == wvf_n))) then
            call redi_face_flux(nz, ns, nx, ny, nx, ny + 1, h_layer, hTr_in, &
                                i, j, i, j + 1, i, j + 1, vPoL, vPoR, vKoL, vKoR, vhEff, &
                                dt*khtr_v(i, j + 1)*dx_cv(i, j + 1)*idyCv(i, j + 1), .true., dTr)
         end if
         iaij = 1.0_wp/areaT(i, j)
         do k = 1, nz
            hTr(i, j, k) = hTr(i, j, k) + dTr(k)*iaij
         end do
      end do
      end do
   end subroutine redi_apply_flux_impl

   ! =====================================================================
   ! OPTIMIZED apply-flux — precompute hoist (portable-Fortran analogue of
   ! the CUDA `kOptApplyFlux`). Bit-identical to redi_apply_flux.
   !
   ! The faithful path visits every T-cell and calls redi_face_flux on each
   ! of its four bounding faces (W/E/S/N). redi_face_flux reconstructs BOTH
   ! its left and right tracer column from scratch -> 8 column rebuilds per
   ! cell, but only 5 distinct columns exist: the centre (i,j) is rebuilt 4x
   ! (it is a column of ALL four faces), each of the 4 neighbours once.
   !
   ! redi_cell_flux reconstructs the centre column ONCE per cell into frame
   ! locals and reuses it across all four faces via redi_face_flux_hoist,
   ! which rebuilds only the distinct neighbour column -> 8 rebuilds -> 5.
   ! redi_tracer_column is a deterministic pure function of the column data,
   ! so computing the centre once vs four times yields the SAME BITS; the
   ! W,E,S,N face order and the ks=1..ns-1 accumulation order into dTr are
   ! preserved exactly. Hence bit-exact, not merely FMA-level.
   ! =====================================================================
   subroutine redi_apply_flux_hoist(grid, metrics, this, ms, dt, khtr_u_ext, khtr_v_ext, bc)
      !! Optimized public Phase-B entry. Identical contract to redi_apply_flux;
      !! see that routine for argument semantics. Only the per-cell column
      !! reconstruction is hoisted (5 rebuilds/cell instead of 8).
      type(hgrid_t), intent(in) :: grid
      type(ocean_metrics_t), intent(in) :: metrics
      type(ocean_redi_t), intent(inout) :: this
      type(multilayer_cgrid_state_t), intent(inout) :: ms
      real(wp), intent(in) :: dt
      real(wp), intent(in), optional :: khtr_u_ext(:, :)
      real(wp), intent(in), optional :: khtr_v_ext(:, :)
      type(ocean_bc_state_t), intent(in), optional :: bc
      integer :: nx, ny, nz, it
      integer :: nghost, nxp, nyp
      logical :: use_ext, wall_w, wall_e, wall_s, wall_n

      if (.not. this%is_init) return
      if (.not. this%enable) return
      if (.not. allocated(ms%h_layer)) return
      if (.not. allocated(ms%tracers)) return

      nx = grid%nx_total
      ny = grid%ny_total
      nz = ms%nz_ml
      if (this%nz_ml /= nz) return
      nghost = grid%nghost
      nxp = grid%nx_phys
      nyp = grid%ny_phys
      wall_w = .true.
      wall_e = .true.
      wall_s = .true.
      wall_n = .true.
      if (present(bc)) then
         wall_w = (bc%west%bc_type == OBC_WALL)
         wall_e = (bc%east%bc_type == OBC_WALL)
         wall_s = (bc%south%bc_type == OBC_WALL)
         wall_n = (bc%north%bc_type == OBC_WALL)
      end if

      use_ext = present(khtr_u_ext) .and. present(khtr_v_ext)
      if (.not. use_ext .and. this%khtr <= 0.0_wp) return

      if (use_ext) then
         call redi_face_copy(nx + 1, ny, khtr_u_ext, this%khtr_u)
         call redi_face_copy(nx, ny + 1, khtr_v_ext, this%khtr_v)
      else
         call redi_face_const(nx + 1, ny, this%khtr, this%khtr_u)
         call redi_face_const(nx, ny + 1, this%khtr, this%khtr_v)
      end if

      do it = 1, size(ms%tracers)
         call redi_snapshot(nx, ny, nz, ms%tracers(it)%hTr, this%tr_snap)
         call redi_apply_flux_hoist_impl(nx, ny, nz, this%nsurf, dt, &
                                         nghost, nxp, nyp, wall_w, wall_e, wall_s, wall_n, &
                                         this%khtr_u, this%khtr_v, &
                                         metrics%dy_cu, metrics%dx_cv, &
                                         metrics%idxCu, metrics%idyCv, metrics%areaT, &
                                         ms%h_layer, this%tr_snap, ms%tracers(it)%hTr, &
                                         this%uPoL, this%uPoR, this%uKoL, this%uKoR, this%uhEff, &
                                         this%vPoL, this%vPoR, this%vKoL, this%vKoR, this%vhEff)
      end do
   end subroutine redi_apply_flux_hoist

   pure subroutine redi_face_flux_hoist(nz, ns, nxc, nyc, nfa, nfb, h_layer, hTr_in, &
                                        iN, jN, fa, fb, PoL, PoR, KoL, KoR, hEff, coef, &
                                        centre_is_left, TlC, TiC, aLC, aRC, dTr)
      DC_ROUTINE_SEQ
      !! One C-grid face flux, given this cell's PRE-BUILT centre column
      !! (TlC/TiC/aLC/aRC). Rebuilds only the distinct NEIGHBOUR column
      !! `(iN,jN)` and runs the ns-1 sublayer sweep. `centre_is_left` = this
      !! (centre) cell is the geometric LEFT (west/south) column of the face:
      !! then L := centre, R := neighbour and +flx goes into nz+1-KoL; else the
      !! centre is the RIGHT column, L := neighbour, R := centre and -flx goes
      !! into nz+1-KoR. Bit-identical to redi_face_flux: the L/R argument order
      !! into redi_sublayer_dT and the +KoL / -KoR sign rule are preserved.
      integer, intent(in) :: nz, ns, nxc, nyc, nfa, nfb
      integer, intent(in) :: iN, jN, fa, fb
      real(wp), intent(in) :: h_layer(nxc, nyc, nz), hTr_in(nxc, nyc, nz)
      real(wp), intent(in) :: PoL(nfa, nfb, ns), PoR(nfa, nfb, ns)
      integer, intent(in) :: KoL(nfa, nfb, ns), KoR(nfa, nfb, ns)
      real(wp), intent(in) :: hEff(nfa, nfb, ns - 1)
      real(wp), intent(in) :: coef
      logical, intent(in) :: centre_is_left
      real(wp), intent(in) :: TlC(NZ_STACK_MAX), TiC(NZ_STACK_MAX + 1), aLC(NZ_STACK_MAX), aRC(NZ_STACK_MAX)
      real(wp), intent(inout) :: dTr(nz)

      integer :: k, ks, knat
      real(wp) :: hcN(NZ_STACK_MAX), trcN(NZ_STACK_MAX)
      real(wp) :: TlN(NZ_STACK_MAX), TiN(NZ_STACK_MAX + 1), aLN(NZ_STACK_MAX), aRN(NZ_STACK_MAX)
      real(wp) :: dtdiff, flx

      do k = 1, nz
         hcN(k) = h_layer(iN, jN, k)
         trcN(k) = hTr_in(iN, jN, k)
      end do
      call redi_tracer_column(nz, hcN, trcN, TlN, TiN, aLN, aRN)
      do ks = 1, ns - 1
         if (hEff(fa, fb, ks) /= 0.0_wp) then
            if (centre_is_left) then
               dtdiff = redi_sublayer_dT(nz, KoL(fa, fb, ks), KoL(fa, fb, ks + 1), &
                                         KoR(fa, fb, ks), KoR(fa, fb, ks + 1), &
                                         PoL(fa, fb, ks), PoL(fa, fb, ks + 1), &
                                         PoR(fa, fb, ks), PoR(fa, fb, ks + 1), &
                                         TlC, TiC, aLC, aRC, TlN, TiN, aLN, aRN)
               flx = dtdiff*hEff(fa, fb, ks)*coef
               knat = nz + 1 - KoL(fa, fb, ks)
               dTr(knat) = dTr(knat) + flx
            else
               dtdiff = redi_sublayer_dT(nz, KoL(fa, fb, ks), KoL(fa, fb, ks + 1), &
                                         KoR(fa, fb, ks), KoR(fa, fb, ks + 1), &
                                         PoL(fa, fb, ks), PoL(fa, fb, ks + 1), &
                                         PoR(fa, fb, ks), PoR(fa, fb, ks + 1), &
                                         TlN, TiN, aLN, aRN, TlC, TiC, aLC, aRC)
               flx = dtdiff*hEff(fa, fb, ks)*coef
               knat = nz + 1 - KoR(fa, fb, ks)
               dTr(knat) = dTr(knat) - flx
            end if
         end if
      end do
   end subroutine redi_face_flux_hoist

   pure subroutine redi_cell_flux(i, j, nx, ny, nz, ns, dt, &
                                  wall_w, wall_e, wall_s, wall_n, &
                                  wuf_w, wuf_e, wvf_s, wvf_n, &
                                  khtr_u, khtr_v, dy_cu, dx_cv, idxCu, idyCv, &
                                  h_layer, hTr_in, &
                                  uPoL, uPoR, uKoL, uKoR, uhEff, &
                                  vPoL, vPoR, vKoL, vKoR, vhEff, dTr)
      DC_ROUTINE_SEQ
      !! Accumulate the four bounding-face Redi fluxes into cell (i,j)'s dTr,
      !! reconstructing the CENTRE column (i,j) ONCE (the hoist) and reusing it
      !! across all four faces. W,E,S,N order and per-face wall guards are
      !! identical to redi_apply_flux_impl's loop body.
      integer, intent(in) :: i, j, nx, ny, nz, ns
      real(wp), intent(in) :: dt
      logical, intent(in) :: wall_w, wall_e, wall_s, wall_n
      integer, intent(in) :: wuf_w, wuf_e, wvf_s, wvf_n
      real(wp), intent(in) :: khtr_u(nx + 1, ny), khtr_v(nx, ny + 1)
      real(wp), intent(in) :: dy_cu(nx + 1, ny), dx_cv(nx, ny + 1)
      real(wp), intent(in) :: idxCu(nx + 1, ny), idyCv(nx, ny + 1)
      real(wp), intent(in) :: h_layer(nx, ny, nz), hTr_in(nx, ny, nz)
      real(wp), intent(in) :: uPoL(nx + 1, ny, ns), uPoR(nx + 1, ny, ns)
      integer, intent(in) :: uKoL(nx + 1, ny, ns), uKoR(nx + 1, ny, ns)
      real(wp), intent(in) :: uhEff(nx + 1, ny, ns - 1)
      real(wp), intent(in) :: vPoL(nx, ny + 1, ns), vPoR(nx, ny + 1, ns)
      integer, intent(in) :: vKoL(nx, ny + 1, ns), vKoR(nx, ny + 1, ns)
      real(wp), intent(in) :: vhEff(nx, ny + 1, ns - 1)
      real(wp), intent(inout) :: dTr(nz)

      integer :: k
      real(wp) :: hcC(NZ_STACK_MAX), trcC(NZ_STACK_MAX)
      real(wp) :: TlC(NZ_STACK_MAX), TiC(NZ_STACK_MAX + 1), aLC(NZ_STACK_MAX), aRC(NZ_STACK_MAX)

      ! ---- HOIST: build this cell's centre column ONCE (was rebuilt 4x) ----
      do k = 1, nz
         hcC(k) = h_layer(i, j, k)
         trcC(k) = hTr_in(i, j, k)
      end do
      call redi_tracer_column(nz, hcC, trcC, TlC, TiC, aLC, aRC)

      ! WEST u-face (face i): centre is the RIGHT column; neighbour (i-1,j) LEFT.
      if (i >= 2 .and. .not. ((wall_w .and. i == wuf_w) .or. (wall_e .and. i == wuf_e))) then
         call redi_face_flux_hoist(nz, ns, nx, ny, nx + 1, ny, h_layer, hTr_in, &
                                   i - 1, j, i, j, uPoL, uPoR, uKoL, uKoR, uhEff, &
                                   dt*khtr_u(i, j)*dy_cu(i, j)*idxCu(i, j), .false., &
                                   TlC, TiC, aLC, aRC, dTr)
      end if
      ! EAST u-face (face i+1): centre is the LEFT column; neighbour (i+1,j) RIGHT.
      if (i <= nx - 1 .and. .not. ((wall_w .and. i + 1 == wuf_w) .or. (wall_e .and. i + 1 == wuf_e))) then
         call redi_face_flux_hoist(nz, ns, nx, ny, nx + 1, ny, h_layer, hTr_in, &
                                   i + 1, j, i + 1, j, uPoL, uPoR, uKoL, uKoR, uhEff, &
                                   dt*khtr_u(i + 1, j)*dy_cu(i + 1, j)*idxCu(i + 1, j), .true., &
                                   TlC, TiC, aLC, aRC, dTr)
      end if
      ! SOUTH v-face (face j): centre is the RIGHT (north) column; neighbour (i,j-1) LEFT.
      if (j >= 2 .and. .not. ((wall_s .and. j == wvf_s) .or. (wall_n .and. j == wvf_n))) then
         call redi_face_flux_hoist(nz, ns, nx, ny, nx, ny + 1, h_layer, hTr_in, &
                                   i, j - 1, i, j, vPoL, vPoR, vKoL, vKoR, vhEff, &
                                   dt*khtr_v(i, j)*dx_cv(i, j)*idyCv(i, j), .false., &
                                   TlC, TiC, aLC, aRC, dTr)
      end if
      ! NORTH v-face (face j+1): centre is the LEFT (south) column; neighbour (i,j+1) RIGHT.
      if (j <= ny - 1 .and. .not. ((wall_s .and. j + 1 == wvf_s) .or. (wall_n .and. j + 1 == wvf_n))) then
         call redi_face_flux_hoist(nz, ns, nx, ny, nx, ny + 1, h_layer, hTr_in, &
                                   i, j + 1, i, j + 1, vPoL, vPoR, vKoL, vKoR, vhEff, &
                                   dt*khtr_v(i, j + 1)*dx_cv(i, j + 1)*idyCv(i, j + 1), .true., &
                                   TlC, TiC, aLC, aRC, dTr)
      end if
   end subroutine redi_cell_flux

   subroutine redi_apply_flux_hoist_impl(nx, ny, nz, ns, dt, &
                                         nghost, nxp, nyp, wall_w, wall_e, wall_s, wall_n, &
                                         khtr_u, khtr_v, dy_cu, dx_cv, &
                                         idxCu, idyCv, areaT, h_layer, hTr_in, hTr, &
                                         uPoL, uPoR, uKoL, uKoR, uhEff, &
                                         vPoL, vPoR, vKoL, vKoR, vhEff)
      !! Hoisted Phase-B kernel for ONE tracer. Same cell-centric double-visit
      !! and divergence as redi_apply_flux_impl; the four faces are handled by
      !! redi_cell_flux, which builds the centre column once. Loop-body DC
      !! footprint stays local(k, dTr, iaij) -- the heavy column arrays live in
      !! redi_cell_flux's frame, exactly as redi_face_flux did for the faithful path.
      integer, intent(in) :: nx, ny, nz, ns
      real(wp), intent(in) :: dt
      integer, intent(in) :: nghost, nxp, nyp
      logical, intent(in) :: wall_w, wall_e, wall_s, wall_n
      real(wp), intent(in) :: khtr_u(nx + 1, ny)
      real(wp), intent(in) :: khtr_v(nx, ny + 1)
      real(wp), intent(in) :: dy_cu(nx + 1, ny)
      real(wp), intent(in) :: dx_cv(nx, ny + 1)
      real(wp), intent(in) :: idxCu(nx + 1, ny)
      real(wp), intent(in) :: idyCv(nx, ny + 1)
      real(wp), intent(in) :: areaT(nx, ny)
      real(wp), intent(in) :: h_layer(nx, ny, nz)
      real(wp), intent(in) :: hTr_in(nx, ny, nz)
      real(wp), intent(inout) :: hTr(nx, ny, nz)
      real(wp), intent(in) :: uPoL(nx + 1, ny, ns), uPoR(nx + 1, ny, ns)
      integer, intent(in) :: uKoL(nx + 1, ny, ns), uKoR(nx + 1, ny, ns)
      real(wp), intent(in) :: uhEff(nx + 1, ny, ns - 1)
      real(wp), intent(in) :: vPoL(nx, ny + 1, ns), vPoR(nx, ny + 1, ns)
      integer, intent(in) :: vKoL(nx, ny + 1, ns), vKoR(nx, ny + 1, ns)
      real(wp), intent(in) :: vhEff(nx, ny + 1, ns - 1)

      integer :: i, j, k
      integer :: wuf_w, wuf_e, wvf_s, wvf_n
      real(wp) :: dTr(NZ_STACK_MAX)
      real(wp) :: iaij

      wuf_w = nghost + 1
      wuf_e = nghost + nxp + 1
      wvf_s = nghost + 1
      wvf_n = nghost + nyp + 1

      do j=1,ny
      do i=1,nx
         do k = 1, nz
            dTr(k) = 0.0_wp
         end do
         call redi_cell_flux(i, j, nx, ny, nz, ns, dt, &
                             wall_w, wall_e, wall_s, wall_n, &
                             wuf_w, wuf_e, wvf_s, wvf_n, &
                             khtr_u, khtr_v, dy_cu, dx_cv, idxCu, idyCv, &
                             h_layer, hTr_in, &
                             uPoL, uPoR, uKoL, uKoR, uhEff, &
                             vPoL, vPoR, vKoL, vKoR, vhEff, dTr)
         iaij = 1.0_wp/areaT(i, j)
         do k = 1, nz
            hTr(i, j, k) = hTr(i, j, k) + dTr(k)*iaij
         end do
      end do
      end do
   end subroutine redi_apply_flux_hoist_impl
end module ocean_redi
