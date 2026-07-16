!! ============================================================================
!! PRECOMP — the portable-Fortran control variant for Redi's Phase B.
!!
!! WHAT THIS TESTS. Production's `redi_apply_flux_impl` is a cell-centric
!! double-visit: cell (i,j) recomputes the flux on each of its 4 bounding faces
!! and `redi_face_flux` rebuilds BOTH adjacent tracer columns from scratch
!! (`redi_tracer_column` x2). That is **8 full PPM column reconstructions per
!! cell** — a plm_diff + (nz-1) ppm_edge + a limiter sweep, each — when only 5
!! distinct columns (self + 4 neighbours) exist, and every column in the domain
!! is rebuilt ~8 times over. The redundancy is ALGORITHMIC: it is in the
!! Fortran and in the CUDA port alike, and neither compiler can remove it (the
!! rebuild lives behind a `!$acc routine seq` call, across threads).
!!
!! THE TRANSFORM. Hoist the column reconstruction out of the flux loop:
!!   pass 1 (new kernel)  every cell builds ITS OWN column ONCE -> Tlay/Tint/
!!                        aLe/aRe in device-resident (nx,ny,*) scratch
!!   pass 2 (flux)        reads those by scalar index; no rebuild, and no
!!                        per-thread column arrays at all
!! 8 reconstructions per cell -> 1. Same arithmetic in the same order, so it is
!! BIT-IDENTICAL rather than an approximation.
!!
!! SECOND-ORDER EFFECT, and the reason this is worth more than the 8x suggests:
!! the per-thread stack arrays leave with the rebuild. Production's
!! `redi_face_flux` holds 12 NZ_STACK_MAX-sized locals (hcL/trcL/hcR/trcR +
!! TlL/TiL/aLL/aRL + TlR/TiR/aLR/aRR); PRECOMP's flux loop holds only `dTr`.
!! Redi's measured bottleneck is thread-local memory traffic at 10.8% occupancy
!! (README), so deleting the locals attacks the bottleneck directly.
!!
!! COST. Four extra device arrays of ~(nx,ny,nz) = ~139 MB at the 0.1 deg
!! config, reused across tracers. And Phase B stops being a single fused pass.
!!
!! DEPENDENCY CHECK (done, not assumed). Pass 1 -> pass 2 IS a real barrier:
!! pass 2 reads the columns of NEIGHBOURS that pass 1 wrote, so the two cannot
!! be fused — exactly the "loop reading NEIGHBOURS of what the previous wrote"
!! case. Two kernels is that dependency, not an implementation limit. Pass 1
!! itself is race-free (each cell writes only its own column) and reads only
!! the read-only snapshot `hTr_in`.
!! ============================================================================
module ocean_redi_pre
   use constants, only: wp, NZ_STACK_MAX, H_DIV_EPS
   use ocean_redi, only: redi_interface_scalar, redi_signum1, redi_ppm_ave
   implicit none
   private

   public :: redi_tracer_precompute, redi_apply_flux_impl_pre

contains

   subroutine redi_tracer_precompute(nx, ny, nz, h_layer, hTr_in, Tlay, Tint, aLe, aRe)
      !! PASS 1. Each cell builds its own TOP-DOWN column ONCE. The body is
      !! `redi_tracer_column` (ocean_redi.F90:818-857) verbatim, with the
      !! outputs written to global scratch instead of the caller's stack.
      integer, intent(in) :: nx, ny, nz
      real(wp), intent(in) :: h_layer(nx, ny, nz), hTr_in(nx, ny, nz)
      real(wp), intent(out) :: Tlay(nx, ny, nz), Tint(nx, ny, nz + 1)
      real(wp), intent(out) :: aLe(nx, ny, nz), aRe(nx, ny, nz)

      integer :: i, j, k, kf
      real(wp) :: htd(NZ_STACK_MAX), tld(NZ_STACK_MAX), tedge(NZ_STACK_MAX + 1)
      real(wp) :: he, alk, ark, tlk

      do concurrent(j=1:ny, i=1:nx) local(k, kf, htd, tld, tedge, he, alk, ark, tlk)
         do k = 1, nz
            kf = nz + 1 - k
            he = max(h_layer(i, j, k), H_DIV_EPS)
            htd(kf) = h_layer(i, j, k)
            tld(kf) = hTr_in(i, j, k)/he
         end do
         call redi_interface_scalar(nz, htd, tld, tedge)
         do k = 1, nz + 1
            Tint(i, j, k) = tedge(k)
         end do
         do k = 1, nz
            Tlay(i, j, k) = tld(k)
            alk = tedge(k)
            ark = tedge(k + 1)
            tlk = tld(k)
            if (redi_signum1(ark - tlk)*redi_signum1(tlk - alk) <= 0.0_wp) then
               alk = tlk
               ark = tlk
            else if (sign(3.0_wp, ark - alk)*((tlk - alk) + (tlk - ark)) > abs(ark - alk)) then
               alk = tlk + 2.0_wp*(tlk - ark)
            else if (sign(3.0_wp, ark - alk)*((tlk - alk) + (tlk - ark)) < -abs(ark - alk)) then
               ark = tlk + 2.0_wp*(tlk - alk)
            end if
            aLe(i, j, k) = alk
            aRe(i, j, k) = ark
         end do
      end do
   end subroutine redi_tracer_precompute

   pure function redi_sublayer_dT_g(nx, ny, nz, iL, jL, iR, jR, klt, klb, krt, krb, &
                                    PoLt, PoLb, PoRt, PoRb, Tlay, Tint, aLe, aRe) result(dT)
      !$acc routine seq
      !! `redi_sublayer_dT` (:859-891) with the eight column arrays replaced by
      !! scalar reads out of the precomputed global scratch. Identical
      !! expressions, identical order => bit-identical.
      integer, intent(in) :: nx, ny, nz, iL, jL, iR, jR, klt, klb, krt, krb
      real(wp), intent(in) :: PoLt, PoLb, PoRt, PoRb
      real(wp), intent(in) :: Tlay(nx, ny, nz), Tint(nx, ny, nz + 1)
      real(wp), intent(in) :: aLe(nx, ny, nz), aRe(nx, ny, nz)
      real(wp) :: dT
      real(wp) :: tlt, tlb, trt, trb, tlay_, trlay, dT_top, dT_bot, dT_ave, dT_layer

      tlt = (1.0_wp - PoLt)*Tint(iL, jL, klt) + PoLt*Tint(iL, jL, klt + 1)
      tlb = (1.0_wp - PoLb)*Tint(iL, jL, klb) + PoLb*Tint(iL, jL, klb + 1)
      trt = (1.0_wp - PoRt)*Tint(iR, jR, krt) + PoRt*Tint(iR, jR, krt + 1)
      trb = (1.0_wp - PoRb)*Tint(iR, jR, krb) + PoRb*Tint(iR, jR, krb + 1)
      tlay_ = redi_ppm_ave(PoLt, PoLb + real(klb - klt, wp), aLe(iL, jL, klt), aRe(iL, jL, klt), Tlay(iL, jL, klt))
      trlay = redi_ppm_ave(PoRt, PoRb + real(krb - krt, wp), aLe(iR, jR, krt), aRe(iR, jR, krt), Tlay(iR, jR, krt))
      dT_top = trt - tlt
      dT_bot = trb - tlb
      dT_ave = 0.5_wp*(dT_top + dT_bot)
      dT_layer = trlay - tlay_
      if (redi_signum1(dT_top)*redi_signum1(dT_bot) <= 0.0_wp .or. &
          redi_signum1(dT_ave)*redi_signum1(dT_layer) <= 0.0_wp) then
         dT = 0.0_wp
      else
         dT = dT_layer
      end if
   end function redi_sublayer_dT_g

   pure subroutine redi_face_flux_g(nz, ns, nxc, nyc, nfa, nfb, &
                                    iL, jL, iR, jR, fa, fb, &
                                    PoL, PoR, KoL, KoR, hEff, coef, is_left, &
                                    Tlay, Tint, aLe, aRe, dTr)
      !$acc routine seq
      !! `redi_face_flux` (:1003-1058) with the two `redi_tracer_column` rebuilds
      !! and ALL twelve NZ_STACK_MAX locals deleted. The sublayer loop is
      !! unchanged.
      integer, intent(in) :: nz, ns, nxc, nyc, nfa, nfb
      integer, intent(in) :: iL, jL, iR, jR, fa, fb
      real(wp), intent(in) :: PoL(nfa, nfb, ns), PoR(nfa, nfb, ns)
      integer, intent(in) :: KoL(nfa, nfb, ns), KoR(nfa, nfb, ns)
      real(wp), intent(in) :: hEff(nfa, nfb, ns - 1)
      real(wp), intent(in) :: coef
      logical, intent(in) :: is_left
      real(wp), intent(in) :: Tlay(nxc, nyc, nz), Tint(nxc, nyc, nz + 1)
      real(wp), intent(in) :: aLe(nxc, nyc, nz), aRe(nxc, nyc, nz)
      real(wp), intent(inout) :: dTr(nz)

      integer :: ks, knat
      real(wp) :: dtdiff, flx

      do ks = 1, ns - 1
         if (hEff(fa, fb, ks) /= 0.0_wp) then
            dtdiff = redi_sublayer_dT_g(nxc, nyc, nz, iL, jL, iR, jR, &
                                        KoL(fa, fb, ks), KoL(fa, fb, ks + 1), &
                                        KoR(fa, fb, ks), KoR(fa, fb, ks + 1), &
                                        PoL(fa, fb, ks), PoL(fa, fb, ks + 1), &
                                        PoR(fa, fb, ks), PoR(fa, fb, ks + 1), &
                                        Tlay, Tint, aLe, aRe)
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
   end subroutine redi_face_flux_g

   subroutine redi_apply_flux_impl_pre(nx, ny, nz, ns, dt, &
                                       nghost, nxp, nyp, wall_w, wall_e, wall_s, wall_n, &
                                       khtr_u, khtr_v, dy_cu, dx_cv, &
                                       idxCu, idyCv, areaT, hTr, &
                                       Tlay, Tint, aLe, aRe, &
                                       uPoL, uPoR, uKoL, uKoR, uhEff, &
                                       vPoL, vPoR, vKoL, vKoR, vhEff)
      !! PASS 2. `redi_apply_flux_impl` (:1060-1147) with `h_layer`/`hTr_in`
      !! swapped for the precomputed columns. Identical face selection, identical
      !! wall logic, identical signs, identical divergence.
      integer, intent(in) :: nx, ny, nz, ns
      real(wp), intent(in) :: dt
      integer, intent(in) :: nghost, nxp, nyp
      logical, intent(in) :: wall_w, wall_e, wall_s, wall_n
      real(wp), intent(in) :: khtr_u(nx + 1, ny), khtr_v(nx, ny + 1)
      real(wp), intent(in) :: dy_cu(nx + 1, ny), dx_cv(nx, ny + 1)
      real(wp), intent(in) :: idxCu(nx + 1, ny), idyCv(nx, ny + 1)
      real(wp), intent(in) :: areaT(nx, ny)
      real(wp), intent(inout) :: hTr(nx, ny, nz)
      real(wp), intent(in) :: Tlay(nx, ny, nz), Tint(nx, ny, nz + 1)
      real(wp), intent(in) :: aLe(nx, ny, nz), aRe(nx, ny, nz)
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

      do concurrent(j=1:ny, i=1:nx) local(k, dTr, iaij)
         do k = 1, nz
            dTr(k) = 0.0_wp
         end do
         if (i >= 2 .and. .not. ((wall_w .and. i == wuf_w) .or. (wall_e .and. i == wuf_e))) then
            call redi_face_flux_g(nz, ns, nx, ny, nx + 1, ny, &
                                  i - 1, j, i, j, i, j, uPoL, uPoR, uKoL, uKoR, uhEff, &
                                  dt*khtr_u(i, j)*dy_cu(i, j)*idxCu(i, j), .false., &
                                  Tlay, Tint, aLe, aRe, dTr)
         end if
         if (i <= nx - 1 .and. .not. ((wall_w .and. i + 1 == wuf_w) .or. (wall_e .and. i + 1 == wuf_e))) then
            call redi_face_flux_g(nz, ns, nx, ny, nx + 1, ny, &
                                  i, j, i + 1, j, i + 1, j, uPoL, uPoR, uKoL, uKoR, uhEff, &
                                  dt*khtr_u(i + 1, j)*dy_cu(i + 1, j)*idxCu(i + 1, j), .true., &
                                  Tlay, Tint, aLe, aRe, dTr)
         end if
         if (j >= 2 .and. .not. ((wall_s .and. j == wvf_s) .or. (wall_n .and. j == wvf_n))) then
            call redi_face_flux_g(nz, ns, nx, ny, nx, ny + 1, &
                                  i, j - 1, i, j, i, j, vPoL, vPoR, vKoL, vKoR, vhEff, &
                                  dt*khtr_v(i, j)*dx_cv(i, j)*idyCv(i, j), .false., &
                                  Tlay, Tint, aLe, aRe, dTr)
         end if
         if (j <= ny - 1 .and. .not. ((wall_s .and. j + 1 == wvf_s) .or. (wall_n .and. j + 1 == wvf_n))) then
            call redi_face_flux_g(nz, ns, nx, ny, nx, ny + 1, &
                                  i, j, i, j + 1, i, j + 1, vPoL, vPoR, vKoL, vKoR, vhEff, &
                                  dt*khtr_v(i, j + 1)*dx_cv(i, j + 1)*idyCv(i, j + 1), .true., &
                                  Tlay, Tint, aLe, aRe, dTr)
         end if
         iaij = 1.0_wp/areaT(i, j)
         do k = 1, nz
            hTr(i, j, k) = hTr(i, j, k) + dTr(k)*iaij
         end do
      end do
   end subroutine redi_apply_flux_impl_pre

end module ocean_redi_pre
