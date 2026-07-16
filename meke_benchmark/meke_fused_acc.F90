!! FUSED MEKE step (6 loops) + `!$acc kernels async(1)` + one `wait(1)`.
!!
!! Auto-generated from meke_fused.F90 -- identical body, every loop wrapped.
!! Tests whether the async wrapper and fusion STACK, and whether the wrapper
!! backfires on a register-heavy fused loop (RESUME_GPU_MRE.md sec.5 "possible
!! bug 3": `!$acc kernels` around a register-heavy DC loop took it 56->90
!! registers with 16 spills, costing 2.3x on that kernel; `private()` did not
!! fix it and the mechanism is unknown). L3/L6 here ARE register-heavy, so this
!! variant is exactly the case that report warns about. Measured, not assumed.
!! MEKE step, FUSED: 16 `do concurrent` loops -> 6.
!!
!! Same arithmetic, same order, plain explicit-shape dummies (production
!! already has those — see README). The ONLY change vs meke.F90 is which
!! loop each statement lives in. Every merge is single-assignment
!! (`uflux = (interior ? computed : 0)`), so this must be BIT-IDENTICAL, not
!! merely close. The driver checks that (`max|d meke| == 0.0` exactly).
!!
!! ================= THE FUSION, AND WHY IT IS LEGAL =================
!! Production's live path (gabight_sph_meke_v100.nml), 16 loops:
!!    1 zero rd_ws        (nx,ny)      9  zero uflux     (nx+1,ny)
!!    2 zero sn_u_ws      (nx+1,ny)    10 fill uflux     i=2:nx
!!    3 zero sn_v_ws      (nx,ny+1)    11 zero vflux     (nx,ny+1)
!!    4 mass              (nx,ny)      12 fill vflux     j=2:ny
!!    5 length_scales     (nx,ny)      13 divergence     (nx,ny)
!!    6 stage ke_diss     (nx,ny)      14 drag (2nd half)(nx,ny)
!!    7 source            (nx,ny)      15 kh_closure     (nx,ny)
!!    8 drag (1st half)   (nx,ny)      16 ku_closure     (nx,ny)
!!
!! FUSED into 6:
!!   L1 = loop 2          — sn_u zero. Kept separate: length_scales reads
!!                          sn_u(i,j) AND sn_u(i+1,j) — a NEIGHBOUR read, so
!!                          this is a real barrier. (It is dead in this config
!!                          because alpha_eady=0, but the fusion must be valid
!!                          for ANY config, so it stays.)
!!   L2 = loop 3          — sn_v zero. Same reason (reads sn_v(i,j+1)).
!!   L3 = loops 1,4,5,6,7,8  — ALL POINTWISE in (i,j). Checked, not assumed:
!!          * 5 reads depth_tot(i,j) that 4 wrote     -> same thread
!!          * 7 reads i_mass(i,j)    that 4 wrote     -> same thread
!!          * 8 reads bottom_fac2(i,j) that 5 wrote   -> same thread
!!          * 8 reads meke(i,j)      that 7 wrote     -> same thread
!!          * 5 reads meke(i,j) BEFORE 7 writes it    -> order preserved
!!            within the thread, so the value read is the old one, as before.
!!          * 5 reads f_centre(i+-1,j+-1) — a neighbour read, but f_centre is
!!            NEVER WRITTEN by any loop here, so it is not a hazard.
!!   L4 = loops 9,10      — wall+producer merge on uflux.
!!   L5 = loops 11,12     — wall+producer merge on vflux.
!!   L6 = loops 13,14,15,16 — all pointwise in (i,j):
!!          * 14 reads meke(i,j) that 13 wrote        -> same thread
!!          * 15,16 read meke(i,j)/le(i,j)/barotr_fac2(i,j) -> same thread
!!
!! ================= WHAT CANNOT BE FUSED (checked) =================
!!   * L3 -> L4/L5 is a REAL BARRIER: uflux(i,j) reads meke(i-1,j) and
!!     meke(i,j), and mass_ws(i-1,j) — NEIGHBOURS of what L3 wrote. Every
!!     thread's producer must have landed first.
!!   * L4/L5 -> L6 is a REAL BARRIER: the divergence reads uflux(i+1,j) and
!!     vflux(i,j+1) — neighbours of what L4/L5 wrote.
!!   * L4 and L5 are not fused with each other: different iteration spaces
!!     ((nx+1,ny) vs (nx,ny+1)). A padded union loop with guards would be
!!     legal and would save one launch; not done (untested, and 1 of 6).
!!   * L1/L2 cannot fold into L3 (neighbour reads of sn_*, above).
!! 6 loops is that dependency structure, not an implementation limit.
module meke_fused_acc
   use constants, only: wp, H_VANISHED
   implicit none
   private
   public :: meke_step_fused_acc

   real(wp), parameter :: MASS_NEGLECT = 1.0e-30_wp

contains

   subroutine meke_step_fused_acc(nx, ny, nz, dt, dtscale, &
                              cd_scale, cb, ct, min_gamma2, cdrag, uscale, &
                              a_deform, a_rhines, a_eady, a_frict, a_grid, &
                              bgsrc, gmcoeff, frcoeff, damping, &
                              kh_bg, k4, khmeke_fac, khcoeff, &
                              backscatter, visc_coeff_ku, rho0, &
                              areaT, idxT, idyT, dy_cu, dx_cv, idxCu, idyCv, iareaT, &
                              h_layer, rho_layer, gm_src, ke_diss_ext, f_centre, u_bbl2, &
                              rd_ws, sn_u_ws, sn_v_ws, ke_diss_ws, &
                              i_mass, depth_tot, mass_ws, bottom_fac2, barotr_fac2, &
                              src, le, uflux, vflux, kh_diff, ku, meke)
      integer, intent(in) :: nx, ny, nz
      real(wp), intent(in) :: dt, dtscale
      real(wp), intent(in) :: cd_scale, cb, ct, min_gamma2, cdrag, uscale
      real(wp), intent(in) :: a_deform, a_rhines, a_eady, a_frict, a_grid
      real(wp), intent(in) :: bgsrc, gmcoeff, frcoeff, damping
      real(wp), intent(in) :: kh_bg, k4, khmeke_fac, khcoeff
      logical, intent(in) :: backscatter
      real(wp), intent(in) :: visc_coeff_ku, rho0
      real(wp), intent(in) :: areaT(nx, ny), idxT(nx, ny), idyT(nx, ny)
      real(wp), intent(in) :: dy_cu(nx + 1, ny), dx_cv(nx, ny + 1)
      real(wp), intent(in) :: idxCu(nx + 1, ny), idyCv(nx, ny + 1)
      real(wp), intent(in) :: iareaT(nx, ny)
      real(wp), intent(in) :: h_layer(nx, ny, nz), rho_layer(nx, ny, nz)
      real(wp), intent(in) :: gm_src(nx, ny), ke_diss_ext(nx, ny), f_centre(nx, ny)
      real(wp), intent(inout) :: rd_ws(nx, ny), sn_u_ws(nx + 1, ny), sn_v_ws(nx, ny + 1)
      real(wp), intent(inout) :: ke_diss_ws(nx, ny)
      real(wp), intent(in) :: u_bbl2(nx, ny)
      real(wp), intent(inout) :: i_mass(nx, ny), depth_tot(nx, ny), mass_ws(nx, ny)
      real(wp), intent(inout) :: bottom_fac2(nx, ny), barotr_fac2(nx, ny)
      real(wp), intent(inout) :: src(nx, ny), le(nx, ny)
      real(wp), intent(inout) :: uflux(nx + 1, ny), vflux(nx, ny + 1)
      real(wp), intent(inout) :: kh_diff(nx, ny), ku(nx, ny), meke(nx, ny)

      integer :: i, j, k
      real(wp) :: sdt, sdt_damp, damp_step, cd2, adummy
      real(wp) :: mass, dsum, hk, rhok, imk
      real(wp) :: lgrid, ldeform, lfrict, ratio, bf2, tf2, ueddy, sn, beta, inv_l, rd
      real(wp) :: lrhines, leady, invl
      real(wp) :: s, e, drag_rate, damp_rate
      real(wp) :: kh_u, kh_v, geo, hm, inv_max, mke
      logical :: kh_flux_enabled

      sdt = dt*dtscale
      damp_step = 1.0_wp
      if (kh_bg >= 0.0_wp .or. k4 >= 0.0_wp) damp_step = 0.5_wp
      sdt_damp = sdt*damp_step
      kh_flux_enabled = (kh_bg >= 0.0_wp)
      cd2 = cdrag*cdrag

      ! ---------------- L1: sn_u zero (barrier: neighbour read downstream) ----
      !$acc kernels async(1)
      do concurrent(j=1:ny, i=1:nx + 1)
         sn_u_ws(i, j) = 0.0_wp
      end do
      !$acc end kernels
      ! ---------------- L2: sn_v zero ----------------------------------------
      !$acc kernels async(1)
      do concurrent(j=1:ny + 1, i=1:nx)
         sn_v_ws(i, j) = 0.0_wp
      end do
      !$acc end kernels

      ! ---------------- L3 = loops 1,4,5,6,7,8 (all pointwise) ---------------
      !$acc kernels async(1)
      do concurrent(j=1:ny, i=1:nx) &
         local(k, mass, dsum, hk, rhok, imk, lgrid, ldeform, lfrict, ratio, bf2, tf2, &
               ueddy, sn, beta, inv_l, rd, lrhines, leady, invl, s, e, drag_rate, damp_rate)
         ! -- loop 1: rd_ws zero (have_ws = .false. in this config) --
         rd_ws(i, j) = 0.0_wp
         rd = rd_ws(i, j)   ! same load production's length_scales does
         ! -- loop 6: stage ke_diss --
         ke_diss_ws(i, j) = ke_diss_ext(i, j)
         ! -- loop 4: meke_mass --
         mass = 0.0_wp
         dsum = 0.0_wp
         do k = 1, nz
            hk = max(h_layer(i, j, k), H_VANISHED)
            rhok = rho_layer(i, j, k)
            mass = mass + rhok*hk
            dsum = dsum + h_layer(i, j, k)
         end do
         depth_tot(i, j) = dsum
         mass_ws(i, j) = mass
         if (mass > 0.0_wp) then
            imk = 1.0_wp/mass
         else
            imk = 0.0_wp
         end if
         i_mass(i, j) = imk
         ! -- loop 5: meke_length_scales (depth_tot/rd from THIS thread) --
         lgrid = sqrt(max(areaT(i, j), 0.0_wp))
         ldeform = lgrid*rd
         lfrict = 0.0_wp
         if (cdrag > 0.0_wp) lfrict = dsum/cdrag
         bf2 = cd_scale*cd_scale
         if (lfrict*cb > 0.0_wp) then
            ratio = ldeform/lfrict
            bf2 = bf2 + 1.0_wp/(1.0_wp + cb*ratio)**0.8_wp
         end if
         bf2 = max(bf2, min_gamma2)
         bottom_fac2(i, j) = bf2
         tf2 = 1.0_wp
         if (lfrict*ct > 0.0_wp) then
            ratio = ldeform/lfrict
            tf2 = 1.0_wp/(1.0_wp + ct*ratio)**0.25_wp
         end if
         tf2 = max(tf2, min_gamma2)
         barotr_fac2(i, j) = tf2
         ! meke(i,j) read here is the OLD value — loop 7 writes it below, in
         ! this same thread, so ordering is preserved exactly.
         ueddy = sqrt(2.0_wp*max(0.0_wp, tf2*meke(i, j)))
         beta = 0.0_wp
         if (i > 1 .and. i < nx) then
            beta = beta + (0.5_wp*(f_centre(i + 1, j) - f_centre(i - 1, j))*idxT(i, j))**2
         end if
         if (j > 1 .and. j < ny) then
            beta = beta + (0.5_wp*(f_centre(i, j + 1) - f_centre(i, j - 1))*idyT(i, j))**2
         end if
         beta = sqrt(beta)
         sn = 0.0_wp
         if (a_eady > 0.0_wp) sn = 0.25_wp*((sn_u_ws(i, j) + sn_u_ws(i + 1, j)) + &
                                            (sn_v_ws(i, j) + sn_v_ws(i, j + 1)))
         ! -- meke_inv_lmix, inlined verbatim (same expression, same order) --
         ldeform = lgrid*rd
         lfrict = 0.0_wp
         if (cdrag > 0.0_wp) lfrict = dsum/cdrag
         lrhines = 0.0_wp
         if (beta > 0.0_wp) lrhines = sqrt(max(ueddy, 0.0_wp)/beta)
         leady = 0.0_wp
         if (sn > 1.0e-15_wp) leady = ueddy/sn
         invl = 0.0_wp
         if (a_deform*ldeform > 0.0_wp) invl = invl + 1.0_wp/(a_deform*ldeform)
         if (a_frict*lfrict > 0.0_wp) invl = invl + 1.0_wp/(a_frict*lfrict)
         if (a_rhines*lrhines > 0.0_wp) invl = invl + 1.0_wp/(a_rhines*lrhines)
         if (a_eady*leady > 0.0_wp) invl = invl + 1.0_wp/(a_eady*leady)
         if (a_grid*lgrid > 0.0_wp) invl = invl + 1.0_wp/(a_grid*lgrid)
         inv_l = invl
         if (inv_l > 0.0_wp) then
            le(i, j) = 1.0_wp/inv_l
         else
            le(i, j) = 0.0_wp
         end if
         ! -- loop 7: meke_source --
         s = bgsrc
         if (gmcoeff >= 0.0_wp) s = s + gmcoeff*imk*gm_src(i, j)
         if (frcoeff >= 0.0_wp) s = s - frcoeff*imk*ke_diss_ws(i, j)
         src(i, j) = s
         e = meke(i, j) + sdt*s
         ! -- loop 8: meke_drag (u_bbl2 = 0; use_bbl_drag off in this config) --
         drag_rate = imk*sqrt(cd2*(max(0.0_wp, 2.0_wp*bf2*e) &
                                   + u_bbl2(i, j) + uscale*uscale))
         damp_rate = damping + drag_rate*bf2
         if (e < 0.0_wp) damp_rate = 0.0_wp
         meke(i, j) = e/(1.0_wp + sdt_damp*damp_rate)
      end do
      !$acc end kernels

      ! ================ BARRIER: L4/L5 read meke/mass_ws NEIGHBOURS ==========

      ! ---------------- L4 = loops 9,10: wall + producer merge on uflux ------
      !$acc kernels async(1)
      do concurrent(j=1:ny, i=1:nx + 1) local(kh_u, geo, hm, inv_max)
         if (i >= 2 .and. i <= nx) then
            geo = dy_cu(i, j)*idxCu(i, j)
            kh_u = max(0.0_wp, kh_bg) + khmeke_fac*0.5_wp*(kh_diff(i - 1, j) + kh_diff(i, j))
            inv_max = 2.0_wp*sdt*(geo*max(iareaT(i - 1, j), iareaT(i, j)))
            if (kh_u*inv_max > 0.25_wp) kh_u = 0.25_wp/inv_max
            hm = 2.0_wp*mass_ws(i - 1, j)*mass_ws(i, j)/ &
                 ((mass_ws(i - 1, j) + mass_ws(i, j)) + MASS_NEGLECT)
            uflux(i, j) = (kh_u*geo)*hm*(meke(i - 1, j) - meke(i, j))
         else
            uflux(i, j) = 0.0_wp
         end if
      end do
      !$acc end kernels
      ! ---------------- L5 = loops 11,12: wall + producer merge on vflux -----
      !$acc kernels async(1)
      do concurrent(j=1:ny + 1, i=1:nx) local(kh_v, geo, hm, inv_max)
         if (j >= 2 .and. j <= ny) then
            geo = dx_cv(i, j)*idyCv(i, j)
            kh_v = max(0.0_wp, kh_bg) + khmeke_fac*0.5_wp*(kh_diff(i, j - 1) + kh_diff(i, j))
            inv_max = 2.0_wp*sdt*(geo*max(iareaT(i, j - 1), iareaT(i, j)))
            if (kh_v*inv_max > 0.25_wp) kh_v = 0.25_wp/inv_max
            hm = 2.0_wp*mass_ws(i, j - 1)*mass_ws(i, j)/ &
                 ((mass_ws(i, j - 1) + mass_ws(i, j)) + MASS_NEGLECT)
            vflux(i, j) = (kh_v*geo)*hm*(meke(i, j - 1) - meke(i, j))
         else
            vflux(i, j) = 0.0_wp
         end if
      end do
      !$acc end kernels

      ! ================ BARRIER: L6 reads uflux(i+1)/vflux(j+1) ==============

      ! ---------------- L6 = loops 13,14,15,16 (all pointwise) ---------------
      !$acc kernels async(1)
      do concurrent(j=1:ny, i=1:nx) local(mke, e, drag_rate, damp_rate, ueddy)
         ! -- loop 13: conservative divergence --
         mke = sdt*(iareaT(i, j)*i_mass(i, j))* &
               ((uflux(i, j) - uflux(i + 1, j)) + (vflux(i, j) - vflux(i, j + 1)))
         e = meke(i, j) + mke
         ! -- loop 14: meke_drag, 2nd half --
         drag_rate = i_mass(i, j)*sqrt(cd2*(max(0.0_wp, 2.0_wp*bottom_fac2(i, j)*e) &
                                            + u_bbl2(i, j) + uscale*uscale))
         damp_rate = damping + drag_rate*bottom_fac2(i, j)
         if (e < 0.0_wp) damp_rate = 0.0_wp
         e = e/(1.0_wp + sdt_damp*damp_rate)
         meke(i, j) = e
         ! -- loop 15: meke_kh_closure --
         if (khcoeff > 0.0_wp) then
            ueddy = sqrt(2.0_wp*max(0.0_wp, barotr_fac2(i, j)*e))
            kh_diff(i, j) = khcoeff*ueddy*le(i, j)
         else
            kh_diff(i, j) = 0.0_wp
         end if
         ! -- loop 16: meke_ku_closure --
         if (backscatter .and. visc_coeff_ku /= 0.0_wp) then
            ueddy = sqrt(2.0_wp*max(0.0_wp, e))
            ku(i, j) = visc_coeff_ku*ueddy*le(i, j)
         else
            ku(i, j) = 0.0_wp
         end if
      end do
      !$acc end kernels

      !$acc wait(1)   ! single drain: the whole step ran on async(1)
      adummy = k4   ! k4 < 0 in this config: the biharmonic block is not reached.
   end subroutine meke_step_fused_acc

end module meke_fused_acc
