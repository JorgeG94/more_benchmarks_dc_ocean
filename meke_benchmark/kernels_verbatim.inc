   pure subroutine meke_zero_2d(n1, n2, a)
      !! Zero a device-resident 2D workspace (absent-source fallback).
      integer, intent(in) :: n1, n2
      real(wp), intent(out) :: a(n1, n2)
      integer :: i, j
      do concurrent(j=1:n2, i=1:n1)
         a(i, j) = 0.0_wp
      end do
   end subroutine meke_zero_2d

   pure subroutine meke_stage_rd(nx, ny, rd_in, rd_ws)
      !! Copy rd_over_dx onto the device workspace.  Thermo-cadence;
      !! explicit-shape; on-device (source slot is device-resident).
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: rd_in(nx, ny)
      real(wp), intent(out) :: rd_ws(nx, ny)
      integer :: i, j
      do concurrent(j=1:ny, i=1:nx)
         rd_ws(i, j) = rd_in(i, j)
      end do
   end subroutine meke_stage_rd

   pure subroutine meke_stage_sn(nx, ny, sn_u_in, sn_v_in, sn_u_ws, sn_v_ws)
      !! Copy the SN faces onto the device workspaces.  Thermo-cadence;
      !! explicit-shape; on-device (VarMix slot is device-resident).
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: sn_u_in(nx + 1, ny)
      real(wp), intent(in) :: sn_v_in(nx, ny + 1)
      real(wp), intent(out) :: sn_u_ws(nx + 1, ny)
      real(wp), intent(out) :: sn_v_ws(nx, ny + 1)
      integer :: i, j
      do concurrent(j=1:ny, i=1:nx + 1)
         sn_u_ws(i, j) = sn_u_in(i, j)
      end do
      do concurrent(j=1:ny + 1, i=1:nx)
         sn_v_ws(i, j) = sn_v_in(i, j)
      end do
   end subroutine meke_stage_sn

   pure subroutine meke_mass(nx, ny, nz, rho0, have_rho, h_layer, rho_layer, &
                             i_mass, depth_tot, mass_ws)
      !! Column mass `mass = Sum_k rho*max(h,H_VANISHED)` (kg/m^2), its
      !! inverse `i_mass` (0 where mass<=0), `depth_tot = Sum_k h` (m), and
      !! `mass_ws = mass` (the harmonic-mass input for the lateral flux).
      integer, intent(in) :: nx, ny, nz
      real(wp), intent(in) :: rho0
      logical, intent(in) :: have_rho
      real(wp), intent(in) :: h_layer(nx, ny, nz)
      real(wp), intent(in) :: rho_layer(nx, ny, nz)
      real(wp), intent(out) :: i_mass(nx, ny)
      real(wp), intent(out) :: depth_tot(nx, ny)
      real(wp), intent(out) :: mass_ws(nx, ny)
      integer :: i, j, k
      real(wp) :: mass, dsum, hk, rhok

      do concurrent(j=1:ny, i=1:nx) local(k, mass, dsum, hk, rhok)
         mass = 0.0_wp
         dsum = 0.0_wp
         do k = 1, nz
            hk = max(h_layer(i, j, k), H_VANISHED)
            rhok = rho0
            if (have_rho) rhok = rho_layer(i, j, k)
            mass = mass + rhok*hk
            dsum = dsum + h_layer(i, j, k)
         end do
         depth_tot(i, j) = dsum
         mass_ws(i, j) = mass
         if (mass > 0.0_wp) then
            i_mass(i, j) = 1.0_wp/mass
         else
            i_mass(i, j) = 0.0_wp
         end if
      end do
   end subroutine meke_mass

   pure function meke_inv_lmix(ueddy, sn, beta, area, rd_over_dx, depth, cdrag, &
                               a_deform, a_rhines, a_eady, a_frict, a_grid) result(inv_l)
      !$acc routine seq
      !! Harmonic inverse mixing length `1/Lmix = Sum aX/LX` over the five
      !! length scales (deformation, frictional, Rhines, Eady, grid).  Each
      !! scale is gated `aX*LX > 0` so a zero weight or a degenerate scale
      !! contributes nothing.  Returns 1/Lmix (0 ⇒ Lmix degenerate).
      real(wp), intent(in) :: ueddy, sn, beta, area, rd_over_dx, depth, cdrag
      real(wp), intent(in) :: a_deform, a_rhines, a_eady, a_frict, a_grid
      real(wp) :: inv_l
      real(wp) :: lgrid, ldeform, lfrict, lrhines, leady

      lgrid = sqrt(max(area, 0.0_wp))
      ldeform = lgrid*rd_over_dx
      lfrict = 0.0_wp
      if (cdrag > 0.0_wp) lfrict = depth/cdrag
      lrhines = 0.0_wp
      if (beta > 0.0_wp) lrhines = sqrt(max(ueddy, 0.0_wp)/beta)
      leady = 0.0_wp
      if (sn > 1.0e-15_wp) leady = ueddy/sn

      inv_l = 0.0_wp
      if (a_deform*ldeform > 0.0_wp) inv_l = inv_l + 1.0_wp/(a_deform*ldeform)
      if (a_frict*lfrict > 0.0_wp) inv_l = inv_l + 1.0_wp/(a_frict*lfrict)
      if (a_rhines*lrhines > 0.0_wp) inv_l = inv_l + 1.0_wp/(a_rhines*lrhines)
      if (a_eady*leady > 0.0_wp) inv_l = inv_l + 1.0_wp/(a_eady*leady)
      if (a_grid*lgrid > 0.0_wp) inv_l = inv_l + 1.0_wp/(a_grid*lgrid)
   end function meke_inv_lmix

   pure subroutine meke_length_scales(nx, ny, cd_scale, cb, ct, min_gamma2, &
                                      cdrag, a_deform, a_rhines, &
                                      a_eady, a_frict, a_grid, &
                                      areaT, idxT, idyT, f_centre, &
                                      depth_tot, meke, &
                                      rd_over_dx, sn_u, sn_v, &
                                      bottom_fac2, barotr_fac2, le)
      !! Fill the structure factors gamma_b^2 (`bottom_fac2`) and gamma_t^2
      !! (`barotr_fac2`) plus the mixing length `le` (Lmix) at each cell
      !! centre.  `Ldeform/Lfrict` drives both gammas; `Lmix` is the harmonic
      !! sum of the alpha-weighted scales.  `beta = |grad f|` from centred
      !! `f_centre` differences scaled by `idxT`/`idyT` (zero when f_centre
      !! is unfilled ⇒ Rhines inert).  SN = 0.25*(sn_u(i)+sn_u(i-1)+
      !! sn_v(j)+sn_v(j-1)) only when aEady>0.
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: cd_scale, cb, ct, min_gamma2, cdrag
      real(wp), intent(in) :: a_deform, a_rhines, a_eady, a_frict, a_grid
      real(wp), intent(in) :: areaT(nx, ny)
      real(wp), intent(in) :: idxT(nx, ny)
      real(wp), intent(in) :: idyT(nx, ny)
      real(wp), intent(in) :: f_centre(nx, ny)
      real(wp), intent(in) :: depth_tot(nx, ny)
      real(wp), intent(in) :: meke(nx, ny)
      real(wp), intent(in) :: rd_over_dx(nx, ny)
      real(wp), intent(in) :: sn_u(nx + 1, ny)
      real(wp), intent(in) :: sn_v(nx, ny + 1)
      real(wp), intent(out) :: bottom_fac2(nx, ny)
      real(wp), intent(out) :: barotr_fac2(nx, ny)
      real(wp), intent(out) :: le(nx, ny)
      integer :: i, j
      real(wp) :: lgrid, ldeform, lfrict, ratio, bf2, tf2
      real(wp) :: ueddy, sn, beta, inv_l, rd

      do concurrent(j=1:ny, i=1:nx) &
         local(lgrid, ldeform, lfrict, ratio, bf2, tf2, ueddy, sn, beta, inv_l, rd)
         rd = rd_over_dx(i, j)
         lgrid = sqrt(max(areaT(i, j), 0.0_wp))
         ldeform = lgrid*rd
         lfrict = 0.0_wp
         if (cdrag > 0.0_wp) lfrict = depth_tot(i, j)/cdrag

         ! gamma_b^2 = cd_scale^2 + 1/(1+Cb*Ldeform/Lfrict)^0.8 (floor).
         bf2 = cd_scale*cd_scale
         if (lfrict*cb > 0.0_wp) then
            ratio = ldeform/lfrict
            bf2 = bf2 + 1.0_wp/(1.0_wp + cb*ratio)**0.8_wp
         end if
         bf2 = max(bf2, min_gamma2)
         bottom_fac2(i, j) = bf2

         ! gamma_t^2 = 1/(1+Ct*Ldeform/Lfrict)^0.25 (floor).
         tf2 = 1.0_wp
         if (lfrict*ct > 0.0_wp) then
            ratio = ldeform/lfrict
            tf2 = 1.0_wp/(1.0_wp + ct*ratio)**0.25_wp
         end if
         tf2 = max(tf2, min_gamma2)
         barotr_fac2(i, j) = tf2

         ! Mixing length: harmonic sum of alpha-weighted scales.
         ueddy = sqrt(2.0_wp*max(0.0_wp, tf2*meke(i, j)))
         ! beta = |grad f|; centred f_centre differences scaled to a true
         ! gradient by the cell-centre inverse metrics (df/dx ~ (f_{i+1} -
         ! f_{i-1})*idxT/2).  When f_centre is unfilled (default) beta=0 ⇒
         ! Rhines weight inert (alpha_rhines default 0).
         beta = 0.0_wp
         if (i > 1 .and. i < nx) then
            beta = beta + (0.5_wp*(f_centre(i + 1, j) - f_centre(i - 1, j))*idxT(i, j))**2
         end if
         if (j > 1 .and. j < ny) then
            beta = beta + (0.5_wp*(f_centre(i, j + 1) - f_centre(i, j - 1))*idyT(i, j))**2
         end if
         beta = sqrt(beta)
         sn = 0.0_wp
         if (a_eady > 0.0_wp) sn = 0.25_wp*((sn_u(i, j) + sn_u(i + 1, j)) + &
                                            (sn_v(i, j) + sn_v(i, j + 1)))

         inv_l = meke_inv_lmix(ueddy, sn, beta, areaT(i, j), rd, &
                               depth_tot(i, j), cdrag, &
                               a_deform, a_rhines, a_eady, a_frict, a_grid)
         if (inv_l > 0.0_wp) then
            le(i, j) = 1.0_wp/inv_l
         else
            le(i, j) = 0.0_wp
         end if
      end do
   end subroutine meke_length_scales

   pure subroutine meke_source(nx, ny, bgsrc, gmcoeff, frcoeff, sdt, i_mass, &
                               gm_src, ke_diss, src, meke)
      !! Aggregate source `src = bgsrc + gmcoeff*I_mass*gm_src
      !! - frcoeff*I_mass*ke_diss` and the explicit bump `E += sdt*src`.
      !! `gmcoeff<0` ⇒ GM source off; `frcoeff<0` ⇒ frictional source off.
      !! `ke_diss` is the lateral-viscosity KE dissipation rate (≤0), so
      !! `-frcoeff*I_mass*ke_diss ≥ 0` is a mean→eddy source (0 ⇒ inert).
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: bgsrc, gmcoeff, frcoeff, sdt
      real(wp), intent(in) :: i_mass(nx, ny)
      real(wp), intent(in) :: gm_src(nx, ny)
      real(wp), intent(in) :: ke_diss(nx, ny)
      real(wp), intent(inout) :: src(nx, ny)
      real(wp), intent(inout) :: meke(nx, ny)
      integer :: i, j
      real(wp) :: s

      do concurrent(j=1:ny, i=1:nx) local(s)
         s = bgsrc
         if (gmcoeff >= 0.0_wp) s = s + gmcoeff*i_mass(i, j)*gm_src(i, j)
         if (frcoeff >= 0.0_wp) s = s - frcoeff*i_mass(i, j)*ke_diss(i, j)
         src(i, j) = s
         meke(i, j) = meke(i, j) + sdt*s
      end do
   end subroutine meke_source

   pure subroutine meke_drag(nx, ny, sdt_damp, damping, cdrag, uscale, &
                             i_mass, bottom_fac2, u_bbl2, meke)
      !! Implicit (backward-Euler) bottom-drag half-step.
      !!   drag_rate = i_mass*sqrt(cdrag^2*(max(0,2*bf2*E)+u_bbl2+uscale^2))
      !!   damp_rate = damping + drag_rate*bf2 ; =0 where E<0
      !!   E <- E/(1 + sdt_damp*damp_rate)
      !! `u_bbl2` is the resolved bed-layer speed² (MOM6 `drag_rate_visc`);
      !! it is 0 unless `use_bbl_drag` is set, so the default is bit-identical.
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: sdt_damp, damping, cdrag, uscale
      real(wp), intent(in) :: i_mass(nx, ny)
      real(wp), intent(in) :: bottom_fac2(nx, ny)
      real(wp), intent(in) :: u_bbl2(nx, ny)
      real(wp), intent(inout) :: meke(nx, ny)
      integer :: i, j
      real(wp) :: drag_rate, damp_rate, cd2, e

      cd2 = cdrag*cdrag
      do concurrent(j=1:ny, i=1:nx) local(drag_rate, damp_rate, e)
         e = meke(i, j)
         drag_rate = i_mass(i, j)*sqrt(cd2*(max(0.0_wp, 2.0_wp*bottom_fac2(i, j)*e) &
                                            + u_bbl2(i, j) + uscale*uscale))
         damp_rate = damping + drag_rate*bottom_fac2(i, j)
         if (e < 0.0_wp) damp_rate = 0.0_wp
         meke(i, j) = e/(1.0_wp + sdt_damp*damp_rate)
      end do
   end subroutine meke_drag

   pure subroutine meke_bbl_speed2(nx, ny, nz, u_face, v_face, u_bbl2)
      !! Resolved bed-layer (k=1, bottom-up) speed² at cell centres:
      !! `u_bbl2 = u_c² + v_c²` with `u_c = ½(u_face(i)+u_face(i+1))`,
      !! `v_c = ½(v_face(j)+v_face(j+1))` from the BED layer.  The bottom
      !! eddy velocity the MEKE drag law needs (MOM6 `drag_rate_visc`).
      integer, intent(in) :: nx, ny, nz
      real(wp), intent(in) :: u_face(nx + 1, ny, nz)
      real(wp), intent(in) :: v_face(nx, ny + 1, nz)
      real(wp), intent(inout) :: u_bbl2(nx, ny)
      integer :: i, j
      real(wp) :: u_c, v_c
      do concurrent(j=1:ny, i=1:nx) local(u_c, v_c)
         u_c = 0.5_wp*(u_face(i, j, 1) + u_face(i + 1, j, 1))
         v_c = 0.5_wp*(v_face(i, j, 1) + v_face(i, j + 1, 1))
         u_bbl2(i, j) = u_c*u_c + v_c*v_c
      end do
   end subroutine meke_bbl_speed2

   pure subroutine meke_kh_closure(nx, ny, khcoeff, barotr_fac2, meke, le, kh_diff)
      !! Derived diffusivity `kh = khcoeff*sqrt(2*max(0,gamma_t2*E))*Lmix`.
      !! `khcoeff<=0` ⇒ kh left at 0.
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: khcoeff
      real(wp), intent(in) :: barotr_fac2(nx, ny)
      real(wp), intent(in) :: meke(nx, ny)
      real(wp), intent(in) :: le(nx, ny)
      real(wp), intent(out) :: kh_diff(nx, ny)
      integer :: i, j
      real(wp) :: ueddy

      do concurrent(j=1:ny, i=1:nx) local(ueddy)
         if (khcoeff > 0.0_wp) then
            ueddy = sqrt(2.0_wp*max(0.0_wp, barotr_fac2(i, j)*meke(i, j)))
            kh_diff(i, j) = khcoeff*ueddy*le(i, j)
         else
            kh_diff(i, j) = 0.0_wp
         end if
      end do
   end subroutine meke_kh_closure

   pure subroutine meke_ku_closure(nx, ny, backscatter, visc_coeff_ku, &
                                   meke, le, ku)
      !! Derived harmonic backscatter viscosity
      !! `ku = visc_coeff_ku*sqrt(2*max(0,E))*Lmix` (m²/s), matching MOM6
      !! `MEKE%Ku = MEKE_VISCOSITY_COEFF_KU*sqrt(2*MEKE)*Lmix`.  Unlike the
      !! kh closure (which carries the barotropic-mode factor `gamma_t2`
      !! inside the eddy velocity), MOM6's Ku uses the PLAIN `sqrt(2*MEKE)` —
      !! no vertical-structure factor — so `gamma_t2` is deliberately absent
      !! here (vertical structure `BS_struct = 1`; EBT/SQG deferred).
      !! Off ⇒ ku left at 0 (bit-identical seam).  Always ≥ 0; the SIGN of
      !! the momentum effect is set by the subtraction downstream.
      integer, intent(in) :: nx, ny
      logical, intent(in) :: backscatter
      real(wp), intent(in) :: visc_coeff_ku
      real(wp), intent(in) :: meke(nx, ny)
      real(wp), intent(in) :: le(nx, ny)
      real(wp), intent(out) :: ku(nx, ny)
      integer :: i, j
      real(wp) :: ueddy

      do concurrent(j=1:ny, i=1:nx) local(ueddy)
         if (backscatter .and. visc_coeff_ku /= 0.0_wp) then
            ueddy = sqrt(2.0_wp*max(0.0_wp, meke(i, j)))
            ku(i, j) = visc_coeff_ku*ueddy*le(i, j)
         else
            ku(i, j) = 0.0_wp
         end if
      end do
   end subroutine meke_ku_closure

   pure subroutine meke_lateral(nx, ny, sdt, kh_bg, k4, khmeke_fac, &
                                kh_flux_enabled, dy_cu, dx_cv, idxCu, idyCv, &
                                iareaT, i_mass, mass, kh_diff, &
                                uflux, vflux, del2, meke)
      !! Harmonic-mass Laplacian diffusion of MEKE (+ optional biharmonic).
      !! Flux-form, conservative on a closed domain (interior faces only;
      !! array-edge faces carry zero flux).
      !!   Kh_u = max(0,kh_bg) + khmeke_fac*0.5*(kh_i+kh_{i+1}), CFL-capped 0.25
      !!   uflux = Kh_u*(dy_cu*idxCu)*[2 m_i m_{i+1}/(m_i+m_{i+1}+eps)]*(E_i-E_{i+1})
      !!   E += sdt*iareaT*i_mass*((uflux_{i-1}-uflux_i)+(vflux_{j-1}-vflux_j))
      !! Biharmonic: del2 = iareaT*(d uflux' + d vflux') with the bare-gradient
      !! flux uflux' = (dy_cu*idxCu)*(E_{i+1}-E_i); then a harmonic-mass flux
      !! of del2 with CFL cap 0.3 and E += that divergence (additive).
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: sdt, kh_bg, k4, khmeke_fac
      logical, intent(in) :: kh_flux_enabled
      real(wp), intent(in) :: dy_cu(nx + 1, ny)
      real(wp), intent(in) :: dx_cv(nx, ny + 1)
      real(wp), intent(in) :: idxCu(nx + 1, ny)
      real(wp), intent(in) :: idyCv(nx, ny + 1)
      real(wp), intent(in) :: iareaT(nx, ny)
      real(wp), intent(in) :: i_mass(nx, ny)
      real(wp), intent(in) :: mass(nx, ny)
      real(wp), intent(in) :: kh_diff(nx, ny)
      real(wp), intent(inout) :: uflux(nx + 1, ny)
      real(wp), intent(inout) :: vflux(nx, ny + 1)
      real(wp), intent(inout) :: del2(nx, ny)
      real(wp), intent(inout) :: meke(nx, ny)
      integer :: i, j
      real(wp) :: kh_u, kh_v, hm, geo, inv_max, k4_u, k4_v, mke

      ! ---------- Biharmonic (computed first; tendency added after diffusion). ----------
      if (k4 >= 0.0_wp) then
         ! bare-gradient flux into uflux/vflux workspaces (units m^2/s^2).
         do concurrent(j=1:ny, i=1:nx + 1)
            uflux(i, j) = 0.0_wp
         end do
         do concurrent(j=1:ny, i=2:nx)
            uflux(i, j) = (dy_cu(i, j)*idxCu(i, j))*(meke(i, j) - meke(i - 1, j))
         end do
         do concurrent(j=1:ny + 1, i=1:nx)
            vflux(i, j) = 0.0_wp
         end do
         do concurrent(j=2:ny, i=1:nx)
            vflux(i, j) = (dx_cv(i, j)*idyCv(i, j))*(meke(i, j) - meke(i, j - 1))
         end do
         do concurrent(j=1:ny, i=1:nx)
            del2(i, j) = iareaT(i, j)*((uflux(i + 1, j) - uflux(i, j)) + &
                                       (vflux(i, j + 1) - vflux(i, j)))
         end do
         ! harmonic-mass flux of del2 with K4 (CFL cap 0.3).
         do concurrent(j=1:ny, i=1:nx + 1)
            uflux(i, j) = 0.0_wp
         end do
         do concurrent(j=1:ny, i=2:nx) local(k4_u, geo, hm, inv_max)
            geo = dy_cu(i, j)*idxCu(i, j)
            inv_max = 64.0_wp*sdt*(geo*max(iareaT(i - 1, j), iareaT(i, j)))**2
            k4_u = k4
            if (k4_u*inv_max > 0.3_wp) k4_u = 0.3_wp/inv_max
            hm = 2.0_wp*mass(i - 1, j)*mass(i, j)/((mass(i - 1, j) + mass(i, j)) + MASS_NEGLECT)
            uflux(i, j) = (k4_u*geo)*hm*(del2(i, j) - del2(i - 1, j))
         end do
         do concurrent(j=1:ny + 1, i=1:nx)
            vflux(i, j) = 0.0_wp
         end do
         do concurrent(j=2:ny, i=1:nx) local(k4_v, geo, hm, inv_max)
            geo = dx_cv(i, j)*idyCv(i, j)
            inv_max = 64.0_wp*sdt*(geo*max(iareaT(i, j - 1), iareaT(i, j)))**2
            k4_v = k4
            if (k4_v*inv_max > 0.3_wp) k4_v = 0.3_wp/inv_max
            hm = 2.0_wp*mass(i, j - 1)*mass(i, j)/((mass(i, j - 1) + mass(i, j)) + MASS_NEGLECT)
            vflux(i, j) = (k4_v*geo)*hm*(del2(i, j) - del2(i, j - 1))
         end do
         do concurrent(j=1:ny, i=1:nx) local(mke)
            mke = sdt*(iareaT(i, j)*i_mass(i, j))* &
                  ((uflux(i, j) - uflux(i + 1, j)) + (vflux(i, j) - vflux(i, j + 1)))
            del2(i, j) = mke   ! stash the biharmonic tendency in del2
         end do
      end if

      ! ---------- Laplacian (harmonic-mass) diffusion. ----------
      if (kh_flux_enabled) then
         do concurrent(j=1:ny, i=1:nx + 1)
            uflux(i, j) = 0.0_wp
         end do
         do concurrent(j=1:ny, i=2:nx) local(kh_u, geo, hm, inv_max)
            geo = dy_cu(i, j)*idxCu(i, j)
            kh_u = max(0.0_wp, kh_bg) + khmeke_fac*0.5_wp*(kh_diff(i - 1, j) + kh_diff(i, j))
            inv_max = 2.0_wp*sdt*(geo*max(iareaT(i - 1, j), iareaT(i, j)))
            if (kh_u*inv_max > 0.25_wp) kh_u = 0.25_wp/inv_max
            hm = 2.0_wp*mass(i - 1, j)*mass(i, j)/((mass(i - 1, j) + mass(i, j)) + MASS_NEGLECT)
            uflux(i, j) = (kh_u*geo)*hm*(meke(i - 1, j) - meke(i, j))
         end do
         do concurrent(j=1:ny + 1, i=1:nx)
            vflux(i, j) = 0.0_wp
         end do
         do concurrent(j=2:ny, i=1:nx) local(kh_v, geo, hm, inv_max)
            geo = dx_cv(i, j)*idyCv(i, j)
            kh_v = max(0.0_wp, kh_bg) + khmeke_fac*0.5_wp*(kh_diff(i, j - 1) + kh_diff(i, j))
            inv_max = 2.0_wp*sdt*(geo*max(iareaT(i, j - 1), iareaT(i, j)))
            if (kh_v*inv_max > 0.25_wp) kh_v = 0.25_wp/inv_max
            hm = 2.0_wp*mass(i, j - 1)*mass(i, j)/((mass(i, j - 1) + mass(i, j)) + MASS_NEGLECT)
            vflux(i, j) = (kh_v*geo)*hm*(meke(i, j - 1) - meke(i, j))
         end do
         do concurrent(j=1:ny, i=1:nx) local(mke)
            mke = sdt*(iareaT(i, j)*i_mass(i, j))* &
                  ((uflux(i, j) - uflux(i + 1, j)) + (vflux(i, j) - vflux(i, j + 1)))
            meke(i, j) = meke(i, j) + mke
         end do
      end if

      ! add the biharmonic tendency (computed above, stashed in del2).
      if (k4 >= 0.0_wp) then
         do concurrent(j=1:ny, i=1:nx)
            meke(i, j) = meke(i, j) + del2(i, j)
         end do
      end if
   end subroutine meke_lateral
