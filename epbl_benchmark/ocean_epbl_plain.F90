!! Signature-fix control for the EPBL column kernel: IDENTICAL body, but every
!! array read through a derived-type component (`this%t0%data`, `ms%h_layer`,
!! `ss%tau_x`, ...) becomes a plain explicit-shape dummy on plain integer bounds.
!!
!! WHY: the btstep benchmark found this worth 1.30x -- `bt_work%` components made
!! nvfortran emit 103 loads/thread where plain dummies emitted 38. `make regs`
!! shows EPBL's DC kernel doing 178 LDG against the CUDA port's 48, the same
!! signature. This file tests whether that gap is worth anything HERE.
!! Generated mechanically from ocean_epbl.F90 -- see the Makefile's
!! `ocean_epbl_plain.F90` rule. Do not hand-edit.
module ocean_epbl_plain
   use constants, only: wp, GRAVITY, H_DIV_EPS
   use grid, only: hgrid_t
   use epbl_stubs, only: multilayer_cgrid_state_t, ocean_surface_stress_t, &
                             ocean_eos_t, eos_specvol_derivs
   use ocean_epbl, only: ocean_epbl_t, epbl_find_mstar, epbl_mixlen_shape, &
                             epbl_lf17_wave_state, epbl_lf17_la, epbl_lt_enhance
   implicit none
   private
   public :: epbl_column_kernel_plain
   real(wp), parameter :: H_NEGLECT = H_DIV_EPS
   integer, parameter :: EPBL_VSTAR_RH18 = 2
contains
   pure subroutine epbl_column_kernel_plain(grid, this, ms, ss, dt, inv_rho0_cp, inv_rho0, &
                                            nx_arg, ny_arg, nxd, nyd, nzd, hT, hS, &
                                            h_layer, wet_mask, tau_x, tau_y, &
                                            Q_heat_field, Q_salt_field, f_centre, &
                                            mld_a, la_a, kd_int, t0_d, s0_d, dpe_t_d, &
                                            dpe_s_d, dch_t_d, dch_s_d, tke_wind_a, &
                                            tke_conv_a, tke_forcing_a, tke_mixing_a, &
                                            tke_mdec_a, tke_cdec_a)
      !! Per-column EPBL solve.  One `do concurrent (j, i)` with the
      !! serial work in k inside (j -> i -> k ordering); ALL sweep
      !! state is carried in scalars (design doc D6) — the only
      !! column arrays are the six iteration-invariant workspaces
      !! filled by the prep sweep.
      !!
      !! Kinematic surface fluxes are read per-column inside the DC:
      !!   q_t_kin = Q_heat_field(i,j) / (rho0 · cp)  [degC·m/s]
      !!   q_s_kin = Q_salt_field(i,j) / rho0           [PSU·m/s]
      !! For the constant-fill default both fields are uniform, giving
      !! arithmetic identical to the former scalar-broadcast path.
      !!
      !! Algorithm:
      !!   prep:  T0, S0 and the pressure-weighted PE / steric
      !!          sensitivities per layer (downward pressure sum).
      !!   outer: MLD root-find (false position / bisection) —
      !!          mstar and the mixing-length shape depend on MLD.
      !!   sweep: interfaces Ki = nz..2 downward.  Decay mech TKE
      !!          across the layer above; rotation-reduce the
      !!          convective reservoir; closed-form energy solve for
      !!          the largest affordable Kd (the gravity-wave
      !!          column-height correction folds into PEc_core);
      !!          advance the embedded tridiagonal forward
      !!          elimination (hp_a / dX_to_dPE_a / Th_a recursions).
      type(hgrid_t), intent(in) :: grid
      type(ocean_epbl_t), intent(inout) :: this
      type(multilayer_cgrid_state_t), intent(in) :: ms
      type(ocean_surface_stress_t), intent(in) :: ss
      real(wp), intent(in) :: dt
      real(wp), intent(in) :: inv_rho0_cp, inv_rho0
      integer, intent(in) :: nx_arg, ny_arg
      ! ---- THE SIGNATURE FIX ----
      ! Identical body; every ARRAY that production reads through a derived-type
      ! component is now a plain EXPLICIT-SHAPE dummy bounded by plain integer
      ! dummies. `this` is still passed, but only its SCALAR knobs are read.
      ! (RESUME_GPU_MRE §2: `h(grid%nx_total, ...)` is explicit-shape and STILL
      ! kills auto-collapse -- the bounds must be plain integers.)
      integer, intent(in) :: nxd, nyd, nzd
      real(wp), intent(in) :: hT(nxd, nyd, nzd), hS(nxd, nyd, nzd)
      real(wp), intent(in) :: h_layer(nxd, nyd, nzd), wet_mask(nxd, nyd)
      real(wp), intent(in) :: tau_x(nxd + 1, nyd), tau_y(nxd, nyd + 1)
      real(wp), intent(in) :: Q_heat_field(nxd, nyd), Q_salt_field(nxd, nyd)
      real(wp), intent(in) :: f_centre(nxd, nyd)
      real(wp), intent(inout) :: mld_a(nxd, nyd), la_a(nxd, nyd)
      real(wp), intent(inout) :: kd_int(nxd, nyd, nzd + 1)
      real(wp), intent(inout) :: t0_d(nxd, nyd, nzd), s0_d(nxd, nyd, nzd)
      real(wp), intent(inout) :: dpe_t_d(nxd, nyd, nzd), dpe_s_d(nxd, nyd, nzd)
      real(wp), intent(inout) :: dch_t_d(nxd, nyd, nzd), dch_s_d(nxd, nyd, nzd)
      real(wp), intent(inout) :: tke_wind_a(nxd, nyd), tke_conv_a(nxd, nyd)
      real(wp), intent(inout) :: tke_forcing_a(nxd, nyd), tke_mixing_a(nxd, nyd)
      real(wp), intent(inout) :: tke_mdec_a(nxd, nyd), tke_cdec_a(nxd, nyd)

      integer :: i, j, nx, ny, nz
      ! prep locals
      integer :: k
      real(wp) :: q_t_kin, q_s_kin
      real(wp) :: hk, hk_eff, inv_h, t0k, s0k, dmass, dpres, p_mid, pres
      real(wp) :: dsv_dt_k, dsv_ds_k, dsv_dt_sfc, dsv_ds_sfc, h_sum
      ! forcing / environment locals
      real(wp) :: tau_xc, tau_yc, ustar, absf, idecay, mech_in
      real(wp) :: b0, ctke_sfc
      ! MLD iteration locals
      integer :: obl_it, n_its
      real(wp) :: min_mld, max_mld, mld_guess, mld_found
      real(wp) :: dmld_min, dmld_max
      logical :: have_min, have_max
      real(wp) :: mstar_val, mech_tke, conv_perel, forcing_clip
      ! sweep carried scalars
      integer :: ki, ka, kb
      real(wp) :: htot, z_int, pres_int, mld_output
      logical :: sfc_connected, sfc_disconnect
      real(wp) :: hp_a, dpe_t_a, dpe_s_a, dch_t_a, dch_s_a
      real(wp) :: th_a, sh_a, te_lag, se_lag, kddt_prev, kddt_cur
      real(wp) :: exp_kh, nstar_fc, tot_tke, dt_h, tke_here, vstar
      real(wp) :: h_ka, h_kb, hb_hs, shape_fn, hbs, mixlen, kd_g0
      real(wp) :: hp_b, th_b, sh_b, hps, bdt1, dt_c, ds_c
      real(wp) :: pec_core, colht_core, dkddt, pe_max
      real(wp) :: kd_val, tke_used, frac_bl, pe_g0, dpe_conv
      real(wp) :: b1, c1, r_reduc, te_new, se_new, surf_scale
      ! Langmuir (LF17) wave state + Langmuir number
      real(wp) :: lt_u10, lt_ustokes, lt_kphil, la_val
      ! TKE budget ledger (final iteration's values survive)
      real(wp) :: d_wind, d_conv, d_forcing, d_mixing, d_mdecay, d_cdecay

      nx = grid%nx_total
      ny = grid%ny_total
      nz = ms%nz_ml

      do concurrent(j=1:ny, i=1:nx) &
         local(k, hk, hk_eff, inv_h, t0k, s0k, dmass, dpres, p_mid, pres, &
               dsv_dt_k, dsv_ds_k, dsv_dt_sfc, dsv_ds_sfc, h_sum, &
               tau_xc, tau_yc, ustar, absf, idecay, mech_in, b0, ctke_sfc, &
               obl_it, n_its, min_mld, max_mld, mld_guess, mld_found, &
               dmld_min, dmld_max, have_min, have_max, &
               mstar_val, mech_tke, conv_perel, forcing_clip, &
               ki, ka, kb, htot, z_int, pres_int, mld_output, &
               sfc_connected, sfc_disconnect, &
               hp_a, dpe_t_a, dpe_s_a, dch_t_a, dch_s_a, &
               th_a, sh_a, te_lag, se_lag, kddt_prev, kddt_cur, &
               exp_kh, nstar_fc, tot_tke, dt_h, tke_here, vstar, &
               h_ka, h_kb, hb_hs, shape_fn, hbs, mixlen, kd_g0, &
               hp_b, th_b, sh_b, hps, bdt1, dt_c, ds_c, &
               pec_core, colht_core, dkddt, pe_max, &
               kd_val, tke_used, frac_bl, pe_g0, dpe_conv, &
               b1, c1, r_reduc, te_new, se_new, surf_scale, &
               lt_u10, lt_ustokes, lt_kphil, la_val, &
               d_wind, d_conv, d_forcing, d_mixing, d_mdecay, d_cdecay, &
               q_t_kin, q_s_kin)

         ! ---- Column prep: T0/S0 + PE/steric weights (downward) ----
         pres = 0.0_wp
         h_sum = 0.0_wp
         dsv_dt_sfc = 0.0_wp
         dsv_ds_sfc = 0.0_wp
         do k = nz, 1, -1
            hk = h_layer(i, j, k)
            if (hk > 0.0_wp) then
               inv_h = 1.0_wp/(hk + H_NEGLECT)
               t0k = hT(i, j, k)*inv_h
               s0k = hS(i, j, k)*inv_h
            else
               t0k = 0.0_wp
               s0k = 0.0_wp
            end if
            dmass = this%rho0*hk
            dpres = GRAVITY*dmass
            p_mid = pres + 0.5_wp*dpres
            call eos_specvol_derivs(this%eos, t0k, s0k, p_mid, dsv_dt_k, dsv_ds_k)
            t0_d(i, j, k) = t0k
            s0_d(i, j, k) = s0k
            dpe_t_d(i, j, k) = dmass*p_mid*dsv_dt_k
            dpe_s_d(i, j, k) = dmass*p_mid*dsv_ds_k
            dch_t_d(i, j, k) = dmass*dsv_dt_k
            dch_s_d(i, j, k) = dmass*dsv_ds_k
            if (k == nz) then
               dsv_dt_sfc = dsv_dt_k
               dsv_ds_sfc = dsv_ds_k
            end if
            pres = pres + dpres
            h_sum = h_sum + hk
         end do
         h_sum = h_sum + H_NEGLECT

         ! ---- Surface forcing energetics ----
         ! Kinematic fluxes at this column:
         !   q_T_kin = Q_heat(i,j) / (rho0·cp)  [degC·m/s]
         !   q_S_kin = Q_salt(i,j) / rho0         [PSU·m/s]
         q_t_kin = inv_rho0_cp*Q_heat_field(i, j)
         q_s_kin = inv_rho0*Q_salt_field(i, j)
         ! B0 = g rho0 (dSV_dT q_T + dSV_dS q_S); > 0 stabilizing.
         ! cTKE_sfc: PE released (> 0) / required (< 0) to homogenize
         ! the freshly applied skin fluxes through the surface layer
         ! — reduces to -0.5 rho0 h_sfc B0 dt.
         b0 = GRAVITY*this%rho0*(dsv_dt_sfc*q_t_kin + dsv_ds_sfc*q_s_kin)
         ctke_sfc = -0.5_wp*GRAVITY*this%rho0**2*h_layer(i, j, nz)* &
                    (q_t_kin*dt*dsv_dt_sfc + q_s_kin*dt*dsv_ds_sfc)

         tau_xc = 0.5_wp*(tau_x(i, j) + tau_x(i + 1, j))
         tau_yc = 0.5_wp*(tau_y(i, j) + tau_y(i, j + 1))
         ustar = max(sqrt(sqrt(tau_xc*tau_xc + tau_yc*tau_yc)/this%rho0), &
                     this%ustar_min)
         absf = sqrt((1.0_wp - this%omega_frac)*f_centre(i, j)**2 + &
                     this%omega_frac*4.0_wp*this%omega**2)
         idecay = this%tke_decay*absf/ustar
         mech_in = dt*this%rho0*ustar**3

         ! LF17 wave state is BLD-independent: compute once per
         ! column; only the surface-layer average inside the MLD
         ! iteration depends on the guess.
         la_val = 0.0_wp
         lt_u10 = 0.0_wp
         lt_ustokes = 0.0_wp
         lt_kphil = 0.0_wp
         if (this%use_lt) then
            call epbl_lf17_wave_state(ustar, this%rho0, lt_u10, lt_ustokes, lt_kphil)
         end if

         d_wind = 0.0_wp
         d_conv = 0.0_wp
         d_forcing = 0.0_wp
         d_mixing = 0.0_wp
         d_mdecay = 0.0_wp
         d_cdecay = 0.0_wp
         mld_found = 0.0_wp

         if (wet_mask(i, j) <= 0.0_wp .or. h_sum <= 2.0_wp*H_NEGLECT) then
            ! Dry / land column: no mixing.
            do k = 1, nz + 1
               kd_int(i, j, k) = 0.0_wp
            end do
         else

            ! ---- Outer MLD iteration ----
            min_mld = 0.0_wp
            max_mld = h_sum
            dmld_min = 0.0_wp
            dmld_max = 0.0_wp
            have_min = .false.
            have_max = .false.
            mld_guess = 0.5_wp*(min_mld + max_mld)
            if (this%mld_use_prev_guess .and. mld_a(i, j) > 0.0_wp) then
               mld_guess = min(mld_a(i, j), max_mld)
            end if
            n_its = 1
            if (this%mld_iteration) n_its = this%mld_max_its

            do obl_it = 1, n_its
               ! Budget ledger restarts each iteration; the last
               ! iteration's values are the ones reported.
               d_wind = 0.0_wp
               d_conv = 0.0_wp
               d_forcing = 0.0_wp
               d_mixing = 0.0_wp
               d_mdecay = 0.0_wp
               d_cdecay = 0.0_wp

               ! (A) mstar at the current MLD guess.
               call epbl_find_mstar(this%mstar_scheme, this%mstar_const, &
                                    this%mstar_cap, this%mstar_coef1, &
                                    this%c_ek, this%mstar_conv_adj, &
                                    this%rh18_cn1, this%rh18_cn2, this%rh18_cn3, &
                                    this%rh18_cs1, this%rh18_cs2, &
                                    b0, ustar, mld_guess, absf, mstar_val)
               if (this%use_lt) then
                  la_val = epbl_lf17_la(ustar, this%la_frac_hbl*mld_guess, &
                                        lt_ustokes, lt_kphil)
                  call epbl_lt_enhance(this%lt_scheme, this%lt_enhance_coef, &
                                       this%lt_enhance_exp, this%lt_max_enhance, &
                                       this%von_karman, &
                                       this%lt_lac1, this%lt_lac2, this%lt_lac3, &
                                       this%lt_lac4, this%lt_lac5, &
                                       la_val, b0, ustar, mld_guess, absf, mstar_val)
               end if
               mech_tke = mstar_val*mech_in
               d_wind = mech_tke

               ! (B) seed the reservoirs from the surface forcing.
               if (ctke_sfc <= 0.0_wp) then
                  forcing_clip = max(ctke_sfc, -mech_tke)
                  mech_tke = mech_tke + forcing_clip
                  conv_perel = 0.0_wp
                  d_forcing = forcing_clip
               else
                  conv_perel = ctke_sfc
                  d_conv = this%nstar*ctke_sfc
               end if

               ! (C) sweep initialization at the surface layer.
               h_ka = h_layer(i, j, nz) + H_NEGLECT
               hp_a = h_ka
               dpe_t_a = dpe_t_d(i, j, nz)
               dpe_s_a = dpe_s_d(i, j, nz)
               dch_t_a = dch_t_d(i, j, nz)
               dch_s_a = dch_s_d(i, j, nz)
               th_a = h_ka*t0_d(i, j, nz)
               sh_a = h_ka*s0_d(i, j, nz)
               te_lag = 0.0_wp
               se_lag = 0.0_wp
               kddt_prev = 0.0_wp
               htot = h_layer(i, j, nz)
               z_int = h_layer(i, j, nz)
               pres_int = GRAVITY*this%rho0*h_layer(i, j, nz)
               mld_output = h_layer(i, j, nz)
               sfc_connected = .true.
               kd_int(i, j, nz + 1) = 0.0_wp

               ! (D) downward sweep over interfaces Ki = nz .. 2.
               ! Layer above the interface: ka = Ki; below: kb = Ki-1.
               do ki = nz, 2, -1
                  ka = ki
                  kb = ki - 1
                  h_ka = h_layer(i, j, ka) + H_NEGLECT
                  h_kb = h_layer(i, j, kb) + H_NEGLECT
                  sfc_disconnect = .false.

                  ! (1) mechanical TKE decays across the layer above
                  !     (Ekman-scale e-folding; no decay at f = 0).
                  exp_kh = exp(-h_ka*idecay)
                  d_mdecay = d_mdecay + (1.0_wp - exp_kh)*mech_tke
                  mech_tke = mech_tke*exp_kh

                  ! (2) per-layer convective forcing accrual: only the
                  !     surface layer carries cTKE today (scalar
                  !     fluxes, no penetrating shortwave), and it
                  !     seeds the reservoirs in (B).  When SW lands,
                  !     positive cTKE(kb) accrues into conv_perel here
                  !     and negative cTKE(kb) drains tot_tke after (3).

                  ! (3) rotation-reduced convective efficiency.
                  nstar_fc = this%nstar
                  if (conv_perel > 0.0_wp .and. absf > 0.0_wp) then
                     nstar_fc = this%nstar*conv_perel/ &
                                (conv_perel + 0.2_wp* &
                                 sqrt(0.5_wp*dt*this%rho0*(absf*htot)**3*conv_perel))
                  end if
                  tot_tke = mech_tke + nstar_fc*conv_perel

                  ! (5) static-stability short-circuit: no energy and
                  !     a stable interface => no mixing here.
                  if (tot_tke <= 0.0_wp .and. &
                      0.0_wp <= (dch_t_d(i, j, kb) + dch_t_d(i, j, ka))* &
                      (t0_d(i, j, ka) - t0_d(i, j, kb)) + &
                      (dch_s_d(i, j, kb) + dch_s_d(i, j, ka))* &
                      (s0_d(i, j, ka) - s0_d(i, j, kb))) then
                     kd_val = 0.0_wp
                     sfc_disconnect = .true.
                  else
                     ! (6) velocity scale, mixing length, first-guess Kd.
                     dt_h = dt/max(0.5_wp*(h_ka + h_kb), 1.0e-15_wp*h_sum)
                     tke_here = mech_tke + this%wstar_ustar_coef*conv_perel
                     hb_hs = (h_sum - z_int)/h_sum
                     shape_fn = epbl_mixlen_shape(z_int, mld_guess, &
                                                  this%translay_scale, &
                                                  this%mixlen_exponent, &
                                                  this%mld_iteration)
                     hbs = min(hb_hs, shape_fn)
                     if (tke_here > 0.0_wp) then
                        if (this%vstar_scheme == EPBL_VSTAR_RH18) then
                           surf_scale = max(0.05_wp, 1.0_wp - htot/mld_guess)
                           vstar = this%vstar_scale_fac*surf_scale* &
                                   (this%vstar_surf_fac*ustar + &
                                    (this%wstar_ustar_coef*conv_perel/ &
                                     (dt*this%rho0))**(1.0_wp/3.0_wp))
                        else
                           vstar = this%vstar_scale_fac* &
                                   (tke_here/(dt*this%rho0))**(1.0_wp/3.0_wp)
                        end if
                        if (this%mld_iteration) then
                           mixlen = max(this%min_mix_len, &
                                        (htot*hbs*vstar)/ &
                                        (this%ekman_scale_coef*absf*htot*hbs + vstar))
                           kd_g0 = vstar*this%von_karman*mixlen
                        else
                           kd_g0 = vstar*this%von_karman*(htot*hbs*vstar)/ &
                                   (this%ekman_scale_coef*absf*htot*hbs + vstar)
                        end if
                     else
                        vstar = 0.0_wp
                        kd_g0 = 0.0_wp
                     end if

                     ! (7) pivot quantities for the layer below.
                     hp_b = h_kb
                     th_b = h_kb*t0_d(i, j, kb)
                     sh_b = h_kb*s0_d(i, j, kb)

                     ! (8) closed-form energy solve (direct path).
                     hps = hp_a + hp_b
                     bdt1 = hp_a*hp_b
                     dt_c = hp_a*th_b - hp_b*th_a
                     ds_c = hp_a*sh_b - hp_b*sh_a
                     pec_core = hp_b*(dpe_t_a*dt_c + dpe_s_a*ds_c) - &
                                hp_a*(dpe_t_d(i, j, kb)*dt_c + &
                                      dpe_s_d(i, j, kb)*ds_c)
                     colht_core = hp_b*(dch_t_a*dt_c + dch_s_a*ds_c) - &
                                  hp_a*(dch_t_d(i, j, kb)*dt_c + &
                                        dch_s_d(i, j, kb)*ds_c)
                     ! Gravity-wave radiation correction: a shrinking
                     ! column radiates energy that cannot drive mixing.
                     if (colht_core < 0.0_wp) then
                        pec_core = pec_core - pres_int*colht_core
                     end if
                     dkddt = kd_g0*dt_h
                     pe_max = pec_core/(bdt1*hps)

                     if (pe_max < 0.0_wp) then
                        ! (8a) convectively unstable: mixing RELEASES
                        ! PE.  Recompute vstar with the released
                        ! energy included; Kd from the mixing length
                        ! (not energy-limited); bank the release.
                        tke_here = mech_tke + &
                                   this%wstar_ustar_coef*(conv_perel - pe_max)
                        if (tke_here > 0.0_wp) then
                           if (this%vstar_scheme == EPBL_VSTAR_RH18) then
                              surf_scale = max(0.05_wp, 1.0_wp - htot/mld_guess)
                              vstar = this%vstar_scale_fac*surf_scale* &
                                      (this%vstar_surf_fac*ustar + &
                                       (this%wstar_ustar_coef*conv_perel/ &
                                        (dt*this%rho0))**(1.0_wp/3.0_wp))
                           else
                              vstar = this%vstar_scale_fac* &
                                      (tke_here/(dt*this%rho0))**(1.0_wp/3.0_wp)
                           end if
                           if (this%mld_iteration) then
                              mixlen = max(this%min_mix_len, &
                                           (htot*hbs*vstar)/ &
                                           (this%ekman_scale_coef*absf*htot*hbs + vstar))
                              kd_val = vstar*this%von_karman*mixlen
                           else
                              kd_val = vstar*this%von_karman*(htot*hbs*vstar)/ &
                                       (this%ekman_scale_coef*absf*htot*hbs + vstar)
                           end if
                        else
                           vstar = 0.0_wp
                           kd_val = 0.0_wp
                        end if
                        pe_g0 = pec_core*dkddt/(bdt1*(bdt1 + dkddt*hps))
                        dpe_conv = pec_core*(kd_val*dt_h)/ &
                                   (bdt1*(bdt1 + (kd_val*dt_h)*hps))
                        if (dpe_conv > 0.0_wp) then
                           kd_val = kd_g0
                           dpe_conv = pe_g0
                        end if
                        ! dpe_conv < 0 => the reservoir grows; the
                        ! reservoirs are NOT proportionally drained on
                        ! this branch.
                        conv_perel = conv_perel - dpe_conv
                        d_conv = d_conv - this%nstar*dpe_conv
                        if (sfc_connected) then
                           mld_output = mld_output + h_layer(i, j, kb)
                        end if
                     else
                        ! (8b) stable: direct closed-form Kd from the
                        ! energy budget; drain the reservoirs.
                        if ((pec_core*dkddt <= &
                             tot_tke*(bdt1*(bdt1 + dkddt*hps))) .or. &
                            (pec_core <= 0.0_wp)) then
                           kd_val = kd_g0
                           tke_used = pec_core*dkddt/(bdt1*(bdt1 + dkddt*hps))
                           frac_bl = 1.0_wp
                        else
                           kd_val = (bdt1**2*tot_tke)/ &
                                    (dt_h*(pec_core - bdt1*hps*tot_tke))
                           tke_used = tot_tke
                           frac_bl = tot_tke*(bdt1*(bdt1 + dkddt*hps))/ &
                                     (pec_core*dkddt)
                        end if
                        if (sfc_connected) then
                           mld_output = mld_output + frac_bl*h_layer(i, j, kb)
                        end if
                        if (frac_bl < 1.0_wp) sfc_disconnect = .true.
                        r_reduc = 0.0_wp
                        if (tot_tke > 0.0_wp .and. tot_tke > tke_used) then
                           r_reduc = (tot_tke - tke_used)/tot_tke
                        end if
                        d_mixing = d_mixing + tke_used
                        d_cdecay = d_cdecay + &
                                   (1.0_wp - r_reduc)*(this%nstar - nstar_fc)*conv_perel
                        mech_tke = r_reduc*mech_tke
                        conv_perel = r_reduc*conv_perel
                     end if
                  end if

                  kd_int(i, j, ki) = kd_val
                  kddt_cur = kd_val*dt_h

                  ! (9) advance the embedded forward elimination.
                  b1 = 1.0_wp/(hp_a + kddt_cur)
                  c1 = kddt_cur*b1
                  if (ki == nz) then
                     te_lag = b1*(h_ka*t0_d(i, j, ka))
                     se_lag = b1*(h_ka*s0_d(i, j, ka))
                  else
                     te_new = b1*(h_ka*t0_d(i, j, ka) + kddt_prev*te_lag)
                     se_new = b1*(h_ka*s0_d(i, j, ka) + kddt_prev*se_lag)
                     te_lag = te_new
                     se_lag = se_new
                  end if
                  hp_a = h_kb + (hp_a*b1)*kddt_cur
                  dpe_t_a = dpe_t_d(i, j, kb) + c1*dpe_t_a
                  dpe_s_a = dpe_s_d(i, j, kb) + c1*dpe_s_a
                  dch_t_a = dch_t_d(i, j, kb) + c1*dch_t_a
                  dch_s_a = dch_s_d(i, j, kb) + c1*dch_s_a
                  th_a = h_kb*t0_d(i, j, kb) + kddt_cur*te_lag
                  sh_a = h_kb*s0_d(i, j, kb) + kddt_cur*se_lag
                  kddt_prev = kddt_cur
                  if (sfc_disconnect) then
                     htot = h_layer(i, j, kb)
                     sfc_connected = .false.
                  else
                     htot = htot + h_layer(i, j, kb)
                  end if
                  z_int = z_int + h_layer(i, j, kb)
                  pres_int = pres_int + GRAVITY*this%rho0*h_layer(i, j, kb)
               end do
               kd_int(i, j, 1) = 0.0_wp

               ! Leftover stocks at the bed count as dissipated.
               d_mdecay = d_mdecay + mech_tke
               d_cdecay = d_cdecay + this%nstar*conv_perel

               mld_found = mld_output
               if (.not. this%mld_iteration) exit
               if (abs(mld_found - mld_guess) < this%mld_tol) exit
               if (obl_it == n_its) exit

               ! Bracket update + next guess (false position with a
               ! bisection fallback; bisection-only when configured).
               if (mld_found > mld_guess) then
                  min_mld = mld_guess
                  dmld_min = mld_found - mld_guess
                  have_min = .true.
               else
                  max_mld = mld_guess
                  dmld_max = mld_found - mld_guess
                  have_max = .true.
               end if
               if (this%mld_bisection) then
                  mld_guess = 0.5_wp*(min_mld + max_mld)
               else if (have_min .and. have_max .and. obl_it > 2 .and. &
                        mod(obl_it - 1, 4) > 0) then
                  mld_guess = min_mld + dmld_min*(max_mld - min_mld)/ &
                              (dmld_min - dmld_max)
               else if (mld_found > min_mld .and. mld_found < max_mld) then
                  mld_guess = mld_found
               else
                  mld_guess = 0.5_wp*(min_mld + max_mld)
               end if
            end do

         end if

         mld_a(i, j) = mld_found
         la_a(i, j) = la_val
         if (this%tke_diags) then
            tke_wind_a(i, j) = d_wind/dt
            tke_conv_a(i, j) = d_conv/dt
            tke_forcing_a(i, j) = d_forcing/dt
            tke_mixing_a(i, j) = d_mixing/dt
            tke_mdec_a(i, j) = d_mdecay/dt
            tke_cdec_a(i, j) = d_cdecay/dt
         end if
      end do
   end subroutine epbl_column_kernel_plain

end module ocean_epbl_plain
