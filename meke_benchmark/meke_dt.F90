!! MEKE step, DERIVED-TYPE-COMPONENT dummies — the COUNTERFACTUAL for H1.
!!
!! ⚠ THIS IS NOT PRODUCTION. It is production run BACKWARDS.
!!
!! The btstep benchmark found that passing arrays through derived-type
!! components (`bt_work%bt_eta(i,j)`) instead of as plain explicit-shape dummies
!! bounded by plain integers cost **1.30x** on an 11-loop, ~145k-cell kernel,
!! and that the cost scaled with the NUMBER OF DISTINCT ARRAYS a loop touches
!! (~19 arrays -> 1.3x; 6 arrays -> nothing).
!!
!! Production MEKE ALREADY passes plain dummies everywhere — there is no
!! signature fix left to apply. So the only way to measure what that existing
!! signature is WORTH is to undo it. This file is that undo: every kernel takes
!! `type(ocean_meke_t)` / `type(ocean_metrics_t)` / `type(ocean_gm_t)` /
!! `type(multilayer_cgrid_state_t)` and indexes the components directly, with
!! loop bounds read from `m%nx_total` / `m%ny_total` (derived-type components,
!! exactly the shape btstep's slow variant had).
!!
!! Everything else is held fixed: same 16 loops, same subroutine-per-loop
!! structure, same arithmetic in the same order. The ONLY variable is the
!! signature. It must be BIT-IDENTICAL to meke.F90 (the driver checks
!! `max|d meke| == 0.0` exactly) — if it is not, this is a transcription bug and
!! the comparison is void.
!!
!! Expected sign: SLOWER than meke.F90. Expected magnitude, from btstep's
!! array-count scaling: SMALL, because MEKE's loops touch 4-11 distinct arrays,
!! not 19. That is a prediction, and the number this file produces tests it.
module meke_dt
   use constants, only: wp, H_VANISHED
   use meke_state, only: ocean_metrics_t, multilayer_cgrid_state_t, &
                             ocean_gm_t, ocean_meke_t
   implicit none
   private
   public :: meke_step_dt

   real(wp), parameter :: MASS_NEGLECT = 1.0e-30_wp

contains

   subroutine meke_step_dt(nx, ny, nz, m, metrics, gm, ms, dt, ke_diss_ext)
      integer, intent(in) :: nx, ny, nz
      type(ocean_meke_t), intent(inout) :: m
      type(ocean_metrics_t), intent(in) :: metrics
      type(ocean_gm_t), intent(in) :: gm
      type(multilayer_cgrid_state_t), intent(in) :: ms
      real(wp), intent(in) :: dt
      real(wp), intent(in) :: ke_diss_ext(nx, ny)
      real(wp) :: sdt, sdt_damp, damp_step

      sdt = dt*m%dtscale
      damp_step = 1.0_wp
      if (m%kh >= 0.0_wp .or. m%k4 >= 0.0_wp) damp_step = 0.5_wp
      sdt_damp = sdt*damp_step

      call dt_zero_rd(m)                              ! 1
      call dt_zero_snu(m)                             ! 2
      call dt_zero_snv(m)                             ! 3
      call dt_mass(m, ms, gm)                         ! 4
      call dt_length_scales(m, metrics)               ! 5
      call dt_stage_kediss(m, ke_diss_ext, nx, ny)    ! 6
      call dt_source(m, gm, sdt)                      ! 7
      call dt_drag(m, sdt_damp)                       ! 8
      call dt_zero_uflux(m)                           ! 9
      call dt_uflux(m, metrics, sdt)                  ! 10
      call dt_zero_vflux(m)                           ! 11
      call dt_vflux(m, metrics, sdt)                  ! 12
      call dt_div(m, metrics, sdt)                    ! 13
      call dt_drag(m, sdt_damp)                       ! 14
      call dt_kh_closure(m)                           ! 15
      call dt_ku_closure(m)                           ! 16
   end subroutine meke_step_dt

   subroutine dt_zero_rd(m)
      type(ocean_meke_t), intent(inout) :: m
      integer :: i, j
      do concurrent(j=1:m%ny_total, i=1:m%nx_total)
         m%rd_ws(i, j) = 0.0_wp
      end do
   end subroutine

   subroutine dt_zero_snu(m)
      type(ocean_meke_t), intent(inout) :: m
      integer :: i, j
      do concurrent(j=1:m%ny_total, i=1:m%nx_total + 1)
         m%sn_u_ws(i, j) = 0.0_wp
      end do
   end subroutine

   subroutine dt_zero_snv(m)
      type(ocean_meke_t), intent(inout) :: m
      integer :: i, j
      do concurrent(j=1:m%ny_total + 1, i=1:m%nx_total)
         m%sn_v_ws(i, j) = 0.0_wp
      end do
   end subroutine

   subroutine dt_mass(m, ms, gm)
      type(ocean_meke_t), intent(inout) :: m
      type(multilayer_cgrid_state_t), intent(in) :: ms
      type(ocean_gm_t), intent(in) :: gm
      integer :: i, j, k
      real(wp) :: mass, dsum, hk, rhok
      do concurrent(j=1:m%ny_total, i=1:m%nx_total) local(k, mass, dsum, hk, rhok)
         mass = 0.0_wp
         dsum = 0.0_wp
         do k = 1, m%nz_ml
            hk = max(ms%h_layer(i, j, k), H_VANISHED)
            rhok = ms%rho_layer(i, j, k)
            mass = mass + rhok*hk
            dsum = dsum + ms%h_layer(i, j, k)
         end do
         m%depth_tot(i, j) = dsum
         m%mass_ws(i, j) = mass
         if (mass > 0.0_wp) then
            m%i_mass(i, j) = 1.0_wp/mass
         else
            m%i_mass(i, j) = 0.0_wp
         end if
      end do
   end subroutine

   pure function dt_inv_lmix(ueddy, sn, beta, area, rd_over_dx, depth, cdrag, &
                             a_deform, a_rhines, a_eady, a_frict, a_grid) result(inv_l)
      !$acc routine seq
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
   end function

   subroutine dt_length_scales(m, metrics)
      type(ocean_meke_t), intent(inout) :: m
      type(ocean_metrics_t), intent(in) :: metrics
      integer :: i, j, nx, ny
      real(wp) :: lgrid, ldeform, lfrict, ratio, bf2, tf2
      real(wp) :: ueddy, sn, beta, inv_l, rd
      nx = m%nx_total; ny = m%ny_total
      do concurrent(j=1:m%ny_total, i=1:m%nx_total) &
         local(lgrid, ldeform, lfrict, ratio, bf2, tf2, ueddy, sn, beta, inv_l, rd)
         rd = m%rd_ws(i, j)
         lgrid = sqrt(max(metrics%areaT(i, j), 0.0_wp))
         ldeform = lgrid*rd
         lfrict = 0.0_wp
         if (m%cdrag > 0.0_wp) lfrict = m%depth_tot(i, j)/m%cdrag
         bf2 = m%cd_scale*m%cd_scale
         if (lfrict*m%cb > 0.0_wp) then
            ratio = ldeform/lfrict
            bf2 = bf2 + 1.0_wp/(1.0_wp + m%cb*ratio)**0.8_wp
         end if
         bf2 = max(bf2, m%min_gamma2)
         m%bottom_fac2(i, j) = bf2
         tf2 = 1.0_wp
         if (lfrict*m%ct > 0.0_wp) then
            ratio = ldeform/lfrict
            tf2 = 1.0_wp/(1.0_wp + m%ct*ratio)**0.25_wp
         end if
         tf2 = max(tf2, m%min_gamma2)
         m%barotr_fac2(i, j) = tf2
         ueddy = sqrt(2.0_wp*max(0.0_wp, tf2*m%meke(i, j)))
         beta = 0.0_wp
         if (i > 1 .and. i < nx) then
            beta = beta + (0.5_wp*(m%f_centre(i + 1, j) - m%f_centre(i - 1, j))*metrics%idxT(i, j))**2
         end if
         if (j > 1 .and. j < ny) then
            beta = beta + (0.5_wp*(m%f_centre(i, j + 1) - m%f_centre(i, j - 1))*metrics%idyT(i, j))**2
         end if
         beta = sqrt(beta)
         sn = 0.0_wp
         if (m%alpha_eady > 0.0_wp) sn = 0.25_wp*((m%sn_u_ws(i, j) + m%sn_u_ws(i + 1, j)) + &
                                                  (m%sn_v_ws(i, j) + m%sn_v_ws(i, j + 1)))
         inv_l = dt_inv_lmix(ueddy, sn, beta, metrics%areaT(i, j), rd, &
                             m%depth_tot(i, j), m%cdrag, &
                             m%alpha_deform, m%alpha_rhines, m%alpha_eady, &
                             m%alpha_frict, m%alpha_grid)
         if (inv_l > 0.0_wp) then
            m%le(i, j) = 1.0_wp/inv_l
         else
            m%le(i, j) = 0.0_wp
         end if
      end do
   end subroutine

   subroutine dt_stage_kediss(m, ke_diss_ext, nx, ny)
      type(ocean_meke_t), intent(inout) :: m
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: ke_diss_ext(nx, ny)
      integer :: i, j
      do concurrent(j=1:m%ny_total, i=1:m%nx_total)
         m%ke_diss_ws(i, j) = ke_diss_ext(i, j)
      end do
   end subroutine

   subroutine dt_source(m, gm, sdt)
      type(ocean_meke_t), intent(inout) :: m
      type(ocean_gm_t), intent(in) :: gm
      real(wp), intent(in) :: sdt
      integer :: i, j
      real(wp) :: s
      do concurrent(j=1:m%ny_total, i=1:m%nx_total) local(s)
         s = m%bgsrc
         if (m%gmcoeff >= 0.0_wp) s = s + m%gmcoeff*m%i_mass(i, j)*gm%gm_src(i, j)
         if (m%frcoeff >= 0.0_wp) s = s - m%frcoeff*m%i_mass(i, j)*m%ke_diss_ws(i, j)
         m%src(i, j) = s
         m%meke(i, j) = m%meke(i, j) + sdt*s
      end do
   end subroutine

   subroutine dt_drag(m, sdt_damp)
      type(ocean_meke_t), intent(inout) :: m
      real(wp), intent(in) :: sdt_damp
      integer :: i, j
      real(wp) :: drag_rate, damp_rate, cd2, e
      cd2 = m%cdrag*m%cdrag
      do concurrent(j=1:m%ny_total, i=1:m%nx_total) local(drag_rate, damp_rate, e)
         e = m%meke(i, j)
         drag_rate = m%i_mass(i, j)*sqrt(cd2*(max(0.0_wp, 2.0_wp*m%bottom_fac2(i, j)*e) &
                                              + m%u_bbl2(i, j) + m%uscale*m%uscale))
         damp_rate = m%damping + drag_rate*m%bottom_fac2(i, j)
         if (e < 0.0_wp) damp_rate = 0.0_wp
         m%meke(i, j) = e/(1.0_wp + sdt_damp*damp_rate)
      end do
   end subroutine

   subroutine dt_zero_uflux(m)
      type(ocean_meke_t), intent(inout) :: m
      integer :: i, j
      do concurrent(j=1:m%ny_total, i=1:m%nx_total + 1)
         m%uflux(i, j) = 0.0_wp
      end do
   end subroutine

   subroutine dt_zero_vflux(m)
      type(ocean_meke_t), intent(inout) :: m
      integer :: i, j
      do concurrent(j=1:m%ny_total + 1, i=1:m%nx_total)
         m%vflux(i, j) = 0.0_wp
      end do
   end subroutine

   subroutine dt_uflux(m, metrics, sdt)
      type(ocean_meke_t), intent(inout) :: m
      type(ocean_metrics_t), intent(in) :: metrics
      real(wp), intent(in) :: sdt
      integer :: i, j
      real(wp) :: kh_u, geo, hm, inv_max
      do concurrent(j=1:m%ny_total, i=2:m%nx_total) local(kh_u, geo, hm, inv_max)
         geo = metrics%dy_cu(i, j)*metrics%idxCu(i, j)
         kh_u = max(0.0_wp, m%kh) + m%khmeke_fac*0.5_wp*(m%kh_diff(i - 1, j) + m%kh_diff(i, j))
         inv_max = 2.0_wp*sdt*(geo*max(metrics%iareaT(i - 1, j), metrics%iareaT(i, j)))
         if (kh_u*inv_max > 0.25_wp) kh_u = 0.25_wp/inv_max
         hm = 2.0_wp*m%mass_ws(i - 1, j)*m%mass_ws(i, j)/ &
              ((m%mass_ws(i - 1, j) + m%mass_ws(i, j)) + MASS_NEGLECT)
         m%uflux(i, j) = (kh_u*geo)*hm*(m%meke(i - 1, j) - m%meke(i, j))
      end do
   end subroutine

   subroutine dt_vflux(m, metrics, sdt)
      type(ocean_meke_t), intent(inout) :: m
      type(ocean_metrics_t), intent(in) :: metrics
      real(wp), intent(in) :: sdt
      integer :: i, j
      real(wp) :: kh_v, geo, hm, inv_max
      do concurrent(j=2:m%ny_total, i=1:m%nx_total) local(kh_v, geo, hm, inv_max)
         geo = metrics%dx_cv(i, j)*metrics%idyCv(i, j)
         kh_v = max(0.0_wp, m%kh) + m%khmeke_fac*0.5_wp*(m%kh_diff(i, j - 1) + m%kh_diff(i, j))
         inv_max = 2.0_wp*sdt*(geo*max(metrics%iareaT(i, j - 1), metrics%iareaT(i, j)))
         if (kh_v*inv_max > 0.25_wp) kh_v = 0.25_wp/inv_max
         hm = 2.0_wp*m%mass_ws(i, j - 1)*m%mass_ws(i, j)/ &
              ((m%mass_ws(i, j - 1) + m%mass_ws(i, j)) + MASS_NEGLECT)
         m%vflux(i, j) = (kh_v*geo)*hm*(m%meke(i, j - 1) - m%meke(i, j))
      end do
   end subroutine

   subroutine dt_div(m, metrics, sdt)
      type(ocean_meke_t), intent(inout) :: m
      type(ocean_metrics_t), intent(in) :: metrics
      real(wp), intent(in) :: sdt
      integer :: i, j
      real(wp) :: mke
      do concurrent(j=1:m%ny_total, i=1:m%nx_total) local(mke)
         mke = sdt*(metrics%iareaT(i, j)*m%i_mass(i, j))* &
               ((m%uflux(i, j) - m%uflux(i + 1, j)) + (m%vflux(i, j) - m%vflux(i, j + 1)))
         m%meke(i, j) = m%meke(i, j) + mke
      end do
   end subroutine

   subroutine dt_kh_closure(m)
      type(ocean_meke_t), intent(inout) :: m
      integer :: i, j
      real(wp) :: ueddy
      do concurrent(j=1:m%ny_total, i=1:m%nx_total) local(ueddy)
         if (m%khcoeff > 0.0_wp) then
            ueddy = sqrt(2.0_wp*max(0.0_wp, m%barotr_fac2(i, j)*m%meke(i, j)))
            m%kh_diff(i, j) = m%khcoeff*ueddy*m%le(i, j)
         else
            m%kh_diff(i, j) = 0.0_wp
         end if
      end do
   end subroutine

   subroutine dt_ku_closure(m)
      type(ocean_meke_t), intent(inout) :: m
      integer :: i, j
      real(wp) :: ueddy
      do concurrent(j=1:m%ny_total, i=1:m%nx_total) local(ueddy)
         if (m%backscatter .and. m%visc_coeff_ku /= 0.0_wp) then
            ueddy = sqrt(2.0_wp*max(0.0_wp, m%meke(i, j)))
            m%ku(i, j) = m%visc_coeff_ku*ueddy*m%le(i, j)
         else
            m%ku(i, j) = 0.0_wp
         end if
      end do
   end subroutine

end module meke_dt
