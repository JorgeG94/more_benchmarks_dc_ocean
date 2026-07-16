!! FULLY HAND-INLINED variant of kernel_flux.F90's HLL path.
!!
!! Question under test: with the compiler's inlining DECISION removed entirely —
!! no `-Minline`, no `acc routine seq`, no call at all — does `do concurrent`
!! close the gap to CUDA C?
!!
!! HOW: the C preprocessor pastes the helper bodies at every call site (.F90 =>
!! CPP runs). This is textual substitution before nvfortran ever sees the code,
!! so there is no inlining heuristic left to blame. The result is one flat loop
!! body: no flux_cell, no minmod, no extract_velocity, no hll_flux_x/y.
!!
!! FAITHFULNESS: every macro is the original body verbatim, same op order, same
!! branches, same parenthesisation. Only the names of the temporaries change (a
!! macro cannot have locals, so each expansion writes into caller-supplied
!! scratch). The bench asserts bit-identity against the un-inlined kernel — if
!! that fails, this file is wrong, not interesting.
!!
!! Source of truth: kernel_flux.F90 (md5 1ea40efad8567a85914561db5bfc3a55).

! minmod(a,b) -> res.   Original at kernel_flux.F90:380.
#define MINMOD(res, aa, bb, t1, t2) \
   t1 = (aa); t2 = (bb); \
   if (t1*t2 <= 0.0_wp) then; res = 0.0_wp; \
   else if (abs(t1) < abs(t2)) then; res = t1; \
   else; res = t2; end if

! extract_velocity(h,hu,hv -> u,v).   Original at :395.
#define EXTRACT_VEL(hh, hhu, hhv, uu, vv) \
   if ((hh) > DRY_TOLERANCE) then; uu = (hhu)/(hh); vv = (hhv)/(hh); \
   else; uu = 0.0_wp; vv = 0.0_wp; end if

! hll_flux_x(h_l,hu_l,hv_l, h_r,hu_r,hv_r -> f_h,f_hu,f_hv).  Original at :410.
! Scratch: xul,xur,xvl,xvr,xal,xar,xsl,xsr,xden,xflh,xflhu,xflhv,xfrh,xfrhu,xfrhv
#define HLL_X(hl, hul, hvl, hr, hur, hvr, fh, fhu, fhv) \
   if ((hl) > DRY_TOLERANCE) then; xul = (hul)/(hl); xvl = (hvl)/(hl); \
   else; xul = 0.0_wp; xvl = 0.0_wp; end if; \
   if ((hr) > DRY_TOLERANCE) then; xur = (hur)/(hr); xvr = (hvr)/(hr); \
   else; xur = 0.0_wp; xvr = 0.0_wp; end if; \
   xal = sqrt(GRAVITY*max((hl), 0.0_wp)); \
   xar = sqrt(GRAVITY*max((hr), 0.0_wp)); \
   xsl = min(xul - xal, xur - xar); \
   xsr = max(xul + xal, xur + xar); \
   xflh = (hul); \
   xflhu = (hul)*xul + 0.5_wp*GRAVITY*(hl)*(hl); \
   xflhv = (hul)*xvl; \
   xfrh = (hur); \
   xfrhu = (hur)*xur + 0.5_wp*GRAVITY*(hr)*(hr); \
   xfrhv = (hur)*xvr; \
   if (xsl >= 0.0_wp) then; fh = xflh; fhu = xflhu; fhv = xflhv; \
   else if (xsr <= 0.0_wp) then; fh = xfrh; fhu = xfrhu; fhv = xfrhv; \
   else; xden = 1.0_wp/(xsr - xsl); \
   fh = (xsr*xflh - xsl*xfrh + xsl*xsr*((hr) - (hl)))*xden; \
   fhu = (xsr*xflhu - xsl*xfrhu + xsl*xsr*((hur) - (hul)))*xden; \
   fhv = (xsr*xflhv - xsl*xfrhv + xsl*xsr*((hvr) - (hvl)))*xden; end if

! hll_flux_y(h_b,hu_b,hv_b, h_t,hu_t,hv_t -> g_h,g_hu,g_hv).  Original at :471.
#define HLL_Y(hb, hub, hvb, ht, hut, hvt, gh, ghu, ghv) \
   if ((hb) > DRY_TOLERANCE) then; xul = (hub)/(hb); xvl = (hvb)/(hb); \
   else; xul = 0.0_wp; xvl = 0.0_wp; end if; \
   if ((ht) > DRY_TOLERANCE) then; xur = (hut)/(ht); xvr = (hvt)/(ht); \
   else; xur = 0.0_wp; xvr = 0.0_wp; end if; \
   xal = sqrt(GRAVITY*max((hb), 0.0_wp)); \
   xar = sqrt(GRAVITY*max((ht), 0.0_wp)); \
   xsl = min(xvl - xal, xvr - xar); \
   xsr = max(xvl + xal, xvr + xar); \
   xflh = (hvb); \
   xflhu = (hub)*xvl; \
   xflhv = (hvb)*xvl + 0.5_wp*GRAVITY*(hb)*(hb); \
   xfrh = (hvt); \
   xfrhu = (hut)*xvr; \
   xfrhv = (hvt)*xvr + 0.5_wp*GRAVITY*(ht)*(ht); \
   if (xsl >= 0.0_wp) then; gh = xflh; ghu = xflhu; ghv = xflhv; \
   else if (xsr <= 0.0_wp) then; gh = xfrh; ghu = xfrhu; ghv = xfrhv; \
   else; xden = 1.0_wp/(xsr - xsl); \
   gh = (xsr*xflh - xsl*xfrh + xsl*xsr*((ht) - (hb)))*xden; \
   ghu = (xsr*xflhu - xsl*xfrhu + xsl*xsr*((hut) - (hub)))*xden; \
   ghv = (xsr*xflhv - xsl*xfrhv + xsl*xsr*((hvt) - (hvb)))*xden; end if

#ifndef INL_VLEN
#define INL_VLEN 512
#endif

module kernel_flux_inline
   use constants, only: wp, GRAVITY, DRY_TOLERANCE, THIN_LAYER_THRESHOLD
   use grid, only: hgrid_t
   implicit none
   private
   public :: compute_flux_hll_inline

contains

   subroutine compute_flux_hll_inline(h, hu, hv, b, flux_h, flux_hu, flux_hv, &
                                      mass_flux_x, mass_flux_y, grid)
      type(hgrid_t), intent(in) :: grid
      real(wp), intent(in) :: h(grid%nx_total, grid%ny_total), &
                              hu(grid%nx_total, grid%ny_total), &
                              hv(grid%nx_total, grid%ny_total), &
                              b(grid%nx_total, grid%ny_total)
      real(wp), intent(out) :: flux_h(grid%nx_total, grid%ny_total), &
                               flux_hu(grid%nx_total, grid%ny_total), &
                               flux_hv(grid%nx_total, grid%ny_total)
      real(wp), intent(out) :: mass_flux_x(grid%nx_total, grid%ny_total), &
                               mass_flux_y(grid%nx_total, grid%ny_total)

      integer :: i, j, nx, ny, nghost
      real(wp) :: dx, dy, half_g
      ! every temporary flux_cell + the helpers used, now all loop-private
      real(wp) :: eta_c, eta_im1, eta_ip1, eta_im2, eta_ip2
      real(wp) :: eta_jm1, eta_jp1, eta_jm2, eta_jp2
      real(wp) :: seta_x, shu_x, shv_x, seta_y, shu_y, shv_y
      real(wp) :: seta_x_nb, shu_x_nb, shv_x_nb, seta_y_nb, shu_y_nb, shv_y_nb
      real(wp) :: eta_L, eta_R, hu_L, hv_L, hu_R, hv_R, h_L, h_R
      real(wp) :: b_max, h_hyd_l, h_hyd_r, u_l, v_l, u_r, v_r
      real(wp) :: fx_h_l, fx_hu_l, fx_hv_l, fx_h_r, fx_hu_r, fx_hv_r
      real(wp) :: fy_h_b, fy_hu_b, fy_hv_b, fy_h_t, fy_hu_t, fy_hv_t
      real(wp) :: h_hyd_left, h_pre_left, h_hyd_right, h_pre_right
      real(wp) :: h_hyd_bottom, h_pre_bottom, h_hyd_top, h_pre_top
      real(wp) :: m1, m2                                    ! MINMOD scratch
      real(wp) :: xul, xur, xvl, xvr, xal, xar, xsl, xsr, xden   ! HLL scratch
      real(wp) :: xflh, xflhu, xflhv, xfrh, xfrhu, xfrhv

      nx = grid%nx_total; ny = grid%ny_total; nghost = grid%nghost
      dx = grid%dx; dy = grid%dy
      half_g = 0.5_wp*GRAVITY

      !$acc parallel loop collapse(2) gang vector vector_length(INL_VLEN) &
      !$acc&   present(h, hu, hv, b, flux_h, flux_hu, flux_hv, mass_flux_x, mass_flux_y) &
      !$acc&   private(eta_c, eta_im1, eta_ip1, eta_im2, eta_ip2, eta_jm1, eta_jp1, &
      !$acc&           eta_jm2, eta_jp2, seta_x, shu_x, shv_x, seta_y, shu_y, shv_y, &
      !$acc&           seta_x_nb, shu_x_nb, shv_x_nb, seta_y_nb, shu_y_nb, shv_y_nb, &
      !$acc&           eta_L, eta_R, hu_L, hv_L, hu_R, hv_R, h_L, h_R, b_max, &
      !$acc&           h_hyd_l, h_hyd_r, u_l, v_l, u_r, v_r, &
      !$acc&           fx_h_l, fx_hu_l, fx_hv_l, fx_h_r, fx_hu_r, fx_hv_r, &
      !$acc&           fy_h_b, fy_hu_b, fy_hv_b, fy_h_t, fy_hu_t, fy_hv_t, &
      !$acc&           h_hyd_left, h_pre_left, h_hyd_right, h_pre_right, &
      !$acc&           h_hyd_bottom, h_pre_bottom, h_hyd_top, h_pre_top, m1, m2, &
      !$acc&           xul, xur, xvl, xvr, xal, xar, xsl, xsr, xden, &
      !$acc&           xflh, xflhu, xflhv, xfrh, xfrhu, xfrhv)
      do j = nghost + 1, ny - nghost
      do i = nghost + 1, nx - nghost

         eta_c = h(i, j) + b(i, j)
         eta_im1 = h(i - 1, j) + b(i - 1, j)
         eta_ip1 = h(i + 1, j) + b(i + 1, j)
         eta_im2 = h(i - 2, j) + b(i - 2, j)
         eta_ip2 = h(i + 2, j) + b(i + 2, j)
         eta_jm1 = h(i, j - 1) + b(i, j - 1)
         eta_jp1 = h(i, j + 1) + b(i, j + 1)
         eta_jm2 = h(i, j - 2) + b(i, j - 2)
         eta_jp2 = h(i, j + 2) + b(i, j + 2)

         if (h(i, j) < THIN_LAYER_THRESHOLD .or. h(i - 1, j) < DRY_TOLERANCE &
             .or. h(i + 1, j) < DRY_TOLERANCE) then
            seta_x = 0.0_wp; shu_x = 0.0_wp; shv_x = 0.0_wp
         else
            MINMOD(seta_x, eta_c - eta_im1, eta_ip1 - eta_c, m1, m2)
            MINMOD(shu_x, hu(i, j) - hu(i - 1, j), hu(i + 1, j) - hu(i, j), m1, m2)
            MINMOD(shv_x, hv(i, j) - hv(i - 1, j), hv(i + 1, j) - hv(i, j), m1, m2)
         end if

         if (h(i, j) < THIN_LAYER_THRESHOLD .or. h(i, j - 1) < DRY_TOLERANCE &
             .or. h(i, j + 1) < DRY_TOLERANCE) then
            seta_y = 0.0_wp; shu_y = 0.0_wp; shv_y = 0.0_wp
         else
            MINMOD(seta_y, eta_c - eta_jm1, eta_jp1 - eta_c, m1, m2)
            MINMOD(shu_y, hu(i, j) - hu(i, j - 1), hu(i, j + 1) - hu(i, j), m1, m2)
            MINMOD(shv_y, hv(i, j) - hv(i, j - 1), hv(i, j + 1) - hv(i, j), m1, m2)
         end if

         ! ---- x: LEFT (i-1/2) ----
         eta_R = eta_c - 0.5_wp*seta_x
         hu_R = hu(i, j) - 0.5_wp*shu_x
         hv_R = hv(i, j) - 0.5_wp*shv_x
         h_R = max(0.0_wp, eta_R - b(i, j))
         h_pre_left = h_R
         if (h(i - 1, j) < THIN_LAYER_THRESHOLD .or. h(i - 2, j) < DRY_TOLERANCE &
             .or. h(i, j) < DRY_TOLERANCE) then
            seta_x_nb = 0.0_wp; shu_x_nb = 0.0_wp; shv_x_nb = 0.0_wp
         else
            MINMOD(seta_x_nb, eta_im1 - eta_im2, eta_c - eta_im1, m1, m2)
            MINMOD(shu_x_nb, hu(i - 1, j) - hu(i - 2, j), hu(i, j) - hu(i - 1, j), m1, m2)
            MINMOD(shv_x_nb, hv(i - 1, j) - hv(i - 2, j), hv(i, j) - hv(i - 1, j), m1, m2)
         end if
         eta_L = eta_im1 + 0.5_wp*seta_x_nb
         hu_L = hu(i - 1, j) + 0.5_wp*shu_x_nb
         hv_L = hv(i - 1, j) + 0.5_wp*shv_x_nb
         h_L = max(0.0_wp, eta_L - b(i - 1, j))
         b_max = max(b(i - 1, j), b(i, j))
         h_hyd_l = max(0.0_wp, h_L + b(i - 1, j) - b_max)
         h_hyd_r = max(0.0_wp, h_R + b(i, j) - b_max)
         h_hyd_left = h_hyd_r
         EXTRACT_VEL(h_L, hu_L, hv_L, u_l, v_l)
         EXTRACT_VEL(h_R, hu_R, hv_R, u_r, v_r)
         HLL_X(h_hyd_l, h_hyd_l*u_l, h_hyd_l*v_l, h_hyd_r, h_hyd_r*u_r, h_hyd_r*v_r, &
               fx_h_l, fx_hu_l, fx_hv_l)

         ! ---- x: RIGHT (i+1/2) ----
         eta_L = eta_c + 0.5_wp*seta_x
         hu_L = hu(i, j) + 0.5_wp*shu_x
         hv_L = hv(i, j) + 0.5_wp*shv_x
         h_L = max(0.0_wp, eta_L - b(i, j))
         h_pre_right = h_L
         if (h(i + 1, j) < THIN_LAYER_THRESHOLD .or. h(i, j) < DRY_TOLERANCE &
             .or. h(i + 2, j) < DRY_TOLERANCE) then
            seta_x_nb = 0.0_wp; shu_x_nb = 0.0_wp; shv_x_nb = 0.0_wp
         else
            MINMOD(seta_x_nb, eta_ip1 - eta_c, eta_ip2 - eta_ip1, m1, m2)
            MINMOD(shu_x_nb, hu(i + 1, j) - hu(i, j), hu(i + 2, j) - hu(i + 1, j), m1, m2)
            MINMOD(shv_x_nb, hv(i + 1, j) - hv(i, j), hv(i + 2, j) - hv(i + 1, j), m1, m2)
         end if
         eta_R = eta_ip1 - 0.5_wp*seta_x_nb
         hu_R = hu(i + 1, j) - 0.5_wp*shu_x_nb
         hv_R = hv(i + 1, j) - 0.5_wp*shv_x_nb
         h_R = max(0.0_wp, eta_R - b(i + 1, j))
         b_max = max(b(i, j), b(i + 1, j))
         h_hyd_l = max(0.0_wp, h_L + b(i, j) - b_max)
         h_hyd_r = max(0.0_wp, h_R + b(i + 1, j) - b_max)
         h_hyd_right = h_hyd_l
         EXTRACT_VEL(h_L, hu_L, hv_L, u_l, v_l)
         EXTRACT_VEL(h_R, hu_R, hv_R, u_r, v_r)
         HLL_X(h_hyd_l, h_hyd_l*u_l, h_hyd_l*v_l, h_hyd_r, h_hyd_r*u_r, h_hyd_r*v_r, &
               fx_h_r, fx_hu_r, fx_hv_r)

         ! ---- y: BOTTOM (j-1/2) ----
         eta_R = eta_c - 0.5_wp*seta_y
         hu_R = hu(i, j) - 0.5_wp*shu_y
         hv_R = hv(i, j) - 0.5_wp*shv_y
         h_R = max(0.0_wp, eta_R - b(i, j))
         h_pre_bottom = h_R
         if (h(i, j - 1) < THIN_LAYER_THRESHOLD .or. h(i, j - 2) < DRY_TOLERANCE &
             .or. h(i, j) < DRY_TOLERANCE) then
            seta_y_nb = 0.0_wp; shu_y_nb = 0.0_wp; shv_y_nb = 0.0_wp
         else
            MINMOD(seta_y_nb, eta_jm1 - eta_jm2, eta_c - eta_jm1, m1, m2)
            MINMOD(shu_y_nb, hu(i, j - 1) - hu(i, j - 2), hu(i, j) - hu(i, j - 1), m1, m2)
            MINMOD(shv_y_nb, hv(i, j - 1) - hv(i, j - 2), hv(i, j) - hv(i, j - 1), m1, m2)
         end if
         eta_L = eta_jm1 + 0.5_wp*seta_y_nb
         hu_L = hu(i, j - 1) + 0.5_wp*shu_y_nb
         hv_L = hv(i, j - 1) + 0.5_wp*shv_y_nb
         h_L = max(0.0_wp, eta_L - b(i, j - 1))
         b_max = max(b(i, j - 1), b(i, j))
         h_hyd_l = max(0.0_wp, h_L + b(i, j - 1) - b_max)
         h_hyd_r = max(0.0_wp, h_R + b(i, j) - b_max)
         h_hyd_bottom = h_hyd_r
         EXTRACT_VEL(h_L, hu_L, hv_L, u_l, v_l)
         EXTRACT_VEL(h_R, hu_R, hv_R, u_r, v_r)
         HLL_Y(h_hyd_l, h_hyd_l*u_l, h_hyd_l*v_l, h_hyd_r, h_hyd_r*u_r, h_hyd_r*v_r, &
               fy_h_b, fy_hu_b, fy_hv_b)

         ! ---- y: TOP (j+1/2) ----
         eta_L = eta_c + 0.5_wp*seta_y
         hu_L = hu(i, j) + 0.5_wp*shu_y
         hv_L = hv(i, j) + 0.5_wp*shv_y
         h_L = max(0.0_wp, eta_L - b(i, j))
         h_pre_top = h_L
         if (h(i, j + 1) < THIN_LAYER_THRESHOLD .or. h(i, j) < DRY_TOLERANCE &
             .or. h(i, j + 2) < DRY_TOLERANCE) then
            seta_y_nb = 0.0_wp; shu_y_nb = 0.0_wp; shv_y_nb = 0.0_wp
         else
            MINMOD(seta_y_nb, eta_jp1 - eta_c, eta_jp2 - eta_jp1, m1, m2)
            MINMOD(shu_y_nb, hu(i, j + 1) - hu(i, j), hu(i, j + 2) - hu(i, j + 1), m1, m2)
            MINMOD(shv_y_nb, hv(i, j + 1) - hv(i, j), hv(i, j + 2) - hv(i, j + 1), m1, m2)
         end if
         eta_R = eta_jp1 - 0.5_wp*seta_y_nb
         hu_R = hu(i, j + 1) - 0.5_wp*shu_y_nb
         hv_R = hv(i, j + 1) - 0.5_wp*shv_y_nb
         h_R = max(0.0_wp, eta_R - b(i, j + 1))
         b_max = max(b(i, j), b(i, j + 1))
         h_hyd_l = max(0.0_wp, h_L + b(i, j) - b_max)
         h_hyd_r = max(0.0_wp, h_R + b(i, j + 1) - b_max)
         h_hyd_top = h_hyd_l
         EXTRACT_VEL(h_L, hu_L, hv_L, u_l, v_l)
         EXTRACT_VEL(h_R, hu_R, hv_R, u_r, v_r)
         HLL_Y(h_hyd_l, h_hyd_l*u_l, h_hyd_l*v_l, h_hyd_r, h_hyd_r*u_r, h_hyd_r*v_r, &
               fy_h_t, fy_hu_t, fy_hv_t)

         ! ---- net divergence + well-balanced correction ----
         flux_h(i, j) = -(fx_h_r - fx_h_l)/dx - (fy_h_t - fy_h_b)/dy
         flux_hu(i, j) = -(fx_hu_r - fx_hu_l)/dx - (fy_hu_t - fy_hu_b)/dy &
                         - half_g*(h_hyd_left**2 - h_pre_left**2 &
                                   - h_hyd_right**2 + h_pre_right**2)/dx
         flux_hv(i, j) = -(fx_hv_r - fx_hv_l)/dx - (fy_hv_t - fy_hv_b)/dy &
                         - half_g*(h_hyd_bottom**2 - h_pre_bottom**2 &
                                   - h_hyd_top**2 + h_pre_top**2)/dy
         mass_flux_x(i, j) = fx_h_r
         mass_flux_y(i, j) = fy_h_t
         if (i == nghost + 1) mass_flux_x(i - 1, j) = fx_h_l
         if (j == nghost + 1) mass_flux_y(i, j - 1) = fy_h_b

      end do
      end do

   end subroutine compute_flux_hll_inline

end module kernel_flux_inline
