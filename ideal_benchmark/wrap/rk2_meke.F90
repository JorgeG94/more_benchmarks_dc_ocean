#include "directives.h"
!! RK2 wrapper for the MEKE kernel.
!! Entry compute: meke_step_ext_fused (fused opt-DC, 6 loops).
!! Init copied verbatim from meke/drivers/dc_ab.F90 (alloc_meke + init_meke).
module rk2_meke_mod
   use, intrinsic :: iso_c_binding, only: c_int, c_double
   use constants, only: wp
   use meke_state, only: ocean_metrics_t, multilayer_cgrid_state_t, ocean_gm_t, ocean_meke_t
   use meke, only: meke_step_ext_fused
   implicit none
   private
   public :: rk2_meke_init, rk2_meke_stage, rk2_meke_stage_cuda, rk2_meke_probe

   integer, parameter :: NGH = 3
   real(wp), parameter :: DXM = 8000.0_wp, DYM = 8000.0_wp, DT_THERMO = 1800.0_wp

   type(ocean_metrics_t), save :: met
   type(multilayer_cgrid_state_t), save :: ms
   type(ocean_gm_t), save :: gm
   type(ocean_meke_t), save :: m
   real(wp), allocatable, target, save :: ke_diss_ext(:, :), scratch(:, :)
   integer, save :: gnx, gny, gnz

   interface
      ! extern "C" meke_opt_launch_flat (opt_kernel.cu) -- from meke_bridge.F90.
      subroutine meke_opt_launch_flat( &
         meke, kh_diff, le, ku, i_mass, depth_tot, bottom_fac2, barotr_fac2, src, &
         uflux, vflux, mass_ws, rd_ws, sn_u_ws, sn_v_ws, ke_diss_ws, &
         u_bbl2, f_centre, gm_src, ke_diss_ext, h_layer, rho_layer, &
         areaT, iareaT, idxT, idyT, dy_cu, dx_cv, idxCu, idyCv, &
         nx, ny, nz, &
         dt, dtscale, cd_scale, cb, ct, min_gamma2, cdrag, uscale, &
         a_deform, a_rhines, a_eady, a_frict, a_grid, &
         bgsrc, gmcoeff, frcoeff, damping, &
         kh_bg, k4, khmeke_fac, khcoeff, visc_coeff_ku, rho0, backscatter, &
         meke_scratch) bind(C, name="meke_opt_launch_flat")
         import :: wp, c_int
         real(wp) :: meke(*), kh_diff(*), le(*), ku(*), i_mass(*), depth_tot(*)
         real(wp) :: bottom_fac2(*), barotr_fac2(*), src(*), uflux(*), vflux(*)
         real(wp) :: mass_ws(*), rd_ws(*), sn_u_ws(*), sn_v_ws(*), ke_diss_ws(*)
         real(wp) :: u_bbl2(*), f_centre(*), gm_src(*), ke_diss_ext(*)
         real(wp) :: h_layer(*), rho_layer(*)
         real(wp) :: areaT(*), iareaT(*), idxT(*), idyT(*)
         real(wp) :: dy_cu(*), dx_cv(*), idxCu(*), idyCv(*)
         real(wp) :: meke_scratch(*)
         integer(c_int), value :: nx, ny, nz, backscatter
         real(wp), value :: dt, dtscale, cd_scale, cb, ct, min_gamma2, cdrag, uscale
         real(wp), value :: a_deform, a_rhines, a_eady, a_frict, a_grid
         real(wp), value :: bgsrc, gmcoeff, frcoeff, damping
         real(wp), value :: kh_bg, k4, khmeke_fac, khcoeff, visc_coeff_ku, rho0
      end subroutine meke_opt_launch_flat
   end interface

contains

   subroutine rk2_meke_init(nxp, nyp, nz) bind(C, name="rk2_meke_init")
      integer(c_int), value :: nxp, nyp, nz
      integer :: i, j, k, nx, ny
      nx = nxp + 2*NGH; ny = nyp + 2*NGH
      gnx = nx; gny = ny; gnz = nz

      allocate (met%areaT(nx, ny), met%iareaT(nx, ny), met%idxT(nx, ny), met%idyT(nx, ny))
      allocate (met%dy_cu(nx + 1, ny), met%dx_cv(nx, ny + 1))
      allocate (met%idxCu(nx + 1, ny), met%idyCv(nx, ny + 1))
      allocate (ms%h_layer(nx, ny, nz), ms%rho_layer(nx, ny, nz))
      allocate (gm%gm_src(nx, ny), ke_diss_ext(nx, ny), scratch(nx, ny))
      scratch = 0.0_wp
      ms%nz_ml = nz; gm%is_init = .true.

      met%areaT = DXM*DYM; met%iareaT = 1.0_wp/(DXM*DYM)
      met%idxT = 1.0_wp/DXM; met%idyT = 1.0_wp/DYM
      met%dy_cu = DYM; met%dx_cv = DXM
      met%idxCu = 1.0_wp/DXM; met%idyCv = 1.0_wp/DYM
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               ms%h_layer(i, j, k) = 4000.0_wp/real(nz, wp)
               ms%rho_layer(i, j, k) = 1025.0_wp + 5.0_wp*real(nz - k, wp)/real(nz, wp)
            end do
         end do
      end do
      do j = 1, ny
         do i = 1, nx
            gm%gm_src(i, j) = 1.0e-2_wp*exp(-((real(i - nx/2, wp)/real(max(nx/6, 1), wp))**2 &
                                              + (real(j - ny/2, wp)/real(max(ny/6, 1), wp))**2))
            ke_diss_ext(i, j) = -1.0e-3_wp
         end do
      end do

      call alloc_meke(nx, ny, nz)
      call init_meke(nx, ny)

      DC_ENTER_IN(met)
      DC_ENTER_IN(met%areaT)
      DC_ENTER_IN(met%iareaT)
      DC_ENTER_IN(met%idxT)
      DC_ENTER_IN(met%idyT)
      DC_ENTER_IN(met%dy_cu)
      DC_ENTER_IN(met%dx_cv)
      DC_ENTER_IN(met%idxCu)
      DC_ENTER_IN(met%idyCv)
      DC_ENTER_IN(ms)
      DC_ENTER_IN(ms%h_layer)
      DC_ENTER_IN(ms%rho_layer)
      DC_ENTER_IN(gm)
      DC_ENTER_IN(gm%gm_src)
      DC_ENTER_IN(ke_diss_ext)
      DC_ENTER_CREATE(scratch)
      DC_ENTER_IN(m)
      DC_ENTER_IN(m%meke)
      DC_ENTER_IN(m%kh_diff)
      DC_ENTER_IN(m%u_bbl2)
      DC_ENTER_IN(m%f_centre)
      DC_ENTER_CREATE(m%le)
      DC_ENTER_CREATE(m%ku)
      DC_ENTER_CREATE(m%i_mass)
      DC_ENTER_CREATE(m%depth_tot)
      DC_ENTER_CREATE(m%bottom_fac2)
      DC_ENTER_CREATE(m%barotr_fac2)
      DC_ENTER_CREATE(m%src)
      DC_ENTER_CREATE(m%uflux)
      DC_ENTER_CREATE(m%vflux)
      DC_ENTER_CREATE(m%del2)
      DC_ENTER_CREATE(m%mass_ws)
      DC_ENTER_CREATE(m%ke_diss_ws)
      DC_ENTER_CREATE(m%rd_ws)
      DC_ENTER_CREATE(m%sn_u_ws)
      DC_ENTER_CREATE(m%sn_v_ws)
      DC_WAIT
   end subroutine rk2_meke_init

   subroutine alloc_meke(nx, ny, nz)
      integer, intent(in) :: nx, ny, nz
      allocate (m%meke(nx, ny), m%kh_diff(nx, ny), m%le(nx, ny), m%ku(nx, ny))
      allocate (m%i_mass(nx, ny), m%depth_tot(nx, ny))
      allocate (m%bottom_fac2(nx, ny), m%barotr_fac2(nx, ny), m%src(nx, ny))
      allocate (m%uflux(nx + 1, ny), m%vflux(nx, ny + 1), m%del2(nx, ny))
      allocate (m%mass_ws(nx, ny), m%u_bbl2(nx, ny), m%ke_diss_ws(nx, ny))
      allocate (m%rd_ws(nx, ny), m%f_centre(nx, ny))
      allocate (m%sn_u_ws(nx + 1, ny), m%sn_v_ws(nx, ny + 1))
      allocate (m%baro_hu(nx + 1, ny), m%baro_hv(nx, ny + 1))
      m%nx_total = nx; m%ny_total = ny; m%nz_ml = nz; m%is_init = .true.
   end subroutine alloc_meke

   subroutine init_meke(nx, ny)
      integer, intent(in) :: nx, ny
      integer :: ii, jj
      m%enable = .true.
      m%gmcoeff = 0.9_wp
      m%khcoeff = 1.0_wp
      m%cdrag = 2.5e-3_wp
      m%kh = 100.0_wp
      do jj = 1, ny
         do ii = 1, nx
            m%meke(ii, jj) = 1.0e-3_wp + 1.0e-2_wp* &
               exp(-((real(ii - nx/3, wp)/real(max(nx/8, 1), wp))**2 &
                     + (real(jj - ny/3, wp)/real(max(ny/8, 1), wp))**2))
            m%f_centre(ii, jj) = abs(-1.0e-4_wp + 2.0e-11_wp*real(jj - ny/2, wp)*DYM)
         end do
      end do
      m%kh_diff = 0.0_wp; m%le = 0.0_wp; m%ku = 0.0_wp
      m%i_mass = 0.0_wp; m%depth_tot = 0.0_wp
      m%bottom_fac2 = 0.0_wp; m%barotr_fac2 = 0.0_wp; m%src = 0.0_wp
      m%uflux = 0.0_wp; m%vflux = 0.0_wp; m%del2 = 0.0_wp
      m%mass_ws = 0.0_wp
      m%u_bbl2 = 0.0_wp
      m%ke_diss_ws = 0.0_wp; m%rd_ws = 0.0_wp
      m%sn_u_ws = 0.0_wp; m%sn_v_ws = 0.0_wp
      m%baro_hu = 0.0_wp; m%baro_hv = 0.0_wp
   end subroutine init_meke

   subroutine rk2_meke_stage() bind(C, name="rk2_meke_stage")
      call meke_step_ext_fused(gnx, gny, gnz, m, met, gm, ms, DT_THERMO, ke_diss_ext)
   end subroutine rk2_meke_stage

   subroutine rk2_meke_stage_cuda() bind(C, name="rk2_meke_stage_cuda")
      integer(c_int) :: bscat
      bscat = 0
      if (m%backscatter) bscat = 1
      !$acc host_data use_device(m%meke, m%kh_diff, m%le, m%ku, m%i_mass, &
      !$acc   m%depth_tot, m%bottom_fac2, m%barotr_fac2, m%src, m%uflux, m%vflux, &
      !$acc   m%mass_ws, m%rd_ws, m%sn_u_ws, m%sn_v_ws, m%ke_diss_ws, &
      !$acc   m%u_bbl2, m%f_centre, gm%gm_src, ke_diss_ext, ms%h_layer, ms%rho_layer, &
      !$acc   met%areaT, met%iareaT, met%idxT, met%idyT, met%dy_cu, met%dx_cv, &
      !$acc   met%idxCu, met%idyCv, scratch)
      call meke_opt_launch_flat( &
         m%meke, m%kh_diff, m%le, m%ku, m%i_mass, m%depth_tot, m%bottom_fac2, &
         m%barotr_fac2, m%src, m%uflux, m%vflux, m%mass_ws, m%rd_ws, &
         m%sn_u_ws, m%sn_v_ws, m%ke_diss_ws, &
         m%u_bbl2, m%f_centre, gm%gm_src, ke_diss_ext, ms%h_layer, ms%rho_layer, &
         met%areaT, met%iareaT, met%idxT, met%idyT, met%dy_cu, met%dx_cv, &
         met%idxCu, met%idyCv, &
         gnx, gny, gnz, &
         DT_THERMO, m%dtscale, m%cd_scale, m%cb, m%ct, m%min_gamma2, m%cdrag, m%uscale, &
         m%alpha_deform, m%alpha_rhines, m%alpha_eady, m%alpha_frict, m%alpha_grid, &
         m%bgsrc, m%gmcoeff, m%frcoeff, m%damping, &
         m%kh, m%k4, m%khmeke_fac, m%khcoeff, m%visc_coeff_ku, gm%rho0, bscat, &
         scratch)
      !$acc end host_data
   end subroutine rk2_meke_stage_cuda

   subroutine rk2_meke_probe(vmin, vmax) bind(C, name="rk2_meke_probe")
      real(c_double), intent(out) :: vmin, vmax
      DC_UPDATE_SELF(m%meke)
      vmin = minval(m%meke); vmax = maxval(m%meke)
   end subroutine rk2_meke_probe

end module rk2_meke_mod
