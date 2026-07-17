#include "directives.h"
!! RK2 wrapper for the redi (ocean_redi) kernel.
!! Entry compute: redi_calc_coeffs (Phase A) + redi_apply_flux_hoist (Phase B).
!! Init copied verbatim from redi/drivers/dc_ab.F90 (build_state + enter_data),
!! then one calc_coeffs is run so the first stage has populated slots too.
module rk2_redi_mod
   use, intrinsic :: iso_c_binding, only: c_int, c_double
   use constants, only: wp, NZ_STACK_MAX
   use grid, only: hgrid_t
   use ocean_metrics, only: ocean_metrics_t
   use multilayer_cgrid_state, only: multilayer_cgrid_state_t
   use ocean_eos, only: ocean_eos_t
   use ocean_boundary_types, only: ocean_bc_state_t, OBC_WALL
   use ocean_redi, only: ocean_redi_t, redi_calc_coeffs, redi_apply_flux_hoist
   implicit none
   private
   public :: rk2_redi_init, rk2_redi_stage, rk2_redi_stage_cuda, rk2_redi_probe

   integer, parameter :: NGHOST = 3
   real(wp), parameter :: DT = 900.0_wp

   interface
      ! extern "C" redi_opt_launch (opt_kernel.cu) -- lifted from redi_bridge.F90.
      subroutine redi_opt_launch(nx, ny, nz, ns, dt, nghost, nxp, nyp, &
                                 ww, we, wsz, wn, rho0, T_ref, S_ref, alpha_T, beta_S, &
                                 h_layer, t_htr, s_htr, wet_u, wet_v, &
                                 uPoL, uPoR, uKoL, uKoR, uhEff, &
                                 vPoL, vPoR, vKoL, vKoR, vhEff, &
                                 khtr_u_ext, khtr_v_ext, khtr_u, khtr_v, &
                                 dy_cu, dx_cv, idxCu, idyCv, areaT, tr_snap) &
         bind(C, name="redi_opt_launch")
         import :: wp
         integer :: nx, ny, nz, ns, nghost, nxp, nyp, ww, we, wsz, wn
         real(wp) :: dt, rho0, T_ref, S_ref, alpha_T, beta_S
         real(wp) :: h_layer(*), t_htr(*), s_htr(*), wet_u(*), wet_v(*)
         real(wp) :: uPoL(*), uPoR(*), uhEff(*), vPoL(*), vPoR(*), vhEff(*)
         integer :: uKoL(*), uKoR(*), vKoL(*), vKoR(*)
         real(wp) :: khtr_u_ext(*), khtr_v_ext(*), khtr_u(*), khtr_v(*)
         real(wp) :: dy_cu(*), dx_cv(*), idxCu(*), idyCv(*), areaT(*), tr_snap(*)
      end subroutine redi_opt_launch
   end interface

   type(hgrid_t), save :: grid
   type(ocean_metrics_t), save :: metrics
   type(ocean_eos_t), save :: eos
   type(ocean_bc_state_t), save :: bc
   type(multilayer_cgrid_state_t), save :: ms
   type(ocean_redi_t), save :: redi
   real(wp), allocatable, save :: khtr_u_ext(:, :), khtr_v_ext(:, :)

contains

   subroutine rk2_redi_init(nxp, nyp, nz) bind(C, name="rk2_redi_init")
      integer(c_int), value :: nxp, nyp, nz
      integer :: nx, ny
      nx = nxp + 2*NGHOST; ny = nyp + 2*NGHOST
      grid%nx_total = nx; grid%ny_total = ny
      grid%nx_phys = nxp; grid%ny_phys = nyp
      grid%nghost = NGHOST; grid%dx = 0.1_wp; grid%dy = 0.1_wp

      bc%west%bc_type = OBC_WALL; bc%east%bc_type = OBC_WALL
      bc%south%bc_type = OBC_WALL; bc%north%bc_type = OBC_WALL

      allocate (metrics%dy_cu(nx + 1, ny), source=11100.0_wp)
      allocate (metrics%dx_cv(nx, ny + 1), source=8000.0_wp)
      allocate (metrics%idxCu(nx + 1, ny), source=1.0_wp/8000.0_wp)
      allocate (metrics%idyCv(nx, ny + 1), source=1.0_wp/11100.0_wp)
      allocate (metrics%areaT(nx, ny), source=8000.0_wp*11100.0_wp)
      allocate (metrics%wet_u(nx + 1, ny), source=1.0_wp)
      allocate (metrics%wet_v(nx, ny + 1), source=1.0_wp)
      allocate (khtr_u_ext(nx + 1, ny), source=100.0_wp)
      allocate (khtr_v_ext(nx, ny + 1), source=100.0_wp)

      call build_state(ms, nx, ny, nz)
      call redi%init(grid, nz)
      redi%enable = .true.; redi%khtr = 100.0_wp

      DC_ENTER_IN(ms%h_layer)
      DC_ENTER_IN(ms%tracers(1)%hTr)
      DC_ENTER_IN(ms%tracers(2)%hTr)
      DC_ENTER_IN(metrics%dy_cu)
      DC_ENTER_IN(metrics%dx_cv)
      DC_ENTER_IN(metrics%idxCu)
      DC_ENTER_IN(metrics%idyCv)
      DC_ENTER_IN(metrics%areaT)
      DC_ENTER_IN(metrics%wet_u)
      DC_ENTER_IN(metrics%wet_v)
      DC_ENTER_IN(khtr_u_ext)
      DC_ENTER_IN(khtr_v_ext)
      DC_ENTER_CREATE(redi%uPoL)
      DC_ENTER_CREATE(redi%uPoR)
      DC_ENTER_CREATE(redi%uKoL)
      DC_ENTER_CREATE(redi%uKoR)
      DC_ENTER_CREATE(redi%uhEff)
      DC_ENTER_CREATE(redi%vPoL)
      DC_ENTER_CREATE(redi%vPoR)
      DC_ENTER_CREATE(redi%vKoL)
      DC_ENTER_CREATE(redi%vKoR)
      DC_ENTER_CREATE(redi%vhEff)
      DC_ENTER_CREATE(redi%khtr_u)
      DC_ENTER_CREATE(redi%khtr_v)
      DC_ENTER_CREATE(redi%tr_snap)

      call redi_calc_coeffs(grid, metrics, eos, redi, ms)
      DC_WAIT
   end subroutine rk2_redi_init

   subroutine rk2_redi_stage() bind(C, name="rk2_redi_stage")
      ! Phase A: neutral-position coefficients; Phase B: hoisted apply-flux.
      call redi_calc_coeffs(grid, metrics, eos, redi, ms)
      call redi_apply_flux_hoist(grid, metrics, redi, ms, DT, khtr_u_ext, khtr_v_ext, bc)
   end subroutine rk2_redi_stage

   subroutine rk2_redi_stage_cuda() bind(C, name="rk2_redi_stage_cuda")
      ! One optimized CUDA launch = Phase A calc-coeffs + hoisted apply-flux,
      ! on the SAME device arrays rk2_redi_init entered. Lifted from redi_bridge.
      integer :: nx, ny, nz, ns
      nx = grid%nx_total; ny = grid%ny_total; nz = ms%nz_ml; ns = redi%nsurf
      !$acc host_data use_device(ms%h_layer, ms%tracers(1)%hTr, ms%tracers(2)%hTr, &
      !$acc                      metrics%wet_u, metrics%wet_v, &
      !$acc                      redi%uPoL, redi%uPoR, redi%uKoL, redi%uKoR, redi%uhEff, &
      !$acc                      redi%vPoL, redi%vPoR, redi%vKoL, redi%vKoR, redi%vhEff, &
      !$acc                      khtr_u_ext, khtr_v_ext, redi%khtr_u, redi%khtr_v, &
      !$acc                      metrics%dy_cu, metrics%dx_cv, metrics%idxCu, metrics%idyCv, &
      !$acc                      metrics%areaT, redi%tr_snap)
      call redi_opt_launch(nx, ny, nz, ns, DT, grid%nghost, grid%nx_phys, grid%ny_phys, &
                           1, 1, 1, 1, &
                           eos%rho0, eos%T_ref, eos%S_ref, eos%alpha_T, eos%beta_S, &
                           ms%h_layer, ms%tracers(1)%hTr, ms%tracers(2)%hTr, &
                           metrics%wet_u, metrics%wet_v, &
                           redi%uPoL, redi%uPoR, redi%uKoL, redi%uKoR, redi%uhEff, &
                           redi%vPoL, redi%vPoR, redi%vKoL, redi%vKoR, redi%vhEff, &
                           khtr_u_ext, khtr_v_ext, redi%khtr_u, redi%khtr_v, &
                           metrics%dy_cu, metrics%dx_cv, metrics%idxCu, metrics%idyCv, &
                           metrics%areaT, redi%tr_snap)
      !$acc end host_data
   end subroutine rk2_redi_stage_cuda

   subroutine rk2_redi_probe(vmin, vmax) bind(C, name="rk2_redi_probe")
      real(c_double), intent(out) :: vmin, vmax
      DC_UPDATE_SELF(ms%tracers(1)%hTr)
      vmin = minval(ms%tracers(1)%hTr); vmax = maxval(ms%tracers(1)%hTr)
   end subroutine rk2_redi_probe

   subroutine build_state(s, nx, ny, nz)
      type(multilayer_cgrid_state_t), intent(inout) :: s
      integer, intent(in) :: nx, ny, nz
      integer :: ii, jj, kk
      real(wp) :: depth, zz, tt, ss, hh, fx, fy
      s%nz_ml = nz
      s%idx_temperature = 1
      s%idx_salinity = 2
      allocate (s%h_layer(nx, ny, nz))
      allocate (s%tracers(2))
      allocate (s%tracers(1)%hTr(nx, ny, nz))
      allocate (s%tracers(2)%hTr(nx, ny, nz))
      depth = 4000.0_wp
      hh = depth/real(nz, wp)
      do kk = 1, nz
         zz = (real(nz - kk, wp) + 0.5_wp)*hh
         do jj = 1, ny
            fy = real(jj - 1, wp)/real(ny - 1, wp)
            do ii = 1, nx
               fx = real(ii - 1, wp)/real(nx - 1, wp)
               tt = 2.0_wp + 16.0_wp*exp(-zz/800.0_wp) + 3.0_wp*fx + 3.0_wp*fy
               ss = 34.5_wp + 0.7_wp*(1.0_wp - exp(-zz/1500.0_wp)) - 0.2_wp*fx
               s%h_layer(ii, jj, kk) = hh
               s%tracers(1)%hTr(ii, jj, kk) = tt*hh
               s%tracers(2)%hTr(ii, jj, kk) = ss*hh
            end do
         end do
      end do
   end subroutine build_state

end module rk2_redi_mod
