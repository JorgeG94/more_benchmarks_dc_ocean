!! MRE stubs for the barotropic substep benchmark.
!!
!! Field names, shapes and defaults follow production:
!!   barotropic_cgrid_workstate.F90  (barotropic_cgrid_workstate_t)
!!   grid.F90, ocean_metrics.F90
!! Only the fields the CLOSED-BASIN substep path touches are here.
module bt_state
   use constants, only: wp
   implicit none
   public

   type :: hgrid_t
      integer :: nx_total = 0, ny_total = 0
      integer :: nx_phys = 0, ny_phys = 0
      integer :: nghost = 3
   end type hgrid_t

   !! 2-D metrics. Curvilinear in production; this MRE fills them with the
   !! uniform-Cartesian values (dy_cu = dy etc.), which is what the gabight
   !! spherical configs are close to and what the kernel arithmetic assumes.
   type :: ocean_metrics_t
      real(wp), allocatable :: dy_cu(:, :)    !! east-face width  [m]   (nx+1, ny)
      real(wp), allocatable :: dx_cv(:, :)    !! north-face width [m]   (nx, ny+1)
      real(wp), allocatable :: iareaT(:, :)   !! 1/cell area    [1/m^2] (nx, ny)
      real(wp), allocatable :: areaCu(:, :)   !! u-cell area    [m^2]   (nx+1, ny)
      real(wp), allocatable :: areaCv(:, :)   !! v-cell area    [m^2]   (nx, ny+1)
      real(wp), allocatable :: dxCu(:, :)     !! (nx+1, ny)
      real(wp), allocatable :: dyCv(:, :)     !! (nx, ny+1)
      real(wp), allocatable :: idxCu(:, :)    !! 1/dxCu (nx+1, ny)
      real(wp), allocatable :: idyCv(:, :)    !! 1/dyCv (nx, ny+1)
      real(wp), allocatable :: iareaBu(:, :)  !! 1/corner area (nx+1, ny+1)
   end type ocean_metrics_t

   type :: coriolis_t
      real(wp), allocatable :: f_corner(:, :) !! (nx+1, ny+1)
   end type coriolis_t

   !! Subset of barotropic_cgrid_workstate_t: the closed-basin path.
   !! Omitted vs production: wd_* (wetdry off), BTCL_u/v + h_face_up_*
   !! (use_bt_cont_type / use_upstream_h_face off), F_slow_*, bt_*_end.
   type :: bt_work_t
      real(wp) :: g_bt = 9.81_wp
      real(wp) :: bebt = 0.0_wp
      real(wp), allocatable :: bt_eta(:, :)          !! (nx, ny)
      real(wp), allocatable :: bt_eta_new(:, :)      !! (nx, ny)
      real(wp), allocatable :: bt_H_ref(:, :)        !! (nx, ny)
      real(wp), allocatable :: bt_ubt(:, :)          !! (nx+1, ny)
      real(wp), allocatable :: bt_vbt(:, :)          !! (nx, ny+1)
      real(wp), allocatable :: bt_ubt_prev(:, :)     !! (nx+1, ny)
      real(wp), allocatable :: bt_vbt_prev(:, :)     !! (nx, ny+1)
      real(wp), allocatable :: bt_rem_u(:, :)        !! (nx+1, ny) drag; 1.0 when off
      real(wp), allocatable :: bt_rem_v(:, :)        !! (nx, ny+1)
      real(wp), allocatable :: bt_zeta_corner(:, :)  !! (nx+1, ny+1)
      real(wp), allocatable :: bt_ke_centre(:, :)    !! (nx, ny)
      real(wp), allocatable :: ubt_sum(:, :)         !! (nx+1, ny)
      real(wp), allocatable :: vbt_sum(:, :)         !! (nx, ny+1)
      real(wp), allocatable :: eta_sum(:, :)         !! (nx, ny)
      real(wp), allocatable :: uhbt_sum(:, :)        !! (nx+1, ny)
      real(wp), allocatable :: vhbt_sum(:, :)        !! (nx, ny+1)
   end type bt_work_t

end module bt_state
