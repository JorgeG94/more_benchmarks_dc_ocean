!! MRE stubs for the kappa-shear column kernel's environment: the horizontal
!! grid, the multilayer C-grid state, and the EOS handle. Every field name and
!! default is verbatim from production; only the UNUSED fields are dropped.
!!
!! Sources:
!!   hgrid_t                : <model>/src/core/structured/grid.F90
!!   multilayer_cgrid_state_t : <model>/src/core/structured/multilayer_cgrid_state.F90
!!   ocean_eos_t / eos_specvol_derivs :
!!                            <model>/src/equation_of_state/structured/ocean_eos.F90
!!
!! ⚠ THE TRACER REGISTRY IS FLATTENED. Production reaches temperature/salinity
!! through `ms%tracers(ms%idx_temperature)%hTr` — an array-of-derived-type
!! indirection that `kappa_shear_compute` host-dereferences before calling the
!! column kernel (that is the whole point of its outer shim). The column kernel
!! itself only ever sees two plain `hT(:,:,:)` / `hS(:,:,:)` dummies, so this
!! stub provides those directly and skips the registry. The kernel under test is
!! unaffected — it never touches `ms%tracers`.

module grid
   use constants, only: wp
   implicit none
   private
   public :: hgrid_t

   type :: hgrid_t
      integer  :: nx_total = 0   !! cells in x incl. ghosts
      integer  :: ny_total = 0   !! cells in y incl. ghosts
      integer  :: nx_phys = 0    !! interior cells in x
      integer  :: ny_phys = 0    !! interior cells in y
      integer  :: nghost = 3     !! production gabight configs: nghost = 3
      real(wp) :: dx = 0.0_wp
      real(wp) :: dy = 0.0_wp
   end type hgrid_t
end module grid

module multilayer_cgrid_state
   use constants, only: wp
   implicit none
   private
   public :: multilayer_cgrid_state_t

   !! Only the components `kappa_shear_column_kernel` reads:
   !!   nz_ml, wet_mask, h_layer, u_face_x_layer, v_face_y_layer.
   !! (idx_temperature / idx_salinity / tracers are consumed by the outer shim
   !! `kappa_shear_compute` on the HOST and never reach the device kernel.)
   type :: multilayer_cgrid_state_t
      integer :: nz_ml = 0
      real(wp), allocatable :: wet_mask(:, :)
      real(wp), allocatable :: h_layer(:, :, :)
      real(wp), allocatable :: u_face_x_layer(:, :, :)   !! (nx+1, ny, nz)
      real(wp), allocatable :: v_face_y_layer(:, :, :)   !! (nx, ny+1, nz)
   end type multilayer_cgrid_state_t
end module multilayer_cgrid_state

module ocean_eos
   use constants, only: wp
   implicit none
   private
   public :: ocean_eos_t, eos_specvol_derivs, eos_specvol_derivs_s
   public :: EOS_VARIANT_LINEAR, EOS_VARIANT_WRIGHT_97

   integer, parameter :: EOS_VARIANT_LINEAR = 1
   integer, parameter :: EOS_VARIANT_WRIGHT_97 = 2

   ! Wright (1997) Table A1 coefficients — verbatim from
   ! ocean_eos.F90:77-89. Retained so the WRIGHT branch is a faithful
   ! control; production gabight runs LINEAR (see the type's `variant` note).
   real(wp), parameter :: WRIGHT_A1 = 3.480336e-7_wp
   real(wp), parameter :: WRIGHT_A2 = -1.112733e-7_wp
   real(wp), parameter :: WRIGHT_B0 = 5.790749e8_wp
   real(wp), parameter :: WRIGHT_B1 = 3.516535e6_wp
   real(wp), parameter :: WRIGHT_B2 = -4.002714e4_wp
   real(wp), parameter :: WRIGHT_B3 = 2.084372e2_wp
   real(wp), parameter :: WRIGHT_B4 = 5.944068e5_wp
   real(wp), parameter :: WRIGHT_B5 = -9.643486e3_wp
   real(wp), parameter :: WRIGHT_C0 = 1.704853e5_wp
   real(wp), parameter :: WRIGHT_C1 = 7.904722e2_wp
   real(wp), parameter :: WRIGHT_C2 = -7.984422e0_wp
   real(wp), parameter :: WRIGHT_C3 = 5.140652e-2_wp
   real(wp), parameter :: WRIGHT_C4 = -2.302158e2_wp
   real(wp), parameter :: WRIGHT_C5 = -3.079464e0_wp

   !! Flat POD — no allocatable — so it copies into registers by value on the
   !! device. Verbatim field set from ocean_eos.F90 minus the unused
   !! T_ref/S_ref/p_ref/ts_convention (the LINEAR dSV branch reads only
   !! alpha_T, beta_S, rho0).
   type :: ocean_eos_t
      integer  :: variant = EOS_VARIANT_LINEAR
         !! Production gabight configs carry NO &ocean_eos_nml block, so the
         !! config default `eos = "linear"` (config.F90:264) stands.
      real(wp) :: rho0 = 1035.0_wp
      real(wp) :: alpha_T = 1.7e-4_wp
      real(wp) :: beta_S = 7.6e-4_wp
   end type ocean_eos_t

contains

   !! Verbatim from ocean_eos.F90 `eos_specvol_derivs`, minus the
   !! ROQUET_SPV branch (its point routine is 200 lines of coefficient table
   !! and production does not select it). LINEAR + WRIGHT retained.
   !! Scalar-argument form of eos_specvol_derivs. Same arithmetic, but takes the
   !! four EOS fields directly instead of the record.
   !!
   !! WHY IT EXISTS: amdflang's `-fdo-concurrent-to-openmp=device` cannot map a
   !! `do concurrent` live-in whose type has a derived-type component. ks.F90's
   !! loop was split so it captures no such type -- but it still had to pass
   !! `ocean_eos_t` down to the column solve. That record is FLAT, so it ought to
   !! be a legal live-in, and on NVIDIA it is. This routine removes the question
   !! entirely: with it, the `do concurrent` captures NO derived type of any
   !! shape, which is the only position that does not depend on how a particular
   !! offload pass treats records.
   pure subroutine eos_specvol_derivs_s(variant, rho0, alpha_T, beta_S, &
                                        T, S, p, dsv_dt, dsv_ds)
      !$acc routine seq
#ifdef DC_DATA_OMP
      !$omp declare target
#endif
      integer, intent(in) :: variant
      real(wp), intent(in) :: rho0, alpha_T, beta_S
      real(wp), intent(in) :: T, S, p
      real(wp), intent(out) :: dsv_dt, dsv_ds

      real(wp) :: T_sq, p_0, lambda, big_p, inv_p, inv_p2
      real(wp) :: dp0_dt, dlam_dt, dp0_ds, dlam_ds

      if (variant == EOS_VARIANT_WRIGHT_97) then
         T_sq = T*T
         p_0 = WRIGHT_B0 + WRIGHT_B1*T + WRIGHT_B2*T_sq + WRIGHT_B3*T_sq*T + &
               WRIGHT_B4*S + WRIGHT_B5*S*T
         lambda = WRIGHT_C0 + WRIGHT_C1*T + WRIGHT_C2*T_sq + WRIGHT_C3*T_sq*T + &
                  WRIGHT_C4*S + WRIGHT_C5*S*T
         dp0_dt = WRIGHT_B1 + 2.0_wp*WRIGHT_B2*T + 3.0_wp*WRIGHT_B3*T_sq + &
                  WRIGHT_B5*S
         dlam_dt = WRIGHT_C1 + 2.0_wp*WRIGHT_C2*T + 3.0_wp*WRIGHT_C3*T_sq + &
                   WRIGHT_C5*S
         dp0_ds = WRIGHT_B4 + WRIGHT_B5*T
         dlam_ds = WRIGHT_C4 + WRIGHT_C5*T
         big_p = p + p_0
         inv_p = 1.0_wp/big_p
         inv_p2 = inv_p*inv_p
         dsv_dt = WRIGHT_A1 + dlam_dt*inv_p - lambda*dp0_dt*inv_p2
         dsv_ds = WRIGHT_A2 + dlam_ds*inv_p - lambda*dp0_ds*inv_p2
      else
         dsv_dt = alpha_T/(rho0*rho0)
         dsv_ds = -beta_S/(rho0*rho0)
      end if
   end subroutine eos_specvol_derivs_s

   pure subroutine eos_specvol_derivs(eos, T, S, p, dsv_dt, dsv_ds)
      !$acc routine seq
      type(ocean_eos_t), intent(in) :: eos
      real(wp), intent(in) :: T, S
      real(wp), intent(in) :: p
      real(wp), intent(out) :: dsv_dt
      real(wp), intent(out) :: dsv_ds

      real(wp) :: T_sq, p_0, lambda, big_p, inv_p, inv_p2
      real(wp) :: dp0_dt, dlam_dt, dp0_ds, dlam_ds

      if (eos%variant == EOS_VARIANT_WRIGHT_97) then
         T_sq = T*T
         p_0 = WRIGHT_B0 + WRIGHT_B1*T + WRIGHT_B2*T_sq + WRIGHT_B3*T_sq*T + &
               WRIGHT_B4*S + WRIGHT_B5*S*T
         lambda = WRIGHT_C0 + WRIGHT_C1*T + WRIGHT_C2*T_sq + WRIGHT_C3*T_sq*T + &
                  WRIGHT_C4*S + WRIGHT_C5*S*T
         dp0_dt = WRIGHT_B1 + 2.0_wp*WRIGHT_B2*T + 3.0_wp*WRIGHT_B3*T_sq + &
                  WRIGHT_B5*S
         dlam_dt = WRIGHT_C1 + 2.0_wp*WRIGHT_C2*T + 3.0_wp*WRIGHT_C3*T_sq + &
                   WRIGHT_C5*S
         dp0_ds = WRIGHT_B4 + WRIGHT_B5*T
         dlam_ds = WRIGHT_C4 + WRIGHT_C5*T
         big_p = p + p_0
         inv_p = 1.0_wp/big_p
         inv_p2 = inv_p*inv_p
         dsv_dt = WRIGHT_A1 + dlam_dt*inv_p - lambda*dp0_dt*inv_p2
         dsv_ds = WRIGHT_A2 + dlam_ds*inv_p - lambda*dp0_ds*inv_p2
      else
         dsv_dt = eos%alpha_T/(eos%rho0*eos%rho0)
         dsv_ds = -eos%beta_S/(eos%rho0*eos%rho0)
      end if
   end subroutine eos_specvol_derivs

end module ocean_eos
