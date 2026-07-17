#include "directives.h"
!! DC A/B driver for the ALE-remap kernel: FAITHFUL vs OPTIMIZED, same binary.
!!
!!  A = ale_remap_step      (production do-concurrent orchestrator, verbatim)
!!  B = ale_remap_step_opt  (T+S geometry fusion + pre/post register hoists)
!!
!! Both run under the SAME device data layer, chosen entirely by directives.h:
!!   -DDC_DATA_ACC  -> OpenACC  (nvfortran -acc=gpu -stdpar=gpu)   GPU
!!   -DDC_DATA_OMP  -> OpenMP target                               GPU
!!   (neither)      -> host: bare DC on the CPU                    NO CUDA
!!
!! The remap has a FIXED POINT (after one call h_old == target_h), so the state
!! MUST be restored from a pristine device copy before every call -- both warm
!! and timed -- or a rep would measure a kernel doing strictly less work than
!! production. Timing is min over reps around a single call (restore untimed).
!!
!! Correctness: B is held bit-identical to A. Every shared output array
!! (h_layer, hTr_T, hTr_S, mass/heat/salt budgets, u, v, bt_eta) is captured
!! from a clean A remap and cross-checked against a clean B remap; the bar is
!! max rel diff < 1e-12.
!!
!! Usage: ./dc_ab [nx_pts] [ny_pts] [nz] [nreps] [nwarm] [hdrift_pct]
program dc_ab
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use constants, only: wp, REMAP_PPM, VCOORD_ZSTAR
   use remap_state, only: hgrid_t, ocean_vcoord_t, multilayer_cgrid_state_t
   use ale_remap, only: ale_remap_step, ale_remap_step_opt
   implicit none

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30
   integer, parameter :: REPS_DEF = 200, WARM_DEF = 10, NGHOST = 3
   integer, parameter :: PERT_DEF = 25

   type(hgrid_t) :: grid
   type(ocean_vcoord_t), target :: vc
   type(multilayer_cgrid_state_t), target :: ms
   real(wp), allocatable, target :: bt_eta(:, :), bt_H_ref(:, :)
   real(wp), allocatable :: p_h_layer(:, :, :), p_hTr_t(:, :, :), p_hTr_s(:, :, :)
   real(wp), allocatable :: p_u(:, :, :), p_v(:, :, :), p_eta(:, :)
   real(wp), allocatable :: p_mass(:, :, :), p_heat(:, :, :), p_salt(:, :, :)
   ! snapshots of the FAITHFUL (A) outputs, for the bit-identity check
   real(wp), allocatable :: a_h(:, :, :), a_hTr_t(:, :, :), a_hTr_s(:, :, :)
   real(wp), allocatable :: a_u(:, :, :), a_v(:, :, :), a_eta(:, :)
   real(wp), allocatable :: a_mass(:, :, :), a_heat(:, :, :), a_salt(:, :, :)

   real(wp) :: t0, t1, best_a, best_b, dt, ms_a, ms_b
   real(wp) :: hbed, sig, pert, cT, cS, csum, rmax
   integer :: i, j, k, rep, nx, ny, nz, nxp, nyp, n_reps, n_warm, pert_pct

   nxp = iarg(1, NXP_DEF); nyp = iarg(2, NYP_DEF); nz = iarg(3, NZ_DEF)
   n_reps = iarg(4, REPS_DEF); n_warm = iarg(5, WARM_DEF); pert_pct = iarg(6, PERT_DEF)
   nx = nxp + 2*NGHOST; ny = nyp + 2*NGHOST
   if (nx < 6 .or. ny < 6 .or. nz < 1) then
      write (output_unit, '(a)') 'ERROR: need nx_pts,ny_pts >= 1 and nz >= 1'; stop 1
   end if

   grid%nx_total = nx; grid%ny_total = ny
   vc%nx_total = nx; vc%ny_total = ny; vc%nz_ml = nz
   vc%coord_type = VCOORD_ZSTAR; vc%is_init = .true.; vc%remap_method = REMAP_PPM
   vc%regrid_time_scale = 0.0_wp; vc%remap_vel_conserve_ke = .false.
   ms%nz_ml = nz; ms%idx_temperature = 1; ms%idx_salinity = 2

   allocate (vc%dsig(nz), vc%target_h(nx, ny, nz), vc%remap_h_old(nx, ny, nz))
   allocate (vc%remap_total_h(nx, ny), vc%remap_h_ref(nx, ny))
   allocate (ms%h_layer(nx, ny, nz), ms%mass_budget_remap(nx, ny, nz))
   allocate (ms%heat_budget_remap(nx, ny, nz), ms%salt_budget_remap(nx, ny, nz))
   allocate (ms%u_face_x_layer(nx + 1, ny, nz), ms%v_face_y_layer(nx, ny + 1, nz))
   allocate (ms%tracers(2))
   allocate (ms%tracers(1)%hTr(nx, ny, nz), ms%tracers(2)%hTr(nx, ny, nz))
   allocate (bt_eta(nx, ny), bt_H_ref(nx, ny))
   allocate (p_h_layer(nx, ny, nz), p_hTr_t(nx, ny, nz), p_hTr_s(nx, ny, nz))
   allocate (p_u(nx + 1, ny, nz), p_v(nx, ny + 1, nz), p_eta(nx, ny))
   allocate (p_mass(nx, ny, nz), p_heat(nx, ny, nz), p_salt(nx, ny, nz))
   allocate (a_h(nx, ny, nz), a_hTr_t(nx, ny, nz), a_hTr_s(nx, ny, nz))
   allocate (a_u(nx + 1, ny, nz), a_v(nx, ny + 1, nz), a_eta(nx, ny))
   allocate (a_mass(nx, ny, nz), a_heat(nx, ny, nz), a_salt(nx, ny, nz))

   do k = 1, nz
      vc%dsig(k) = 1.0_wp + 3.0_wp*real(k - 1, wp)/real(max(1, nz - 1), wp)
   end do
   vc%dsig = vc%dsig/sum(vc%dsig)

   ! --- identical init to dc_main.F90 / ale_bench.F90 ---
   do j = 1, ny
      do i = 1, nx
         hbed = 15.0_wp + 4485.0_wp*abs(sin(0.013_wp*real(i, wp))*cos(0.017_wp*real(j, wp)))
         if (mod(i + j, 97) == 0) hbed = 3.0_wp     ! very shallow => vanished layers
         bt_H_ref(i, j) = hbed
         bt_eta(i, j) = 0.4_wp*sin(0.05_wp*real(i, wp) + 0.03_wp*real(j, wp))
         do k = 1, nz
            sig = vc%dsig(k)
            pert = (0.01_wp*real(pert_pct, wp))*sin(3.0_wp*real(k, wp) + 0.02_wp*real(i, wp) + 0.03_wp*real(j, wp))
            ms%h_layer(i, j, k) = (hbed + bt_eta(i, j))*sig*(1.0_wp + pert)
         end do
         csum = sum(ms%h_layer(i, j, 1:nz))
         ms%h_layer(i, j, 1:nz) = ms%h_layer(i, j, 1:nz)*(hbed + bt_eta(i, j))/csum
         do k = 1, nz
            cT = 2.0_wp + 12.0_wp*exp(-real(nz - k, wp)/6.0_wp) + 0.5_wp*sin(0.02_wp*real(i, wp))
            cS = 34.5_wp + 0.8_wp*real(k, wp)/real(nz, wp)
            ms%tracers(1)%hTr(i, j, k) = ms%h_layer(i, j, k)*cT
            ms%tracers(2)%hTr(i, j, k) = ms%h_layer(i, j, k)*cS
         end do
      end do
   end do
   ms%mass_budget_remap = 0.0_wp; ms%heat_budget_remap = 0.0_wp; ms%salt_budget_remap = 0.0_wp
   do k = 1, nz
      do j = 1, ny
         do i = 1, nx + 1
            ms%u_face_x_layer(i, j, k) = 0.3_wp*sin(0.01_wp*real(i, wp))* &
                                         cos(0.02_wp*real(j, wp))*(1.0_wp + 0.1_wp*real(k, wp))
         end do
      end do
      do j = 1, ny + 1
         do i = 1, nx
            ms%v_face_y_layer(i, j, k) = 0.2_wp*cos(0.015_wp*real(i, wp))* &
                                         sin(0.018_wp*real(j, wp))*(1.0_wp - 0.05_wp*real(k, wp))
         end do
      end do
   end do

   ! pristine copies for the mandatory per-rep restore (fixed-point kernel)
   p_h_layer = ms%h_layer; p_hTr_t = ms%tracers(1)%hTr; p_hTr_s = ms%tracers(2)%hTr
   p_u = ms%u_face_x_layer; p_v = ms%v_face_y_layer; p_eta = bt_eta
   p_mass = 0.0_wp; p_heat = 0.0_wp; p_salt = 0.0_wp

   write (output_unit, '(a)') repeat('=', 70)
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  domain: ', nxp, ' x ', nyp, ' x ', nz, &
      ' pts (+', NGHOST, ' ghosts/side)'
   write (output_unit, '(a,i0)') '  cells : ', nx*ny*nz
   write (output_unit, '(3a,i0,a,i0,a,i0,a)') '  DATA layer: ', DC_DATA_NAME, &
      '   (reps ', n_reps, ', warm ', n_warm, ', h-drift ', pert_pct, '%)'
   write (output_unit, '(a)') repeat('=', 70)

   ! ---- map the working set (no-ops when the DATA layer is 'host') ----------
   DC_ENTER_IN(grid)
   DC_ENTER_IN(vc)
   DC_ENTER_IN(ms)
   DC_ENTER_IN(vc%dsig)
   DC_ENTER_CREATE(vc%target_h)
   DC_ENTER_CREATE(vc%remap_h_old)
   DC_ENTER_CREATE(vc%remap_total_h)
   DC_ENTER_CREATE(vc%remap_h_ref)
   DC_ENTER_CREATE(ms%h_layer)
   DC_ENTER_CREATE(ms%mass_budget_remap)
   DC_ENTER_CREATE(ms%heat_budget_remap)
   DC_ENTER_CREATE(ms%salt_budget_remap)
   DC_ENTER_CREATE(ms%u_face_x_layer)
   DC_ENTER_CREATE(ms%v_face_y_layer)
   DC_ENTER_IN(ms%tracers)
   DC_ENTER_CREATE(ms%tracers(1)%hTr)
   DC_ENTER_CREATE(ms%tracers(2)%hTr)
   DC_ENTER_CREATE(bt_eta)
   DC_ENTER_IN(bt_H_ref)
   DC_ENTER_IN(p_h_layer)
   DC_ENTER_IN(p_hTr_t)
   DC_ENTER_IN(p_hTr_s)
   DC_ENTER_IN(p_u)
   DC_ENTER_IN(p_v)
   DC_ENTER_IN(p_eta)
   DC_ENTER_IN(p_mass)
   DC_ENTER_IN(p_heat)
   DC_ENTER_IN(p_salt)

   ! ======================= A: FAITHFUL ale_remap_step =======================
   do rep = 1, n_warm
      call restore()
      call ale_remap_step(grid, vc, ms, bt_eta, bt_H_ref)
   end do
   DC_WAIT

   best_a = huge(1.0_wp)
   do rep = 1, n_reps
      call restore()
      DC_WAIT
      t0 = wall()
      call ale_remap_step(grid, vc, ms, bt_eta, bt_H_ref)
      DC_WAIT
      t1 = wall()
      dt = t1 - t0
      if (dt < best_a) best_a = dt
   end do
   ms_a = best_a*1000.0_wp

   ! clean single remap -> snapshot every shared output on the host
   call restore()
   call ale_remap_step(grid, vc, ms, bt_eta, bt_H_ref)
   DC_WAIT
   call pull_all()
   a_h = ms%h_layer; a_hTr_t = ms%tracers(1)%hTr; a_hTr_s = ms%tracers(2)%hTr
   a_u = ms%u_face_x_layer; a_v = ms%v_face_y_layer; a_eta = bt_eta
   a_mass = ms%mass_budget_remap; a_heat = ms%heat_budget_remap; a_salt = ms%salt_budget_remap

   ! ======================= B: OPTIMIZED ale_remap_step_opt ==================
   do rep = 1, n_warm
      call restore()
      call ale_remap_step_opt(grid, vc, ms, bt_eta, bt_H_ref)
   end do
   DC_WAIT

   best_b = huge(1.0_wp)
   do rep = 1, n_reps
      call restore()
      DC_WAIT
      t0 = wall()
      call ale_remap_step_opt(grid, vc, ms, bt_eta, bt_H_ref)
      DC_WAIT
      t1 = wall()
      dt = t1 - t0
      if (dt < best_b) best_b = dt
   end do
   ms_b = best_b*1000.0_wp

   ! clean single remap -> pull B outputs for the cross-check
   call restore()
   call ale_remap_step_opt(grid, vc, ms, bt_eta, bt_H_ref)
   DC_WAIT
   call pull_all()

   ! ---- bit-identity: worst relative diff across every shared output --------
   rmax = 0.0_wp
   rmax = max(rmax, reldiff3(ms%h_layer, a_h, nx, ny, nz))
   rmax = max(rmax, reldiff3(ms%tracers(1)%hTr, a_hTr_t, nx, ny, nz))
   rmax = max(rmax, reldiff3(ms%tracers(2)%hTr, a_hTr_s, nx, ny, nz))
   rmax = max(rmax, reldiff3(ms%mass_budget_remap, a_mass, nx, ny, nz))
   rmax = max(rmax, reldiff3(ms%heat_budget_remap, a_heat, nx, ny, nz))
   rmax = max(rmax, reldiff3(ms%salt_budget_remap, a_salt, nx, ny, nz))
   rmax = max(rmax, reldiff3(ms%u_face_x_layer, a_u, nx + 1, ny, nz))
   rmax = max(rmax, reldiff3(ms%v_face_y_layer, a_v, nx, ny + 1, nz))
   rmax = max(rmax, reldiff2(bt_eta, a_eta, nx, ny))

   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  grid ', nxp, 'x', nyp, 'x', nz, ' (', nx*ny*nz, ' cells)'
   write (output_unit, '(3a)') '  data layer       : ', DC_DATA_NAME, ''
   write (output_unit, '(a,es12.5)') '  correctness      : max rel diff (all outputs) ', rmax
   if (rmax < 1.0e-12_wp) then
      write (output_unit, '(a)') '  verdict          : OK (optimized == faithful, <1e-12 rel)'
   else
      write (output_unit, '(a)') '  verdict          : *** DIFF -- optimized kernel bug ***'
   end if
   write (output_unit, '(a,es14.6,a,es14.6)') '  A h_layer min/max: ', minval(a_h), '  /  ', maxval(a_h)
   write (output_unit, '(a,es14.6,a,es14.6)') '  B h_layer min/max: ', minval(ms%h_layer), '  /  ', maxval(ms%h_layer)
   write (output_unit, '(a,f10.4,a)') '  faithful  DC     : ', ms_a, ' ms/rep'
   write (output_unit, '(a,f10.4,a,f7.3,a)') '  optimized DC     : ', ms_b, ' ms/rep   -> ', ms_a/ms_b, 'x'
   write (output_unit, '(a)') repeat('=', 70)

contains

   ! Restore working state from the pristine device copy (bare DC: device when
   ! mapped, host otherwise). Identical to dc_main.F90's restore().
   subroutine restore()
      integer :: i, j, k
      do concurrent(k=1:nz, j=1:ny, i=1:nx)
         ms%h_layer(i, j, k) = p_h_layer(i, j, k)
         ms%tracers(1)%hTr(i, j, k) = p_hTr_t(i, j, k)
         ms%tracers(2)%hTr(i, j, k) = p_hTr_s(i, j, k)
         ms%mass_budget_remap(i, j, k) = p_mass(i, j, k)
         ms%heat_budget_remap(i, j, k) = p_heat(i, j, k)
         ms%salt_budget_remap(i, j, k) = p_salt(i, j, k)
      end do
      do concurrent(k=1:nz, j=1:ny, i=1:nx + 1)
         ms%u_face_x_layer(i, j, k) = p_u(i, j, k)
      end do
      do concurrent(k=1:nz, j=1:ny + 1, i=1:nx)
         ms%v_face_y_layer(i, j, k) = p_v(i, j, k)
      end do
      do concurrent(j=1:ny, i=1:nx)
         bt_eta(i, j) = p_eta(i, j)
      end do
   end subroutine restore

   ! Bring every compared output back to the host (no-op under DATA=none).
   subroutine pull_all()
      DC_UPDATE_SELF(ms%h_layer)
      DC_UPDATE_SELF(ms%tracers(1)%hTr)
      DC_UPDATE_SELF(ms%tracers(2)%hTr)
      DC_UPDATE_SELF(ms%mass_budget_remap)
      DC_UPDATE_SELF(ms%heat_budget_remap)
      DC_UPDATE_SELF(ms%salt_budget_remap)
      DC_UPDATE_SELF(ms%u_face_x_layer)
      DC_UPDATE_SELF(ms%v_face_y_layer)
      DC_UPDATE_SELF(bt_eta)
   end subroutine pull_all

   real(wp) function reldiff3(x, y, l, m, n) result(rmax)
      integer, intent(in) :: l, m, n
      real(wp), intent(in) :: x(l, m, n), y(l, m, n)
      real(wp) :: sc
      integer :: i, j, k
      rmax = 0.0_wp
      do k = 1, n
         do j = 1, m
            do i = 1, l
               sc = max(abs(x(i, j, k)), abs(y(i, j, k)))
               if (sc > 1.0e-30_wp) rmax = max(rmax, abs(x(i, j, k) - y(i, j, k))/sc)
            end do
         end do
      end do
   end function reldiff3

   real(wp) function reldiff2(x, y, l, m) result(rmax)
      integer, intent(in) :: l, m
      real(wp), intent(in) :: x(l, m), y(l, m)
      real(wp) :: sc
      integer :: i, j
      rmax = 0.0_wp
      do j = 1, m
         do i = 1, l
            sc = max(abs(x(i, j)), abs(y(i, j)))
            if (sc > 1.0e-30_wp) rmax = max(rmax, abs(x(i, j) - y(i, j))/sc)
         end do
      end do
   end function reldiff2

   integer function iarg(k, dflt)
      integer, intent(in) :: k, dflt
      character(len=32) :: buf
      integer :: ln, st
      iarg = dflt
      if (command_argument_count() < k) return
      call get_command_argument(k, buf, ln, st)
      if (st /= 0 .or. ln == 0) return
      read (buf, *, iostat=st) iarg
      if (st /= 0) iarg = dflt
   end function iarg

   function wall() result(t)
      real(wp) :: t
      integer(int64) :: cnt, rate
      call system_clock(cnt, rate)
      t = real(cnt, wp)/real(rate, wp)
   end function wall

end program dc_ab
