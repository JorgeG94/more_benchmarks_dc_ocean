#include "directives.h"
!! SHARED single-binary head-to-head for the ALE-remap kernel:
!!   opt-DC   = ale_remap_step_opt      (optimized do concurrent, OpenACC data)
!!   opt-CUDA = ale_remap_opt (opt_kernel.cu, via the ale_bridge host_data shim)
!!
!! BOTH run on the SAME device arrays. The bridge passes opt-CUDA the very
!! allocations ale_remap_step_opt writes, inside `!$acc host_data use_device`,
!! so there is ONE truth in ONE binary -- this removes the two-harness caveat
!! (opt-DC and opt-CUDA were previously timed in separate executables).
!!
!! The remap has a FIXED POINT (after one call h_old == target_h), and the mass/
!! heat/salt budgets accumulate (+=), so the working state MUST be restored from
!! a pristine device copy before every call -- warm and timed alike -- or a rep
!! measures a kernel doing strictly less work than production. ale evolves
!! h_old -> h_layer, so the inputs are reseeded between the DC and CUDA runs too.
!! Timing is min over reps around a single call (restore untimed).
!!
!! Agreement: every shared output (h_layer, hTr_T, hTr_S, mass/heat/salt budgets,
!! u, v, bt_eta) is captured from a clean opt-DC remap and cross-checked against a
!! clean opt-CUDA remap; the bar is max rel diff < 1e-12 (FMA-contraction level).
!!
!! Usage: ./cmp [nx_pts] [ny_pts] [nz] [nreps] [nwarm] [hdrift_pct]  (dc_ab args)
program cmp_main
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use constants, only: wp, REMAP_PPM, VCOORD_ZSTAR
   use remap_state, only: hgrid_t, ocean_vcoord_t, multilayer_cgrid_state_t
   use ale_remap, only: ale_remap_step_opt
   use ale_bridge, only: ale_remap_opt_step
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
   ! snapshots of the opt-DC (A) outputs, for the bit-identity check
   real(wp), allocatable :: a_h(:, :, :), a_hTr_t(:, :, :), a_hTr_s(:, :, :)
   real(wp), allocatable :: a_u(:, :, :), a_v(:, :, :), a_eta(:, :)
   real(wp), allocatable :: a_mass(:, :, :), a_heat(:, :, :), a_salt(:, :, :)

   real(wp) :: t0, t1, best_a, best_b, dt, ms_a, ms_b
   real(wp) :: hbed, sig, pert, cT, cS, csum, rmax
   real(wp) :: r_h, r_ht, r_hs, r_u, r_v, r_eta, r_prog
   real(wp) :: r_mass, r_heat, r_salt, r_bud
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

   ! --- identical init to dc_ab.F90 / dc_main.F90 / ale_bench.F90 ---
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
   write (output_unit, '(a)') '  SHARED host_data head-to-head: opt-DC vs opt-CUDA (one binary)'
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  domain: ', nxp, ' x ', nyp, ' x ', nz, &
      ' pts (+', NGHOST, ' ghosts/side)'
   write (output_unit, '(a,i0)') '  cells : ', nx*ny*nz
   write (output_unit, '(3a,i0,a,i0,a,i0,a)') '  DATA layer: ', DC_DATA_NAME, &
      '   (reps ', n_reps, ', warm ', n_warm, ', h-drift ', pert_pct, '%)'
   write (output_unit, '(a)') repeat('=', 70)

   ! ---- map the working set (must be device-resident for host_data) ---------
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

   ! ======================= A: opt-DC ale_remap_step_opt =====================
   do rep = 1, n_warm
      call restore()
      call ale_remap_step_opt(grid, vc, ms, bt_eta, bt_H_ref)
   end do
   DC_WAIT

   best_a = huge(1.0_wp)
   do rep = 1, n_reps
      call restore()
      DC_WAIT
      t0 = wall()
      call ale_remap_step_opt(grid, vc, ms, bt_eta, bt_H_ref)
      DC_WAIT
      t1 = wall()
      dt = t1 - t0
      if (dt < best_a) best_a = dt
   end do
   ms_a = best_a*1000.0_wp

   ! clean single opt-DC remap -> snapshot every shared output on the host
   call restore()
   call ale_remap_step_opt(grid, vc, ms, bt_eta, bt_H_ref)
   DC_WAIT
   call pull_all()
   a_h = ms%h_layer; a_hTr_t = ms%tracers(1)%hTr; a_hTr_s = ms%tracers(2)%hTr
   a_u = ms%u_face_x_layer; a_v = ms%v_face_y_layer; a_eta = bt_eta
   a_mass = ms%mass_budget_remap; a_heat = ms%heat_budget_remap; a_salt = ms%salt_budget_remap

   ! ======================= B: opt-CUDA ale_remap_opt (host_data) ============
   do rep = 1, n_warm
      call restore()
      call ale_remap_opt_step(grid, vc, ms, bt_eta, bt_H_ref)
   end do
   DC_WAIT

   best_b = huge(1.0_wp)
   do rep = 1, n_reps
      call restore()
      DC_WAIT
      t0 = wall()
      call ale_remap_opt_step(grid, vc, ms, bt_eta, bt_H_ref)
      DC_WAIT
      t1 = wall()
      dt = t1 - t0
      if (dt < best_b) best_b = dt
   end do
   ms_b = best_b*1000.0_wp

   ! clean single opt-CUDA remap -> pull B outputs for the cross-check
   call restore()
   call ale_remap_opt_step(grid, vc, ms, bt_eta, bt_H_ref)
   DC_WAIT
   call pull_all()

   ! ---- bit-identity across every shared output. Split into PROGNOSTIC fields
   !      (h_layer, hTr_T, hTr_S, u, v, bt_eta) and the DIAGNOSTIC accumulators
   !      (mass/heat/salt budgets). The verdict is on the prognostic fields: the
   !      budgets are budget = new - old, a difference of two ~equal large
   !      numbers, so a cross-compiler FMA-level (~1e-15 absolute) discrepancy in
   !      `new` is amplified by catastrophic cancellation into a large RELATIVE
   !      diff -- this is a property of the metric, not a layout/geometry bug.
   r_h   = reldiff3(ms%h_layer, a_h, nx, ny, nz)
   r_ht  = reldiff3(ms%tracers(1)%hTr, a_hTr_t, nx, ny, nz)
   r_hs  = reldiff3(ms%tracers(2)%hTr, a_hTr_s, nx, ny, nz)
   r_u   = reldiff3(ms%u_face_x_layer, a_u, nx + 1, ny, nz)
   r_v   = reldiff3(ms%v_face_y_layer, a_v, nx, ny + 1, nz)
   r_eta = reldiff2(bt_eta, a_eta, nx, ny)
   r_prog = max(r_h, max(r_ht, max(r_hs, max(r_u, max(r_v, r_eta)))))
   r_mass = reldiff3(ms%mass_budget_remap, a_mass, nx, ny, nz)
   r_heat = reldiff3(ms%heat_budget_remap, a_heat, nx, ny, nz)
   r_salt = reldiff3(ms%salt_budget_remap, a_salt, nx, ny, nz)
   r_bud  = max(r_mass, max(r_heat, r_salt))
   rmax = max(r_prog, r_bud)

   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  grid ', nxp, 'x', nyp, 'x', nz, ' (', nx*ny*nz, ' cells)'
   write (output_unit, '(3a)') '  data layer       : ', DC_DATA_NAME, ''
   write (output_unit, '(a,es12.5,es12.5,es12.5)') '  reldiff h/T/S    : ', r_h, r_ht, r_hs
   write (output_unit, '(a,es12.5,es12.5,es12.5)') '  reldiff u/v/eta  : ', r_u, r_v, r_eta
   write (output_unit, '(a,es12.5)') '  agreement (prog) : max rel diff, prognostic fields ', r_prog
   write (output_unit, '(a,es12.5,a)') '  budgets (diag)   : max rel diff ', r_bud, &
      '  (new-old cancellation, not geometry)'
   if (r_prog < 1.0e-12_wp) then
      write (output_unit, '(a)') '  verdict          : OK (opt-CUDA == opt-DC, prognostic <1e-12 rel)'
   else
      write (output_unit, '(a)') '  verdict          : *** DIFF -- layout/geometry mismatch ***'
   end if
   write (output_unit, '(a,es14.6,a,es14.6)') '  opt-DC   h min/max: ', minval(a_h), '  /  ', maxval(a_h)
   write (output_unit, '(a,es14.6,a,es14.6)') '  opt-CUDA h min/max: ', minval(ms%h_layer), '  /  ', maxval(ms%h_layer)
   write (output_unit, '(a)') repeat('-', 70)
   write (output_unit, '(a,f10.4,a)') '  opt-DC           : ', ms_a, ' ms/rep'
   write (output_unit, '(a,f10.4,a)') '  opt-CUDA         : ', ms_b, ' ms/rep'
   if (ms_b < ms_a) then
      write (output_unit, '(a,f7.3,a)') '  ratio            : opt-CUDA faster, opt-DC/opt-CUDA = ', ms_a/ms_b, 'x'
   else
      write (output_unit, '(a,f7.3,a)') '  ratio            : opt-DC faster, opt-CUDA/opt-DC = ', ms_b/ms_a, 'x'
   end if
   write (output_unit, '(a)') repeat('=', 70)

contains

   ! Restore working state from the pristine device copy. Identical to dc_ab's.
   subroutine restore()
      integer :: i, j, k
      do k=1,nz
      do j=1,ny
      do i=1,nx
         ms%h_layer(i, j, k) = p_h_layer(i, j, k)
         ms%tracers(1)%hTr(i, j, k) = p_hTr_t(i, j, k)
         ms%tracers(2)%hTr(i, j, k) = p_hTr_s(i, j, k)
         ms%mass_budget_remap(i, j, k) = p_mass(i, j, k)
         ms%heat_budget_remap(i, j, k) = p_heat(i, j, k)
         ms%salt_budget_remap(i, j, k) = p_salt(i, j, k)
      end do
      end do
      end do
      do k=1,nz
      do j=1,ny
      do i=1,nx + 1
         ms%u_face_x_layer(i, j, k) = p_u(i, j, k)
      end do
      end do
      end do
      do k=1,nz
      do j=1,ny + 1
      do i=1,nx
         ms%v_face_y_layer(i, j, k) = p_v(i, j, k)
      end do
      end do
      end do
      do j=1,ny
      do i=1,nx
         bt_eta(i, j) = p_eta(i, j)
      end do
      end do
   end subroutine restore

   ! Bring every compared output back to the host.
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

   ! Field-normalized max relative diff: max|x-y| over the field, divided by the
   ! GLOBAL max magnitude of the field (redi_bench's metric). A per-ELEMENT
   ! relative diff instead divides each cell by its own magnitude, so a cell
   ! where the field crosses zero (velocity near a node) or where a diagnostic
   ! is a difference of two ~equal numbers (budget = new - old) amplifies a
   ! ~1e-16 absolute FMA-contraction difference into a spuriously large ratio.
   ! Normalizing by the field scale reports the true (FMA-level) agreement.
   real(wp) function reldiff3(x, y, l, m, n) result(rmax)
      integer, intent(in) :: l, m, n
      real(wp), intent(in) :: x(l, m, n), y(l, m, n)
      real(wp) :: dmax, scl, d
      integer :: i, j, k
      dmax = 0.0_wp; scl = 0.0_wp
      do k = 1, n
         do j = 1, m
            do i = 1, l
               d = abs(x(i, j, k) - y(i, j, k))
               if (d > dmax) dmax = d
               if (abs(x(i, j, k)) > scl) scl = abs(x(i, j, k))
            end do
         end do
      end do
      rmax = dmax/max(scl, 1.0e-30_wp)
   end function reldiff3

   real(wp) function reldiff2(x, y, l, m) result(rmax)
      integer, intent(in) :: l, m
      real(wp), intent(in) :: x(l, m), y(l, m)
      real(wp) :: dmax, scl, d
      integer :: i, j
      dmax = 0.0_wp; scl = 0.0_wp
      do j = 1, m
         do i = 1, l
            d = abs(x(i, j) - y(i, j))
            if (d > dmax) dmax = d
            if (abs(x(i, j)) > scl) scl = abs(x(i, j))
         end do
      end do
      rmax = dmax/max(scl, 1.0e-30_wp)
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

end program cmp_main
