#include "directives.h"
!! DC-only driver for the production ALE-remap kernel (`ale_remap_step`).
!!
!! COMPUTE is the production `do concurrent` orchestrator (ale_remap_step, the
!! as-shipped variant with NO -DASYNC / -DPPM_DIRECT / -DCOLLAPSE_FIX / -DFLATSIG).
!! The device data layer is chosen ENTIRELY by directives.h at compile time:
!!   -DDC_DATA_ACC  -> OpenACC  (nvfortran -acc=gpu -stdpar=gpu)   GPU
!!   -DDC_DATA_OMP  -> OpenMP target                               GPU (AMD/Intel too)
!!   (neither)      -> host: bare DC on the CPU (-stdpar=multicore/serial)
!! There is NO CUDA and NO nvcc in this binary. It builds and runs the
!! do-concurrent remap on the CPU / AMD / Intel unchanged.
!!
!! Init is IDENTICAL to ale_bench.F90 (Gaussian bathymetry + zstar stratification
!! perturbed by a h-drift %, the ONE knob the cost is sensitive to). The state is
!! restored from a pristine device copy before every timed rep -- MANDATORY here:
!! the remap has a fixed point (after one call h_old == target_h), so un-restored
!! reps would measure a kernel doing strictly less work than production's.
!!
!! Cross-check (proves the macro'd data layer did not change the numbers):
!!   DC_DUMP=file  writes nx,ny,nz + h_layer  (a reference)
!!   DC_REF=file   reads that reference and reports max|diff| vs this run
!! Run once with DC_DATA_ACC (GPU) dumping a ref, then again on the CPU/OMP
!! reading it: agreement to FMA level means the OpenACC->host swap is inert.
!!
!! Usage: ./dc_main [nx_pts] [ny_pts] [nz] [nreps] [nwarm] [hdrift_pct]
program dc_main
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use constants, only: wp, REMAP_PPM, VCOORD_ZSTAR
   use remap_state, only: hgrid_t, ocean_vcoord_t, multilayer_cgrid_state_t
   use ale_remap, only: ale_remap_step
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

   real(wp) :: t0, t1, best, dt, ms_dc, gib
   real(wp) :: hbed, sig, pert, cT, cS, csum, hmn, hmx, hsm
   integer :: i, j, k, rep, nx, ny, nz, nxp, nyp, n_reps, n_warm, pert_pct, iu, ios
   character(len=256) :: ref_path, dump_path

   nxp = iarg(1, NXP_DEF); nyp = iarg(2, NYP_DEF); nz = iarg(3, NZ_DEF)
   n_reps = iarg(4, REPS_DEF); n_warm = iarg(5, WARM_DEF); pert_pct = iarg(6, PERT_DEF)
   nx = nxp + 2*NGHOST; ny = nyp + 2*NGHOST
   if (nx < 6 .or. ny < 6 .or. nz < 1) then
      write (output_unit, '(a)') 'ERROR: need nx_pts,ny_pts >= 1 and nz >= 1'; stop 1
   end if
   gib = 8.0_wp*real(nx, wp)*real(ny, wp)*real(nz, wp)*8.0_wp/(1024.0_wp**3)

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

   do k = 1, nz
      vc%dsig(k) = 1.0_wp + 3.0_wp*real(k - 1, wp)/real(max(1, nz - 1), wp)
   end do
   vc%dsig = vc%dsig/sum(vc%dsig)

   ! --- identical init to ale_bench.F90 (branch coverage over a mix of columns) ---
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
   ! parent structs first, then each payload (mapping inside a helper never
   ! attaches the components) -- mirrors ale_bench.F90's enter-data list.
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

   ! ---- do concurrent, production verbatim (restore OUTSIDE the timed span) --
   do rep = 1, n_warm
      call restore()
      call ale_remap_step(grid, vc, ms, bt_eta, bt_H_ref)
   end do
   DC_WAIT

   best = huge(1.0_wp)
   do rep = 1, n_reps
      call restore()
      DC_WAIT
      t0 = wall()
      call ale_remap_step(grid, vc, ms, bt_eta, bt_H_ref)
      DC_WAIT
      t1 = wall()
      dt = t1 - t0
      if (dt < best) best = dt
   end do
   ms_dc = best*1000.0_wp

   DC_UPDATE_SELF(ms%h_layer)

   hmn = minval(ms%h_layer); hmx = maxval(ms%h_layer); hsm = sum(ms%h_layer)

   write (output_unit, '(3a,f10.4,a)') '  do concurrent (', DC_DATA_NAME, ')  : ', ms_dc, ' ms/rep (min)'
   write (output_unit, '(a,es14.6)') '  min h_layer : ', hmn
   write (output_unit, '(a,es14.6)') '  max h_layer : ', hmx
   write (output_unit, '(a,es14.6)') '  sum h_layer : ', hsm
   if (hmn /= hmn .or. (hmn == 0.0_wp .and. hmx == 0.0_wp)) then
      write (output_unit, '(a)') '  sanity      : *** garbage (NaN or all-zero) ***'; stop 2
   else
      write (output_unit, '(a)') '  sanity      : OK (finite, non-zero)'
   end if

   ! ---- optional reference dump / cross-check ------------------------------
   call get_environment_variable('DC_DUMP', dump_path, status=ios)
   if (ios == 0 .and. len_trim(dump_path) > 0) then
      open (newunit=iu, file=trim(dump_path), access='stream', form='unformatted', status='replace')
      write (iu) nx, ny, nz
      write (iu) ms%h_layer
      close (iu)
      write (output_unit, '(3a)') '  wrote ref   : ', trim(dump_path), ' (nx,ny,nz, h_layer)'
   end if

   call get_environment_variable('DC_REF', ref_path, status=ios)
   if (ios == 0 .and. len_trim(ref_path) > 0) call compare_ref(trim(ref_path))

   write (output_unit, '(a)') repeat('=', 70)

contains

   ! Restore the working state from the pristine device copy. Bare `do
   ! concurrent` => runs on the device when the DATA layer maps it, on the CPU
   ! when it does not. Identical to ale_bench.F90's restore().
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

   subroutine compare_ref(path)
      character(len=*), intent(in) :: path
      real(wp), allocatable :: ref(:, :, :)
      real(wp) :: dmax, rmax, sc
      integer :: rnx, rny, rnz, u, st, nbad
      open (newunit=u, file=path, access='stream', form='unformatted', status='old', iostat=st)
      if (st /= 0) then
         write (output_unit, '(3a)') '  cross-check : ref ', path, ' not found -- skipped'; return
      end if
      read (u) rnx, rny, rnz
      if (rnx /= nx .or. rny /= ny .or. rnz /= nz) then
         write (output_unit, '(a)') '  cross-check : ref has a different shape -- skipped'
         close (u); return
      end if
      allocate (ref(rnx, rny, rnz)); read (u) ref; close (u)
      dmax = 0.0_wp; rmax = 0.0_wp; nbad = 0
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               dmax = max(dmax, abs(ms%h_layer(i, j, k) - ref(i, j, k)))
               sc = max(abs(ms%h_layer(i, j, k)), abs(ref(i, j, k)))
               if (sc > 1.0e-30_wp) then
                  rmax = max(rmax, abs(ms%h_layer(i, j, k) - ref(i, j, k))/sc)
                  if (abs(ms%h_layer(i, j, k) - ref(i, j, k))/sc > 1.0e-12_wp) nbad = nbad + 1
               end if
            end do
         end do
      end do
      write (output_unit, '(a,es12.5,a,es12.5)') '  cross-check vs ref: max|diff| ', dmax, '  max rel ', rmax
      if (rmax < 1.0e-12_wp) then
         write (output_unit, '(a)') '  cross-check : OK (<1e-12 rel -> data layer is numerically inert)'
      else
         write (output_unit, '(a,i0,a)') '  cross-check : *** ', nbad, ' cells >1e-12 rel -- INVESTIGATE ***'
      end if
   end subroutine compare_ref

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

end program dc_main
