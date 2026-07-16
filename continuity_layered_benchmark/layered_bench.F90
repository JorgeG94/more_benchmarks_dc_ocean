!! Driver for the ocean model's PRODUCTION LAYERED continuity-PPM kernel:
!! `do concurrent` vs a faithful CUDA C port.
!!
!! WHY THIS EXISTS: the barotropic benchmark next door
!! (../continuity_ppm_benchmark) measured the WRONG KERNEL for the regime that
!! matters. The production profiler splits continuity two ways:
!!   ocean_continuity        11.1% (EPBL on) / ~20% (EPBL off)  <- THIS kernel
!!   ocean_barotropic_solver  8.6% (EPBL on) / ~16% (EPBL off)  <- the other one
!! `ocean_continuity` wraps continuity_compute_fluxes -- the LAYERED routine --
!! at 2.16 ms/call in production against the barotropic kernel's 0.16 ms.
!! (Profile: <scratch>/<user>/tassie_runs/gabight_acc_120d.*.log, 0.1 deg,
!! 473x297x30, 34560 steps. With EPBL+kappa-shear OFF, vmix's 39% leaves the
!! budget and continuity becomes co-dominant with ale_remap -- that is the
!! regime this kernel is worth optimising for.)
!!
!! ⚠ NOTHING measured on the 2-D barotropic kernel predicts this one. It is 3-D
!! (11 loops over (nx,ny,nz), auto-collapsed to collapse(3)) with 2-D metrics.
!! Different iteration space, different register pressure, different launch
!! count. The "~1.09x at 4.2M cells" figure in the barotropic README is an
!! extrapolation from a 2-D sweep and must NOT be quoted for this kernel --
!! that is what this driver replaces with a measurement.
!!
!! STATE: all-wet (wet_T = 1), so ppm_mirror_h and the slope-flatten multiply by
!! 1 and the land branches never fire -- same choice as the other benchmarks.
!! use_ppm_limit_pos = .false. (production default) so ppm_limit_pos never runs.
!!
!! MEMORY: -gpu=mem:separate + MANUAL DEEP COPY (parent struct, then payload).
!! Both toolchains then read the SAME device allocation via host_data
!! use_device, so the only variable is who generated the code.
!!
!! Usage: ./layered_bench [nx_phys] [ny_phys] [nz] [nreps] [nwarm] [cuda_sync] [dump]
!!   ./layered_bench                  -> 473 297 30 (the 0.1 deg config), 200 reps
!!   ./layered_bench 945 594 30       -> the 0.05 deg hi-res config
!!   ./layered_bench 473 297 30 200 10 0  -> CUDA fully async
!!   ./layered_bench 473 297 30 200 10 2 1 -> also write fh_ref.bin for
!!                                            ./layered_native to verify against
!! nx_phys/ny_phys are INTERIOR cells; nx_total = nx_phys + 2*nghost. This
!! kernel is ghost-aware (it zeroes mass flux at nghost+1 / nghost+nx_phys+1),
!! unlike the barotropic one, so that relationship must actually hold.
program layered_bench
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use, intrinsic :: iso_c_binding, only: c_double, c_int
   use constants, only: wp
   use grid, only: hgrid_t
   use ocean_metrics, only: ocean_metrics_t
   use multilayer_cgrid_state, only: multilayer_cgrid_state_t
   use continuity_layered, only: continuity_t, continuity_compute_fluxes
   implicit none

   interface
      subroutine continuity_layered_cuda_launch(h, u, v, wet_T, dy_cu, dx_cv, iareaT, &
                                                hfl_x, hfr_x, hfl_y, hfr_y, mfx, mfy, fh, &
                                                nx, ny, nz, nghost, nx_phys, ny_phys, &
                                                do_pos, h_min_pos, sync) &
         bind(C, name="continuity_layered_cuda_launch")
         import :: c_double, c_int
         implicit none
         real(c_double), intent(in) :: h(*), u(*), v(*), wet_T(*)
         real(c_double), intent(in) :: dy_cu(*), dx_cv(*), iareaT(*)
         real(c_double), intent(inout) :: hfl_x(*), hfr_x(*), hfl_y(*), hfr_y(*)
         real(c_double), intent(inout) :: mfx(*), mfy(*), fh(*)
         integer(c_int), value :: nx, ny, nz, nghost, nx_phys, ny_phys, do_pos, sync
         real(c_double), value :: h_min_pos
      end subroutine
   end interface

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30
   integer, parameter :: REPS_DEF = 200, WARM_DEF = 10, NGHOST = 3
   real(wp), parameter :: DX = 10.0_wp, DY = 10.0_wp

   type(hgrid_t) :: grid
   type(ocean_metrics_t) :: metrics
   type(multilayer_cgrid_state_t) :: ms
   type(continuity_t) :: cont
   real(wp), allocatable :: hfl_x_cu(:, :, :), hfr_x_cu(:, :, :)
   real(wp), allocatable :: hfl_y_cu(:, :, :), hfr_y_cu(:, :, :)
   real(wp), allocatable :: mfx_cu(:, :, :), mfy_cu(:, :, :), fh_cu(:, :, :)
   real(wp) :: t0, t1, ms_dc, ms_cu, dmax, rmax, scale, gib
   real(wp) :: fh_min, fh_max, fh_sum
   integer :: i, j, k, rep, nx, ny, nz, nxp, nyp, n_reps, n_warm, cu_sync, nbad, do_pos_i
   integer :: dump_i, iu

   nxp = iarg(1, NXP_DEF)
   nyp = iarg(2, NYP_DEF)
   nz = iarg(3, NZ_DEF)
   n_reps = iarg(4, REPS_DEF)
   n_warm = iarg(5, WARM_DEF)
   cu_sync = iarg(6, 2)
   dump_i = iarg(7, 0)

   nx = nxp + 2*NGHOST
   ny = nyp + 2*NGHOST
   ! PPM runs i=3:nx-2 / j=3:ny-2 and the boundary loops touch nx-1, ny-1.
   if (nx < 6 .or. ny < 6 .or. nz < 1) then
      write (output_unit, '(a)') 'ERROR: need nx_phys,ny_phys >= 1 and nz >= 1'
      stop 1
   end if
   grid%nx_total = nx; grid%ny_total = ny
   grid%nx_phys = nxp; grid%ny_phys = nyp
   grid%nghost = NGHOST
   grid%dx = DX; grid%dy = DY

   ! ~17 3-D arrays of nx*ny*nz doubles. A domain that overruns the device
   ! fails inside `enter data` with a CUDA OOM that never mentions the size.
   gib = 17.0_wp*real(nx, wp)*real(ny, wp)*real(nz, wp)*8.0_wp/(1024.0_wp**3)

   allocate (ms%h_layer(nx, ny, nz))
   allocate (ms%u_face_x_layer(nx + 1, ny, nz), ms%v_face_y_layer(nx, ny + 1, nz))
   allocate (ms%mass_flux_x_layer(nx + 1, ny, nz), ms%mass_flux_y_layer(nx, ny + 1, nz))
   allocate (ms%flux_h_layer(nx, ny, nz))
   allocate (metrics%dy_cu(nx + 1, ny), metrics%dx_cv(nx, ny + 1))
   allocate (metrics%iareaT(nx, ny), metrics%wet_T(nx, ny))
   allocate (cont%h_face_left_x%data(nx + 1, ny, nz), cont%h_face_right_x%data(nx + 1, ny, nz))
   allocate (cont%h_face_left_y%data(nx, ny + 1, nz), cont%h_face_right_y%data(nx, ny + 1, nz))
   allocate (hfl_x_cu(nx + 1, ny, nz), hfr_x_cu(nx + 1, ny, nz))
   allocate (hfl_y_cu(nx, ny + 1, nz), hfr_y_cu(nx, ny + 1, nz))
   allocate (mfx_cu(nx + 1, ny, nz), mfy_cu(nx, ny + 1, nz), fh_cu(nx, ny, nz))
   ms%nz_ml = nz

   ! Gaussian bump + a k-dependent stratification, so the PPM limiter earns its
   ! keep: a flat h would collapse every limited slope into the same branch.
   do k = 1, nz
      do j = 1, ny
         do i = 1, nx
            ms%h_layer(i, j, k) = 20.0_wp + real(k, wp)*0.5_wp &
                                  + 5.0_wp*exp(-((real(i - nx/2, wp)/real(max(nx/8, 1), wp))**2 &
                                                 + (real(j - ny/2, wp)/real(max(ny/8, 1), wp))**2)) &
                                  + 0.4_wp*sin(0.01_wp*real(i, wp))*cos(0.013_wp*real(j, wp))
         end do
      end do
   end do
   do j = 1, ny
      do i = 1, nx
         metrics%wet_T(i, j) = 1.0_wp
         metrics%iareaT(i, j) = 1.0_wp/(DX*DY)
      end do
   end do
   ! Both signs of u/v so the upwind branch in the transport loops goes both ways.
   do k = 1, nz
      do j = 1, ny
         do i = 1, nx + 1
            ms%u_face_x_layer(i, j, k) = 0.5_wp*sin(0.02_wp*real(i, wp))*cos(0.03_wp*real(k, wp))
         end do
      end do
   end do
   do k = 1, nz
      do j = 1, ny + 1
         do i = 1, nx
            ms%v_face_y_layer(i, j, k) = 0.3_wp*cos(0.017_wp*real(j, wp))
         end do
      end do
   end do
   do j = 1, ny
      do i = 1, nx + 1
         metrics%dy_cu(i, j) = DY
      end do
   end do
   do j = 1, ny + 1
      do i = 1, nx
         metrics%dx_cv(i, j) = DX
      end do
   end do
   ms%mass_flux_x_layer = 0.0_wp; ms%mass_flux_y_layer = 0.0_wp; ms%flux_h_layer = 0.0_wp
   cont%h_face_left_x%data = 0.0_wp; cont%h_face_right_x%data = 0.0_wp
   cont%h_face_left_y%data = 0.0_wp; cont%h_face_right_y%data = 0.0_wp
   cont%h_min = 1.0e-6_wp
   cont%use_ppm_limit_pos = .false.
   hfl_x_cu = 0.0_wp; hfr_x_cu = 0.0_wp; hfl_y_cu = 0.0_wp; hfr_y_cu = 0.0_wp
   mfx_cu = 0.0_wp; mfy_cu = 0.0_wp; fh_cu = 0.0_wp
   do_pos_i = merge(1, 0, cont%use_ppm_limit_pos)

   write (output_unit, '(a)') repeat('=', 70)
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  domain: ', nxp, ' x ', nyp, ' x ', nz, &
      ' interior (+', NGHOST, ' ghosts/side)'
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  arrays: ', nx, ' x ', ny, ' x ', nz, &
      ' = ', nx*ny*nz, ' cells'
   write (output_unit, '(a,i0,a,i0,a)') '  reps:   ', n_reps, ' timed, ', n_warm, ' warm-up'
   write (output_unit, '(a,f0.2,a,i0)') '  device memory ~', gib, ' GiB;  cuda_sync = ', cu_sync
   write (output_unit, '(a)') repeat('=', 70)

   !$acc enter data copyin(ms)
   !$acc enter data copyin(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer)
   !$acc enter data create(ms%mass_flux_x_layer, ms%mass_flux_y_layer, ms%flux_h_layer)
   !$acc enter data copyin(metrics)
   !$acc enter data copyin(metrics%dy_cu, metrics%dx_cv, metrics%iareaT, metrics%wet_T)
   !$acc enter data copyin(cont)
   !$acc enter data copyin(cont%h_face_left_x, cont%h_face_right_x)
   !$acc enter data copyin(cont%h_face_left_y, cont%h_face_right_y)
   !$acc enter data create(cont%h_face_left_x%data, cont%h_face_right_x%data)
   !$acc enter data create(cont%h_face_left_y%data, cont%h_face_right_y%data)
   !$acc enter data create(hfl_x_cu, hfr_x_cu, hfl_y_cu, hfr_y_cu)
   !$acc enter data create(mfx_cu, mfy_cu, fh_cu)

   ! ---- variant A: do concurrent (production, verbatim) ------------------
   do rep = 1, n_warm
      call continuity_compute_fluxes(grid, metrics, cont, ms)
   end do
   !$acc wait
   t0 = wall()
   do rep = 1, n_reps
      call continuity_compute_fluxes(grid, metrics, cont, ms)
   end do
   !$acc wait
   t1 = wall()
   ms_dc = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   ! ---- variant B: faithful CUDA C port ---------------------------------
   do rep = 1, n_warm
      !$acc host_data use_device(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, &
      !$acc                      metrics%wet_T, metrics%dy_cu, metrics%dx_cv, metrics%iareaT, &
      !$acc                      hfl_x_cu, hfr_x_cu, hfl_y_cu, hfr_y_cu, mfx_cu, mfy_cu, fh_cu)
      call continuity_layered_cuda_launch(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, &
                                          metrics%wet_T, metrics%dy_cu, metrics%dx_cv, metrics%iareaT, &
                                          hfl_x_cu, hfr_x_cu, hfl_y_cu, hfr_y_cu, &
                                          mfx_cu, mfy_cu, fh_cu, nx, ny, nz, NGHOST, nxp, nyp, &
                                          do_pos_i, cont%h_min, 1)
      !$acc end host_data
   end do
   t0 = wall()
   do rep = 1, n_reps
      !$acc host_data use_device(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, &
      !$acc                      metrics%wet_T, metrics%dy_cu, metrics%dx_cv, metrics%iareaT, &
      !$acc                      hfl_x_cu, hfr_x_cu, hfl_y_cu, hfr_y_cu, mfx_cu, mfy_cu, fh_cu)
      call continuity_layered_cuda_launch(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, &
                                          metrics%wet_T, metrics%dy_cu, metrics%dx_cv, metrics%iareaT, &
                                          hfl_x_cu, hfr_x_cu, hfl_y_cu, hfr_y_cu, &
                                          mfx_cu, mfy_cu, fh_cu, nx, ny, nz, NGHOST, nxp, nyp, &
                                          do_pos_i, cont%h_min, cu_sync)
      !$acc end host_data
   end do
   !$acc host_data use_device(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, &
   !$acc                      metrics%wet_T, metrics%dy_cu, metrics%dx_cv, metrics%iareaT, &
   !$acc                      hfl_x_cu, hfr_x_cu, hfl_y_cu, hfr_y_cu, mfx_cu, mfy_cu, fh_cu)
   call continuity_layered_cuda_launch(ms%h_layer, ms%u_face_x_layer, ms%v_face_y_layer, &
                                       metrics%wet_T, metrics%dy_cu, metrics%dx_cv, metrics%iareaT, &
                                       hfl_x_cu, hfr_x_cu, hfl_y_cu, hfr_y_cu, &
                                       mfx_cu, mfy_cu, fh_cu, nx, ny, nz, NGHOST, nxp, nyp, &
                                       do_pos_i, cont%h_min, 1)
   !$acc end host_data
   t1 = wall()
   ms_cu = (t1 - t0)*1000.0_wp/real(n_reps + 1, wp)

   !$acc update self(ms%flux_h_layer)
   !$acc update self(fh_cu)

   ! Reference dump for ./layered_native (the no-Fortran, no-OpenACC driver).
   ! Inputs go in as well as outputs: that lets the native driver prove its
   ! OWN init matches this one before it claims its fluxes match, so an init
   ! mismatch can never be mistaken for a kernel or an indexing bug.
   if (dump_i /= 0) then
      open (newunit=iu, file='fh_ref.bin', access='stream', form='unformatted', status='replace')
      write (iu) nx, ny, nz
      write (iu) ms%h_layer
      write (iu) ms%u_face_x_layer
      write (iu) ms%v_face_y_layer
      write (iu) ms%flux_h_layer
      write (iu) fh_cu
      close (iu)
      write (output_unit, '(a)') '  wrote fh_ref.bin (nx,ny,nz, h, u, v, fh_dc, fh_cu)'
   end if

   ! DC vs CUDA: not necessarily bit-exact -- FMA contraction can differ between
   ! nvfortran and nvcc. Beyond ~1e-12 relative is a PORT BUG, which would mean
   ! the timing compares two different algorithms.
   dmax = 0.0_wp; rmax = 0.0_wp; nbad = 0
   do k = 1, nz
      do j = 1, ny
         do i = 1, nx
            dmax = max(dmax, abs(ms%flux_h_layer(i, j, k) - fh_cu(i, j, k)))
            scale = max(abs(ms%flux_h_layer(i, j, k)), abs(fh_cu(i, j, k)))
            if (scale > 1.0e-30_wp) then
               rmax = max(rmax, abs(ms%flux_h_layer(i, j, k) - fh_cu(i, j, k))/scale)
               if (abs(ms%flux_h_layer(i, j, k) - fh_cu(i, j, k))/scale > 1.0e-12_wp) nbad = nbad + 1
            end if
         end do
      end do
   end do
   fh_min = minval(ms%flux_h_layer)
   fh_max = maxval(ms%flux_h_layer)
   fh_sum = sum(ms%flux_h_layer)

   write (output_unit, '(a,f10.4,a)') '  layered continuity (do concurrent)    : ', ms_dc, ' ms/rep'
   write (output_unit, '(a,f10.4,a)') '  CUDA C (faithful port, nvcc)          : ', ms_cu, ' ms/rep'
   write (output_unit, '(a,f10.3,a)') '  ratio  dc / cuda                      : ', ms_dc/ms_cu, ' x'
   write (output_unit, '(a)') ''
   write (output_unit, '(a,es12.5)') '  DC vs CUDA  max |diff|  : ', dmax
   write (output_unit, '(a,es12.5)') '              max rel diff: ', rmax
   if (rmax < 1.0e-12_wp) then
      write (output_unit, '(a)') '              agreement   : OK (<1e-12 rel -> FMA contraction only)'
   else
      write (output_unit, '(a,i0,a)') '              agreement   : *** SUSPECT *** ', nbad, &
         ' cells >1e-12 rel -- likely a PORT BUG'
   end if
   write (output_unit, '(a)') ''
   write (output_unit, '(a,es14.6)') '  min flux_h_layer       : ', fh_min
   write (output_unit, '(a,es14.6)') '  max flux_h_layer       : ', fh_max
   write (output_unit, '(a,es14.6)') '  sum flux_h_layer       : ', fh_sum
   if (fh_min /= fh_min .or. fh_max /= fh_max) then
      write (output_unit, '(a)') '  sanity                 : *** NaN -- results are garbage ***'
   else if (fh_min == 0.0_wp .and. fh_max == 0.0_wp) then
      write (output_unit, '(a)') '  sanity                 : *** all-zero -- kernel did nothing ***'
   else
      write (output_unit, '(a)') '  sanity                 : OK (finite, non-zero)'
   end if
   write (output_unit, '(a)') repeat('=', 70)

contains

   !! Positional integer argument `k`, or `dflt` if absent/unparseable.
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

end program layered_bench
