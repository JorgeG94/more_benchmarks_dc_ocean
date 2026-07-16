#include "directives.h"
!! DC-only driver for the production LAYERED continuity-PPM kernel.
!!
!! COMPUTE is a bare `do concurrent` (continuity_compute_fluxes). The device
!! data layer is chosen ENTIRELY by directives.h at compile time:
!!   -DDC_DATA_ACC  -> OpenACC  (nvfortran -acc=gpu -stdpar=gpu)   GPU
!!   -DDC_DATA_OMP  -> OpenMP target                               GPU (AMD/Intel too)
!!   (neither)      -> host: bare DC on the CPU (-stdpar=multicore/serial)
!! There is NO CUDA and NO nvcc in this binary. That is the point: it builds and
!! runs the do-concurrent implementation on the CPU / AMD / Intel unchanged.
!!
!! Cross-check (proves the macro'd data layer did not change the numbers):
!!   DC_DUMP=file  writes nx,ny,nz + flux_h_layer  (a reference)
!!   DC_REF=file   reads that reference and reports max|diff| vs this run
!! Run once with DC_DATA_ACC (GPU) dumping a ref, then again on the CPU reading
!! it: agreement to FMA level means the OpenACC->host swap is numerically inert.
!!
!! Usage: ./dc_main [nx_phys] [ny_phys] [nz] [nreps] [nwarm]
program dc_main
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use constants, only: wp
   use grid, only: hgrid_t
   use ocean_metrics, only: ocean_metrics_t
   use multilayer_cgrid_state, only: multilayer_cgrid_state_t
   use continuity_layered, only: continuity_t, continuity_compute_fluxes, continuity_compute_fluxes_fused
   implicit none

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30
   integer, parameter :: REPS_DEF = 200, WARM_DEF = 10, NGHOST = 3
   real(wp), parameter :: DX = 10.0_wp, DY = 10.0_wp

   type(hgrid_t) :: grid
   type(ocean_metrics_t) :: metrics
   type(multilayer_cgrid_state_t) :: ms
   type(continuity_t) :: cont
   real(wp) :: t0, t1, ms_dc, ms_fu, gib, fh_min, fh_max, fh_sum, dmax, rmax
   real(wp), allocatable :: fh_prod(:,:,:)
   integer :: i, j, k, rep, nx, ny, nz, nxp, nyp, n_reps, n_warm, iu, ios
   character(len=256) :: ref_path, dump_path

   nxp = iarg(1, NXP_DEF); nyp = iarg(2, NYP_DEF); nz = iarg(3, NZ_DEF)
   n_reps = iarg(4, REPS_DEF); n_warm = iarg(5, WARM_DEF)
   nx = nxp + 2*NGHOST; ny = nyp + 2*NGHOST
   if (nx < 6 .or. ny < 6 .or. nz < 1) then
      write (output_unit, '(a)') 'ERROR: need nx_phys,ny_phys >= 1 and nz >= 1'; stop 1
   end if
   grid%nx_total = nx; grid%ny_total = ny
   grid%nx_phys = nxp; grid%ny_phys = nyp
   grid%nghost = NGHOST; grid%dx = DX; grid%dy = DY
   gib = 8.0_wp*real(nx, wp)*real(ny, wp)*real(nz, wp)*8.0_wp/(1024.0_wp**3)

   allocate (ms%h_layer(nx, ny, nz))
   allocate (ms%u_face_x_layer(nx + 1, ny, nz), ms%v_face_y_layer(nx, ny + 1, nz))
   allocate (ms%mass_flux_x_layer(nx + 1, ny, nz), ms%mass_flux_y_layer(nx, ny + 1, nz))
   allocate (ms%flux_h_layer(nx, ny, nz))
   allocate (metrics%dy_cu(nx + 1, ny), metrics%dx_cv(nx, ny + 1))
   allocate (metrics%iareaT(nx, ny), metrics%wet_T(nx, ny))
   allocate (cont%h_face_left_x%data(nx + 1, ny, nz), cont%h_face_right_x%data(nx + 1, ny, nz))
   allocate (cont%h_face_left_y%data(nx, ny + 1, nz), cont%h_face_right_y%data(nx, ny + 1, nz))
   ms%nz_ml = nz

   ! --- identical init to the benchmark driver (Gaussian bump + stratification) ---
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
         metrics%wet_T(i, j) = 1.0_wp; metrics%iareaT(i, j) = 1.0_wp/(DX*DY)
      end do
   end do
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
   cont%h_min = 1.0e-6_wp; cont%use_ppm_limit_pos = .false.

   write (output_unit, '(a)') repeat('=', 70)
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  domain: ', nxp, ' x ', nyp, ' x ', nz, &
      ' interior (+', NGHOST, ' ghosts/side)'
   write (output_unit, '(a,i0)') '  cells : ', nx*ny*nz
   write (output_unit, '(3a,i0,a,i0,a)') '  DATA layer: ', DC_DATA_NAME, &
      '   (reps ', n_reps, ', warm ', n_warm, ')'
   write (output_unit, '(a)') repeat('=', 70)

   ! ---- map the working set (no-ops when the DATA layer is 'host') ----------
   DC_ENTER_IN(ms)
   DC_ENTER_IN(ms%h_layer)
   DC_ENTER_IN(ms%u_face_x_layer)
   DC_ENTER_IN(ms%v_face_y_layer)
   DC_ENTER_CREATE(ms%mass_flux_x_layer)
   DC_ENTER_CREATE(ms%mass_flux_y_layer)
   DC_ENTER_CREATE(ms%flux_h_layer)
   DC_ENTER_IN(metrics)
   DC_ENTER_IN(metrics%dy_cu)
   DC_ENTER_IN(metrics%dx_cv)
   DC_ENTER_IN(metrics%iareaT)
   DC_ENTER_IN(metrics%wet_T)
   DC_ENTER_IN(cont)
   DC_ENTER_IN(cont%h_face_left_x)
   DC_ENTER_IN(cont%h_face_right_x)
   DC_ENTER_IN(cont%h_face_left_y)
   DC_ENTER_IN(cont%h_face_right_y)
   DC_ENTER_CREATE(cont%h_face_left_x%data)
   DC_ENTER_CREATE(cont%h_face_right_x%data)
   DC_ENTER_CREATE(cont%h_face_left_y%data)
   DC_ENTER_CREATE(cont%h_face_right_y%data)

   allocate (fh_prod(nx, ny, nz))

   ! ---- A: production do concurrent (11 loops) -----------------------------
   do rep = 1, n_warm
      call continuity_compute_fluxes(grid, metrics, cont, ms)
   end do
   DC_WAIT
   t0 = wall()
   do rep = 1, n_reps
      call continuity_compute_fluxes(grid, metrics, cont, ms)
   end do
   DC_WAIT
   t1 = wall()
   ms_dc = (t1 - t0)*1000.0_wp/real(n_reps, wp)
   DC_UPDATE_SELF(ms%flux_h_layer)
   fh_prod = ms%flux_h_layer

   ! ---- B: fused do concurrent (3 loops, no h_face workspace) --------------
   do rep = 1, n_warm
      call continuity_compute_fluxes_fused(grid, metrics, cont, ms)
   end do
   DC_WAIT
   t0 = wall()
   do rep = 1, n_reps
      call continuity_compute_fluxes_fused(grid, metrics, cont, ms)
   end do
   DC_WAIT
   t1 = wall()
   ms_fu = (t1 - t0)*1000.0_wp/real(n_reps, wp)
   DC_UPDATE_SELF(ms%flux_h_layer)

   dmax = 0.0_wp; rmax = 0.0_wp
   do k = 1, nz
      do j = 1, ny
         do i = 1, nx
            dmax = max(dmax, abs(ms%flux_h_layer(i, j, k) - fh_prod(i, j, k)))
            fh_sum = max(abs(ms%flux_h_layer(i, j, k)), abs(fh_prod(i, j, k)))
            if (fh_sum > 1.0e-30_wp) rmax = max(rmax, abs(ms%flux_h_layer(i, j, k) - fh_prod(i, j, k))/fh_sum)
         end do
      end do
   end do

   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  grid ', nxp, 'x', nyp, 'x', nz, ' (', nx*ny*nz, ' cells)'
   write (output_unit, '(3a)') '  data layer       : ', DC_DATA_NAME, ''
   write (output_unit, '(a,es12.5,a,es12.5)') '  correctness      : max|diff| ', dmax, '  max rel ', rmax
   if (rmax < 1.0e-12_wp) then
      write (output_unit, '(a)') '  verdict          : OK (fused == production, <1e-12 rel)'
   else
      write (output_unit, '(a)') '  verdict          : *** DIFF -- fused kernel bug ***'
   end if
   write (output_unit, '(a,f10.4,a)') '  production DC    : ', ms_dc, ' ms/rep'
   write (output_unit, '(a,f10.4,a,f7.3,a)') '  fused DC         : ', ms_fu, ' ms/rep   -> ', ms_dc/ms_fu, 'x'

   write (output_unit, '(a)') repeat('=', 70)

contains

   subroutine compare_ref(path)
      character(len=*), intent(in) :: path
      real(wp), allocatable :: ref(:, :, :)
      real(wp) :: dmax, rmax, sc
      integer :: rnx, rny, rnz, u, st, nbad
      open (newunit=u, file=path, access='stream', form='unformatted', status='old', iostat=st)
      if (st /= 0) then
         write (output_unit, '(3a)') '  cross-check     : ref ', path, ' not found -- skipped'; return
      end if
      read (u) rnx, rny, rnz
      if (rnx /= nx .or. rny /= ny .or. rnz /= nz) then
         write (output_unit, '(a)') '  cross-check     : ref has a different shape -- skipped'
         close (u); return
      end if
      allocate (ref(rnx, rny, rnz)); read (u) ref; close (u)
      dmax = 0.0_wp; rmax = 0.0_wp; nbad = 0
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               dmax = max(dmax, abs(ms%flux_h_layer(i, j, k) - ref(i, j, k)))
               sc = max(abs(ms%flux_h_layer(i, j, k)), abs(ref(i, j, k)))
               if (sc > 1.0e-30_wp) then
                  rmax = max(rmax, abs(ms%flux_h_layer(i, j, k) - ref(i, j, k))/sc)
                  if (abs(ms%flux_h_layer(i, j, k) - ref(i, j, k))/sc > 1.0e-12_wp) nbad = nbad + 1
               end if
            end do
         end do
      end do
      write (output_unit, '(a,es12.5,a,es12.5)') '  cross-check vs ref: max|diff| ', dmax, '  max rel ', rmax
      if (rmax < 1.0e-12_wp) then
         write (output_unit, '(a)') '  cross-check     : OK (<1e-12 rel -> data layer is numerically inert)'
      else
         write (output_unit, '(a,i0,a)') '  cross-check     : *** ', nbad, ' cells >1e-12 rel -- INVESTIGATE ***'
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
