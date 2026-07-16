#include "directives.h"
!! DC-only driver for the production HLL flux kernel (kernel_flux.F90).
!!
!! COMPUTE is a bare `do concurrent` (compute_flux_hll, the shipped kernel). The
!! device data layer is chosen ENTIRELY by directives.h at compile time:
!!   -DDC_DATA_ACC  -> OpenACC  (nvfortran -acc=gpu -stdpar=gpu)   GPU
!!   -DDC_DATA_OMP  -> OpenMP target                               GPU (AMD/Intel too)
!!   (neither)      -> host: bare DC on the CPU (-stdpar=multicore/serial)
!! There is NO CUDA and NO nvcc in this binary. That is the point: it builds and
!! runs the do-concurrent implementation on the CPU / AMD / Intel unchanged.
!!
!! STATE: mirrors hll_fluxes_benchmark/flux_bench.F90's init exactly -- a 2-D
!! radial dam break on a wet bed (10 m inside, 2 m outside, both far above
!! THIN_LAYER_THRESHOLD so the dry branches never fire), flat bed, hu=0.05*i/nx,
!! hv=-0.03*j/ny. This is a 2-D kernel: nghost=2 (the stencil reads i-2..i+2).
!!
!! Cross-check (proves the macro'd data layer did not change the numbers):
!!   DC_DUMP=file  writes nx,ny + flux_h  (a reference)
!!   DC_REF=file   reads that reference and reports max|diff| vs this run
!! Run once with DC_DATA_ACC (GPU) dumping a ref, then again on the CPU reading
!! it: agreement to FMA level means the OpenACC->host swap is numerically inert.
!!
!! Usage: ./dc_main [nx_phys] [ny_phys] [nreps] [nwarm]
program dc_main
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use constants, only: wp
   use grid, only: hgrid_t
   use kernel_flux, only: compute_flux_hll
   implicit none

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297
   integer, parameter :: REPS_DEF = 200, WARM_DEF = 10, NGHOST = 2
   real(wp), parameter :: DX = 10.0_wp, DY = 10.0_wp

   type(hgrid_t) :: grid
   real(wp), allocatable :: h(:, :), hu(:, :), hv(:, :), b(:, :)
   real(wp), allocatable :: flux_h(:, :), flux_hu(:, :), flux_hv(:, :)
   real(wp), allocatable :: mass_flux_x(:, :), mass_flux_y(:, :)
   real(wp) :: t0, t1, ms_dc, fh_min, fh_max, fh_sum
   integer :: i, j, rep, nx, ny, nxp, nyp, n_reps, n_warm, iu, ios
   character(len=256) :: ref_path, dump_path

   nxp = iarg(1, NXP_DEF); nyp = iarg(2, NYP_DEF)
   n_reps = iarg(3, REPS_DEF); n_warm = iarg(4, WARM_DEF)
   nx = nxp + 2*NGHOST; ny = nyp + 2*NGHOST
   if (nx < 6 .or. ny < 6) then
      write (output_unit, '(a)') 'ERROR: need nx_phys,ny_phys >= 2'; stop 1
   end if
   grid%nghost = NGHOST; grid%nx_total = nx; grid%ny_total = ny
   grid%dx = DX; grid%dy = DY

   allocate (h(nx, ny), hu(nx, ny), hv(nx, ny), b(nx, ny))
   allocate (flux_h(nx, ny), flux_hu(nx, ny), flux_hv(nx, ny))
   allocate (mass_flux_x(nx, ny), mass_flux_y(nx, ny))

   ! --- identical init to flux_bench.F90: 2-D radial dam break, wet bed --------
   ! nx/2 and nx/4 are INTEGER division, evaluated before the real() conversion;
   ! doing the divide in double would move the dam radius. Arithmetic order of
   ! hu/hv matches Fortran's left-to-right (0.05*i)/nx.
   do j = 1, ny
      do i = 1, nx
         b(i, j) = 0.0_wp
         if ((real(i - nx/2, wp)**2 + real(j - ny/2, wp)**2) < real(nx/4, wp)**2) then
            h(i, j) = 10.0_wp
         else
            h(i, j) = 2.0_wp
         end if
         hu(i, j) = 0.05_wp*real(i, wp)/real(nx, wp)
         hv(i, j) = -0.03_wp*real(j, wp)/real(ny, wp)
      end do
   end do
   flux_h = 0.0_wp; flux_hu = 0.0_wp; flux_hv = 0.0_wp
   mass_flux_x = 0.0_wp; mass_flux_y = 0.0_wp

   write (output_unit, '(a)') repeat('=', 70)
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  grid  : ', nx, ' x ', ny, ' total (', &
      nxp, ' x ', nyp, ' interior), nghost=2'
   write (output_unit, '(a,i0)') '  cells : ', nx*ny
   write (output_unit, '(3a,i0,a,i0,a)') '  DATA layer: ', DC_DATA_NAME, &
      '   (reps ', n_reps, ', warm ', n_warm, ')'
   write (output_unit, '(a)') '  state : 2-D dam break, wet bed (no dry cells -> no branch divergence)'
   write (output_unit, '(a)') repeat('=', 70)

   ! ---- map the working set (no-ops when the DATA layer is 'host') -----------
   DC_ENTER_IN(h)
   DC_ENTER_IN(hu)
   DC_ENTER_IN(hv)
   DC_ENTER_IN(b)
   DC_ENTER_CREATE(flux_h)
   DC_ENTER_CREATE(flux_hu)
   DC_ENTER_CREATE(flux_hv)
   DC_ENTER_CREATE(mass_flux_x)
   DC_ENTER_CREATE(mass_flux_y)

   ! ---- do concurrent, production kernel verbatim ---------------------------
   do rep = 1, n_warm
      call compute_flux_hll(h, hu, hv, b, flux_h, flux_hu, flux_hv, &
                            mass_flux_x, mass_flux_y, grid)
   end do
   DC_WAIT
   t0 = wall()
   do rep = 1, n_reps
      call compute_flux_hll(h, hu, hv, b, flux_h, flux_hu, flux_hv, &
                            mass_flux_x, mass_flux_y, grid)
   end do
   DC_WAIT
   t1 = wall()
   ms_dc = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   DC_UPDATE_SELF(flux_h)

   fh_min = minval(flux_h)
   fh_max = maxval(flux_h)
   fh_sum = sum(flux_h)

   write (output_unit, '(3a,f10.4,a)') '  do concurrent (', DC_DATA_NAME, ')  : ', ms_dc, ' ms/rep'
   write (output_unit, '(a,es14.6)') '  min flux_h : ', fh_min
   write (output_unit, '(a,es14.6)') '  max flux_h : ', fh_max
   write (output_unit, '(a,es14.6)') '  sum flux_h : ', fh_sum
   if (fh_min /= fh_min .or. (fh_min == 0.0_wp .and. fh_max == 0.0_wp)) then
      write (output_unit, '(a)') '  sanity     : *** garbage (NaN or all-zero) ***'; stop 2
   else
      write (output_unit, '(a)') '  sanity     : OK (finite, non-zero)'
   end if

   ! ---- optional reference dump / cross-check ------------------------------
   call get_environment_variable('DC_DUMP', dump_path, status=ios)
   if (ios == 0 .and. len_trim(dump_path) > 0) then
      open (newunit=iu, file=trim(dump_path), access='stream', form='unformatted', status='replace')
      write (iu) nx, ny
      write (iu) flux_h
      close (iu)
      write (output_unit, '(3a)') '  wrote ref  : ', trim(dump_path), ' (nx,ny, flux_h)'
   end if

   call get_environment_variable('DC_REF', ref_path, status=ios)
   if (ios == 0 .and. len_trim(ref_path) > 0) call compare_ref(trim(ref_path))

   write (output_unit, '(a)') repeat('=', 70)

   deallocate (h, hu, hv, b, flux_h, flux_hu, flux_hv, mass_flux_x, mass_flux_y)

contains

   subroutine compare_ref(path)
      character(len=*), intent(in) :: path
      real(wp), allocatable :: ref(:, :)
      real(wp) :: dmax, rmax, sc
      integer :: rnx, rny, u, st, nbad
      open (newunit=u, file=path, access='stream', form='unformatted', status='old', iostat=st)
      if (st /= 0) then
         write (output_unit, '(3a)') '  cross-check: ref ', path, ' not found -- skipped'; return
      end if
      read (u) rnx, rny
      if (rnx /= nx .or. rny /= ny) then
         write (output_unit, '(a)') '  cross-check: ref has a different shape -- skipped'
         close (u); return
      end if
      allocate (ref(rnx, rny)); read (u) ref; close (u)
      dmax = 0.0_wp; rmax = 0.0_wp; nbad = 0
      ! Score over the interior the kernel actually writes; the halo is untouched.
      do j = NGHOST + 1, ny - NGHOST
         do i = NGHOST + 1, nx - NGHOST
            dmax = max(dmax, abs(flux_h(i, j) - ref(i, j)))
            sc = max(abs(flux_h(i, j)), abs(ref(i, j)))
            if (sc > 1.0e-30_wp) then
               rmax = max(rmax, abs(flux_h(i, j) - ref(i, j))/sc)
               if (abs(flux_h(i, j) - ref(i, j))/sc > 1.0e-12_wp) nbad = nbad + 1
            end if
         end do
      end do
      write (output_unit, '(a,es12.5,a,es12.5)') '  cross-check vs ref: max|diff| ', dmax, '  max rel ', rmax
      if (rmax < 1.0e-12_wp) then
         write (output_unit, '(a)') '  cross-check: OK (<1e-12 rel -> data layer is numerically inert)'
      else
         write (output_unit, '(a,i0,a)') '  cross-check: *** ', nbad, ' cells >1e-12 rel -- INVESTIGATE ***'
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
