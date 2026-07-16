#include "directives.h"
!! DC-only driver for the ocean model's kappa-shear (JHL08) column kernel.
!!
!! COMPUTE is a bare `do concurrent (j, i)` (kappa_shear_column_kernel), one
!! thread per COLUMN with a fully sequential, ITERATIVE (Picard x adaptive
!! substep) solve inside. The device data layer is chosen ENTIRELY by
!! directives.h at compile time:
!!   -DDC_DATA_ACC  -> OpenACC  (nvfortran -acc=gpu -stdpar=gpu)   GPU
!!   -DDC_DATA_OMP  -> OpenMP target                               GPU (AMD/Intel too)
!!   (neither)      -> host: bare DC on the CPU (-stdpar=multicore/serial)
!! There is NO CUDA and NO nvcc in this binary -- the comparison-only CUDA
!! variant, its counters (n_out/n_in), and the ks_par_t bridge are all dropped.
!!
!! Cross-check (proves the macro'd data layer did not change the numbers):
!!   DC_DUMP=file  writes nx,ny,nz + kd_int  (a reference)
!!   DC_REF=file   reads that reference and reports max|diff| vs this run
!! Run once with DC_DATA_ACC (GPU) dumping a ref, then again on the CPU reading
!! it: agreement to FMA level means the OpenACC->host swap is numerically inert.
!!
!! NOTE (the old bench's "*** PORT BUG ***" line does NOT apply here): the old
!! DC-vs-CUDA metric is a reduction that can hide sign-paired/permuted
!! differences (documented false alarm -- see the old README). This driver's
!! cross-check is DC-acc vs DC-none/omp -- the SAME do-concurrent kernel across
!! data layers -- which is clean by construction.
!!
!! Usage: ./dc_main [nxp] [nyp] [nz] [nreps] [nwarm] [land_pct]
program dc_main
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   use constants, only: wp, NZ_STACK_MAX
   use grid, only: hgrid_t
   use multilayer_cgrid_state, only: multilayer_cgrid_state_t
   use ocean_eos, only: EOS_VARIANT_LINEAR
   use ks, only: ocean_kappa_shear_t, kappa_shear_column_kernel
   implicit none

   integer, parameter :: NXP_DEF = 473, NYP_DEF = 297, NZ_DEF = 30
   integer, parameter :: REPS_DEF = 200, WARM_DEF = 10, NGHOST = 3
   real(wp), parameter :: DT_THERM = 300.0_wp   !! &time_nml dt_fixed, therm ratio 1

   type(hgrid_t) :: grid
   type(multilayer_cgrid_state_t) :: ms
   type(ocean_kappa_shear_t) :: ks
   real(wp), allocatable :: hT(:, :, :), hS(:, :, :)

   real(wp) :: t0, t1, ms_dc, gib, kd_min, kd_max, kd_sum
   integer :: i, j, k, rep, nx, ny, nz, nxp, nyp, n_reps, n_warm
   integer :: ncol, nwet, land_pct, iu, ios
   character(len=256) :: ref_path, dump_path

   nxp = iarg(1, NXP_DEF); nyp = iarg(2, NYP_DEF); nz = iarg(3, NZ_DEF)
   n_reps = iarg(4, REPS_DEF); n_warm = iarg(5, WARM_DEF); land_pct = iarg(6, 0)

   nx = nxp + 2*NGHOST; ny = nyp + 2*NGHOST
   if (nx < 1 .or. ny < 1 .or. nz < 2) then
      write (output_unit, '(a)') 'ERROR: need nxp,nyp >= 1 and nz >= 2'; stop 1
   end if
   if (nz + 1 > NZ_STACK_MAX) then
      write (output_unit, '(a,i0,a,i0)') 'ERROR: nz+1 = ', nz + 1, &
         ' exceeds NZ_STACK_MAX = ', NZ_STACK_MAX
      stop 1
   end if
   grid%nx_total = nx; grid%ny_total = ny
   grid%nx_phys = nxp; grid%ny_phys = nyp
   grid%nghost = NGHOST; grid%dx = 0.1_wp; grid%dy = 0.1_wp
   ncol = nx*ny

   allocate (ms%h_layer(nx, ny, nz))
   allocate (ms%u_face_x_layer(nx + 1, ny, nz), ms%v_face_y_layer(nx, ny + 1, nz))
   allocate (ms%wet_mask(nx, ny))
   allocate (hT(nx, ny, nz), hS(nx, ny, nz))
   allocate (ks%f_centre(nx, ny))
   allocate (ks%kd_int(nx, ny, nz + 1), ks%tke_int(nx, ny, nz + 1))
   ms%nz_ml = nz

   gib = 8.0_wp*real(nx, wp)*real(ny, wp)*real(nz, wp)*8.0_wp/(1024.0_wp**3)

   call build_state()

   ks%enable = .true.
   ks%eos%variant = EOS_VARIANT_LINEAR
   ks%eos%rho0 = 1035.0_wp
   ks%eos%alpha_T = 0.2_wp     ! &ocean_ic_nml alpha_T -> ocean_state.F90:474
   ks%eos%beta_S = 7.6e-4_wp   ! ocean_eos.F90 default (no nml override)
   ks%rho0 = 1035.0_wp
   ks%kd_int = 0.0_wp; ks%tke_int = 0.0_wp

   nwet = count(ms%wet_mask > 0.0_wp)
   write (output_unit, '(a)') repeat('=', 74)
   write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') '  domain: ', nxp, ' x ', nyp, &
      ' x ', nz, ' interior (+', NGHOST, ' ghosts/side)'
   write (output_unit, '(a,i0,a,i0,a,i0,a)') '  arrays: ', nx, ' x ', ny, &
      ' -> ', ncol, ' COLUMNS (the parallel width)'
   write (output_unit, '(a,i0,a,f5.1,a)') '  wet columns: ', nwet, '  (', &
      100.0_wp*real(nwet, wp)/real(ncol, wp), ' %)'
   write (output_unit, '(a,i0,a,i0,a)') '  NZ_STACK_MAX = ', NZ_STACK_MAX, &
      ' (production); nz = ', nz
   write (output_unit, '(3a,i0,a,i0,a)') '  DATA layer: ', DC_DATA_NAME, &
      '   (reps ', n_reps, ', warm ', n_warm, ')'
   write (output_unit, '(a)') repeat('=', 74)

   ! ---- map the DC working set (no-ops when the DATA layer is 'host'). The
   ! deep copy MUST happen where the variables are owned: parent struct first,
   ! then each allocatable payload. The comparison-only CUDA arrays (kd_cu,
   ! tke_cu, kd_wp_, tke_wp_, n_out, n_in) are gone with the CUDA variant.
   DC_ENTER_IN(ms)
   DC_ENTER_IN(ms%h_layer)
   DC_ENTER_IN(ms%u_face_x_layer)
   DC_ENTER_IN(ms%v_face_y_layer)
   DC_ENTER_IN(ms%wet_mask)
   DC_ENTER_IN(ks)
   DC_ENTER_IN(ks%f_centre)
   DC_ENTER_CREATE(ks%kd_int)
   DC_ENTER_CREATE(ks%tke_int)
   DC_ENTER_IN(hT)
   DC_ENTER_IN(hS)

   ! ---- do concurrent, production verbatim ---------------------------------
   do rep = 1, n_warm
      call kappa_shear_column_kernel(grid, ks, ms, hT, hS, DT_THERM)
   end do
   DC_WAIT
   t0 = wall()
   do rep = 1, n_reps
      call kappa_shear_column_kernel(grid, ks, ms, hT, hS, DT_THERM)
   end do
   DC_WAIT
   t1 = wall()
   ms_dc = (t1 - t0)*1000.0_wp/real(n_reps, wp)

   DC_UPDATE_SELF(ks%kd_int)

   kd_min = minval(ks%kd_int); kd_max = maxval(ks%kd_int); kd_sum = sum(ks%kd_int)

   write (output_unit, '(3a,f10.4,a)') '  kappa-shear do concurrent (', DC_DATA_NAME, ') : ', ms_dc, ' ms/rep'
   write (output_unit, '(a,es14.6)') '  min kd_int : ', kd_min
   write (output_unit, '(a,es14.6)') '  max kd_int : ', kd_max
   write (output_unit, '(a,es14.6)') '  sum kd_int : ', kd_sum
   if (kd_min /= kd_min .or. kd_max /= kd_max) then
      write (output_unit, '(a)') '  sanity     : *** NaN -- results are garbage ***'; stop 2
   else if (kd_min == 0.0_wp .and. kd_max == 0.0_wp) then
      write (output_unit, '(a)') '  sanity     : *** all-zero -- NO COLUMN MIXED (state too stable) ***'; stop 2
   else
      write (output_unit, '(a)') '  sanity     : OK (finite, non-zero)'
   end if

   ! ---- optional reference dump / cross-check ------------------------------
   call get_environment_variable('DC_DUMP', dump_path, status=ios)
   if (ios == 0 .and. len_trim(dump_path) > 0) then
      open (newunit=iu, file=trim(dump_path), access='stream', form='unformatted', status='replace')
      write (iu) nx, ny, nz
      write (iu) ks%kd_int
      close (iu)
      write (output_unit, '(3a)') '  wrote ref       : ', trim(dump_path), ' (nx,ny,nz, kd_int)'
   end if

   call get_environment_variable('DC_REF', ref_path, status=ios)
   if (ios == 0 .and. len_trim(ref_path) > 0) call compare_ref(trim(ref_path))

   write (output_unit, '(a)') repeat('=', 74)

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
      allocate (ref(rnx, rny, rnz + 1)); read (u) ref; close (u)
      dmax = 0.0_wp; rmax = 0.0_wp; nbad = 0
      do k = 1, nz + 1
         do j = 1, ny
            do i = 1, nx
               dmax = max(dmax, abs(ks%kd_int(i, j, k) - ref(i, j, k)))
               sc = max(abs(ks%kd_int(i, j, k)), abs(ref(i, j, k)))
               if (sc > 1.0e-30_wp) then
                  rmax = max(rmax, abs(ks%kd_int(i, j, k) - ref(i, j, k))/sc)
                  if (abs(ks%kd_int(i, j, k) - ref(i, j, k))/sc > 1.0e-12_wp) nbad = nbad + 1
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

   !! Build the physically realistic state described in the old bench header:
   !! z* grid, a surface mixed layer straddling Ri_crit, a surface-intensified
   !! jet, and a per-column iteration count that actually varies. VERBATIM from
   !! ks_bench.F90's build_state.
   subroutine build_state()
      integer, parameter :: NZ_DEF_MAX = 512
      real(wp) :: depth, mld, us, vs, zt, zb, zm, hh, tt, ss, uu, vv
      real(wp) :: w(NZ_DEF_MAX), wsum, r, xr, yr, lat
      integer :: m, kg

      r = 1.18_wp   ! geometric stretch -> ~5 m surface layer at H = 4000 m
      wsum = 0.0_wp
      do m = 1, nz
         w(m) = r**(m - 1)
         wsum = wsum + w(m)
      end do
      do m = 1, nz
         w(m) = w(m)/wsum
      end do

      do j = 1, ny
         yr = real(j - NGHOST, wp)/real(max(nyp, 1), wp)      ! 0..1 south->north
         lat = -60.61_wp + 0.1_wp*real(j - NGHOST, wp)        ! gabight lat span
         do i = 1, nx
            xr = real(i - NGHOST, wp)/real(max(nxp, 1), wp)

            ! |f| from the planetary metric (coriolis_scheme = "planetary").
            ks%f_centre(i, j) = abs(2.0_wp*7.2921e-5_wp*sin(lat*3.141592653589793_wp/180.0_wp))

            ! Bathymetry: shelf -> abyssal, smooth, 200..4500 m.
            depth = 200.0_wp + 4300.0_wp* &
                    (0.5_wp*(1.0_wp + tanh(3.0_wp*(yr - 0.25_wp))))* &
                    (0.85_wp + 0.15_wp*sin(6.0_wp*xr))

            ! Land: a contiguous NW block, so the mask is spatially coherent.
            if (land_pct > 0) then
               if (xr < 0.01_wp*real(land_pct, wp) .and. yr > 0.55_wp) then
                  ms%wet_mask(i, j) = 0.0_wp
               else
                  ms%wet_mask(i, j) = 1.0_wp
               end if
            else
               ms%wet_mask(i, j) = 1.0_wp
            end if

            ! Mixed-layer depth 40..100 m, and the jet strength that competes
            ! with the stratification at its base.
            mld = 40.0_wp + 60.0_wp*(0.5_wp*(1.0_wp + sin(5.0_wp*xr)*cos(4.0_wp*yr)))
            us = 0.25_wp + 0.60_wp*(0.5_wp*(1.0_wp + sin(7.0_wp*xr + 2.0_wp*yr)* &
                                            cos(3.0_wp*yr)))
            vs = 0.10_wp*sin(4.0_wp*xr)*cos(5.0_wp*yr)

            zt = 0.0_wp
            do m = 1, nz          ! m = local, surface-down
               hh = max(depth*w(m), 1.0e-3_wp)
               zb = zt + hh
               zm = 0.5_wp*(zt + zb)

               ! T: uniform in the ML, thermocline below (T_surf 14 -> T_bot 2).
               if (zm <= mld) then
                  tt = 14.0_wp
               else
                  tt = 2.0_wp + 12.0_wp*exp(-(zm - mld)/600.0_wp)
               end if
               ss = 35.0_wp - 0.5_wp*exp(-zm/300.0_wp)

               ! u: surface jet, shear concentrated at the ML base.
               uu = us*0.5_wp*(1.0_wp - tanh((zm - mld)/25.0_wp))
               vv = vs*0.5_wp*(1.0_wp - tanh((zm - mld)/25.0_wp))

               kg = nz + 1 - m    ! global bottom-up: kg = nz is the surface
               ms%h_layer(i, j, kg) = hh
               hT(i, j, kg) = tt*hh
               hS(i, j, kg) = ss*hh
               ms%u_face_x_layer(i, j, kg) = uu
               ms%v_face_y_layer(i, j, kg) = vv
               zt = zb
            end do
         end do
      end do
      ! Face arrays carry one extra row/column; fill the overhang by copy so
      ! the face average at i = nx / j = ny is not reading uninitialised memory.
      do kg = 1, nz
         do j = 1, ny
            ms%u_face_x_layer(nx + 1, j, kg) = ms%u_face_x_layer(nx, j, kg)
         end do
         do i = 1, nx
            ms%v_face_y_layer(i, ny + 1, kg) = ms%v_face_y_layer(i, ny, kg)
         end do
      end do
   end subroutine build_state

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
