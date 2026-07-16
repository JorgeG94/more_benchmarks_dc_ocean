program repro
   use, intrinsic :: iso_fortran_env, only: wp => real64, int64, output_unit
   use repro_m
   implicit none

   integer, parameter :: NX = 4096, NY = 4096, NREP = 50
   type(grid_t) :: grid
   real(wp), allocatable :: a(:, :), c1(:, :), c2(:, :), c3(:, :), c4(:, :)
   real(wp) :: t0, t1, ms_a, ms_b, ms_c, ms_d, dmax
   integer :: i, j, r

   grid%nx_total = NX; grid%ny_total = NY; grid%nghost = 1

   allocate (a(NX, NY), c1(NX, NY), c2(NX, NY), c3(NX, NY), c4(NX, NY))
   do concurrent(j=1:NY, i=1:NX)
      a(i, j) = real(i, wp) + 0.5_wp*real(j, wp)
   end do
   c1 = 0.0_wp; c2 = 0.0_wp; c3 = 0.0_wp; c4 = 0.0_wp

   !$acc enter data copyin(a) create(c1, c2, c3, c4)

   call a_plain_call(a, c1, NX, NY); !$acc wait
   t0 = wall(); do r = 1, NREP; call a_plain_call(a, c1, NX, NY); end do
   !$acc wait
   t1 = wall(); ms_a = (t1 - t0)*1000.0_wp/NREP

   call b_dt_inline(a, c2, grid); !$acc wait
   t0 = wall(); do r = 1, NREP; call b_dt_inline(a, c2, grid); end do
   !$acc wait
   t1 = wall(); ms_b = (t1 - t0)*1000.0_wp/NREP

   call c_dt_call(a, c3, grid); !$acc wait
   t0 = wall(); do r = 1, NREP; call c_dt_call(a, c3, grid); end do
   !$acc wait
   t1 = wall(); ms_c = (t1 - t0)*1000.0_wp/NREP

   call d_workaround(a, c4, NX, NY, grid); !$acc wait
   t0 = wall(); do r = 1, NREP; call d_workaround(a, c4, NX, NY, grid); end do
   !$acc wait
   t1 = wall(); ms_d = (t1 - t0)*1000.0_wp/NREP

   !$acc update self(c1, c2, c3, c4)
   !$acc exit data delete(a, c1, c2, c3, c4)

   ! all four compute the same thing
   dmax = 0.0_wp
   do j = 2, NY - 1
      do i = 2, NX - 1
         dmax = max(dmax, abs(c1(i, j) - c3(i, j)), abs(c1(i, j) - c2(i, j)), &
                    abs(c1(i, j) - c4(i, j)))
      end do
   end do

   write (output_unit, '(a,i0,a,i0,a,i0,a)') 'grid ', NX, ' x ', NY, ', ', NREP, ' reps'
   write (output_unit, '(a)') repeat('-', 64)
   write (output_unit, '(a,f9.4,a)') '  A  plain bounds  + call    : ', ms_a, ' ms   [collapses]'
   write (output_unit, '(a,f9.4,a)') '  B  grid% bounds  + inline  : ', ms_b, ' ms   [collapses]'
   write (output_unit, '(a,f9.4,a)') '  C  grid% bounds  + call    : ', ms_c, ' ms   *** NO COLLAPSE ***'
   write (output_unit, '(a,f9.4,a)') '  D  workaround (plain dims) : ', ms_d, ' ms   [collapses]'
   write (output_unit, '(a)') repeat('-', 64)
   write (output_unit, '(a,f6.2,a)') '  C is ', ms_c/ms_a, 'x slower than A (same maths)'
   write (output_unit, '(a,es10.3,a)') '  max |difference| between all four: ', dmax, '  (0 = identical)'
   deallocate (a, c1, c2, c3, c4)

contains
   function wall() result(t)
      real(wp) :: t
      integer(int64) :: cnt, rate
      call system_clock(cnt, rate)
      t = real(cnt, wp)/real(rate, wp)
   end function wall
end program repro
