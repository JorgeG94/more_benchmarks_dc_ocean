!! Shared problem setup + verification for both MRE variants.
!!
!! Deliberately identical between the `do concurrent` and the CUDA C version so
!! the ONLY difference between the two programs is which kernel touches the
!! device data. Anything else here would confound the comparison.
module daxpy_common
   use, intrinsic :: iso_fortran_env, only: real64, int64, output_unit
   implicit none
   private

   public :: wp, nx, ny, nz, alpha, n_reps
   public :: fill_inputs, verify, report, wall_seconds

   integer, parameter :: wp = real64

   ! 512^3 doubles = 128 MiB per array, 384 MiB for three. Comfortable on a
   ! 32 GiB V100 and big enough that the kernel is bandwidth-bound rather than
   ! launch-overhead-bound, which is the regime the comparison is about.
   integer, parameter :: nx = 512, ny = 512, nz = 512
   real(wp), parameter :: alpha = 2.5_wp
   integer, parameter :: n_reps = 20

contains

   !! a(i,j,k) and b(i,j,k) get values that depend on ALL THREE indices, so a
   !! transposed/wrong linearisation in the CUDA kernel produces a wrong answer
   !! instead of accidentally matching. (Fill with a(i,j,k)=i and any index
   !! permutation still "verifies" — a trap worth avoiding in an MRE whose
   !! whole subject is index mapping.)
   pure subroutine fill_inputs(a, b)
      real(wp), intent(out) :: a(nx, ny, nz), b(nx, ny, nz)
      integer :: i, j, k
      do concurrent(k=1:nz, j=1:ny, i=1:nx)
         a(i, j, k) = real(i, wp) + 1000.0_wp*real(j, wp) + 1000000.0_wp*real(k, wp)
         b(i, j, k) = real(i, wp) - 0.5_wp*real(j, wp) + 0.25_wp*real(k, wp)
      end do
   end subroutine fill_inputs

   !! Recompute on the HOST and compare. Returns max abs error.
   !! The expected value is exact in binary64 for these inputs, so the bar is
   !! literal equality — not a tolerance. A tolerance here would hide exactly
   !! the indexing bug this MRE exists to catch.
   subroutine verify(c, max_err, n_bad)
      real(wp), intent(in) :: c(nx, ny, nz)
      real(wp), intent(out) :: max_err
      integer, intent(out) :: n_bad
      real(wp) :: expect, err
      integer :: i, j, k

      max_err = 0.0_wp
      n_bad = 0
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               expect = alpha*(real(i, wp) + 1000.0_wp*real(j, wp) + 1000000.0_wp*real(k, wp)) &
                        + (real(i, wp) - 0.5_wp*real(j, wp) + 0.25_wp*real(k, wp))
               err = abs(c(i, j, k) - expect)
               if (err > 0.0_wp) then
                  n_bad = n_bad + 1
                  if (err > max_err) max_err = err
               end if
            end do
         end do
      end do
   end subroutine verify

   subroutine report(label, secs, max_err, n_bad)
      character(*), intent(in) :: label
      real(wp), intent(in) :: secs, max_err
      integer, intent(in) :: n_bad
      real(wp) :: gbytes, gbps

      ! daxpy moves 3 arrays: read a, read b, write c.
      gbytes = 3.0_wp*real(nx, wp)*real(ny, wp)*real(nz, wp)*8.0_wp/1.0e9_wp

      write (output_unit, '(a)') repeat('-', 62)
      write (output_unit, '(a,a)') '  variant        : ', label
      write (output_unit, '(a,i0,a,i0,a,i0,a,i0,a)') &
         '  grid           : ', nx, ' x ', ny, ' x ', nz, '  (', nx*ny*nz, ' elements)'
      write (output_unit, '(a,f8.3,a)') '  moved / rep    : ', gbytes, ' GB'
      write (output_unit, '(a,i0)') '  reps           : ', n_reps
      write (output_unit, '(a,f10.5,a)') '  time / rep     : ', secs*1000.0_wp, ' ms'
      gbps = gbytes/secs
      write (output_unit, '(a,f8.1,a)') '  bandwidth      : ', gbps, ' GB/s'
      if (n_bad == 0) then
         write (output_unit, '(a)') '  verify         : PASS (bit-exact vs host)'
      else
         write (output_unit, '(a,i0,a,es12.5)') &
            '  verify         : *** FAIL *** ', n_bad, ' wrong, max err ', max_err
      end if
      write (output_unit, '(a)') repeat('-', 62)
   end subroutine report

   function wall_seconds() result(t)
      real(wp) :: t
      integer(int64) :: cnt, rate
      call system_clock(cnt, rate)
      t = real(cnt, wp)/real(rate, wp)
   end function wall_seconds

end module daxpy_common
