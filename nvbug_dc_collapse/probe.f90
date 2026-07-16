module m
   use, intrinsic :: iso_fortran_env, only: wp => real64
   implicit none
contains
   ! A trivial pure subroutine. Nothing exotic: no I/O, no state, no allocation.
   pure subroutine axpb(x, y, r)
      real(wp), intent(in) :: x, y
      real(wp), intent(out) :: r
      r = 2.5_wp*x + y
   end subroutine axpb

   ! (A) body written inline
   subroutine inline_body(a, b, c, nx, ny)
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: a(nx, ny), b(nx, ny)
      real(wp), intent(out) :: c(nx, ny)
      integer :: i, j
      do concurrent(j=1:ny, i=1:nx)
         c(i, j) = 2.5_wp*a(i, j) + b(i, j)
      end do
   end subroutine inline_body

   ! (B) IDENTICAL maths, but via a call to the pure subroutine above
   subroutine called_body(a, b, c, nx, ny)
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: a(nx, ny), b(nx, ny)
      real(wp), intent(out) :: c(nx, ny)
      integer :: i, j
      real(wp) :: t
      do concurrent(j=1:ny, i=1:nx) local(t)
         call axpb(a(i, j), b(i, j), t)
         c(i, j) = t
      end do
   end subroutine called_body
end module m
