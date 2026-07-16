module m3
   use, intrinsic :: iso_fortran_env, only: wp => real64
   implicit none
contains
   pure function leaf(x, y) result(r)     ! depth-2 leaf
      real(wp), intent(in) :: x, y
      real(wp) :: r
      if (x*y <= 0.0_wp) then; r = 0.0_wp
      else if (abs(x) < abs(y)) then; r = x
      else; r = y; end if
   end function leaf

   pure subroutine mid(a, nx, ny, i, j, r)   ! calls leaf -> nested depth 2
      integer, intent(in) :: nx, ny, i, j
      real(wp), intent(in) :: a(nx, ny)
      real(wp), intent(out) :: r
      r = leaf(a(i, j) - a(i - 1, j), a(i + 1, j) - a(i, j))
   end subroutine mid

   ! (F) NESTED call: DC -> mid -> leaf
   subroutine nested_call(a, c, nx, ny)
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: a(nx, ny)
      real(wp), intent(out) :: c(nx, ny)
      integer :: i, j
      real(wp) :: r
      do concurrent(j=2:ny - 1, i=2:nx - 1) local(r)
         call mid(a, nx, ny, i, j, r)
         c(i, j) = r
      end do
   end subroutine nested_call

   ! (G) BIG called body: many locals, lots of arithmetic (flux_cell-ish size)
   pure subroutine big(a, nx, ny, i, j, r)
      integer, intent(in) :: nx, ny, i, j
      real(wp), intent(in) :: a(nx, ny)
      real(wp), intent(out) :: r
      real(wp) :: t(40)
      integer :: k
      t(1) = a(i, j)
      do k = 2, 40
         t(k) = sqrt(abs(t(k - 1))) + a(i - 1, j)*real(k, wp) - a(i + 1, j)/real(k, wp)
      end do
      r = sum(t)
   end subroutine big

   subroutine big_call(a, c, nx, ny)
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: a(nx, ny)
      real(wp), intent(out) :: c(nx, ny)
      integer :: i, j
      real(wp) :: r
      do concurrent(j=2:ny - 1, i=2:nx - 1) local(r)
         call big(a, nx, ny, i, j, r)
         c(i, j) = r
      end do
   end subroutine big_call
end module m3
