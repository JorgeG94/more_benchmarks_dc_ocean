module m2
   use, intrinsic :: iso_fortran_env, only: wp => real64
   implicit none
contains
   ! like flux_cell: takes the ARRAYS + the indices, returns several scalars
   pure subroutine cell_kernel(a, b, nx, ny, i, j, r1, r2)
      integer, intent(in) :: nx, ny, i, j
      real(wp), intent(in) :: a(nx, ny), b(nx, ny)
      real(wp), intent(out) :: r1, r2
      r1 = 2.5_wp*a(i, j) + b(i - 1, j)
      r2 = a(i + 1, j) - b(i, j)
   end subroutine cell_kernel

   ! (C) call takes ARRAYS + indices
   subroutine arrays_passed(a, b, c, d, nx, ny)
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: a(nx, ny), b(nx, ny)
      real(wp), intent(out) :: c(nx, ny), d(nx, ny)
      integer :: i, j
      real(wp) :: r1, r2
      do concurrent(j=2:ny - 1, i=2:nx - 1) local(r1, r2)
         call cell_kernel(a, b, nx, ny, i, j, r1, r2)
         c(i, j) = r1
         d(i, j) = r2
      end do
   end subroutine arrays_passed

   ! (D) inline body + the CONDITIONAL NEIGHBOUR WRITE the flux kernel does
   subroutine cond_write(a, b, c, nx, ny)
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: a(nx, ny), b(nx, ny)
      real(wp), intent(out) :: c(nx, ny)
      integer :: i, j
      real(wp) :: r1
      do concurrent(j=2:ny - 1, i=2:nx - 1) local(r1)
         r1 = 2.5_wp*a(i, j) + b(i, j)
         c(i, j) = r1
         if (i == 2) c(i - 1, j) = r1
      end do
   end subroutine cond_write

   ! (E) arrays passed AND conditional neighbour write (= the flux kernel shape)
   subroutine both(a, b, c, d, nx, ny)
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: a(nx, ny), b(nx, ny)
      real(wp), intent(out) :: c(nx, ny), d(nx, ny)
      integer :: i, j
      real(wp) :: r1, r2
      do concurrent(j=2:ny - 1, i=2:nx - 1) local(r1, r2)
         call cell_kernel(a, b, nx, ny, i, j, r1, r2)
         c(i, j) = r1
         d(i, j) = r2
         if (i == 2) c(i - 1, j) = r1
      end do
   end subroutine both
end module m2
