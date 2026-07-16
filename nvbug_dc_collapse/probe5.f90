module m5
   use, intrinsic :: iso_fortran_env, only: wp => real64
   implicit none
   type :: grid_t
      integer :: nx_total = 0, ny_total = 0, nghost = 2
   end type grid_t
contains
   pure subroutine cell(a, nx, ny, i, j, r)
      integer, intent(in) :: nx, ny, i, j
      real(wp), intent(in) :: a(nx, ny)
      real(wp), intent(out) :: r
      r = 2.5_wp*a(i, j) + a(i - 1, j) - a(i + 1, j)
   end subroutine cell

   ! (K) derived type PRESENT but arrays+bounds from plain integer dummies
   pure subroutine k_dt_unused(a, c, nx, ny, grid)
      type(grid_t), intent(in) :: grid
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: a(nx, ny)
      real(wp), intent(out) :: c(nx, ny)
      integer :: i, j, ng
      real(wp) :: r
      ng = grid%nghost
      do concurrent(j=ng + 1:ny - ng, i=ng + 1:nx - ng) local(r)
         call cell(a, nx, ny, i, j, r)
         c(i, j) = r
      end do
   end subroutine k_dt_unused

   ! (L) ARRAY BOUNDS from grid% ; loop bounds from plain local ints
   pure subroutine l_arraybounds(a, c, grid)
      type(grid_t), intent(in) :: grid
      real(wp), intent(in) :: a(grid%nx_total, grid%ny_total)
      real(wp), intent(out) :: c(grid%nx_total, grid%ny_total)
      integer :: i, j, nx, ny, ng
      real(wp) :: r
      nx = grid%nx_total; ny = grid%ny_total; ng = grid%nghost
      do concurrent(j=ng + 1:ny - ng, i=ng + 1:nx - ng) local(r)
         call cell(a, nx, ny, i, j, r)
         c(i, j) = r
      end do
   end subroutine l_arraybounds

   ! (M) array bounds from grid% but NO call in the body (inline)
   pure subroutine m_arraybounds_inline(a, c, grid)
      type(grid_t), intent(in) :: grid
      real(wp), intent(in) :: a(grid%nx_total, grid%ny_total)
      real(wp), intent(out) :: c(grid%nx_total, grid%ny_total)
      integer :: i, j, nx, ny, ng
      real(wp) :: r
      nx = grid%nx_total; ny = grid%ny_total; ng = grid%nghost
      do concurrent(j=ng + 1:ny - ng, i=ng + 1:nx - ng) local(r)
         r = 2.5_wp*a(i, j) + a(i - 1, j) - a(i + 1, j)
         c(i, j) = r
      end do
   end subroutine m_arraybounds_inline
end module m5
