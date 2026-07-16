module m4
   use, intrinsic :: iso_fortran_env, only: wp => real64
   implicit none
   type :: grid_t
      integer :: nx_total = 0, ny_total = 0, nghost = 2
      real(wp) :: dx = 0.0_wp, dy = 0.0_wp
   end type grid_t
contains
   pure subroutine cell(a, nx, ny, i, j, r)
      integer, intent(in) :: nx, ny, i, j
      real(wp), intent(in) :: a(nx, ny)
      real(wp), intent(out) :: r
      r = 2.5_wp*a(i, j) + a(i - 1, j) - a(i + 1, j)
   end subroutine cell

   ! (H) NON-pure driver, plain nx/ny args, call  -> baseline (collapses)
   subroutine drv_nonpure(a, c, nx, ny)
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: a(nx, ny)
      real(wp), intent(out) :: c(nx, ny)
      integer :: i, j
      real(wp) :: r
      do concurrent(j=2:ny - 1, i=2:nx - 1) local(r)
         call cell(a, nx, ny, i, j, r)
         c(i, j) = r
      end do
   end subroutine drv_nonpure

   ! (I) PURE driver  <-- the shipped kernel is `pure subroutine compute_flux_hll`
   pure subroutine drv_pure(a, c, nx, ny)
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: a(nx, ny)
      real(wp), intent(out) :: c(nx, ny)
      integer :: i, j
      real(wp) :: r
      do concurrent(j=2:ny - 1, i=2:nx - 1) local(r)
         call cell(a, nx, ny, i, j, r)
         c(i, j) = r
      end do
   end subroutine drv_pure

   ! (J) derived-type arg, bounds from grid%  <-- the shipped kernel's shape
   pure subroutine drv_grid(a, c, grid)
      type(grid_t), intent(in) :: grid
      real(wp), intent(in) :: a(grid%nx_total, grid%ny_total)
      real(wp), intent(out) :: c(grid%nx_total, grid%ny_total)
      integer :: i, j, nx, ny, nghost
      real(wp) :: r
      nx = grid%nx_total; ny = grid%ny_total; nghost = grid%nghost
      do concurrent(j=nghost + 1:ny - nghost, i=nghost + 1:nx - nghost) local(r)
         call cell(a, nx, ny, i, j, r)
         c(i, j) = r
      end do
   end subroutine drv_grid
end module m4
