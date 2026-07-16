module m6
   use, intrinsic :: iso_fortran_env, only: wp => real64
   implicit none
   type :: grid_t
      integer :: nx_total = 0, ny_total = 0
   end type grid_t
contains
   ! (i) plain pure callee — implicit acc routine seq (what we had)
   pure subroutine cell_plain(a, nx, ny, i, j, r)
      integer, intent(in) :: nx, ny, i, j
      real(wp), intent(in) :: a(nx, ny)
      real(wp), intent(out) :: r
      r = 2.5_wp*a(i, j) + a(i - 1, j) - a(i + 1, j)
   end subroutine cell_plain

   ! (ii) SAME callee, but EXPLICITLY marked !$acc routine seq
   pure subroutine cell_seq(a, nx, ny, i, j, r)
      !$acc routine seq
      integer, intent(in) :: nx, ny, i, j
      real(wp), intent(in) :: a(nx, ny)
      real(wp), intent(out) :: r
      r = 2.5_wp*a(i, j) + a(i - 1, j) - a(i + 1, j)
   end subroutine cell_seq

   ! C (baseline bug): grid% bounds + call to the PLAIN callee
   pure subroutine c_plain(a, c, grid)
      type(grid_t), intent(in) :: grid
      real(wp), intent(in) :: a(grid%nx_total, grid%ny_total)
      real(wp), intent(out) :: c(grid%nx_total, grid%ny_total)
      integer :: i, j, nx, ny
      real(wp) :: r
      nx = grid%nx_total; ny = grid%ny_total
      do concurrent(j=2:ny - 1, i=2:nx - 1) local(r)
         call cell_plain(a, nx, ny, i, j, r)
         c(i, j) = r
      end do
   end subroutine c_plain

   ! E: grid% bounds + call to the !$acc routine seq callee
   pure subroutine e_seq(a, c, grid)
      type(grid_t), intent(in) :: grid
      real(wp), intent(in) :: a(grid%nx_total, grid%ny_total)
      real(wp), intent(out) :: c(grid%nx_total, grid%ny_total)
      integer :: i, j, nx, ny
      real(wp) :: r
      nx = grid%nx_total; ny = grid%ny_total
      do concurrent(j=2:ny - 1, i=2:nx - 1) local(r)
         call cell_seq(a, nx, ny, i, j, r)
         c(i, j) = r
      end do
   end subroutine e_seq

   ! F: PLAIN bounds + !$acc routine seq callee (does seq alone break collapse?)
   pure subroutine f_plain_seq(a, c, nx, ny)
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: a(nx, ny)
      real(wp), intent(out) :: c(nx, ny)
      integer :: i, j
      real(wp) :: r
      do concurrent(j=2:ny - 1, i=2:nx - 1) local(r)
         call cell_seq(a, nx, ny, i, j, r)
         c(i, j) = r
      end do
   end subroutine f_plain_seq
end module m6
