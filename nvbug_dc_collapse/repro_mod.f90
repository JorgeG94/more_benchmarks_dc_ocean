! nvfortran: `do concurrent` auto-collapse is silently disabled when an
! explicit-shape array dummy is bounded by a derived-type component AND the
! loop body contains a procedure call.
!
! Four variants of the SAME 2-D stencil. They differ only in
!   (a) where the array dummy's bounds come from, and
!   (b) whether the loop body calls a procedure or is written inline.
!
! Expected: all four map the 2-D iteration space to one thread per cell.
! Actual  : variant C does not collapse -- it maps j to blocks and i to 128
!           threads, so each thread serially walks 32 cells at nx=4096.
!
! Build:
!   nvfortran -O2 -stdpar=gpu -gpu=cc70,mem:separate -Minfo=stdpar,accel -c repro.f90
!
! Compiler: nvfortran 26.5-0 (also seen on 25.5), CUDA 12.9, Tesla V100 (cc70).

module repro_m
   use, intrinsic :: iso_fortran_env, only: wp => real64
   implicit none

   type :: grid_t
      integer :: nx_total = 0, ny_total = 0, nghost = 1
   end type grid_t

contains

   ! The called leaf. Trivial, pure, side-effect free.
   pure subroutine cell(a, nx, ny, i, j, r)
      integer, intent(in) :: nx, ny, i, j
      real(wp), intent(in) :: a(nx, ny)
      real(wp), intent(out) :: r
      r = 2.5_wp*a(i, j) + a(i - 1, j) - a(i + 1, j)
   end subroutine cell

   !-----------------------------------------------------------------------
   ! A. bounds from plain integer dummies + CALL      -> collapses (OK)
   !-----------------------------------------------------------------------
   pure subroutine a_plain_call(a, c, nx, ny)
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: a(nx, ny)
      real(wp), intent(out) :: c(nx, ny)
      integer :: i, j
      real(wp) :: r
      do concurrent(j=2:ny - 1, i=2:nx - 1) local(r)
         call cell(a, nx, ny, i, j, r)
         c(i, j) = r
      end do
   end subroutine a_plain_call

   !-----------------------------------------------------------------------
   ! B. bounds from grid%component + INLINE body      -> collapses (OK)
   !-----------------------------------------------------------------------
   pure subroutine b_dt_inline(a, c, grid)
      type(grid_t), intent(in) :: grid
      real(wp), intent(in) :: a(grid%nx_total, grid%ny_total)
      real(wp), intent(out) :: c(grid%nx_total, grid%ny_total)
      integer :: i, j, nx, ny
      real(wp) :: r
      nx = grid%nx_total; ny = grid%ny_total
      do concurrent(j=2:ny - 1, i=2:nx - 1) local(r)
         r = 2.5_wp*a(i, j) + a(i - 1, j) - a(i + 1, j)
         c(i, j) = r
      end do
   end subroutine b_dt_inline

   !-----------------------------------------------------------------------
   ! C. bounds from grid%component + CALL      -> *** DOES NOT COLLAPSE ***
   !    Identical maths to A and B. Only the combination differs.
   !-----------------------------------------------------------------------
   pure subroutine c_dt_call(a, c, grid)
      type(grid_t), intent(in) :: grid
      real(wp), intent(in) :: a(grid%nx_total, grid%ny_total)
      real(wp), intent(out) :: c(grid%nx_total, grid%ny_total)
      integer :: i, j, nx, ny
      real(wp) :: r
      nx = grid%nx_total; ny = grid%ny_total
      do concurrent(j=2:ny - 1, i=2:nx - 1) local(r)
         call cell(a, nx, ny, i, j, r)
         c(i, j) = r
      end do
   end subroutine c_dt_call

   !-----------------------------------------------------------------------
   ! D. THE WORKAROUND: same as C, but the bounds are passed as plain
   !    integers alongside the derived type. Collapses again.
   !-----------------------------------------------------------------------
   pure subroutine d_workaround(a, c, nx, ny, grid)
      type(grid_t), intent(in) :: grid
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: a(nx, ny)
      real(wp), intent(out) :: c(nx, ny)
      integer :: i, j, ng
      real(wp) :: r
      ng = grid%nghost
      do concurrent(j=1 + ng:ny - ng, i=1 + ng:nx - ng) local(r)
         call cell(a, nx, ny, i, j, r)
         c(i, j) = r
      end do
   end subroutine d_workaround

end module repro_m


