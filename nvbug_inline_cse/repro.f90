! nvfortran fails to CSE global loads in an INLINED procedure inside
! `do concurrent`: the same loop body written flat vs. via a `pure` helper
! call compiles to 6 vs. 10 LDG.64 (+67%) for bit-identical semantics.
!
! Expected: k_call and k_flat compile to the same code — the helper is fully
!           inlined (no CALL in SASS, zero spills, -Minfo reports both loops
!           auto-collapsed identically).
! Actual  : k_call re-loads array elements whose values are already live —
!           every element read by the IF condition is loaded AGAIN on the
!           other side of the branch. k_flat merges them. The trigger is a
!           branch in the callee whose condition reads array elements also
!           used elsewhere in the body; a branchless callee (merge()) or a
!           branch on a scalar does not trigger it.
! Impact  : this toy is bandwidth-bound at full occupancy, so its wall-clock
!           ties — the defect is the instruction stream. On our production
!           shallow-water HLL flux kernel (branchy 500-line helper, 126 regs,
!           25% occupancy, V100) the same pattern costs 22% wall-clock vs.
!           the hand-flattened identical source: 66 vs 56 LDG per thread,
!           +25% DFMA, 12 extra sqrt expansions, +53% long-scoreboard stalls
!           — the flat form matches a hand-written CUDA C port exactly.
!
! Build + count (no GPU needed):
!   nvfortran -O2 -stdpar=gpu -gpu=cc70 -c repro.f90 -o repro.o
!   cuobjdump -sass repro.o | awk '/Function : /{fn=$3} /LDG/{n[fn]++} \
!                                  END{for(f in n) print f, n[f]}'
!     -> probe_m_k_call_*_gpu  10
!        probe_m_k_flat_*_gpu   6
! Run (verifies bit-identical results, times both):
!   nvfortran -O2 -stdpar=gpu -gpu=cc70,mem:separate repro.f90 -o repro && ./repro
!
! nvfortran 26.5-0, CUDA 12.9, Tesla V100 (cc70). Same counts at -O3 -fast.

module probe_m
   use, intrinsic :: iso_fortran_env, only: wp => real64
   implicit none
   real(wp), parameter :: TOL = 1.0e-6_wp
contains

   ! the callee: one branch whose condition reads h at 3 stencil points,
   ! all of which are also used in the arithmetic below the branch
   pure subroutine cell(h, b, nx, ny, i, j, o1, o2)
      integer, intent(in) :: nx, ny, i, j
      real(wp), intent(in) :: h(nx, ny), b(nx, ny)
      real(wp), intent(out) :: o1, o2
      real(wp) :: ec, el, er, s
      ec = h(i, j) + b(i, j)
      el = h(i - 1, j) + b(i - 1, j)
      er = h(i + 1, j) + b(i + 1, j)
      if (h(i, j) < TOL .or. h(i - 1, j) < TOL .or. h(i + 1, j) < TOL) then
         s = 0.0_wp
      else
         s = (er - ec) + (ec - el)
      end if
      o1 = s*ec + sqrt(ec)
      o2 = s*(er - el) + h(i, j)*b(i + 1, j)
   end subroutine cell

   ! loop body = the call                      -> 10 LDG per iteration
   subroutine k_call(h, b, c, d, nx, ny)
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: h(nx, ny), b(nx, ny)
      real(wp), intent(out) :: c(nx, ny), d(nx, ny)
      integer :: i, j
      real(wp) :: o1, o2
      do concurrent(j=2:ny - 1, i=2:nx - 1) local(o1, o2)
         call cell(h, b, nx, ny, i, j, o1, o2)
         c(i, j) = o1
         d(i, j) = o2
      end do
   end subroutine k_call

   ! loop body = the callee pasted verbatim    ->  6 LDG per iteration
   subroutine k_flat(h, b, c, d, nx, ny)
      integer, intent(in) :: nx, ny
      real(wp), intent(in) :: h(nx, ny), b(nx, ny)
      real(wp), intent(out) :: c(nx, ny), d(nx, ny)
      integer :: i, j
      real(wp) :: ec, el, er, s, o1, o2
      do concurrent(j=2:ny - 1, i=2:nx - 1) local(ec, el, er, s, o1, o2)
         ec = h(i, j) + b(i, j)
         el = h(i - 1, j) + b(i - 1, j)
         er = h(i + 1, j) + b(i + 1, j)
         if (h(i, j) < TOL .or. h(i - 1, j) < TOL .or. h(i + 1, j) < TOL) then
            s = 0.0_wp
         else
            s = (er - ec) + (ec - el)
         end if
         o1 = s*ec + sqrt(ec)
         o2 = s*(er - el) + h(i, j)*b(i + 1, j)
         c(i, j) = o1
         d(i, j) = o2
      end do
   end subroutine k_flat

end module probe_m

program repro
   use, intrinsic :: iso_fortran_env, only: wp => real64, int64, output_unit
   use probe_m
   implicit none
   integer, parameter :: N = 4096, NREP = 50
   real(wp), allocatable :: h(:, :), b(:, :), c1(:, :), d1(:, :), c2(:, :), d2(:, :)
   real(wp) :: t0, t1, ms_call, ms_flat, dmax
   integer :: i, j, r

   allocate (h(N, N), b(N, N), c1(N, N), d1(N, N), c2(N, N), d2(N, N))
   do concurrent(j=1:N, i=1:N)
      h(i, j) = 2.0_wp + sin(0.01_wp*real(i, wp))*cos(0.01_wp*real(j, wp))
      b(i, j) = 0.1_wp*real(i + j, wp)/real(2*N, wp)
   end do
   c1 = 0.0_wp; d1 = 0.0_wp; c2 = 0.0_wp; d2 = 0.0_wp

   !$acc enter data copyin(h, b) create(c1, d1, c2, d2)

   call k_call(h, b, c1, d1, N, N); !$acc wait
   t0 = wall(); do r = 1, NREP; call k_call(h, b, c1, d1, N, N); end do
   !$acc wait
   t1 = wall(); ms_call = (t1 - t0)*1000.0_wp/NREP

   call k_flat(h, b, c2, d2, N, N); !$acc wait
   t0 = wall(); do r = 1, NREP; call k_flat(h, b, c2, d2, N, N); end do
   !$acc wait
   t1 = wall(); ms_flat = (t1 - t0)*1000.0_wp/NREP

   !$acc update self(c1, d1, c2, d2)
   !$acc exit data delete(h, b, c1, d1, c2, d2)

   dmax = 0.0_wp
   do j = 2, N - 1
      do i = 2, N - 1
         dmax = max(dmax, abs(c1(i, j) - c2(i, j)), abs(d1(i, j) - d2(i, j)))
      end do
   end do

   write (output_unit, '(a,f9.4,a)') '  body = call helper : ', ms_call, ' ms'
   write (output_unit, '(a,f9.4,a)') '  body written flat  : ', ms_flat, ' ms'
   write (output_unit, '(a,es10.3,a)') '  max |difference|   : ', dmax, '  (0 = bit-identical)'

contains
   function wall() result(t)
      real(wp) :: t
      integer(int64) :: cnt, rate
      call system_clock(cnt, rate)
      t = real(cnt, wp)/real(rate, wp)
   end function wall
end program repro
