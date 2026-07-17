!! ideal_benchmark / rk2_main.F90  -- the whole-model RK2 harness.
!!
!! A "dumb" RK2 (2-stage) time-stepping harness that runs the 8-kernel ocean
!! core back-to-back per stage, keeping the GPU (or CPU) hot, to measure the
!! aggregate whole-model cost. Physical correctness does NOT matter: each kernel
!! runs on its own resident device state, driven in lockstep.
!!
!! 3-way model-level comparison, selected at run time:
!!   MODE    dc | cuda | both     which side(s) to TIME
!!   VERSION opt | unopt          optimized routines vs the faithful ports
!! and a CPU build (rk2_cpu, -DRK2_NO_CUDA) runs the whole thing in pure
!! `do concurrent` on the host (DC-only).
!!
!! Stage order (hll excluded -> the 8-kernel ocean core):
!!   continuity, redi, ale, hvisc, btstep, kappa, epbl, meke
!!
!! CLI (GPU):  ./rk2      NXP NYP NZ NSTEPS NWARM [MODE] [VERSION]
!! CLI (CPU):  ./rk2_cpu  NXP NYP NZ NSTEPS NWARM [VERSION]        (mode = dc)
!!
!! Machine-readable line, ONE per timed side:
!!   RESULT target=<gpu|cpu> mode=<dc|cuda> version=<opt|unopt> ms_per_stage=<f> ms_per_step=<f>
program rk2_main
   use, intrinsic :: iso_c_binding, only: c_int, c_double
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   implicit none

   interface
      subroutine rk2_continuity_init(nx, ny, nz) bind(C, name="rk2_continuity_init")
         import :: c_int; integer(c_int), value :: nx, ny, nz
      end subroutine
      subroutine rk2_redi_init(nx, ny, nz) bind(C, name="rk2_redi_init")
         import :: c_int; integer(c_int), value :: nx, ny, nz
      end subroutine
      subroutine rk2_kappa_init(nx, ny, nz) bind(C, name="rk2_kappa_init")
         import :: c_int; integer(c_int), value :: nx, ny, nz
      end subroutine
      subroutine rk2_ale_init(nx, ny, nz) bind(C, name="rk2_ale_init")
         import :: c_int; integer(c_int), value :: nx, ny, nz
      end subroutine
      subroutine rk2_btstep_init(nx, ny, nz) bind(C, name="rk2_btstep_init")
         import :: c_int; integer(c_int), value :: nx, ny, nz
      end subroutine
      subroutine rk2_epbl_init(nx, ny, nz) bind(C, name="rk2_epbl_init")
         import :: c_int; integer(c_int), value :: nx, ny, nz
      end subroutine
      subroutine rk2_meke_init(nx, ny, nz) bind(C, name="rk2_meke_init")
         import :: c_int; integer(c_int), value :: nx, ny, nz
      end subroutine
      subroutine rk2_hvisc_init(nx, ny, nz) bind(C, name="rk2_hvisc_init")
         import :: c_int; integer(c_int), value :: nx, ny, nz
      end subroutine

      subroutine rk2_continuity_stage() bind(C, name="rk2_continuity_stage"); end subroutine
      subroutine rk2_redi_stage()       bind(C, name="rk2_redi_stage");       end subroutine
      subroutine rk2_kappa_stage()      bind(C, name="rk2_kappa_stage");      end subroutine
      subroutine rk2_ale_stage()        bind(C, name="rk2_ale_stage");        end subroutine
      subroutine rk2_btstep_stage()     bind(C, name="rk2_btstep_stage");     end subroutine
      subroutine rk2_epbl_stage()       bind(C, name="rk2_epbl_stage");       end subroutine
      subroutine rk2_meke_stage()       bind(C, name="rk2_meke_stage");       end subroutine
      subroutine rk2_hvisc_stage()      bind(C, name="rk2_hvisc_stage");      end subroutine

      subroutine rk2_continuity_stage_unopt() bind(C, name="rk2_continuity_stage_unopt"); end subroutine
      subroutine rk2_redi_stage_unopt()       bind(C, name="rk2_redi_stage_unopt");       end subroutine
      subroutine rk2_kappa_stage_unopt()      bind(C, name="rk2_kappa_stage_unopt");      end subroutine
      subroutine rk2_ale_stage_unopt()        bind(C, name="rk2_ale_stage_unopt");        end subroutine
      subroutine rk2_btstep_stage_unopt()     bind(C, name="rk2_btstep_stage_unopt");     end subroutine
      subroutine rk2_epbl_stage_unopt()       bind(C, name="rk2_epbl_stage_unopt");       end subroutine
      subroutine rk2_meke_stage_unopt()       bind(C, name="rk2_meke_stage_unopt");       end subroutine
      subroutine rk2_hvisc_stage_unopt()      bind(C, name="rk2_hvisc_stage_unopt");      end subroutine

#ifndef RK2_NO_CUDA
      subroutine rk2_continuity_stage_cuda() bind(C, name="rk2_continuity_stage_cuda"); end subroutine
      subroutine rk2_redi_stage_cuda()       bind(C, name="rk2_redi_stage_cuda");       end subroutine
      subroutine rk2_kappa_stage_cuda()      bind(C, name="rk2_kappa_stage_cuda");      end subroutine
      subroutine rk2_ale_stage_cuda()        bind(C, name="rk2_ale_stage_cuda");        end subroutine
      subroutine rk2_btstep_stage_cuda()     bind(C, name="rk2_btstep_stage_cuda");     end subroutine
      subroutine rk2_epbl_stage_cuda()       bind(C, name="rk2_epbl_stage_cuda");       end subroutine
      subroutine rk2_meke_stage_cuda()       bind(C, name="rk2_meke_stage_cuda");       end subroutine
      subroutine rk2_hvisc_stage_cuda()      bind(C, name="rk2_hvisc_stage_cuda");      end subroutine

      subroutine rk2_continuity_stage_cuda_unopt() bind(C, name="rk2_continuity_stage_cuda_unopt"); end subroutine
      subroutine rk2_redi_stage_cuda_unopt()       bind(C, name="rk2_redi_stage_cuda_unopt");       end subroutine
      subroutine rk2_kappa_stage_cuda_unopt()      bind(C, name="rk2_kappa_stage_cuda_unopt");      end subroutine
      subroutine rk2_ale_stage_cuda_unopt()        bind(C, name="rk2_ale_stage_cuda_unopt");        end subroutine
      subroutine rk2_btstep_stage_cuda_unopt()     bind(C, name="rk2_btstep_stage_cuda_unopt");     end subroutine
      subroutine rk2_epbl_stage_cuda_unopt()       bind(C, name="rk2_epbl_stage_cuda_unopt");       end subroutine
      subroutine rk2_meke_stage_cuda_unopt()       bind(C, name="rk2_meke_stage_cuda_unopt");       end subroutine
      subroutine rk2_hvisc_stage_cuda_unopt()      bind(C, name="rk2_hvisc_stage_cuda_unopt");      end subroutine

      integer(c_int) function cuda_sync() bind(C, name="cudaDeviceSynchronize")
         import :: c_int
      end function
#endif

      subroutine rk2_continuity_probe(a, b) bind(C, name="rk2_continuity_probe")
         import :: c_double; real(c_double), intent(out) :: a, b
      end subroutine
      subroutine rk2_redi_probe(a, b) bind(C, name="rk2_redi_probe")
         import :: c_double; real(c_double), intent(out) :: a, b
      end subroutine
      subroutine rk2_kappa_probe(a, b) bind(C, name="rk2_kappa_probe")
         import :: c_double; real(c_double), intent(out) :: a, b
      end subroutine
      subroutine rk2_ale_probe(a, b) bind(C, name="rk2_ale_probe")
         import :: c_double; real(c_double), intent(out) :: a, b
      end subroutine
      subroutine rk2_btstep_probe(a, b) bind(C, name="rk2_btstep_probe")
         import :: c_double; real(c_double), intent(out) :: a, b
      end subroutine
      subroutine rk2_epbl_probe(a, b) bind(C, name="rk2_epbl_probe")
         import :: c_double; real(c_double), intent(out) :: a, b
      end subroutine
      subroutine rk2_meke_probe(a, b) bind(C, name="rk2_meke_probe")
         import :: c_double; real(c_double), intent(out) :: a, b
      end subroutine
      subroutine rk2_hvisc_probe(a, b) bind(C, name="rk2_hvisc_probe")
         import :: c_double; real(c_double), intent(out) :: a, b
      end subroutine
   end interface

   integer, parameter :: NK = 8
   character(len=12), parameter :: KNAME(NK) = [character(len=12) :: &
      'continuity', 'redi', 'ale', 'hvisc', 'btstep', 'kappa', 'epbl', 'meke']

   integer :: nxp, nyp, nz, nsteps, nwarm, kk
   logical :: unopt, do_dc, do_cuda
   character(len=8) :: mode, version, target
   real(8) :: ms_dc, ms_cu, per_dc(NK), per_cu(NK)
   real(8) :: vmn(NK), vmx(NK)
   logical :: ok(NK), all_ok

   nxp = iarg(1, 473); nyp = iarg(2, 297); nz = iarg(3, 30)
   nsteps = iarg(4, 100); nwarm = iarg(5, 10)

#ifdef RK2_NO_CUDA
   target = 'cpu'
   mode = 'dc'
   version = carg(6, 'opt')      ! rk2_cpu: arg 6 is VERSION (mode is implicit dc)
#else
   target = 'gpu'
   mode = carg(6, 'both')
   version = carg(7, 'opt')
#endif
   unopt = (trim(version) == 'unopt')
   do_dc = (trim(mode) == 'dc') .or. (trim(mode) == 'both')
   do_cuda = .false.
#ifndef RK2_NO_CUDA
   do_cuda = (trim(mode) == 'cuda') .or. (trim(mode) == 'both')
#endif

   write (output_unit, '(a)') repeat('=', 78)
   write (output_unit, '(a)') '  IDEAL BENCHMARK -- whole-model RK2 harness  (8-kernel ocean core)'
   write (output_unit, '(a,i0,a,i0,a,i0)') '  domain (phys): ', nxp, ' x ', nyp, ' x ', nz
   write (output_unit, '(a,i0,a,i0,a)') '  RK2: ', nsteps, ' timed steps x 2 stages   (', nwarm, ' warm-up steps)'
   write (output_unit, '(7a)') '  target=', trim(target), '  mode=', trim(mode), '  version=', trim(version), ''
   write (output_unit, '(a)') '  stage order: continuity redi ale hvisc btstep kappa epbl meke'
   write (output_unit, '(a)') repeat('=', 78)

   call rk2_continuity_init(nxp, nyp, nz)
   call rk2_redi_init(nxp, nyp, nz)
   call rk2_ale_init(nxp, nyp, nz)
   call rk2_hvisc_init(nxp, nyp, nz)
   call rk2_btstep_init(nxp, nyp, nz)
   call rk2_kappa_init(nxp, nyp, nz)
   call rk2_epbl_init(nxp, nyp, nz)
   call rk2_meke_init(nxp, nyp, nz)
   !$acc wait
   write (output_unit, '(a)') '  init: all 8 kernels allocated + enter_data complete.'

   ms_dc = -1.0d0; ms_cu = -1.0d0; per_dc = 0.0d0; per_cu = 0.0d0

   ! ------------------------------ DC side --------------------------------
   if (do_dc) then
      call warm_dc()
      ms_dc = time_agg_dc()
      do kk = 1, NK
         per_dc(kk) = time_kernel_dc(kk)
      end do
      call probe_all(vmn, vmx, ok); all_ok = all(ok)
      call emit_result('dc', ms_dc)
      call human_table('dc', ms_dc, per_dc, vmn, vmx, ok)
   end if

#ifndef RK2_NO_CUDA
   ! ----------------------------- CUDA side -------------------------------
   if (do_cuda) then
      call warm_cuda()
      ms_cu = time_agg_cuda()
      do kk = 1, NK
         per_cu(kk) = time_kernel_cuda(kk)
      end do
      call probe_all(vmn, vmx, ok); all_ok = all(ok)
      call emit_result('cuda', ms_cu)
      call human_table('cuda', ms_cu, per_cu, vmn, vmx, ok)
   end if

   if (do_dc .and. do_cuda) then
      write (output_unit, '(a,f8.3,a)') '  ratio  DC / CUDA (per step)  version='//trim(version)//' : ', &
         ms_dc/ms_cu, ' x'
   end if
#endif
   write (output_unit, '(a)') repeat('=', 78)

contains

   ! ---- one full stage over the 8 kernels, opt or faithful, DC side ------
   subroutine one_stage_dc()
      if (unopt) then
         call rk2_continuity_stage_unopt(); call rk2_redi_stage_unopt(); call rk2_ale_stage_unopt()
         call rk2_hvisc_stage_unopt(); call rk2_btstep_stage_unopt(); call rk2_kappa_stage_unopt()
         call rk2_epbl_stage_unopt(); call rk2_meke_stage_unopt()
      else
         call rk2_continuity_stage(); call rk2_redi_stage(); call rk2_ale_stage()
         call rk2_hvisc_stage(); call rk2_btstep_stage(); call rk2_kappa_stage()
         call rk2_epbl_stage(); call rk2_meke_stage()
      end if
   end subroutine one_stage_dc

   subroutine dispatch_dc(idx)
      integer, intent(in) :: idx
      if (unopt) then
         select case (idx)
         case (1); call rk2_continuity_stage_unopt()
         case (2); call rk2_redi_stage_unopt()
         case (3); call rk2_ale_stage_unopt()
         case (4); call rk2_hvisc_stage_unopt()
         case (5); call rk2_btstep_stage_unopt()
         case (6); call rk2_kappa_stage_unopt()
         case (7); call rk2_epbl_stage_unopt()
         case (8); call rk2_meke_stage_unopt()
         end select
      else
         select case (idx)
         case (1); call rk2_continuity_stage()
         case (2); call rk2_redi_stage()
         case (3); call rk2_ale_stage()
         case (4); call rk2_hvisc_stage()
         case (5); call rk2_btstep_stage()
         case (6); call rk2_kappa_stage()
         case (7); call rk2_epbl_stage()
         case (8); call rk2_meke_stage()
         end select
      end if
   end subroutine dispatch_dc

   subroutine warm_dc()
      integer :: step, s
      do step = 1, nwarm
         do s = 1, 2
            call one_stage_dc()
         end do
      end do
      !$acc wait
   end subroutine warm_dc

   real(8) function time_agg_dc() result(msout)
      integer :: step, s
      integer(int64) :: c0, c1, crate
      call system_clock(c0, crate)
      do step = 1, nsteps
         do s = 1, 2
            call one_stage_dc()
         end do
      end do
      !$acc wait
      call system_clock(c1)
      msout = real(c1 - c0, 8)*1000.0d0/real(crate, 8)/real(nsteps, 8)
   end function time_agg_dc

   real(8) function time_kernel_dc(idx) result(msout)
      integer, intent(in) :: idx
      integer :: rep, ncall
      integer(int64) :: a0, a1, r
      ncall = nsteps*2
      do rep = 1, 2*nwarm
         call dispatch_dc(idx)
      end do
      !$acc wait
      call system_clock(a0, r)
      do rep = 1, ncall
         call dispatch_dc(idx)
      end do
      !$acc wait
      call system_clock(a1)
      msout = real(a1 - a0, 8)*1000.0d0/real(r, 8)/real(ncall, 8)
   end function time_kernel_dc

#ifndef RK2_NO_CUDA
   subroutine one_stage_cuda()
      if (unopt) then
         call rk2_continuity_stage_cuda_unopt(); call rk2_redi_stage_cuda_unopt(); call rk2_ale_stage_cuda_unopt()
         call rk2_hvisc_stage_cuda_unopt(); call rk2_btstep_stage_cuda_unopt(); call rk2_kappa_stage_cuda_unopt()
         call rk2_epbl_stage_cuda_unopt(); call rk2_meke_stage_cuda_unopt()
      else
         call rk2_continuity_stage_cuda(); call rk2_redi_stage_cuda(); call rk2_ale_stage_cuda()
         call rk2_hvisc_stage_cuda(); call rk2_btstep_stage_cuda(); call rk2_kappa_stage_cuda()
         call rk2_epbl_stage_cuda(); call rk2_meke_stage_cuda()
      end if
   end subroutine one_stage_cuda

   subroutine dispatch_cuda(idx)
      integer, intent(in) :: idx
      if (unopt) then
         select case (idx)
         case (1); call rk2_continuity_stage_cuda_unopt()
         case (2); call rk2_redi_stage_cuda_unopt()
         case (3); call rk2_ale_stage_cuda_unopt()
         case (4); call rk2_hvisc_stage_cuda_unopt()
         case (5); call rk2_btstep_stage_cuda_unopt()
         case (6); call rk2_kappa_stage_cuda_unopt()
         case (7); call rk2_epbl_stage_cuda_unopt()
         case (8); call rk2_meke_stage_cuda_unopt()
         end select
      else
         select case (idx)
         case (1); call rk2_continuity_stage_cuda()
         case (2); call rk2_redi_stage_cuda()
         case (3); call rk2_ale_stage_cuda()
         case (4); call rk2_hvisc_stage_cuda()
         case (5); call rk2_btstep_stage_cuda()
         case (6); call rk2_kappa_stage_cuda()
         case (7); call rk2_epbl_stage_cuda()
         case (8); call rk2_meke_stage_cuda()
         end select
      end if
   end subroutine dispatch_cuda

   subroutine warm_cuda()
      integer :: step, s, js
      do step = 1, nwarm
         do s = 1, 2
            call one_stage_cuda()
         end do
      end do
      js = cuda_sync()
   end subroutine warm_cuda

   real(8) function time_agg_cuda() result(msout)
      integer :: step, s, js
      integer(int64) :: c0, c1, crate
      call system_clock(c0, crate)
      do step = 1, nsteps
         do s = 1, 2
            call one_stage_cuda()
         end do
      end do
      js = cuda_sync()
      call system_clock(c1)
      msout = real(c1 - c0, 8)*1000.0d0/real(crate, 8)/real(nsteps, 8)
   end function time_agg_cuda

   real(8) function time_kernel_cuda(idx) result(msout)
      integer, intent(in) :: idx
      integer :: rep, ncall, js
      integer(int64) :: a0, a1, r
      ncall = nsteps*2
      do rep = 1, 2*nwarm
         call dispatch_cuda(idx)
      end do
      js = cuda_sync()
      call system_clock(a0, r)
      do rep = 1, ncall
         call dispatch_cuda(idx)
      end do
      js = cuda_sync()
      call system_clock(a1)
      msout = real(a1 - a0, 8)*1000.0d0/real(r, 8)/real(ncall, 8)
   end function time_kernel_cuda
#endif

   subroutine probe_all(vn, vx, okv)
      real(8), intent(out) :: vn(NK), vx(NK)
      logical, intent(out) :: okv(NK)
      integer :: j
      call rk2_continuity_probe(vn(1), vx(1))
      call rk2_redi_probe(vn(2), vx(2))
      call rk2_ale_probe(vn(3), vx(3))
      call rk2_hvisc_probe(vn(4), vx(4))
      call rk2_btstep_probe(vn(5), vx(5))
      call rk2_kappa_probe(vn(6), vx(6))
      call rk2_epbl_probe(vn(7), vx(7))
      call rk2_meke_probe(vn(8), vx(8))
      do j = 1, NK
         okv(j) = finite(vx(j)) .and. finite(vn(j)) .and. .not. (vn(j) == 0.0d0 .and. vx(j) == 0.0d0)
      end do
   end subroutine probe_all

   subroutine emit_result(mode_s, msstep)
      character(len=*), intent(in) :: mode_s
      real(8), intent(in) :: msstep
      write (output_unit, '(7a,f0.6,a,f0.6)') 'RESULT target=', trim(target), &
         ' mode=', trim(mode_s), ' version=', trim(version), &
         ' ms_per_stage=', msstep/2.0d0, ' ms_per_step=', msstep
   end subroutine emit_result

   subroutine human_table(side, msstep, per, vn, vx, okv)
      character(len=*), intent(in) :: side
      real(8), intent(in) :: msstep, per(NK), vn(NK), vx(NK)
      logical, intent(in) :: okv(NK)
      integer :: j
      write (output_unit, '(a)') repeat('-', 78)
      write (output_unit, '(3a,f10.4,a,f9.4,a)') '  [', trim(side), '] aggregate ms/RK2-step: ', &
         msstep, '   (', msstep/2.0d0, ' ms/stage)'
      write (output_unit, '(a)') '    kernel          ms/stage     min output       max output    sane'
      do j = 1, NK
         write (output_unit, '(4x,a12,f11.5,2x,es13.5,2x,es13.5,4x,l1)') &
            KNAME(j), per(j), vn(j), vx(j), okv(j)
      end do
      write (output_unit, '(a,f11.5,a)') '    sum of isolated per-kernel ms/stage: ', sum(per), ''
      if (all(okv)) then
         write (output_unit, '(3a)') '    sanity [', trim(side), ']: OK -- all 8 finite + non-zero.'
      else
         write (output_unit, '(3a)') '    sanity [', trim(side), ']: *** FAILED (see flags) ***'
      end if
   end subroutine human_table

   logical function finite(x)
      real(8), intent(in) :: x
      finite = (x == x) .and. (abs(x) <= huge(1.0d0))
   end function finite

   integer function iarg(k, dflt)
      integer, intent(in) :: k, dflt
      character(len=32) :: buf
      integer :: ln, st
      iarg = dflt
      if (command_argument_count() < k) return
      call get_command_argument(k, buf, ln, st)
      if (st /= 0 .or. ln == 0) return
      read (buf, *, iostat=st) iarg
      if (st /= 0) iarg = dflt
   end function iarg

   function carg(k, dflt) result(s)
      integer, intent(in) :: k
      character(len=*), intent(in) :: dflt
      character(len=8) :: s
      integer :: ln, st
      s = dflt
      if (command_argument_count() < k) return
      call get_command_argument(k, s, ln, st)
      if (st /= 0 .or. ln == 0) s = dflt
   end function carg

end program rk2_main
