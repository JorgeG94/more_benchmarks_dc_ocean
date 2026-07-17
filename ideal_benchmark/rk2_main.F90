!! ideal_benchmark / rk2_main.F90  -- the whole-model RK2 harness.
!!
!! A "dumb" RK2 (2-stage) time-stepping harness that exercises ALL 9 build-mode
!! kernels back-to-back per stage, keeping the GPU hot, to measure the aggregate
!! whole-model cost. Physical correctness does NOT matter: each kernel runs on
!! its own resident device state, driven in lockstep.
!!
!! TWO PASSES on the SAME device state:
!!   Phase 1  all-DC   : each stage calls the kernel's opt-DC `do concurrent`.
!!   Phase 2  all-CUDA : each stage calls the kernel's opt-CUDA launcher via
!!                       `!$acc host_data use_device` (same device arrays).
!! Per-kernel bit-identity DC==CUDA is already proven by each kernel's `cmp`
!! build, so this harness only TIMES the two paths and checks finite+non-zero.
!! The CUDA pass runs on the state the DC pass left behind (fine for timing).
!!
!! Order of the 9 stage calls:
!!   continuity, redi, ale, hvisc, btstep, kappa, epbl, meke, hll
!!
!! Usage: ./rk2 [nx_phys] [ny_phys] [nz] [nsteps] [nwarm]
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
      subroutine rk2_hll_init(nx, ny, nz) bind(C, name="rk2_hll_init")
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
      subroutine rk2_hll_stage()        bind(C, name="rk2_hll_stage");        end subroutine
      subroutine rk2_hvisc_stage()      bind(C, name="rk2_hvisc_stage");      end subroutine

      subroutine rk2_continuity_stage_cuda() bind(C, name="rk2_continuity_stage_cuda"); end subroutine
      subroutine rk2_redi_stage_cuda()       bind(C, name="rk2_redi_stage_cuda");       end subroutine
      subroutine rk2_kappa_stage_cuda()      bind(C, name="rk2_kappa_stage_cuda");      end subroutine
      subroutine rk2_ale_stage_cuda()        bind(C, name="rk2_ale_stage_cuda");        end subroutine
      subroutine rk2_btstep_stage_cuda()     bind(C, name="rk2_btstep_stage_cuda");     end subroutine
      subroutine rk2_epbl_stage_cuda()       bind(C, name="rk2_epbl_stage_cuda");       end subroutine
      subroutine rk2_meke_stage_cuda()       bind(C, name="rk2_meke_stage_cuda");       end subroutine
      subroutine rk2_hll_stage_cuda()        bind(C, name="rk2_hll_stage_cuda");        end subroutine
      subroutine rk2_hvisc_stage_cuda()      bind(C, name="rk2_hvisc_stage_cuda");      end subroutine

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
      subroutine rk2_hll_probe(a, b) bind(C, name="rk2_hll_probe")
         import :: c_double; real(c_double), intent(out) :: a, b
      end subroutine
      subroutine rk2_hvisc_probe(a, b) bind(C, name="rk2_hvisc_probe")
         import :: c_double; real(c_double), intent(out) :: a, b
      end subroutine

      integer(c_int) function cuda_sync() bind(C, name="cudaDeviceSynchronize")
         import :: c_int
      end function
   end interface

   integer, parameter :: NK = 8
   character(len=12), parameter :: KNAME(NK) = [character(len=12) :: &
      'continuity', 'redi', 'ale', 'hvisc', 'btstep', 'kappa', 'epbl', 'meke']

   integer :: nxp, nyp, nz, nsteps, nwarm, step, s, kk, istat
   integer(int64) :: c0, c1, crate
   real(8) :: ms_dc, ms_cu
   real(8) :: per_dc(NK), per_cu(NK)
   real(8) :: vmn_dc(NK), vmx_dc(NK), vmn_cu(NK), vmx_cu(NK)
   logical :: ok_dc(NK), ok_cu(NK)

   nxp = iarg(1, 473); nyp = iarg(2, 297); nz = iarg(3, 30)
   nsteps = iarg(4, 100); nwarm = iarg(5, 10)

   write (output_unit, '(a)') repeat('=', 78)
   write (output_unit, '(a)') '  IDEAL BENCHMARK -- whole-model RK2 harness  (all-DC vs all-CUDA)'
   write (output_unit, '(a,i0,a,i0,a,i0)') '  domain (phys): ', nxp, ' x ', nyp, ' x ', nz
   write (output_unit, '(a,i0,a,i0,a)') '  RK2: ', nsteps, ' timed steps x 2 stages   (', nwarm, ' warm-up steps)'
   write (output_unit, '(a)') '  stage order: continuity redi ale hvisc btstep kappa epbl meke  (ocean core; hll excluded)'
   write (output_unit, '(a)') repeat('=', 78)

   ! ---- init every kernel's resident device state --------------------------
   call rk2_continuity_init(nxp, nyp, nz)
   call rk2_redi_init(nxp, nyp, nz)
   call rk2_ale_init(nxp, nyp, nz)
   call rk2_hvisc_init(nxp, nyp, nz)
   call rk2_btstep_init(nxp, nyp, nz)
   call rk2_kappa_init(nxp, nyp, nz)
   call rk2_epbl_init(nxp, nyp, nz)
   call rk2_meke_init(nxp, nyp, nz)
   !$acc wait
   write (output_unit, '(a)') '  init: all 8 ocean kernels allocated + enter_data complete.'

   ! ============================ PHASE 1: all-DC ============================
   do step = 1, nwarm
      do s = 1, 2
         call one_stage_dc()
      end do
   end do
   !$acc wait
   call system_clock(c0, crate)
   do step = 1, nsteps
      do s = 1, 2
         call one_stage_dc()
      end do
   end do
   !$acc wait
   call system_clock(c1)
   ms_dc = real(c1 - c0, 8)*1000.0d0/real(crate, 8)/real(nsteps, 8)

   call time_kernel_dc(1, per_dc(1)); call time_kernel_dc(2, per_dc(2))
   call time_kernel_dc(3, per_dc(3)); call time_kernel_dc(4, per_dc(4))
   call time_kernel_dc(5, per_dc(5)); call time_kernel_dc(6, per_dc(6))
   call time_kernel_dc(7, per_dc(7)); call time_kernel_dc(8, per_dc(8))
   call probe_all(vmn_dc, vmx_dc, ok_dc)

   ! =========================== PHASE 2: all-CUDA ==========================
   do step = 1, nwarm
      do s = 1, 2
         call one_stage_cuda()
      end do
   end do
   istat = cuda_sync()
   call system_clock(c0, crate)
   do step = 1, nsteps
      do s = 1, 2
         call one_stage_cuda()
      end do
   end do
   istat = cuda_sync()
   call system_clock(c1)
   ms_cu = real(c1 - c0, 8)*1000.0d0/real(crate, 8)/real(nsteps, 8)

   call time_kernel_cuda(1, per_cu(1)); call time_kernel_cuda(2, per_cu(2))
   call time_kernel_cuda(3, per_cu(3)); call time_kernel_cuda(4, per_cu(4))
   call time_kernel_cuda(5, per_cu(5)); call time_kernel_cuda(6, per_cu(6))
   call time_kernel_cuda(7, per_cu(7)); call time_kernel_cuda(8, per_cu(8))
   call probe_all(vmn_cu, vmx_cu, ok_cu)

   ! ================================ REPORT ================================
   write (output_unit, '(a)') ''
   write (output_unit, '(a)') repeat('=', 78)
   write (output_unit, '(a)') '  HEADLINE  (ms / RK2-step = 2 stages, 8 ocean kernels; hll excluded)'
   write (output_unit, '(a,f10.4,a,f9.4,a)') '    all-DC   : ', ms_dc, ' ms/step   (', ms_dc/2.0d0, ' ms/stage)'
   write (output_unit, '(a,f10.4,a,f9.4,a)') '    all-CUDA : ', ms_cu, ' ms/step   (', ms_cu/2.0d0, ' ms/stage)'
   write (output_unit, '(a,f8.3,a)') '    ratio  DC / CUDA (per step)              : ', ms_dc/ms_cu, ' x'
   write (output_unit, '(a)') repeat('-', 78)
   write (output_unit, '(a)') '  per-kernel ms/stage (isolated loop):'
   write (output_unit, '(a)') '    kernel          DC ms      CUDA ms     DC/CUDA   sane(DC,CUDA)'
   do kk = 1, NK
      write (output_unit, '(4x,a12,f10.5,2x,f10.5,3x,f8.3,6x,l1,1x,l1)') &
         KNAME(kk), per_dc(kk), per_cu(kk), safe_ratio(per_dc(kk), per_cu(kk)), ok_dc(kk), ok_cu(kk)
   end do
   write (output_unit, '(4x,a12,f10.5,2x,f10.5,3x,f8.3)') 'SUM', sum(per_dc), sum(per_cu), &
      safe_ratio(sum(per_dc), sum(per_cu))
   write (output_unit, '(a)') repeat('-', 78)
   write (output_unit, '(a)') '  sanity min/max (DC pass, then CUDA pass -- same field per kernel):'
   do kk = 1, NK
      write (output_unit, '(4x,a12,a,es12.4,a,es12.4,a,es12.4,a,es12.4,a)') KNAME(kk), &
         '  DC[', vmn_dc(kk), ',', vmx_dc(kk), ']  CU[', vmn_cu(kk), ',', vmx_cu(kk), ']'
   end do
   if (all(ok_dc) .and. all(ok_cu)) then
      write (output_unit, '(a)') '  SANITY: OK -- all 8 outputs finite + non-zero on BOTH passes.'
   else
      write (output_unit, '(a)') '  SANITY: *** FAILED (see flags above) ***'
   end if
   write (output_unit, '(a)') '  note: the CUDA pass runs on the device state the DC pass left behind;'
   write (output_unit, '(a)') '        fine for timing (per-kernel bit-identity is proven by each cmp build).'
   write (output_unit, '(a)') repeat('=', 78)

contains

   subroutine one_stage_dc()
      call rk2_continuity_stage(); call rk2_redi_stage(); call rk2_ale_stage()
      call rk2_hvisc_stage(); call rk2_btstep_stage(); call rk2_kappa_stage()
      call rk2_epbl_stage(); call rk2_meke_stage()
   end subroutine one_stage_dc

   subroutine one_stage_cuda()
      call rk2_continuity_stage_cuda(); call rk2_redi_stage_cuda(); call rk2_ale_stage_cuda()
      call rk2_hvisc_stage_cuda(); call rk2_btstep_stage_cuda(); call rk2_kappa_stage_cuda()
      call rk2_epbl_stage_cuda(); call rk2_meke_stage_cuda()
   end subroutine one_stage_cuda

   subroutine dispatch_dc(idx)
      integer, intent(in) :: idx
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
   end subroutine dispatch_dc

   subroutine dispatch_cuda(idx)
      integer, intent(in) :: idx
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
   end subroutine dispatch_cuda

   subroutine time_kernel_dc(idx, ms_out)
      integer, intent(in) :: idx
      real(8), intent(out) :: ms_out
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
      ms_out = real(a1 - a0, 8)*1000.0d0/real(r, 8)/real(ncall, 8)
   end subroutine time_kernel_dc

   subroutine time_kernel_cuda(idx, ms_out)
      integer, intent(in) :: idx
      real(8), intent(out) :: ms_out
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
      ms_out = real(a1 - a0, 8)*1000.0d0/real(r, 8)/real(ncall, 8)
   end subroutine time_kernel_cuda

   subroutine probe_all(vmn, vmx, ok)
      real(8), intent(out) :: vmn(NK), vmx(NK)
      logical, intent(out) :: ok(NK)
      integer :: j
      call rk2_continuity_probe(vmn(1), vmx(1))
      call rk2_redi_probe(vmn(2), vmx(2))
      call rk2_ale_probe(vmn(3), vmx(3))
      call rk2_hvisc_probe(vmn(4), vmx(4))
      call rk2_btstep_probe(vmn(5), vmx(5))
      call rk2_kappa_probe(vmn(6), vmx(6))
      call rk2_epbl_probe(vmn(7), vmx(7))
      call rk2_meke_probe(vmn(8), vmx(8))
      do j = 1, NK
         ok(j) = finite(vmx(j)) .and. finite(vmn(j)) .and. .not. (vmn(j) == 0.0d0 .and. vmx(j) == 0.0d0)
      end do
   end subroutine probe_all

   real(8) function safe_ratio(a, b)
      real(8), intent(in) :: a, b
      if (b > 0.0d0) then
         safe_ratio = a/b
      else
         safe_ratio = 0.0d0
      end if
   end function safe_ratio

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

end program rk2_main
