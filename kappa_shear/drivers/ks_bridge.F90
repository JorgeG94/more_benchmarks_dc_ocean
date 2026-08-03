!! Fortran -> CUDA bridge for the kappa-shear column kernels.
!!
!! WHY THIS EXISTS: the hand-written CUDA kernels used to be reachable only
!! through their own C++ `main()` (drivers/cpp_main.cu), which meant a second
!! harness with a HAND-WRITTEN MIRROR of build_state in C++. That mirror is a
!! liability -- if it drifts even slightly the CUDA timing measures a different
!! problem, silently, and a wall-clock number cannot tell you. It also walks
!! straight into the cross-language libm trap documented in CLAUDE.md
!! (nvfortran's sin/exp and glibc's disagree in the last ulp), which is why
!! cpp_main.cu carries ~150 lines of [RISK A]/[RISK B] verification machinery
!! just to establish that its inputs are the same inputs.
!!
!! With this bridge, ONE Fortran driver builds the state once and launches BOTH
!! toolchains on the SAME device allocation via `!$acc host_data use_device`.
!! There is no mirror to be wrong, no libm question, and no second binary: the
!! only variable left is who generated the code, which is the thing being
!! measured. (Lifted from legacy_testing/kappa_shear_benchmark/ks_bench.F90,
!! where this pattern was already proven.)
!!
!! ⚠ LINKING: `-cuda` is LINK-ONLY. It also switches nvfortran into CUDA-Fortran
!! mode, which type-checks device attributes and rejects a host_data bridge that
!! declares plain host arrays -- COMPILING with it fails with
!! NVFORTRAN-S-0528. See the kernel Makefile's CMP_LDFLAGS.
module ks_bridge
   use, intrinsic :: iso_c_binding, only: c_double, c_int
   implicit none
   public

   !! Mirrors `struct KsPar` in ks_kernel.cu / opt_kernel.cu. bind(C) so the
   !! layout is the compiler's problem, not ours. A drift here would feed the
   !! kernel garbage knobs and it would still happily run, so all three
   !! declarations (here, ks_kernel.cu, opt_kernel.cu) must stay in step.
   type, bind(C) :: ks_par_t
      real(c_double) :: dt, ri_crit, shearmix_rate, fri_curvature
      real(c_double) :: c_n, c_s, lambda, lz_rescale
      real(c_double) :: kappa_0, kappa_seed, kappa_trunc, tke_bg
      real(c_double) :: tol_err, src_max_chg, vel_underflow, rho0
      integer(c_int) :: max_inner_it, max_substep_it
      integer(c_int) :: eos_variant
      real(c_double) :: eos_rho0, eos_alpha_T, eos_beta_S
   end type ks_par_t

   interface
      !! The faithful port (ks_kernel.cu). `sync /= 0` makes the launcher
      !! cudaDeviceSynchronize before returning.
      subroutine ks_cuda_launch(h, u, v, hT, hS, wet, fc, kd, tke, n_out, n_in, &
                                nx, ny, nz, p, sync) bind(C, name="ks_cuda_launch")
         import :: c_double, c_int, ks_par_t
         implicit none
         real(c_double), intent(in) :: h(*), u(*), v(*), hT(*), hS(*)
         real(c_double), intent(in) :: wet(*), fc(*)
         real(c_double), intent(inout) :: kd(*), tke(*)
         integer(c_int), intent(inout) :: n_out(*), n_in(*)
         integer(c_int), value :: nx, ny, nz, sync
         type(ks_par_t), intent(in) :: p
      end subroutine

      subroutine ks_cuda_attrs(nregs, lmem, smem, maxtpb) bind(C, name="ks_cuda_attrs")
         import :: c_int
         implicit none
         integer(c_int), intent(out) :: nregs, lmem, smem, maxtpb
      end subroutine

      !! Barrier only -- lets the driver time N asynchronous launches followed by
      !! ONE synchronise, which is the same window the `do concurrent` side is
      !! timed over. Without it the choice is a sync per rep (which taxes small
      !! grids) or an extra launch (which biases the rep count).
      subroutine ks_cuda_sync() bind(C, name="ks_cuda_sync")
      end subroutine

      !! The optimised port (opt_kernel.cu). Same signature by construction; it
      !! REFUSES to launch when nz+1 > KS_OPT_NZMAX rather than corrupt results,
      !! so the driver must check that bound before trusting a timing.
      subroutine ks_opt_launch(h, u, v, hT, hS, wet, fc, kd, tke, n_out, n_in, &
                               nx, ny, nz, p, sync) bind(C, name="ks_opt_launch")
         import :: c_double, c_int, ks_par_t
         implicit none
         real(c_double), intent(in) :: h(*), u(*), v(*), hT(*), hS(*)
         real(c_double), intent(in) :: wet(*), fc(*)
         real(c_double), intent(inout) :: kd(*), tke(*)
         integer(c_int), intent(inout) :: n_out(*), n_in(*)
         integer(c_int), value :: nx, ny, nz, sync
         type(ks_par_t), intent(in) :: p
      end subroutine

      subroutine ks_opt_attrs(nregs, lmem, smem, maxtpb) bind(C, name="ks_opt_attrs")
         import :: c_int
         implicit none
         integer(c_int), intent(out) :: nregs, lmem, smem, maxtpb
      end subroutine
   end interface

end module ks_bridge
