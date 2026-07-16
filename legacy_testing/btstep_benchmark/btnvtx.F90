!! NVTX range helper — same pattern as production's profiler.F90:49-60
!! (`use nvtx, only: nvtxStartRange, nvtxEndRange`, gated on a CPP flag).
!!
!! Build with `make NVTX=1` (-DUSE_NVTX -cudalib=nvtx); without it every call
!! here compiles to nothing, so the timed path is byte-identical to the
!! un-instrumented build. That matters: NVTX ranges around an ASYNC region
!! mark HOST time, so they must not be in the build we quote numbers from.
!!
!! Ranges live in btstep_bench.F90 ONLY -- one per timed loop, so each variant
!! is a single contiguous band in the visual profiler. Do NOT push ranges
!! inside the substep: per-phase ranges nest 11-deep per substep x 24 substeps
!! and make the timeline unreadable.
module btnvtx
#ifdef USE_NVTX
   use nvtx, only: nvtxStartRange, nvtxEndRange
#endif
   implicit none
   private
   public :: nvtx_push, nvtx_pop

contains

   subroutine nvtx_push(name)
      character(len=*), intent(in) :: name
#ifdef USE_NVTX
      call nvtxStartRange(name)
#endif
   end subroutine nvtx_push

   subroutine nvtx_pop()
#ifdef USE_NVTX
      call nvtxEndRange()
#endif
   end subroutine nvtx_pop

end module btnvtx
