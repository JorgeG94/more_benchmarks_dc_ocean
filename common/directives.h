/* continuity_layered/directives.h
 *
 * Backend-toggled data-mapping + device-routine directives for the Fortran
 * `do concurrent` side. The COMPUTE is always a bare `do concurrent`; only the
 * lines below change per backend, so ONE source builds for NVIDIA (OpenACC or
 * OpenMP-target) and for the CPU (no data layer at all).
 *
 * Selected by a -D on the compile line (see the Makefile / ../config.mk):
 *   -DDC_DATA_ACC   OpenACC data  (nvfortran -acc=gpu)          — the measured path
 *   -DDC_DATA_OMP   OpenMP target (nvfortran/ifx/amdflang -mp)  — AMD/Intel portable
 *   (neither)       host: bare DC on the CPU (-stdpar=multicore / serial)
 *
 * A macro that expands to `!$acc ...` / `!$omp ...` lands verbatim on its own
 * source line; the sentinel is honoured when the matching flag (-acc / -mp) is
 * on and ignored otherwise. This mirrors ../dc_patterns/common/dcp_directives.h.
 */
#ifndef DC_DIRECTIVES_H
#define DC_DIRECTIVES_H

#if defined(DC_DATA_ACC)
#  define DC_DATA_NAME "acc"
#  define DC_ROUTINE_SEQ !$acc routine seq
#  define DC_ENTER_IN(x) !$acc enter data copyin(x)
#  define DC_ENTER_CREATE(x) !$acc enter data create(x)
#  define DC_EXIT(x) !$acc exit data delete(x)
#  define DC_UPDATE_SELF(x) !$acc update self(x)
#  define DC_WAIT !$acc wait

#elif defined(DC_DATA_OMP)
#  define DC_DATA_NAME "omp"
#  define DC_ROUTINE_SEQ !$omp declare target
#  define DC_ENTER_IN(x) !$omp target enter data map(to: x)
#  define DC_ENTER_CREATE(x) !$omp target enter data map(alloc: x)
#  define DC_EXIT(x) !$omp target exit data map(delete: x)
#  define DC_UPDATE_SELF(x) !$omp target update from(x)
#  define DC_WAIT !$omp taskwait

#else
#  define DC_DATA_NAME "host"
#  define DC_ROUTINE_SEQ
#  define DC_ENTER_IN(x)
#  define DC_ENTER_CREATE(x)
#  define DC_EXIT(x)
#  define DC_UPDATE_SELF(x)
#  define DC_WAIT
#endif

#endif /* DC_DIRECTIVES_H */
