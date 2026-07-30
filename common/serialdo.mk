# common/serialdo.mk -- the "serial do-loop" build MODE, shared by every kernel.
#
# Included near the end of each kernel Makefile (after DC_SRCS + BASE_FFLAGS are
# defined). Compiles the COMMITTED serialdo/ sources -- plain nested `do` loops,
# generated from the `do concurrent` sources by tools/dc_to_serialdo.py and
# refreshed with tools/gen_serialdo.sh. This is the paper's SERIAL DO baseline,
# distinct from "do concurrent, serial": same numerics, ordinary loops.
#
# It is compiled with NO -stdpar / -acc / -mp and NO DC_DATA_* macro, so it is
# serial and portable to ANY F2008+ compiler -- including gfortran, which cannot
# compile the `do concurrent ... local(...)` source at all. Override the
# compiler/flags the usual way: `make serialdo FC=gfortran`, or
# `make serialdo FC=ifx FFLAGS_BASE='-O3 -xHost'`.
#
# Needs from the kernel Makefile: DC_SRCS, BASE_FFLAGS.  From ../config.mk:
# FC, MODFLAG.  The serialdo/ dir holds ONLY the files that differ from their DC
# original (those with `do concurrent`); every other source compiles in place.
#
# Binary is `sdo` (NOT `serialdo`) so `clean` never collides with the serialdo/
# SOURCE directory.

SD_BASES  := $(notdir $(DC_SRCS))
SDBLD     := build/serialdo
SDBIN     := sdo
SDOBJS    := $(patsubst %.F90,$(SDBLD)/%.o,$(SD_BASES))
SD_FFLAGS := $(BASE_FFLAGS)          # base flags only: serial, no device layer

# resolved source for a basename: the committed serialdo/ copy wins, else the
# ordinary DC source (top-level, then drivers/).
sd_src = $(firstword $(wildcard serialdo/$(1)) $(wildcard $(1)) $(wildcard drivers/$(1)))

$(SDBLD):
	@mkdir -p $@

# one explicit rule per source so the right file is a real prerequisite
# (rebuilds when either the serialdo/ copy or the header changes).
define SD_RULE
$(SDBLD)/$(basename $(1)).o: $(call sd_src,$(1)) ../common/directives.h | $(SDBLD)
	$$(FC) $$(SD_FFLAGS) $$(MODFLAG) $(SDBLD) -c $$< -o $$@
endef
$(foreach s,$(SD_BASES),$(eval $(call SD_RULE,$(s))))

$(SDBIN): $(SDOBJS)
	$(FC) $(SD_FFLAGS) $(SDOBJS) -o $@

.PHONY: serialdo run-serialdo verify-serialdo clean-serialdo
serialdo: $(SDBIN)

SD_ARGS ?= $(ARGS)
run-serialdo: serialdo
	@./$(SDBIN) $(SD_ARGS)

# bit-identity gate (runs where the DC source compiles -- e.g. nvfortran): the
# do-concurrent build dumps a reference field, the serial-do build cross-checks
# it (max rel diff < 1e-12). Proves the loop rewrite is numerically inert.
verify-serialdo:
	@$(MAKE) --no-print-directory dc DATA=acc
	@DC_DUMP=serialdo_ref.bin ./dc_acc $(SD_ARGS)
	@$(MAKE) --no-print-directory serialdo
	@DC_REF=serialdo_ref.bin ./$(SDBIN) $(SD_ARGS)

clean-serialdo:
	@rm -rf $(SDBLD) $(SDBIN) serialdo_ref.bin

# introspection: `make print-SD_FFLAGS FC=ifx` echoes the resolved value, so the
# run harness can record the EXACT compiler flags in its provenance CSV.
print-%:
	@echo '$($*)'
