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

# SD_TAG: optional per-configuration suffix so a kernel that exposes
# compile-time sweep axes (kappa_shear: NZSTACK/VARIANT/VLEN/...) gets one build
# dir + binary per configuration instead of clobbering the previous one. EMPTY
# by default -> the historical `sdo` / `build/serialdo` names are unchanged.
SD_TAG    ?=
SD_BASES  := $(notdir $(DC_SRCS))
SDBLD     := build/serialdo$(SD_TAG)
SDBIN     := sdo$(SD_TAG)
SDOBJS    := $(patsubst %.F90,$(SDBLD)/%.o,$(SD_BASES))
SD_FFLAGS := $(BASE_FFLAGS)          # base flags only: serial, no device layer

# resolved source for a basename: the committed serialdo/ copy wins, else the
# ordinary DC source (top-level, then drivers/).
sd_src = $(firstword $(wildcard serialdo/$(1)) $(wildcard $(1)) $(wildcard drivers/$(1)))

$(SDBLD):
	@mkdir -p $@

# Stamp the resolved compiler + flags into the build dir and make every object
# depend on it. `make` keys on TIMESTAMPS, so without this a second build with a
# different FC reuses the first one's objects -- and, worse, its .mod files:
#   make serialdo               # nvfortran, writes build/serialdo/constants.mod
#   make serialdo FC=gfortran   # "constants.mod ... is not a GNU Fortran module"
# which is the FRIENDLY failure. The quiet one is two compilers' objects linked
# into one binary that runs and reports a number. The stamp is only touched when
# its contents change, so a repeated identical build still does nothing.
$(SDBLD)/.flags: ; @true
$(shell mkdir -p $(SDBLD) 2>/dev/null; printf '%s\n' '$(FC) $(SD_FFLAGS)' > $(SDBLD)/.flags.new; \
        cmp -s $(SDBLD)/.flags.new $(SDBLD)/.flags 2>/dev/null || mv -f $(SDBLD)/.flags.new $(SDBLD)/.flags; \
        rm -f $(SDBLD)/.flags.new)

# one explicit rule per source so the right file is a real prerequisite
# (rebuilds when either the serialdo/ copy or the header changes).
define SD_RULE
$(SDBLD)/$(basename $(1)).o: $(call sd_src,$(1)) ../common/directives.h $(SDBLD)/.flags | $(SDBLD)
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
# SD_REF_DATA picks WHICH do-concurrent build produces the reference. `acc` (the
# GPU) is the interesting cross-check where nvfortran exists; `none` is the
# CPU do-concurrent build, which is what a machine with only gfortran/ifx has --
# and it still proves the point, since the claim under test is that the loop
# rewrite is numerically inert, not that a GPU is involved:
#   make verify-serialdo FC=gfortran-15 SD_REF_DATA=none
SD_REF_DATA ?= acc
SD_REF_BIN ?= dc_$(SD_REF_DATA)$(SD_TAG)
verify-serialdo:
	@$(MAKE) --no-print-directory dc DATA=$(SD_REF_DATA)
	@DC_DUMP=serialdo_ref.bin ./$(SD_REF_BIN) $(SD_ARGS)
	@$(MAKE) --no-print-directory serialdo
	@DC_REF=serialdo_ref.bin ./$(SDBIN) $(SD_ARGS)

clean-serialdo:
	@rm -rf $(SDBLD) $(SDBIN) serialdo_ref.bin

# introspection: `make print-SD_FFLAGS FC=ifx` echoes the resolved value, so the
# run harness can record the EXACT compiler flags in its provenance CSV.
print-%:
	@echo '$($*)'
