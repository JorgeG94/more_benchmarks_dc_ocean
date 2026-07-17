# benchmark_config.mk -- the ONE place the paper's problem size lives.
#
# Every ./build_and_run_*.sh reads these five numbers (via benchmark_common.sh).
# There are NO size arguments on the scripts: to change the problem, edit here.
#
# Default is REDUCED for fast iteration (serial CPU column solvers stay quick).
# For the paper's final numbers, set the production 0.1-deg size:
#     NXP=473  NYP=297  NZ=30  REPS=200  WARM=10
#
# NOTE: keep these as bare `VAR ?= value` with NO trailing inline `# comment` --
# the spaces before a `#` land inside the value and corrupt the arg string
# (same trap documented for config.mk in CLAUDE.md).
#
# NXP,NYP = interior cells (ghosts added by each kernel). NZ = layers.
# REPS = timed reps / RK2 steps. WARM = warm-up reps.

NXP  ?= 160
NYP  ?= 100
NZ   ?= 30
REPS ?= 50
WARM ?= 10
