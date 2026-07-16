# Optimizing the kappa-shear CUDA column kernel

A best-CUDA-practices pass over `ks_kernel.cu` (the faithful monolithic
one-thread-per-column JHL08 solver), held **bit-identical** to it — `ab_main.cu`
diffs the FULL `kd_int` field (not just its sum, which is a reduction that can
hide sign-paired / permuted differences). V100, 473×297×30.

## Result: register- and divergence-bound, ~0% headroom

**No meaningful speedup exists for this kernel, and the profile says exactly
why.** The best optimized build is bit-identical and **0.998×** (i.e. a wash).
Every attempt to trade registers for occupancy made it *worse*. This is a valid,
evidenced result: the faithful kernel is already at the algorithm's ceiling for
the one-thread-per-column strategy.

| build | ms/rep (opt) | vs faithful | regs | local B/thr | kd_int |
|---|---|---|---|---|---|
| faithful (baseline) | 34.86 | 1.000× | 80 | 66512 | — |
| **opt (default: NZMAX=48, IDX32)** | **34.93** | **0.998×** | 80 | 25536 | **bit-identical** |
| opt + `KS_MINBLOCKS=6` | 35.10 | 0.991× | 80 | 25504 | bit-identical |
| opt + `KS_MINBLOCKS=8` | 35.95 | 0.971× | 64 | 25568 | bit-identical |
| opt + `KS_MINBLOCKS=10` | 37.83 | 0.924× | 48 | 25632 | bit-identical |
| opt + `KS_MINBLOCKS=12` | 40.40 | 0.860× | 40 | 25680 | bit-identical |
| opt + `KS_OPT_NZMAX=32` | 34.90 | 0.996× | 80 | 17344 | bit-identical |

All builds: **0 of 4,499,247 `kd_int` cells differ in any bit**;
`sum kd_int = 3.299881663e+05` matches the faithful run and ks_bench's DC figure.

## What the optimized kernel changes (all bit-identical, all free)

1. **Frame sized to actual depth, not the production max (`KS_OPT_NZMAX=48`).**
   The faithful kernel declares every per-column array to `NZL =
   MODEL_NZ_STACK_MAX = 128` and indexes `[1..nz+1]`, so the compiler reserves
   129 doubles × ~55 arrays = **66,512 B/thread** even though the benchmark only
   ever touches indices 1..31 (`nz=30`). Sizing the frame to `KS_OPT_NZMAX`
   drops the reservation to **25,536 B/thread** (17,344 at `NZMAX=32`). Elements
   above `nz+1` are provably never read (see `ks_kernel.cu`'s header), so results
   are identical for any `nz+1 ≤ KS_OPT_NZMAX`; the launch guards this and
   refuses a deeper column rather than corrupt it.
2. **32-bit global indexing (`IDX32`).** `nx*ny*nz = 4.4e6 << 2³¹`, so the
   `size_t` address math on the column load/store is pure overhead.

Both are genuine waste removal and both are correct — but **neither moves the
clock**, because neither was the bottleneck (below).

## Why there is no headroom — the ncu evidence

`ncu` on both kernels (96×96×30, one launch each) is essentially identical:

| metric | faithful | optimized |
|---|---|---|
| registers/thread | 80 | 80 |
| occupancy limit (registers) | 6 blocks | 6 blocks |
| **achieved active warps** | **6.19 %** | **6.20 %** |
| **IPC (inst/cycle active)** | **0.03** | **0.03** |
| SM throughput | 3.06 % | 3.07 % |
| DRAM throughput | 35.2 % | 35.2 % |
| L1 throughput | 15.5 % | 15.6 % |
| local ld / st sectors | 41.2M / 22.5M | 41.1M / 22.5M |

Reading this:

- **Nothing is saturated.** DRAM 35 %, L1 15 %, SM 3 %. The kernel is neither
  memory-bound nor compute-bound — it is **latency-bound**. IPC of 0.03 (out of
  ~4 possible) means the SM sits idle ~99 % of cycles waiting on long-latency
  local-memory dependency chains.
- **Occupancy is the lever, and it is jammed.** Registers cap the theoretical
  occupancy at 6 blocks/SM (37.5 %), and *achieved* occupancy is only **6.2 %** —
  far below even that cap. The gap is **warp divergence**: this is a sequential
  Picard iterative solver (`max_inner_it=50`) inside an adaptive substepper
  (`max_substep_it=13`), both with **data-dependent trip counts**. A warp cannot
  retire until its slowest of 32 columns converges, so warps stall and drain,
  and the SM cannot even fill to its register-limited occupancy.
- **So the two ways to hide latency are both closed.** (a) Raise occupancy by
  capping registers → the `KS_MINBLOCKS` sweep does exactly this (80→64→48→40
  regs) and it is **monotonically worse** (0.99×→0.86×): the ~26 KB/thread of
  live spill state just spills harder, and the added local traffic costs more
  than the extra warps buy. (b) More work per thread / warp-per-column → **does
  not apply**: the recursion is sequential (each layer depends on the one above),
  so there is no independent parallelism inside a column to map onto a warp.

The frame-size reduction confirms the diagnosis: shrinking the *reserved* frame
3.8× (66,512→17,344 B) changes performance by <0.5%, because the reserved-but-
untouched tail never generated any traffic in the first place. The binding
resource is the **latency of the touched local-memory dependency chain under
data-dependent divergence**, and that is a property of the algorithm, not of the
codegen.

## Verdict

The faithful CUDA kernel is already optimal for the one-thread-per-column
strategy. The optimized kernel is offered as a strictly-not-worse, bit-identical
cleanup (smaller reserved frame, 32-bit addressing) with headroom knobs wired
for the record; it does not and cannot beat the faithful kernel meaningfully on
this V100 because the workload is latency/divergence-bound at 6% occupancy and
0.03 IPC, with every occupancy lever either capped by registers or defeated by
the sequential recursion. Real speedup would require a different algorithm
(e.g. batching columns by iteration count to cut divergence), which is out of
scope for a codegen-faithful port comparison.

## Reproduce

```bash
nvcc ab_main.cu opt_kernel.cu ks_kernel.cu -O3 -arch=sm_70 -I../common \
     -DMODEL_NZ_STACK_MAX=128 -o ab
./ab 473 297 30 20                 # correctness (full-field diff) + timing
# occupancy/register sweep (all bit-identical, all >= as slow):
nvcc ... -DKS_MINBLOCKS=8 -o ab_lb8    # launch_bounds(128,8) -> 64 regs
nvcc ... -DKS_OPT_NZMAX=32 -o ab_nz32  # tighter frame (nz+1 <= 32)
# limiter evidence:
ncu --launch-count 2 --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__inst_executed.avg.per_cycle_active ./ab 96 96 30 1
```

All GPU runs go through `tmp_local_artifacts/gpu_run.sh` (shared-V100 lock).
