# Optimizing the ALE remap CUDA kernel

An optimized launcher (`opt_kernel.cu`, `ale_remap_opt`) over the **faithful**
port (`ale_kernel.cu`, `ale_remap_cuda(..., fused=0)` — one CUDA kernel per
production do-concurrent loop). Held **bit-identical** to the faithful path:
`ab_main.cu` checks `max|diff| == 0` on every output array the two paths share.
V100-PCIE-32GB, nvcc 12.9, `-O3 -arch=sm_70`.

## Result

**7.82 → 6.92 ms/rep, 1.13× (bit-identical).** Every output array —
`h_layer`, `hTr_T`, `hTr_S`, `u`, `v`, `bt_eta`, and the heat/salt/mass budgets —
matches the faithful path to `max|diff| = 0.0`.

| grid (phys) | faithful ms/rep | optimized ms/rep | speedup | max rel |
|---|---|---|---|---|
| 473×297×30 | 7.82 | 6.92 | 1.13× | 0 |
| 240×180×30 | 2.50 | 2.17 | 1.16× | 0 |

## The one real lever: T and S share the column geometry

The faithful path remaps the two tracers with **two separate `k_tracer`
launches**. Each one, per column, rebuilds the *entire* PPM machinery from
scratch: the cumulative interface depths `z_old`/`z_new`, the PPM edge
reconstruction (`q_L`/`q_R`/`q6` with the parabola limiters), and then walks the
old↔new overlap sweep. T and S ride on **exactly the same `h_old`/`h_new`
column**, so all of that geometry is identical between them — the faithful path
just pays for it twice.

`k_opt_tracer2` builds the geometry **once** and reuses it for both tracers in a
single pass:

* `z_old`/`z_new` accumulated once,
* one overlap sweep whose per-`ko` interval (`z_lo`, `z_hi`, `overlap`, `xi_lo`,
  `xi_hi`) is computed once and consumed by both tracers' integrals,
* the only per-tracer work kept is the arithmetic that genuinely differs — the
  two edge reconstructions and the two integral accumulations, **in the same
  order as the faithful kernel**, which is why the result is bit-identical.

The cheap surrounding do-concurrent loops are folded too:
`k_opt_pre` does `total_h + h_ref + h_old-snapshot + target_h` in one per-column
kernel, and `k_opt_post` does `mass-budget + h_layer-commit + bt_eta` in one.
10 launches → 5.

## No intermediate global scratch

`total_h` and `h_ref` are consumed only inside the pre-pass. The faithful path
round-trips both through DRAM (`k_total_h` writes `total_h`, `k_h_ref` reads it
and writes `h_ref`, `k_target_h` reads that). `k_opt_pre` keeps both in
registers — `h_ref = Σh − eta` and the column total `= h_ref + eta` never leave
the thread. `ale_remap_opt` does not even allocate them.

## What was tried and did NOT help: 32-bit indexing

Worth **1.06× on the sibling continuity kernel**, it was a **no-op here**
(1.134× with `size_t` indexing vs 1.129× with `int32`, i.e. inside the noise;
`-DIDX32=0` reproduces). Why the difference: continuity is memory/address-bound
with a flat one-thread-per-face layout, so address-register pressure mattered.
This kernel is **compute- and register-bound on the per-column PPM** — each
thread carries ~10 `NZ_STACK_MAX`-sized stack arrays, and the bottleneck is that
register/occupancy pressure and the serial overlap sweep, not global address
math. Cutting the index width frees nothing the schedule was waiting on. It is
left as the default anyway (correct, free, and future-proof if the balance
shifts), but the honest attribution is: **the 1.13× is entirely the T+S geometry
fusion.**

## Why not more

The two face-remap kernels (`k_opt_xface`, `k_opt_yface`) are the momentum
remap on u/v faces; they have no T/S-style twin to fuse with and are already a
single launch each, so they are carried through unchanged (bit-identical). They
are a large, irreducible share of the runtime, which caps the whole-remap
speedup: the fusion halves the *tracer* geometry cost, but the tracer step is
only one of the three heavy PPM passes (T+S, u, v).

## Reproduce

```bash
nvcc ab_main.cu opt_kernel.cu ale_kernel.cu -O3 -arch=sm_70 -I../common -o ab
# ALL GPU runs go through the shared-GPU lock:
bash ../tmp_local_artifacts/gpu_run.sh ale ./ab 473 297 30 100 20 25
# isolate the 32-bit-index contribution:  nvcc -DIDX32=0 ... -o ab_i64
```

Args: `nx ny nz reps warmup hdrift%`. State is restored from a pristine copy
before every timed rep (the remap has a fixed point; un-restored reps measure a
kernel doing strictly less work than production). Timing is CUDA events around a
single call, restore untimed, min over reps.
