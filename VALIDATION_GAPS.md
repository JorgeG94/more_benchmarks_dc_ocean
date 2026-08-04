# Validation gaps in the +7.8% verdict

**Status: proposed, nothing run yet. Uncommitted.**

The headline — *a hand-CUDA rewrite of the ocean core is **+7.8% end-to-end***
— is a product of three things measured in different places:

```
whole-model verdict  =  per-kernel DC-vs-CUDA ratios  x  per-kernel shares
                        (reproducers, V100)              (live model, H200)
```

Two joins in that product are unvalidated. Neither invalidates the work; both
are worth closing before the number is load-bearing in a paper.

---

## Gap 1 — reproducer vs live kernel

The benchmark kernels are minimal reproducers extracted from the model: close to
the live kernels, not identical, and a snapshot that drifts as the model moves.

**What "faithful" means here — and what it doesn't.** In this repo, *faithful*
consistently means "the CUDA port is faithful to the DC reproducer" — controlled
tightly, bit-identity at `max rel diff < 1e-12`. It has never meant "the
reproducer is faithful to the live kernel". That second axis has no check, no
tolerance, and no owner.

**Which conclusions this threatens, and how much:**

* **"Does DC cost anything vs CUDA?" — fairly robust.** This is a codegen
  comparison, and it transfers as long as the reproducer sits in the same
  *performance regime* as the live kernel (register pressure, occupancy,
  divergence, memory traffic). It does not require the arithmetic to match.
* **"+7.8% end-to-end" — materially exposed.** It multiplies reproducer ratios
  by live shares. If a reproducer is cheaper or simpler than its live twin — the
  usual direction, since extraction tends to drop masking branches, vanishing-
  layer guards and halo handling — then both its ratio *and* its weight are off,
  and the errors do not obviously cancel.

The kernels most at risk are the ones whose cost is *data-dependent*:
`kappa_shear` is explicitly "latency/divergence-bound at 6% occupancy", and
divergence depends on realistic land fraction and thin layers. A reproducer fed
idealised state can sit in a completely different divergence regime from the same
kernel over a real coastline — while looking correct.

**The cheap decisive check: compare ms/call, not source.** Profile the live model
(`nsys` on a production-size run) and the reproducer at the same grid, and
compare per-kernel time per call. Agreement within ~10-20% means the reproducer
stands in for the live kernel and everything downstream holds. A large gap
localises which reproducer to fix, and is far quicker than diffing kernels by
eye. This also re-derives the shares on the same footing, which closes half of
gap 2 for free.

**If a reproducer does need to be made more faithful,** feed it real dumped state
rather than synthetic — land fraction and vanishing-layer counts should be
reported next to any timing for a divergence-bound kernel.

---

## Gap 2 — V100 timings vs H200 profile

Every DC-vs-CUDA timing here is **V100 (cc70)**. The profile that ranks the
kernels — the shares that decide which kernels matter — came from an **H200
(cc90)**, `build_hopper`. The repo already flags this in three READMEs:

> *"Both assume the V100 speedup ratio transfers to the H200. That is
> unverified."*
> *"Nothing here has been measured on the GPU that produced the profile."*

Two things must hold on Hopper for the verdict to survive:

1. **The giants stay near-parity.** `redi` + `kappa_shear` are ~80% of a stage.
   If nvfortran's cc90 codegen trails nvcc by more than its cc70 codegen does —
   plausible, since stdpar/ACC lowering for a newer arch is less mature — the gap
   widens exactly where it costs. The known `maxregcount`-quality gap (Bug 3) is
   config-dependent (hurt EPBL, *helped* kappa_shear on Volta), so its sign on
   Hopper is genuinely unknown — and both giants are register-bound.
2. **The shares stay roughly the same.** ~5x the bandwidth (HBM2 ~900 GB/s →
   HBM3e ~4.8 TB/s) and 132 vs 80 SMs. A stencil at ~84% of V100 DRAM peak may be
   occupancy- or latency-bound on H200 — which reorders the profile and opens
   room for hand-tuning that did not exist on Volta.

**What to run** — a re-run at `ARCH=cc90 NVARCH=sm_90`, no new code:

1. `cd meke && make cmp ARCH=cc90 NVARCH=sm_90` — largest V100 CUDA win,
   cheapest build, smoke-tests the toolchain at cc90.
2. `make cmp` at cc90 for the six kernels with an opt on both sides (redi, hvisc,
   ale_remap, btstep, meke, continuity_layered). **Add a cc90 column beside the
   cc70 one — do not overwrite it.** The V100→H200 delta is the result.
3. `cd ideal_benchmark && make run` at cc90 — confirms or replaces "+7.8%".
4. Diagnose only what moved: registers/thread and achieved occupancy, both sides,
   cc70 vs cc90. Name the regime before tuning (launch-bound → fuse/async;
   spill-bound → less per-column work; divergence-bound → algorithmic).

---

## Order of work

**Gap 1 first**, and specifically the ms/call comparison — it is cheaper, it can
be done on the V100 already here, and it re-derives the shares on a consistent
footing so gap 2's re-run has something trustworthy to multiply. Doing gap 2
first risks producing precise Hopper ratios for kernels that don't represent the
model.

---

## Discipline (unchanged — this is why the V100 numbers are trustworthy)

* **Exclusive GPU.** `nvidia-smi --query-compute-apps=pid,process_name
  --format=csv` empty. An 8% effect sits inside contention noise; a co-running
  job once flipped the sign of the whole-model result. 4-5 runs, ratio stable to
  <0.2%.
* **Production size `473 297 30`.** Small grids hide launch amortization — and
  under-filling 132 SMs distorts more than under-filling 80.
* **Keep `-gpu=tripcount:host`** (TPR #38714). Without it timings are ~2x wrong,
  and the penalty falls on the DC side.
* **`make cmp`, never two standalone binaries.** Ratio-of-ratios is how the two
  retracted claims happened.
* **Bit-identity first** (`max rel diff < 1e-12`). A timing from a kernel that
  hasn't passed its check is not a datum.
* **No toolkit constraint at cc90.** The CUDA 12.9 pin exists only because CUDA
  13 dropped `sm_70`. H200 is `sm_90` — current CUDA is fine.

---

## Likely outcome

Probably "the reproducers check out within noise, +8% becomes +12%, conclusion
holds". That is a good result: it converts two documented assumptions into
measured facts for a few hours of GPU time, and it is the difference between a
claim that survives review and one that doesn't.

The case worth wanting is the one where the giants move on Hopper. "Single-digit
% on Volta, X% on Hopper" makes the contribution *hardware-dependence of
performance portability* rather than a single-number verdict — stronger, provided
it is measured deliberately rather than found by a reviewer.
