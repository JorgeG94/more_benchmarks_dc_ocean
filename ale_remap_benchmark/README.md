# ALE remap: production `do concurrent` vs improved Fortran vs hand-written CUDA C

`ocean_ale_remap` is **11.2% of production runtime** — the #2 region in the profile
(`tassie_runs/gabight_acc_120d.173705307.<sched>.log`: 149.8 s of 1340 s, 34560 calls,
one per outer step). This is its first measurement.

```bash
source ../../<model>-sea-ice/environments/toolkits/<system>/nvhpc.sh
make && make run          # ~1 min, 7 variants + bit-identity
make native               # NATIVE C++/CUDA driver: cudaMalloc + main(), no Fortran
make native-verify        # + prove it matches the Fortran run bit-for-bit (§5b)
make sweep                # the real config sizes
make verbatim             # re-check the verbatim kernel against production
make collapse regs        # -Minfo collapse report / register+stack usage
```

---

## 0. TL;DR

**The prime hypothesis (async) is dead. The answer is register pressure and
local-memory spill traffic, and the fix is loop fusion — not `!$acc async`.**

Portable Fortran gets **1.46x**, bit-identical (`max|diff| = 0.0` exactly on every
field). Hand-written CUDA C gets **1.54x**, i.e. a full CUDA rewrite buys **6% more**
than portable Fortran. Measured on **V100**, 473x297x30 (the 0.1 deg gabight geometry):

| variant | ms | vs DC | bit-identical? |
|---|---|---|---|
| DC as shipped | 10.731 | 1.000x | — |
| DC + `!$acc kernels async(1)` + one drain | 10.648 | **1.008x** | yes (0.0) |
| DC + fixes (dispatch hoist + collapse fix + signature) | 9.357 | 1.147x | yes (0.0) |
| **DC + fixes + fused (best portable Fortran)** | **7.374** | **1.455x** | **yes (0.0)** |
| DC + fixes + fused + async | 7.338 | 1.462x | yes (0.0) |
| CUDA faithful (one kernel per loop) | 7.820 | 1.372x | 1.8e-12 (FMA) |
| **CUDA fused (the ceiling)** | **6.961** | **1.542x** | 1.8e-12 (FMA) |

⚠ **The 11.2% profile is from an H200, not a V100 — see §6 before translating any of
this into % of total runtime.** This is the single biggest caveat in the report.

**The CUDA rows above are launched from Fortran through `!$acc host_data use_device`, so
the OpenACC runtime sits on the CUDA path (15 present-table lookups per call, inside the
timed region). A native `cudaMalloc` + `main()` driver with no Fortran and no OpenACC —
`ale_native.cu`, calling the same `ale_remap_cuda()` — measures the SAME time (1.000x).
The CUDA numbers here are not understated; the comparison is already fair. See §5b.**

---

## 1. ⚠ THE PRIME HYPOTHESIS IS FALSIFIED: async is worth 1.008x

The brief predicted ~1.5x from `!$acc kernels async(1)` + one `!$acc wait(1)` drain,
by analogy with the MEKE module (16 loops, no async, 1.48x).

**Measured: 1.008x.** It is worth nothing here, at every size tested (1.00–1.05x).

This was predictable from arithmetic *before* building anything, and the prediction
held. The profile says 149.756 s / 34560 calls = **4.33 ms per call**. The live path
has ~10 `do concurrent` launches. At the established ~8 us per launch, the *entire*
host-scheduling budget is ~80 us = **1.8% of the call**. There is no 1.5x to recover
because there is no host gap to hide.

**Why MEKE and ALE-remap differ, measured:** MEKE's 16 loops are cheap memory-bound 2-D
sweeps, so an ~8 us launch gap dominates each one. ALE remap's time is concentrated in
**4 heavy per-column kernels that are 91% of the call**:

| kernel (DC as shipped) | ms/call | share |
|---|---|---|
| `ocean_remap_tracer_field` x2 (T, S) | 5.19 | 47% |
| `remap_y_face_velocity` | 2.44 | 22% |
| `remap_x_face_velocity` | 2.41 | 22% |
| `compute_target_h` | 0.70 | 6% |
| the other 6 (cheap) loops | ~0.4 | 4% |

**Lesson (generalises):** "N loops, no async" is not a sufficient condition for the
MEKE result. The async win is a function of *loop cheapness*, not loop count. Compute
`profiled_time_per_call / n_loops` first: if it is >> 8 us, async cannot help. Here it
is 433 us per loop.

---

## 2. What the time actually is: local-memory spill traffic (MEASURED)

The four column kernels keep per-thread column arrays (`NZ_STACK_MAX` doubles each):
5 in `ocean_remap_tracer_field` + 5 more inside `remap_column_ppm` (`z_old`, `z_new`,
`q_L`, `q_R`, `q6`). They are indexed by a **data-dependent** index (`q_L(ko)`, where
`ko` is each thread's own overlap-sweep cursor), so they cannot be register-allocated
and cannot be uniformly addressed across a warp.

`ncu`, shipped `ocean_remap_tracer_field`, one launch at 473x297x30:

| metric | value |
|---|---|
| duration | 2.54 ms |
| global load sectors | 5,624,055 |
| **local load sectors** | **33,432,311** |
| **local store sectors** | **13,519,165** |
| achieved occupancy | **11.90%** (register-limited to 2 blocks/SM) |
| registers / stack frame | **254 / 28,856 B** |

Local traffic = 46.95M sectors x 32 B = **1.50 GB per call**, in 2.54 ms = **592 GB/s,
~73% of the V100's ~810 GB/s peak**. The kernel moves 1.5 GB of spill traffic to do
180 MB of useful global work — an **8.3x amplification**. It is not compute bound and
not launch bound: **it is at the local-memory bandwidth wall.**

### The decisive control: nvfortran and CUDA C spill the SAME amount

| kernel | ms | global ld | local ld | local st | occ % | REG |
|---|---|---|---|---|---|---|
| DC shipped `tracer_field` | 2.54 | 5.62M | 33.43M | 13.52M | 11.9 | 254 |
| DC fixed `tracer_field` | 2.06 | 5.65M | 33.27M | 13.41M | 22.3 | 126 |
| CUDA `k_tracer` | 2.00 | 5.70M | 31.53M | 13.24M | 49.5 | 56 |
| Fortran fused `two_tracers` | 3.18 | 9.08M | 45.77M | 22.08M | 22.5 | 126 |
| CUDA fused `k_tracer2` | 3.06 | 9.06M | 42.24M | 19.90M | 45.5 | 64 |

**This is the core result.** Local traffic is within 6% between nvfortran and CUDA C
(33.4M vs 31.5M sectors). **nvfortran is not generating extra spills — the algorithm
demands them.** CUDA's only real advantage is register allocation (56 vs 126 vs 254),
and past ~22% occupancy that advantage buys almost nothing: 22.3% -> 49.5% occupancy is
worth just **1.03x** (2.06 -> 2.00 ms). Below the knee it matters: 11.9% -> 22.3% is
worth **1.23x**.

That is why a full CUDA rewrite is only worth 6% here.

### Stack frame SIZE is a red herring (hypothesis tested and rejected)

I hypothesised the 28,856-byte frame itself was the cost. **It is not.** Rebuilding with
`NZ_STACK_MAX=256` (production `build_hopper`'s value) **doubles the frame to 57,528 B
and changes the time by nothing**:

| `NZ_STACK_MAX` | STACK (shipped) | DC time |
|---|---|---|
| 128 | 28,856 B | 10.75 ms |
| 256 | 57,528 B | 10.80 ms |

Only elements `1..nz` are ever touched, and dead branches are never executed, so frame
size costs address space, not bandwidth. **Raising `NZ_STACK_MAX` is free; lowering it
would not help.** The `RESUME` note about `NZ_STACK_MAX` locals being "a known pathology"
is right about the *traffic* but wrong if read as being about the *declared size*.

---

## 3. The three Fortran fixes (1.147x combined, all bit-identical)

### 3a. Hoist the method dispatch out of the loop body — registers 254 -> 126

`remap_column(method, ...)` does a 5-way `select case` **inside** the `do concurrent`
body, so nvfortran inlines **all five** remap methods per thread and the frames sum:

| | REG | STACK |
|---|---|---|
| `remap_column(method,…)` (as shipped) | **254** | 28,856 |
| `remap_column_ppm(…)` called directly | **126** | 12,376 |

Worth **1.082x** alone. The mechanism is the **register** drop (254 -> 126 lifts
occupancy 11.9% -> 22.3%, through the knee), **not** the stack drop (§2).

Production reads `method` from `vcoord%remap_method` (namelist, default PPM) — it is a
*runtime* value but a *loop-invariant* one. The fix keeps all five methods available:
dispatch once on the **host**, calling a per-method specialised kernel. This is a real,
cheap change to `ocean_remap.F90`.

### 3b. `compute_target_h` does not auto-collapse — a genuine bug, in production

This one is new and is not in `RESUME`. Every 3-D `do concurrent` in the module reports
`collapse(3) auto-collapsed` — **except** `ocean_vcoord_compute_target_h_impl`:

```
212, Loop run sequentially                       <-- k is SERIAL
     Loop parallelized across CUDA threads(128)
     Loop parallelized across CUDA thread blocks
```

Production's loop bounds are `associate`-names aliasing derived-type components
(`nx_total => this%nx_total`, `ocean_vcoord.F90:561-567`). Hoisting them to plain
local integers restores `collapse(3)`:

```fortran
nxl = this%nx_total          ! plain integers, outside the associate
nyl = this%ny_total
associate (target_h => this%target_h, dsig => this%dsig)
   do concurrent(k=1:nz, j=1:nyl, i=1:nxl)
```

Worth **1.06x** overall (0.70 ms -> ~0.09 ms on that kernel). The loop is a trivial
`target_h = (h_ref+eta)*dsig(k)` that was running at ~52 GB/s while the `h_old` snapshot
loop beside it — same shape, same size — runs at 740 GB/s.

**This extends `RESUME` §1.** The documented trigger requires an explicit-shape array
dummy bounded by a DT component **AND** a procedure call in the body. Here there is **no
procedure call** and the DT component is a **loop bound**, not an array bound. So either
the documented trigger is incomplete, or this is a distinct defect. **I did not isolate
it further** — flagged for the NVIDIA report, not claimed as understood.

### 3c. Signature fix: applies, but only in one place, and it is small

- The four heavy column kernels **already take plain explicit-shape dummies bounded by
  plain integers** (`h_old(nx,ny,nz)`). Production's comment says so explicitly
  ("Flat-arg so GPU codegen doesn't chase the array-of-derived-types pointer"). **The
  signature fix is already applied there. Nothing to win.**
- `compute_target_h` takes **assumed-shape** `total_h(:,:)` / `eta(:,:)` (marked
  "assumed-shape-ok"). Making them explicit-shape is worth **1.19x on that kernel**
  (0.655 -> 0.551 ms) = **~0.9% overall**. Real, but a rounding error next to §3b's
  collapse fix on the *same* loop.

Consistent with the brief's scaling rule (cost scales with the number of distinct arrays
a loop touches): these loops touch 2–5 arrays, so the ~1.30x seen on the barotropic
substep's 19-array Pass 1 was never on the table.

---

## 4. Fusion: 10 launches -> 5, worth 1.27x on top of the fixes

**The fusion win here is NOT about launch overhead** (§1 killed that). It is one real
algorithmic saving:

**T and S remap across the same `h_old -> target_h` column geometry.** Production builds
`z_old`, `z_new` and runs the entire overlap sweep (`z_lo`/`z_hi`/`overlap`/`xi_lo`/
`xi_hi`) **twice, identically**. Fused, they are built once and reused; only the
per-tracer arithmetic is duplicated, in the same order.

Measured: 5.19 ms (two separate) -> 3.18 ms (fused) = **1.63x on 47% of the call**, and
the mechanism is visible in the counters — local traffic per call drops from
2x(33.43+13.52) = 93.9M sectors to 67.85M, i.e. **-28%**, with global loads roughly
halved too.

### What was fused

| group | loops | why safe |
|---|---|---|
| G1 | L1+L2+L3+L4 (total_h, h_ref, h_old, target_h) | same-index chains; L3 rides L1's k-sweep |
| G2 | L5+L6 (T and S) | independent tracers, shared geometry |
| G5 | L9+L10 (mass budget, h_layer, bt_eta) | L10 sums what L9 wrote at the same (i,j,k) |

### What could NOT be fused, and why

- **L7 (x-face) and L8 (y-face) are real barriers.** Both read **neighbours** of what G1
  wrote: `h_old_face(k) = 0.5*(h_old(I-1,j,k) + h_old(I,j,k))`. Column `(I,j)` needs
  `h_old`/`target_h` from column `I-1`, produced by a *different thread*. Fusing them
  into G1 would be a **race**, not a slowdown. They stay separate.
- **L7 and L8 cannot be fused with each other**: different iteration spaces
  ((nx+1)*ny vs nx*(ny+1)) and different stencils. A merged loop is a divergent branch
  over the union — not a saving.
- G5 *could* legally fold into G1+G2 (it writes only `h_layer`/`mass_budget`, which the
  face loops never read). Kept separate: it measured no faster.

### Bit-identity trap this hit, worth recording

The first fused implementation was **1 ulp off** (`max|diff| = 9.1e-13` on `hTr_T`).
Cause: I factored `d3 = (xi_hi**3 - xi_lo**3)/3.0` then `d3*q6`, where production does
`(xi_hi**3 - xi_lo**3)*q6/3.0`. Same value in exact arithmetic, different rounding.
Restoring production's association order gives **exactly 0.0**. Loop fusion must
preserve *association order*, not just operation order.

---

## 5. Verification

- **All four Fortran variants are bit-identical to DC-as-shipped: `max|diff| = 0.0`
  exactly**, on `h_layer`, `hTr_T`, `u_face_x`, `bt_eta` and `heat_budget`.
- **CUDA differs by 1.8e-12 on `hTr_T`** (~1.3e-15 relative) and 6.7e-16 on `u_face_x`
  — FMA contraction between nvcc and nvfortran, as expected; `h_layer` and `bt_eta` are
  exactly 0.0 even for CUDA.
- **The check discriminates** (per the brief — a check that cannot fail proves nothing):
  perturbing one cell of the DC result by **1 ulp** trips the comparison at 8.882e-16.
  The probe is in the driver and runs every time.
- **Every variant has its own result slot** and the working state is restored from a
  pristine device copy before every rep, so no variant can inherit another's "agreement
  OK".
- **The restore is load-bearing for the timing, not just correctness**: the remap has a
  **fixed point** (after one call `h_old == target_h`, so every new layer aligns exactly
  with an old one and the overlap sweep collapses to one iteration per layer). Timing
  reps without restoring would measure a kernel doing strictly less work than
  production's.
- **Cost is insensitive to the state knob it might have depended on.** The h-drift the
  remap must undo (CLI arg 6) does not change the answer — 1% and 50% drift both give
  10.73–10.76 ms — because the `ko_start` cursor makes the sweep amortised O(nz)
  regardless. So the numbers do not hinge on my choice of synthetic perturbation.

---

## 5b. The native C++/CUDA driver: is the OpenACC runtime taxing the CUDA path?

**No. Measured 1.000x. The CUDA numbers in this report are already fair.**

Every CUDA row above is launched *from Fortran*: `ale_bench.F90:run_cuda` opens an
`!$acc host_data use_device(...)` region over **15 arrays**, and each one is a runtime
present-table lookup. That block is **inside the timed region**. So the OpenACC runtime
sits on the CUDA path, and if it costs anything, every "CUDA buys N%" number here is
understated and the comparison is rigged against CUDA. That is a real objection and it
cannot be answered from the Fortran driver.

`ale_native.cu` answers it: `cudaMalloc` + `main()`, no Fortran, no OpenACC, no
`host_data`, no `-cuda`. It links **`ale_kernel.o` — the same object `ale_bench` links**,
so the kernels are not duplicated and the two drivers cannot drift apart. `ldd` confirms
the separation: `ale_bench` pulls in `libacchost.so`, `libnvf.so` and `libcudafor.so`;
`ale_native` pulls in none of them, only `libstdc++`/`libm`/`libc` (cudart is static).

| variant | via Fortran (`host_data`) | native (`cudaMalloc`) | ratio |
|---|---|---|---|
| CUDA faithful | 7.766 ms | 7.798 ms | **0.996x** |
| CUDA fused | 6.979 ms | 6.977 ms | **1.000x** |

⚠ **These timings are PROVISIONAL — taken with up to 9 sibling agents on this one V100.**
Contention was severe and directly observed: the same native fused run gave anything from
6.977 to 22.198 ms (a **3.2x** swing), and `nvidia-smi` reported 23–100% utilisation with
an **empty** `--query-compute-apps` list (sibling processes are in other PID namespaces
and are invisible here — an empty process list is NOT evidence of an idle GPU). The table
is **min over 26 alternating runs** of the two binaries; contention can only *add* time,
so the minimum converges on the uncontended value from above. **A serial re-run on an idle
GPU is required to quote these to 3 digits.** The cleanest paired sample — both binaries
hitting a quiet window seconds apart — agrees: Fortran 7.792 / 6.985, native 7.805 / 6.981.

### The sensitive probe: the tie is not just noise-hiding

A ~15 µs fixed cost is invisible against a 7 ms kernel, so the production-size tie alone
would be weak evidence. Shrink the grid until the kernel cost collapses but the fixed
per-call cost does not — both drivers issue the same 10 (or 5) launches, and only the
Fortran one also does 15 present-table lookups:

| grid | total call | fortran − native, faithful | fortran − native, fused |
|---|---|---|---|
| 32x32 (nxt=38) | ~0.54 ms | **+8 µs** | **+4 µs** |
| 64x64 (nxt=70) | ~0.79 ms | −8 µs | +5 µs |

The difference is ~±8 µs, i.e. **the `host_data` bridge costs ≲10 µs per call** — under
0.15% of the 7 ms production-size call, which is why the tie is real and not merely
unresolvable. (A 16x16 batch was discarded: its *Fortran* time came out **higher** than
32x32's, which is physically impossible and marks the batch as contention-poisoned. Noted
rather than silently dropped.)

### Verification — three independent legs, unaffected by contention

`./ale_bench 473 297 30 20 10 25 1` writes `ale_ref.bin` (pristine input state + the
results of variants 1, 6, 7); `./ale_native --ref ale_ref.bin` diffs against it. `make
native-verify` does both.

1. **PATH — the load-bearing one.** Fed the *same* state, the native driver's output is
   **bit-identical** to the same kernel launched from Fortran via `host_data`: `0.0`
   exactly on `h_layer`, `hTr_T`, `u_face_x`, `bt_eta` and `heat_budget`, for **both**
   faithful and fused. This is what proves the native driver is a like-for-like
   replacement and not a different computation that merely runs at a similar speed — a
   timing tie between two different computations would prove nothing.
2. **PHYSICS.** vs variant 1 (DC-as-shipped Fortran), the native driver reproduces the
   Fortran-driven CUDA's agreement **exactly**: `0.0 / 1.819e-12 / 6.661e-16 / 0.0 /
   1.819e-12`, the identical row §0 reports for CUDA fused. The native driver inherits the
   port's FMA-level agreement with production; it does not launder it.
3. **INIT.** The native init is transcribed from `ale_bench.F90` and diffed against the
   pristine state that run actually used. `dsig` is **exactly** 0.0 (it has no
   transcendentals — which also confirms the `sum()` association order was reproduced);
   the other 7 fields agree to **1–2 ulp** (worst `9.1e-12` on `hTr_S`, ~1e-15 relative).

**The init is NOT bit-identical, and that is a host-libm difference, not a modelling
choice**: the synthetic init calls `sin`/`cos`/`exp`, and glibc's (via nvcc's host
compiler) differ from nvfortran's by 1–2 ulp. It is confined to the benchmark's synthetic
state and never touches the kernel. `--use-ref-state` isolates it by running the native
driver on the *dumped* state — and that is the mode in which legs 1 and 2 return exact
`0.0`, which localises the entire discrepancy to the init and clears the CUDA path. **It
cannot affect the timings**: §5 already measured that h-drift from 1% to 50% does not move
the time, so a 1e-15 relative perturbation cannot.

**The check discriminates**: perturbing one cell of the native result by 1 ulp trips the
comparison at `4.441e-16`. Note this probe is deliberately diffed against the *unperturbed
native* result rather than against the Fortran result the way `ale_bench.F90`'s is — that
form is only meaningful when the baseline is exactly `0.0`, which is true under
`--use-ref-state` but **false** under the computed init, where ~1e-13 of libm noise would
swamp a 1-ulp probe and let it report a pass it had not earned.

### What this means for the rest of the repo

**Nothing shifts, and that is the useful result.** The concern that every CUDA port in this
repo is measured through an OpenACC tax — and so `LOGBOOK.md` §0's "CUDA's edge is a flat
6–10%" is really "CUDA's edge is ≥6–10%" — is **falsified here**: the tax is ≲10 µs/call.
CUDA's 1.059x on this kernel stands as measured.

⚠ **Scope.** This is measured **on this kernel only**. The bridge cost is ~fixed per call
(15 lookups), so the conclusion transfers to any kernel whose call is milliseconds —
`redi` (33 ms), `kshear` (36 ms), `epbl` (6.5 ms) are all safely in that regime. It is
**not** automatically safe for a *cheap*, launch-bound kernel: MEKE's whole call is
**0.213 ms**, where ≲10 µs would be up to ~5%. That is untested and is the one place this
result should not be extrapolated to. Same lesson as §1: **cheapness, not count.**

---

## 6. ⚠ CAVEATS — read before quoting a % of runtime

**1. The profile is from an H200; this benchmark ran on a V100.** The log header reads
`host: <h200-node>`, `GPU 0: NVIDIA H200`, `running: build_hopper/<model>`. The
two builds differ in both ways that matter to this kernel:

| | profile (`build_hopper`) | this benchmark (`build`) |
|---|---|---|
| GPU | **H200 (cc90)** | V100 (cc70) |
| `NZ_STACK_MAX` | **256** | 128 |

This fully explains my 10.73 ms/call vs the profile's 4.33 ms/call (~2.5x, in line with
H200 vs V100). I could not run on an H200 — this node has a V100 and no H200 was
reachable. **Mitigations:** I re-ran everything at `NZ_STACK_MAX=256` and the ratios are
unchanged (fused 1.460x vs 1.455x), so the stack difference does not matter (§2). The
**architecture** difference is untested and is a genuine risk *specifically here*,
because the whole result is about registers, occupancy and local-memory bandwidth —
exactly what differs most between Volta and Hopper. Hopper has more L1/SMEM per SM and a
different local-memory path, so the spill wall may sit elsewhere.

**2. The % translation is therefore EXTRAPOLATED, not measured.**
Applying the V100 **ratio** to the profiled 11.2%:

- best portable Fortran, 1.455x -> ALE remap 11.2% -> 7.7% of runtime = **~3.5% of total
  wall time**
- CUDA fused, 1.542x -> **~4.0% of total**, i.e. a full CUDA rewrite buys ~0.5% of total
  over portable Fortran

**Both assume the V100 speedup ratio transfers to the H200. That is unverified.** If
someone can run `make` + `make run` with `ARCH=cc90 NVARCH=sm_90 NZSTACK=256` on an H200
node, that single number would replace the weakest link in this report.

**3. `ale_remap_dc.F90` is a TRANSCRIPTION, not a verbatim copy.** `kernel_remap.F90`
*is* byte-identical to production (`make verbatim` enforces it, md5
`a079bd1e00b179d3233195c73953c2ee`) and is deliberately kept **whole** — including
`remap_column_ppm_h4`/`_pqm`, which production never calls at these configs but which are
reachable through the in-body `select case` and therefore change register pressure.
Dropped from the orchestrator, each verified dead for every config in
`~/analysis_gebco/*.nml` (**all** set `vcoord_type = "zstar"`): the EULERIAN_Z/LAGRANGIAN
early return; the `VCOORD_RHO`/`HYCOM` branch (`build_ts_concentration` +
`compute_target_h_rho` + `eos`); the grid time-filter (`regrid_time_scale`, default 0.0);
and `compute_target_h`'s non-zstar branches. The last is safe because those branches are
*separate loops* behind a `select case` **outside** the loops — untaken branches cannot
affect the taken loop's codegen. Full list in the file header.

**4. `conserve_ke` is kept but does not fire** (`remap_vel_conserve_ke` default
`.false.`), matching production. `rescale_anomaly_ke` is still compiled into the face
kernels, as in production.

**5. Synthetic state.** Bathymetry, stratification and velocities are analytic, not from
`gabight_bathy_sph_0p1_smooth.nc`. Chosen for *branch coverage* (the PPM limiter, the
vanishing-layer guard and the overlap sweep all fire on a mix of columns, including
~1% near-dry columns) rather than physical realism. §5 shows the result is insensitive
to the one knob that plausibly mattered.

**6. No land mask.** Production's domain has land columns; the `do concurrent` runs them
anyway, so this should not bias the comparison — but it is untested.

---

## 7. What I could NOT explain

- **Why `compute_target_h` fails to auto-collapse.** The trigger is reproducible and the
  fix is one line, but `RESUME` §1's documented condition (DT-component array bound
  **AND** a call in the body) does **not** hold here — there is no call, and the DT
  component is a loop bound. Either that condition is incomplete or this is a second,
  distinct defect. **Not isolated. Stated as a symptom, not a mechanism.**
- **Why CUDA's `k_tracer` needs only 56 registers where nvfortran-fixed needs 126** for
  the same algorithm. It costs little (§2: 1.03x past the occupancy knee), so I did not
  chase it — but it is unexplained.
- **The face kernels (44% of the call) resisted everything.** Fusion cannot touch them
  (§4, real barriers) and they gain only from the dispatch hoist. CUDA is 1.12x/1.04x
  faster on them. Their local traffic is inherent to one-thread-per-column. A
  **one-warp-per-column** rewrite with the column arrays in shared memory is the obvious
  next lever and is **untested** — it is the main thing left on the table.

## 8. Files

```
kernel_remap.F90     VERBATIM production copy (make verbatim enforces)
constants.F90        stub: only what the remap path reads; NZ_STACK_MAX=128
remap_state.F90      stubbed hgrid_t / ocean_vcoord_t / multilayer_cgrid_state_t,
                         preserving the tracers(t)%hTr array-of-DT indirection
ale_remap_dc.F90         ONE source, compiled 3 ways -- this is what makes the async
                         test airtight (only the directives differ):
                           (default)                        DC as shipped
                           -DASYNC                      + !$acc kernels async(1)
                           -DPPM_DIRECT -DCOLLAPSE_FIX -DFLATSIG   + fixes
ale_remap_fused.F90  fused variant (+ -DASYNC), incl. remap_column_ppm2
ale_kernel.cu            CUDA C port: faithful (one kernel per loop) + fused
ale_bench.F90            driver: 7 variants, own state each, bit-identity + discriminate
                         arg 7 = 1 dumps ale_ref.bin for ale_native.cu (opt-in, ~571 MB,
                         removed by `make clean`; `make run` never writes it)
ale_native.cu            NATIVE driver: cudaMalloc + main(), no Fortran, no OpenACC (§5b).
                         Links ale_kernel.o -- the SAME object ale_bench links. The
                         kernels are NOT duplicated: two copies could diverge and would
                         silently corrupt the very comparison the file exists to make.
sweep.sh                 the real config sizes, min-of-N
```
