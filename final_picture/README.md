# final_picture — "is it worth rewriting in C++?", assembled

Collates all 8 kernel benchmarks into one weighted answer. Each benchmark
already compares Fortran `do concurrent` against a hand-written CUDA C port
(launched via OpenACC `host_data`) **and** a native `cudaMalloc` driver with no
Fortran/OpenACC at all. This harness runs them serially on an idle GPU, parses
each into a CSV, and weights every kernel by its share of production wall time.

## Run it

```bash
make all-dc                 # build every benchmark (BACKEND=cuda; see ../config.mk)
make run-all                # ...then run bench+native for all, idle-gated, and collate
# or drive the harness directly:
./final_picture/build_and_run_all.sh          # run + parse + collate
PARSE_ONLY=1 ./final_picture/build_and_run_all.sh   # re-collate existing logs
make picture                # just re-run the collator over results/*.csv
```

Production size is 473×297×30 (the profiled 0.1° config). GPU util is checked
<15% before each run — the LOGBOOK documents 15–40× swings under contention.

## What each piece does

| file | role |
|---|---|
| `build_and_run_all.sh` | serial, idle-gated runner: bench + native per kernel → `logs/`, then parse + collate |
| `parse_one.py` | one kernel's raw stdout (`logs/<key>_{bench,native}.txt`) → tidy `results/<key>.csv` (variant, family, ms) |
| `collate.py` | `results/*.csv` + `shares.csv` → `results/summary.csv` + the printed verdict |
| `shares.csv` | profile weights (H200, LOGBOOK §4) — the one file to edit if shares change |
| `logs/` | raw run output (kept; scratch/jobfs is ephemeral) |
| `results/` | per-kernel CSVs + `summary.csv` |

## Reading the output

- **ms columns are within-row only** — each kernel's unit differs (ms/call, /rep,
  /step; see `shares.csv`) and fires a different number of times per timestep.
  Cross-kernel currency is `share × (1 − fast/slow)`, which is what the totals sum.
- **`brdg` = native ÷ best-CUDA.** ~1.0 in every kernel ⇒ the OpenACC→CUDA
  `host_data` bridge is free, so the CUDA numbers are a fair comparison.
- **`rewrite` = bestFortran ÷ bestCUDA.** >1 ⇒ CUDA faster; <1 ⇒ Fortran wins.

## The answer

A full CUDA/HIP rewrite buys **single-digit %** of wall time; portable-Fortran
fixes buy **~3× more**, and on the biggest kernel (Redi, 40%) Fortran already
wins. See the caveats the collator prints: kappa-shear's gap is pre-`maxregcount`
flag, and EPBL's 1.33× is a compiler register-allocator bug, not a language win.
Reconciled with those, this matches the LOGBOOK headline (~16% Fortran, ~1.7% C++).

## HIP

`make all-dc BACKEND=hip` flips the comparison-kernel compiler to `hipcc` via
`../config.mk`. It is a documented seam, not yet validated (no ROCm here): the
`.cu` need a `hipify-perl` pass and the benchmark `CUFLAGS` carry nvcc-isms
(`--compiler-options` → `-Xcompiler`). `GPU_ARCHFLAG` already handles `-arch`.
