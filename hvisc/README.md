# hvisc — ocean horizontal viscosity (Smagorinsky), build-mode

The `ocean_hvisc` Smagorinsky closure in the **build-mode layout**: compute is
single-sourced, strategy is a build MODE. This is the CPU/OpenMP-capable twin of
`../hvisc_benchmark/` (which is the GPU-only in-process DC-vs-CUDA comparison).

Two `do concurrent` kernels (`hvisc_kernel.F90`): `hvisc_compute_smag`
(strain → per-face viscosity `A_h = clamp((Cs·√(dxT·dyT))²·|D|)`) then
`hvisc_compute_face` (`A_h` × curvilinear-FV velocity Laplacian). Anonymised,
verbatim from `<model>`. Plain-integer explicit-shape dummies, no calls in the
loops → auto-collapses, no compiler-bug exposure.

## Build / run

```bash
make dc                 # do concurrent, OpenACC on the GPU            [default]
make dc DATA=omp        # do concurrent via OpenMP target (GPU)
make dc DATA=none       # do concurrent on the CPU (-stdpar=multicore) -- NO CUDA
make cpp                # native C++/CUDA (cudaMalloc)
make cpp BACKEND=hip    # ...or HIP (structural; needs ROCm)
make verify             # OpenACC dumps a ref; the CPU build cross-checks it
make run-dc / run-cpp / clean
```

## Result (V100, 473×297×30)

| mode | ms/rep | cross-check |
|---|---|---|
| `dc DATA=acc`  (OpenACC GPU) | 1.82 | reference |
| `dc DATA=omp`  (OpenMP GPU) | 1.95 | **max\|diff\| = 0.0** (bit-identical to acc) |
| `dc DATA=none` (CPU, no CUDA) | 12.7 | field-rel 7e-15 vs acc (FMA-level; sqrt) |
| `cpp BACKEND=cuda` (native) | 1.81 | field-rel 3.8e-15 vs DC (adopts DC's inputs) |

So `do concurrent` ≈ faithful CUDA (~1.81 vs 1.82 on the GPU), and the whole
closure runs on the CPU unchanged. The GPU is ~7× the CPU here.

## Note on the cross-check

The velocity init uses `sin/cos`; nvfortran's libm and glibc's differ ~1 ulp,
which the strain's cancellation amplifies on near-zero cells. So the ref dump
carries `u_face`/`v_face` and `cpp_main` **adopts** them (rather than
recomputing) — otherwise the check measures libm, not the kernel. The reported
metric is field-relative (`max|diff| / max|field|`), robust to near-zero cells.
