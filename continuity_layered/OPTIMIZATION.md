# Optimizing the layered-continuity CUDA kernel

A best-CUDA-practices pass over `layered_kernel.cu` (the faithful 11-kernel
transliteration), freed from the faithfulness rule but held **bit-identical** to
the original (`ab_main.cu` checks `max|diff| == 0` on `fh`). V100, 473×297×30.

## Result

**0.93 → 0.62 ms/rep, 1.50× (bit-identical).** 1.40–1.70× across sizes — more on
smaller grids, where cutting 11 launches to 3 matters most.

| size | speedup | max\|diff\| |
|---|---|---|
| 108×137×30 | 1.70× | 0 |
| 473×297×30 | 1.50× | 0 |
| 473×297×50 | 1.51× | 0 |
| 945×594×30 | 1.40× | 0 |

## What worked (the default build)

Two changes, both *removing genuine waste* rather than being clever:

1. **Fuse PPM + boundary + transport + walls per direction (11 kernels → 3).**
   Each x-face thread reconstructs the upwind cell in registers and writes `mfx`
   directly; the `hfl_x/hfr_x/hfl_y/hfr_y` scratch arrays never touch global
   memory. Removes ~280 MB of intermediate traffic per call. **1.41×.**
2. **32-bit indexing.** The domain fits `int32` (`nx*ny*nz < 2³¹`), so the
   `size_t` address math was pure overhead — fewer registers (60→48), cheaper
   addressing. **→1.50×.**

`kO_div` (the divergence) is left as its own kernel: it already runs at **~84%
of DRAM peak**, essentially the memory-bandwidth floor.

## What was tried and LOST (kept as `OPTVER=3..8`)

Every "clever" restructuring cost more than it saved — this kernel is
memory-bound and the flat one-thread-per-face layout lets the GPU's L2 +
occupancy do the reuse better than any hand-rolled scheme.

| idea | `OPTVER` | result | why it lost |
|---|---|---|---|
| Full recompute-fusion → 1 kernel | 3 | 0.75× | 4× redundant recon → compute-bound |
| `__launch_bounds__(128,≥14)` | — | 0.80× | register spill |
| block size 256 / 512 | — | ≤ | register-heavy → fewer blocks |
| k-loop (thread per face-line) | 4 | 1.27× | 145k threads < 1 wave, lost parallelism |
| k-blocking (KB k/thread) | 5 | ≤1.28× | same; ILP doesn't beat occupancy here |
| div fused with x- or y-flux | 6, 7 | 1.30/1.40× | 2× recon recompute > saved round-trip |
| shared-memory tiled y-flux | 8 | 1.46× | `__syncthreads` + smem occupancy cost > L2 relief |

The profile explains it: after the fusion, `kO_div` is DRAM-bound (~84%),
`kO_yflux` is L2-bound (~79%, the strided j-stencil), `kO_xflux` is
latency-bound (nothing saturated, register-limited to ~62% occupancy). None of
them has slack that recompute or tiling can cheaply buy back.

## Reproduce

```bash
nvcc ab_main.cu opt_kernel.cu layered_kernel.cu -O3 -arch=sm_70 -I../common -o ab
./ab 473 297 30 300          # correctness (max|diff|) + timing, original vs optimized
# sweep a variant:  nvcc -DOPTVER=8 -DBYS=8 ...   (see the launcher switch)
```

`opt_kernel.cu` is a standalone best-CUDA artifact; the faithful
`layered_kernel.cu` stays as the codegen-comparison baseline. Wiring the
optimized launcher into a driver (its signature drops the `hfl/hfr` scratch) is
a small follow-up if this is worth shipping.
