// common/gpu_rt.h  —  GPU-runtime redirect shared by every kernel's C++ side.
//
// The .cu sources stay CUDA-spelled (cudaMalloc, <<<>>>, cudaGraph*, ...); this
// header redirects those names to the HIP runtime when -DUSE_HIP is set, so ONE
// .cu builds under both nvcc (CUDA) and hipcc (HIP). Selected by ../config.mk's
// BACKEND (cuda|hip). hipcc accepts <<<...>>> launch syntax as-is, and HIP's
// kernel-side intrinsics (__global__/__device__/threadIdx/...) are identical to
// CUDA's, so kernels need no changes at all.
//
// This is the C++ analogue of directives.h: macro the plumbing, keep the
// compute single-sourced. If a kernel uses a cuda* symbol not mapped below, it
// will fail to compile under hip until a line is added here — an intended
// tripwire, never a silent fallthrough. Add the mapping here, not per-kernel.
//
// NB naming is NOT a uniform cuda->hip prefix swap. The pinned-host allocators
// reorder the words: cudaMallocHost / cudaHostAlloc  ->  hipHostMalloc, and
// cudaFreeHost -> hipHostFree. They are mapped explicitly below.
#ifndef GPU_RT_H
#define GPU_RT_H

#if defined(USE_HIP)
#  include <hip/hip_runtime.h>

// --- core memory + control ---
#  define cudaError_t                 hipError_t
#  define cudaSuccess                 hipSuccess
#  define cudaGetErrorString          hipGetErrorString
#  define cudaGetLastError            hipGetLastError
#  define cudaDeviceSynchronize       hipDeviceSynchronize
#  define cudaMalloc                  hipMalloc
#  define cudaFree                    hipFree
#  define cudaMemcpy                  hipMemcpy
#  define cudaMemset                  hipMemset
#  define cudaMemcpyHostToDevice      hipMemcpyHostToDevice
#  define cudaMemcpyDeviceToHost      hipMemcpyDeviceToHost
#  define cudaMemcpyDeviceToDevice    hipMemcpyDeviceToDevice

// --- pinned host memory (words reordered vs CUDA -- see header note) ---
#  define cudaMallocHost              hipHostMalloc
#  define cudaHostAlloc               hipHostMalloc
#  define cudaFreeHost                hipHostFree

// --- events + streams ---
#  define cudaEvent_t                 hipEvent_t
#  define cudaEventCreate             hipEventCreate
#  define cudaEventRecord             hipEventRecord
#  define cudaEventSynchronize        hipEventSynchronize
#  define cudaEventElapsedTime        hipEventElapsedTime
#  define cudaEventDestroy            hipEventDestroy
#  define cudaStream_t                hipStream_t
#  define cudaStreamCreate            hipStreamCreate
#  define cudaStreamDestroy           hipStreamDestroy
#  define cudaStreamSynchronize       hipStreamSynchronize
#  define cudaStreamBeginCapture      hipStreamBeginCapture
#  define cudaStreamEndCapture        hipStreamEndCapture
#  define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal

// --- CUDA graphs ---
#  define cudaGraph_t                 hipGraph_t
#  define cudaGraphExec_t             hipGraphExec_t
#  define cudaGraphInstantiate        hipGraphInstantiate
#  define cudaGraphLaunch             hipGraphLaunch
#  define cudaGraphDestroy            hipGraphDestroy
#  define cudaGraphExecDestroy        hipGraphExecDestroy

// --- introspection (attributes, device props, pointer kind) ---
#  define cudaFuncAttributes         hipFuncAttributes
#  define cudaFuncGetAttributes      hipFuncGetAttributes
#  define cudaDeviceProp             hipDeviceProp_t
#  define cudaGetDeviceProperties    hipGetDeviceProperties
#  define cudaPointerAttributes      hipPointerAttribute_t
#  define cudaPointerGetAttributes   hipPointerGetAttributes
#  define cudaMemoryTypeDevice       hipMemoryTypeDevice
#  define cudaMemoryTypeUnregistered hipMemoryTypeUnregistered

#  define GPU_RT_NAME "hip"
#else
#  include <cuda_runtime.h>
#  define GPU_RT_NAME "cuda"
#endif

#endif // GPU_RT_H
