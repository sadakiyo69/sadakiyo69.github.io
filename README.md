# GPUs vs CPUs – Lab 1: Architecture and Performance

***Alexey Wratschinski, David Schell***

In this lab we compare a CPU and a GPU on a simple, highly parallel task (vector addition), look at why the GPU is so much faster, and finally run into a classic parallel-programming bug: the **race condition**.

**Hardware used**

| | CPU | GPU |
|---|---|---|
| Model | Intel Xeon w3-2525 | NVIDIA RTX PRO 4000 (Blackwell) |
| Compute units | 8 cores / 16 threads | 70 SMs, 8,960 CUDA cores (128 per SM) |

Source GPU information: https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/workstation-datasheet-blackwell-rtx-pro-4000-nvidia-3662515.pdf, 
https://www.techpowerup.com/gpu-specs/rtx-pro-4000-blackwell-workstation.c4279

Source CPU information: https://www.intel.com/content/www/us/en/products/sku/236882/intel-xeon-w32525-processor-22-5m-cache-3-50-ghz/specifications.html

---

## 1. What are GPUs used for?

**Graphics.** GPUs were built to render images: every pixel (or vertex) can be computed independently, so thousands of small computations run at the same time. This is still the core of games, 3D design, video editing and ray tracing.

**AI / Machine Learning.** Training and running neural networks is mostly matrix multiplication. The same simple operation applied to huge amounts of numbers. That maps perfectly onto a GPU, which is why modern AI like language models depends on GPUs.

Other areas: scientific simulation, video encoding, data analytics, crypto, ...

---

## 2. Experiment: Vector Addition

Task: `C[i] = A[i] + B[i]` for N floats. The same operation is implemented on both devices (see `addition.cu`):

```cuda
// GPU: one thread per element
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) C[i] = A[i] + B[i];
}

// CPU: one loop, one thread
void vectorAddCPU(const float *A, const float *B, float *C, int numElements) {
    for (int i = 0; i < numElements; ++i) C[i] = A[i] + B[i];
}
```

**Measurement**
- GPU: `cudaEvent` timing around the kernel launch.
- CPU: `clock()` around `vectorAddCPU`.
- Results of GPU and CPU are compared element by element ("Test PASSED").
- N from 50,000 to 500,000,000, block sizes 256 / 512 / 1024.

### Results (ms)

| Elements | CPU | GPU (256) | GPU (512) | GPU (1024) |
|---:|---:|---:|---:|---:|
| 50,000 | 0.17–0.20 | 0.16 | 0.20 | 0.17 |
| 500,000 | 1.66–1.74 | 0.27 | 0.26 | 0.25 |
| 5,000,000 | 16.5–17.7 | 0.36 | 0.34 | 0.35 |
| 50,000,000 | 137–171 | 1.34 | 1.28 | 1.30 |
| 500,000,000 | 1360–1367 | 11.73 | 11.90 | 11.77 |

![CPU vs GPU](plot_cpu_vs_gpu.png)

### What we observe

1. **Small data: no benefit.** At 50,000 elements CPU and GPU are equally fast. The kernel launch overhead and the fact that the GPU is not fully utilised eat the advantage.
2. **Growing data: speed-up explodes.** The CPU time grows linearly with N, while the GPU time stays almost flat until the GPU is saturated. Beyond ~5 million elements the speed-up is ~100×.
3. **Block size barely matters** for this kernel. All three configurations are within a few percent of each other, because vector addition is limited by memory bandwidth, not by how threads are grouped. (Block size matters more for kernels that use shared memory or many registers.)

---

## 3. Architecture: why is the GPU faster here?

| | CPU (Xeon w3-2525) | GPU (RTX PRO 4000) |
|---|---|---|
| Design goal | Low latency for a few complex tasks | High throughput for many simple tasks |
| Cores | 8 big cores, out-of-order execution, branch prediction | 8,960 small CUDA cores grouped in 70 SMs |
| Threads | 16 hardware threads | tens of thousands of threads, switched with almost no cost |
| Cache | Large caches per core | Small caches, instead very high memory bandwidth |
| Best for | Complex logic, branching, sequential code | Same operation on huge data sets |

On the GPU, threads are organised in **blocks**, blocks form a **grid**, and each block runs on one SM. Inside a block, threads are executed in groups of 32 (**warps**) that execute the same instruction at the same time. This is why vector addition fits so well: every thread does the same thing on a different element.

Block size selects how many threads form one block (256 / 512 / 1024 here) and therefore how many blocks are created, e.g. 500M elements → 488,282 blocks of 1024 threads, or 976,563 blocks of 512 threads.

---

## 4. Race conditions

### The bug

`race_condition_bug.cu` counts how many elements of a vector are above a threshold. The naive kernel:

```cuda
if (i < numElements && A[i] > threshold) {
    (*count)++;          // BUG
}
```

`(*count)++` is not one operation, but three: **read** the value, **add 1**, **write** it back. Thousands of threads do this on the same memory address at the same time. Two threads can read the same old value, both add 1, and both write the same result, so one increment is **lost**. The final count is too small, and it is **different in every run**, which is why the program runs the kernel several times and compares with the CPU result.

### The fix

```cuda
if (i < numElements && A[i] > threshold) {
    atomicAdd(count, 1); // read-modify-write as one indivisible operation
}
```

`atomicAdd` makes the hardware execute the whole read-modify-write without interruption, so no update is lost. The result now matches the CPU in every run (**PASSED**). Furthermore, if we scale the number of elements up, so will that result in significantly more execution time, when using atomicADD.

### How race conditions are usually handled

- **Atomic operations** like `atomicAdd`: simple, correct, but can be slow under high contention.
- **Reduction instead of a shared counter**: each block first counts locally (shared memory or warp shuffles), then only **one** `atomicAdd` per block is done. This is the standard high-performance solution.
- **Avoid sharing**: give every thread its own output location (like `C[i]` in the vector addition, which is why that kernel has no race condition).

On the CPU the same problem exists with multithreading and is solved with mutexes, atomics or thread-local counters.

---

## 5. Conclusion

- For massively parallel, simple operations the GPU was **~100× faster** than a single CPU thread on 50M–500M elements, while for small inputs there was no advantage.
- The block size had almost no influence for vector addition, because the kernel is limited by memory bandwidth.
- Parallelism brings new bug classes: the race condition showed that *correct-looking* code can give wrong and non-reproducible results, and that `atomicAdd` fixes it at a performance cost.