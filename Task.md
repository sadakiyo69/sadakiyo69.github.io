# GPUs vs CPUs (lab 1: architecture and performance)
The goal of this lab is to develop a deeper understanding of the fundamental architectural differences between GPUs and CPUs and to observe how these differences impact performance.

## Tasks (system info)
1. GPU architecture: ...    
2. No. SMs: ...
3. No. SPs: ...
4. CPU: ...
5. No. threads CPU vs no. threads GPU: ...

## Tasks (performance)
1. Compile and run addition
2. Measure execution time GPU (hint: `cudaEvent_t start, stop;` \n `cudaEventCreate(&start);`)
3. Implement CPU version
4. Measure execution time CPU
5. Compare CPU and GPU execution with changing block sizes (as low and as high, e.g., 512, 2048, so that you can gather performance trends) and data sizes (up to and including 50 billion)
6. Plot measurements and speedups (block sizes and increasing data)
7. Compile and run race condition exercise
8. Pass the race condition test by identifying the bug in the _countAboveThreshold_ CUDA kernel and solving it.

## Wrap up and deliverables
1. Create blog post (or GitHub repo)
2. Describe main applications for GPUs (graphics and AI)
3. Showcase your experiments and speed ups achieved
4. Segway to architectural differences (describe CPU vs GPU during lab, cores, threads, and other specifics)
5. Showcase speed up experiments and describe differences (especially for today)
6. Discuss the race condition problem, how they might appear, and how they are usually treated.
7. Wrap up and conclude with what's next    