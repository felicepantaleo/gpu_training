# Introduction to GPU Programming 
### Presentation:
You can find the presentation here:
<a href="
https://docs.google.com/presentation/d/1Id7dLpzC0UlsJCJeXdJBvwigQ1aFaDmk0l0Pgkv7bwo/edit?usp=sharing" target="_blank">
https://docs.google.com/presentation/d/1Id7dLpzC0UlsJCJeXdJBvwigQ1aFaDmk0l0Pgkv7bwo/edit?usp=sharing</a>

### Hands-on

The CUDA Runtime API reference manual is a very useful source of information:
<a href="http://docs.nvidia.com/cuda/cuda-runtime-api/index.html" target="_blank">http://docs.nvidia.com/cuda/cuda-runtime-api/index.html</a>


Check that your environment is correctly configured to compile CUDA code by running:
```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
```


Compile and run the `deviceQuery` application:
```bash
$ cd hands-on/gpu/utils/deviceQuery
$ make
```

You can get some useful information about the features and the limits that you will find on the device you will be running your code on. For example:

```shell
$ ./deviceQuery
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 2 CUDA Capable device(s)

Device 0: "NVIDIA A100-PCIE-40GB"
  CUDA Driver Version / Runtime Version          13.0 / 13.0
  CUDA Capability Major/Minor version number:    8.0
  Total amount of global memory:                 40441 MBytes (42405855232 bytes)
  (108) Multiprocessors, (064) CUDA Cores/MP:    6912 CUDA Cores
  GPU Max Clock rate:                            1410 MHz (1.41 GHz)
  Memory Clock rate:                             1215 Mhz
  Memory Bus Width:                              5120-bit
  L2 Cache Size:                                 41943040 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        167936 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 5
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

Device 1: "NVIDIA A100-PCIE-40GB"
  CUDA Driver Version / Runtime Version          13.0 / 13.0
  CUDA Capability Major/Minor version number:    8.0
  Total amount of global memory:                 40441 MBytes (42405855232 bytes)
  (108) Multiprocessors, (064) CUDA Cores/MP:    6912 CUDA Cores
  GPU Max Clock rate:                            1410 MHz (1.41 GHz)
  Memory Clock rate:                             1215 Mhz
  Memory Bus Width:                              5120-bit
  L2 Cache Size:                                 41943040 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        167936 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 6
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
> Peer access from NVIDIA A100-PCIE-40GB (GPU0) -> NVIDIA A100-PCIE-40GB (GPU1) : No
> Peer access from NVIDIA A100-PCIE-40GB (GPU1) -> NVIDIA A100-PCIE-40GB (GPU0) : No

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 13.0, CUDA Runtime Version = 13.0, NumDevs = 2
Result = PASS

```

Some of you are sharing the same machine and some time measurements can be influenced by other users running at the very same moment. It can be necessary to run time measurements multiple times.


# CUDA Exercises

These short, self‑contained labs walk students from CUDA basics to a tiny predator–prey simulation.  
Every exercise follows the same workflow:

```
1. Fill in all  ►►► TODO ◄◄◄  and/or kernel bodies.
2. Build with the line shown in the banner comment.
3. Run → the program prints “…PASSED 🎉” when assertions succeed.
```

---

## Exercise 1 – CUDA Memory Model

*Goal – feel the separation between CPU (host) and GPU (device) address spaces.*

| Step | What you implement                                   |
| ---- | ---------------------------------------------------- |
|  1   | `cudaMallocAsync` two device buffers `d_a` and `d_b` |
|  2   | Copy host array `h_a → d_a`                          |
|  3   | Copy device array `d_a → d_b` (device‑to‑device)     |
|  4   | Copy back `d_b → h_a`                                |
|  5   | Free `d_a`, `d_b` with `cudaFreeAsync`               |

```bash
nvcc -std=c++20 memory_model.cu -o ex01
./ex01   # prints “Exercise 1 – memory model: PASSED 🎉”
```

### Variation

Add a non‑blocking version using **streams** + `cudaMemcpyAsync` and time a 100 MB H↔D copy to estimate PCIe bandwidth.

---

## Exercise 2 – Launch Your First Kernel

*Goal – understand grid/block configuration and indexing.*

1. `cudaMallocAsync` a device array `d_a[N]`
2. Launch a **1‑D grid** of **1‑D blocks**

   ```c++
   d_a[i] = i + 42;  // each thread writes one element
   ```
3. Copy back, verify, free the memory.

Compile & run:

```bash
nvcc -std=c++20 launch_kernel.cu -o ex02 && ./ex02
```

**Hint:** Global thread index = `blockIdx.x * blockDim.x + threadIdx.x`.

---

## Exercise 3 – 2‑D Grid & Block

*Goal – move from 1‑D to 2‑D indexing.*

Matrix **M\[numRows × numCols]**.

| Task | Detail                                                                                          |
| ---- | ----------------------------------------------------------------------------------------------- |
| 1    | Set `numRows`, `numCols` (start with 4 × 4, then 19 × 67).                                      |
| 2    | Launch a **2‑D grid of 2‑D blocks** so each thread writes<br>`M[row,col] = row * numCols + col` |
| 3    | Copy to host, assert correctness                                                                |
| 4    | Experiment: fix block = `16×16`, compute `blocksPerGrid` with ceiling division                  |

Compile:

```bash
nvcc -std=c++20 ex03_fill_matrix.cu -o ex03 && ./ex03
```

---

## Exercise 4 – Parallel Reduction ∑

*Goal – sum a 1‑D array faster than the CPU.*

> **Rule of thumb:** keep each block power‑of‑two and reduce in shared memory.

1. Kernel #1: each block loads its slice, does **shared‑mem tree reduction**, writes one partial sum.
2. Kernel #2: single block reduces those partials into the final total.
3. Copy result, compare to `std::accumulate`.

*Bonus*: one‑step reduction (single kernel).

---

## Parallel Challenge – **The Circle of Life**

A toroidal predator–prey world.  Build the starter **CPU version** first, then port to CUDA

Reference CPU build:

```bash
make serial
./circle_of_life --width 256 --height 256 --seed 42
```

Can you use the asynchronous GPU kernel launch to execute the generation of a git frame on the CPU while the GPU is running the next iteration?

<img src="simulation.gif" alt="circleoflife" title="Circle of Life" width="500" height="500" />

---

### Common Pitfalls & Tips

* **Always** `cudaGetLastError()` after a launch when debugging.
* Use **asserts** on the host to check results before optimising.
* Remember `cudaStreamSynchronize()` before timing or freeing async memory.
* `dim3` defaults `z=1`; you almost never need a non‑unit Z for these labs.
* For reductions, `blockDim.x` **must** be a power‑of‑two when you half the stride each step.

