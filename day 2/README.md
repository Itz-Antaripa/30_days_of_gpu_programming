# DAY 2: GPU Parallelism and Triton Fundamentals 🚀

Today, we will be working on Triton GPU kernels to perform vector addition efficiently. For that whatever theoretical understanding is required, we will be covering that here.

We will be covering today these key topics:
- How GPUs work (SIMT Model)
- Threading concepts (Thread Indexing, Block Indexing)
- CUDA Memory Types: Global, Shared, Local, Register, Constant, and Texture Memory
- Memory access optimization (Coalescing, Throughput, Contiguous Access)
- Triton-specific execution model and work balancing
- CUDA memory management (cudaMalloc, cudaMemcpy, and error handling)

Understanding these concepts is crucial for writing high-performance GPU kernels and ensuring efficient execution.

---

## 1. How GPUs Execute Threads: Understanding SIMT Execution Model

### 1.1 Why GPUs Are Different from CPUs
Unlike CPUs that are optimized for low-latency, sequential execution, GPUs are designed for high-throughput execution with thousands of parallel threads.

A CPU core executes one instruction per cycle per thread. In contrast, a GPU core (CUDA Core) executes the same instruction on multiple threads simultaneously using **SIMT (Single Instruction, Multiple Threads)**.

Example
```cpp
// CUDA kernel for vector addition
__global__ void vector_add(float *A, float *B, float *C, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        C[index] = A[index] + B[index];
    }
}
```
- Each thread computes one element of the vector.
- Thousands of threads execute this in parallel.

### 1.2 SIMT (Single Instruction, Multiple Threads)
A warp (group of 32 threads) executes the same instruction in parallel but on different pieces of data.
This is not the same as **SIMD (Single Instruction, Multiple Data)** because each thread has its own register state and can execute different instructions if necessary (though it reduces efficiency).

---

## 2. Threading Concepts: Thread Indexing and Block Indexing  

### 2.1 Thread Hierarchy in CUDA  

CUDA organizes threads into a hierarchical structure:  

| Level            | Description                                        |
|-----------------|----------------------------------------------------|
| **Thread**      | Smallest execution unit, operates on a subset of data |
| **Warp** (32 threads) | A group of 32 threads executed in lockstep |
| **Thread Block** | A collection of warps that share shared memory |
| **Grid**        | A collection of blocks executing a kernel |

---

### 2.2 Thread Indexing  

Each thread in CUDA has a **unique index** to determine which part of the data it should process.  

**For a 1D grid:**  
Each thread's **global index** is calculated as:  
```cpp
int index = blockIdx.x * blockDim.x + threadIdx.x;
```

`threadIdx.x`: Thread index within a block
`blockIdx.x`: Block index within the grid
`blockDim.x`: Total threads in a block

**For a 2D grid:**
Threads are indexed as:
```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

---

## 3. CUDA Memory Types and Their Importance  

### 3.1. Memory Hierarchy in GPUs  

CUDA provides different types of memory, each optimized for specific use cases. Understanding this hierarchy is crucial for writing efficient GPU programs.   

| **Memory Type**      | **Description** | **Scope** | **Best Used For** | **Speed** | **Size** | **Example Usage** |
|----------------------|----------------|-----------|--------------------|-----------|----------|------------------|
| **Register Memory**  | Fastest memory type, directly accessible by each thread. | Per-thread (private to each thread). | Storing frequently accessed variables and intermediate calculations. | Extremely fast but limited in size. | Smallest | Loop counters, temporary and frequently accessed variables. |
| **Shared Memory**    | Small but fast on-chip memory shared among threads within the same block. | Per-block (shared among threads in the same block). | Temporary storage of data that needs to be shared between threads in the same block. | Much faster than global memory because it is located on-chip. Reduces redundant global memory accesses. | Small | Data reuse within a block. Using shared memory to store intermediate results in matrix multiplication. |
| **Global Memory**    | Largest and most general-purpose memory space on the GPU. Accessible by all threads in a kernel. | Entire GPU (accessible by all threads across all blocks). | Storing large datasets that need to be shared across multiple thread blocks. | Relatively slow due to high latency. However, it has very high capacity. | Largest | Storing input data or output results for a computation. |
| **Local Memory**     | Private memory for each thread, used when variables cannot fit into registers. | Per-thread (private to each thread). | Automatically managed by the compiler for variables that do not fit in registers. | Slower than registers but faster than global memory. Resides off-chip. | Medium | Large arrays declared inside a kernel function that cannot fit into registers (Spillover from registers) |
| **Constant Memory**  | Small, read-only memory space optimized for broadcast reads. | Entire GPU (accessible by all threads, but read-only). | Best suited for data that does not change during kernel execution and is accessed uniformly by all threads. | Very fast when accessed uniformly by all threads in a warp. | Small | Lookup tables, configuration parameters, static, read-heavy data |
| **Texture Memory**   | Specialized memory designed for spatial locality and caching. | Entire GPU (accessible by all threads, but optimized for 2D spatial locality). | Commonly used in graphics applications for image processing tasks where spatial locality is important. | Optimized for 2D spatial locality, providing efficient caching for nearby memory accesses. | Medium | Image filtering operations. |


- L1 Cache sits between registers and shared memory, reducing latency when accessing frequently used data. Each SM (Streaming Multiprocessor) has its own private L1 cache.
- L2 Cache is shared across all SMs, improving repeated accesses to global memory.
- Global Memory is the slowest, but its performance can be improved by caching and coalescing memory accesses.

---

### 3.2. L1 and L2 Cache in CUDA

### L1 Cache (Private to Each Streaming Multiprocessor)  
- Each **SM (Streaming Multiprocessor)** has its own private **L1 cache**.  
- **Size**: Typically up to **128KB per SM** (configurable in some GPUs).  

**Purpose**: Reduces **register spills**. Speeds up **local variable access**. Caches **frequently accessed global memory**.  

**Programmable?** CUDA **does not** provide direct control over L1 cache. Some architectures allow **configuring L1 cache vs. shared memory split**.  

**Example: How L1 Cache Helps**  
Imagine **1000 threads** accessing array `A` in **global memory**.  
1. The **first access is slow** (**400-600 cycles**).  
2. The **next access is fast** because the value is **cached in L1** (**~10-30 cycles**).

---

### L2 Cache (Global Cache for All SMs)  
- **Shared across all SMs** (**not private like L1**).  
- **Size**: **1MB to 6MB**, depending on the GPU model.  

**Purpose**: Reduces **global memory latency** for repeated accesses. Improves **coalesced memory access efficiency**. Serves as a **write-back cache** for global memory.  

**Example: How L2 Cache Helps**  
1. **Thread 0** reads `A[0]` from **global memory** → **Slow (~400 cycles)**.  
2. **Thread 1** reads `A[0]` again → **Fast (~100 cycles, pulled from L2 cache)**.  

---

## 4. Coalescing, Throughput, and Optimizing Memory Access

### 4.1. What is Coalesced Memory Access?

- When consecutive threads access consecutive memory locations, memory accesses are coalesced.
- This reduces the number of memory transactions, improving throughput.

✅ Good (Coalesced Memory Access)
```c
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  // Consecutive memory access
a = tl.load(A_ptr + offsets, mask=mask)
```
Each thread accesses:
```css
Thread 0 → A[0], Thread 1 → A[1], Thread 2 → A[2] ...
```

❌ Bad (Strided Memory Access - Low Throughput)
```c
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) * 2  // Stride of 2
a = tl.load(A_ptr + offsets, mask=mask)
```
Each thread accesses:
```css
Thread 0 → A[0], Thread 1 → A[2], Thread 2 → A[4] ...
```
This results in multiple slow memory transactions instead of one efficient transaction.

### 4.2. How to Maximize Memory Throughput
Throughput = How many bytes the GPU can read/write per second.

**To Maximize Throughput:**
- Use coalesced memory access.
- Use shared memory to buffer global memory accesses.
- Avoid excessive register spilling into local memory.
- Ensure frequently accessed data is cached in L1 or L2.

---





