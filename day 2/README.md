# DAY 2: GPU Parallelism and Triton Fundamentals

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
Unlike CPUs, which are optimized for low-latency, sequential execution, GPUs are designed for high-throughput execution with thousands of parallel threads.

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




