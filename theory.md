# Writing our first CUDA program

## **1️⃣ Introduction**
CUDA (Compute Unified Device Architecture) is NVIDIA’s parallel computing platform and API for general-purpose GPU computing. It enables programmers to leverage the **massive parallelism** of GPUs for accelerating computations.

In this guide, we will:
- Write and understand our **first CUDA program** (`Hello, World!`).
- Learn about **CUDA thread hierarchy**.
- Explore **CUDA timing mechanisms** for performance analysis.

---
## **2️⃣ Hello World Program in CUDA**

```cpp
__global__ void hello_world() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello, World! from GPU thread %d\n", idx);
}

int main() {

    // Run GPU version with timing
    std::cout << "\n=== Running on GPU ===\n";
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);  // Start GPU timing
    hello_kernel<<<4, 4>>>(); // Launch kernel with 4 blocks of 4 threads
    cudaDeviceSynchronize();  // Wait for GPU execution to finish
    cudaEventRecord(stop);   // Stop GPU timing

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "GPU Execution Time: " << milliseconds / 1000.0 << " seconds\n";

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
```

In the above program, each GPU thread runs this function in parallel, meaning multiple threads will print "Hello, World!" at the same time.

## **3️⃣ CUDA Functions Explained**

| **CUDA Function**               | **Purpose**                                                 |
|----------------------------------|--------------------------------------------------------    |
| `__global__`                    | Defines a CUDA kernel (a function that runs on the GPU)     |
| `threadIdx.x`                   | The index (position) of the current thread within a block   |
| `blockIdx.x`                    | The index (position) of the block in the grid                           |
| `blockDim.x`                    | Total Number of threads per block                           |
| `cudaEventCreate(&event)`       | Creates a CUDA event for timing                        |
| `cudaEventRecord(event)`        | Starts/stops timing for GPU execution                  |
| `cudaDeviceSynchronize()`       | Ensures GPU has completed all work before proceeding   |
| `cudaEventElapsedTime(&time, start, stop)` | Calculates execution time in milliseconds             |
| `cudaEventDestroy(event)`       | Cleans up CUDA events to free memory                   |

Formula of Global Thread ID: `idx = threadIdx.x + blockIdx.x * blockDim.x` -> This calculates the global thread ID for parallel execution.

## **4️⃣ CUDA Hierarchy: Understanding GPU Threading**

CUDA follows a hierarchical execution model that organizes threads into blocks and blocks into a grid and through this hierarchical structure it defines how threads are organized and executed. Here's an explanation:

### 1. GPU Architecture: Cores and Multiprocessors
GPU cores: GPUs have thousands of small processing cores that can execute computations simultaneously. These cores are divided into Streaming Multiprocessors (SMs).

Example: An NVIDIA GPU might have 80 SMs, each containing 128 cores.
Total cores = 80 SMs × 128 cores/SM = 10240 cores
Unlike a CPU (with a few powerful cores), a GPU has many lightweight cores, designed for massive parallelism.

### 2. CUDA Execution Hierarchy: Hierarchical Thread Organization
CUDA organizes threads into blocks and blocks into a grid. The execution model has three levels:

| Level   | Description                                              |
|---------|----------------------------------------------------------|
| Grid    | A collection of blocks.                                  |
| Block   | A collection of threads.                                 |
| Thread  | A single execution unit that performs a specific computation. |

CUDA's thread-block-grid hierarchy maps onto the GPU's physical architecture to leverage its parallelism:

**Threads:**
- A thread is the smallest execution unit in CUDA.
- Each thread performs one piece of computation (e.g., adding two numbers, processing one pixel in an image, etc.).

**Block:**
- A block is a group of threads that execute together.
- Threads in a block share resources (e.g., shared memory) and are assigned to the same Streaming Multiprocessor (SM).

**Grid:**
- A grid is a collection of blocks.
- Blocks in a grid execute independently and are distributed across multiple SMs.

### 3. Thread Identification
Each thread in the grid needs a unique ID to determine which portion of the data it should process. CUDA uses indices to achieve this:

Within a Block
`threadIdx.x`: The thread's unique index within a single block (range: 0 to blockDim.x - 1).
Example: If a block has 128 threads, threadIdx.x will range from 0 to 127.
Across Blocks
`blockIdx.x`: The block's unique index within the grid (range: 0 to gridDim.x - 1).
Example: If there are 10 blocks in the grid, blockIdx.x will range from 0 to 9.

### 4. Example Calculation
Let’s say:

Number of Threads per Block (`blockDim.x`) = 128
Number of Blocks (`gridDim.x`) = 4
Blocks and Threads:
Each block has 128 threads.
A total of 4 × 128 = 512 threads exist in the grid.
## Global Thread IDs

| **Block ID (blockIdx.x)** | **Thread Range (threadIdx.x)** | **Global Thread ID (idx)** |
|---------------------------|-------------------------------|----------------------------|
| Block 0                  | Threads 0 - 127              | 0 - 127                   |
| Block 1                  | Threads 0 - 127              | 128 - 255                 |
| Block 2                  | Threads 0 - 127              | 256 - 383                 |
| Block 3                  | Threads 0 - 127              | 384 - 511                 |

For Thread 130, for example:

`blockIdx.x` = 1 (it’s in Block 1).
`threadIdx.x` = 2 (it’s the 3rd thread within Block 1).
Therefore, Global Thread ID, idx = 130.

### 5. Threads and Cores: How They Map

#### Step-by-Step:
1. **GPU assigns threads to cores**:
   - Each thread is mapped to a core of the GPU.
   - A single Streaming Multiprocessor (SM) can execute multiple threads simultaneously (using CUDA's thread scheduler).

2. **Blocks are mapped to SMs**:
   - A block is assigned to an SM, and the threads within the block are distributed to the cores within that SM.

3. **Grids span multiple SMs**:
   - If there are many blocks in the grid, they are distributed across multiple SMs, allowing the GPU to execute thousands (or millions) of threads in parallel.

**Example:** Suppose:
- Each block has **256 threads**.
- The grid contains **100 blocks**.

**Total threads** =  256 threads/block × 100 blocks = 25600 threads.

If the GPU has **10240 cores**:
- In one cycle, the GPU will execute **10240 threads in parallel**.
- Remaining threads will wait in a queue (**warps**), and CUDA's scheduler will allocate them in subsequent cycles.


### 5. Advantages of This Hierarchy
Parallelism: Each thread works independently on its own portion of the data.
Scalability: By increasing grid and block dimensions, more threads can be used for larger data sets.
Flexibility: We can control the number of blocks and threads to balance performance and resource usage.


### Importance of `cudaDeviceSynchronize()`
`cudaDeviceSynchronize()` is a CUDA function that ensures the GPU has completed all its work (kernels and memory operations) before the CPU continues execution.
By default, CUDA kernel launches and memory operations are asynchronous, meaning they return control to the CPU immediately without waiting for the GPU to finish. This is efficient but can lead to incorrect results if synchronization is not handled properly.
When should you use it?
- If you need accurate GPU execution time measurements.
- If you access GPU-computed results from the CPU after kernel execution.

