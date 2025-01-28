%%writefile hello_world.cu

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// CUDA Kernel
__global__ void hello_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello, World! from GPU thread %d in block %d\n", idx, blockIdx.x);
}


// CPU version
void hello_cpu() {
    auto start = std::chrono::high_resolution_clock::now(); // Start timer

    for (int i = 0; i < 16; i++) {
        std::cout << "Hello, World! from CPU thread " << i << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now(); // End timer
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "CPU Execution Time: " << elapsed.count() << " seconds\n";
}



int main() {
    std::cout << "Running CPU version...\n";
    hello_cpu();

    std::cout << "\nRunning GPU version...\n";
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);  // Start GPU timing
    hello_kernel<<<4, 4>>>();  // Launch kernel with 4 blocks of 4 threads
    cudaEventRecord(stop);   // Stop timing

    cudaDeviceSynchronize(); // Wait for GPU execution to finish

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    std::cout << "GPU Execution Time: " << elapsed_time / 1000.0 << " seconds\n";

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
