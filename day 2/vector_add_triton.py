import torch
import triton
import triton.language as tl


@triton.jit
def vec_add_triton(A_ptr,  # *Pointer* to first input vector.
                   B_ptr,  # *Pointer* to second input vector.
                   C_ptr,  # *Pointer* to output vector.
                   N,  # Size of the vector
                   BLOCK_SIZE: tl.constexpr, ):  # Number of elements each program should process.
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load data from global memory
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)

    # Compute and store result
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)


# Initialize input tensors
N = 98432
A_ptr = torch.arange(0, N, dtype=torch.float32, device='cuda')
B_ptr = 2 * A_ptr
C_ptr = torch.empty_like(A_ptr)

BLOCK_SIZE = 1024
grid = (triton.cdiv(N, BLOCK_SIZE),)  # dynamically compute the optimal grid size

# CUDA Events for accurate timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()  # Record start time

# Launch Triton kernel
vec_add_triton[grid](A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE=BLOCK_SIZE)  # gried specifies how many thread blocks to launch, BLOCK_SIZE specifies how many threads per block.

end_event.record()  # Record end time

torch.cuda.synchronize()  # Ensures all GPU execution is finished before measuring time

# Get elapsed time
elapsed_time = start_event.elapsed_time(end_event)  # Time in seconds
print(f"Kernel execution time: {elapsed_time/1000:.3f} sec")