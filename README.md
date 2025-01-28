This is a very optimistic goal that I am setting for myself, currently, I don’t understand and know anything about CUDA, GPU programming, writing kernels and all. I want to learn. I don’t believe in the word getting “cracked”, but I definitely want to become better in certain skills which I feel will be important in the coming months or years. Let’s see how far I can go. I will document everything here in this repo.

my learning path (created using chatgpt and deepseek): https://www.notion.so/antaripasaha/Learning-GPU-Programming-30-days-plan-1885314a563980be95a6e9fd43c4c217

# 🚀 My 30-Day Journey into GPU Programming  

This repository documents my **30-day deep dive into GPU programming**, covering **CUDA, Triton, and PyTorch** with a focus on **high-performance computing, deep learning optimizations, and large-scale training techniques**.  

---

## **Why I'm Doing This**  
I wanted to go beyond using pre-built deep learning models and truly understand **how computations happen at the GPU level**. My goal is to:  
- Gain **deep theoretical and hands-on experience** in CUDA, Triton, and PyTorch.
- Cover CUDA Fundamentals: Memory management, parallelism, and kernel optimizations.
- Writing optimized GPU kernels with high-level abstractions.  
- Train and fine-tune **large models (LLMs, Transformers) efficiently**.  
- Master **performance profiling, debugging, and optimization** for GPU workloads.  
- Implement **real-world deep learning optimizations (FlashAttention, RoPE, LoRA, QLoRA, FSDP, sparse matrices)**.
- Understand **Inference acceleration**: Deploying models efficiently using ONNX, TensorRT, and quantization.  
- End with a **fully optimized Transformer model that runs efficiently on GPU**.

---

## **What This Repository Contains**  
Each day, I tackle a **specific GPU programming concept** and implement it from scratch.  

| **Week** | **Focus** |
|----------|--------------------------------------|
| **Week 1** | CUDA fundamentals, memory hierarchy, parallelism, and kernel optimization. |
| **Week 2** | Triton programming, memory-efficient operations, and kernel fusion. |
| **Week 3** | PyTorch GPU optimizations, FlashAttention, and large-scale training techniques. |
| **Week 4** | Sparse matrices, Fully Sharded Data Parallel (FSDP), LoRA, QLoRA, and efficient Transformer fine-tuning. |
| **Week 5** | Inference acceleration with TensorRT, ONNX, and quantization. |
| **Week 6** | Distributed training, multi-GPU strategies, and a final project. |

Each day consists of:  
✅ A **brief theoretical explanation** of the concept.  
✅ **Hands-on implementation** of a CUDA or Triton kernel.  
✅ **Experiments to analyze performance** and debug inefficiencies.  
✅ **Comparisons between CPU vs. GPU execution**.  
✅ **Optimization techniques** for real-world scenarios.  

---

## **How I'm Approaching This**  
### **1️⃣ Learning by Doing**
From **Day 1**, I start writing CUDA and Triton kernels—no waiting for "theory first, coding later." Every concept is reinforced with **experiments, performance tests, and real-world optimizations**.

### **2️⃣ Debugging & Profiling**
Understanding **why a GPU kernel is slow** is just as important as writing one. I profile every major implementation using **Nsight Systems, `torch.profiler`, and `cuda-memcheck`**.

### **3️⃣ Pushing the Limits**
I’m not just learning GPU programming—I’m applying it to **deep learning workloads**. I focus on **Transformer optimization, multi-GPU scaling, and inference efficiency**.

### **4️⃣ Ending with a Full Project**
This journey culminates in a **fully optimized Transformer model** incorporating:  
- **Custom CUDA and Triton kernels** for efficiency.  
- **Optimized memory management (FSDP, ZeRO, Sparse Matrices).**  
- **Deployment-ready inference optimizations (TensorRT, ONNX, Quantization).**  

---

## Where will I be running the code?

I have mac so nvidia cuda won't be supported there. I will be running on Google Colab, I have Colab Pro subscription
For cloud execution:
Open Google Colab.
Select Runtime → Change runtime type → GPU.

Check if GPU is there:
`!nvidia-smi`

Compile the CUDA program
`!nvcc file_name.cu -o file_name`

Run the program
`!./file_name`
