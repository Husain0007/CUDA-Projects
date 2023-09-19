# Tutorial 1
CUDA (**Compute Unified Device Architecture**) is a platform for programming CUDA-enabled GPUs. In CUDA Programming, both CPUs and GPUs are used for computing. The CPUS is refered to as the *host*, while the GPU is refered to as the *device*. CPU and GPU are separate platforms wiht their own memory space. Typically we run serial workloads on CPUs and offload parallel computation to GPUs.
- - -
## Comparision between C & CUDA

The major difference between C and CUDA implementation is `__global__` specifier and `<<<...>>>` syntax. The `__global__` specifier indicates a function that runs on device (GPU). Such a function can be called through host code, ie; the `main()` function. The device function is also called the **kernel**. The kernel executation configuration `<<<1,1>>>` indicates that the kernel is launched with only 1 thread. 
- - -
## Compiling CUDA programs
NVIDIA provides a CUDA compiler called `nvcc` in the CUDA toolkit to compile CUDA code, typically stored in a file with extension `.cu`. 
```
    nvcc hello_world.cu -o hello
```
- - -
## Processing Vector Addition in Device Memory
For data to be accessible by GPU, it must be presented in the device memory. CUDA provides APIs for allocating device memory and data transfer betweek host and device memory. Following is the common workflow of CUDA programs. 
1. Allocate host memory and initialize host data.
2. Allocate device memory.
3. Transfer input data from host to device memory.
4. Execute kernels.
5. Transfer output from device memory to host memory.  
### Device Memory Management
Use `cudaMalloc()` and `cudaFree()` to allocate memory device memory. 
```
cudaMalloc(void **devPtr, size_T count);
cudaFree(void *devPtr);
```
`cudaMalloc()` allocates memory of size `count` in the device memory and updates the device pointer `devPtr` to the allocated memory. `cudaFree()` deallocates a region of the device memory where the device pointer `devPtr` points to. These are comparable to `malloc()` and `free()` in C. 

### Memory Transfer

Tranferring data between host and device memory ccan be done through `cudaMemcpy` function, which is similar to `memcpy` in C. The syntax of `cudaMemcpy` is as follow
```
cudaMemcpy(void *dst, void *src, size_t count, cudaMemcpyKind kind)
```
The function copies a memory of size `count` from `src` to `dst`. `kind` indicates the direction. Typically `kind` has the values `cudaMemcpyHostToDevice` or `cudaMemcpyDeviceToHost`. 

### Running a CUDA Program

```
    nvcc add.cu -o add_cuda
    ./add_cuda
```

### Profile for execution time
To find out how long the kernel takes to runs jus tun it with `nvprof`.
```
nvprof ./add_cuda
```

### Picking up Threads

The **execution configuration** `<<<1,1>>>` tells the CUDA runtime how many parallel threads to use for the launch on the GPU. The second parameter is is the number of threads in a thread block. CUDA GPUs run kernels using blocks of threads that re multiples of 32 in size.
```
add<<<1, 256>>>(N, x, y)
```
For specificity we can mention the thread index within a vlock by `threadIdx.x` where `x`x is the variable name. `blockDim.x` contains the number of threads in the block. 

The first parameter of the **execution configuration** refers to hte number of thread blocks, together the block of parallel threads make up what is known as the grid. For example if we want to use the `add` function to add `N` elements of two vectors in parallel we simply need to calculate the appropriate number of blocks with 256 threads each. A block can have upto 1024 parallel threads.
```
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;
add<<<numBlocks, blockSize>>>(N, x, y);
```
CUDA provides `gridDim.x` which contains the number of blocks in the grid, and `blockIdx.x` which contains the index of the current thread block in the grid. The below kernel function is adapted to work on a block and thread level.
```
__global__
void add (int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i= index, i < n; I+= stride)
        y[i] = x[i] + y[i];
}
```

The above is a **grid-style** loop. 

### Resources Used
* [An Even Easier Introduction to CUDA by Mark Harris](https://developer.nvidia.com/blog/even-easier-introduction-cuda/).
* [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
* [CUDA Tutorial](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/).
* [Udacity: Introduction to Parallel Programming](https://www.youtube.com/playlist?list=PLAwxTw4SYaPm0z11jGTXRF7RuEEAgsIwH).
* [Introduction to Parallel Programming with CUDA](https://www.coursera.org/learn/introduction-to-parallel-programming-with-cuda/home/week/1).


