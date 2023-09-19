#include<stdio.h>
#include<stdlib.h>

#define N 32 // Each block can support upto 1024 blocks and operates on multiples of 32

__global__ void device_add_one_block_multiple_threads(int *a, int *b, int *c){
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

__global__ void device_add_multiple_blocks_one_thread(int *a, int *b, int *c){
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

__global__ void device_add_multiple_blocks_and_multiple_threads(int *a, int *b, int *c){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    c[index] = a[index] + b[index];
}

void host_add(int *a, int *b, int *c) {
    for(int idx = 0; idx < N; idx++){
        c[idx] = a[idx] + b[idx];
    }
}

// Fills Array with Index values
void fill_array(int *data) {
    for(int idx=0; idx < N; idx++){
        data[idx] = idx;
    }
}

void print_output(int *a, int *b, int *c){
    for(int idx = 0; idx < N; idx++){
        printf("\n %d + %d = %d", a[idx], b[idx], c[idx]);
    }
}

int main(void){
    int *a, *b, *c;
    int *d_a, *d_b, *d_c; //device copies of a, b, c

    int size = N * sizeof(int);

    // Allocate space for host copies of a, b, & c and pass input values
    a = (int *)malloc(size); 
    fill_array(a);
    b = (int *)malloc(size);
    fill_array(b);
    c = (int *)malloc(size);

    host_add(a,b,c);

    printf("Host Output\n");
    print_output(a, b, c);
    printf("\n");


    // Allocate space for device copies of a, b, & c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy input values to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    device_add_one_block_multiple_threads<<<1, N>>>(d_a, d_b, d_c);

    // Copy results back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToDevice);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    printf("Single Block and %d Threads output\n", N);
    print_output(a, b, c);
    printf("\n");

    device_add_multiple_blocks_one_thread<<<N, 1>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    printf("%d Blocks and single Thread output\n", N);
    print_output(a, b, c);
    printf("\n");

    int threads_per_block = 4;
    int no_of_blocks = 0;

    no_of_blocks = N/threads_per_block;

    device_add_multiple_blocks_and_multiple_threads<<<no_of_blocks, threads_per_block>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    printf("%d Blocks and %d Threads output\n", no_of_blocks, threads_per_block);
    print_output(a, b, c);
    printf("\n");

    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}