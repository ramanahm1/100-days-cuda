#include <iostream>

__global__ void scanWithSharedMemory(int *device_input, int *device_output, int total_elements) {
    extern __shared__ int local_buffer[];

    int local_id = threadIdx.x;
    int global_id = 2 * blockDim.x * blockIdx.x + local_id;

    if (global_id + blockDim.x < total_elements) {
        local_buffer[local_id] = device_input[global_id] + device_input[global_id + blockDim.x];
        __syncthreads();

        for (int step = 1; step < blockDim.x; step *= 2) {
            int temp = 0;
            if (local_id >= step) {
                temp = local_buffer[local_id - step];
            }
            __syncthreads();
            local_buffer[local_id] += temp;
            __syncthreads();
        }

        device_output[global_id] = local_buffer[local_id];
    }
}

int main() {
    const int NUM_ELEMENTS = 16;
    const int THREADS_PER_BLOCK = 8;

    int host_input[NUM_ELEMENTS]  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int host_output[NUM_ELEMENTS] = {0};

    int *gpu_input, *gpu_output;
    size_t mem_size = NUM_ELEMENTS * sizeof(int);

    cudaMalloc((void**)&gpu_input, mem_size);
    cudaMalloc((void**)&gpu_output, mem_size);

    cudaMemcpy(gpu_input, host_input, mem_size, cudaMemcpyHostToDevice);

    int blocks = NUM_ELEMENTS / THREADS_PER_BLOCK;
    int shared_memory_bytes = THREADS_PER_BLOCK * sizeof(int);
    scanWithSharedMemory<<<blocks, THREADS_PER_BLOCK, shared_memory_bytes>>>(gpu_input, gpu_output, NUM_ELEMENTS);

    cudaMemcpy(host_output, gpu_output, mem_size, cudaMemcpyDeviceToHost);

    printf("Original Array: ");
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        printf("%d ", host_input[i]);
    }

    printf("\nScanned Result: ");
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        printf("%d ", host_output[i]);
    }
    printf("\n");

    cudaFree(gpu_input);
    cudaFree(gpu_output);

    return 0;
}
