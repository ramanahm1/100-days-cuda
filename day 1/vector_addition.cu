#include <iostream>

__global__ void addVectors(const float *v1, const float *v2, float *result, int N) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < N) {
      result[index] = v1[index] + v2[index];
    }
}

int main() {
  const int N = 10;
  float A[N], B[N], RES[N];

  float *device_A, *device_B, *device_RES;

  cudaMalloc(&device_A, N * sizeof(float));
  cudaMalloc(&device_B, N * sizeof(float));
  cudaMalloc(&device_RES, N * sizeof(float));


  for (int i = 0; i < N; i++) {
    A[i] = i;
    B[i] = i;
  }

  std::cout << "Initialized A[0] = " << A[0] << ", B[0] = " << B[0] << std::endl;

  cudaMemcpy(device_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_B, B, N * sizeof(float), cudaMemcpyHostToDevice);


  int blockSize = 256; // 32 * 2 (must be a multiple of 32)
  // int gridSize = ceil((float)N/blockSize);
  int gridSize = (N + blockSize - 1) / blockSize;
  addVectors<<<gridSize, blockSize>>>(device_A, device_B, device_RES, N);

  cudaDeviceSynchronize();


  cudaMemcpy(RES, device_RES, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(device_A);
  cudaFree(device_B);
  cudaFree(device_RES);

  for (int i = 0; i < N; i++) {
    printf("%f %f %f\n", A[i], B[i], RES[i]);
  }

  return 0;

}
