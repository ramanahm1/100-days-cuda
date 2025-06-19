#include <iostream>
#include <cmath>

__global__ void vecMatMul(float *A, float *B, float *C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    float summ = 0.0f;
    for (int j = 0; j < N; j++) {
      summ += A[i * N + j] * B[j];
    }
    C[i] = summ;
  }
}

int main() {
  int N = 10;

  float *A, *B, *C;

  A = (float *)malloc(N * N * sizeof(float));
  B = (float *)malloc(N * sizeof(float));
  C = (float *)malloc(N * sizeof(float));

  // Fill A, B, C;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = 1.0f;
    }
    B[i] = 2.0f;
    C[i] = 0.0f;
  }

  float *d_a, *d_b, *d_c;

  cudaMalloc((void **)&d_a, N * N * sizeof(float));
  cudaMalloc((void **)&d_b, N * sizeof(float));
  cudaMalloc((void **)&d_c, N * sizeof(float));


  cudaMemcpy(d_a, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, C, N*sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  vecMatMul<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

  cudaDeviceSynchronize();

  cudaMemcpy(C, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

  printf("A:\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f ", A[i * N + j]);
    }
    printf("\n");
  }

  printf("B:\n");
  for (int i = 0; i < N; i++) {
    printf("%f ", B[i]);
  }

  printf("C:\n");
  for (int i = 0; i < N; i++) {
    printf("%f ", C[i]);
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
