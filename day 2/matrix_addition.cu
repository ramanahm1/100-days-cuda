#include <iostream>
#include <cmath>

__global__ void matrixAdd(const float* A, const float* B, float* C, int N) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx_x < N && idx_y < N) {
    C[idx_x * N + idx_y] = A[idx_x * N + idx_y] + B[idx_x * N + idx_y];
  }
}

int main() {
  int N = 10;
  float *A, *B, *C;

  A = (float *)malloc(sizeof(float) * N * N);
  B = (float *)malloc(sizeof(float) * N * N);
  C = (float *)malloc(sizeof(float) * N * N);

  for (int i = 0; i < N * N; i++) {
    A[i] = 1.0f;
    B[i] = 2.0f;
    C[i] = 0.0f;
  }

  float *d_a, *d_b, *d_c;

  cudaMalloc((void **)&d_a, N * N * sizeof(float));
  cudaMalloc((void **)&d_b, N * N * sizeof(float));
  cudaMalloc((void **)&d_c, N * N * sizeof(float));

  cudaMemcpy(d_a, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimBlock(32, 16); // Threads per block
  dim3 dimGrid(ceil(N / 32.0f), ceil(N / 16.0f));

  matrixAdd<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();

  cudaMemcpy(C, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  printf("A:\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f ", A[i * N + j]);
    }
    printf("\n");
  }

  printf("B:\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f ", B[i * N + j]);
    }
    printf("\n");
  }

  printf("RESULT:\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f ", C[i * N + j]);
    }
    printf("\n");
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(A);
  free(B);
  free(C);
}
