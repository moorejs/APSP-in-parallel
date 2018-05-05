#include <iostream>
#include <algorithm>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "floyd_warshall.hpp"

#define BLOCK_DIM 16

__forceinline__
__host__ void check_cuda_error() {
  cudaError_t errCode = cudaPeekAtLastError();
  if (errCode != cudaSuccess) {
    std::cerr << "WARNING: A CUDA error occured: code=" << errCode << "," <<
                cudaGetErrorString(errCode) << "\n";
  }
}

__forceinline__
__device__ void calc(int* graph, int n, int k, int i, int j) {
  if ((i >= n) || (j >= n) || (k >= n)) return;
  const unsigned int kj = k*n + j;
  const unsigned int ij = i*n + j;
  const unsigned int ik = i*n + k;
  int t1 = graph[ik] + graph[kj];
  int t2 = graph[ij];
  graph[ij] = (t1 < t2) ? t1 : t2;
}


__global__ void floyd_warshall_kernel(int n, int k, int* graph) {
  const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
  calc(graph, n, k, i, j);
}

/*****************************************************************************
                         Blocked Floyd-Warshall Kernel
  ***************************************************************************/

__forceinline__
__device__ void block_calc(int* C, int* A, int* B, int bj, int bi) {
  for (int k = 0; k < BLOCK_DIM; k++) {
    int sum = A[bi*BLOCK_DIM + k] + B[k*BLOCK_DIM + bj];
    if (C[bi*BLOCK_DIM + bj] > sum) {
      C[bi*BLOCK_DIM + bj] = sum;
    }
    __syncthreads();
  }
}

__global__ void floyd_warshall_block_kernel_phase1(int n, int k, int* graph) {
  const unsigned int bi = threadIdx.y;
  const unsigned int bj = threadIdx.x;

  __shared__ int C[BLOCK_DIM * BLOCK_DIM];

  __syncthreads();

  // Transfer to temp shared arrays
  C[bi*BLOCK_DIM + bj] = graph[k*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj];

  __syncthreads();
  
  block_calc(C, C, C, bi, bj);

  __syncthreads();

  // Transfer back to graph
  graph[k*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj] = C[bi*BLOCK_DIM + bj];

}


__global__ void floyd_warshall_block_kernel_phase2(int n, int k, int* graph) {
  // BlockDim is one dimensional (Straight along diagonal)
  // Blocks themselves are two dimensional
  const unsigned int i = blockIdx.x;
  const unsigned int bi = threadIdx.y;
  const unsigned int bj = threadIdx.x;

  if (i == k) return;

  __shared__ int A[BLOCK_DIM * BLOCK_DIM];
  __shared__ int B[BLOCK_DIM * BLOCK_DIM];
  __shared__ int C[BLOCK_DIM * BLOCK_DIM];

  __syncthreads();

  C[bi*BLOCK_DIM + bj] = graph[i*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj];
  B[bi*BLOCK_DIM + bj] = graph[k*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj];

  __syncthreads();

  block_calc(C, C, B, bi, bj);

  __syncthreads();

  graph[i*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj] = C[bi*BLOCK_DIM + bj];

  // Phase 2 1/2

  C[bi*BLOCK_DIM + bj] = graph[k*BLOCK_DIM*n + i*BLOCK_DIM + bi*n + bj];
  A[bi*BLOCK_DIM + bj] = graph[k*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj];

  __syncthreads();

  block_calc(C, A, C, bi, bj);

  __syncthreads();

  // Block C is the only one that could be changed
  graph[k*BLOCK_DIM*n + i*BLOCK_DIM + bi*n + bj] = C[bi*BLOCK_DIM + bj];
}


__global__ void floyd_warshall_block_kernel_phase3(int n, int k, int* graph) {
  // BlockDim is one dimensional (Straight along diagonal)
  // Blocks themselves are two dimensional
  const unsigned int j = blockIdx.x;
  const unsigned int i = blockIdx.y;
  const unsigned int bi = threadIdx.y;
  const unsigned int bj = threadIdx.x;

  if (i == k && j == k) return;
  __shared__ int A[BLOCK_DIM * BLOCK_DIM];
  __shared__ int B[BLOCK_DIM * BLOCK_DIM];
  __shared__ int C[BLOCK_DIM * BLOCK_DIM];

  __syncthreads();

  C[bi*BLOCK_DIM + bj] = graph[i*BLOCK_DIM*n + j*BLOCK_DIM + bi*n + bj];
  A[bi*BLOCK_DIM + bj] = graph[i*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj];
  B[bi*BLOCK_DIM + bj] = graph[k*BLOCK_DIM*n + j*BLOCK_DIM + bi*n + bj];

  __syncthreads();

  block_calc(C, A, B, bi, bj);

  __syncthreads();

  graph[i*BLOCK_DIM*n + j*BLOCK_DIM + bi*n + bj] = C[bi*BLOCK_DIM + bj];
}

/************************************************************************
                    Floyd-Warshall's Algorithm CUDA
************************************************************************/


__host__ void floyd_warshall_blocked_cuda(int* input, int* output, int n) {

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);

    std::cout << "Device " << i << ": " << deviceProps.name << "\n"
	      << "\tSMs: " << deviceProps.multiProcessorCount << "\n"
	      << "\tGlobal mem: " << static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024 * 1024) << "GB \n"
	      << "\tCUDA Cap: " << deviceProps.major << "." << deviceProps.minor << "\n";
  }

  int* device_graph;
  const size_t size = sizeof(int) * n * n;
  cudaMalloc(&device_graph, size);
  cudaMemcpy(device_graph, input, size, cudaMemcpyHostToDevice);

  const int blocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;
  dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);
  dim3 phase4_grid(blocks, blocks, 1);

  std::cout << "Launching Kernels Blocks: " << blocks << " Size " << n << "\n";
  for (int k = 0; k < blocks; k++) {
    floyd_warshall_block_kernel_phase1<<<1, block_dim>>>(n, k, device_graph);

    floyd_warshall_block_kernel_phase2<<<blocks, block_dim>>>(n, k, device_graph);

    floyd_warshall_block_kernel_phase3<<<phase4_grid, block_dim>>>(n, k, device_graph);
  }
  
  cudaMemcpy(output, device_graph, size, cudaMemcpyDeviceToHost);
  check_cuda_error();

  cudaFree(device_graph);
}

__host__ void floyd_warshall_cuda(int* input, int* output, int n) {

  // from assignment 1
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);

    std::cout << "Device " << i << ": " << deviceProps.name << "\n"
	      << "\tSMs: " << deviceProps.multiProcessorCount << "\n"
	      << "\tGlobal mem: " << static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024 * 1024) << "GB \n"
	      << "\tCUDA Cap: " << deviceProps.major << "." << deviceProps.minor << "\n";
  }

  std::cout << "Size " << n << "\n";

  int* device_graph;

  const size_t size = sizeof(int) * n * n;

  cudaMalloc(&device_graph, size);

  cudaMemcpy(device_graph, input, size, cudaMemcpyHostToDevice);

  dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);
  dim3 grid_dim((n + block_dim.x - 1) / block_dim.x,
                (n + block_dim.y - 1) / block_dim.y);
  
  for (int k = 0; k < n; k++) {
    floyd_warshall_kernel<<<grid_dim, block_dim>>>(n, k, device_graph);
    cudaThreadSynchronize();
  }

  cudaMemcpy(output, device_graph, size, cudaMemcpyDeviceToHost);

  cudaError_t errCode = cudaPeekAtLastError();
  if (errCode != cudaSuccess) {
      std::cerr << "WARNING: A CUDA error occured: code=" << errCode << "," <<
                cudaGetErrorString(errCode) << "\n";
  }

  cudaFree(device_graph);

}
