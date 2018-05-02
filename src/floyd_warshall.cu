#include <iostream> // cout
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>

#include "floyd_warshall.hpp"

#define THREADS_DIM 32

__host__ void floyd_warshall_blocked_cuda() {
  std::cout << "We reached this far!\n";


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

  dim3 block_dim(THREADS_DIM, THREADS_DIM, 1);
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
