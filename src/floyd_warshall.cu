#include <iostream> // cout
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "floyd_warshall.h"

__global__ void kernel() {

}

__host__ void floyd_warshall_blocked_cuda() {
  std::cout << "We reached this far!\n";

  kernel<<<1,1>>>();

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

__global__ void floyd_warshall_kernel(int n, int* graph) {

  int in = (threadIdx.x + blockIdx.x * blockDim.x) * n;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int inj = in + j;

  for (int k = 0; k < n; k++) {
    if (graph[inj] > graph[in + k] + graph[k*n + j]) {
      graph[inj] = graph[in + k] + graph[k*n + j];
    }
    __syncthreads();
  }

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

  int* device_graph;

  const size_t size = sizeof(int) * n;

  cudaMalloc(&device_graph, size);

  cudaMemcpy(input, device_graph, size, cudaMemcpyHostToDevice);

  dim3 block_dim(32, 32, 1);
  dim3 grid_dim((n + block_dim.x - 1) / block_dim.x,
                (n + block_dim.y - 1) / block_dim.y);

  floyd_warshall_kernel<<<grid_dim, block_dim>>>(n, device_graph);
  cudaThreadSynchronize();

  cudaMemcpy(device_graph, output, size, cudaMemcpyDeviceToHost);

  cudaError_t errCode = cudaPeekAtLastError();
  if (errCode != cudaSuccess) {
      std::cerr << "WARNING: A CUDA error occured: code=" << errCode << "," <<
                cudaGetErrorString(errCode) << "\n";
  }

  cudaFree(device_graph);

}
