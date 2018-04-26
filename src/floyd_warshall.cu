#include <iostream> // cout

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
