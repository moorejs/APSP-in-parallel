#include "johnson.hpp"

__global__ void johnson_kernel() {

}

__global__ void bellman_ford_kernel(int E, int* dist,
                                    int* weights, edge_t* edges) {
  int e = threadIdx.x + blockDim.x * blockIdx.x;
  int u = edges[e].u;
  int v = edges[e].v;
  int new_dist = weights[e] + dist[u];
  // Make ATOMIC
  if (dist[u] != INT_MAX && new_dist < dist[v])
    dist[v] = new_dist;
}

__host__ bool bellman_ford_cuda(graph_cuda_t* gr, int* dist, int s) {
  int V = gr->V;
  int E = gr->E;
  edge_t* edges = gr->edge_array;
  int* weights = gr->weights;

  // use OMP to parallelize. Not worth sending to GPU
  for (int i = 0; i < V; i++) {
    dist[i] = INT_MAX;
  }
  dist[s] = 0;

  int* device_dist;
  int* device_weights;
  edge_t* device_edges;

  cudaMalloc(&device_dist, sizeof(int) * V);
  cudaMalloc(&device_weights, sizeof(int) * E);
  cudaMalloc(&device_edges, sizeof(edge_t) * E);

  cudaMemcpy(device_dist, dist, sizeof(int) * V, cudaMemcpyHostToDevice);
  cudaMemcpy(device_weights, weights, sizeof(int) * E, cudaMemcpyHostToDevice);
  cudaMemcpy(device_edges, edges, sizeof(edge_t) * E, cudaMemcpyHostToDevice);

  int blocks = (E + 1024 - 1) / 1024;
  for (int i = 1; i <= V-1; i++) {
    bellman_ford_kernel<<<blocks, 1024>>>(E, device_dist,
                                        device_weights, device_edges);
    cudaThreadSynchronize();
  }

  bool no_neg_cycle = true;

  // use OMP to parallelize. Not worth sending to GPU
  for (int i = 0; i < E; i++) {
    int u = edges[i].u;
    int v = edges[i].v;
    int weight = weights[i];
    if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
      no_neg_cycle = false;
  }
  return no_neg_cycle;
}

__host__ void johnson_cuda(graph_t* gr, int* output, int n) {

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

}

