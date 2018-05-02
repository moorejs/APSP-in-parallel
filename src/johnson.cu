#include "johnson.hpp"

__device__ int min_distance(int* dist, char* visited, int n) {
  int min = INT_MAX;
  int min_index = 0;
  for (int v = 0; v < n; v++) {
    if (!visited[v] && dist[v] <= min) {
      min = dist[v];
      min_index = v;
    }
  }
  return min_index;
}

__global__ void dijkstra_kernel(int V, int E, edge_t* edge_array,
                                int* starts, int* weights, int* output,
                                char* visited_global) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= V) return;
  int* dist = &output[s * V];
  char* visited = &visited_global[s * V];
  for (int i = 0; i < V; i++) {
    dist[i] = INT_MAX;
    visited[i] = 0;
  }
  dist[s] = 0;
  for (int count = 0; count < V-1; count++) {
    int u = min_distance(dist, visited, V);
    int u_start = starts[u];
    int u_end = starts[u+1];
    visited[u] = 1;
    for (int v_i = u_start; v_i < u_end; v_i++) {
      int v = edge_array[v_i].v;
      if (!visited[v] && dist[u] != INT_MAX && dist[u] + weights[v_i] < dist[v])
          dist[v] = dist[u] + weights[v_i];
    }
  }

}

__global__ void bellman_ford_kernel(int E, int* dist,
                                    int* weights, edge_t* edges) {
  int e = threadIdx.x + blockDim.x * blockIdx.x;
  if (e >= E) return;
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

  cudaMemcpy(dist, device_dist, sizeof(int) * V, cudaMemcpyDeviceToHost);
  bool no_neg_cycle = true;

  // use OMP to parallelize. Not worth sending to GPU
  for (int i = 0; i < E; i++) {
    int u = edges[i].u;
    int v = edges[i].v;
    int weight = weights[i];
    if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
      no_neg_cycle = false;
  }

  cudaFree(device_dist);
  cudaFree(device_weights);
  cudaFree(device_edges);

  return no_neg_cycle;
}

__host__ void johnson_cuda(graph_cuda_t* gr, int* output) {

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

  int V = gr->V;
  int E = gr->E;

  graph_cuda_t* bf_graph = new graph_cuda_t;
  bf_graph->V = V + 1;
  bf_graph->E = gr->E + V;
  bf_graph->edge_array = new edge_t[bf_graph->E];
  bf_graph->weights = new int[bf_graph->E];

  std::memcpy(bf_graph->edge_array, gr->edge_array, gr->E * sizeof(edge_t));
  std::memcpy(bf_graph->weights, gr->weights, gr->E * sizeof(int));
  std::memset(&bf_graph->weights[gr->E], 0, V * sizeof(int));

  int* h = new int[bf_graph->V];
  bool r = bellman_ford_cuda(bf_graph, h, V);
  if (!r) {
    std::cerr << "\nNegative Cycles Detected! Terminating Early\n";
    exit(1);
  }
  for (int e = 0; e < E; e++) {
    int u = gr->edge_array[e].u;
    int v = gr->edge_array[e].v;
    gr->weights[e] = gr->weights[e] + h[u] - h[v];
  }

  int threads = 1024;
  int blocks = (V + threads - 1) / threads;

  // Structure of the graph
  edge_t* device_edge_array;
  int* device_weights;
  int* device_output;
  int* device_starts;
  // Needed to run dijkstra
  char* device_visited;
  // Allocating memory
  cudaMalloc(&device_edge_array, sizeof(edge_t) * E);
  cudaMalloc(&device_weights, sizeof(int) * E);
  cudaMalloc(&device_output, sizeof(int) * V * V);
  cudaMalloc(&device_visited, sizeof(char) * V * V);
  cudaMalloc(&device_starts, sizeof(int) * (V + 1));

  cudaMemcpy(device_edge_array, gr->edge_array, sizeof(edge_t) * E,
                                                cudaMemcpyHostToDevice);
  cudaMemcpy(device_weights, gr->weights, sizeof(int) * E, cudaMemcpyHostToDevice);
  cudaMemcpy(device_starts, gr->starts, sizeof(int) * (V+1), cudaMemcpyHostToDevice);

  std::cout << "Launching Kernel\n";
  dijkstra_kernel<<<blocks, threads>>>(V, E, device_edge_array, device_starts,
                                        device_weights, device_output,
                                        device_visited);

  cudaError_t errCode2 = cudaPeekAtLastError();
  if (errCode2 != cudaSuccess) {
    std::cerr << "WARNING: A CUDA error occured: code=" << errCode2 << "," <<
                cudaGetErrorString(errCode2) << "\n";
  }
  std::cout << "Kernel Finished\n";
  cudaMemcpy(output, device_output, sizeof(int) * V * V, cudaMemcpyDeviceToHost);


  cudaError_t errCode3 = cudaPeekAtLastError();
  if (errCode3 != cudaSuccess) {
    std::cerr << "WARNING: A CUDA error occured: code=" << errCode3 << "," <<
                cudaGetErrorString(errCode3) << "\n";
  }
  // REMEMBER TO REWEIGHT ALL EDGES

  cudaFree(device_edge_array);
  cudaFree(device_weights);
  cudaFree(device_output);
  cudaFree(device_starts);
  cudaFree(device_visited);

}

