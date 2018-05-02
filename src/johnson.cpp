#include <iostream> // cerr
#include <random> // mt19937_64, uniform_x_distribution
#include <vector>
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>

#include "johnson.hpp"

graph_t *johnson_init(const int n, const double p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  static std::uniform_int_distribution<int> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  int *adj_matrix = new int[n * n];
  size_t E = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        adj_matrix[i*n + j] = 0;
      } else if (flip(rand_engine) < p) {
        adj_matrix[i*n + j] = choose_weight(rand_engine);
        E ++;
      } else {
        adj_matrix[i*n + j] = INT_MAX;
      }
    }
  }
  Edge *edge_array = new Edge[E];
  int *weights = new int[E];
  int ei = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (adj_matrix[i*n + j] != 0
          && adj_matrix[i*n + j] != INT_MAX) {
        edge_array[ei] = Edge(i,j);
        weights[ei] = adj_matrix[i*n + j];
        ei++;
      }
    }
  }

  delete[] adj_matrix;

  graph_t *gr = new graph_t;
  gr->V = n;
  gr->E = E;
  gr->edge_array = edge_array;
  gr->weights = weights;

  return gr;
}

#ifdef CUDA
void free_graph_cuda(graph_cuda_t *g) {
  delete[] g->edge_array;
  delete[] g->weights;
  delete g;
}

void set_edge(edge_t *edge, int u, int v) {
  edge->u = u;
  edge->v = v;
}

graph_cuda_t *johnson_cuda_init(const int n, const double p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  static std::uniform_int_distribution<int> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  int *adj_matrix = new int[n * n];
  size_t E = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        adj_matrix[i*n + j] = 0;
      } else if (flip(rand_engine) < p) {
        adj_matrix[i*n + j] = choose_weight(rand_engine);
        E ++;
      } else {
        adj_matrix[i*n + j] = INT_MAX;
      }
    }
  }
  edge_t *edge_array = new edge_t[E];
  int* starts = new int[n + 1];  // Starting point for each edge
  int* weights = new int[E];
  int ei = 0;
  for (int i = 0; i < n; i++) {
    starts[i] = ei;
    for (int j = 0; j < n; j++) {
      if (adj_matrix[i*n + j] != 0
          && adj_matrix[i*n + j] != INT_MAX) {
        set_edge(&edge_array[ei], i, j);
        weights[ei] = adj_matrix[i*n + j];
        ei++;
      }
    }
  }
  starts[n] = ei; // One extra

  delete[] adj_matrix;

  graph_cuda_t *gr = new graph_cuda_t;
  gr->V = n;
  gr->E = E;
  gr->edge_array = edge_array;
  gr->weights = weights;
  gr->starts = starts;

  return gr;
}

void free_cuda_graph(graph_cuda_t* g) {
  delete[] g->edge_array;
  delete[] g->weights;
  delete[] g->starts;
  delete g;
}

#endif

void free_graph(graph_t* g) {
  delete[] g->edge_array;
  delete[] g->weights;
  delete g;
}

inline bool bellman_ford(graph_t* gr, int* dist, int src) {
  int V = gr->V;
  int E = gr->E;
  Edge* edges = gr->edge_array;
  int* weights = gr->weights;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < V; i++) {
    dist[i] = INT_MAX;
  }
  dist[src] = 0;


  for (int i = 1; i <= V-1; i++) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int j = 0; j < E; j++) {
      int u = std::get<0>(edges[j]);
      int v = std::get<1>(edges[j]);
      int new_dist = weights[j] + dist[u];
      if (dist[u] != INT_MAX && new_dist < dist[v])
        dist[v] = new_dist;
    }
  }

  bool no_neg_cycle = true;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < E; i++) {
    int u = std::get<0>(edges[i]);
    int v = std::get<1>(edges[i]);
    int weight = weights[i];
    if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
      no_neg_cycle = false;
  }
  return no_neg_cycle;
}

void johnson_parallel(graph_t* gr, int* output) {

  int V = gr->V;

  // Make new graph for Bellman-Ford
  // First, a new node q is added to the graph, connected by zero-weight edges
  // to each of the other nodes.
  graph_t* bf_graph = new graph_t;
  bf_graph->V = V + 1;
  bf_graph->E = gr->E + V;
  bf_graph->edge_array = new Edge[bf_graph->E];
  bf_graph->weights = new int[bf_graph->E];

  std::memcpy(bf_graph->edge_array, gr->edge_array, gr->E  * sizeof(Edge));
  std::memcpy(bf_graph->weights, gr->weights, gr->E * sizeof(int));
  std::memset(&bf_graph->weights[gr->E], 0, V * sizeof(int));

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int e = 0; e < V; e++) {
    bf_graph->edge_array[e + gr->E] = Edge(V, e);
  }

  // Second, the Bellman–Ford algorithm is used, starting from the new vertex q,
  // to find for each vertex v the minimum weight h(v) of a path from q to v. If
  // this step detects a negative cycle, the algorithm is terminated.
  // TODO Can run parallel version?
  int* h = new int[bf_graph->V];
  bool r = bellman_ford(bf_graph, h, V);
  if (!r) {
    std::cerr << "\nNegative Cycles Detected! Terminating Early\n";
    exit(1);
  }
  // Next the edges of the original graph are reweighted using the values computed
  // by the Bellman–Ford algorithm: an edge from u to v, having length
  // w(u,v), is given the new length w(u,v) + h(u) − h(v).
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int e = 0; e < gr->E; e++) {
    int u = std::get<0>(gr->edge_array[e]);
    int v = std::get<1>(gr->edge_array[e]);
    gr->weights[e] = gr->weights[e] + h[u] - h[v];
  }

  Graph G(gr->edge_array, gr->edge_array + gr->E, gr->weights, V);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int s = 0; s < V; s++) {
    std::vector<int> d(num_vertices(G));
    dijkstra_shortest_paths(G, s, distance_map(&d[0]));
    for (int v = 0; v < V; v++) {
      output[s*V + v] = d[v] + h[v] - h[s];
    }
  }

  delete[] h;
  free_graph(bf_graph);
}
