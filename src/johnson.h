#include <random> // mt19937_64, uniform_x_distribution
#include <boost/config.hpp>
#include <vector>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>

//using namespace std;
using namespace boost;

typedef adjacency_list<listS, vecS, directedS,
                                 no_property, property<edge_weight_t, int> > Graph;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef std::pair<int,int> Edge;

/*
typedef struct node {
  int weight;
  int u;

  node(int v, int w) : weight(w), u(v) {}
} node_t;

typedef struct compare_node {
  bool operator()(const node& n1, const node& n2) const {
    return n1.weight < n2.weight;
  }
} compare_node_t;

typedef struct Graph {
  int V;
  std::vector< tuple <int, int> >* edge_list;
} Graph_t; */


inline int* johnson_init(const int n, const int p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  // TODO: create negative edges without negative cycles
  static std::uniform_int_distribution<int> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  int *adj_matrix = new int[n * n];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        adj_matrix[i*n + j] = 0;
      } else if (flip(rand_engine) > p) {
        adj_matrix[i*n + j] = choose_weight(rand_engine);
      } else {
        adj_matrix[i*n + j] = 0;
      }
    }
  }

  return adj_matrix;
}

/*
inline Graph_t *johnson_init2(const int n, const int p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  static std::uniform_int_distribution<int> choose_weight(0, 100);

  std::mt19937_64 rand_engine(seed);

  int *adj_matrix = new int[n * n];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        adj_matrix[i*n + j] = 0;
      } else if (flip(rand_engine) > p) {
        adj_matrix[i*n + j] = choose_weight(rand_engine);
      } else {
        adj_matrix[i*n + j] = INT_MAX;
      }
    }
  }

  Graph_t *graph = new Graph_t;
  graph->V = n;
  graph->edge_list = new std::vector< tuple <int, int> >[n];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i != j && adj_matrix[i*n + j] != INT_MAX) {
        tuple <int, int> pair = std::make_tuple (j, adj_matrix[i*n + j]);
        graph->edge_list[i].push_back(pair);
      }
    }
  }
  delete[] adj_matrix;
  return graph;
} */

typedef struct graph {
  int V;
  int E;
  Edge *edge_array;
  int *weights;
} graph_t;

inline graph_t *johnson_init3(const int n, const int p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  static std::uniform_int_distribution<int> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  int *adj_matrix = new int[n * n];
  size_t E = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        adj_matrix[i*n + j] = 0;
      } else if (flip(rand_engine) > p) {
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
      if (adj_matrix[i*n + j] != 0 && adj_matrix[i*n + j] != INT_MAX) {
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

/*
inline bool bellman_ford(Graph *input, int *dist, int src) {
  int V = graph->V;
  int E = graph->E;

  for (int i = 0; i < V; i++) {
    dist[i] = INT_MAX;
  }
  dist[src] = 0;

  for (int i = 1; i < V; i++) {
    for (in j = 0; j < E; j++) {
      int u = graph->edge[j].src;
      int v = grpah->edge[j].dest;
      if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
          dist[v] = dist[u] + weight;
    }
  }

  for (int i = 0; i < E; i++) {
    it u = graph->edge[i].src;
    int v = graph->edge[i].dest;
    int weight = graph->edge[i].weight;
    if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
      std::cerr << "Graph contains a negative cycle!";
      return false;
    }
  }

  return true;
}
*/
/*
inline int min_distance(int dist[], bool sptSet[], int n) {
  int min = INT_MAX;
  int min_index = 0;

  for (int v = 0; v < n; v++) {
    if (sptSet[v] == false && dist[v] <= min) {
      min = dist[v];
      min_index = v;
    }
  }
  return min_index;
}

inline void dijkstra(int *input, int *dist, int src, int n) {
  bool sptSet[n];

  for (int i = 0; i < n; i++) {
    dist[i] = INT_MAX;
    sptSet[i] = false;
  }

  dist[src] = 0;
  for (int count = 0; count < n-1; count++) {
    int u = min_distance(dist, sptSet, n);
    int u_row = u * n;
    sptSet[u] = true;
    for (int v = 0; v < n; v++) {
      if (!sptSet[v] && input[u_row + v] && dist[u] != INT_MAX
                     && dist[u]+input[u_row + v] < dist[v]) {
        dist[v] = dist[u] + input[u_row + v];
      }

    }
  }
}


inline void dijkstra2(Graph_t *input, int *dist, int src) {
  int n = input->V;
  boost::heap::fibonacci_heap<node, boost::heap::compare<compare_node>> pq;

  for (int u = 0; u < n; u++) {
    dist[u] = INT_MAX;
  }
  dist[src] = 0;
  pq.push(node(src, INT_MIN));

  while (!pq.empty()) {
    node_t u = pq.top();
    pq.pop();
    std::vector< tuple <int, int> > neighbors = input->edge_list[u.u];
    for (tuple <int, int> edge : neighbors) {
      int v = std::get<0>(edge);
      int w = std::get<1>(edge);
      if (dist[v] > dist[u.u] + w) {
        dist[v] = dist[u.u] + w;
        pq.push(node(v, dist[v]));
      }
    }
  }
}

inline void johnson(int *input, int *output, int n) {
  // add temp source vertex s with edges to all other vertices
  // edges have weight 0
  //
  // run BF to get dist[]
  //
  // reweight edges

  for (int u = 0; u < n; u++) {
    dijkstra(input, &output[u*n], u, n);
  }
}

inline void johnson2(Graph_t *input, int *output, int n) {
  for (int s = 0; s < n; s++) {
    dijkstra2(input, &output[s*n], s);
  }
} */
