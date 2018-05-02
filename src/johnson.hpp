#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>

using namespace boost;

typedef adjacency_list<listS, vecS, directedS,
                        no_property, property<edge_weight_t, int> > Graph;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef std::pair<int,int> Edge;

typedef struct graph {
  int V;
  int E;
  Edge *edge_array;
  int* weights;
} graph_t;

graph_t* johnson_init(const int n, const double p, const unsigned long seed);

typedef struct edge {
  int u;
  int v;
} edge_t;

typedef struct graph_cuda {
  int V;
  int E;
  int* starts;
  int* weights;
  edge_t* edge_array;
} graph_cuda_t;


graph_cuda_t* johnson_cuda_init(const int n, const double p, const unsigned long seed);
void johnson_cuda(graph_cuda_t* gr, int* output);
void free_cuda_graph(graph_cuda_t* g);

void free_graph(graph_t* g);
void johnson_parallel(graph_t *gr, int* output);

