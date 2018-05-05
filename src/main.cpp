#include <sys/stat.h> // stat
#include <unistd.h> // getopt
#include <chrono> // high_resolution_clock
#include <iostream> // cout
#include <cstdio> // printf
#include <fstream> // ifstream, ofstream
#include <sstream> // stringstream
#include <ratio>  // milli

#include <boost/graph/johnson_all_pairs_shortest.hpp> // seq algorithm, distance_map

#ifdef _OPENMP
#include "omp.h" // omp_set_num_threads
#endif

#include "util.hpp"
#include "johnson.hpp"
#include "floyd_warshall.hpp"

void bench_floyd_warshall(int interations, unsigned long seed, int block_size, bool check_correctness);
void bench_johnson(int iterations, unsigned long seed, bool check_correctness);

int main(int argc, char* argv[]) {
  // parameter defaults
  unsigned long seed = 0;
  int n = 1024;
  double p = 0.5;
  bool use_floyd_warshall = true;
  bool benchmark = false;
  bool check_correctness = false;
  int block_size = 32;
  int thread_count = 1;

  extern char* optarg;
  int opt;
  while ((opt = getopt(argc, argv, "ha:n:p:s:bd:ct:")) != -1) {
    switch (opt) {
      case 'h':
      case '?': // illegal command
      case ':': // forgot command's argument
        print_usage();
        return 0;

      case 'a':
        if (optarg[0] == 'j') {
          use_floyd_warshall = false;
        } else if (optarg[0] != 'f') {
          std::cerr << "Illegal algorithm argument, must be f or j\n";
          return -1;
        }
        break;

      case 'p':
        p = std::stod(optarg);
        break;

      case 's':
        seed = std::stoul(optarg);
        break;

      case 'b':
        benchmark = true;
        break;

      case 'n':
        n = std::stoi(optarg);
        break;

      case 'd':
        block_size = std::stoi(optarg);
        break;

      case 'c':
        check_correctness = true;
        break;

      case 't':
	thread_count = std::stoi(optarg);
        break;

    }
  }

#ifdef _OPENMP
  omp_set_num_threads(thread_count);
#else
  (void)thread_count; // suppress unused warning
#endif

  int* solution = nullptr; // both algorithms share the same solution
  if (!benchmark && check_correctness) {
    bool write_solution_to_file = true;

    // have we cached the solution before?
    std::string solution_filename = get_solution_filename("apsp", n, p, seed);
    struct stat file_stat;
    bool solution_available = stat(solution_filename.c_str(), &file_stat) != -1 || errno != ENOENT;

    solution = new int[n * n];
    if (solution_available) {
      std::cout << "Reading reference solution from file: " << solution_filename << "\n";

      std::ifstream in(solution_filename, std::ios::in | std::ios::binary);
      in.read(reinterpret_cast<char*>(solution), n * n * sizeof(int));
      in.close();
    } else {
      int* matrix = floyd_warshall_init(n, p, seed);

      auto start = std::chrono::high_resolution_clock::now();

      floyd_warshall(matrix, solution, n);

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> start_to_end = end - start;
      std::cout << "Algorithm runtime: " << start_to_end.count() << "ms\n";

      if (write_solution_to_file) {
        std::cout << "Writing solution to file: " << solution_filename << "\n";

        if (system("mkdir -p solution_cache") == -1) {
          std::cerr << "mkdir failed!";
          return -1;
        }

        std::ofstream out(solution_filename, std::ios::out | std::ios::binary);
        out.write(reinterpret_cast<const char*>(solution), n * n * sizeof(int));
        out.close();
      }
    }
  }

  if (use_floyd_warshall) {
    int* matrix = nullptr;
    int* output = nullptr;
    if (benchmark) {
      bench_floyd_warshall(1, seed, block_size, check_correctness);
    } else {
      matrix = floyd_warshall_blocked_init(n, block_size, p, seed);
      int n_blocked = n;
      int block_remainder = n % block_size;
      if (block_remainder != 0) {
	n_blocked = n + block_size - block_remainder;
      }
      output = new int[n_blocked * n_blocked];

      std::cout << "Using Floyd-Warshall's on " << n_blocked << "x" << n_blocked
		<< " with p=" << p << " and seed=" << seed << "\n";
      auto start = std::chrono::high_resolution_clock::now();
#ifdef CUDA
      floyd_warshall_blocked_cuda(matrix, output, n_blocked);
#else
      floyd_warshall_blocked(matrix, output, n_blocked, block_size);
#endif
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> start_to_end = end - start;
      std::cout << "Algorithm runtime: " << start_to_end.count() << "ms\n\n";

      if (check_correctness) {
        correctness_check(output, n_blocked, solution, n);
      }

      delete[] matrix;
      delete[] output;
    }
  } else {  // Using Johnson's Algorithm
    if (benchmark) {
      bench_johnson(1, seed, check_correctness);
    } else {
      int *output = new int[n * n];
      std::cout << "Using Johnson's on " << n << "x" << n
                << " with p=" << p << " and seed=" << seed << "\n";
#ifdef CUDA
      std::cout << "CUDA!\n";
      graph_cuda_t* cuda_gr = johnson_cuda_init(n, p, seed);
      auto start = std::chrono::high_resolution_clock::now();
      johnson_cuda(cuda_gr, output);
      auto end = std::chrono::high_resolution_clock::now();
      free_cuda_graph(cuda_gr);
#else
      graph_t *gr = johnson_init(n, p, seed);
      auto start = std::chrono::high_resolution_clock::now();
      johnson_parallel(gr, output);
      auto end = std::chrono::high_resolution_clock::now();
#endif
      std::chrono::duration<double, std::milli> start_to_end = end - start;
      std::cout << "Algorithm runtime: " << start_to_end.count() << "ms\n\n";

      if (check_correctness) correctness_check(output, n, solution, n);

      //free_graph(gr);
      delete[] output;
    }
  }

  if (check_correctness) {
    delete[] solution;
  }

  return 0;
}

void bench_floyd_warshall(int iterations, unsigned long seed, int block_size, bool check_correctness) {
  std::cout << "\n\nFloyd-Warshall's Algorithm benchmarking results for seed=" << seed << " and block size=" << block_size << "\n";

  print_table_header(check_correctness);
  for (double p = 0.25; p < 1.0; p += 0.25) {
    for (int v = 64; v <= 1024; v *= 2) {
      int* matrix = floyd_warshall_init(v, p, seed);
      int* solution = new int[v * v];

      int* matrix_blocked = matrix; // try to reuse inputs
      int v_blocked = v;
      int block_remainder = v % block_size;
      if (block_remainder != 0) {
	// we may have to add some verts to fit to a multiple of block_size
        matrix_blocked = floyd_warshall_blocked_init(v, block_size, p, seed);
	    v_blocked = v + block_size - block_remainder;
      }
      int* output = new int[v_blocked * v_blocked];

      bool correct = false;

      double seq_total_time = 0.0;
      double total_time = 0.0;
      for (int b = 0; b < iterations; b++) {
        // clear solution
        std::memset(solution, 0, v*v*sizeof(int));

        auto seq_start = std::chrono::high_resolution_clock::now();
        floyd_warshall(matrix, solution, v);
        auto seq_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> seq_start_to_end = seq_end - seq_start;
        seq_total_time += seq_start_to_end.count();

        // clear output
        std::memset(output, 0, v_blocked*v_blocked*sizeof(int));
        auto start = std::chrono::high_resolution_clock::now();
        floyd_warshall_blocked(matrix_blocked, output, v_blocked, block_size);
        auto end = std::chrono::high_resolution_clock::now();

        if (check_correctness) {
          correct = correct || correctness_check(output, v_blocked, solution, v);
        }

        std::chrono::duration<double, std::milli> start_to_end = end - start;
        total_time += start_to_end.count();
      }
      delete[] matrix;

      print_table_row(p, v, seq_total_time, total_time, check_correctness, correct);
    }
    print_table_break(check_correctness);
  }
  std::cout << "\n\n";
}

void bench_johnson(int iterations, unsigned long seed, bool check_correctness) {
  std::cout << "\n\nJohnson's Algorithm benchmarking results for seed=" << seed << "\n";

  print_table_header(check_correctness);
  for (double p = 0.25; p < 1.0; p += 0.25) {
    for (int v = 64; v <= 2048; v *= 2) {
      // johnson init
      graph_t* gr = johnson_init(v, p, seed);
      int* matrix = floyd_warshall_init(v, p, seed);
      int* output = new int[v* v];

      int* solution = new int[v*v];
      int** out_sol = new int*[v];
      for (int i = 0; i < v; i++) out_sol[i] = &solution[i*v];
      Graph G(gr->edge_array, gr->edge_array + gr->E, gr->weights, gr->V);
      std::vector<int> d(num_vertices(G));

      bool correct = false;

      double seq_total_time = 0.0;
      double total_time = 0.0;
      for (int b = 0; b < iterations; b++) {
        // clear solution
        std::memset(solution, 0, v*v*sizeof(int));

        auto seq_start = std::chrono::high_resolution_clock::now();
        johnson_all_pairs_shortest_paths(G, out_sol, distance_map(&d[0]));
        auto seq_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> seq_start_to_end = seq_end - seq_start;
        seq_total_time += seq_start_to_end.count();

        // clear output
        std::memset(output, 0, v*v*sizeof(int));
        auto start = std::chrono::high_resolution_clock::now();
        // TODO: johnson parallel -- temporarily putting floyd_warshall here
        //floyd_warshall_blocked(matrix, output, v, block_size);
        johnson_parallel(gr, output);
        auto end = std::chrono::high_resolution_clock::now();

        if (check_correctness) {
          correct = correct || correctness_check(output, v, solution, v);
        }

        std::chrono::duration<double, std::milli> start_to_end = end - start;
        total_time += start_to_end.count();
      }
      delete[] solution;
      delete[] out_sol;
      delete[] output;
      delete[] matrix;

      print_table_row(p, v, seq_total_time, total_time, check_correctness, correct);
    }
    print_table_break(check_correctness);
  }
  std::cout << "\n\n";
}
