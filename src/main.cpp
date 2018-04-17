#include <sys/stat.h> // stat
#include <unistd.h> // getopt
#include <chrono> // high_resolution_clock
#include <iostream> // cout
#include <cstdio> // printf
#include <fstream> // ifstream, ofstream
#include <sstream> // stringstream
#include <ratio>  // milli

#include "util.h"
#include "johnson.h"
#include "floyd_warshall.h"

void bench_floyd_warshall(double p, unsigned long seed, int block_size, bool check_correctness);

int main(int argc, char* argv[]) {
  // parameter defaults
  unsigned long seed = 0;
  int n = 1024;
  double p = 0.5;
  bool use_floyd_warshall = true;
  bool benchmark = false;
  bool check_correctness = false;
  int block_size = 32;

  extern char* optarg;
  int opt;
  while ((opt = getopt(argc, argv, "ha:n:p:s:bd:c")) != -1) {
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
    }
  }

    std::cout << "\nGenerating " << n << "x" << n << " adjacency matrix with seed " << seed << "\n";

    int* solution = nullptr;
    if (check_correctness) {
      bool write_solution_to_file = true;

      // have we cached the solution before?
      std::string solution_filename = get_solution_filename("fw", n, p, seed);
      struct stat file_stat;
      bool solution_available = stat(solution_filename.c_str(), &file_stat) != -1 || errno != ENOENT;

      solution = new int[n * n];
      if (solution_available) {
        std::cout << "Reading reference solution from file\n";

        std::ifstream in(solution_filename, std::ios::in | std::ios::binary);
        in.read(reinterpret_cast<char*>(solution), n * n * sizeof(int));
        in.close();
      } else {
        std::cout << "Solving APSP with Floyd-Warshall sequentially for reference solution\n";

        int* matrix = floyd_warshall_init(n, p, seed);

        auto start = std::chrono::high_resolution_clock::now();

        floyd_warshall(matrix, solution, n);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> start_to_end = end - start;
        std::cout << "Algorithm runtime: " << start_to_end.count() << "ms\n";

        if (write_solution_to_file) {
          std::cout << "Writing solution to file\n";

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
    std::cout << "\nSolving APSP with Floyd-Warshall blocked sequentially\n";

    int* matrix = nullptr;
    int* output = nullptr;
    if (benchmark) {
      bench_floyd_warshall(p, seed, block_size, check_correctness);
    } else {
      matrix = floyd_warshall_init(n, p, seed);
      output = new int[n * n];

      auto start = std::chrono::high_resolution_clock::now();
      floyd_warshall_blocked(matrix, output, n, block_size);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> start_to_end = end - start;
      std::cout << "Algorithm runtime: " << start_to_end.count() << "ms\n\n";

      if (check_correctness) correctness_check(output, solution, n);

      delete[] matrix;
      delete[] output;
    }

  } else {  // Using Johnson's Algorithm
    std::cout << "Solving APSP with Johnson's sequentially" << "\n";
    //int *matrix_john = johnson_init(n, p, seed);
    //Graph_t *graph_john = johnson_init2(n, p, seed);
    graph_t *gr = johnson_init3(n, p, seed);
    int **out = new int*[n];
    for (int i = 0; i < n; i++) out[i] = new int[n];

    Graph G(gr->edge_array, gr->edge_array + gr->E, gr->weights, gr->V);
    std::vector<int> d(num_vertices(G));

    auto start = std::chrono::high_resolution_clock::now();
    johnson_all_pairs_shortest_paths(G, out, distance_map(&d[0]));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> start_to_end = end - start;
    std::cout << "Algorithm runtime: " << start_to_end.count() << "ms\n\n";

    int *output = new int[n * n];
    for (int i = 0; i < n; i++) {
      std::memcpy(&output[i*n], out[i], n * sizeof(int));
    }

    if (check_correctness) correctness_check(output, solution, n);

    for (int i = 0; i < n; i++) delete[] out[i];
    delete[] out;
    delete[] output;
  }

  if (check_correctness) {
    delete[] solution;
  }

  return 0;
}

void bench_floyd_warshall(double p, unsigned long seed, int block_size, bool check_correctness) {
  std::cout << "Benchmarking Results for p=" << p << ", block size=" << block_size << " and seed=" << seed << "\n";

  if (check_correctness) {
    std::printf("\n --------------------------------------------------------------- \n");
  } else {
    std::printf("\n ---------------------------------------------------- \n");
  }

  std::printf("| %-7s | %-12s | %-12s | %-10s |", "verts", "seq (ms)", "par (ms)", "speedup");
  if (check_correctness) {
    std::printf(" %-8s |", "correct");
    std::printf("\n --------------------------------------------------------------- \n");
  } else {
    std::printf("\n ---------------------------------------------------- \n");
  }

  for (int v = 64; v <= 4096; v *= 2) {
    int* matrix = floyd_warshall_init(v, p, seed);
    int* solution = new int[v * v];
    int* output = new int[v * v];

    bool correct = false;

    double seq_total_time = 0.0;
    double total_time = 0.0;
    for (int b = 0; b < 1; b++) {
      // clear solution
      std::memset(solution, 0, sizeof(int));

      auto seq_start = std::chrono::high_resolution_clock::now();
      floyd_warshall(matrix, solution, v);
      auto seq_end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double, std::milli> seq_start_to_end = seq_end - seq_start;
      seq_total_time += seq_start_to_end.count();

      // clear output
      std::memset(output, 0, sizeof(int));
      auto start = std::chrono::high_resolution_clock::now();
      floyd_warshall_blocked(matrix, output, v, block_size);
      auto end = std::chrono::high_resolution_clock::now();

      if (check_correctness) {
        correct = correct || correctness_check(output, solution, v);
      }

      std::chrono::duration<double, std::milli> start_to_end = end - start;
      total_time += start_to_end.count();
    }
    delete[] matrix;

    if (check_correctness) {
      std::printf("| %-7d | %-12.3f | %-12.3f | %-10.3f | %-8s |\n", v, seq_total_time / 2, total_time / 2, seq_total_time / total_time, (correct ? "x" : "!"));
    } else {
      std::printf("| %-7d | %-12.3f | %-12.3f | %-10.3f |\n", v, seq_total_time / 2, total_time / 2, seq_total_time / total_time);
    }
  }
  if (check_correctness) {
    std::printf("\n -------------------------------------------------------------- \n");
  } else {
    std::printf("\n ---------------------------------------------------- \n");
  }
}
