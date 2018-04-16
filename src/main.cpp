#include <sys/stat.h>
#include <unistd.h>	// getopt, S_ISDIR
#include <chrono>		 // high_resolution_clock
#include <iostream>	// cout
#include <fstream> // ifstream, ofstream
#include <sstream> // stringstream
#include <ratio>		 // milli

#include "floyd_warshall.h"

void print_usage();
std::string get_solution_filename(std::string prefix, int n, double p, unsigned long seed);

int main(int argc, char* argv[]) {
  // parameter defaults
  unsigned long seed = 0;
  int n = 1024;
  double p = 0.5;
  bool use_floyd_warshall = true;
  int bench_count = 1;
  bool check_correctness = false;
  int block_size = 32;

  extern char* optarg;
  int opt;
  while ((opt = getopt(argc, argv, "ha:n:p:s:b:d:c")) != -1) {
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

      case 'n':
        n = std::stoi(optarg);
        break;

      case 'p':
        p = std::stod(optarg);
        break;

      case 's':
        seed = std::stoul(optarg);
        break;

      case 'b':
        bench_count = std::stoi(optarg);
        break;

      case 'd':
        block_size = std::stoi(optarg);
        break;

      case 'c':
        check_correctness = true;
        break;
    }
  }

  if (use_floyd_warshall) {
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

    int* output = new int[n * n];
    std::cout << "\nSolving APSP with Floyd-Warshall blocked sequentially\n";

    int* matrix = floyd_warshall_init(n, p, seed);

    double total_time = 0.0;
    for (int b = 0; b < bench_count; b++) {
      auto start = std::chrono::high_resolution_clock::now();
      floyd_warshall_blocked(matrix, output, n, block_size);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> start_to_end = end - start;
      total_time += start_to_end.count();
    }
    std::cout << "Algorithm runtime: " << total_time / bench_count << "ms\n\n";

    if (check_correctness) {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          if (output[i*n + j] != solution[i*n + j]) {
            std::cerr << "Output did not match at [" << i << "][" << j << "]: " << output[i*n+j] 
                << " vs solution's " << solution[i*n+j] << "!\n";
          }
        }
      }
    }

    delete[] matrix;
    delete[] output;
    if (check_correctness) {
      delete[] solution;
    }
  } else {
  }

  return 0;
}

void print_usage() {
  std::cout << "\nUsage: asap [-n INT] [-p DOUBLE] [-a (f|j)] [-s LONG] [-b INT] [-c]\n";
  std::cout << "\t-h\t\tPrint this message\n";
  std::cout << "\t-n INT\t\tGraph size, default 1024\n";
  std::cout << "\t-p DOUBLE\t\tProbability of edge from a given node to another (0.0 to 1.0), default 0.5\n";
  std::cout << "\t-a CHAR\t\tAlgorithm to use for all pairs shortest path\n";
  std::cout << "\t\t\t\tf: Floyd-Warshall (default)\n";
  std::cout << "\t\t\t\tj: Johnson's Algorithm\n";
  std::cout << "\t-s LONG\t\tSeed for graph generation\n";
  std::cout << "\t-b INT\t\tNumber of times to run the benchmark, default 1\n";
  std::cout << "\t-c\t\tCheck correctness\n";
  std::cout << "\n";
}

std::string get_solution_filename(std::string prefix, int n, double p, unsigned long seed) {
  std::stringstream solution_filename;
  solution_filename << "solution_cache/" << prefix << "-sol-n"<< n << "-p" << p << "-s" << seed << ".bin";
  return solution_filename.str();
}