#include <unistd.h>	// getopt
#include <chrono>		 // high_resolution_clock
#include <iostream>	// cout
#include <ratio>		 // milli

#include "floyd_warshall.h"

void print_usage();

int main(int argc, char* argv[]) {
  unsigned long seed;

  // parameter defaults
  int n = 1000;
  double p = 0.5;
  bool use_floyd_warshall = true;

  extern char* optarg;
  int opt;
  while ((opt = getopt(argc, argv, "han::p:s:")) != -1) {
    switch (opt) {
      case 'h':
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
    }
  }

  if (use_floyd_warshall) {
    std::cout << "Generating " << n << "x" << n << " adjacency matrix with seed " << seed << "\n";

    // have we cached the solution before?
    bool solution_available = false;

    int* solution;
    if (solution_available) {
      std::cout << "Reading reference solution from file\n";
    } else {
      std::cout << "Solving APSP with Floyd-Warshall sequentially for reference solution\n";

      solution = new int[n * n];
      int* matrix = floyd_warshall_init(n, p, seed);

      auto start = std::chrono::high_resolution_clock::now();

      floyd_warshall(matrix, solution, n);

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> start_to_end = end - start;
      std::cout << "Algorithm runtime: " << start_to_end.count() << "ms\n\n";

      if (false) {
        std::cout << "Writing solution to file\n";
      }
    }

    int* output = new int[n * n];
    std::cout << "Solving APSP with Floyd-Warshall blocked sequentially\n";

    int* matrix = floyd_warshall_init(n, p, seed);

    auto start = std::chrono::high_resolution_clock::now();

    floyd_warshall_blocked(matrix, output, n, 32);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> start_to_end = end - start;
    std::cout << "Algorithm runtime: " << start_to_end.count() << "ms\n\n";


    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (output[i*n + j] != solution[i*n + j]) {
          std::cerr << "Output did not match at [" << i << "][" << j << "]: " << output[i*n+j] 
              << " vs solution's " << solution[i*n+j] << "!\n";
        }
      }
    }

    delete[] matrix;
    if (!solution_available) {
      delete[] solution;
    }
    delete[] output;
  } else {
  }

  return 0;
}

void print_usage() {
  std::cout << "\nUsage: asap [-a (f|j)] [-s seed]\n";
  std::cout << "\t-h\tPrint this message\n";
  std::cout << "\t-a\tAlgorithm to use for all pairs shortest path\n";
  std::cout << "\t\t\tf: Floyd-Warshall (default)\n";
  std::cout << "\t\t\tj: Johnson's Algorithm\n";
  std::cout << "\t-s\tSeed for graph generation\n";
  std::cout << "\n";
}