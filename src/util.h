#include <iostream> // cout
#include <string> // string
#include <sstream>	// stringstream

// Util functions

bool correctness_check(int* output, int n_output, int* solution, int n_solution) {
  for (int i = 0; i < n_solution; i++) {
    for (int j = 0; j < n_solution; j++) {
      if (output[i*n_output + j] != solution[i*n_solution + j]) {
        std::cerr << "Output did not match at [" << i << "][" << j << "]: " << output[i*n_output+j]
                << " vs solution's " << solution[i*n_solution+j] << "!\n";
        return false;
      }
    }
  }

  return true;
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

void print_table_row(double p, int v, double seq, double par, bool check_correctness, bool correct) {
  std::printf("\n| %-3.2f | %-7d | %-12.3f | %-12.3f | %-10.3f |", p, v, seq, par, seq /par);
  if (check_correctness) {
    std::printf(" %-8s |", (correct ? "x" : "!"));
  }
}

void print_table_break(bool check_correctness) {
  if (check_correctness) {
    std::printf("\n ----------------------------------------------------------------------");
  } else {
    std::printf("\n -----------------------------------------------------------");
  }
}

void print_table_header(bool check_correctness) {
  print_table_break(check_correctness);
  std::printf("\n| %-4s | %-7s | %-12s | %-12s | %-10s |",
      "p", "verts", "seq (ms)", "par (ms)", "speedup");
  if (check_correctness) {
    std::printf(" %-8s |", "correct");
  }
  print_table_break(check_correctness);
}

std::string get_solution_filename(std::string prefix, int n, double p, unsigned long seed) {
  std::stringstream solution_filename;
  solution_filename << "solution_cache/" << prefix << "-sol-n"<< n << "-p" << p << "-s" << seed << ".bin";
  return solution_filename.str();
}
