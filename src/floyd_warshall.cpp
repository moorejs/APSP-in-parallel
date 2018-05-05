#include <cstring> // memcpy
#include <random> // mt19937_64, uniform_x_distribution

#ifdef _OPENMP
#include "omp.h" // omp_get_num_threads
#endif

#include "floyd_warshall.hpp"

int* floyd_warshall_init(const int n, const double p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  // TODO: create negative edges without negative cycles
  static std::uniform_int_distribution<int> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  int* out = new int[n * n];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        out[i*n + j] = 0;
      } else if (flip(rand_engine) < p) {
        out[i*n + j] = choose_weight(rand_engine);
      } else {
        // "infinity" - the highest value we can still safely add two infinities
        out[i*n + j] = std::numeric_limits<int>::max() / 2;
      }
    }
  }

  return out;
}

int* floyd_warshall_blocked_init(const int n, const int block_size, const double p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  // TODO: create negative edges without negative cycles
  static std::uniform_int_distribution<int> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  int n_oversized;
  int block_remainder = n % block_size;
  if (block_remainder == 0) {
    n_oversized = n;
  } else {
    n_oversized = n + block_size - block_remainder;
  }

  int* out = new int[n_oversized * n_oversized];
  for (int i = 0; i < n_oversized; i++) {
    for (int j = 0; j < n_oversized; j++) {
      if (i == j) {
        out[i*n_oversized + j] = 0;
      } else if (i < n && j < n && flip(rand_engine) < p) {
        out[i*n_oversized + j] = choose_weight(rand_engine);
      } else {
        // "infinity" - the highest value we can still safely add two infinities
        out[i*n_oversized + j] = std::numeric_limits<int>::max() / 2;
      }
    }
  }

  return out;
}

void floyd_warshall(const int* input, int* output, const int n) {
  std::memcpy(output, input, n * n * sizeof(int));

  for (int k = 0; k < n; k++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (output[i*n + j] > output[i*n + k] + output[k*n + j]) {
          output[i*n + j] = output[i*n + k] + output[k*n + j];
        }
      }
    }
  }
}

void floyd_warshall_blocked(const int* input, int* output, const int n, const int b) {
  std::memcpy(output, input, n * n * sizeof(int));

  // for now, assume b divides n
  const int blocks = n / b;

  // note that [i][j] == [i * input_width * block_width + j * block_width]
  for (int k = 0; k < blocks; k++) {
    floyd_warshall_in_place(&output[k*b*n + k*b], &output[k*b*n + k*b], &output[k*b*n + k*b], b, n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int j = 0; j < blocks; j++) {
      if (j == k) continue;
      floyd_warshall_in_place(&output[k*b*n + j*b], &output[k*b*n + k*b], &output[k*b*n + j*b], b, n);
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < blocks; i++) {
      if (i == k) continue;
      floyd_warshall_in_place(&output[i*b*n + k*b], &output[i*b*n + k*b], &output[k*b*n + k*b], b, n);
      for (int j = 0; j < blocks; j++) {
	    if (j == k) continue;
	    floyd_warshall_in_place(&output[i*b*n + j*b], &output[i*b*n + k*b], &output[k*b*n + j*b], b, n);
      }
    }
  }
}
