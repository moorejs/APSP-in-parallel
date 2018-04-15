#include <cstring> // memcpy
#include <random>		 // mt19937_64, uniform_x_distribution

// we need this to initialized to 0 on the diagonal, infinity anywhere there is no edge 
inline int* floyd_warshall_init(const int n, const int p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  // TODO: create negative edges without negative cycles
  static std::uniform_int_distribution<int> choose_weight(0, 100);

  std::mt19937_64 rand_engine(seed);

  int* out = new int[n * n];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        out[i*n + j] = 0;
      } else if (flip(rand_engine) > p) {
        out[i*n + j] = choose_weight(rand_engine);
      } else {
        out[i*n + j] = 9999; // infinity
      }
    }
  }

  return out;
}

// expects len(input) == len(output) == n*n
inline void floyd_warshall(const int* input, int* output, const int n) {
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

inline void floyd_warshall_in_place(int* C, const int* A, const int* B, const int n) {
  for (int k = 0; k < n; k++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (C[i*n + j] > A[i*n + k] + B[k*n + j]) {
          C[i*n + j] = A[i*n + k] + B[k*n + j];
        }
      }
    }
  }
}

// expects len(input) == len(output) == n*n
inline void floyd_warshall_blocked(const int* input, int* output, const int n, const int b) {
  std::memcpy(output, input, n * n * sizeof(int));

  // for now, assume b divides n
  const int blocks = n / b;

  // note that [i][j] == [i * input_width * block_width + j * block_width]

  for (int k = 0; k < blocks; k++) {
    floyd_warshall_in_place(&output[k*b*n + k*b], &output[k*b*n + k*b], &output[k*b*n + k*b], b);
    for (int j = 0; j < blocks && j != k; j++) {
      floyd_warshall_in_place(&output[k*b*n + j*b], &output[k*b*n + k*b], &output[k*b*n + j*b], b);
    }
    for (int i = 0; i < blocks && i != k; i++) {
      floyd_warshall_in_place(&output[i*b*n + k*b], &output[i*b*n + k*b], &output[k*b*n + k*b], b);
      for (int j = 0; j < blocks && j != k; j++) {
        floyd_warshall_in_place(&output[i*b*n + j*b], &output[i*b*n + k*b], &output[k*b*n + j*b], b);
      }
    }
  }
}