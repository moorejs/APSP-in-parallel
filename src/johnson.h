#include <random>		 // mt19937_64, uniform_x_distribution

inline int*[] johnson_init(const int n, const int p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  // TODO: create negative edges without negative cycles
  static std::uniform_int_distribution<int> choose_weight(0, 100);

  std::mt19937_64 rand_engine(seed);

  int* out[] = new int*[n];
  return out;
}