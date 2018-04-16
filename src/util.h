// Util functions

void correctness_check(int *output, int *solution, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (output[i*n + j] != solution[i*n + j]) {
        std::cerr << "Output did not match at [" << i << "][" << j << "]: " << output[i*n+j]
                << " vs solution's " << solution[i*n+j] << "!\n";
      }
    }
  }
}
