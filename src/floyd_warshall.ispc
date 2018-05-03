export void floyd_warshall_in_place(uniform int C[], const uniform int A[], const uniform int B[], 
				    const uniform int b, const uniform int n) {
  for (int k = 0; k < b; k++) {
    foreach (i = 0 ... b, j = 0 ... b) {
      if (C[i*n + j] > A[i*n + k] + B[k*n + j]) {
	C[i*n + j] = A[i*n + k] + B[k*n + j];
      }
    }
  }
}
