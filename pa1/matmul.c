// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!

void matmul(int N, const double* A, const double* B, double* C) {
  int i, j, k;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      for (k = 0; k < N; k++)
        C[i*N + j] += A[i*N+k]*B[k*N+j];
}
