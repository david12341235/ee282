#include <emmintrin.h>

// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!

 #define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

void matmul(int N, const double* A, const double* B, double* C) {
  int i, j, k, ii, jj, kk, BS = 4;
  __builtin_prefetch(A);
  __builtin_prefetch(B);
  __builtin_prefetch(C,1);

  __m128d* mA = (__m128d*) A;
  __m128d* mB = (__m128d*) B;
  __m128d* mC = (__m128d*) C;

  N /= 2;

  for (ii = 0; ii < N; ii += BS)
  for (jj = 0; jj < N; jj += BS)
  for (kk = 0; kk < N; kk += BS)
  for (i = ii; i < min(ii+BS,N); i++)
    for (j = jj; j < min(jj+BS,N); j++) {
      __builtin_prefetch(mC+i*N+j+1,1);
      for (k = kk; k < min(kk+BS,N); k++) {
        __builtin_prefetch(mA+i*N+k+1,0);
        __builtin_prefetch(mB+(k+1)*N+j,0);
        mC[i*N + j] = _mm_add_pd( mC[i*N + j], _mm_mul_pd(mA[i*N+k],mB[k*N+j]));
      }
    }
}
