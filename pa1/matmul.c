#include <emmintrin.h>
#include <stdio.h>

// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!

 #define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

void matmul(int N, const double* A, const double* Bp, double* C) {
  int i, j, k, ii, jj, kk;
//  int BS = (N > 256) ? 32 : 8;
  int BS = 32;

  double* B = malloc(N*N*sizeof(double));
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      B[i*N+j] = Bp[j*N+i];

  __builtin_prefetch(A);
  __builtin_prefetch(B);

  __m128d* mA = (__m128d*) A;
  __m128d* mB = (__m128d*) B;
  __m128d* mC = (__m128d*) C;

if (N > 32) {
  int n = N / 2;

  for (ii = 0; ii < N; ii += BS)
  for (jj = 0; jj < N; jj += BS)
  for (kk = 0; kk < N; kk += BS/2)
  for (i = ii; i < min(ii+BS,N); i = i + 1)
    for (j = jj; j < min(jj+BS,N); j = j + 1) {
      __m128d t1 = _mm_set_pd(0.0, 0.0);
      __m128d t2 = _mm_set_pd(0.0, 0.0);
//      __builtin_prefetch(mC+i*N + j + 1,1);
//      _mm_prefetch(mC+i*N + j + 1, _MM_HINT_NTA);
      for (k = kk; k < min(kk+BS/2,n); k = k + 1) {
//        __builtin_prefetch(mA+i*n+k+7,0,3);
//        __builtin_prefetch(mB+(2*k+2)*n+j,0,3);
//        __builtin_prefetch(mB+(2*k+3)*n+j,0,3);
//        __builtin_prefetch(mB+N*j+k+7,0,3);

//        __m128d b1 = _mm_shuffle_pd(mB[2*k*n+j], mB[(2*k+1)*n+j], 0x4);
//        __m128d b2 = _mm_shuffle_pd(mB[2*k*n+j], mB[(2*k+1)*n+j], 0x3);
//        __m128d b1 = _mm_shuffle_pd(mB[N*j+k], mB[N*j+k+1], 0x4);
//        __m128d b2 = _mm_shuffle_pd(mB[N*j+k], mB[N*j+k+1], 0x3);

        t1 = _mm_add_pd( t1, _mm_mul_pd(mA[i*n+k],mB[j*n+k]));
//        t2 = _mm_add_pd( t2, _mm_mul_pd(mA[i*n+k+1],mB[j*n+k+1]));
//        t1 = _mm_add_pd( t1, _mm_mul_pd(mA[i*n+k],b1));
//        t2 = _mm_add_pd( t2, _mm_mul_pd(mA[i*n+k],b2));
      }
//      __m128d x,y,c;
//      x = _mm_shuffle_pd(t1, t2, _MM_SHUFFLE2(1,1));
//      y = _mm_shuffle_pd(t1, t2, _MM_SHUFFLE2(0,0));
//      _mm_stream_pd(C + i*N + 2*j, _mm_add_pd(x,y));
//      x = _mm_add_pd(x,y);
//      mC[i*n + j] = _mm_add_pd(mC[i*N + j], x);
      C[i*N + j] += *((double*)&t1) + *(((double*)&t1)+1);
//      C[i*N + 2*j + 1] += *((double*)&t2) + *(((double*)&t2)+1);
    }

} else {
  int n = N / 2;

  for (ii = 0; ii < N; ii += BS)
  for (jj = 0; jj < N; jj += BS)
  for (kk = 0; kk < N; kk += BS/2)
  for (i = ii; i < min(ii+BS,N); i = i + 1)
    for (j = jj; j < min(jj+BS,N); j = j + 1) {
      __m128d t1 = _mm_set_pd(0.0, 0.0);
      __m128d t2 = _mm_set_pd(0.0, 0.0);
//      __builtin_prefetch(mC+i*N + j + 1,1);
//      _mm_prefetch(mC+i*N + j + 1, _MM_HINT_NTA);
      for (k = kk; k < min(kk+BS/2,n); k = k + 1) {
//        __builtin_prefetch(mA+i*n+k+7,0,3);
//        __builtin_prefetch(mB+(2*k+2)*n+j,0,3);
//        __builtin_prefetch(mB+(2*k+3)*n+j,0,3);
//        __builtin_prefetch(mB+N*j+k+7,0,3);

//        __m128d b1 = _mm_shuffle_pd(mB[2*k*n+j], mB[(2*k+1)*n+j], 0x4);
//        __m128d b2 = _mm_shuffle_pd(mB[2*k*n+j], mB[(2*k+1)*n+j], 0x3);
//        __m128d b1 = _mm_shuffle_pd(mB[N*j+k], mB[N*j+k+1], 0x4);
//        __m128d b2 = _mm_shuffle_pd(mB[N*j+k], mB[N*j+k+1], 0x3);

        t1 = _mm_add_pd( t1, _mm_mul_pd(mA[i*n+k],mB[j*n+k]));
//        t2 = _mm_add_pd( t2, _mm_mul_pd(mA[i*n+k+1],mB[j*n+k+1]));
//        t1 = _mm_add_pd( t1, _mm_mul_pd(mA[i*n+k],b1));
//        t2 = _mm_add_pd( t2, _mm_mul_pd(mA[i*n+k],b2));
      }
//      __m128d x,y,c;
//      x = _mm_shuffle_pd(t1, t2, _MM_SHUFFLE2(1,1));
//      y = _mm_shuffle_pd(t1, t2, _MM_SHUFFLE2(0,0));
//      _mm_stream_pd(C + i*N + 2*j, _mm_add_pd(x,y));
//      x = _mm_add_pd(x,y);
//      mC[i*n + j] = _mm_add_pd(mC[i*N + j], x);
      C[i*N + j] += *((double*)&t1) + *(((double*)&t1)+1);
//      C[i*N + 2*j + 1] += *((double*)&t2) + *(((double*)&t2)+1);
    }
}

