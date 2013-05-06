#include <pmmintrin.h>
#include <stdio.h>

// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!

 #define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

void matmul(int N, const double* A, const double* Bp, double* C) {
  // use straightforward multiply for 2x2 case
  if (N == 2) {
    C[0] += A[0]*Bp[0] + A[1]*Bp[2];
    C[1] += A[0]*Bp[1] + A[1]*Bp[3];
    C[2] += A[2]*Bp[0] + A[3]*Bp[2];
    C[3] += A[2]*Bp[1] + A[3]*Bp[3];
    return;
  }

  // allows us to access packed sse2 DFP values individually 
  union pd_d {
    __m128d pd;
    double d[2];
  } r1, r2;

  // cast to sse2 FP 
  __m128d* mA = (__m128d*) A;
  __m128d* mC = (__m128d*) C;


  int i, j, k, ii, jj, kk;
//  int BS = (N > 256) ? 32 : 8;
  int BS = 32;

//  __builtin_prefetch(A);
//  __builtin_prefetch(Bp);

  int n = N / 2;

if (N > 32) {
  double *B = malloc(N*N*sizeof(double));
  __m128d* mB = (__m128d*) B;
  // transpose array to line up memory access
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      B[i*N+j] = Bp[j*N+i];


  for (ii = 0; ii < N; ii += BS)
  for (jj = 0; jj < N; jj += BS)
//  for (kk = 0; kk < N; kk += BS/2)
//  for (i = 0; i < N; ++i)
  for (i = ii; i < min(ii+BS,N); ++i)
    for (j = jj; j < min(jj+BS,N); ++j) {
      // sse2 register to accumulate 2 additions at a time,
      // for every 2*kth and 2*k+1th array element (= kth element
      // of __m128d* array)
      r1.pd = _mm_set_pd(0.0, 0.0);
      for (k = 0; k < n; ++k) {
//      for (k = kk; k < min(kk+BS/2,n); ++k) {
//        __builtin_prefetch(mA+i*n+k+7,0);
//        __builtin_prefetch(mB+j*n+k+7,0);
          // multiply elements from mA and mB and add to cumulative total
          r1.pd = _mm_add_pd( r1.pd, _mm_mul_pd(mA[i*n+k],mB[j*n+k]));
      }
      C[i*N + j] += r1.d[0] + r1.d[1];
    }

} else {
  __m128d* mB = (__m128d*) Bp;
  for (i = 0; i < N; ++i)
    for (j = 0; j < n; ++j) {
      // now B is not transposed, so we do two columns at a time
      // since we're already reading 2 row elements of A at a time
      r1.pd = _mm_set_pd(0.0, 0.0);
      r2.pd = _mm_set_pd(0.0, 0.0);
      for (k = 0; k < n; ++k) {
        // transpose the two columns into b1 and b2:
        // x = mB[2*k*n+j]
        // y = mB[(2*k+1)*n+j]
        // b1 = x0, y0
        // b2 = x1, y1 
        __m128d b1 = _mm_shuffle_pd(mB[2*k*n+j], mB[(2*k+1)*n+j], _MM_SHUFFLE2(0,0));
        __m128d b2 = _mm_shuffle_pd(mB[2*k*n+j], mB[(2*k+1)*n+j], _MM_SHUFFLE2(1,1));

        r1.pd = _mm_add_pd( r1.pd, _mm_mul_pd(mA[i*n+k],b1));
        r2.pd = _mm_add_pd( r2.pd, _mm_mul_pd(mA[i*n+k],b2));
      }
      C[i*N + 2*j] += r1.d[0] + r1.d[1];
      C[i*N + 2*j + 1] += r2.d[0] + r2.d[1];
    }
}
}

