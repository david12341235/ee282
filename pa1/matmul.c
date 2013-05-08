#include <pmmintrin.h>
#include <stdio.h>

// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!

 #define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

void matmul(int N, const double* A, const double* Bp, double* C) {

  // cast to sse2 FP 
  __m128d* mA = (__m128d*) A;
  __m128d* mC = (__m128d*) C;

  // use straightforward multiply for 2x2 case
  if (N == 2) {
    __m128d b1 = _mm_load_pd(Bp);
    __m128d b2 = _mm_load_pd(Bp+2);
    mC[0] += _mm_add_pd(_mm_mul_pd(_mm_set1_pd(A[0]), b1), _mm_mul_pd(_mm_set1_pd(A[1]), b2));
    mC[1] += _mm_add_pd(_mm_mul_pd(_mm_set1_pd(A[2]), b1), _mm_mul_pd(_mm_set1_pd(A[3]), b2));
    return;
  }

  // allows us to access packed sse2 DFP values individually 
  union pd_d {
    __m128d pd;
    double d[2];
  } r1, r2, b1, b2, a;


  int i, j, k, ii, jj, kk, iii, jjj, kkk;
  int BSI = 256;
  int BSI2 = 2048;
  int BSJ = (N == 64) ? 64 : 256;
  int BSJ2 = 2048;
  int BSK = 16;
  int BSK2 = 2048;


  int n = N / 2;

//if (N < 3000) {
//  double *B = malloc(N*N*sizeof(double));
  __m128d* mB = (__m128d*) Bp;

//  for (iii = 0; iii < N; iii += BSI2)
//  for (jjj = 0; jjj < n; jjj += BSJ2/2)
//  for (kkk = 0; kkk < n; kkk += BSK2/2)
//  for (ii = iii; ii < min(iii+BSI2,N); ii += BSI)
//  for (jj = jjj; jj < min(jjj+BSJ2/2,n); jj += BSJ/2)
//  for (kk = kkk; kk < min(kkk+BSK2/2,n); kk += BSK/2)
  for (ii = 0; ii < N; ii += BSI)
  for (jj = 0; jj < n; jj += BSJ/2)
  for (kk = 0; kk < n; kk += BSK/2)
//  for (kk = 0; kk < n; kk += BS/2)
  for (i = ii; i < min(ii+BSI,N); ++i) {
    __m128d *__restrict__ c = mC + n*i;
    for (k = kk; k < min(kk+BSK/2,n); ++k) {
//    for (k = 0; k < n; ++k) {
      a.pd = mA[n*i + k];
      b1.pd = _mm_set1_pd(a.d[0]);
      b2.pd = _mm_set1_pd(a.d[1]);
      // sse2 register to accumulate 2 additions at a time,
      // for every 2*kth and 2*k+1th array element (= kth element
      // of __m128d* array)
      __m128d *__restrict__ mb1 = mB + n*2*k;
      __m128d *__restrict__ mb2 = mB + n*(2*k+1);
      for (j = jj; j < min(jj+BSJ/2,n); ++j) {
          c[j] = _mm_add_pd( _mm_add_pd(c[j], _mm_mul_pd(b1.pd,mb1[j])),  _mm_mul_pd(b2.pd,mb2[j]));
      }
    }
  }  

/*} else {
  double *B = malloc(N*N*sizeof(double));
  __m128d* mB = (__m128d*) B;
  // transpose array to line up memory access
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      B[i*N+j] = Bp[j*N+i];


  for (jjj = 0; jjj < N; jjj += BS2)
  for (jj = jjj; jj < min(jjj+BS2,N); jj += BS)
  for (i = 0; i < N; ++i)
    for (j = jj; j < min(jj+BS,N); ++j) {
      // sse2 register to accumulate 2 additions at a time,
      // for every 2*kth and 2*k+1th array element (= kth element
      // of __m128d* array)
      r1.pd = _mm_set_pd(0.0, 0.0);
      for (k = 0; k < n; ++k) {
          // multiply elements from mA and mB and add to cumulative total
          r1.pd = _mm_add_pd( r1.pd, _mm_mul_pd(mA[i*n+k],mB[j*n+k]));
      }
      C[i*N + j] += r1.d[0] + r1.d[1];
    }

}*/
}

