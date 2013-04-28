#define _GNU_SOURCE
#include <sys/time.h>

#include <stdlib.h>
#include <time.h>

#include "utils.h"

// initialize RNG using time()
void rseed() {
  int i;
  unsigned short seed16v[3];

  for (i = 0; i < 3; i++) seed16v[i] = time(0);

  seed48(seed16v);
}

// return a random number uniformly in [l,u] inclusive, l < u
int rrand(int l, int u) { 
  return (l + (int)((1 + u - l)*drand48())); 
}

void mat_init(double *mat,int rows,int cols) {
  int i;

  for (i = 0; i < rows*cols; i++)
    *mat++ = 2.0*drand48() - 1.0;
}

double l1_norm(double *mat, int rows, int cols) {
  double sum=0;
  int i;

  for (i = 0; i < rows*cols; i++) {
    double val = *mat++;
    sum += ABS(val);
  }

  return sum;
}

double l1_norm_diff(double *mat1, double *mat2, int rows, int cols) {
  double sum = 0;
  int i;

  for (i = 0; i < rows*cols; i++) {
    double val = *mat1++ - *mat2++;
    sum += ABS(val);
  }

  return sum;
}

// error: Error formula to compare two matrices.
// norm(C1-C2)/(macheps*norm(A)*norm(B)),
// Ci=float(A*B)
// macheps=2^(-24) in single prec.
//        =2^(-53) in double prec.

double error(double *mat1, double *mat2, int rows, int cols)
{
  const double macheps = 1.110223024625157e-16; /* = 2^(-53) */

  return l1_norm_diff(mat1, mat2, rows, cols) /
    (macheps * l1_norm(mat1, rows, cols) * l1_norm(mat2, rows, cols));
}

double timeval_diff(struct timeval tv1, struct timeval tv2) {
  struct timeval utime;

  utime.tv_sec = tv2.tv_sec - tv1.tv_sec;

  if (tv2.tv_usec < tv1.tv_usec) {
    utime.tv_sec--;
    utime.tv_usec = 1000000 - tv1.tv_usec + tv2.tv_usec;
  } else {
    utime.tv_usec = tv2.tv_usec - tv1.tv_usec;
  }

  return (double)utime.tv_sec + (double)utime.tv_usec * 1e-6;
}
