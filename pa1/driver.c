#include <getopt.h>
#include <string.h>
#include <strings.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <unistd.h>
#include <signal.h>

// #define ALLOC_ONCE

#ifdef PAPI
#include <papi.h>
#endif

#ifdef BLAS
#include <cblas.h>
#endif

#include "utils.h"

// matmul() can be found in matmul.c.
void matmul(int N, const double* A, const double* B, double* C);

// For simplicity, test sizes are restricted to be power of 2.
int test_sizes[] = { 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 };
#define NUM_TESTS (sizeof(test_sizes) / sizeof(int))

#ifdef PAPI
// You may replace these events with others of your choice. Refer to
// the PAPI documentation.
//
// Note: There are a limited number of hardware performance counters,
// so adding too many counters here may break PAPI.

int events[] = { 
  PAPI_TOT_INS,
  PAPI_TOT_CYC,
  PAPI_L1_DCM,
  PAPI_L2_TCM,
};

#define NUM_EVENTS (sizeof(events) / sizeof(int))

#endif

volatile int long_enough = 0;
volatile int expired = 0;

#define MAX_ERROR     (2.0)

void alarm_handler(int signum) {
  if (expired) {
    printf(" --- Measurement is taking too long... aborting.\n");
    exit(-1);
  } else if (!long_enough) {
    long_enough = 1;
    alarm(9);
  } else {
    expired = 1;
    alarm(20);
  }
}

// This function is for checking your matmul() for correctness.
void naive_matmul(int n, const double *A, const double *B, double *C) {
  int i, j, k;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++)
        C[i*n + j] += A[i*n+k]*B[k*n+j];
}

void check_correct () {
  double *A, *B, *C, *cA, *cB, *cC;

  int maxnbytes = sizeof(double) * SQR(test_sizes[NUM_TESTS - 1]);
  int i;

  A  = (double*) malloc(maxnbytes);
  B  = (double*) malloc(maxnbytes);
  C  = (double*) malloc(maxnbytes);
  cA = (double*) malloc(maxnbytes);
  cB = (double*) malloc(maxnbytes);
  cC = (double*) malloc(maxnbytes);
 
  if (A  == NULL || B  == NULL || C  == NULL || 
      cA == NULL || cB == NULL || cC == NULL) {
    printf("check_correct(): malloc() failed\n");
    exit(-1);
  }
 
  printf("Checking for correctness.\n\n"); 

  for (i = 0; i < NUM_TESTS; i++)  {
    int matdim = test_sizes[i];
    double err;
    int nbytes = sizeof(double) * SQR(matdim);

    printf("%4d x %4d   ", matdim, matdim); 

    mat_init(A, matdim, matdim);
    mat_init(B, matdim, matdim);
    mat_init(C, matdim, matdim);

    bcopy((void*)A, (void*)cA, nbytes);
    bcopy((void*)B, (void*)cB, nbytes);
    bcopy((void*)C, (void*)cC, nbytes);

    expired = 1; alarm(30);

    // Use an optimized BLAS library if available.
#ifndef BLAS
    naive_matmul(matdim, cA, cB, cC);
#else
    cblas_dgemm (CblasRowMajor,
                 CblasNoTrans,
                 CblasNoTrans,
                 matdim, matdim, matdim,
                 1.0, cA, matdim, cB, matdim, 1.0, cC, matdim);
#endif

    matmul(matdim, A, B, C);

    alarm(0);

    if (bcmp((void*)A, (void*)cA, nbytes) != 0 ||
        bcmp((void*)B, (void*)cB, nbytes) != 0) {
      printf("\nSource matrices have changed.\nFAILED.\n");
      exit(1);
    }

    if ((err = error(C, cC, matdim, matdim)) > MAX_ERROR) {
      printf("FAILED  Calculated error %f > %f\n", err, MAX_ERROR);
    } else {
      printf("PASSED\n");
    }
  }

  free(A);
  free(B);
  free(C); 
  free(cA);
  free(cB);
  free(cC);
}

void measure_performance () {
  struct rusage ru_start, ru_end;

  double *A,  *B,  *C;
  double *oA, *oB, *oC;
  int i, j, k;
  int test;
  int papi = 0;

  long long values[4];

  printf("\n");
  printf("Each measurement is average per iteration. "
         "Runtime is given in milliseconds.\n");
  printf("MFLOPS is estimated assuming a naive matmul().\n");

#ifdef PAPI
  printf("\n");
  for (i = 0; i < NUM_EVENTS; i++) {
    PAPI_event_info_t info;
    char name[PAPI_MAX_STR_LEN];
    PAPI_event_code_to_name(events[i], name);
    PAPI_get_event_info(events[i], &info);

    printf(" 0x%08x = %s\t%s\n", events[i], name, info.short_descr);
  }
#endif

  printf("\n");
  printf("Dim.   MFLOPS     Runtime   ");

#ifdef PAPI
  for (i = 0; i < NUM_EVENTS; i++)
    //    printf("%13s", event_names[i]);
    printf("   0x%8x", events[i]);
#endif

  printf("\n");

#ifdef ALLOC_ONCE
  A = oA = (double *) valloc(SQR(test_sizes[NUM_TESTS - 1]) * sizeof(double));
  B = oB = (double *) valloc(SQR(test_sizes[NUM_TESTS - 1]) * sizeof(double));
  C = oC = (double *) valloc(SQR(test_sizes[NUM_TESTS - 1]) * sizeof(double));

  if (!A || !B || !C) {
    printf("measure_performance: valloc() failed\n");
    exit(-1);
  }
#endif

  for (test = 0; test < NUM_TESTS; test++) {
    int matdim = test_sizes[test];
    int iter;
    int ret = 0;

    double mflops;
    double utime;

#ifndef ALLOC_ONCE
    ret |= posix_memalign((void **) &A, 64, SQR(matdim) * sizeof(double));
    ret |= posix_memalign((void **) &B, 64, SQR(matdim) * sizeof(double));
    ret |= posix_memalign((void **) &C, 64, SQR(matdim) * sizeof(double));
    oA = A; oB = B; oC = C;

    if (ret) {
      printf("measure_performance: posix_memalign() failed\n");
      exit(-1);
    }
#endif

    // Fill matrices with random data.
    mat_init(A, matdim, matdim);
    mat_init(B, matdim, matdim);
    mat_init(C, matdim, matdim);

    printf("%4d ", matdim);

    long_enough = expired = 0;
    alarm(1);

    // Mark the current time.
    getrusage(RUSAGE_SELF, &ru_start);

#ifdef PAPI     
    // Enable performance counters.
    if (PAPI_start_counters(events, NUM_EVENTS) == PAPI_OK) papi = 1;
#endif

    // Loop for at least 1 second and stop ASAP after 10 seconds.
    for (iter = 0; !expired && !long_enough; iter++) {
      matmul(matdim, A, B, C);
    }

#ifdef PAPI
    // Stop and read performance counters.
    if (papi) PAPI_stop_counters(values, 4);
#endif

    // Mark the current time.
    getrusage(RUSAGE_SELF, &ru_end);

    alarm(0);

    // Calculate the measured user time.
    utime = timeval_diff(ru_start.ru_utime, ru_end.ru_utime);

    // A silly estimate of MFLOPS..
    mflops = 2.0 * CUBE((long long) matdim) * iter * 1e-6 / utime;

    printf("%8.3f % 11.4f   ", mflops, utime / iter * 1e3);

#ifdef PAPI
    if (papi) {
      for (i = 0; i < NUM_EVENTS; i++)
        printf("% 13.0f", ((double) values[i]) / iter);
    } else {
      printf("   PAPI not available.");
    }
#endif

    printf("\n");

#ifndef ALLOC_ONCE
    free(oA); free(oB); free(oC);
#endif
  }

#ifdef ALLOC_ONCE
  free(oA); free(oB); free(oC);
#endif
}

int main(int argc, char ** argv) {
  int check = 0, measure = 0, help = 0;
  int retval;

  while (1) {
    int c = getopt(argc, argv, "cph");

    if (c == -1) break;

    switch (c) {
    case 'c': check = 1;   break;
    case 'p': measure = 1; break;
    case 'h': help = 1;    break;
    default: break;
    }
  }

  if (help) {
    char help_text[] =
      "EE282 Programming Assignment 1 -- Matrix Multiplication\n\
\n\
Usage: matmul [-c] [-p] [-h]\n\
           -c    Check matmul() for correctness.\n\
           -p    Measure matmul() performance. (default)\n\
           -h    Display this help text.\n\
\n\
";
    fwrite(help_text, strlen(help_text), 1, stdout);
    exit(0);
  }

  signal(SIGALRM, alarm_handler);
  rseed();
  setbuf(stdout, NULL);

#ifdef PAPI
  retval = PAPI_library_init(PAPI_VER_CURRENT);

  if  (retval != PAPI_VER_CURRENT) {
    fprintf(stderr, "PAPI library init error!\n");
    exit(1); 
  }
#endif

  if (check) check_correct();

  if (!check || measure) measure_performance();

  return 0;
}
