#define ABS(val) ((val) > 0 ? (val) : -(val))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define SQR(a)   ((a) * (a))
#define CUBE(a)  ((a) * (a) * (a))

void   rseed();
int    rrand(int lower, int upper);
double error(double *mat1, double *mat2, int rows, int cols);
void   mat_init(double *mat,int rows,int cols);
double timeval_diff(struct timeval tv1, struct timeval tv2);
