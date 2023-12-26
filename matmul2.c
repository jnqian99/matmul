// https://rosettacode.org/wiki/Matrix_multiplication#C

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "matmul.h"
#include "vectorclass.h"

#define MAT_ELEM(rows,cols,r,c) (r*cols+c)

//Improve performance by assuming output matrices do not overlap with
//input matrices. If this is C++, use the __restrict extension instead
#ifdef __cplusplus
    typedef double * const __restrict MAT_OUT_t;
    typedef const double * const __restrict MAT_IN_t;
#else
    typedef double * const restrict MAT_OUT_t;
    typedef const double * const restrict MAT_IN_t;
#endif


#ifdef __cplusplus
    typedef float * const __restrict MAT_OUT_f;
    typedef const float * const __restrict MAT_IN_f;
#else
    typedef float * const restrict MAT_OUT_f;
    typedef const float * const restrict MAT_IN_f;
#endif


// adapted from timing.c
#define TIMING_RESULT(descr, CODE) do { \
    struct timespec start, end; \
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start); \
    CODE; \
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end); \
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; \
    printf("%25s took %7.2f ms\n", descr, elapsed * 1000); \
} while(0)

static inline void mat_mult(
    const int m,
    const int n,
    const int p, 
    MAT_IN_t a,
    MAT_IN_t b,
    MAT_OUT_t c)
{
    for (int row=0; row<m; row++) {
        for (int col=0; col<p; col++) {
            c[MAT_ELEM(m,p,row,col)] = 0;
            for (int i=0; i<n; i++) {
                c[MAT_ELEM(m,p,row,col)] += a[MAT_ELEM(m,n,row,i)]*b[MAT_ELEM(n,p,i,col)];
            }
        }
    }
}

static inline void mat_mult_f(
    const int m,
    const int n,
    const int p, 
    MAT_IN_f a,
    MAT_IN_f b,
    MAT_OUT_f c)
{
    for (int row=0; row<m; row++) {
        for (int col=0; col<p; col++) {
            c[MAT_ELEM(m,p,row,col)] = 0;
            for (int i=0; i<n; i++) {
                c[MAT_ELEM(m,p,row,col)] += a[MAT_ELEM(m,n,row,i)]*b[MAT_ELEM(n,p,i,col)];
            }
        }
    }
}

static inline void mat_show(
    const int m,
    const int p,
    MAT_IN_t a)
{
    for (int row=0; row<m;row++) {
        for (int col=0; col<p;col++) {
            printf("\t%7.3f", a[MAT_ELEM(m,p,row,col)]);
        }
        putchar('\n');
    }
}

static inline void mat_show_f(
    const int m,
    const int p,
    double* a)
{
    for (int row=0; row<m;row++) {
        for (int col=0; col<p;col++) {
            printf("\t%7.3f", a[MAT_ELEM(m,p,row,col)]);
        }
        putchar('\n');
    }
}

static inline double* mat_ccreate(int row, int col) {
    srand(time(NULL));
    double *m = (double*)malloc(row*col*sizeof(double));
    for(int i=0; i<row*col; i++) {
        m[i]=((double)rand() / RAND_MAX - 0.25);;
    }
    return m;
}

static inline float* mat_ccreate_f(int row, int col) {
    srand(time(NULL));
    float *m = (float*)malloc(row*col*sizeof(float));
    for(int i=0; i<row*col; i++) {
        m[i]=((float)rand() / RAND_MAX - 0.25);;
    }
    return m;
}


static inline double *mat_t(double *mat, int row, int col) {
    double * mat2 = (double*) malloc(row*col*sizeof(double));
    for (int r=0; r<row; r++) {
        for (int c=0; c<col; c++) {
            mat2[MAT_ELEM(col,row,c,r)]=mat[MAT_ELEM(row,col,r,c)];
        }
    }
    return mat2;
}

static inline float *mat_t_f(float *mat, int row, int col) {
    float * mat2 = (float*) malloc(row*col*sizeof(float));
    for (int r=0; r<row; r++) {
        for (int c=0; c<col; c++) {
            mat2[MAT_ELEM(col,row,c,r)]=mat[MAT_ELEM(row,col,r,c)];
        }
    }
    return mat2;
}

#define BUF_LEN 16
static inline void mat_mult4(
    const int m,
    const int n,
    const int p, 
    MAT_IN_t a,
    MAT_IN_t b_t,
    MAT_OUT_t c)
{
    double tmp[BUF_LEN] = {0};
    for (int row=0; row<m; row++) {
        for (int col=0; col<p; col++) {
            for(int i=0; i<BUF_LEN; i++)
                tmp[i]=0.0;
            for (int i=0; i<n; i+=BUF_LEN) {
                for(int j=0; j<BUF_LEN; j++)
                    tmp[j]+= a[MAT_ELEM(m,n,row,(i+j))]*b_t[MAT_ELEM(p,n,col,(i+j))];
            }
            double total=0.0;
            for(int i=0; i<BUF_LEN; i++)
                total+=tmp[i];
            c[MAT_ELEM(m,p,row,col)] = total;
        }
    }
}

static inline void mat_mult4_f(
    const int m,
    const int n,
    const int p, 
    MAT_IN_f a,
    MAT_IN_f b_t,
    MAT_OUT_f c)
{
    float tmp[BUF_LEN] = {0};
    for (int row=0; row<m; row++) {
        for (int col=0; col<p; col++) {
            for(int i=0; i<BUF_LEN; i++)
                tmp[i]=0.0;
            for (int i=0; i<n; i+=BUF_LEN) {
                for(int j=0; j<BUF_LEN; j++)
                    tmp[j]+= a[MAT_ELEM(m,n,row,(i+j))]*b_t[MAT_ELEM(p,n,col,(i+j))];
            }
            float total=0.0;
            for(int i=0; i<BUF_LEN; i++)
                total+=tmp[i];
            c[MAT_ELEM(m,p,row,col)] = total;
        }
    }
}

static inline void mat_mult5(
    const int m,
    const int n,
    const int p, 
    MAT_IN_t a,
    MAT_IN_t b_t,
    MAT_OUT_t c)
{
    double tmp[BUF_LEN] = {0};
    for (int row=0; row<m; row++) {
        for (int col=0; col<p; col++) {
            for(int i=0; i<BUF_LEN; i++)
                tmp[i]=0.0;
            for (int i=0; i<n; i+=(BUF_LEN*4)) {
                for(int j=0; j<BUF_LEN; j++) {
                    Vec4d v1, v2;
                    v1.load(a+MAT_ELEM(m,n,row,(i+j*4)));
                    v2.load(b_t+MAT_ELEM(p,n,col,(i+j*4)));
                    v1 *= v2;
                    tmp[j] += horizontal_add(v1);
                }
            }
            double total=0.0;
            for(int i=0; i<BUF_LEN; i++)
                total+=tmp[i];
            c[MAT_ELEM(m,p,row,col)] = total;
        }
    }
}

static inline void mat_mult5_f(
    const int m,
    const int n,
    const int p, 
    MAT_IN_f a,
    MAT_IN_f b_t,
    MAT_OUT_f c)
{
    float tmp[BUF_LEN] = {0};
    for (int row=0; row<m; row++) {
        for (int col=0; col<p; col++) {
            for(int i=0; i<BUF_LEN; i++)
                tmp[i]=0.0;
            for (int i=0; i<n; i+=(BUF_LEN*8)) {
                for(int j=0; j<BUF_LEN; j++) {
                    Vec8f v1, v2;
                    v1.load(a+MAT_ELEM(m,n,row,(i+j*8)));
                    v2.load(b_t+MAT_ELEM(p,n,col,(i+j*8)));
                    v1 *= v2;
                    tmp[j] += horizontal_add(v1);
                }
            }
            float total=0.0;
            for(int i=0; i<BUF_LEN; i++)
                total+=tmp[i];
            c[MAT_ELEM(m,p,row,col)] = total;
        }
    }
}


static inline void mat_mult2(
    const int m,
    const int n,
    const int p, 
    MAT_IN_t a,
    MAT_IN_t b_t,
    MAT_OUT_t c)
{
    for (int row=0; row<m; row++) {
        for (int col=0; col<p; col++) {
            int idx=MAT_ELEM(m,p,row,col);
            c[idx] = 0;
            int idx1=n*row;
            int idx2=n*col;
            for (int i=0; i<n; i++) {
                //c[idx] += a[MAT_ELEM(m,n,row,i)]*b_t[MAT_ELEM(p,n,col,i)];
                c[idx] += a[idx1++]*b_t[idx2++];
            }
        }
    }

}

static inline void mat_mult2_f(
    const int m,
    const int n,
    const int p, 
    MAT_IN_f a,
    MAT_IN_f b_t,
    MAT_OUT_f c)
{
    for (int row=0; row<m; row++) {
        for (int col=0; col<p; col++) {
            int idx=MAT_ELEM(m,p,row,col);
            c[idx] = 0;
            int idx1=n*row;
            int idx2=n*col;
            for (int i=0; i<n; i++) {
                //c[idx] += a[MAT_ELEM(m,n,row,i)]*b_t[MAT_ELEM(p,n,col,i)];
                c[idx] += a[idx1++]*b_t[idx2++];
            }
        }
    }
}


static inline void mat_mult3(
    const int m,
    const int n,
    const int p, 
    MAT_IN_t a,
    MAT_IN_t b_t,
    MAT_OUT_t c)
{
    for (int row=0; row<m; row++) {
        for (int col=0; col<p; col++) {
            int idx=MAT_ELEM(m,p,row,col);
            c[idx] = 0;
            int idx1=n*row;
            int idx2=n*col;
            for (int i=0; i<n; i+=4) {
                Vec4d v1, v2;
                v1.load(a+idx1);
                v2.load(b_t+idx2);
                v1 += v2;
                //c[idx] += a[MAT_ELEM(m,n,row,i)]*b_t[MAT_ELEM(p,n,col,i)];
                c[idx] += horizontal_add(v1);
                idx1+=4;
                idx2+=4;
            }
        }
    }
}

static inline void mat_mult3_f(
    const int m,
    const int n,
    const int p, 
    MAT_IN_f a,
    MAT_IN_f b_t,
    MAT_OUT_f c)
{
    for (int row=0; row<m; row++) {
        for (int col=0; col<p; col++) {
            int idx=MAT_ELEM(m,p,row,col);
            c[idx] = 0;
            int idx1=n*row;
            int idx2=n*col;
            for (int i=0; i<n; i+=8) {
                Vec8f v1, v2;
                v1.load(a+idx1);
                v2.load(b_t+idx2);
                v1 += v2;
                //c[idx] += a[MAT_ELEM(m,n,row,i)]*b_t[MAT_ELEM(p,n,col,i)];
                c[idx] += horizontal_add(v1);
                idx1+=8;
                idx2+=8;
            }
        }
    }
}


#define M 256*28
#define N 256*28
#define P 256*28

int main(void)
{
    printf("m=%d, n=%d, p=%d\n", M, N, P);

    double *a = mat_ccreate(M,N);

    double *b = mat_ccreate(N,P);
    double *b_t = mat_t(b, N, P);

    double *c = (double*)malloc(M*P*sizeof(double));
/*
    double a1[] = {1.1,1.2,1.3,1.4,
                  2.1,2.2,2.3,2.4,
                  3.1,3.2,3.3,3.4,
                  4.1,4.2,4.3,4.4};

    double b1[] = {1.1,1.2,1.3,1.4,
                  2.1,2.2,2.3,2.4,
                  3.1,3.2,3.3,3.4,
                  4.1,4.2,4.3,4.4};

    double b1_t[] = {1.1,2.1,3.1,4.1,
                  1.2,2.2,3.2,4.2,
                  1.3,2.3,3.3,4.3,
                  1.4,2.4,3.4,4.4};
    
    
    TIMING_RESULT("matmul_d", matmul_d(M,N,P,a,b_t,c));

    memset(c, 0, M*P*sizeof(double));
    TIMING_RESULT("matmul_d", matmul_d(M,N,P,a,b_t,c));
    //mat_show(M,P,c);

    memset(c, 0, M*P*sizeof(double));

    TIMING_RESULT("mat_mult5", mat_mult5(M,N,P,a,b_t,c));
    //mat_show(M,P,c);

    memset(c, 0, M*P*sizeof(double));

    TIMING_RESULT("mat_mult4", mat_mult4(M,N,P,a,b_t,c));
    //mat_show(M,P,c);

    memset(c, 0, M*P*sizeof(double));

    TIMING_RESULT("mat_mult3", mat_mult3(M,N,P,a,b_t,c));
    //mat_show(M,P,c);

    memset(c, 0, M*P*sizeof(double));
*/
    TIMING_RESULT("mat_mult2", mat_mult2(M,N,P,a,b_t,c));
    //mat_show(M,P,c);

    //memset(c, 0, M*P*sizeof(double));
    
    //TIMING_RESULT("mat_mult", mat_mult(M,N,P,a,b,c));
    //mat_show(M,P,c);

    free(a);
    free(b);
    free(b_t);
    free(c);

    ////////////////////////////////////////////
/*    
    float *af = mat_ccreate_f(M,N);

    float *bf = mat_ccreate_f(N,P);
    float *bf_t = mat_t_f(bf, N, P);

    float *cf = (float*)malloc(M*P*sizeof(float));

    TIMING_RESULT("matmul_f", matmul_f(M,N,P,af,bf_t,cf));
    //mat_show(M,P,c);

    memset(cf, 0, M*P*sizeof(float));

    TIMING_RESULT("mat_mult5_f", mat_mult5_f(M,N,P,af,bf_t,cf));
    //mat_show(M,P,c);

    memset(cf, 0, M*P*sizeof(float));

    TIMING_RESULT("mat_mult4_f", mat_mult4_f(M,N,P,af,bf_t,cf));
    //mat_show(M,P,c);

    memset(cf, 0, M*P*sizeof(float));

    TIMING_RESULT("mat_mult3_f", mat_mult3_f(M,N,P,af,bf_t,cf));
    //mat_show(M,P,c);

    memset(cf, 0, M*P*sizeof(float));

    //TIMING_RESULT("mat_mult2_f", mat_mult2_f(M,N,P,af,bf_t,cf));
    //mat_show(M,P,c);

    //memset(cf, 0, M*P*sizeof(float));

    TIMING_RESULT("mat_mult_f", mat_mult_f(M,N,P,af,bf,cf));
    //mat_show(M,P,c);

    free(af);
    free(bf);
    free(bf_t);
    free(cf);
*/
    return 0;
}