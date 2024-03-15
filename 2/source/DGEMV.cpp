#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

void matrix_vector_product(double *a, double *b, double *c, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n, int nt)
{
#pragma omp parallel num_threads(nt)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++)
        {
            c[i] = 0.0;
            for (int j = 0; j < n; j++)
                c[i] += a[i * n + j] * b[j];
        }
    }
}

void run_serial(size_t n, size_t m)
{
    double *a = (double*)malloc(sizeof(*a) * m * n);
    double *b = (double*)malloc(sizeof(*b) * n);
    double *c = (double*)malloc(sizeof(*c) * m);

    if (a == NULL || b == NULL || c == NULL)
    {
        free(a);
        free(b);
        free(c);
        printf("Error allocate memory!\n");
        exit(1);
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }

    for (int j = 0; j < n; j++)
        b[j] = j;

    double t = cpuSecond();
    matrix_vector_product(a, b, c, m, n);
    t = cpuSecond() - t;

    printf("Elapsed time (serial): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);
}

void filling(double* arr, double* vec, size_t m, size_t n, int nt){

    #pragma omp parallel num_threads(nt)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++)
        {
            vec[i] = i;
            for (int j = 0; j < n; j++)
                arr[i * n + j] = i + j;
        }
    }
}

void run_parallel(size_t n, size_t m, int nt)
{
    double *a = (double*)malloc(sizeof(*a) * m * n);
    double *b = (double*)malloc(sizeof(*b) * n);
    double *c = (double*)malloc(sizeof(*c) * m);

    if (a == NULL || b == NULL || c == NULL)
    {
        free(a);
        free(b);
        free(c);
        printf("Error allocate memory!\n");
        exit(1);
    }
    filling(a, b, m, n, nt);
    double t = cpuSecond();
    matrix_vector_product_omp(a, b, c, m, n, nt);
    t = cpuSecond() - t;

    printf("Elapsed time (parallel): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);
}

int main(int argc, char *argv[])
{
    int N, nt;
    if (argc > 1)
    {
        N = atoi(argv[1]);
        nt = atoi(argv[2]);
    }
    else return 1;
    run_parallel(N, N, nt);
    return 0;
}
