#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <cmath>

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

void matrix_vector_product_v1(double *a, double *b, double *c, int n, int nt)
{
#pragma omp parallel num_threads(nt)
    {
        #pragma omp for
        for (int i = 0; i < n; i++)
        {
            c[i] = 0.0;
            for (int j = 0; j < n; j++)
                c[i] += a[i * n + j] * b[j];
        }
    }
}

void filling_v1(double* arr, double* vec, double* x, size_t n, int nt)
{
    #pragma omp parallel num_threads(nt)
    {
        double v = n + 1;
        #pragma omp for
        for (int i = 0; i < n; i++)
        {
            vec[i] = v;
            x[i] = 0;
            for (int j = 0; j < n; j++)
                if (i == j) arr[i * n + j] = 2;
                else arr[i * n + j] = 1;
        }
    }
}

void matrix_vector_product_v2(double *a, double *b, double *c, int n, int nt)
{
#pragma omp parallel num_threads(nt)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++)
        {
            c[i] = 0.0;
            for (int j = 0; j < n; j++)
                c[i] += a[i * n + j] * b[j];
        }
    }
}

void filling_v2(double* arr, double* vec, double* x, size_t n, int nt)
{
    #pragma omp parallel num_threads(nt)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        double v = n + 1;
        for (int i = lb; i <= ub; i++)
        {
            vec[i] = v;
            x[i] = 0;
            for (int j = 0; j < n; j++)
                if (i == j) arr[i * n + j] = 2;
                else arr[i * n + j] = 1;
        }
    }
}

double norm2(double* u, int n)
{
    double s = 0;
    for (int i = 0; i < n; i++)
        s += u[i] * u[i];
    return sqrt(s);
}

double* sub_v1(double* a, double* b, int n, int nt)
{
    double* res = (double*)malloc(sizeof(*res) * n);
    #pragma omp parallel num_threads(nt)
    {
        #pragma omp for
        for(int i = 0; i < n; i++)
            res[i] = a[i] - b[i];
    }
    return res;
}

double* sub_v2(double* a, double* b, int n, int nt)
{
    double* res = (double*)malloc(sizeof(*res) * n);
    #pragma omp parallel num_threads(nt)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        for(int i = lb; i <= ub; i++)
            res[i] = a[i] - b[i];
    }
    return res;
}

double* mul_v1(double* a, double b, int n, int nt)
{
    double* res = (double*)malloc(sizeof(*res) * n);
    #pragma omp parallel num_threads(nt)
    {
        #pragma omp for
        for(int i = 0; i < n; i++)
            res[i] = a[i] * b;
    }
    return res;
}

double* mul_v2(double* a, double b, int n, int nt)
{
    double* res = (double*)malloc(sizeof(*res) * n);
    #pragma omp parallel num_threads(nt)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        for(int i = lb; i <= ub; i++)
            res[i] = a[i] * b;
    }
    return res;
}

void sol(int n, int nt, int var)
{
    double *a = new double[n * n];
    double *x = new double[n];
    double *b = new double[n];
    double *c = new double[n];
    double eps = 0.00001, tau = 0.00001, t = 0;
    if (var)
    {
        filling_v1(a, b, x, n, nt);
        matrix_vector_product_v1(a, x, c, n, nt);
        c = sub_v1(c, b, n, nt);
        t = cpuSecond();
        while (norm2(c, n) / norm2(b, n) >= eps)
        {
            matrix_vector_product_v1(a, x, c, n, nt);
            c = sub_v1(c, b, n, nt);
            x = sub_v1(x, mul_v1(c, tau, n, nt), n, nt);
            matrix_vector_product_v1(a, x, c, n, nt);
            c = sub_v1(c, b, n, nt);
        }
    }
    else
    {
        filling_v2(a, b, x, n, nt);
        matrix_vector_product_v2(a, x, c, n, nt);
        c = sub_v2(c, b, n, nt);
        t = cpuSecond();
        while (norm2(c, n) / norm2(b, n) >= eps)
        {
            matrix_vector_product_v2(a, x, c, n, nt);
            c = sub_v2(c, b, n, nt);
            x = sub_v2(x, mul_v1(c, tau, n, nt), n, nt);
            matrix_vector_product_v2(a, x, c, n, nt);
            c = sub_v2(c, b, n, nt);
        }
    }
    std:: cout << "Time: " << cpuSecond() - t << std::endl;
    std::cout <<  x[0] << " ";
    std::cout << std::endl;
    delete a;
    delete b;
    delete c;
    delete x;
}

int main(int argc, char *argv[])
{
    int N, nt, var;
    if (argc > 1)
    {
        N = atoi(argv[1]);
        nt = atoi(argv[2]);
        var = atoi(argv[3]);
    }
    else return 1;
    sol(N, nt, var);
    return 0;
}