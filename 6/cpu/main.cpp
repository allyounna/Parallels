#include <iostream>
#include <boost/program_options.hpp>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>

namespace po = boost::program_options;

void initialize(std::unique_ptr<double[]> &A, std::unique_ptr<double[]> &Anew, int n)
{
    memset(A.get(), 0, n * n * sizeof(double));
    A[0] = 10;
    A[n - 1] = 20;
    A[n * n - 1] = 30;
    A[n * (n - 1)] =  20;
    double step = (A[n - 1] - A[0]) / (n - 1);
    for (int i = 1; i < n - 1; i++) 
    {
        A[i] = A[0] + i * step;
        A[n * i] =A[0] + i * step;
        A[(n-1) + n * i] = A[n - 1] + i * step;
        A[n * (n-1) + i] = A[n * (n - 1)] + i * step;
    }
    std::memcpy(Anew.get(), A.get(), n * n * sizeof(double));
}


int main(int argc, char* argv[]) 
{
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "Help message")
        ("precision", po::value<double>()->default_value(0.000001), "Set precision")
        ("grid-size", po::value<int>()->default_value(1024), "Set grid size")
        ("iterations", po::value<int>()->default_value(1000000), "Set number of iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) 
    {
        std::cout << desc << std::endl;
        return 1;
    }
    po::notify(vm);

    double precision = vm["precision"].as<double>();
    int n = vm["grid-size"].as<int>();
    int max_it= vm["iterations"].as<int>();

    double error = 1.0;

    std::unique_ptr<double[]> A_ptr(new double[n*n]);
    std::unique_ptr<double[]> Anew_ptr(new double[n*n]);
    initialize(std::ref(A_ptr), std::ref(Anew_ptr), n);
    double* A = A_ptr.get();
    double* Anew = Anew_ptr.get();

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, n);

    auto time_point = std::chrono::high_resolution_clock::now();
    int it = 0;
    #pragma acc data copyin(A[:n*n], Anew[:n*n], error)
    {
        while (error > precision && it < max_it)
        {
            if(it % 100 == 0)
            {
                error = 0;
                #pragma acc update device(error) async(1)
                #pragma acc parallel loop independent collapse(2) vector vector_length(n) gang num_gangs(n) reduction(max:error) present(A, Anew)
                for(int j = 1; j < n - 1; j++)
                    for(int i = 1; i < n - 1; i++) 
                    {
                        Anew[j * n + i] = (A[j * n + i + 1] + A[j * n + i - 1] + A[(j - 1) * n + i] + A[(j + 1) * n + i]) * 0.25;
                        error = fmax(error, fabs(Anew[j * n + i] - A[j * n + i]));
                    }
                #pragma acc update host(error) async(1)	
                #pragma acc wait(1)
                printf("%5d, %0.6f\n", it, error);
            }
            else
            {
                #pragma acc parallel loop independent collapse(2) vector vector_length(n) gang num_gangs(n) present(A, Anew)
                for(int j = 1; j < n - 1; j++)
                    for(int i = 1; i < n - 1; i++)
                        Anew[j * n + i] = (A[j * n + i + 1] + A[j * n + i - 1] + A[(j - 1) * n + i] + A[(j + 1) * n + i]) * 0.25;
            }
            double* buf = A;
            A = Anew, Anew = buf;
            it++;
        }
        std::chrono::duration<double> t = std::chrono::high_resolution_clock::now() - time_point;
        printf("%5d, %0.6f\n", it, error);
        printf(" total time (sec): %f s\n", t);
    }
    std::ofstream file("out.txt");
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
           file << std::left << std::setw(10) << A[j * n + i] << " ";
        file << std::endl;
    }
    file.close();
    return 0;
}