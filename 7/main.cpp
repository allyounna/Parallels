#include <iostream>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cublas_v2.h>
#include <boost/program_options.hpp>
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


int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "Produce help message")
        ("precision", po::value<double>()->default_value(0.000001), "Set precision")
        ("grid-size", po::value<int>()->default_value(128), "Set grid size")
        ("iterations", po::value<int>()->default_value(1000000), "Set number of iterations");

    po::variables_map vm;
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    double precision = vm["precision"].as<double>();
    int n = vm["grid-size"].as<int>();
    int iter_max= vm["iterations"].as<int>();

    std::unique_ptr<double[]> A_ptr(new double[n*n]);
    std::unique_ptr<double[]> Anew_ptr(new double[n*n]);
    initialize(std::ref(A_ptr), std::ref(Anew_ptr), n);
    double* A = A_ptr.get();
    double* Anew = Anew_ptr.get();

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, n);

    cublasHandle_t handler;
	cublasStatus_t status;
	
	int iter = 0, idx = 0;
	double alpha = -1, error = 1;
	status = cublasCreate(&handler);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        std::cerr << "cublasCreate failed with error code: " << status << std::endl;
        exit(13);
    }
    #pragma acc data copyin(A[:n*n],Anew[:n*n], alpha, n, idx)
    {
        auto start = std::chrono::high_resolution_clock::now();
        while (error > precision && iter < iter_max)
        {
            #pragma acc parallel loop independent collapse(2) vector vector_length(n) gang num_gangs(n) present(A,Anew)
                for(int j = 1; j < n - 1; j++)
                    for(int i = 1; i < n - 1; i++) 
                    {
                        Anew[j * n + i] = (A[j * n + i + 1] + A[j * n + i - 1] + A[(j - 1) * n + i] + A[(j + 1) * n + i]) * 0.25;
                        error = fmax(error, fabs(Anew[j * n + i] - A[j * n + i]));
                    }
            if (iter % 100 == 0)
            {
                #pragma acc data present (A, Anew) wait
                #pragma acc host_data use_device(A, Anew)
                {
                    status = cublasDaxpy(handler, n * n, &alpha, Anew, 1, A, 1);
                    if (status != CUBLAS_STATUS_SUCCESS) 
                    {
                        std::cerr << "cublasDaxpy failed with error code: " << status << std::endl;
                        exit (13);
                    }
                    status = cublasIdamax(handler, n * n, A, 1, &idx);
                    if (status != CUBLAS_STATUS_SUCCESS) 
                    {
                        std::cerr << "cublasIdamax failed with error code: " << status << std::endl;
                        exit (13);
                    }
                }

                #pragma acc update host(A[idx - 1]) 
                error = std::fabs(A[idx - 1]);
                #pragma acc host_data use_device(A, Anew)
                status = cublasDcopy(handler, n * n, Anew, 1, A, 1);
                if (status != CUBLAS_STATUS_SUCCESS) 
                {
                    std::cerr << "cublasDcopy failed with error code: " << status << std::endl;
                    exit (13);
                }
                printf("%5d, %0.6f\n", iter, error);
			}

            double* temp = A;
            A = Anew;
            Anew = temp;
            iter++;

        }
      
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> runtime = end - start;
        printf("%5d, %0.6f\n", iter, error);
        printf(" total: %f s\n", runtime);
        cublasDestroy(handler);
    }
    return 0;
}