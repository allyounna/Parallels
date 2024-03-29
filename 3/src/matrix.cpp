#include <iostream>
#include <vector>
#include <thread>
#include <time.h>

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

void filling(std::vector<long long>& vector, int start, int end, int n, std::vector<long long>& matrix)
 {
    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < n; j++)
            matrix[i * n + j] = i + j;
        vector[i] = i;
    }
        
}

void product(std::vector<long long>& vector, std::vector<long long>& matrix, std::vector<long long>& result, int startIndex, int endIndex, int n) 
{
    for (int i = startIndex; i < endIndex; i++)
    {
        result[i] = 0;
        for (int j = 0; j < n; j++)
            result[i] += matrix[i * n + j] * vector[j];
    }
}

void sol(int n, int thrNum)
{
    std::vector<long long> a(n * n);
    std::vector<long long> b(n);
    std::vector<long long> result(n, 0);

    int chunkSizeVec = n / thrNum, stIdx = 0, endIdx = 0;
    std::vector<std::jthread> threads;

     for (int i = 0; i < thrNum; i++) 
     {
        endIdx = (i == thrNum - 1) ? n : stIdx + chunkSizeVec;
        threads.emplace_back(filling, std::ref(b), stIdx, endIdx, n, std::ref(a));
        stIdx = endIdx;
    }

    for (auto& thread : threads)
        thread.join();
    
    threads.clear();
    stIdx = 0;

    double t = cpuSecond();
    for (int i = 0; i < thrNum; ++i)
    {
        endIdx = (i == thrNum - 1) ? n : stIdx + chunkSizeVec;
        threads.emplace_back(product, std::ref(b), std::ref(a), std::ref(result), stIdx, endIdx, n);
        stIdx = endIdx;
    }
    for (auto& thread : threads)
        thread.join();
    threads.clear();
    t = cpuSecond() - t;
    std::cout <<"time "<< t << std::endl;
}

int main(int argc, char *argv[])
{
    int N, nt, var;
    if (argc == 3)
    {
        N = atoi(argv[1]);
        nt = atoi(argv[2]);
    }
    else return 1;
    sol(N, nt);
    return 0;
}
