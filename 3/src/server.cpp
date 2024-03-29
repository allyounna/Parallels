#include <iostream>
#include <fstream>
#include <random>
#include <thread>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <iomanip>

template<typename T>
T sinus(T x) { return std::sin(x); }

template<typename T>
T square_root(T x) { return std::sqrt(x);}

template<typename T>
T pow(T x, T z) { return std::pow(x, z); }

template<typename T>
class Task
{
private:
    int type;
    T arg1, arg2;

public:
    Task() {}
    Task(int type, T arg1, T arg2)
    : type(type), arg1(arg1), arg2(arg2) {}

    int getType() { return type; }
    T getArg1() { return arg1; }
    T getArg2() { return arg2; }
};


template<typename T>
class Server
{
private:
    std::thread server_thr;
    std::mutex mut;
    std::unordered_map<int, Task<T>> tasks;
    std::unordered_map<int, T> results;
    std::unordered_map<int, int> ready;
    size_t id = 0;
    int status = 0;

    void run() 
    {
        while (status) 
        {
            std::lock_guard<std::mutex> lock(mut);
            for (auto& task : tasks) 
            {
                T result;
                if(task.second.getType() == 1) result = std::sin(task.second.getArg1());
                else if(task.second.getType() == 2) result = square_root(task.second.getArg1());
                else result = pow(task.second.getArg1(), task.second.getArg2());
                results[task.first] = result;
            }
            tasks.clear();
        }
    }

public:
    Server() { server_thr = std::thread(&Server::run, this); }

    void start()
    {
        status = 1;
        std::cout<<"The server is launched"<< std::endl;
    }

    void stop()
    {
        status = 0;
        server_thr.join();
        std::cout<<"The server is closed"<< std::endl;
    }

    size_t add_task(Task<T> task) 
    {
        std::unique_lock<std::mutex> lock(mut);
        tasks[id] = task;
        return id++;
    }

    T request_result(size_t id) 
    {
        while(1)
        {
            std::lock_guard<std::mutex> lock(mut);
            if (results.find(id) != results.end())
            {
                T result = results[id];
                results.erase(id);
                return result;
            }
        }
    }
};

template<typename T>
class Client
{
private:
    Server<T>& server;
    int n, type;
    std::string file;

public:
    Client(Server<T>& server, int n, int type, std::string file) 
    : server(server), n(n), type(type), file(file){}

    void run()
    {
        std::ofstream f(file);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis1(0, 10);
        std::uniform_int_distribution<int> dis2(0, 10);
        f << type << ' ' << n << std::endl;
        for (int i = 0; i < n; i++) 
        {
            T a1 = dis1(gen);
            T a2 = dis2(gen);
            size_t id = server.add_task(Task(type, a1, a2));
            T ans = server.request_result(id);
            if (type == 3) f << id << ' ' << a1 << ' ' << a2  << ' ' << ans << std::endl;
            else f << id << ' ' << a1 << ' ' << ans << std::endl;
        }
        f.close();
    }

};

int main(int argc, char *argv[])
{
    int num_tasks;
    if (argc == 2) num_tasks = atoi(argv[1]);
    else return 1;

    Server<double> server;
    server.start();

    Client<double> client1(server, num_tasks, 1, "sin.txt");
    Client<double> client2(server, num_tasks, 2, "sqrt.txt");
    Client<double> client3(server, num_tasks, 3, "power.txt");

    std::thread serv1(&Client<double>::run, &client1);
    std::thread serv2(&Client<double>::run, &client2);
    std::thread serv3(&Client<double>::run, &client3);

    serv1.join();
    serv2.join();
    serv3.join();

    server.stop();
    return 0;
}