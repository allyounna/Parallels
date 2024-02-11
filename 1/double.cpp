#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>

double arr[10000000];

int main()
{
    double s = 0, step = (M_PI * 2) / 10000000;
    for (int i = 0; i < 10000000; i++)
    {
        arr[i] = sin(i * step);
        s += arr[i];
    }
    std::cout << s << std::endl;
    return 0;
}   
