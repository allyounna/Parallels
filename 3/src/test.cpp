#include <iostream>
#include <fstream>
#include <cmath>

int main(int argc, char *argv[])
{
    int type, n, err = 0;
    double id, arg1, arg2, ans;
    std::string filename;
    if (argc == 2) filename = argv[1];
    else return 1;

    std::ifstream f(filename);
    f >> type >> n;
    for (int i = 0; i < n; i++)
    {
        if (type == 3)
        {
            f >> id >> arg1 >> arg2 >> ans;
            if (std::abs(std::pow(arg1, arg2) - ans) > 1e-5) err++;
        }
        else if (type == 2)
        {
            f >> id >> arg1 >> ans;
            if (std::abs(std::sqrt(arg1) - ans) > 1e-5) err++;
        }
        else if (type == 1)
        {
            f >> id >> arg1 >> ans;
            if (std::abs(std::sin(arg1) - ans) > 1e-5) err++;
        }
    }
    f.close();
    std::cout << "There're " << err << " errors" << std::endl;
}