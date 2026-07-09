#include "DataLoader.h"
#include <fstream>
#include <iostream>
#include <cstdlib>

std::vector<std::vector<int>> DataLoader::loadX(const std::string& path,
                                                  size_t N, size_t T) {
    std::vector<int> flat(N * T);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "ERROR: cannot open " << path << "\n";
        std::exit(1);
    }
    f.read(reinterpret_cast<char*>(flat.data()), flat.size() * sizeof(int));

    std::vector<std::vector<int>> out(N, std::vector<int>(T));
    for (size_t i = 0; i < N; ++i)
        for (size_t t = 0; t < T; ++t)
            out[i][t] = flat[i * T + t];
    return out;
}

std::vector<int> DataLoader::loadY(const std::string& path, size_t N) {
    std::vector<int> y(N);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "ERROR: cannot open " << path << "\n";
        std::exit(1);
    }
    f.read(reinterpret_cast<char*>(y.data()), y.size() * sizeof(int));
    return y;
}

std::vector<std::vector<int>> DataLoader::slice(
    const std::vector<std::vector<int>>& X, size_t begin, size_t end)
{
    return {X.begin() + begin, X.begin() + end};
}

std::vector<int> DataLoader::slice(const std::vector<int>& y,
                                    size_t begin, size_t end)
{
    return {y.begin() + begin, y.begin() + end};
}