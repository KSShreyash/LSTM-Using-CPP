#include "Matrix.h"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <numeric>

Matrix::Matrix(size_t r, size_t c, float init)
    : rows(r), cols(c), data(r * c, init) {}

float& Matrix::operator()(size_t i, size_t j) {
    return data[i * cols + j];
}
float Matrix::operator()(size_t i, size_t j) const {
    return data[i * cols + j];
}

void Matrix::zero() {
    std::fill(data.begin(), data.end(), 0.f);
}

Matrix Matrix::copy() const {
    Matrix m(rows, cols);
    m.data = data;
    return m;
}

Matrix Matrix::matmul(const Matrix& B) const {
    assert(cols == B.rows);
    Matrix C(rows, B.cols, 0.f);
    for (size_t i = 0; i < rows; ++i)
        for (size_t k = 0; k < cols; ++k) {
            float a = (*this)(i, k);
            for (size_t j = 0; j < B.cols; ++j)
                C(i, j) += a * B(k, j);
        }
    return C;
}

Matrix Matrix::transpose() const {
    Matrix R(cols, rows);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            R(j, i) = (*this)(i, j);
    return R;
}

void Matrix::axpy(float alpha, const Matrix& B) {
    assert(data.size() == B.data.size());
    for (size_t i = 0; i < data.size(); ++i)
        data[i] += alpha * B.data[i];
}

void Matrix::scale(float s) {
    for (auto& x : data) x *= s;
}

Matrix Matrix::load(const std::string& path, size_t r, size_t c) {
    Matrix m(r, c);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "ERROR: cannot open " << path << "\n";
        std::exit(1);
    }
    f.read(reinterpret_cast<char*>(m.data.data()),
           m.data.size() * sizeof(float));
    return m;
}

void Matrix::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "ERROR: cannot write " << path << "\n";
        std::exit(1);
    }
    f.write(reinterpret_cast<const char*>(data.data()),
            data.size() * sizeof(float));
}