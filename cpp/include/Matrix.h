#pragma once
#include <vector>
#include <string>
#include <cstddef>
#include <cassert>

/**
 * Simple row-major dense matrix.
 * All LSTM math uses this type.
 */
class Matrix {
public:
    size_t rows = 0, cols = 0;
    std::vector<float> data;

    Matrix() = default;
    Matrix(size_t r, size_t c, float init = 0.f);

    // Element access
    float& operator()(size_t i, size_t j);
    float  operator()(size_t i, size_t j) const;

    // Utilities
    void   zero();
    Matrix copy() const;

    // Math
    Matrix matmul(const Matrix& B) const;
    Matrix transpose() const;
    void   axpy(float alpha, const Matrix& B);   // this += alpha * B
    void   scale(float s);

    // I/O
    static Matrix load(const std::string& path, size_t r, size_t c);
    void          save(const std::string& path) const;
};