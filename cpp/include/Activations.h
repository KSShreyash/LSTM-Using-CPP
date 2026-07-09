#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <algorithm>
#include "Matrix.h"        // ← ADD THIS, remove the forward declaration

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Activations {

    inline float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    inline float sigmoid_deriv(float y) {
        return y * (1.0f - y);
    }

    inline float tanh_act(float x) {
        return std::tanh(x);
    }
    inline float tanh_deriv(float y) {
        return 1.0f - y * y;
    }

    inline float relu(float x) { return x > 0.f ? x : 0.f; }
    inline float relu_deriv(float x) { return x > 0.f ? 1.f : 0.f; }

    // ← Parameter is now the real ::Matrix, NOT Activations::Matrix
    void softmax_cols(Matrix& m, size_t B);

} // namespace Activations