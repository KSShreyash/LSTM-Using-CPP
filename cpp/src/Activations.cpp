#include "Activations.h"   // already pulls in Matrix.h transitively
#include <cmath>

namespace Activations {

void softmax_cols(Matrix& m, size_t B) {
    size_t rows = m.rows;
    for (size_t b = 0; b < B; ++b) {
        float max_val = m(0, b);
        for (size_t i = 1; i < rows; ++i)
            if (m(i, b) > max_val) max_val = m(i, b);

        float sum = 0.f;
        for (size_t i = 0; i < rows; ++i) {
            m(i, b) = std::exp(m(i, b) - max_val);
            sum += m(i, b);
        }
        for (size_t i = 0; i < rows; ++i)
            m(i, b) /= sum;
    }
}

} // namespace Activations