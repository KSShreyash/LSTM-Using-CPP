#pragma once
#include "Matrix.h"
#include <vector>
#include <string>

class EmbeddingLayer {
public:
    explicit EmbeddingLayer(size_t vocab_size, size_t embedding_dim);

    void loadWeights(const std::string& dir, const std::string& tag = "init");
    void saveWeights(const std::string& dir, const std::string& tag) const;

    Matrix forward(const std::vector<std::vector<int>>& X_batch, size_t t) const;

    void accumulateGrad(const Matrix& d_xh,
                        const std::vector<std::vector<int>>& X_batch,
                        size_t t, size_t H);

    void applyGradients(float lr);

    const Matrix& weights() const;   // ← declaration only, NO inline body

private:
    Matrix emb_;
    Matrix d_emb_;
};