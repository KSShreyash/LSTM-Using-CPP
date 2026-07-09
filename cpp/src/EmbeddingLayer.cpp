#include "EmbeddingLayer.h"
#include <iostream>

EmbeddingLayer::EmbeddingLayer(size_t vocab_size, size_t embedding_dim)
    : emb_(vocab_size, embedding_dim, 0.f),
      d_emb_(vocab_size, embedding_dim, 0.f) {}

void EmbeddingLayer::loadWeights(const std::string& dir, const std::string& tag) {
    emb_ = Matrix::load(dir + "/embedding_" + tag + ".bin",
                        emb_.rows, emb_.cols);
    std::cout << "  [Embedding] weights loaded (" << tag << ")\n";
}

void EmbeddingLayer::saveWeights(const std::string& dir, const std::string& tag) const {
    emb_.save(dir + "/embedding_trained_" + tag + ".bin");
    std::cout << "  [Embedding] weights saved (" << tag << ")\n";
}

Matrix EmbeddingLayer::forward(const std::vector<std::vector<int>>& X_batch,
                                size_t t) const {
    size_t B = X_batch.size();
    size_t E = emb_.cols;
    Matrix x_t(B, E);
    for (size_t b = 0; b < B; ++b) {
        int idx = X_batch[b][t];
        for (size_t e = 0; e < E; ++e)
            x_t(b, e) = emb_(idx, e);
    }
    return x_t;
}

void EmbeddingLayer::accumulateGrad(const Matrix& d_xh,
                                     const std::vector<std::vector<int>>& X_batch,
                                     size_t t, size_t H) {
    size_t B = X_batch.size();
    size_t E = emb_.cols;
    for (size_t b = 0; b < B; ++b) {
        int idx = X_batch[b][t];
        for (size_t e = 0; e < E; ++e)
            d_emb_(idx, e) += d_xh(H + e, b);
    }
}

void EmbeddingLayer::applyGradients(float lr) {
    emb_.axpy(-lr, d_emb_);
    d_emb_.zero();
}

const Matrix& EmbeddingLayer::weights() const { return emb_; }