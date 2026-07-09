#pragma once
#include "LSTMCell.h"
#include "EmbeddingLayer.h"
#include "Matrix.h"
#include <vector>
#include <string>

/**
 * Full unrolled LSTM over T time steps.
 * Owns all four gate weight matrices and biases.
 * Delegates per-step math to LSTMCell.
 */
class LSTMLayer {
public:
    LSTMLayer(size_t hidden_size, size_t embedding_dim);

    void loadWeights(const std::string& dir, const std::string& tag = "init");
    void saveWeights(const std::string& dir, const std::string& tag) const;

    /**
     * Forward pass over the full sequence.
     * @param X_batch   [B][T] token ids
     * @param emb       EmbeddingLayer (to look up x_t)
     * @param cache_out per-step caches for BPTT
     * @returns h_final [H x B] (last hidden state)
     */
    Matrix forward(const std::vector<std::vector<int>>& X_batch,
                   const EmbeddingLayer& emb,
                   std::vector<LSTMStepCache>& cache_out) const;

    /**
     * Backward pass through time.
     * Updates own weights via SGD and returns embedding grad for the caller.
     * @param dh_final  gradient w.r.t. h_final [H x B]
     * @param cache     per-step caches from forward()
     * @param X_batch   [B][T] token ids (to route embedding grads)
     * @param emb       EmbeddingLayer — accumulates embedding grad
     * @param lr        SGD learning rate
     */
    void backward(const Matrix& dh_final,
                  const std::vector<LSTMStepCache>& cache,
                  const std::vector<std::vector<int>>& X_batch,
                  EmbeddingLayer& emb,
                  float lr);

private:
    size_t H_, E_;

    Matrix Wf_, Wi_, Wc_, Wo_;   // [H x (H+E)]
    Matrix bf_, bi_, bc_, bo_;   // [H x 1]

    // Gradient accumulators
    Matrix dWf_, dWi_, dWc_, dWo_;
    Matrix dbf_, dbi_, dbc_, dbo_;

    void zeroGrads();
    void applyGradients(float lr);
};