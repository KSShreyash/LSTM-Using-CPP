#pragma once
#include "Matrix.h"
#include <string>

/**
 * Fully-connected output layer with softmax activation.
 * Applies: probs = softmax(Wy * h + by)
 */
class DenseLayer {
public:
    DenseLayer(size_t output_size, size_t hidden_size);

    void loadWeights(const std::string& dir, const std::string& tag = "init");
    void saveWeights(const std::string& dir, const std::string& tag) const;

    /**
     * Forward: logits → softmax probabilities.
     * @param h_final  [H x B]
     * @returns probs  [OUT x B]
     */
    Matrix forward(const Matrix& h_final, size_t B) const;

    /**
     * Backward: given softmax probs and true labels,
     * computes dh (gradient w.r.t. h_final) and updates Wy, by via SGD.
     * @param probs     [OUT x B] from forward()
     * @param y_batch   true labels, length B
     * @param h_final   [H x B]
     * @param lr        SGD learning rate
     * @returns dh      [H x B]
     */
    Matrix backward(const Matrix& probs,
                    const std::vector<int>& y_batch,
                    const Matrix& h_final,
                    float lr);

    float computeLoss(const Matrix& probs, const std::vector<int>& y) const;

private:
    size_t OUT_, H_;
    Matrix Wy_;   // [OUT x H]
    Matrix by_;   // [OUT x 1]

    Matrix dWy_, dby_;

    void zeroGrads();
    void applyGradients(float lr);
};