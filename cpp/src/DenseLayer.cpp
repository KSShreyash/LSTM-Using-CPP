#include "DenseLayer.h"
#include "Activations.h"
#include <cmath>
#include <iostream>

DenseLayer::DenseLayer(size_t output_size, size_t hidden_size)
    : OUT_(output_size), H_(hidden_size),
      Wy_(output_size, hidden_size, 0.f),
      by_(output_size, 1, 0.f),
      dWy_(output_size, hidden_size, 0.f),
      dby_(output_size, 1, 0.f)
{}

void DenseLayer::loadWeights(const std::string& dir, const std::string& tag) {
    Wy_ = Matrix::load(dir + "/w_y_" + tag + ".bin", OUT_, H_);
    by_ = Matrix::load(dir + "/b_y_" + tag + ".bin", OUT_, 1);
    std::cout << "  [DenseLayer] weights loaded (" << tag << ")\n";
}

void DenseLayer::saveWeights(const std::string& dir, const std::string& tag) const {
    Wy_.save(dir + "/w_y_trained_" + tag + ".bin");
    by_.save(dir + "/b_y_trained_" + tag + ".bin");
    std::cout << "  [DenseLayer] weights saved (" << tag << ")\n";
}

Matrix DenseLayer::forward(const Matrix& h_final, size_t B) const {
    Matrix logits = Wy_.matmul(h_final);
    for (size_t i = 0; i < OUT_; ++i)
        for (size_t b = 0; b < B; ++b)
            logits(i, b) += by_(i, 0);

    Activations::softmax_cols(logits, B);
    return logits;   // now contains probs
}

float DenseLayer::computeLoss(const Matrix& probs, const std::vector<int>& y) const {
    float loss = 0.f;
    for (size_t b = 0; b < y.size(); ++b)
        loss -= std::log(probs(y[b], b) + 1e-8f);
    return loss / y.size();
}

Matrix DenseLayer::backward(const Matrix& probs,
                             const std::vector<int>& y_batch,
                             const Matrix& h_final,
                             float lr)
{
    size_t B = y_batch.size();

    // dlogits = (probs - one_hot) / B
    Matrix dlogits = probs.copy();
    for (size_t b = 0; b < B; ++b)
        dlogits(y_batch[b], b) -= 1.f;
    dlogits.scale(1.f / B);

    // Weight grads
    zeroGrads();
    dWy_ = dlogits.matmul(h_final.transpose());
    for (size_t i = 0; i < OUT_; ++i)
        for (size_t b = 0; b < B; ++b)
            dby_(i, 0) += dlogits(i, b);

    // Gradient w.r.t. h_final
    Matrix dh = Wy_.transpose().matmul(dlogits);

    applyGradients(lr);
    return dh;
}

void DenseLayer::zeroGrads() {
    dWy_.zero();
    dby_.zero();
}

void DenseLayer::applyGradients(float lr) {
    Wy_.axpy(-lr, dWy_);
    by_.axpy(-lr, dby_);
}