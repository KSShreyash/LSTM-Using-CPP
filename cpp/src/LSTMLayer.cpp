#include <iostream>
#include "LSTMLayer.h"
#include "LSTMCell.h"
#include <cassert>

LSTMLayer::LSTMLayer(size_t hidden_size, size_t embedding_dim)
    : H_(hidden_size), E_(embedding_dim),
      Wf_(H_, H_+E_, 0.f), Wi_(H_, H_+E_, 0.f),
      Wc_(H_, H_+E_, 0.f), Wo_(H_, H_+E_, 0.f),
      bf_(H_, 1, 0.f), bi_(H_, 1, 0.f),
      bc_(H_, 1, 0.f), bo_(H_, 1, 0.f),
      dWf_(H_, H_+E_, 0.f), dWi_(H_, H_+E_, 0.f),
      dWc_(H_, H_+E_, 0.f), dWo_(H_, H_+E_, 0.f),
      dbf_(H_, 1, 0.f), dbi_(H_, 1, 0.f),
      dbc_(H_, 1, 0.f), dbo_(H_, 1, 0.f)
{}

void LSTMLayer::loadWeights(const std::string& dir, const std::string& tag) {
    Wf_ = Matrix::load(dir + "/w_f_" + tag + ".bin", H_, H_+E_);
    Wi_ = Matrix::load(dir + "/w_i_" + tag + ".bin", H_, H_+E_);
    Wc_ = Matrix::load(dir + "/w_c_" + tag + ".bin", H_, H_+E_);
    Wo_ = Matrix::load(dir + "/w_o_" + tag + ".bin", H_, H_+E_);
    bf_ = Matrix::load(dir + "/b_f_" + tag + ".bin", H_, 1);
    bi_ = Matrix::load(dir + "/b_i_" + tag + ".bin", H_, 1);
    bc_ = Matrix::load(dir + "/b_c_" + tag + ".bin", H_, 1);
    bo_ = Matrix::load(dir + "/b_o_" + tag + ".bin", H_, 1);
    std::cout << "  [LSTMLayer] weights loaded (" << tag << ")\n";
}

void LSTMLayer::saveWeights(const std::string& dir, const std::string& tag) const {
    Wf_.save(dir + "/w_f_trained_" + tag + ".bin");
    Wi_.save(dir + "/w_i_trained_" + tag + ".bin");
    Wc_.save(dir + "/w_c_trained_" + tag + ".bin");
    Wo_.save(dir + "/w_o_trained_" + tag + ".bin");
    bf_.save(dir + "/b_f_trained_" + tag + ".bin");
    bi_.save(dir + "/b_i_trained_" + tag + ".bin");
    bc_.save(dir + "/b_c_trained_" + tag + ".bin");
    bo_.save(dir + "/b_o_trained_" + tag + ".bin");
    std::cout << "  [LSTMLayer] weights saved (" << tag << ")\n";
}

Matrix LSTMLayer::forward(const std::vector<std::vector<int>>& X_batch,
                           const EmbeddingLayer& emb,
                           std::vector<LSTMStepCache>& cache_out) const
{
    size_t B = X_batch.size();
    size_t T = X_batch[0].size();

    Matrix h(H_, B, 0.f);
    Matrix c(H_, B, 0.f);
    cache_out.clear();

    for (size_t t = 0; t < T; ++t) {
        Matrix x_t = emb.forward(X_batch, t);

        Matrix xh(H_ + E_, B);
        for (size_t i = 0; i < H_; ++i)
            for (size_t b = 0; b < B; ++b)
                xh(i, b) = h(i, b);
        for (size_t i = 0; i < E_; ++i)
            for (size_t b = 0; b < B; ++b)
                xh(H_ + i, b) = x_t(b, i);

        LSTMStepCache sc;
        auto result = LSTMCell::forward(
            xh, h, c,
            Wf_, Wi_, Wc_, Wo_,
            bf_, bi_, bc_, bo_,
            sc);

        cache_out.push_back(sc);
        h = result.first;
        c = result.second;
    }
    return h;
}

void LSTMLayer::backward(const Matrix& dh_final,
                          const std::vector<LSTMStepCache>& cache,
                          const std::vector<std::vector<int>>& X_batch,
                          EmbeddingLayer& emb,
                          float lr)
{
    size_t B = X_batch.size();
    size_t T = cache.size();

    zeroGrads();

    Matrix dh = dh_final.copy();
    Matrix dc(H_, B, 0.f);

    for (int t = (int)T - 1; t >= 0; --t) {
        Matrix d_xh = LSTMCell::backward(
            dh, dc, cache[t],
            Wf_, Wi_, Wc_, Wo_,
            dWf_, dWi_, dWc_, dWo_,
            dbf_, dbi_, dbc_, dbo_,
            B);

        for (size_t i = 0; i < H_; ++i)
            for (size_t b = 0; b < B; ++b)
                dh(i, b) = d_xh(i, b);

        emb.accumulateGrad(d_xh, X_batch, t, H_);
    }

    applyGradients(lr);
    emb.applyGradients(lr);
}

void LSTMLayer::zeroGrads() {
    dWf_.zero(); dWi_.zero(); dWc_.zero(); dWo_.zero();
    dbf_.zero(); dbi_.zero(); dbc_.zero(); dbo_.zero();
}

void LSTMLayer::applyGradients(float lr) {
    Wf_.axpy(-lr, dWf_); Wi_.axpy(-lr, dWi_);
    Wc_.axpy(-lr, dWc_); Wo_.axpy(-lr, dWo_);
    bf_.axpy(-lr, dbf_); bi_.axpy(-lr, dbi_);
    bc_.axpy(-lr, dbc_); bo_.axpy(-lr, dbo_);
}
