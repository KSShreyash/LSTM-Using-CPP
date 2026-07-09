#include "LSTMModel.h"
#include "DataLoader.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

static const char* LABELS[] = {"NEGATIVE", "NEUTRAL", "POSITIVE"};

LSTMModel::LSTMModel(const Config& cfg)
    : cfg_(cfg),
      embedding_(cfg.vocab_size, cfg.embedding_dim),
      lstm_(cfg.hidden_size, cfg.embedding_dim),
      dense_(cfg.output_size, cfg.hidden_size)
{}

void LSTMModel::loadWeights(const std::string& dir, const std::string& tag) {
    std::cout << "\n[LSTMModel] Loading weights (" << tag << ") from: " << dir << "\n";
    embedding_.loadWeights(dir, tag);
    lstm_.loadWeights(dir, tag);
    dense_.loadWeights(dir, tag);
    std::cout << "[LSTMModel] All weights loaded.\n";
}

void LSTMModel::saveWeights(const std::string& dir, const std::string& tag) const {
    std::cout << "\n[LSTMModel] Saving weights (" << tag << ") to: " << dir << "\n";
    embedding_.saveWeights(dir, tag);
    lstm_.saveWeights(dir, tag);
    dense_.saveWeights(dir, tag);
    std::cout << "[LSTMModel] All weights saved.\n";
}

float LSTMModel::trainBatch(const std::vector<std::vector<int>>& X_batch,
                             const std::vector<int>& y_batch,
                             float lr)
{
    size_t B = X_batch.size();
    std::vector<LSTMStepCache> cache;

    Matrix h_final = lstm_.forward(X_batch, embedding_, cache);
    Matrix probs   = dense_.forward(h_final, B);
    float  loss    = dense_.computeLoss(probs, y_batch);

    // backward: dense → lstm (which also updates embedding)
    Matrix dh = dense_.backward(probs, y_batch, h_final, lr);
    lstm_.backward(dh, cache, X_batch, embedding_, lr);

    return loss;
}

float LSTMModel::trainEpoch(const std::vector<std::vector<int>>& X_train,
                             const std::vector<int>& y_train,
                             float lr)
{
    size_t N  = X_train.size();
    size_t BS = cfg_.batch_size;
    size_t nb = N / BS;
    float  total_loss = 0.f;

    for (size_t b = 0; b < nb; ++b) {
        auto Xb = DataLoader::slice(X_train, b * BS, (b + 1) * BS);
        auto yb = DataLoader::slice(y_train, b * BS, (b + 1) * BS);
        total_loss += trainBatch(Xb, yb, lr);
    }
    return total_loss / nb;
}

std::vector<int> LSTMModel::predict(const std::vector<std::vector<int>>& X) {
    size_t B = X.size();
    std::vector<LSTMStepCache> cache;
    Matrix h      = lstm_.forward(X, embedding_, cache);
    Matrix probs  = dense_.forward(h, B);
    std::vector<int> preds(B);
    for (size_t b = 0; b < B; ++b) {
        int mx = 0;
        for (size_t i = 1; i < cfg_.output_size; ++i)
            if (probs(i, b) > probs(mx, b)) mx = i;
        preds[b] = mx;
    }
    return preds;
}

float LSTMModel::evaluate(const std::vector<std::vector<int>>& X,
                           const std::vector<int>& y)
{
    auto preds = predict(X);
    int correct = 0;
    for (size_t i = 0; i < y.size(); ++i)
        if (preds[i] == y[i]) ++correct;
    return static_cast<float>(correct) / y.size();
}

void LSTMModel::showPredictions(const std::vector<std::vector<int>>& X,
                                 const std::vector<int>& y,
                                 const std::string& title,
                                 size_t n) const
{
    // const-cast to allow calling predict (non-mutating but no const overload)
    LSTMModel* self = const_cast<LSTMModel*>(this);

    size_t show = std::min(n, X.size());
    auto   Xn   = DataLoader::slice(X, 0, show);
    auto   preds = self->predict(Xn);

    std::vector<LSTMStepCache> cache;
    Matrix h     = lstm_.forward(Xn, embedding_, cache);
    Matrix probs = dense_.forward(h, show);

    std::cout << "\n" << std::string(64, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(64, '=') << "\n";
    for (size_t i = 0; i < show; ++i) {
        int p    = preds[i];
        float cf = probs(p, i);
        const char* mark = (p == y[i]) ? "OK" : "XX";
        std::cout << "  Seq " << std::setw(2) << (i + 1)
                  << ": Pred: " << std::setw(9) << std::left << LABELS[p]
                  << " [" << mark << "]"
                  << "  Conf: " << std::fixed << std::setprecision(4) << cf
                  << "  (GT: " << LABELS[y[i]] << ")\n";
    }
}