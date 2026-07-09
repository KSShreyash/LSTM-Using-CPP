#pragma once
#include "Config.h"
#include "EmbeddingLayer.h"
#include "LSTMLayer.h"
#include "DenseLayer.h"
#include <vector>
#include <string>

/**
 * Top-level model that wires together:
 *   EmbeddingLayer → LSTMLayer → DenseLayer
 *
 * Provides train(), evaluate(), predict(), and weight I/O.
 */
class LSTMModel {
public:
    explicit LSTMModel(const Config& cfg);

    // Weight I/O
    void loadWeights(const std::string& dir, const std::string& tag = "init");
    void saveWeights(const std::string& dir, const std::string& tag) const;

    /**
     * One training step on a single batch.
     * @returns cross-entropy loss for this batch
     */
    float trainBatch(const std::vector<std::vector<int>>& X_batch,
                     const std::vector<int>& y_batch,
                     float lr);

    /**
     * Full epoch: iterate over all batches sequentially (no shuffle).
     * @returns average loss over the epoch
     */
    float trainEpoch(const std::vector<std::vector<int>>& X_train,
                     const std::vector<int>& y_train,
                     float lr);

    /**
     * Predict class labels for all samples.
     */
    std::vector<int> predict(const std::vector<std::vector<int>>& X);

    /**
     * Accuracy on a labeled dataset.
     */
    float evaluate(const std::vector<std::vector<int>>& X,
                   const std::vector<int>& y);

    /**
     * Print first n predictions with ground-truth labels.
     */
    void showPredictions(const std::vector<std::vector<int>>& X,
                         const std::vector<int>& y,
                         const std::string& title,
                         size_t n = 5) const;

private:
    Config          cfg_;
    EmbeddingLayer  embedding_;
    LSTMLayer       lstm_;
    DenseLayer      dense_;
};