/**
 * main.cpp — Modular C++ LSTM Sentiment Trainer
 *
 * Wires together:
 *   DataLoader → LSTMModel (EmbeddingLayer + LSTMLayer + DenseLayer) → train/eval
 *
 * Build:   make
 * Run:     ./lstm_train
 *          (from the lstm_sentiment/ root, after running train_tensorflow.py)
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>

#include "Config.h"
#include "DataLoader.h"
#include "LSTMModel.h"

int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "  C++ LSTM SENTIMENT TRAINER  (Modular Class-Based)\n";
    std::cout << std::string(70, '=') << "\n";

    // ── Configuration ──────────────────────────────────────────────────
    Config cfg;   // uses defaults matching train_tensorflow.py

    size_t TRN = cfg.n_trunc();  // 1184  — exact multiple of batch_size
    size_t TST = cfg.num_test;
    size_t T   = cfg.seq_length;

    std::cout << "\nConfig:\n"
              << "  vocab=" << cfg.vocab_size
              << "  seq_len=" << T
              << "  emb_dim=" << cfg.embedding_dim
              << "  hidden=" << cfg.hidden_size
              << "  output=" << cfg.output_size << "\n"
              << "  train_trunc=" << TRN
              << "  test=" << TST
              << "  batch=" << cfg.batch_size
              << "  lr=" << cfg.learning_rate
              << "  epochs=" << cfg.epochs << "\n";

    // ── Load Data ──────────────────────────────────────────────────────
    std::cout << "\nLoading data...\n";
    auto X_train_full = DataLoader::loadX("lstm_data/X_train.bin", cfg.num_train, T);
    auto y_train_full = DataLoader::loadY("lstm_data/y_train.bin", cfg.num_train);
    auto X_test       = DataLoader::loadX("lstm_data/X_test.bin",  TST, T);
    auto y_test       = DataLoader::loadY("lstm_data/y_test.bin",  TST);

    // Truncate training set — no shuffle (matches TF shuffle=False)
    auto X_train = DataLoader::slice(X_train_full, 0, TRN);
    auto y_train = DataLoader::slice(y_train_full, 0, TRN);
    std::cout << "  X_train=" << TRN << "  X_test=" << TST << "  OK\n";

    // ── Build & Initialise Model ───────────────────────────────────────
    LSTMModel model(cfg);
    model.loadWeights("lstm_weights", "init");

    model.showPredictions(X_test, y_test,
                          "INITIAL PREDICTIONS  (before training)");

    // ── Training Loop ──────────────────────────────────────────────────
    std::cout << "\n" << std::string(64, '-') << "\n";
    std::cout << "  TRAINING  (sequential batches, no shuffle)\n";
    std::cout << "  lr=" << cfg.learning_rate
              << "  epochs=" << cfg.epochs
              << "  batch=" << cfg.batch_size
              << "  samples=" << TRN << "\n";
    std::cout << std::string(64, '-') << "\n\n";

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        auto t0 = std::chrono::high_resolution_clock::now();

        float avg_loss = model.trainEpoch(X_train, y_train, cfg.learning_rate);

        float tr_acc = model.evaluate(X_train, y_train);
        float ts_acc = model.evaluate(X_test,  y_test);

        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration_cast<std::chrono::milliseconds>(
                          t1 - t0).count() / 1000.0;

        std::cout << "  Epoch " << std::setw(2) << (epoch + 1)
                  << "/" << cfg.epochs
                  << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << " | Train: " << tr_acc
                  << " | Test: "  << ts_acc
                  << " | " << secs << "s\n";
    }

    model.showPredictions(X_test, y_test,
                          "FINAL PREDICTIONS  (after training)");

    // ── Save Trained Weights ───────────────────────────────────────────
    model.saveWeights("lstm_weights", "cpp");

    std::cout << "\nC++ training complete.\n";
    std::cout << std::string(70, '=') << "\n";
    return 0;
}