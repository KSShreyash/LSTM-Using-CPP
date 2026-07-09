#pragma once
#include <cstddef>

struct Config {
    size_t vocab_size    = 100;
    size_t seq_length    = 10;
    size_t embedding_dim = 32;
    size_t hidden_size   = 16;
    size_t output_size   = 3;
    size_t num_train     = 1200;
    size_t num_test      = 300;
    size_t batch_size    = 32;
    float  learning_rate = 0.05f;
    int    epochs        = 30;

    size_t n_trunc() const {
        return (num_train / batch_size) * batch_size; // 1184
    }
};

// Global singleton — all layers read from this
inline Config& globalConfig() {
    static Config cfg;
    return cfg;
}