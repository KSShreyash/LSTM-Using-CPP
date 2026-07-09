#pragma once
#include <vector>
#include <string>
#include <cstddef>

/**
 * Loads flat binary files (int32 / float32) produced by train_tensorflow.py.
 */
class DataLoader {
public:
    /**
     * Load token ids from a flat binary file and reshape to [N][T].
     * @param path   path to *.bin file
     * @param N      number of sequences
     * @param T      sequence length
     */
    static std::vector<std::vector<int>> loadX(const std::string& path,size_t N, size_t T);

    /**
     * Load integer labels from a flat binary file.
     */
    static std::vector<int> loadY(const std::string& path, size_t N);

    /**
     * Slice [begin, end) from a 2D array.
     */
    static std::vector<std::vector<int>> slice(
        const std::vector<std::vector<int>>& X, size_t begin, size_t end);

    static std::vector<int> slice(const std::vector<int>& y,size_t begin, size_t end);
};