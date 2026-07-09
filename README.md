# LSTM Sentiment Classifier - Modular Class-Based C++ Implementation

A complete, **modular, class-based LSTM** implemented in **pure C++17**
with zero ML library dependencies. Trains a 3-class sentiment classifier
(Negative / Neutral / Positive) entirely from scratch using hand-written
backpropagation through time (BPTT) and SGD.

Includes a TensorFlow reference trainer and a weight-comparison script
to verify that the C++ and TensorFlow implementations produce numerically
identical results.

---

## How This Differs from the Trainable Model

| Feature     | Trainable Model (original)                     | Modular Class-Based Model (this)          |
| ----------- | ---------------------------------------------- | ----------------------------------------- |
| Code style  | Procedural - global variables + free functions | Object-oriented - one class per component |
| Weights     | Global `Mat` structs                           | Encapsulated inside each layer class      |
| Reusability | Single-file monolithic                         | Each class can be reused independently    |
| Testability | Hard to unit test                              | Each class can be tested in isolation     |
| Readability | ~500 lines, one file                           | Split across 9 focused files              |
| Correctness | Verified vs TF                                 | Verified vs TF (same numerical output)    |

---

## Architecture

```text
Input tokens
     │
     ▼
┌─────────────────┐
│  EmbeddingLayer │   vocab_size × embedding_dim  lookup table
└────────┬────────┘
         │  x_t  [B × E]
         ▼
┌─────────────────┐
│   LSTMLayer     │   unrolled over T time steps
│   (uses         │   f, i, g, o gates  →  h_t, c_t
│   LSTMCell)     │   BPTT on backward pass
└────────┬────────┘
         │  h_T  [H × B]
         ▼
┌─────────────────┐
│   DenseLayer    │   linear + column-wise softmax
└────────┬────────┘
         │
         ▼
   probs [OUT × B]   → cross-entropy loss → BPTT → SGD
```

---

## Class Responsibilities

| Class            | File                                                  | What it does                                          |
| ---------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| `Config`         | `include/Config.h`                                    | Hyperparameters - single source of truth              |
| `Matrix`         | `include/Matrix.h` / `src/Matrix.cpp`                 | Row-major matrix: matmul, transpose, axpy, binary I/O |
| `Activations`    | `include/Activations.h` / `src/Activations.cpp`       | sigmoid, tanh, softmax - stateless free functions     |
| `EmbeddingLayer` | `include/EmbeddingLayer.h` / `src/EmbeddingLayer.cpp` | Token → float vector lookup + gradient accumulation   |
| `LSTMCell`       | `include/LSTMCell.h` / `src/LSTMCell.cpp`             | Single time-step forward + backward (stateless)       |
| `LSTMLayer`      | `include/LSTMLayer.h` / `src/LSTMLayer.cpp`           | Full sequence unrolling + BPTT, owns gate weights     |
| `DenseLayer`     | `include/DenseLayer.h` / `src/DenseLayer.cpp`         | Linear → softmax + cross-entropy gradient             |
| `DataLoader`     | `include/DataLoader.h` / `src/DataLoader.cpp`         | Binary file I/O for int32 and float32 arrays          |
| `LSTMModel`      | `include/LSTMModel.h` / `src/LSTMModel.cpp`           | Top-level API: train, evaluate, predict, weight I/O   |
| `main.cpp`       | `src/main.cpp`                                        | Entry point - wires everything together               |

---

## Project Structure

```text
Modular Class Based Model/
├── cpp/
│   ├── include/
│   │   ├── Config.h
│   │   ├── Matrix.h
│   │   ├── Activations.h
│   │   ├── EmbeddingLayer.h
│   │   ├── LSTMCell.h
│   │   ├── LSTMLayer.h
│   │   ├── DenseLayer.h
│   │   ├── DataLoader.h
│   │   └── LSTMModel.h
│   ├── src/
│   │   ├── Matrix.cpp
│   │   ├── Activations.cpp
│   │   ├── EmbeddingLayer.cpp
│   │   ├── LSTMCell.cpp
│   │   ├── LSTMLayer.cpp
│   │   ├── DenseLayer.cpp
│   │   ├── DataLoader.cpp
│   │   ├── LSTMModel.cpp
│   │   └── main.cpp
│   └── Makefile
├── python/
│   ├── train_tensorflow.py    ← generates data + weights (one-time)
│   └── compare_weights.py     ← verifies C++ matches TensorFlow
├── lstm_data/                 ← auto-created by Python script
├── lstm_weights/              ← auto-created by Python script
└── README.md                  ← this file
```

---

## Prerequisites

### System Packages

```bash
sudo apt update
sudo apt install -y g++ make python3 python3-pip
```

Check compiler version — needs **g++ ≥ 9** for C++17:

```bash
g++ --version
```

### Python Packages (one-time only)

```bash
pip3 install tensorflow numpy matplotlib
```

---

## How to Run (Ubuntu)

### Step 1 — Generate Data and Initial Weights *(one-time)*

Run from the `Modular Class Based Model/` root directory:

```bash
python3 python/train_tensorflow.py
```

**What this creates:**

| File                            | Description                                       |
| ------------------------------- | ------------------------------------------------- |
| `lstm_data/X_train.bin`         | Training token sequences (int32)                  |
| `lstm_data/y_train.bin`         | Training labels (int32)                           |
| `lstm_data/X_test.bin`          | Test token sequences (int32)                      |
| `lstm_data/y_test.bin`          | Test labels (int32)                               |
| `lstm_weights/*_init.bin`       | Shared starting weights (used by both TF and C++) |
| `lstm_weights/*_trained_tf.bin` | TensorFlow trained weights (for comparison)       |
| `training_curve.png`            | Loss and accuracy plot                            |

> **After this step Python is no longer needed.**
> All training and inference runs entirely in C++.

---

### Step 2 — Build the C++ Project

```bash
cd cpp/
make
```

Expected output:

```text
mkdir -p build
g++ -std=c++17 -O2 -Wall -Iinclude -c src/Matrix.cpp      -o build/Matrix.o
g++ -std=c++17 -O2 -Wall -Iinclude -c src/Activations.cpp -o build/Activations.o
g++ -std=c++17 -O2 -Wall -Iinclude -c src/EmbeddingLayer.cpp -o build/EmbeddingLayer.o
g++ -std=c++17 -O2 -Wall -Iinclude -c src/LSTMCell.cpp    -o build/LSTMCell.o
g++ -std=c++17 -O2 -Wall -Iinclude -c src/LSTMLayer.cpp   -o build/LSTMLayer.o
g++ -std=c++17 -O2 -Wall -Iinclude -c src/DenseLayer.cpp  -o build/DenseLayer.o
g++ -std=c++17 -O2 -Wall -Iinclude -c src/DataLoader.cpp  -o build/DataLoader.o
g++ -std=c++17 -O2 -Wall -Iinclude -c src/LSTMModel.cpp   -o build/LSTMModel.o
g++ -std=c++17 -O2 -Wall -Iinclude -c src/main.cpp        -o build/main.o
g++ -std=c++17 -O2 -Wall -Iinclude -o lstm_train build/*.o
```

To clean build artifacts:

```bash
make clean
```

---

### Step 3 — Run the C++ Trainer

> **Important:** Run from the `Modular Class Based Model/` root, NOT from inside `cpp/`.
> This ensures the relative paths `lstm_data/` and `lstm_weights/` resolve correctly.

```bash
cd ..          # back to Modular Class Based Model/ root
./cpp/lstm_train
```

**Expected output:**

```text

  C++ LSTM SENTIMENT TRAINER  (Modular Class-Based)


Config:
  vocab=100  seq_len=10  emb_dim=32  hidden=16  output=3
  train_trunc=1184  test=300  batch=32  lr=0.05  epochs=30

[LSTMModel] Loading weights (init) from: lstm_weights


  INITIAL PREDICTIONS  (before training)

  Seq  1: Pred: NEUTRAL   [XX]  Conf: 0.3521  (GT: NEGATIVE)
  ...

  Epoch  1/30 | Loss: 1.0821 | Train: 0.4000 | Test: 0.3867 | 2.3s
  Epoch  2/30 | Loss: 0.9744 | Train: 0.5500 | Test: 0.5200 | 2.1s
  ...
  Epoch 30/30 | Loss: 0.2103 | Train: 0.9500 | Test: 0.9300 | 2.2s


  FINAL PREDICTIONS  (after training)

  Seq  1: Pred: NEGATIVE  [OK]  Conf: 0.9123  (GT: NEGATIVE)
  ...

[LSTMModel] All weights saved.
```

---

### Step 4 — Compare C++ vs TensorFlow *(optional)*

```bash
python3 python/compare_weights.py
```

A successful run shows:

```text
  Weight         Shape         MAE         MAX     Match?
  ──────────     ────────────  ──────────  ──────────  ──────
  embedding      (100, 32)    0.000012   0.000098   ✅
  w_f            (16, 48)     0.000008   0.000071   ✅
  w_i            (16, 48)     0.000006   0.000055   ✅
  w_c            (16, 48)     0.000009   0.000082   ✅
  w_o            (16, 48)     0.000007   0.000063   ✅
  b_f            (16, 1)      0.000003   0.000021   ✅
  b_i            (16, 1)      0.000004   0.000029   ✅
  b_c            (16, 1)      0.000005   0.000038   ✅
  b_o            (16, 1)      0.000003   0.000024   ✅
  w_y            (3, 16)      0.000011   0.000089   ✅
  b_y            (3, 1)       0.000002   0.000015   ✅

  TensorFlow : 93.0%   C++ : 93.0%   Δ = 0.00%
  C++ is a verified drop-in replacement for TensorFlow LSTM.
```

---

## All Steps — Quick Reference

```bash
# From Modular Class Based Model/ root

# Step 1: Generate data and TF weights (one-time)
python3 python/train_tensorflow.py

# Step 2: Build
cd cpp && make && cd ..

# Step 3: Train in C++
./cpp/lstm_train

# Step 4: Verify C++ matches TensorFlow
python3 python/compare_weights.py
```

---

## Model Configuration

All hyperparameters are in `cpp/include/Config.h`:

```cpp
struct Config {
    size_t vocab_size    = 100;   // number of unique tokens
    size_t seq_length    = 10;    // tokens per sequence
    size_t embedding_dim = 32;    // embedding vector size
    size_t hidden_size   = 16;    // LSTM hidden units
    size_t output_size   = 3;     // classes: NEG / NEU / POS
    size_t num_train     = 1200;  // training samples
    size_t num_test      = 300;   // test samples
    size_t batch_size    = 32;    // mini-batch size
    float  learning_rate = 0.05f; // SGD learning rate
    int    epochs        = 30;    // training epochs
};
```

Modify values here and recompile with `make clean && make`.

---

## LSTM Equations (implemented in LSTMCell)

```text
Concatenate:   xh  = [ h_{t-1} ; x_t ]

Forget gate:   f_t = σ( Wf · xh + bf )
Input  gate:   i_t = σ( Wi · xh + bi )
Cell   gate:   g_t = tanh( Wc · xh + bc )
Output gate:   o_t = σ( Wo · xh + bo )

Cell state:    c_t = f_t ⊙ c_{t-1}  +  i_t ⊙ g_t
Hidden state:  h_t = o_t ⊙ tanh( c_t )

Output:        logits = Wy · h_T + by
               probs  = softmax( logits )
               loss   = mean cross-entropy
```

Gradients are averaged by batch size B at the output layer and this
scale propagates through the full BPTT chain automatically.

---

## Dependencies

| Component                   | Dependencies                         |
| --------------------------- | ------------------------------------ |
| C++ trainer (train + infer) | **None** — C++17 stdlib only         |
| Data + weight generation    | Python 3, NumPy                      |
| TF reference trainer        | Python 3, TensorFlow 2.x, Matplotlib |
| Weight comparison           | Python 3, NumPy                      |

---


