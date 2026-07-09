"""
compare_weights.py
──────────────────────────────────────────────────────────────────────────────
Run on your LOCAL machine AFTER:
  1. Colab produced:  lstm_weights/*_trained_tf.bin  (inside lstm_for_cpp.zip)
  2. C++ produced:    lstm_weights/*_trained_cpp.bin  (via ./lstm_train)

Shows:
  • TensorFlow vs C++ weight differences
  • Side-by-side accuracy and sample predictions
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import os, sys

SEQ_LENGTH    = 10
NUM_TEST      = 300
VOCAB_SIZE    = 100
EMBEDDING_DIM = 32
HIDDEN_SIZE   = 16
OUTPUT_SIZE   = 3
H, E          = HIDDEN_SIZE, EMBEDDING_DIM
LABELS        = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

WEIGHT_SHAPES = [
    ("embedding", (VOCAB_SIZE, EMBEDDING_DIM)),
    ("w_f", (H, H+E)), ("w_i", (H, H+E)),
    ("w_c", (H, H+E)), ("w_o", (H, H+E)),
    ("b_f", (H, 1)),   ("b_i", (H, 1)),
    ("b_c", (H, 1)),   ("b_o", (H, 1)),
    ("w_y", (OUTPUT_SIZE, H)),
    ("b_y", (OUTPUT_SIZE, 1)),
]

def load(name, tag):
    p = f"lstm_weights/{name}_trained_{tag}.bin"
    if not os.path.exists(p): return None
    shape = next(s for n, s in WEIGHT_SHAPES if n == name)
    return np.fromfile(p, dtype=np.float32).reshape(shape)

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))
def softmax_cols(x):
    e = np.exp(x - x.max(0, keepdims=True))
    return e / e.sum(0, keepdims=True)

def run_inference(tag):
    e  = np.fromfile(f"lstm_weights/embedding_trained_{tag}.bin", np.float32).reshape(VOCAB_SIZE, E)
    wf = np.fromfile(f"lstm_weights/w_f_trained_{tag}.bin", np.float32).reshape(H, H+E)
    wi = np.fromfile(f"lstm_weights/w_i_trained_{tag}.bin", np.float32).reshape(H, H+E)
    wc = np.fromfile(f"lstm_weights/w_c_trained_{tag}.bin", np.float32).reshape(H, H+E)
    wo = np.fromfile(f"lstm_weights/w_o_trained_{tag}.bin", np.float32).reshape(H, H+E)
    bf = np.fromfile(f"lstm_weights/b_f_trained_{tag}.bin", np.float32).reshape(H, 1)
    bi = np.fromfile(f"lstm_weights/b_i_trained_{tag}.bin", np.float32).reshape(H, 1)
    bc = np.fromfile(f"lstm_weights/b_c_trained_{tag}.bin", np.float32).reshape(H, 1)
    bo = np.fromfile(f"lstm_weights/b_o_trained_{tag}.bin", np.float32).reshape(H, 1)
    wy = np.fromfile(f"lstm_weights/w_y_trained_{tag}.bin", np.float32).reshape(OUTPUT_SIZE, H)
    by = np.fromfile(f"lstm_weights/b_y_trained_{tag}.bin", np.float32).reshape(OUTPUT_SIZE, 1)
    B = len(X_test)
    h = np.zeros((H, B), np.float32); c = np.zeros((H, B), np.float32)
    for t in range(SEQ_LENGTH):
        xt = e[X_test[:, t]].T
        xh = np.vstack([h, xt])
        ft = sigmoid(wf @ xh + bf); it = sigmoid(wi @ xh + bi)
        ct = np.tanh(wc @ xh + bc); ot = sigmoid(wo @ xh + bo)
        c  = ft * c + it * ct;      h  = ot * np.tanh(c)
    probs = softmax_cols(wy @ h + by)
    preds = probs.argmax(0)
    return preds, probs, (preds == y_test).mean()

X_test = np.fromfile("lstm_data/X_test.bin", np.int32).reshape(NUM_TEST, SEQ_LENGTH)
y_test = np.fromfile("lstm_data/y_test.bin", np.int32)

has_tf  = os.path.exists("lstm_weights/w_f_trained_tf.bin")
has_cpp = os.path.exists("lstm_weights/w_f_trained_cpp.bin")

print("=" * 66)
print("  WEIGHT COMPARISON: TensorFlow (Colab) vs C++ (local)")
print("=" * 66)
print(f"  TensorFlow weights (from Colab) : {'✅ found' if has_tf  else '❌ missing — run Colab first'}")
print(f"  C++ weights (local)             : {'✅ found' if has_cpp else '❌ missing — run ./lstm_train'}")

if not has_tf:
    print("\n  ⛔  Run the Colab notebook first, download lstm_for_cpp.zip,")
    print("      extract it here, then run ./lstm_train before this script.")
    sys.exit(1)
if not has_cpp:
    print("\n  ⛔  Run C++ trainer:  make && ./lstm_train")
    sys.exit(1)

print(f"\n  {'─'*66}")
print("  Weight-by-weight differences  (TensorFlow vs C++)")
print(f"  {'─'*66}")
print(f"  {'Weight':14s}  {'Shape':12s}  {'MAE':>10s}  {'MAX':>10s}  Match?")
print(f"  {'─'*14}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*6}")

all_match = True
for name, shape in WEIGHT_SHAPES:
    w_tf = load(name,"tf"); w_cpp = load(name,"cpp")
    if w_tf is None or w_cpp is None:
        print(f"  {name:14s}  missing"); continue
    mae = float(np.abs(w_tf - w_cpp).mean())
    mx  = float(np.abs(w_tf - w_cpp).max())
    ok  = "✅" if mx < 0.05 else "⚠️ "
    if mx >= 0.05: all_match = False
    print(f"  {name:14s}  {str(shape):12s}  {mae:10.6f}  {mx:10.6f}  {ok}")

print(f"  {'─'*66}")
msg = "✅  All weights match." if all_match else "⚠️   Some weights differ."
print(f"  {msg}")

print(f"\n  {'─'*66}")
print("  Final test accuracy")
print(f"  {'─'*66}")
results = {}
for tag, label in [("tf","TensorFlow (Colab)"), ("cpp","C++ (local)")]:
    preds, probs, acc = run_inference(tag)
    results[tag] = (preds, probs, acc)
    print(f"  {label:22s}  {acc:.4f}  ({acc*100:.1f}%)")

print(f"\n  {'─'*66}")
print("  Sample predictions — first 5 test sequences")
print(f"  {'─'*66}")
print(f"\n  {'Seq':>4}  {'Ground Truth':<10}  {'TensorFlow':<11} {'Conf':>8}  {'C++':<11} {'Conf':>8}")
print(f"  {'─'*4}  {'─'*10}  {'─'*11} {'─'*8}  {'─'*11} {'─'*8}")
preds_tf,  probs_tf,  _ = results["tf"]
preds_cpp, probs_cpp, _ = results["cpp"]
for i in range(5):
    gt    = LABELS[y_test[i]]
    p_tf  = preds_tf[i];  mk_tf  = "✅" if p_tf  == y_test[i] else "❌"
    p_cpp = preds_cpp[i]; mk_cpp = "✅" if p_cpp == y_test[i] else "❌"
    print(f"  {i+1:>4}  {gt:<10}  "
          f"{LABELS[p_tf]:<9}{mk_tf}  {probs_tf[p_tf,i]:8.4f}  "
          f"{LABELS[p_cpp]:<9}{mk_cpp}  {probs_cpp[p_cpp,i]:8.4f}")

acc_tf  = results["tf"][2];   acc_cpp = results["cpp"][2]
diff    = abs(acc_tf - acc_cpp)
print(f"\n{'='*66}")
print("  VERDICT")
print(f"{'='*66}")
print(f"  TensorFlow : {acc_tf*100:.1f}%   C++ : {acc_cpp*100:.1f}%   Δ = {diff*100:.2f}%")
if diff < 0.02:
    print("  ✅  C++ is a verified drop-in replacement for TensorFlow LSTM.")
    print("      Zero Python or TF dependency needed for training or inference.")
else:
    print("  ⚠️   Accuracy gap > 2% — check both used the same init weights.")
print(f"{'='*66}")
