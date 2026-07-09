"""
train_tensorflow.py
─────────────────────────────────────────────────────────────────────────────
Local replacement for lstm_sentiment_colab.ipynb
Run inside the lstm_sentiment/ project folder.

Usage:
    python3 python/train_tensorflow.py
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import os, zipfile, time
import matplotlib
matplotlib.use('Agg')          # no display needed — saves plot to file
import matplotlib.pyplot as plt

# ── suppress TF info/warning logs ────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL']    = '2'
# ── CRITICAL: disable oneDNN fused LSTM kernel ───────────────────────────────
# oneDNN reorders floating-point operations which produces different gradients
# than our manual C++ BPTT. Disabling it makes TF use standard ops that
# match our C++ computation order → weights match to float32 precision.
os.environ['TF_ENABLE_ONEDNN_OPTS']   = '0'
os.environ['TF_DETERMINISTIC_OPS']    = '1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Setup
# ═════════════════════════════════════════════════════════════════════════════
os.makedirs('lstm_data',    exist_ok=True)
os.makedirs('lstm_weights', exist_ok=True)
print('✅  Folders ready')
print(f'    TensorFlow {tf.__version__}')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Configuration  (must match C++ Config struct exactly)
# ═════════════════════════════════════════════════════════════════════════════
VOCAB_SIZE    = 100
SEQ_LENGTH    = 10
EMBEDDING_DIM = 32
HIDDEN_SIZE   = 16
OUTPUT_SIZE   = 3        # 0=NEGATIVE  1=NEUTRAL  2=POSITIVE
NUM_TRAIN     = 1200
NUM_TEST      = 300
LEARNING_RATE = 0.05
EPOCHS        = 30
BATCH_SIZE    = 32
SEED          = 42
LABELS        = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']

np.random.seed(SEED)
tf.random.set_seed(SEED)
print('✅  Config set')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Generate synthetic sentiment dataset
# ═════════════════════════════════════════════════════════════════════════════
# Word-id ranges carry sentiment signal:
#   0–29  → negative words
#  30–59  → neutral  words
#  60–99  → positive words
# Each sequence: 80% words from its class + 20% random noise

NEG_WORDS = list(range(0,  30))
NEU_WORDS = list(range(30, 60))
POS_WORDS = list(range(60, 100))

def make_sequence(label, noise=0.2):
    pools = [NEG_WORDS, NEU_WORDS, POS_WORDS]
    primary = pools[label]
    seq = []
    for _ in range(SEQ_LENGTH):
        if np.random.rand() < noise:
            seq.append(np.random.randint(0, VOCAB_SIZE))
        else:
            seq.append(np.random.choice(primary))
    return seq

def make_dataset(n):
    X, y = [], []
    for i in range(n):
        label = i % 3
        X.append(make_sequence(label))
        y.append(label)
    idx = np.random.permutation(n)
    return np.array(X, dtype=np.int32)[idx], np.array(y, dtype=np.int32)[idx]

X_train, y_train = make_dataset(NUM_TRAIN)
X_test,  y_test  = make_dataset(NUM_TEST)

X_train.tofile('lstm_data/X_train.bin')
y_train.tofile('lstm_data/y_train.bin')
X_test.tofile( 'lstm_data/X_test.bin')
y_test.tofile( 'lstm_data/y_test.bin')

print(f'✅  Dataset saved')
print(f'    X_train={X_train.shape}  y_train={y_train.shape}')
print(f'    X_test ={X_test.shape}   y_test ={y_test.shape}')
print(f'    Labels (train): { {LABELS[i]: int((y_train==i).sum()) for i in range(3)} }')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Generate & save initial weights  (shared by TF and C++)
# ═════════════════════════════════════════════════════════════════════════════
def xavier(shape):
    fan_in  = shape[1] if len(shape) > 1 else shape[0]
    fan_out = shape[0]
    limit   = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape).astype(np.float32)

combined = HIDDEN_SIZE + EMBEDDING_DIM

init_weights = {
    'embedding': xavier((VOCAB_SIZE,    EMBEDDING_DIM)),
    'w_f':       xavier((HIDDEN_SIZE,   combined)),
    'w_i':       xavier((HIDDEN_SIZE,   combined)),
    'w_c':       xavier((HIDDEN_SIZE,   combined)),
    'w_o':       xavier((HIDDEN_SIZE,   combined)),
    'b_f':       np.zeros((HIDDEN_SIZE, 1), dtype=np.float32),
    'b_i':       np.zeros((HIDDEN_SIZE, 1), dtype=np.float32),
    'b_c':       np.zeros((HIDDEN_SIZE, 1), dtype=np.float32),
    'b_o':       np.zeros((HIDDEN_SIZE, 1), dtype=np.float32),
    'w_y':       xavier((OUTPUT_SIZE,   HIDDEN_SIZE)),
    'b_y':       np.zeros((OUTPUT_SIZE, 1), dtype=np.float32),
}

for name, w in init_weights.items():
    w.tofile(f'lstm_weights/{name}_init.bin')

print('✅  Initial weights saved to lstm_weights/')
for name, w in init_weights.items():
    print(f'    {name}_init.bin  {w.shape}')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Build TensorFlow model & load initial weights
# ═════════════════════════════════════════════════════════════════════════════
# TF LSTM gate order: i, f, c, o   (our C++ order: f, i, c, o) — must reorder

inputs  = keras.Input(shape=(SEQ_LENGTH,), dtype='int32')
x       = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, name='embedding')(inputs)
# implementation=1: column-wise matrix ops — closest to manual C++ BPTT
# implementation=2 (default): uses batch matmul with different op order → mismatch
x       = layers.LSTM(HIDDEN_SIZE, name='lstm', implementation=1)(x)
outputs = layers.Dense(OUTPUT_SIZE, activation='softmax', name='dense')(x)
model   = keras.Model(inputs, outputs)
model.summary()

def load_init(name, shape):
    return np.fromfile(f'lstm_weights/{name}_init.bin', np.float32).reshape(shape)

H, E = HIDDEN_SIZE, EMBEDDING_DIM

w_f = load_init('w_f', (H, H+E));  w_i = load_init('w_i', (H, H+E))
w_c = load_init('w_c', (H, H+E));  w_o = load_init('w_o', (H, H+E))
b_f = load_init('b_f', (H, 1)).flatten()
b_i = load_init('b_i', (H, 1)).flatten()
b_c = load_init('b_c', (H, 1)).flatten()
b_o = load_init('b_o', (H, 1)).flatten()

def split_rec_inp(W):
    return W[:, :H], W[:, H:]    # (H,H), (H,E)

rec_f, inp_f = split_rec_inp(w_f)
rec_i, inp_i = split_rec_inp(w_i)
rec_c, inp_c = split_rec_inp(w_c)
rec_o, inp_o = split_rec_inp(w_o)

# TF gate order: i, f, c, o
kernel     = np.concatenate([inp_i, inp_f, inp_c, inp_o], axis=0).T   # (E, 4H)
rec_kernel = np.concatenate([rec_i, rec_f, rec_c, rec_o], axis=0).T   # (H, 4H)
bias       = np.concatenate([b_i,   b_f,   b_c,   b_o])               # (4H,)

model.get_layer('embedding').set_weights([load_init('embedding', (VOCAB_SIZE, E))])
model.get_layer('lstm').set_weights([kernel.astype(np.float32),
                                     rec_kernel.astype(np.float32),
                                     bias.astype(np.float32)])
wy = load_init('w_y', (OUTPUT_SIZE, H)).T
by = load_init('b_y', (OUTPUT_SIZE, 1)).flatten()
model.get_layer('dense').set_weights([wy, by])

print('\n✅  Initial weights loaded into TF model')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Predictions BEFORE training
# ═════════════════════════════════════════════════════════════════════════════
def show_predictions(X_sample, y_sample, title, n=5):
    probs = model.predict(X_sample[:n], verbose=0)
    preds = probs.argmax(axis=1)
    print(f'\n{"="*64}')
    print(f'  {title}')
    print(f'{"="*64}')
    for i in range(n):
        p, gt = preds[i], y_sample[i]
        mark = '✅' if p == gt else '❌'
        print(f'  Sequence {i+1}: Prediction: {LABELS[p]:<8}  {mark}  '
              f'Confidence: {probs[i,p]:.4f}   (GT: {LABELS[gt]})')

show_predictions(X_test, y_test, 'INITIAL PREDICTIONS  (before training)')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — Train
# ═════════════════════════════════════════════════════════════════════════════
# KEY: truncate to exact multiple of BATCH_SIZE so both TF and C++ run
# exactly 37 batches per epoch — prevents gradient count mismatch.
# shuffle=False so both see batches in identical sequential order.

N_TRUNC = (NUM_TRAIN // BATCH_SIZE) * BATCH_SIZE   # 1184
X_tr = X_train[:N_TRUNC]
y_tr = y_train[:N_TRUNC]

print(f'\nTraining samples truncated: {NUM_TRAIN} → {N_TRUNC}  ({N_TRUNC // BATCH_SIZE} full batches of {BATCH_SIZE})')

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

sep = '─' * 64
print(sep)
print('TRAINING  —  TensorFlow / Keras LSTM')
print(f'  lr={LEARNING_RATE}   epochs={EPOCHS}   batch={BATCH_SIZE}')
print(f'  samples={N_TRUNC} ({N_TRUNC//BATCH_SIZE} batches)   shuffle=False')
print(sep)
print()

t0 = time.time()
history = model.fit(
    X_tr, y_tr,
    batch_size      = BATCH_SIZE,
    epochs          = EPOCHS,
    validation_data = (X_test, y_test),
    shuffle         = False,
    verbose         = 1
)
print(f'\nTotal training time: {time.time()-t0:.1f}s')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — Predictions AFTER training
# ═════════════════════════════════════════════════════════════════════════════
show_predictions(X_test, y_test, 'FINAL PREDICTIONS  (after TensorFlow training)')

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f'\nTest Loss: {loss:.4f}   Test Accuracy: {acc:.4f}')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 9 — Save trained weights in C++-compatible binary format
# ═════════════════════════════════════════════════════════════════════════════
def save_w(name, arr):
    arr.astype(np.float32).tofile(f'lstm_weights/{name}_trained_tf.bin')

save_w('embedding', model.get_layer('embedding').get_weights()[0])

k, rk, bs = model.get_layer('lstm').get_weights()   # (E,4H), (H,4H), (4H,)

# TF order i,f,c,o → our order f,i,c,o
def unpack_tf(mat, axis):
    parts = np.split(mat, 4, axis=axis)
    return parts[1], parts[0], parts[2], parts[3]

k_f,k_i,k_c,k_o     = unpack_tf(k,  axis=1)
rk_f,rk_i,rk_c,rk_o = unpack_tf(rk, axis=1)
b_parts = [bs[i*H:(i+1)*H] for i in range(4)]
b_f_t, b_i_t, b_c_t, b_o_t = b_parts[1], b_parts[0], b_parts[2], b_parts[3]

for tag, kr, ki in [('w_f',rk_f,k_f), ('w_i',rk_i,k_i),
                    ('w_c',rk_c,k_c), ('w_o',rk_o,k_o)]:
    W = np.concatenate([kr.T, ki.T], axis=1)   # (H, H+E)
    save_w(tag, W)

save_w('b_f', b_f_t.reshape(H,1));  save_w('b_i', b_i_t.reshape(H,1))
save_w('b_c', b_c_t.reshape(H,1));  save_w('b_o', b_o_t.reshape(H,1))

wy_out, by_out = model.get_layer('dense').get_weights()
save_w('w_y', wy_out.T)
save_w('b_y', by_out.reshape(OUTPUT_SIZE, 1))

print('\n✅  TF trained weights saved to lstm_weights/*_trained_tf.bin')
for f in sorted(os.listdir('lstm_weights')):
    if 'trained_tf' in f:
        print(f'    {f}')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 10 — Training curve plot
# ═════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['loss'],     label='Train', linewidth=2, color='steelblue')
ax1.plot(history.history['val_loss'], label='Test',  linewidth=2, color='tomato')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Cross-Entropy Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history.history['accuracy'],     label='Train', linewidth=2, color='steelblue')
ax2.plot(history.history['val_accuracy'], label='Test',  linewidth=2, color='tomato')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
ax2.set_title('Classification Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.3, 1.0])

plt.suptitle('TensorFlow LSTM — Sentiment Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('training_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅  Plot saved → training_curve.png')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 11 — Package everything into ZIP
# ═════════════════════════════════════════════════════════════════════════════
zip_path = 'lstm_for_cpp.zip'

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in ['X_train.bin','y_train.bin','X_test.bin','y_test.bin']:
        zf.write(f'lstm_data/{fname}', f'lstm_data/{fname}')
    for fname in os.listdir('lstm_weights'):
        zf.write(f'lstm_weights/{fname}', f'lstm_weights/{fname}')
    if os.path.exists('training_curve.png'):
        zf.write('training_curve.png', 'training_curve.png')

size_mb = os.path.getsize(zip_path) / 1024**2
print(f'\n✅  {zip_path} created ({size_mb:.2f} MB)')
print()
print('=' * 60)
print('  NEXT STEPS:')
print('=' * 60)
print('  1.  make       (compile C++)')
print('  2.  ./lstm_train')
print('  3.  python3 python/compare_weights.py')
print('=' * 60)