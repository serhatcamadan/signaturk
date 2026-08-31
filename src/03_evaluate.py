#!/usr/bin/env python3
"""
03_evaluate.py
==============
Test seti üzerinde model performansını ölçer.
Train veya Val verisine dokunmaz.
"""

import numpy as np
import json, os
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH   = "models/best_model.keras"
LABEL_MAP    = "models/label_map.json"
KEYPOINT_DIR = "data/keypoints"
MAX_FRAMES   = 60
FEATURE_DIM  = 126

# ─── ÖZEL KATMANI KAYDET ──────────────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable()
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = tf.keras.layers.Dense(units, use_bias=False)
        self.V = tf.keras.layers.Dense(1,     use_bias=False)

    def call(self, hidden_states):
        score   = self.V(tf.nn.tanh(self.W(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(weights * hidden_states, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# ─── YÜKLEME ──────────────────────────────────────────────────────────────────
print("Model yükleniyor...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"BahdanauAttention": BahdanauAttention}
)

with open(LABEL_MAP, "r", encoding="utf-8") as f:
    label_map = json.load(f)

idx_to_label = {int(k): v for k, v in label_map.items()}
label_to_idx = {v: k for k, v in idx_to_label.items()}
labels       = [idx_to_label[i] for i in range(len(idx_to_label))]

# ─── TEST VERİSİ ──────────────────────────────────────────────────────────────
print("Test verisi yükleniyor...")
test_dir = os.path.join(KEYPOINT_DIR, "Test")
X, y_true = [], []

for label in sorted(os.listdir(test_dir)):
    label_dir = os.path.join(test_dir, label)
    if not os.path.isdir(label_dir) or label not in label_to_idx:
        continue
    files = [f for f in os.listdir(label_dir) if f.endswith(".npy")]
    print(f"  {label:15s}: {len(files)} test örneği")
    for fname in files:
        seq = np.load(os.path.join(label_dir, fname))
        if len(seq) < MAX_FRAMES:
            pad = np.zeros((MAX_FRAMES - len(seq), FEATURE_DIM))
            seq = np.vstack([seq, pad])
        else:
            seq = seq[:MAX_FRAMES]
        X.append(seq)
        y_true.append(label_to_idx[label])

X = np.array(X, dtype=np.float32)
print(f"\nToplam {len(X)} test örneği değerlendiriliyor...")

# ─── TAHMİN ───────────────────────────────────────────────────────────────────
y_pred_probs = model.predict(X, batch_size=32, verbose=1)
y_pred       = np.argmax(y_pred_probs, axis=1)

# ─── RAPOR ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST SETİ SINIFLANDIRMA RAPORU")
print("="*60)
print(classification_report(y_true, y_pred, target_names=labels))

accuracy = np.mean(np.array(y_true) == np.array(y_pred))
print(f"Genel Test Doğruluğu: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ─── CONFUSION MATRIX ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
fig_size = max(10, len(labels))
plt.figure(figsize=(fig_size, fig_size - 1))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="YlOrRd",
    xticklabels=labels, yticklabels=labels,
    linewidths=0.5
)
plt.title(f"Confusion Matrix — Test Seti (Doğruluk: {accuracy*100:.1f}%)", fontsize=13)
plt.ylabel("Gerçek", fontsize=11)
plt.xlabel("Tahmin", fontsize=11)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150, bbox_inches="tight")
print("\n✅ Confusion matrix → models/confusion_matrix.png")
plt.show()

# ─── KELIME BAZLI DOĞRULUK ────────────────────────────────────────────────────
print("\nKelime bazlı doğruluk:")
print(f"{'Kelime':15s}  {'Doğru':>5}  {'Toplam':>6}  {'%':>6}")
print("-" * 38)
for i, lbl in enumerate(labels):
    mask    = np.array(y_true) == i
    total   = mask.sum()
    correct = (np.array(y_pred)[mask] == i).sum()
    pct     = correct / total * 100 if total > 0 else 0
    bar     = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
    print(f"  {lbl:13s}  {correct:5d}  {total:6d}  {pct:5.1f}%  {bar}")
