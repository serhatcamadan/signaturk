#!/usr/bin/env python3
"""
02_train_model.py
=================
Hazır Train/Val ayrımını kullanarak model eğitir.
Veri artırma SADECE Train setine uygulanır.
Test seti bu scriptte kullanılmaz — 03_evaluate.py için saklanır.
"""

import numpy as np
import os, json
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from sklearn.preprocessing import LabelEncoder

# ─── AYARLAR ──────────────────────────────────────────────────────────────────
KEYPOINT_DIR = "data/keypoints"
MODEL_DIR    = "models"
MAX_FRAMES   = 60
FEATURE_DIM  = 126
EPOCHS       = 100
BATCH_SIZE   = 32   # M4 için ideal; OOM alırsan 8'e düşür

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ─── GPU KONTROL ──────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"✅ Metal GPU aktif: {gpus[0].name}")
else:
    print("⚠️  GPU bulunamadı — CPU ile devam (yavaş olacak)")

# ─── VERİ YÜKLEME ─────────────────────────────────────────────────────────────
def load_split(split_name):
    """
    data/keypoints/{split_name}/{kelime}/*.npy dosyalarını yükler.
    Döner: X (N, 60, 126) float32, y (N,) string listesi
    """
    split_dir = os.path.join(KEYPOINT_DIR, split_name)
    if not os.path.exists(split_dir):
        print(f"  UYARI: {split_dir} bulunamadı, boş döndürülüyor.")
        return np.array([]), []

    X, y = [], []
    labels = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])

    for label in labels:
        label_dir = os.path.join(split_dir, label)
        files = [f for f in os.listdir(label_dir) if f.endswith(".npy")]
        for fname in files:
            seq = np.load(os.path.join(label_dir, fname))
            # Uzunluğu MAX_FRAMES'e eşitle
            if len(seq) < MAX_FRAMES:
                pad = np.zeros((MAX_FRAMES - len(seq), FEATURE_DIM), dtype=np.float32)
                seq = np.vstack([seq, pad])
            else:
                seq = seq[:MAX_FRAMES]
            X.append(seq)
            y.append(label)

    print(f"  [{split_name}] {len(labels)} kelime, {len(X)} örnek yüklendi.")
    return np.array(X, dtype=np.float32), y


print("\nVeriler yükleniyor...")
X_train, y_train_raw = load_split("Train")
X_val,   y_val_raw   = load_split("Val")

if len(X_train) == 0:
    print("HATA: Train verisi bulunamadı. 01_extract_keypoints.py çalıştırıldı mı?")
    exit(1)

# ─── ETİKET KODLAMA ───────────────────────────────────────────────────────────
# Train + Val etiketlerinden birleşik encoder oluştur
all_labels = sorted(set(y_train_raw + y_val_raw))
le = LabelEncoder()
le.fit(all_labels)
n_classes = len(le.classes_)

y_train_enc = tf.keras.utils.to_categorical(le.transform(y_train_raw), n_classes)
y_val_enc   = tf.keras.utils.to_categorical(le.transform(y_val_raw),   n_classes)

# Etiket haritasını kaydet (inference için şart)
label_map = {str(i): cls for i, cls in enumerate(le.classes_)}
with open(os.path.join(MODEL_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print(f"\n{n_classes} kelime: {list(le.classes_)}")
print(f"Train: {len(X_train)} örnek | Val: {len(X_val)} örnek")

# ─── VERİ ARTIRMA (sadece Train) ──────────────────────────────────────────────
from scipy.ndimage import uniform_filter1d

def augment_sequence(seq):
    """5 farklı augmentasyon — Train setini ~6x büyütür"""
    aug = []

    # 1. Gaussian gürültü
    aug.append(seq + np.random.normal(0, 0.015, seq.shape).astype(np.float32))

    # 2. Zamansal ölçekleme (%75–125 hız)
    factor  = np.random.uniform(0.75, 1.25)
    new_len = max(10, min(int(MAX_FRAMES * factor), MAX_FRAMES))
    idx     = np.linspace(0, MAX_FRAMES - 1, new_len, dtype=int)
    scaled  = seq[idx]
    if len(scaled) < MAX_FRAMES:
        pad    = np.zeros((MAX_FRAMES - len(scaled), FEATURE_DIM), dtype=np.float32)
        scaled = np.vstack([scaled, pad])
    aug.append(scaled[:MAX_FRAMES])

    # 3. Zamansal kaydırma
    shift   = np.random.randint(-8, 9)
    shifted = np.roll(seq, shift, axis=0)
    if shift > 0:
        shifted[:shift] = 0
    aug.append(shifted)

    # 4. Zamansal yumuşatma
    aug.append(uniform_filter1d(seq, size=3, axis=0).astype(np.float32))

    # 5. Ölçek pertürbasyonu
    aug.append((seq * np.random.uniform(0.85, 1.15)).astype(np.float32))

    return aug


print("\nVeri artırma uygulanıyor (Train)...")
X_aug = list(X_train)
y_aug = list(y_train_enc)

for i in range(len(X_train)):
    for aug_seq in augment_sequence(X_train[i]):
        arr = aug_seq[:MAX_FRAMES]
        if len(arr) < MAX_FRAMES:
            pad = np.zeros((MAX_FRAMES - len(arr), FEATURE_DIM), dtype=np.float32)
            arr = np.vstack([arr, pad])
        X_aug.append(arr.astype(np.float32))
        y_aug.append(y_train_enc[i])

X_aug = np.array(X_aug, dtype=np.float32)
y_aug = np.array(y_aug, dtype=np.float32)
print(f"Train artırma sonrası: {len(X_aug)} örnek ({len(X_aug)//len(X_train)}x)")

# ─── ATTENTION KATMANI ────────────────────────────────────────────────────────
class BahdanauAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units, use_bias=False)
        self.V = layers.Dense(1,     use_bias=False)

    def call(self, hidden_states):
        score   = self.V(tf.nn.tanh(self.W(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(weights * hidden_states, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# ─── MODEL ────────────────────────────────────────────────────────────────────
def build_model(n_classes):
    inp = layers.Input(shape=(MAX_FRAMES, FEATURE_DIM))
    x   = layers.Masking(mask_value=0.0)(inp)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2),
        name="bilstm_1"
    )(x)
    x = layers.LayerNormalization()(x)

    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True, dropout=0.2),
        name="bilstm_2"
    )(x)

    x   = BahdanauAttention(64, name="attention")(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    return Model(inp, out)


model = build_model(n_classes)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001, epsilon=1e-7),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ─── CALLBACK'LER ─────────────────────────────────────────────────────────────
cb_list = [
    callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1
    ),
    callbacks.TensorBoard(log_dir="logs", histogram_freq=0),
]

# ─── XQEĞİTİM ──────────────────────────────────────────────────────────────────
print(f"\nEğitim başlıyor... ({EPOCHS} epoch max, early stopping aktif)")
history = model.fit(
    X_aug, y_aug,
    validation_data=(X_val, y_val_enc),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb_list,
    verbose=1
)

model.save(os.path.join(MODEL_DIR, "final_model.keras"))
best_val = max(history.history["val_accuracy"])
print(f"\n✅ Eğitim tamamlandı!")
print(f"   En iyi val_accuracy : {best_val:.4f} ({best_val*100:.2f}%)")
print(f"   Model kaydedildi    : {MODEL_DIR}/best_model.keras")
