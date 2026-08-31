#!/usr/bin/env python3
"""
04_realtime.py — Motion Detection + Gürültü Engelleme + 30 Kelime
3 Katmanlı Gürültü Engelleme:
  1. Motion Detection: Hareketsizken tahmin yapma
  2. Güçlendirilmiş Onay: 5 ardışık + cooldown
  3. Güven Eşiği: %80
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import json, os, time
from collections import deque

# ─── AYARLAR ──────────────────────────────────────────────────────────────────
WEIGHTS_PATH   = "models/model_weights.weights.h5"
LABEL_MAP      = "models/label_map.json"
MAX_FRAMES     = 60
FEATURE_DIM    = 126
THRESHOLD      = 0.80   # %80 güven eşiği (eskiden 0.75)
CONFIRM_N      = 5      # 5 ardışık onay (eskiden 3)
PREDICT_EVERY  = 10
COOLDOWN_SEC   = 2.0    # Aynı kelime bu süreden önce tekrar onaylanmaz
MOTION_THRESH  = 0.015  # Bu altında → hareketsiz sayılır

# ─── DOSYA KONTROLÜ ───────────────────────────────────────────────────────────
for f in [WEIGHTS_PATH, LABEL_MAP]:
    if not os.path.exists(f):
        print(f"HATA: {f} bulunamadı!")
        exit(1)

with open(LABEL_MAP, "r", encoding="utf-8") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}
n_classes = len(label_map)
print(f"✅ {n_classes} kelime yüklendi")

# ─── MODEL ────────────────────────────────────────────────────────────────────
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

def build_model(n_classes):
    inp = layers.Input(shape=(MAX_FRAMES, FEATURE_DIM))
    x   = layers.Masking(mask_value=0.0)(inp)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2), name="bilstm_1")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(64,  return_sequences=True, dropout=0.2), name="bilstm_2")(x)
    x   = BahdanauAttention(64, name="attention")(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(64,  activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inp, out)

print("Model yükleniyor...")
model = build_model(n_classes)
dummy = np.zeros((1, MAX_FRAMES, FEATURE_DIM), dtype=np.float32)
model(dummy, training=False)
model.load_weights(WEIGHTS_PATH)
print("✅ Model hazır!")

# ─── MEDİAPIPE ────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# ─── KEYPOINT + NORMALİZASYON ─────────────────────────────────────────────────
def extract_and_normalize(frame):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    left_kp  = np.zeros(63, dtype=np.float32)
    right_kp = np.zeros(63, dtype=np.float32)

    if result.multi_hand_landmarks:
        for hand_lm, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            label  = hand_info.classification[0].label
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32).flatten()
            if label == "Left":  left_kp  = coords
            else:                right_kp = coords

    def norm(kp):
        if np.any(kp != 0):
            pts   = kp.reshape(21, 3)
            pts   = pts - pts[0]
            scale = np.max(np.abs(pts)) + 1e-6
            return (pts / scale).flatten()
        return kp

    hands_on = result.multi_hand_landmarks is not None
    return np.concatenate([norm(left_kp), norm(right_kp)]), hands_on, result

# ─── KATMAN 1: HAREKET TESPİTİ ────────────────────────────────────────────────
def motion_score(frame_buffer):
    """
    Son N frame arasındaki ortalama değişimi hesaplar.
    Düşük değer → el hareketsiz → tahmin yapma.
    """
    if len(frame_buffer) < 15:
        return 0.0
    frames = np.array(list(frame_buffer)[-15:])  # Son 15 frame
    # Sadece el olan frame'leri değerlendir
    nonzero = frames[np.any(frames != 0, axis=1)]
    if len(nonzero) < 5:
        return 0.0
    diffs = np.diff(nonzero, axis=0)
    return float(np.mean(np.abs(diffs)))

# ─── UI ───────────────────────────────────────────────────────────────────────
def draw_ui(frame, pred, conf, confirmed_hist, hands_on, frame_buffer, is_moving, cooldown_left):
    h, w    = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 155), (w, h), (12, 12, 28), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # ── El göstergesi
    dot = (0, 220, 80) if hands_on else (80, 80, 200)
    cv2.circle(frame, (28, h - 130), 9, dot, -1)
    cv2.putText(frame, "El algilandi" if hands_on else "El yok",
                (48, h - 123), cv2.FONT_HERSHEY_SIMPLEX, 0.55, dot, 2)

    # ── Hareket göstergesi (Katman 1)
    motion_color = (0, 200, 100) if is_moving else (60, 60, 180)
    motion_txt   = "HAREKET VAR" if is_moving else "Hareketsiz — bekleniyor"
    cv2.putText(frame, motion_txt,
                (w - 280, h - 123), cv2.FONT_HERSHEY_SIMPLEX, 0.52, motion_color, 2)

    # ── Cooldown göstergesi
    if cooldown_left > 0:
        cv2.putText(frame, f"Bekleme: {cooldown_left:.1f}s",
                    (w - 200, h - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 0), 1)

    # ── Tampon doluluk
    fill  = len(frame_buffer) / MAX_FRAMES
    bar_w = int(200 * fill)
    cv2.rectangle(frame, (28, h - 100), (228, h - 86), (40, 40, 60), -1)
    cv2.rectangle(frame, (28, h - 100), (28 + bar_w, h - 86), (0, 160, 120), -1)
    cv2.putText(frame, "tampon", (28, h - 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

    # ── Ana tahmin
    if not is_moving:
        color = (60, 60, 140)
        txt   = "El hareketsiz..."
    elif pred and conf >= THRESHOLD:
        color = (0, 255, 160)
        txt   = f"{pred.upper()}   {conf*100:.1f}%"
    elif pred:
        color = (0, 140, 220)
        txt   = f"{pred}  ({conf*100:.1f}%)  dusuk guven"
    else:
        color = (120, 120, 120)
        txt   = "Bekleniyor..."

    cv2.putText(frame, txt, (18, h - 55),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2)

    # ── Onaylanan kelimeler geçmişi
    hist = "  |  ".join(list(confirmed_hist)[-7:])
    cv2.putText(frame, hist, (18, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    # ── Kontroller
    cv2.putText(frame, "Q:cik  C:temizle",
                (w - 170, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 100), 1)

    # ── Güven çubuğu (sadece hareket varsa)
    if is_moving and pred:
        bar = int(w * min(conf, 1.0))
        cv2.rectangle(frame, (0, h - 7), (bar, h), color, -1)

    return frame

# ─── ANA DÖNGÜ ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("HATA: Kamera açılamadı!")
    print("Sistem Ayarları → Gizlilik → Kamera → Terminal'e izin ver")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_buffer    = deque(maxlen=MAX_FRAMES)
recent_preds    = deque(maxlen=CONFIRM_N)
confirmed_words = deque(maxlen=8)

cur_pred      = None
cur_conf      = 0.0
frame_count   = 0
last_confirmed_word = None
last_confirmed_time = 0.0

print("\n✅ Kamera başlatıldı!")
print("   Q = çık   |   C = geçmişi temizle\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    # Keypoint çıkar
    kp, hands_on, mp_result = extract_and_normalize(frame)
    frame_buffer.append(kp)

    # ── KATMAN 1: Hareket tespiti
    m_score   = motion_score(frame_buffer)
    is_moving = m_score > MOTION_THRESH

    # El iskeleti çiz
    if mp_result.multi_hand_landmarks:
        for hand_lm in mp_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1)
            )

    # Hareketsizse tahmin sıfırla
    if not is_moving:
        cur_pred = None
        cur_conf = 0.0
        recent_preds.clear()

    # ── Tahmin (hareket varsa + tampon doluysa + doğru aralıkta)
    elif len(frame_buffer) == MAX_FRAMES and frame_count % PREDICT_EVERY == 0:
        seq      = np.array(frame_buffer, dtype=np.float32)[np.newaxis]
        probs    = model.predict(seq, verbose=0)[0]
        top_idx  = int(np.argmax(probs))
        cur_conf = float(probs[top_idx])
        cur_pred = label_map[top_idx]

        # ── KATMAN 2: Güçlendirilmiş ardışık onay
        recent_preds.append(cur_pred)

        if (len(recent_preds) == CONFIRM_N and
                len(set(recent_preds)) == 1 and
                cur_conf >= THRESHOLD):

            word = recent_preds[-1]
            now  = time.time()

            # ── KATMAN 2: Cooldown kontrolü
            cooldown_ok = (
                word != last_confirmed_word or
                (now - last_confirmed_time) >= COOLDOWN_SEC
            )

            if cooldown_ok:
                confirmed_words.append(word)
                last_confirmed_word = word
                last_confirmed_time = now
                print(f">>> {word}  ({cur_conf:.1%})")
                recent_preds.clear()  # Onaydan sonra sıfırla

    # Cooldown kalan süre
    now           = time.time()
    cooldown_left = max(0.0, COOLDOWN_SEC - (now - last_confirmed_time)) if last_confirmed_word else 0.0

    # UI çiz
    frame = draw_ui(frame, cur_pred, cur_conf,
                    confirmed_words, hands_on, frame_buffer,
                    is_moving, cooldown_left)

    cv2.imshow("TID Tanima  [Q:cik | C:temizle]", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        confirmed_words.clear()
        recent_preds.clear()
        last_confirmed_word = None
        print("Gecmis temizlendi.")

cap.release()
cv2.destroyAllWindows()
print("\nProgram sonlandı.")