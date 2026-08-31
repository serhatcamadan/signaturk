#!/usr/bin/env python3
"""
07_collect_my_data.py — Kişisel Harf Verisi Toplama
Her harf için kamera karşısında pozisyonu tut, otomatik kaydeder.
Toplanan veri mevcut veriyle birleştirilecek ve model fine-tune edilecek.
"""

import cv2
import mediapipe as mp
import numpy as np
import os, time, json
from collections import deque

# ─── AYARLAR ──────────────────────────────────────────────────────────────────
SAVE_DIR       = "data/my_letter_data"   # Kişisel veriler buraya
SAMPLES_NEEDED = 100   # Her harf için kaç örnek toplanacak
COUNTDOWN_SEC  = 3     # Harf değişiminde geri sayım
HOLD_THRESH    = 0.008 # Hareketsizlik eşiği — el sabit mi?
MIN_CONFIDENCE = 0.5

# Tüm sınıflar — eğitim verisindeki sırayla aynı
LABELS = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
          'N','O','P','R','S','T','U','V','Y','Z','del']

# ─── MEDİAPIPE ────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=MIN_CONFIDENCE,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# ─── KEYPOINT ÇIKARMA ─────────────────────────────────────────────────────────
def extract_and_normalize(frame):
    # NOT: flip YOK — gerçek zamanlı ile aynı olması için
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    left_kp  = np.zeros(63, dtype=np.float32)
    right_kp = np.zeros(63, dtype=np.float32)
    if result.multi_hand_landmarks:
        for hand_lm, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            label  = hand_info.classification[0].label
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark],
                               dtype=np.float32).flatten()
            if label == "Left":  left_kp  = coords
            else:                right_kp = coords
    def norm(kp):
        if np.any(kp != 0):
            pts = kp.reshape(21, 3); pts = pts - pts[0]
            return (pts / (np.max(np.abs(pts)) + 1e-6)).flatten()
        return kp
    hands_on = result.multi_hand_landmarks is not None
    return np.concatenate([norm(left_kp), norm(right_kp)]), hands_on, result

# ─── HAREKET TESPİTİ ──────────────────────────────────────────────────────────
kp_history = deque(maxlen=8)

def is_hand_stable():
    if len(kp_history) < 6:
        return False
    frames  = np.array(list(kp_history))
    nonzero = frames[np.any(frames != 0, axis=1)]
    if len(nonzero) < 4:
        return False
    return float(np.mean(np.abs(np.diff(nonzero, axis=0)))) < HOLD_THRESH

# ─── MEVCUT İLERLEMEYİ YÜKLE ──────────────────────────────────────────────────
def count_existing(label):
    d = os.path.join(SAVE_DIR, label)
    if not os.path.exists(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith('.npy')])

def get_start_label():
    """Kaldığı yerden devam et"""
    for label in LABELS:
        if count_existing(label) < SAMPLES_NEEDED:
            return label
    return None

# ─── ANA DÖNGÜ ────────────────────────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)

# Mevcut ilerlemeyi göster
print("\n=== Mevcut İlerleme ===")
for label in LABELS:
    count = count_existing(label)
    bar   = '█' * int(count / SAMPLES_NEEDED * 20) + '░' * (20 - int(count / SAMPLES_NEEDED * 20))
    done  = "✓" if count >= SAMPLES_NEEDED else " "
    print(f"  {done} {label:4s}: {count:3d}/{SAMPLES_NEEDED}  {bar}")

current_label = get_start_label()
if current_label is None:
    print("\n✅ Tüm harfler tamamlandı!")
    exit(0)

print(f"\nBaşlangıç: {current_label}")
print("Kontroller: Q=çık | SPACE=sonraki harf | ENTER=bu harfi atla\n")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

label_idx      = LABELS.index(current_label)
collected      = count_existing(current_label)
countdown_end  = time.time() + COUNTDOWN_SEC
in_countdown   = True
save_counter   = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # NOT: flip YOK — eğitim verisiyle tutarlı olması için
    h, w = frame.shape[:2]

    kp, hands_on, mp_result = extract_and_normalize(frame)
    kp_history.append(kp)
    stable = is_hand_stable() and hands_on

    # El iskeleti çiz
    if mp_result.multi_hand_landmarks:
        for hand_lm, hand_info in zip(mp_result.multi_hand_landmarks,
                                       mp_result.multi_handedness):
            side  = hand_info.classification[0].label
            color = (0, 200, 255) if side == "Right" else (255, 150, 0)
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(255,255,255), thickness=2))

    now = time.time()

    # ── Veri kaydet (countdown bittiyse ve el sabitsa)
    if not in_countdown and stable and collected < SAMPLES_NEEDED:
        fname = os.path.join(SAVE_DIR, current_label,
                             f"my_{current_label}_{save_counter:04d}.npy")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        np.save(fname, kp)
        collected    += 1
        save_counter += 1

    # ── Harf tamamlandıysa sonrakine geç
    if collected >= SAMPLES_NEEDED:
        label_idx += 1
        while label_idx < len(LABELS):
            current_label = LABELS[label_idx]
            collected     = count_existing(current_label)
            save_counter  = collected
            if collected < SAMPLES_NEEDED:
                break
            label_idx += 1

        if label_idx >= len(LABELS):
            # Tümü tamamlandı
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (w,h), (0,100,0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, "TUM HARFLER TAMAMLANDI!", (w//2-300, h//2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,100), 3)
            cv2.imshow("Veri Toplama", frame)
            cv2.waitKey(3000)
            break

        countdown_end = now + COUNTDOWN_SEC
        in_countdown  = True
        print(f"\n→ Sıradaki: {current_label}")

    # ── UI ────────────────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h-110), (w, h), (15, 15, 30), -1)
    cv2.addWeighted(ov2, 0.82, frame, 0.18, 0, frame)

    # Büyük harf etiketi
    cv2.putText(frame, current_label, (30, 78),
                cv2.FONT_HERSHEY_DUPLEX, 2.8, (255, 255, 100), 4)

    # İlerleme çubuğu
    progress = collected / SAMPLES_NEEDED
    bar_w    = int((w - 60) * progress)
    cv2.rectangle(frame, (30, 85), (w-30, 98), (40,40,60), -1)
    cv2.rectangle(frame, (30, 85), (30+bar_w, 98), (0,200,100), -1)
    cv2.putText(frame, f"{collected}/{SAMPLES_NEEDED}", (w//2-40, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    if in_countdown:
        remaining = countdown_end - now
        if remaining > 0:
            # Geri sayım ekranı
            ov3 = frame.copy()
            cv2.rectangle(ov3, (w//2-200, h//2-80), (w//2+200, h//2+80), (20,20,20), -1)
            cv2.addWeighted(ov3, 0.8, frame, 0.2, 0, frame)
            cv2.putText(frame, f"{current_label} harfini hazırla",
                        (w//2-180, h//2-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,100), 2)
            cv2.putText(frame, f"{remaining:.1f}",
                        (w//2-30, h//2+60), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0,200,255), 3)
        else:
            in_countdown = False
            print(f"  ▶ {current_label} kaydediliyor...")

    else:
        # Kayıt durumu
        if stable:
            dot_color = (0, 255, 100)
            status    = f"KAYDEDILIYOR  {collected}/{SAMPLES_NEEDED}"
        elif hands_on:
            dot_color = (0, 140, 220)
            status    = "El sabit degil — hareketsiz tut"
        else:
            dot_color = (80, 80, 200)
            status    = "El bulunamadi — kameraya goster"

        cv2.circle(frame, (30, h-75), 10, dot_color, -1)
        cv2.putText(frame, status, (50, h-68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, dot_color, 2)

        # Stabilite göstergesi
        if hands_on:
            kp_arr    = np.array(list(kp_history))
            nonzero   = kp_arr[np.any(kp_arr != 0, axis=1)]
            stab_val  = 1.0 - min(float(np.mean(np.abs(np.diff(nonzero, axis=0)))) / HOLD_THRESH, 1.0) \
                        if len(nonzero) >= 2 else 0.0
            stab_w    = int(300 * stab_val)
            cv2.putText(frame, "Sabitlik:", (30, h-42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140,140,140), 1)
            cv2.rectangle(frame, (105, h-53), (405, h-38), (40,40,60), -1)
            sc = (0,220,80) if stab_val > 0.7 else (0,140,220) if stab_val > 0.4 else (0,80,180)
            cv2.rectangle(frame, (105, h-53), (105+stab_w, h-38), sc, -1)

    # Tüm harfler özeti (sağ panel)
    panel_x = w - 160
    cv2.rectangle(frame, (panel_x-10, 105), (w, h-115), (20,20,35), -1)
    for i, lbl in enumerate(LABELS):
        cnt   = count_existing(lbl) if lbl != current_label else collected
        done  = cnt >= SAMPLES_NEEDED
        color = (0,200,80) if done else (255,255,100) if lbl == current_label else (100,100,100)
        check = "✓" if done else "→" if lbl == current_label else " "
        cv2.putText(frame, f"{check}{lbl}:{cnt}", (panel_x, 125 + i*26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1 if not done else 2)

    # Kontroller
    cv2.putText(frame, "Q:cik  SPACE:sonraki  ENTER:atla",
                (30, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80,80,80), 1)

    cv2.imshow(f"Veri Toplama — {current_label} ({collected}/{SAMPLES_NEEDED})", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print(f"\nDurduruldu. {current_label}: {collected}/{SAMPLES_NEEDED}")
        break
    elif key == ord(" "):
        # Sonraki harfe atla (bu harfi tamamlanmış say)
        print(f"  → {current_label} atlandı ({collected} örnek)")
        collected = SAMPLES_NEEDED  # Bir sonraki iterasyonda geçecek
    elif key == 13:  # ENTER
        # Bu harfi tamamen atla (0 örnek bırak)
        print(f"  → {current_label} tamamen atlandı")
        label_idx += 1
        if label_idx < len(LABELS):
            current_label = LABELS[label_idx]
            collected     = count_existing(current_label)
            save_counter  = collected
            countdown_end = time.time() + COUNTDOWN_SEC
            in_countdown  = True

cap.release()
cv2.destroyAllWindows()

print("\n=== Final İlerleme ===")
total = 0
for label in LABELS:
    cnt = count_existing(label)
    total += cnt
    bar  = '█' * int(cnt/SAMPLES_NEEDED*20) + '░' * (20-int(cnt/SAMPLES_NEEDED*20))
    done = "✓" if cnt >= SAMPLES_NEEDED else " "
    print(f"  {done} {label:4s}: {cnt:3d}/{SAMPLES_NEEDED}  {bar}")
print(f"\nToplam: {total} örnek toplandı")
print(f"Veriler: {SAVE_DIR}/")
print("\nSonraki adım: python src/08_finetune.py")