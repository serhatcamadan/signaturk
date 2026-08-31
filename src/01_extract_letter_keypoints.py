#!/usr/bin/env python3
"""
01_extract_letter_keypoints.py
==============================
Fotoğraflardan (jpg/png) MediaPipe ile el keypoint'i çıkarır.
Her fotoğraf → 63 boyutlu vektör (21 keypoint × 3 koordinat)
Tek el kullanılır (harf modeli için yeterli).

Klasör yapısı:
  data/raw_photos/
    train/A/*.jpg
    train/B/*.jpg
    ...
    test/A/*.jpg
    ...

Çıktı:
  data/letter_keypoints/
    train/A/*.npy
    test/A/*.npy
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

# ─── AYARLAR ──────────────────────────────────────────────────────────────────
RAW_PHOTO_DIR  = "data/raw_photos"
KEYPOINT_DIR   = "data/letter_keypoints"
SPLITS         = ["train", "test"]
SKIP_EXISTING  = True
MIN_CONFIDENCE = 0.5   # El algılama güven eşiği

# ─── MEDİAPIPE (statik fotoğraf modu) ────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,    # Fotoğraf modu — her frame bağımsız
    max_num_hands=2,           # İki el — tek ve çift el harfleri için
    min_detection_confidence=MIN_CONFIDENCE,
    model_complexity=1
)

# ─── FONKSİYONLAR ─────────────────────────────────────────────────────────────
def extract_keypoints_from_image(img_path):
    """
    Fotoğraftan 126 boyutlu keypoint vektörü çıkarır.
    [sol_el: 63 | sağ_el: 63]
    Tek elle gösterilen harflerde olmayan el sıfır kalır.
    En az 1 el bulunamazsa None döner.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    left_kp  = np.zeros(63, dtype=np.float32)
    right_kp = np.zeros(63, dtype=np.float32)

    for hand_lm, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
        label  = hand_info.classification[0].label
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark],
            dtype=np.float32
        ).flatten()  # (63,)
        if label == "Left":
            left_kp = coords
        else:
            right_kp = coords

    return np.concatenate([left_kp, right_kp])  # (126,)

def normalize_keypoints(kp):
    """
    Her el için ayrı ayrı normalize et:
    Bileği (index 0) merkeze al, max genliğe böl.
    Sıfır olan el (görünmeyen) normalize edilmez.
    """
    left  = kp[:63].reshape(21, 3)
    right = kp[63:].reshape(21, 3)

    if np.any(left != 0):
        left = left - left[0]
        scale = np.max(np.abs(left)) + 1e-6
        left  = left / scale

    if np.any(right != 0):
        right = right - right[0]
        scale = np.max(np.abs(right)) + 1e-6
        right = right / scale

    return np.concatenate([left.flatten(), right.flatten()])

# ─── ANA DÖNGÜ ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(RAW_PHOTO_DIR):
        print(f"HATA: '{RAW_PHOTO_DIR}' bulunamadı.")
        print("cd ~/tsl_project komutunu çalıştırdın mı?")
        exit(1)

    grand_total = grand_ok = grand_skip = grand_fail = grand_no_hand = 0

    for split in SPLITS:
        split_dir = os.path.join(RAW_PHOTO_DIR, split)
        if not os.path.exists(split_dir):
            print(f"[{split}] bulunamadı, atlanıyor.")
            continue

        labels = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
            and not d.startswith('.')
        ])

        print(f"\n{'='*55}")
        print(f"[{split.upper()}]  {len(labels)} sınıf: {labels}")
        print(f"{'='*55}")

        split_ok = split_fail = split_skip = split_no_hand = 0

        for label in labels:
            label_dir = os.path.join(split_dir, label)
            images    = sorted([
                f for f in os.listdir(label_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

            ok = fail = skip = no_hand = 0

            for img_file in tqdm(images, desc=f"  {label:4s}", leave=False):
                grand_total += 1
                src  = os.path.join(label_dir, img_file)
                name = os.path.splitext(img_file)[0]
                dst  = os.path.join(KEYPOINT_DIR, split, label, name + ".npy")

                if SKIP_EXISTING and os.path.exists(dst):
                    skip += 1
                    grand_skip += 1
                    continue

                kp = extract_keypoints_from_image(src)

                if kp is None:
                    no_hand += 1
                    grand_no_hand += 1
                    continue

                kp_norm = normalize_keypoints(kp)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                np.save(dst, kp_norm)
                ok += 1
                grand_ok += 1

            total = len(images)
            print(f"  {label:4s} → ✓{ok:4d}  el_yok:{no_hand:3d}  atlandı:{skip:4d}  (toplam:{total})")
            split_ok += ok; split_fail += fail
            split_skip += skip; split_no_hand += no_hand

        print(f"  [{split.upper()} TOPLAM] ✓{split_ok}  el_yok:{split_no_hand}  atlandı:{split_skip}")

    print(f"\n{'='*55}")
    print(f"GENEL TOPLAM: {grand_total} fotoğraf")
    print(f"  ✓ Başarılı   : {grand_ok}")
    print(f"  ✗ El bulunamadı: {grand_no_hand}  ({grand_no_hand/grand_total*100:.1f}%)")
    print(f"  ↷ Atlandı    : {grand_skip}")
    print(f"{'='*55}")
    print("Harf keypoint çıkarma tamamlandı!")