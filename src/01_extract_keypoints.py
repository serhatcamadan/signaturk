#!/usr/bin/env python3
"""
01_extract_keypoints.py
=======================
Gerçek veri yapısına uygun keypoint çıkarma scripti.

Klasör yapısı:
  data/raw_videos/
    Train/kelime/signer{N}_sample{N}_color.mp4
    Test/kelime/signer{N}_sample{N}_color.mp4
    Val/kelime/signer{N}_sample{N}_color.mp4

Çıktı:
  data/keypoints/
    Train/kelime/signer{N}_sample{N}_color.npy
    Test/kelime/signer{N}_sample{N}_color.npy
    Val/kelime/signer{N}_sample{N}_color.npy
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

# ─── AYARLAR ──────────────────────────────────────────────────────────────────
RAW_VIDEO_DIR = "data/raw_videos"
KEYPOINT_DIR  = "data/keypoints"
SPLITS        = ["Train", "Test", "Val"]   # Klasör adları büyük/küçük harf önemli
MAX_FRAMES    = 60
MIN_FRAMES    = 10   # Bu sayının altındaki videolar atlanır
SKIP_EXISTING = True  # True: zaten işlenmiş dosyaları atla (hız kazanımı)

# ─── MEDİAPIPE ────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# ─── FONKSİYONLAR ─────────────────────────────────────────────────────────────
def extract_keypoints(frame):
    """Frame → 126 boyutlu keypoint vektörü (sol_el:63 + sağ_el:63)"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    left_kp  = np.zeros(63, dtype=np.float32)
    right_kp = np.zeros(63, dtype=np.float32)

    if result.multi_hand_landmarks:
        for hand_lm, hand_info in zip(
            result.multi_hand_landmarks,
            result.multi_handedness
        ):
            label  = hand_info.classification[0].label
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark],
                dtype=np.float32
            ).flatten()
            if label == "Left":
                left_kp = coords
            else:
                right_kp = coords

    return np.concatenate([left_kp, right_kp])  # (126,)


def normalize_sequence(sequence):
    """
    Her frame için:
    - Bileği (index 0) merkeze al → konumdan bağımsız
    - Max genliğe böl → ölçekten bağımsız
    """
    normalized = sequence.copy()
    for i, frame in enumerate(sequence):
        left  = frame[:63].reshape(21, 3)
        right = frame[63:].reshape(21, 3)

        if np.any(left != 0):
            origin = left[0].copy()
            left   = left - origin
            scale  = np.max(np.abs(left)) + 1e-6
            left   = left / scale

        if np.any(right != 0):
            origin = right[0].copy()
            right  = right - origin
            scale  = np.max(np.abs(right)) + 1e-6
            right  = right / scale

        normalized[i] = np.concatenate([left.flatten(), right.flatten()])
    return normalized


def process_video(video_path, output_path):
    """Video → keypoint dizisi → .npy dosyası"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "open_error"

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(extract_keypoints(frame))
    cap.release()

    if len(frames) < MIN_FRAMES:
        return f"too_short({len(frames)})"

    # Eşit aralıklı örnekleme ile MAX_FRAMES'e indir
    if len(frames) > MAX_FRAMES:
        indices = np.linspace(0, len(frames) - 1, MAX_FRAMES, dtype=int)
        frames  = [frames[idx] for idx in indices]

    sequence = np.array(frames, dtype=np.float32)   # (T, 126)
    sequence = normalize_sequence(sequence)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, sequence)
    return "ok"


# ─── ANA DÖNGÜ ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(RAW_VIDEO_DIR):
        print(f"HATA: '{RAW_VIDEO_DIR}' bulunamadı.")
        print("cd ~/tsl_project komutunu çalıştırdın mı?")
        exit(1)

    grand_total = grand_ok = grand_skip = grand_fail = 0

    for split in SPLITS:
        split_dir = os.path.join(RAW_VIDEO_DIR, split)
        if not os.path.exists(split_dir):
            print(f"[{split}] klasörü bulunamadı, atlanıyor.")
            continue

        labels = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])
        print(f"\n{'='*55}")
        print(f"[{split}]  {len(labels)} kelime bulundu: {labels}")
        print(f"{'='*55}")

        split_ok = split_fail = split_skip = 0

        for label in labels:
            label_dir = os.path.join(split_dir, label)
            videos = sorted([
                f for f in os.listdir(label_dir)
                if f.lower().endswith((".mp4", ".avi", ".mov"))
            ])

            ok = fail = skip = 0
            for video_file in tqdm(videos, desc=f"  {label:15s}", leave=False):
                grand_total += 1
                src  = os.path.join(label_dir, video_file)
                name = os.path.splitext(video_file)[0]
                dst  = os.path.join(KEYPOINT_DIR, split, label, name + ".npy")

                if SKIP_EXISTING and os.path.exists(dst):
                    skip += 1
                    grand_skip += 1
                    continue

                result = process_video(src, dst)
                if result == "ok":
                    ok += 1
                    grand_ok += 1
                else:
                    fail += 1
                    grand_fail += 1
                    tqdm.write(f"    ✗ {video_file} → {result}")

            print(f"  {label:15s} → ✓{ok:3d}  ✗{fail:2d}  atlandı:{skip:3d}  (toplam:{len(videos)})")
            split_ok += ok; split_fail += fail; split_skip += skip

        print(f"  [{split} TOPLAM] ✓{split_ok}  ✗{split_fail}  atlandı:{split_skip}")

    print(f"\n{'='*55}")
    print(f"GENEL TOPLAM: {grand_total} video")
    print(f"  ✓ Başarılı : {grand_ok}")
    print(f"  ✗ Başarısız: {grand_fail}")
    print(f"  ↷ Atlandı  : {grand_skip}")
    print(f"{'='*55}")
    print("Keypoint çıkarma tamamlandı!")
    print(f"Çıktılar: {KEYPOINT_DIR}/")
