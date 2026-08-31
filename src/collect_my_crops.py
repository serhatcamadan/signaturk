#!/usr/bin/env python3
"""
collect_my_crops.py — Kişisel CNN Verisi Toplama
SPACE tuşuna basınca kaydeder — otomatik değil, manuel tetik
"""

import cv2
import mediapipe as mp
import numpy as np
import os, time

SAVE_DIR       = "data/my_hand_crops"
SAMPLES_NEEDED = 100
COUNTDOWN_SEC  = 3
PADDING        = 0.25
IMG_SIZE       = 224

LABELS = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
          'N','O','P','R','S','T','U','V','Y','Z','del']

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5, model_complexity=1)

def get_crop(frame, result):
    if not result.multi_hand_landmarks: return None
    h, w = frame.shape[:2]
    all_x, all_y = [], []
    for hand_lm in result.multi_hand_landmarks:
        for lm in hand_lm.landmark:
            all_x.append(lm.x*w); all_y.append(lm.y*h)
    bw = max(all_x)-min(all_x); bh = max(all_y)-min(all_y)
    x1 = max(0, int(min(all_x)-bw*PADDING))
    y1 = max(0, int(min(all_y)-bh*PADDING))
    x2 = min(w, int(max(all_x)+bw*PADDING))
    y2 = min(h, int(max(all_y)+bh*PADDING))
    bw2=x2-x1; bh2=y2-y1
    if bw2>bh2: d=bw2-bh2; y1=max(0,y1-d//2); y2=min(h,y2+d//2)
    else:        d=bh2-bw2; x1=max(0,x1-d//2); x2=min(w,x2+d//2)
    if x2-x1<20 or y2-y1<20: return None
    return cv2.resize(frame[y1:y2, x1:x2], (IMG_SIZE,IMG_SIZE)), (x1,y1,x2,y2)

def count_existing(label):
    d = os.path.join(SAVE_DIR, label)
    if not os.path.exists(d): return 0
    return len([f for f in os.listdir(d) if f.endswith('.jpg')])

def get_start_label():
    for label in LABELS:
        if count_existing(label) < SAMPLES_NEEDED:
            return label
    return None

os.makedirs(SAVE_DIR, exist_ok=True)

print("\n=== Mevcut İlerleme ===")
for label in LABELS:
    count = count_existing(label)
    bar   = '█'*int(count/SAMPLES_NEEDED*20) + '░'*(20-int(count/SAMPLES_NEEDED*20))
    done  = "✓" if count >= SAMPLES_NEEDED else " "
    print(f"  {done} {label:4s}: {count:3d}/{SAMPLES_NEEDED}  {bar}")

current_label = get_start_label()
if current_label is None:
    print("\n✅ Tüm harfler tamamlandı!"); exit(0)

print(f"\nBaşlangıç: {current_label}")
print("SPACE=kaydet | ENTER=sonraki harf | Q=çık\n")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap.set(cv2.CAP_PROP_FPS,30)

label_idx     = LABELS.index(current_label)
collected     = count_existing(current_label)
save_counter  = collected
countdown_end = time.time() + COUNTDOWN_SEC
in_countdown  = True
flash_time    = 0  # Kayıt flash efekti

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    hands_on = result.multi_hand_landmarks is not None

    if result.multi_hand_landmarks:
        for hand_lm, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            side  = hand_info.classification[0].label
            color = (0,200,255) if side=="Right" else (255,150,0)
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=color,thickness=2,circle_radius=4),
                mp_draw.DrawingSpec(color=(255,255,255),thickness=2))

    now = time.time()

    # UI
    h, w = frame.shape[:2]
    ov=frame.copy(); cv2.rectangle(ov,(0,0),(w,100),(15,15,30),-1); cv2.addWeighted(ov,0.85,frame,0.15,0,frame)
    ov2=frame.copy(); cv2.rectangle(ov2,(0,h-100),(w,h),(15,15,30),-1); cv2.addWeighted(ov2,0.82,frame,0.18,0,frame)

    cv2.putText(frame, current_label, (30,78), cv2.FONT_HERSHEY_DUPLEX, 2.8, (255,255,100), 4)
    prog = collected/SAMPLES_NEEDED
    cv2.rectangle(frame,(30,85),(w-30,98),(40,40,60),-1)
    cv2.rectangle(frame,(30,85),(30+int((w-60)*prog),98),(0,200,100),-1)
    cv2.putText(frame,f"{collected}/{SAMPLES_NEEDED}",(w//2-40,95),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)

    if in_countdown:
        remaining = countdown_end - now
        if remaining > 0:
            ov3=frame.copy()
            cv2.rectangle(ov3,(w//2-220,h//2-80),(w//2+220,h//2+80),(20,20,20),-1)
            cv2.addWeighted(ov3,0.8,frame,0.2,0,frame)
            cv2.putText(frame,f"{current_label} harfini hazirla",
                        (w//2-190,h//2-20),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,100),2)
            cv2.putText(frame,f"{remaining:.1f}",
                        (w//2-30,h//2+60),cv2.FONT_HERSHEY_DUPLEX,2.0,(0,200,255),3)
        else:
            in_countdown = False
            print(f"  ▶ {current_label} — SPACE ile kaydet")
    else:
        # Kayıt ekranı
        if hands_on:
            # Crop önizleme — sağ alt köşe
            cr = get_crop(frame, result)
            if cr:
                crop_preview, bbox = cr
                x1,y1,x2,y2 = bbox
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,160),2)
                # Küçük önizleme
                preview = cv2.resize(crop_preview, (120,120))
                frame[h-130:h-10, w-130:w-10] = preview
                cv2.rectangle(frame,(w-130,h-130),(w-10,h-10),(0,255,160),2)
                cv2.putText(frame,"onizleme",(w-125,h-135),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,255,160),1)

            dot = (0,255,100); msg = "El hazir — SPACE ile kaydet"
        else:
            dot = (80,80,200); msg = "El bulunamadi"

        cv2.circle(frame,(28,h-65),9,dot,-1)
        cv2.putText(frame,msg,(48,h-58),cv2.FONT_HERSHEY_SIMPLEX,0.65,dot,2)

        # Flash efekti
        if now - flash_time < 0.15:
            ov_f=frame.copy(); cv2.rectangle(ov_f,(0,0),(w,h),(255,255,255),-1)
            cv2.addWeighted(ov_f,0.3,frame,0.7,0,frame)

    # Sağ panel
    px = w-155
    cv2.rectangle(frame,(px-5,105),(w,h-105),(20,20,35),-1)
    for i,lbl in enumerate(LABELS):
        cnt  = count_existing(lbl) if lbl!=current_label else collected
        done = cnt>=SAMPLES_NEEDED
        col  = (0,200,80) if done else (255,255,100) if lbl==current_label else (100,100,100)
        chk  = "v" if done else ">" if lbl==current_label else " "
        cv2.putText(frame,f"{chk}{lbl}:{cnt}",(px,120+i*25),cv2.FONT_HERSHEY_SIMPLEX,0.45,col,1 if not done else 2)

    cv2.putText(frame,"SPACE:kaydet  ENTER:atla  Q:cik",
                (30,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.42,(80,80,80),1)

    cv2.imshow(f"Veri Toplama — {current_label} ({collected}/{SAMPLES_NEEDED})",frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' ') and not in_countdown and hands_on:
        cr = get_crop(frame, result)
        if cr:
            crop, bbox = cr
            out_dir = os.path.join(SAVE_DIR, current_label)
            os.makedirs(out_dir, exist_ok=True)
            fname = os.path.join(out_dir, f"my_{current_label}_{save_counter:04d}.jpg")
            cv2.imwrite(fname, crop)
            collected += 1; save_counter += 1
            flash_time = now
            print(f"  ✓ {current_label} [{collected}/{SAMPLES_NEEDED}]")

            if collected >= SAMPLES_NEEDED:
                print(f"✅ {current_label} tamamlandı!")
                label_idx += 1
                while label_idx < len(LABELS):
                    current_label=LABELS[label_idx]; collected=count_existing(current_label)
                    save_counter=collected
                    if collected<SAMPLES_NEEDED: break
                    label_idx+=1
                if label_idx>=len(LABELS):
                    print("\n✅ Tüm harfler tamamlandı!"); break
                countdown_end=now+COUNTDOWN_SEC; in_countdown=True
                print(f"→ Sıradaki: {current_label}")
        else:
            print("  ✗ El crop alınamadı")

    elif key == 13:  # ENTER
        print(f"  → {current_label} atlandı")
        label_idx+=1
        if label_idx<len(LABELS):
            current_label=LABELS[label_idx]; collected=count_existing(current_label)
            save_counter=collected; countdown_end=time.time()+COUNTDOWN_SEC; in_countdown=True

cap.release(); cv2.destroyAllWindows()

print("\n=== Final İlerleme ===")
total=0
for label in LABELS:
    cnt=count_existing(label); total+=cnt
    bar='█'*int(cnt/SAMPLES_NEEDED*20)+'░'*(20-int(cnt/SAMPLES_NEEDED*20))
    done="✓" if cnt>=SAMPLES_NEEDED else " "
    print(f"  {done} {label:4s}: {cnt:3d}/{SAMPLES_NEEDED}  {bar}")
print(f"\nToplam: {total} örnek → {SAVE_DIR}/")
