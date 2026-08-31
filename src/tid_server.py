#!/usr/bin/env python3
"""
tid_server.py — TID İletişim Sistemi  (Flask + SocketIO)
Algoritma: Python (mediapipe, TF, Whisper, Groq)
Arayüz:    Tarayıcı üzerinden WhatsApp tarzı sohbet
"""

import cv2, mediapipe as mp, numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
import json, os, time, threading, subprocess, queue, tempfile, base64
from collections import deque
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from groq import Groq
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit

# ── AYARLAR ───────────────────────────────────────────────────────────────────
CNN_WEIGHTS    = "models/best_cnn.keras"
CNN_LABEL_MAP  = "models/cnn_label_map.json"
WORD_WEIGHTS   = "models/model_weights.weights.h5"
WORD_LABEL_MAP = "models/label_map.json"
ANIMATIONS_DIR = "animations"
IMG_SIZE       = 224
CONFIRM_N      = 8
THRESHOLD      = 0.85
COOLDOWN_SEC   = 2.0
PADDING        = 0.25
FLIP           = True
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
WHISPER_MODEL  = "medium"
LANGUAGE       = "tr"
SAMPLE_RATE    = 16000
MAX_FRAMES     = 60
FEATURE_DIM    = 126
MOTION_THRESH  = 0.002
PREDICT_EVERY  = 10
WORD_THRESHOLD = 0.80
WORD_CONFIRM_N = 3
WORD_COOLDOWN  = 3.0

app    = Flask(__name__)
sio    = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── MODEL BUILDER'LAR ─────────────────────────────────────────────────────────
def build_cnn_model(n):
    base = MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3), include_top=False, weights=None)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    return Model(base.input, layers.Dense(n, activation='softmax')(x))

class BahdanauAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs); self.units = units
        self.W = layers.Dense(units, use_bias=False)
        self.V = layers.Dense(1, use_bias=False)
    def call(self, h):
        return tf.reduce_sum(tf.nn.softmax(self.V(tf.nn.tanh(self.W(h))), axis=1)*h, axis=1)
    def get_config(self):
        c = super().get_config(); c.update({"units": self.units}); return c

def build_word_model(n):
    inp = layers.Input(shape=(MAX_FRAMES, FEATURE_DIM))
    x   = layers.Masking(mask_value=0.0)(inp)
    x   = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
    x   = layers.LayerNormalization()(x)
    x   = layers.Bidirectional(layers.LSTM(64,  return_sequences=True, dropout=0.2))(x)
    x   = BahdanauAttention(64)(x)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(64,  activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    return Model(inp, layers.Dense(n, activation='softmax')(x))

# ── UYGULAMA DURUMU ───────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.running         = True
        self.mode            = "kelime" # "harf" | "kelime"

        # Modeller
        self.cnn_model  = None; self.cnn_labels  = {}
        self.word_model = None; self.word_labels = {}
        self.whisper    = None
        self.groq_client= None

        # Harf modu
        self._pred_lock      = threading.Lock()
        self.cur_pred        = None
        self.cur_conf        = 0.0
        self.top5            = []
        self.recent_preds    = deque(maxlen=CONFIRM_N)
        self.last_conf_time  = 0.0
        self._predict_busy   = False
        self.harf_buffer     = ""

        # Kelime modu
        self.frame_buffer        = deque(maxlen=MAX_FRAMES)
        self.frame_count         = 0
        self.word_recent_preds   = deque(maxlen=WORD_CONFIRM_N)
        self.last_word           = None
        self.last_word_time      = 0.0
        self._word_predict_busy  = False
        self.kelime_buffer       = deque(maxlen=20)

        # Ses kayıt
        self.recording    = False
        self.audio_frames = []
        self.stream       = None

        # Animasyon
        self.anim_queue   = queue.Queue()
        self.sag_state    = "bekle"   # bekle | kayit | isleniyor | oynatma

        # Kamera frame (JPEG bytes) — browser'a stream edilecek
        self.cam_frame_lock = threading.Lock()
        self.cam_frame_jpg  = None

        # Sohbet geçmişi (SocketIO broadcast için)
        self.chat_history = []

        # LLM intent modu — "ifade" veya "soru"
        self.llm_intent = "ifade"

state = AppState()

# ── MODEL YÜKLEME ─────────────────────────────────────────────────────────────
def load_models():
    msgs = []
    if os.path.exists(CNN_WEIGHTS) and os.path.exists(CNN_LABEL_MAP):
        with open(CNN_LABEL_MAP, 'r', encoding='utf-8') as f:
            state.cnn_labels = {int(k): v for k, v in json.load(f).items()}
        state.cnn_model = build_cnn_model(len(state.cnn_labels))
        state.cnn_model(np.zeros((1,IMG_SIZE,IMG_SIZE,3), dtype=np.float32), training=False)
        try:
            state.cnn_model.load_weights(CNN_WEIGHTS)
        except Exception:
            loaded = tf.keras.models.load_model(CNN_WEIGHTS, compile=False)
            state.cnn_model.set_weights(loaded.get_weights())
        state.cnn_model(np.zeros((1,IMG_SIZE,IMG_SIZE,3), dtype=np.float32), training=False)
        msgs.append("CNN:OK")
    else:
        msgs.append("CNN:YOK")

    if os.path.exists(WORD_WEIGHTS) and os.path.exists(WORD_LABEL_MAP):
        with open(WORD_LABEL_MAP, 'r', encoding='utf-8') as f:
            state.word_labels = {int(k): v for k, v in json.load(f).items()}
        state.word_model = build_word_model(len(state.word_labels))
        state.word_model(np.zeros((1,MAX_FRAMES,FEATURE_DIM), dtype=np.float32), training=False)
        state.word_model.load_weights(WORD_WEIGHTS)
        msgs.append("Kelime:OK")
    else:
        msgs.append("Kelime:YOK")

    try:
        state.whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        msgs.append("Whisper:OK")
    except Exception:
        msgs.append("Whisper:HATA")

    if GROQ_API_KEY:
        state.groq_client = Groq(api_key=GROQ_API_KEY)
        msgs.append("Groq:OK")
    else:
        msgs.append("Groq:YOK")

    sio.emit("system_status", {"msg": "  |  ".join(msgs), "ok": True})
    print("Modeller:", "  |  ".join(msgs))

# ── KAMERA & TAHMİN ───────────────────────────────────────────────────────────
def _get_crop(frame, result):
    if not result.multi_hand_landmarks: return None
    h, w = frame.shape[:2]
    all_x, all_y = [], []
    for hand_lm in result.multi_hand_landmarks:
        for lm in hand_lm.landmark:
            all_x.append(lm.x*w); all_y.append(lm.y*h)
    bw=max(all_x)-min(all_x); bh=max(all_y)-min(all_y)
    x1=max(0,int(min(all_x)-bw*PADDING)); y1=max(0,int(min(all_y)-bh*PADDING))
    x2=min(w,int(max(all_x)+bw*PADDING)); y2=min(h,int(max(all_y)+bh*PADDING))
    bw2=x2-x1; bh2=y2-y1
    if bw2>bh2: d=bw2-bh2; y1=max(0,y1-d//2); y2=min(h,y2+d//2)
    else:        d=bh2-bw2; x1=max(0,x1-d//2); x2=min(w,x2+d//2)
    if x2-x1<20 or y2-y1<20: return None
    crop=cv2.resize(frame[y1:y2,x1:x2],(IMG_SIZE,IMG_SIZE))
    return cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),(x1,y1,x2,y2)

def _extract_word_kp(result):
    left_kp=np.zeros(63,dtype=np.float32); right_kp=np.zeros(63,dtype=np.float32)
    if result.multi_hand_landmarks:
        for hand_lm, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            label=hand_info.classification[0].label
            coords=np.array([[lm.x,lm.y,lm.z] for lm in hand_lm.landmark],dtype=np.float32).flatten()
            if label=="Left": left_kp=coords
            else: right_kp=coords
    def norm(kp):
        if np.any(kp!=0):
            pts=kp.reshape(21,3); pts=pts-pts[0]
            return (pts/(np.max(np.abs(pts))+1e-6)).flatten()
        return kp
    return np.concatenate([norm(left_kp),norm(right_kp)])

def _motion_score():
    if len(state.frame_buffer)<15: return 0.0
    frames=np.array(list(state.frame_buffer)[-15:])
    nonzero=frames[np.any(frames!=0,axis=1)]
    if len(nonzero)<5: return 0.0
    return float(np.mean(np.abs(np.diff(nonzero,axis=0))))

def _draw_overlay(frame, hands_on):
    h, w = frame.shape[:2]
    with state._pred_lock:
        pred = state.cur_pred; conf = state.cur_conf; t5 = list(state.top5)

    # Şeritler
    ov=frame.copy(); cv2.rectangle(ov,(0,0),(w,32),(8,8,18),-1)
    cv2.addWeighted(ov,0.85,frame,0.15,0,frame)
    ov2=frame.copy(); cv2.rectangle(ov2,(0,h-36),(w,h),(8,8,18),-1)
    cv2.addWeighted(ov2,0.80,frame,0.20,0,frame)

    # Top-5
    if t5 and hands_on:
        for ri,(lbl,p) in enumerate(t5):
            col=(0,230,140) if ri==0 and p>=THRESHOLD else (140,140,140)
            cv2.putText(frame,f"{lbl}:{p*100:.0f}%",(w-105,16+ri*14),
                        cv2.FONT_HERSHEY_SIMPLEX,0.32,col,1)

    # Progress bar
    if state.mode=="harf":
        rc=(len(state.recent_preds)
            if len(state.recent_preds)>0 and len(set(state.recent_preds))==1 else 0)
        if rc>0:
            cv2.rectangle(frame,(0,h-6),(int(w*rc/CONFIRM_N),h),(0,140,220),-1)

    # Durum
    if pred and conf>=THRESHOLD: txt=f"{pred.upper()}  {conf*100:.0f}%"; col=(0,230,140)
    elif pred:                   txt=f"{pred.upper()} ({conf*100:.0f}%)"; col=(0,130,210)
    elif not hands_on:           txt="El yok"; col=(80,80,200)
    else:                        txt="Bekleniyor..."; col=(110,110,110)
    cv2.putText(frame,txt,(8,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
    cv2.putText(frame,f"MOD:{state.mode.upper()}",(8,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.4,(140,140,190),1)


def _harf_update(lbl, p, t5):
    state._predict_busy = False
    with state._pred_lock:
        state.cur_pred=lbl; state.cur_conf=p; state.top5=t5

    if p>=THRESHOLD: state.recent_preds.append(lbl)
    else:            state.recent_preds.clear()

    # Overlay güncelle
    sio.emit("pred_update", {"pred": lbl, "conf": round(p,3),
                              "top5": [[l,round(c,3)] for l,c in t5]})

    if len(state.recent_preds)==CONFIRM_N and len(set(state.recent_preds))==1:
        char=state.recent_preds[-1]
        if char=="del":
            if state.harf_buffer: state.harf_buffer=state.harf_buffer[:-1]
        else:
            state.harf_buffer+=char.lower()
        state.last_conf_time=time.time()
        state.recent_preds.clear()
        sio.emit("buffer_update", {"text": state.harf_buffer, "mode": "harf"})

def _kelime_update(lbl, p, t5):
    state._word_predict_busy = False
    with state._pred_lock:
        state.cur_pred=lbl; state.cur_conf=p; state.top5=t5

    state.word_recent_preds.append(lbl)
    sio.emit("pred_update", {"pred": lbl, "conf": round(p,3),
                              "top5": [[l,round(c,3)] for l,c in t5]})

    if (len(state.word_recent_preds)==WORD_CONFIRM_N and
            len(set(state.word_recent_preds))==1 and p>=WORD_THRESHOLD):
        word=state.word_recent_preds[-1]; now=time.time()
        if word!=state.last_word or (now-state.last_word_time)>=WORD_COOLDOWN:
            state.kelime_buffer.append(word)
            state.last_word=word; state.last_word_time=now
            state.word_recent_preds.clear()
            sio.emit("buffer_update",{"text":" ".join(state.kelime_buffer),"mode":"kelime"})

def camera_loop():
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    mp_hands=mp.solutions.hands; mp_draw=mp.solutions.drawing_utils
    hands_det=mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.6,
                              min_tracking_confidence=0.5,model_complexity=1)
    while state.running:
        ret,frame=cap.read()
        if not ret: time.sleep(0.01); continue
        if FLIP: frame=cv2.flip(frame,1)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result=hands_det.process(rgb)
        hands_on=result.multi_hand_landmarks is not None

        if result.multi_hand_landmarks:
            for hand_lm,hand_info in zip(result.multi_hand_landmarks,result.multi_handedness):
                side=hand_info.classification[0].label
                col=(0,200,255) if side=="Right" else (255,150,0)
                mp_draw.draw_landmarks(frame,hand_lm,mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=col,thickness=2,circle_radius=3),
                    mp_draw.DrawingSpec(color=(255,255,255),thickness=1))

        # Tahmin
        if state.mode=="harf" and state.cnn_model:
            _harf_step(frame, result, hands_on)
        elif state.mode=="kelime" and state.word_model:
            _kelime_step(frame, result, hands_on)

        _draw_overlay(frame, hands_on)

        # JPEG encode → browser stream
        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with state.cam_frame_lock:
            state.cam_frame_jpg = jpg.tobytes()

        time.sleep(0.015)
    cap.release()

def _harf_step(frame, result, hands_on):
    now=time.time(); in_cd=(now-state.last_conf_time)<COOLDOWN_SEC
    if not hands_on or in_cd:
        with state._pred_lock: state.cur_pred=None; state.cur_conf=0.0
        if not in_cd: state.recent_preds.clear()
        return
    cr=_get_crop(frame,result)
    if not cr: return
    crop,bbox=cr
    bc=(0,255,160) if (state.cur_pred and state.cur_conf>=THRESHOLD) else (200,200,200)
    cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),bc,2)
    if state._predict_busy: return
    state._predict_busy=True
    inp=crop.astype(np.float32)/255.0
    def _run():
        probs=state.cnn_model.predict(inp[np.newaxis],verbose=0)[0]
        ti=int(np.argmax(probs)); p=float(probs[ti])
        lbl=state.cnn_labels.get(ti,"?")
        t5=[(state.cnn_labels.get(i,"?"),float(probs[i])) for i in np.argsort(probs)[::-1][:5]]
        _harf_update(lbl,p,t5)
    threading.Thread(target=_run,daemon=True).start()

def _kelime_step(frame, result, hands_on):
    kp=_extract_word_kp(result)
    state.frame_buffer.append(kp); state.frame_count+=1
    is_moving=_motion_score()>MOTION_THRESH

    if not is_moving:
        if (len(state.word_recent_preds)>=WORD_CONFIRM_N and
                len(set(state.word_recent_preds))==1):
            word=list(state.word_recent_preds)[0]; now=time.time()
            with state._pred_lock: conf=state.cur_conf; pred=state.cur_pred
            if (pred==word and conf>=WORD_THRESHOLD and
                    (word!=state.last_word or (now-state.last_word_time)>=WORD_COOLDOWN)):
                state.kelime_buffer.append(word)
                state.last_word=word; state.last_word_time=now
                sio.emit("buffer_update",{"text":" ".join(state.kelime_buffer),"mode":"kelime"})
        with state._pred_lock: state.cur_pred=None; state.cur_conf=0.0
        state.word_recent_preds.clear(); return

    if (len(state.frame_buffer)==MAX_FRAMES and
            state.frame_count%PREDICT_EVERY==0 and not state._word_predict_busy):
        state._word_predict_busy=True
        seq=np.array(list(state.frame_buffer),dtype=np.float32)[np.newaxis]
        def _run():
            probs=state.word_model.predict(seq,verbose=0)[0]
            ti=int(np.argmax(probs)); p=float(probs[ti])
            lbl=state.word_labels.get(ti,"?")
            t5=[(state.word_labels.get(i,"?"),float(probs[i])) for i in np.argsort(probs)[::-1][:5]]
            _kelime_update(lbl,p,t5)
        threading.Thread(target=_run,daemon=True).start()

# ── SES & WHİSPER ─────────────────────────────────────────────────────────────
def ses_callback(indata, frames, time_info, status):
    # Çift kontrol: hem recording hem sag_state == kayit olmalı
    if state.recording and state.sag_state == "kayit":
        state.audio_frames.append(indata.copy())

def start_audio():
    state.recording = False
    state.audio_frames = []
    state.stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1,
        dtype='float32', callback=ses_callback)
    state.stream.start()

def whisper_isle():
    # Hemen snapshot al ve state'i temizle — işlem sürerken yeni ses birikmesin
    audio_snapshot = list(state.audio_frames)
    state.audio_frames = []
    state.recording = False   # Garanti: kayıt kesin kapalı

    if not audio_snapshot or not state.whisper:
        state.sag_state="bekle"; sio.emit("sag_state",{"state":"bekle"}); return

    audio_data = np.concatenate(audio_snapshot, axis=0)
    print(f"Whisper: {len(audio_snapshot)} frame, {len(audio_data)/SAMPLE_RATE:.1f}s ses")
    with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as f: tmp=f.name
    sf.write(tmp,audio_data,SAMPLE_RATE)
    segs,_=state.whisper.transcribe(tmp,language=LANGUAGE)
    metin=" ".join(s.text.strip() for s in segs).strip()
    os.unlink(tmp)
    sio.emit("whisper_result",{"text":metin})
    if metin:
        # Normal birey mesajı → sağ balon
        _add_message("normal", metin)
        animler=_metni_animasyona(metin)
        for a in animler: state.anim_queue.put(a)
        state.sag_state="oynatma"
        sio.emit("sag_state",{"state":"oynatma"})
        # Animasyon thread
        threading.Thread(target=_play_animations,daemon=True).start()
    else:
        state.sag_state="bekle"; sio.emit("sag_state",{"state":"bekle"})

def _play_animations():
    """
    Animasyon kuyruğunu sırayla oynatır.
    Düzeltmeler:
      1. queue.empty() race condition → get(timeout) ile güvenli okuma
      2. Harf videosu yoksa tarayıcıya text emit edilir (0.3s beklemek yerine)
      3. VideoCapture her durumda release() edilir (try/finally)
      4. Kuyruk dışarıdan temizlenirse (sıfırla) gracefully çıkar
    """
    try:
        while True:
            # Timeout ile al — kuyruk boşsa queue.Empty fırlatır, döngü biter
            try:
                tip, icerik, yol = state.anim_queue.get(timeout=0.5)
            except queue.Empty:
                break

            sio.emit("anim_play", {"tip": tip, "label": icerik, "yol": yol or ""})

            if yol and os.path.exists(yol):
                cap2 = cv2.VideoCapture(yol)
                try:
                    fps = cap2.get(cv2.CAP_PROP_FPS) or 30
                    frame_delay = 1.0 / fps
                    while cap2.isOpened():
                        ret, frm = cap2.read()
                        if not ret:
                            break
                        frm = cv2.resize(frm, (480, 360))
                        _, jpg = cv2.imencode('.jpg', frm, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        b64 = base64.b64encode(jpg.tobytes()).decode()
                        sio.emit("anim_frame", {"data": b64, "label": icerik})
                        time.sleep(frame_delay)
                finally:
                    cap2.release()  # Her durumda kamera serbest bırakılır
            else:
                if tip == "harf":
                    # Video yoksa harfi text olarak tarayıcıya gönder
                    sio.emit("anim_harf", {"harf": icerik})
                    time.sleep(0.5)  # Harfin okunması için yeterli süre
                else:
                    # Kelime videosu bulunamadı, kısa bekleme
                    sio.emit("anim_harf", {"harf": icerik, "tip": "kelime_yok"})
                    time.sleep(0.4)

    finally:
        state.sag_state = "bekle"
        sio.emit("sag_state", {"state": "bekle"})

# ── LLM & ANIMASYON ───────────────────────────────────────────────────────────
_SORU_PROMPT=(
    "Sen bir Turk isaret dili cevirmenisin.\n"
    "Sana isaret dili kelimeleri verilecek. Kisi SORU soruyor.\n"
    "Gorev: Kelimeleri dogal Turkce SORU cumlesine cevir, soru isareti koy.\n"
    "Sadece sonuc cumleyi yaz, baska hicbir sey yazma.\n"
    "Ornekler:\n"
    "Girdi: sen anne sevmek -> Cikti: Sen anneni seviyor musun?\n"
    "Girdi: ben anne sevmek -> Cikti: ben annemi seviyor muyum?\n"
    "Girdi: saat -> Cikti: Saat kac?\n"
    "Girdi: ev nerede -> Cikti: Ev nerede?\n"
    "Girdi: sen iyi -> Cikti: Iyi misin?"
    "Girdi: ben iyi -> Cikti: Iyi miyim?"
)
 
_SYSTEM_PROMPT=(
    "Sen bir Turk isaret dili (TID) cevirmenisin.\n"
    "Sana kelime kelime yazilmis isaret dili metni verilecek.\n"
    "Asagidaki kurallari uygulayarak dogal Turkce cumle uret.\n"
    "Sadece sonuc cumleyi yaz, baska hicbir sey yazma.\n"
    "\n"
    "KURALLAR:\n"
    "1. ZAMAN BELIRTECI: 'dun' varsa cumle basina al.\n"
    "   Ornek: 'dun baba kacmak' -> 'Dun babam kacti.'\n"
    "2. TEK SOSYAL KELIMELER tam cumleye donusur:\n"
    "   'selam' -> 'Merhaba!' | 'tesekkur' -> 'Tesekkur ederim.' | 'hoscakal' -> 'Gule gule.'\n"
    "3. SORU KELIMELERI (nerede, neden, ne, kim, kac) iceren cumle otomatik soru olur.\n"
    "   Ornek: 'saat nerede' -> 'Saat nerede?' | 'neden kacmak' -> 'Neden kactin?'\n"
    "4. EVET/HAYIR tepki cumleleridir:\n"
    "   'evet iyi' -> 'Evet, iyiyim.' | 'hayir' -> 'Hayir.'\n"
    "5. OZNE YOK ise 'ben' varsay:\n"
    "   'iyi' -> 'Iyiyim.' | 'dolu' -> 'Dolu.' | 'calismak' -> 'Calisiyorum.'\n"
    "6. EYLEM + NESNE:\n"
    "   'alisveris yapmak' -> 'Alisveris yapiyorum.' | 'telefon beklemek' -> 'Telefon bekliyorum.'\n"
    "7. KELIME SIRASI: TID'de zaman once gelir, eylem sona gelir.\n"
    "   'ben anne sevmek' -> 'Ben annemi seviyorum.'\n"
    "   'sen anne sevmek' -> 'Sen anneni seviyorsun.'\n"
    "   'onlar okul gitmek' -> 'Onlar okula gidiyor.'\n"
    "8. SAHIPLIK: kisi + nesne -> ilgi eki ekle:\n"
    "   'baba telefon' -> 'Babanin telefonu.' | 'anne ev' -> 'Annenin evi.'\n"
    "\n"
    "Ornekler:\n"
    "Girdi: selam ben iyi -> Cikti: Merhaba, ben iyiyim.\n"
    "Girdi: ben anne sevmek -> Cikti: Ben annemi seviyorum.\n"
    "Girdi: ev nerede -> Cikti: Ev nerede?\n"
    "Girdi: dun baba kacmak -> Cikti: Dun babam kacti.\n"
    "Girdi: saat kac -> Cikti: Saat kac?\n"
    "Girdi: tesekkur -> Cikti: Tesekkur ederim.\n"
    "Girdi: onlar tamam -> Cikti: Onlar tamam.\n"
    "Girdi: calismak neden -> Cikti: Neden calisiyorsun?"

    "9. bu conversation ı bu şekilde kullan" """  {
        "satir": "Selam anne, ev dolu mu?",
        "kelimeler": ["selam", "anne", "ev", "dolu", "mu"],
        "eslesmeler": ["selam", "anne", "ev", "dolu", None, None]
    },
    {
        "satir": "Evet, baba ve çocuk evde. Sen okuldan iyi geldin mi?",
        "kelimeler": ["evet", "baba", "ve", "çocuk", "evde", "sen", "okuldan", "iyi", "geldin", "mi"],
        "eslesmeler": ["evet", "baba", None, "çocuk", "ev", "sen", "okul", "iyi", None]
    },
    {
        "satir": "Evet anne, ben bekledim.",
        "kelimeler": ["evet", "anne", "ben", "bekledim"],
        "eslesmeler": ["evet", "anne", "ben", "beklemek"]
    },
    {
        "satir": "Telefon yoktu.",
        "kelimeler": ["telefon", "yoktu"],
        "eslesmeler": ["telefon", "hayır"]
    },
        {
        "satir": "babam nerede.",
        "kelimeler": ["babam", "nerede"],
        "eslesmeler": ["baba", "nerede"]
    },
    {
        "satir": "onlar alışverişte.",
        "kelimeler": ["onlar", "alışverişte"],
        "eslesmeler": ["onlar", "alışveriş"]
    },
    {
        "satir": "Tamam. Sen evde bekle, kaçmak yok.",
        "kelimeler": ["tamam", "sen", "evde", "bekle", "kaçmak", "yok"],
        "eslesmeler": ["tamam", "sen", "ev", "beklemek", "kaçmak", "hayır"]
    },
    {
        "satir": "Tamam anne, ben evdeyim.",
        "kelimeler": ["tamam", "anne", "ben", "evdeyim"],
        "eslesmeler": ["tamam", "anne", "ben", "ev"]
    },
    {
        "satir": "İyi. Okul nasıl?",
        "kelimeler": ["iyi", "okul", "nasıl"],
        "eslesmeler": ["iyi", "okul", None]
    },
    {
        "satir": "İyi. Ben memnunum.",
        "kelimeler": ["iyi", "ben", "memnunum"],
        "eslesmeler": ["iyi", "ben", "memnun_olmak"]
    },
    {
        "satir": "Sen çalış, tamam mı?",
        "kelimeler": ["sen", "çalış", "tamam", "mı"],
        "eslesmeler": ["sen", "çalışmak", "tamam", None]
    },
    {
        "satir": "Tamam anne. Teşekkürler.",
        "kelimeler": ["tamam", "anne", "teşekkürler"],
        "eslesmeler": ["tamam", "anne", "teşekkür"]
    },
    {
        "satir": "Teşekkürler, hoşça kal.",
        "kelimeler": ["teşekkürler", "hoşça kal"],
        "eslesmeler": ["teşekkür", "hoşçakal"]
    } """


)

def groq_isle(metin, mod):
    if not state.groq_client: return metin
    if mod=="kelime":
        kelimeler=metin.split(); temiz=[]
        for k in kelimeler:
            if not temiz or temiz[-1]!=k: temiz.append(k)
        user_msg=f"Kelimeler: {' '.join(temiz)}"
    else:
        user_msg=f"Harfler: {metin}"
    try:
        r=state.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"system","content":_SORU_PROMPT if state.llm_intent=="soru" else _SYSTEM_PROMPT},
                      {"role":"user","content":user_msg}],
            max_tokens=200,temperature=0.3)
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM hatasi: {e}"); return metin

def _normalize(s):
    import unicodedata
    return unicodedata.normalize("NFC",s.lower().strip())

def _get_kelime_listesi():
    import unicodedata
    kd=os.path.join(ANIMATIONS_DIR,"kelimeler")
    if not os.path.exists(kd): return []
    return [unicodedata.normalize("NFC",f).lower().replace(".mp4","")
            for f in os.listdir(kd) if f.lower().endswith(".mp4")]

# ── FEW-SHOT ÖRNEKLER (konuşmadan türetilmiş) ────────────────────────────────
_ESLESME_ORNEKLER = [
        {
        "satir": "Evet, baba ve çocuk evde. Sen okuldan iyi geldin mi?",
        "kelimeler": ["evet", "baba", "ve", "çocuk", "evde", "sen", "okuldan", "iyi", "geldin", "mi"],
        "eslesmeler": ["evet", "baba", None, "çocuk", "ev", "sen", "okul", "iyi", None]
    },
    {
        "satir": "Evet anne, ben bekledim.",
        "kelimeler": ["evet", "anne", "ben", "bekledim"],
        "eslesmeler": ["evet", "anne", "ben", "beklemek"]
    },
    {
        "satir": "Telefon yoktu.",
        "kelimeler": ["telefon", "yoktu"],
        "eslesmeler": ["telefon", "hayır"]
    },
        {
        "satir": "babam nerede.",
        "kelimeler": ["babam", "nerede"],
        "eslesmeler": ["baba", "nerede"]
    },
    {
        "satir": "onlar alışverişte.",
        "kelimeler": ["onlar", "alışverişte"],
        "eslesmeler": ["onlar", "alışveriş"]
    },
    {
        "satir": "Tamam. Sen evde bekle, kaçmak yok.",
        "kelimeler": ["tamam", "sen", "evde", "bekle", "kaçmak", "yok"],
        "eslesmeler": ["tamam", "sen", "ev", "beklemek", "kaçmak", "hayır"]
    },
    {
        "satir": "tamam anne, ben evdeyim.",
        "kelimeler": ["tamam", "anne", "ben", "evdeyim"],
        "eslesmeler": ["tamam", "anne", "ben", "ev"]
    },
    {
        "satir": "İyi. Okul nasıl?",
        "kelimeler": ["iyi", "okul", "nasıl"],
        "eslesmeler": ["iyi", "okul", None]
    },
    {
        "satir": "İyi. Ben memnunum.",
        "kelimeler": ["iyi", "ben", "memnunum"],
        "eslesmeler": ["iyi", "ben", "memnun_olmak"]
    },
    {
        "satir": "Sen çalış, tamam mı?",
        "kelimeler": ["sen", "çalış", "tamam", "mı"],
        "eslesmeler": ["sen", "çalışmak", "tamam", None]
    },
    {
        "satir": "Tamam anne. Teşekkürler.",
        "kelimeler": ["tamam", "anne", "teşekkürler"],
        "eslesmeler": ["tamam", "anne", "teşekkür"]
    },
    {
        "satir": "Teşekkürler, hoşça kal.",
        "kelimeler": ["teşekkürler", "hoşça kal"],
        "eslesmeler": ["teşekkür", "hoşçakal"]
    } 
]

def _build_ornek_metin():
    """Few-shot ornekleri prompt icin okunabilir metne cevirir."""
    satirlar = []
    for ornek in _ESLESME_ORNEKLER:
        kelime_str = ", ".join(ornek["kelimeler"])
        eslesme_str = ", ".join(
            (e if e is not None else "null") for e in ornek["eslesmeler"]
        )
        satir = (
            '  Cumle: "' + ornek["satir"] + '"\n' +
            '  Kelimeler: [' + kelime_str + ']\n' +
            '  Eslesmeler: [' + eslesme_str + ']'
        )
        satirlar.append(satir)
    return "\n\n".join(satirlar)

def _llm_eslestime(metin, kelimeler, kelime_listesi):
    if not state.groq_client or not kelime_listesi: return None
    liste_str = ", ".join(kelime_listesi)
    numbered = "\n".join(f"{i+1}. {w}" for i, w in enumerate(kelimeler))
    ornek_metin = _build_ornek_metin()
    prompt = (
        f"Mevcut animasyon dosyaları (SADECE bunlardan seç): {liste_str}\n\n"
        f"Eşleştirilecek kelimeler:\n{numbered}\n\n"
        "Her kelime için yukarıdaki animasyon listesinden en uygun olanı seç.\n"
        "Kurallar:\n"
        "1. Türkçe çekim/yapım/iyelik eklerini çıkar, kök kelimeyi bul:\n"
        "   - İyelik: \"babam\"→\"baba\", \"evdeyim\"→\"ev\", \"annemi\"→\"anne\"\n"
        "   - Hal eki: \"evde\"→\"ev\", \"okuldan\"→\"okul\", \"alışverişte\"→\"alışveriş\"\n"
        "   - Fiil çekimi: \"bekledim\"→\"beklemek\", \"çalış\"→\"çalışmak\", \"memnunum\"→\"memnun_olmak\"\n"
        "   - Çoğul/ek: \"teşekkürler\"→\"teşekkür\"\n"
        "2. Kök listede varsa o dosyayı seç\n"
        "3. Listede karşılık yoksa null yaz (bağlaçlar, soru ekleri vb.)\n"
        "4. SADECE JSON array döndür, başka hiçbir şey yazma\n"
        f"5. Array tam {len(kelimeler)} elemanlı olmalı\n"
        "6. Aynı animasyonu arka arkaya kullanma\n\n"
        f"Referans örnekler (bu konuşmadan alınmıştır):\n{ornek_metin}"
    )
    try:
        r = state.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Sadece geçerli JSON array döndür. Başka hiçbir şey yazma."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=400, temperature=0.0)
        raw = r.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        if isinstance(parsed, list): return parsed
    except Exception as e:
        print(f"LLM eslesme hatasi: {e}")
    return None

def _metni_animasyona(metin):
    kd = os.path.join(ANIMATIONS_DIR, "kelimeler")
    hd = os.path.join(ANIMATIONS_DIR, "harfler")
    kelimeler = [w for w in metin.lower().split() if ''.join(c for c in w if c.isalpha())]
    kelime_listesi = _get_kelime_listesi()
    eslesmeler = _llm_eslestime(metin, kelimeler, kelime_listesi)

    # LLM basarisiz olduysa veya eleman sayisi tutarsizsa fallback
    if eslesmeler is None or len(eslesmeler) != len(kelimeler):
        eslesmeler = [None] * len(kelimeler)
        for i, kelime in enumerate(kelimeler):
            temiz = _normalize(''.join(c for c in kelime if c.isalpha()))

            # 1. Birebir esleme
            for da in kelime_listesi:
                if _normalize(da) == temiz:
                    eslesmeler[i] = da
                    break

            # 2. Kok esleme - listede olan en uzun kelimeyle basliyorsa
            # (iyelik/cekim ekleri icin: "evde"->"ev", "annemi"->"anne")
            if eslesmeler[i] is None:
                for da in sorted(kelime_listesi, key=len, reverse=True):
                    da_norm = _normalize(da)
                    if temiz.startswith(da_norm) and len(da_norm) >= 2:
                        eslesmeler[i] = da
                        break

    animler = []
    for kelime, eslesme in zip(kelimeler, eslesmeler):
        temiz = ''.join(c for c in kelime if c.isalpha())
        if not temiz:
            continue
        if eslesme:
            yol = None
            if os.path.exists(kd):
                import unicodedata
                en = _normalize(eslesme)
                for f in os.listdir(kd):
                    if _normalize(f.replace(".mp4", "").replace(".MP4", "")) == en:
                        yol = os.path.join(kd, f)
                        break
            if yol:
                animler.append(("kelime", eslesme, yol))
                continue
        # Video bulunamadi -> harf harf yaz
        for h in temiz:
            hp = os.path.join(hd, f"{h.upper()}.mp4")
            animler.append(("harf", h, hp if os.path.exists(hp) else None))
    return animler

def _add_message(sender, text):
    """Sohbet geçmişine ekle ve tüm bağlı tarayıcılara yayınla."""
    import datetime
    msg={"sender":sender,"text":text,
         "time":datetime.datetime.now().strftime("%H:%M")}
    state.chat_history.append(msg)
    sio.emit("new_message", msg)

def tts(metin):
    def _t():
        try: subprocess.run(["say","-v","Yelda",metin])
        except: subprocess.run(["say",metin])
    threading.Thread(target=_t,daemon=True).start()

# ── FLASK ROUTES ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    def gen():
        while state.running:
            with state.cam_frame_lock:
                jpg = state.cam_frame_jpg
            if jpg:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
            time.sleep(0.015)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ── SOCKETIO EVENTS ───────────────────────────────────────────────────────────
@sio.on("connect")
def on_connect():
    # Client bağlandığında (sayfa yenileme dahil) kayıt durumunu sıfırla
    # Böylece eski kayıt "bekle" butonu basılmadan devam etmez
    if state.recording or state.sag_state == "kayit":
        state.recording    = False
        state.sag_state    = "bekle"
        state.audio_frames = []
        print(">>> Client yeniden bağlandı — kayıt sıfırlandı")

    emit("chat_history", state.chat_history)
    emit("mode_change", {"mode": state.mode})
    emit("sag_state", {"state": state.sag_state})
    emit("intent_change", {"intent": state.llm_intent})

@sio.on("disconnect")
def on_disconnect():
    # Client koptuğunda aktif kayıt varsa durdur
    if state.recording or state.sag_state == "kayit":
        state.recording    = False
        state.sag_state    = "bekle"
        state.audio_frames = []
        print(">>> Client koptu — kayıt durduruldu")

@sio.on("set_mode")
def on_set_mode(data):
    state.mode = data.get("mode","harf")
    state.harf_buffer=""; state.kelime_buffer.clear()
    state.recent_preds.clear(); state.word_recent_preds.clear()
    with state._pred_lock: state.cur_pred=None; state.cur_conf=0.0; state.top5=[]
    sio.emit("mode_change",{"mode":state.mode})
    sio.emit("buffer_update",{"text":"","mode":state.mode})

@sio.on("llm_isle")
def on_llm_isle(data=None):
    metin=(state.harf_buffer if state.mode=="harf" else " ".join(state.kelime_buffer))
    if not metin: return
    sio.emit("llm_status",{"status":"thinking"})
    def _t():
        sonuc=groq_isle(metin,state.mode)
        # Biriken buffer'ı sıfırla
        state.harf_buffer=""; state.kelime_buffer.clear()
        state.recent_preds.clear(); state.word_recent_preds.clear()
        sio.emit("llm_status",{"status":"done","text":sonuc})
        sio.emit("buffer_update",{"text":"","mode":state.mode})
        # İşitme engelli birey mesajı → sol balon
        _add_message("engelli", sonuc)
        tts(sonuc)
    threading.Thread(target=_t,daemon=True).start()

@sio.on("set_intent")
def on_set_intent(data):
    state.llm_intent = data.get("intent","ifade")
    sio.emit("intent_change",{"intent":state.llm_intent})

@sio.on("temizle")
def on_temizle(data=None):
    state.harf_buffer=""; state.kelime_buffer.clear()
    state.recent_preds.clear(); state.word_recent_preds.clear()
    with state._pred_lock: state.cur_pred=None; state.cur_conf=0.0; state.top5=[]
    sio.emit("buffer_update",{"text":"","mode":state.mode})

@sio.on("backspace")
def on_backspace(data=None):
    if state.mode=="harf" and state.harf_buffer:
        state.harf_buffer=state.harf_buffer[:-1]
    elif state.mode=="kelime" and state.kelime_buffer:
        state.kelime_buffer.pop()
    sio.emit("buffer_update",{
        "text": state.harf_buffer if state.mode=="harf" else " ".join(state.kelime_buffer),
        "mode": state.mode})

@sio.on("kayit_baslat")
def on_kayit_baslat(data=None):
    if state.sag_state != "bekle": return
    state.audio_frames = []       # Eski ses verisini KESİNLİKLE temizle
    state.recording    = True
    state.sag_state    = "kayit"
    print(f">>> KAYIT BASLADI (önceki frames silindi)")
    sio.emit("sag_state", {"state": "kayit"})

@sio.on("kayit_bitir")
def on_kayit_bitir(data=None):
    if state.sag_state != "kayit": return
    state.recording = False       # ÖNCE kaydı durdur
    state.sag_state = "isleniyor"
    print(f">>> KAYIT DURDU ({len(state.audio_frames)} frame yakalandı)")
    sio.emit("sag_state", {"state": "isleniyor"})
    threading.Thread(target=whisper_isle, daemon=True).start()

@sio.on("sifirla")
def on_sifirla(data=None):
    state.recording    = False
    state.sag_state    = "bekle"
    state.audio_frames = []
    while not state.anim_queue.empty():
        try: state.anim_queue.get_nowait()
        except: break
    sio.emit("sag_state", {"state": "bekle"})

# ── BAŞLATMA ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=load_models,  daemon=True).start()
    threading.Thread(target=camera_loop, daemon=True).start()
    start_audio()
    print("\n✅ TID Sunucu başlatıldı!")
    print("   Tarayıcıda aç: http://localhost:5050\n")
    sio.run(app, host="0.0.0.0", port=5050, debug=False)