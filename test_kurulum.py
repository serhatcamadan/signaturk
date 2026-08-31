# test_kurulum.py adıyla kaydet ve python test_kurulum.py ile çalıştır
import sys
print(f"Python: {sys.version}")
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPU aygıtları: {tf.config.list_physical_devices()}")

import mediapipe as mp
print(f"MediaPipe: {mp.__version__}")
import cv2
print(f"OpenCV: {cv2.__version__}")
import numpy as np
print(f"NumPy: {np.__version__}")
print("\n✅ Tüm kütüphaneler başarıyla yüklendi!")