# 보충 자료 및 구현 세부사항

## 1. MoveNet 모델 다운로드 및 변환 가이드

### 옵션 1: Kaggle에서 직접 다운로드
```bash
# Kaggle CLI 설치
pip install kaggle

# 모델 다운로드
kaggle models instances versions google/movenet/1
```

### 옵션 2: TensorFlow Hub에서 다운로드
```python
# download_movenet.py
import tensorflow_hub as hub
import tensorflow as tf

# MoveNet 모델 URL
model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"

# 모델 로드 (자동 다운로드)
model = hub.load(model_url)

# TFLite로 변환
converter = tf.lite.TFLiteConverter.from_saved_model("movenet_model")
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()

# 저장
with open('./models/movenet_singlepose_lightning.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ MoveNet TFLite 변환 완료!")
```

### 옵션 3: Google Coral 최적화 버전
```bash
# Coral 보드용 최적화된 MoveNet (RPi 호환)
# https://github.com/google-coral/examples-camera/tree/master/ml/pose_estimation
```

---

## 2. 오디오 모델 재학습 완전 가이드

### 단계 1: 학습 데이터 수집

#### 데이터셋 구조
```
audio_data/
├── scream/              # 비명 (600개 샘플)
│   ├── 001.wav (1초, 16kHz)
│   ├── 002.wav
│   └── ...
├── help/                # 도움 신호 (400개 샘플)
│   ├── 001.wav
│   └── ...
├── words/               # 특정 단어 (도움, 119 등)
│   ├── help_001.wav
│   └── ...
└── background/          # 거짓 양성 방지 (900개)
    ├── traffic.wav
    ├── music.wav
    ├── speech.wav
    └── ...
```

#### 데이터 특성 (중요!)
- **샘플링 레이트**: 16kHz
- **비트 깊이**: 16-bit PCM
- **지속 시간**: 정확히 1초
- **형식**: WAV 파일

#### 데이터 수집 도구
```python
# record_training_audio.py
import sounddevice as sd
import scipy.io.wavfile as wavfile
import os

sample_rate = 16000
duration = 1  # 1초

def record_sample(category, number):
    print(f"Recording {category} sample {number}...")
    audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    
    os.makedirs(f'audio_data/{category}', exist_ok=True)
    filename = f'audio_data/{category}/{number:03d}.wav'
    wavfile.write(filename, sample_rate, audio)
    print(f"Saved: {filename}")

# 사용 예
# for i in range(1, 21):
#     record_sample('scream', i)
```

### 단계 2: 데이터 전처리

```python
# preprocess_audio.py
import numpy as np
import librosa
import os
from scipy.io import wavfile

def preprocess_audio_file(filename, target_sr=16000, duration=1.0):
    """오디오 파일을 모델 입력 형식으로 전처리"""
    
    # 오디오 로드
    y, sr = librosa.load(filename, sr=target_sr, duration=duration)
    
    # 정확히 1초로 패딩/자르기
    target_samples = int(target_sr * duration)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')
    else:
        y = y[:target_samples]
    
    # 정규화
    y = y / (np.abs(y).max() + 1e-8)
    
    # MFCC 특성 추출 (선택사항 - 더 나은 모델 성능)
    # mfcc = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=13)
    
    return y.reshape(-1, 1)  # [16000, 1] 형태

def prepare_dataset(data_dir='audio_data'):
    """전체 데이터셋 준비"""
    X = []
    y = []
    
    label_map = {'scream': 0, 'help': 1, 'words': 2, 'background': 3}
    
    for category, label in label_map.items():
        category_path = os.path.join(data_dir, category)
        if not os.path.exists(category_path):
            continue
        
        for filename in os.listdir(category_path):
            if filename.endswith('.wav'):
                filepath = os.path.join(category_path, filename)
                try:
                    audio = preprocess_audio_file(filepath)
                    X.append(audio)
                    y.append(label)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
    
    return np.array(X), np.array(y)

# 사용
# X, y = prepare_dataset()
# print(f"Dataset shape: {X.shape}, Labels: {np.unique(y)}")
```

### 단계 3: 신경망 모델 구축 및 학습

```python
# train_audio_model.py
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def build_audio_model(input_shape=(16000, 1)):
    """오디오 분류 신경망"""
    model = keras.Sequential([
        # Input
        keras.layers.Input(shape=input_shape),
        
        # Conv1D 블록 1
        keras.layers.Conv1D(64, 80, strides=4, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=4),
        keras.layers.Dropout(0.3),
        
        # Conv1D 블록 2
        keras.layers.Conv1D(128, 3, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=4),
        keras.layers.Dropout(0.3),
        
        # Conv1D 블록 3
        keras.layers.Conv1D(256, 3, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=4),
        keras.layers.Dropout(0.3),
        
        # Global pooling
        keras.layers.GlobalAveragePooling1D(),
        
        # Dense 블록
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        
        # Output (4개 클래스: scream, help, words, background)
        keras.layers.Dense(4, activation='softmax')
    ])
    
    return model

def train_model(X, y):
    """모델 학습"""
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 정규화
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)
    X_train = X_train_flat.reshape(X_train.shape)
    X_test = X_test_flat.reshape(X_test.shape)
    
    # 모델 빌드
    model = build_audio_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # 학습
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
    )
    
    # 평가
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return model, history

# 사용 예
# from preprocess_audio import prepare_dataset
# X, y = prepare_dataset()
# model, history = train_model(X, y)
```

### 단계 4: TFLite 변환 및 최적화

```python
# convert_to_tflite.py
import tensorflow as tf
import numpy as np

def convert_to_tflite(keras_model, output_path='models/keyword_audio.tflite'):
    """Keras 모델을 TFLite로 변환 및 최적화"""
    
    # 기본 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    
    # 최적화 옵션 (RPi 5 환경)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 양자화 (모델 크기 75% 감소, 약간의 정확도 손실)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # 변환
    tflite_model = converter.convert()
    
    # 저장
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # 파일 크기 확인
    file_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"✅ TFLite 모델 저장 완료!")
    print(f"   경로: {output_path}")
    print(f"   크기: {file_size_mb:.2f} MB")
    
    return tflite_model

# 사용
# tflite_model = convert_to_tflite(model)
```

### 단계 5: TFLite 모델 테스트

```python
# test_audio_tflite.py
from tflite_runtime.interpreter import Interpreter
import numpy as np

def test_tflite_model(model_path='models/keyword_audio.tflite', audio_file='test.wav'):
    """TFLite 모델 테스트"""
    
    # 인터프리터 로드
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"입력 shape: {input_details[0]['shape']}")
    print(f"출력 shape: {output_details[0]['shape']}")
    
    # 테스트 오디오 로드
    import librosa
    y, sr = librosa.load(audio_file, sr=16000, duration=1.0)
    y = (y / (np.abs(y).max() + 1e-8)).reshape(1, -1, 1).astype(np.float32)
    
    # 추론
    interpreter.set_tensor(input_details[0]['index'], y)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # 결과 해석
    label_names = ['Scream', 'Help', 'Words', 'Background']
    class_idx = np.argmax(output[0])
    confidence = output[0][class_idx]
    
    print(f"\n결과:")
    print(f"  클래스: {label_names[class_idx]}")
    print(f"  신뢰도: {confidence:.4f}")
    print(f"  전체 출력: {output[0]}")
    
    return output[0]

# 사용
# test_tflite_model()
```

---

## 3. RPi 5 성능 최적화

### A. 프로세스 우선순위 설정
```python
# main.py 수정
import os

def set_process_priority():
    """프로세스 우선순위 상향"""
    try:
        os.nice(-10)  # 높은 우선순위 (-20 ~ 19, 낮을수록 높음)
        print("[Main] 프로세스 우선순위 상향 설정")
    except:
        print("[Main] 우선순위 설정 실패 (sudo 필요할 수 있음)")

if __name__ == "__main__":
    set_process_priority()
    main()
```

### B. GPU 가속화 (TensorFlow Lite Delegate)
```python
# processors.py 수정
from tflite_runtime.interpreter import Interpreter, load_delegate

# GPU delegate 로드 (가능한 경우)
try:
    gpu_delegate = load_delegate('/usr/lib/libGPUDelegate.so')
    interpreter = Interpreter(
        model_path=config.MOVENET_MODEL_PATH,
        experimental_delegates=[gpu_delegate]
    )
    print("[Video] GPU 가속 활성화")
except:
    interpreter = Interpreter(model_path=config.MOVENET_MODEL_PATH)
    print("[Video] CPU 모드 실행")
```

### C. 메모리 최적화
```python
# RPi 5 시스템 설정
# /boot/firmware/cmdline.txt 에 추가:
# cgroup_enable=memory swapaccount=1
```

---

## 4. 데이터베이스 연동 (선택사항)

낙상 감지 기록을 저장하려면:

```python
# db_manager.py
import sqlite3
from datetime import datetime

class FallEventLogger:
    def __init__(self, db_path='fall_events.db'):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS fall_events (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            fall_detected BOOLEAN,
            keypoints_quality REAL,
            state TEXT
        )''')
        conn.commit()
        conn.close()
    
    def log_event(self, fall_detected, quality, state):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO fall_events 
                     (timestamp, fall_detected, keypoints_quality, state)
                     VALUES (?, ?, ?, ?)''',
                  (datetime.now().isoformat(), fall_detected, quality, state))
        conn.commit()
        conn.close()
```

---

## 5. 클라우드 연동 (선택사항)

알림 및 모니터링을 위한 클라우드 서비스:

```python
# cloud_notifier.py
import requests
import json

class CloudNotifier:
    def __init__(self, api_endpoint, api_key):
        self.endpoint = api_endpoint
        self.api_key = api_key
    
    def send_alert(self, message, severity='HIGH'):
        payload = {
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        
        headers = {'Authorization': f'Bearer {self.api_key}'}
        
        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Cloud notification failed: {e}")
            return False
```

---

## 6. 문제 해결 및 디버깅

### 로그 레벨 설정
```python
# 상세 로깅 추가 (config.py)
LOG_LEVEL = 'DEBUG'  # DEBUG, INFO, WARNING, ERROR

# processors.py에서 사용
import logging

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, config.LOG_LEVEL))
```

### 성능 프로파일링
```python
# profile_performance.py
import cProfile
import pstats

def profile_video_processor():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 비디오 처리 코드 실행
    # ...
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

---

## 📝 최종 체크리스트

### MoveNet 설치
- [ ] 모델 파일 다운로드 완료
- [ ] 모델 형식 확인 (TFLite)
- [ ] 입출력 shape 확인

### 오디오 모델 준비
- [ ] 학습 데이터 수집 (최소 1000개 샘플)
- [ ] 데이터 전처리 스크립트 실행
- [ ] 모델 학습 완료
- [ ] TFLite 변환 완료
- [ ] config.py에서 AUDIO_ENABLED = True

### 배포
- [ ] RPi 5에 라이브러리 설치
- [ ] 모델 파일 복사
- [ ] main.py 실행 및 테스트
- [ ] 성능 모니터링
