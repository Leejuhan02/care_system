# 독거노인 케어 시스템 - MoveNet 기반 낙상 감지 설정 가이드

## 📋 개요

이 프로젝트는 **MoveNet** 관절 감지 모델을 기반으로 한 고급 낙상 감지 시스템입니다. 기존의 분류 기반 모델(fall_detection.tflite)에서 관절 좌표 기반 분석으로 전환하여 더 정확한 낙상 감지를 제공합니다.

---

## 🔧 시스템 아키텍처

### 비디오 처리 (VideoProcessor)
- **모델**: MoveNet Lightning (192x192 입력)
- **출력**: 17개 관절의 좌표(x, y)와 신뢰도(confidence)
- **기능**: 관절 좌표 기반 낙상 감지

### 낙상 감지 알고리즘 (FallDetectionAnalyzer)

낙상을 판정하기 위해 3가지 조건을 분석합니다:

#### 1️⃣ **체격 비율 분석 (신체 너비/높이)**
```
ASPECT_RATIO = 신체_너비 / 신체_높이

- ASPECT_RATIO < 0.3 (아주 낮음)
  → 누워있거나 웅크린 자세 (낙상 의심)
  
- ASPECT_RATIO > 0.8 (높음)
  → 누워있거나 옆으로 누운 자세 (낙상 의심)
```
**감지 가능한 상황**: 
- 쓰러질 때 앉아있는 모습
- 웅크린 자세
- 눕는 동작

#### 2️⃣ **머리 위치 분석**
```
머리 Y좌표 > 골반 Y좌표 + 임계값

→ 머리가 골반보다 훨씬 낮은 상태
```
**감지 가능한 상황**:
- 물건을 주우려고 구부린 자세
- 넘어지면서 머리가 아래로 향한 자세

#### 3️⃣ **신체 안정성 분석 (기울기)**
```
기울기 = 양쪽 어깨 또는 엉덩이의 높이 차이

기울기 > 15% 
→ 불안정한 자세 (낙상 중)
```

#### 🎯 **최종 판정 로직**
```
fall_score = 0
if 체격비율_낙상신호:
    fall_score += 1
if 머리위치_낙상신호:
    fall_score += 1
if 기울기_낙상신호:
    fall_score += 1

IF fall_score >= 2:
    → 낙상 감지!
```

**버퍼링**: 오차 감소를 위해 최근 30프레임(약 1초) 중 50% 이상에서 낙상이 감지되면 최종 판정

---

## 📦 필수 설치 파일

### 1. MoveNet 모델 파일
```
models/movenet_singlepose_lightning.tflite
```
**다운로드 위치**: 
- TensorFlow Hub: https://www.kaggle.com/models/google/movenet/frameworks/tfLite/variations/singlepose-lightning
- 또는 TFLite 변환 스크립트로 자체 생성

**파일 크기**: ~3-4MB

### 2. 오디오 모델 (재학습 필수)
```
models/keyword_audio.tflite  (현재 미지원, 재학습 필요)
```

---

## ⚙️ 설정 파일 (config.py)

### MoveNet 관련 설정
```python
MOVENET_INPUT_SIZE = 192  # 192x192 픽셀 입력
MOVENET_CONFIDENCE_THRESHOLD = 0.3  # 관절 신뢰도 30% 이상만 인정

# 낙상 판정 파라미터
FALL_DETECTION_THRESHOLD = 0.5  # 50% 이상 낙상 신호 필요
ASPECT_RATIO_LOW = 0.3   # 너비/높이 비율 하한
ASPECT_RATIO_HIGH = 0.8  # 너비/높이 비율 상한
HEAD_POSITION_THRESHOLD = 0.2  # 머리 위치 임계값
```

### 오디오 관련 설정
```python
AUDIO_ENABLED = False  # 현재 비활성화 (재학습 대기)
```

---

## 🚀 사용 방법

### 1단계: 환경 설정
```bash
# 필수 라이브러리 설치
pip install opencv-python numpy sounddevice tflite-runtime

# RPi 5 환경
sudo apt-get install -y python3-dev libatlas-base-dev
pip install numpy
```

### 2단계: MoveNet 모델 다운로드
```bash
# models/ 디렉토리에 저장
wget https://www.kaggle.com/...movenet_singlepose_lightning.tflite
mv movenet_singlepose_lightning.tflite ./models/
```

### 3단계: 실행
```bash
python main.py
```

### 출력 예시
```
=== 독거노인 케어 시스템 (RPi 5 / Python 3.11) ===
[Hardware] GPIO 초기화 완료 (Pin 17)
[Video] MoveNet 프로세스 초기화 중...
[Video] 모델 입력 shape: (1, 192, 192, 3)
[Video] 모델 출력 shape: (1, 1, 17, 3)
[Video] MoveNet 낙상 감지 모니터링 시작...
[Audio] 오디오 모델 비활성화 상태 (재학습 대기 중)
[Main] 시스템 가동 시작. (Ctrl+C로 종료)

[Video] Frame 10: 유효한 관절 15/17, 코(0.87), 엉덩이(0.95, 0.94)
...
[Video] 낙상 감지! (Frame 145)
>> [ALERT] 응답 대기 중...
```

---

## 📊 MoveNet 17개 관절 정의

| 인덱스 | 관절명 | 인덱스 | 관절명 |
|--------|--------|--------|--------|
| 0 | 코 (Nose) | 9 | 왼쪽 손목 |
| 1 | 왼쪽 눈 | 10 | 오른쪽 손목 |
| 2 | 오른쪽 눈 | 11 | 왼쪽 골반 |
| 3 | 왼쪽 귀 | 12 | 오른쪽 골반 |
| 4 | 오른쪽 귀 | 13 | 왼쪽 무릎 |
| 5 | 왼쪽 어깨 | 14 | 오른쪽 무릎 |
| 6 | 오른쪽 어깨 | 15 | 왼쪽 발목 |
| 7 | 왼쪽 팔꿈치 | 16 | 오른쪽 발목 |
| 8 | 오른쪽 팔꿈치 | | |

---

## 🔴 오디오 모델 재학습 가이드

### 현재 상태
- 원본 오디오 모델의 문제점:
  - 비명 감지 정확도 낮음
  - 다른 큰소리를 오신호로 인식
  - 거짓 양성(False Positive) 많음

### 재학습 절차

#### 1. 학습 데이터 수집
```
data/
├── scream/          # 비명 샘플 (1초 단위, 16kHz WAV)
│   ├── scream_001.wav
│   ├── scream_002.wav
│   └── ...
├── help/            # 도움 신호 샘플
│   ├── help_001.wav
│   └── ...
└── background/      # 배경 소음 (거짓 음성 방지)
    ├── noise_001.wav
    └── ...
```

#### 2. TensorFlow 모델 재학습
```python
# transfer_learning_audio.py
import tensorflow as tf
from tensorflow import keras

# 기본 모델 로드 (또는 새로 구축)
model = keras.Sequential([
    keras.layers.Input(shape=(16000, 1)),  # 1초, 16kHz
    keras.layers.Conv1D(128, 80, activation='relu'),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(3, activation='softmax')  # [scream, help, background]
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 학습 데이터로 학습
# ... 학습 코드 ...

# TFLite로 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('keyword_audio.tflite', 'wb') as f:
    f.write(tflite_model)
```

#### 3. TFLite 모델 배포
```bash
cp keyword_audio.tflite ./models/
```

#### 4. 활성화
config.py에서:
```python
AUDIO_ENABLED = True
```

---

## 🧪 테스트 및 튜닝

### 낙상 감지 파라미터 조정

#### 가양성(오류)이 많은 경우:
```python
# config.py
ASPECT_RATIO_LOW = 0.25   # 더 낮게 (더 극단적인 자세만 감지)
FALL_DETECTION_THRESHOLD = 0.6  # 60%로 상향 (더 많은 프레임 필요)
HEAD_POSITION_THRESHOLD = 0.25  # 더 엄격하게
```

#### 미감지(거짓 음성)가 많은 경우:
```python
# config.py
ASPECT_RATIO_LOW = 0.35   # 더 높게 (더 쉽게 감지)
FALL_DETECTION_THRESHOLD = 0.4  # 40%로 하향 (더 적은 프레임 필요)
HEAD_POSITION_THRESHOLD = 0.15  # 더 너그럽게
```

### 성능 모니터링

1. **프레임 처리 속도**
   ```
   Target: 30 FPS
   MoveNet Lightning: ~10-15ms/프레임 (RPi 5)
   ```

2. **메모리 사용량**
   ```
   Model: ~50MB
   Runtime: ~100-200MB (Total)
   ```

3. **낙상 감지 정확도**
   - 테스트: 실제 낙상 및 일상 활동 영상으로 평가
   - 목표: 95% 이상

---

## 📝 로그 분석

### 정상 작동 로그
```
[Video] Frame 10: 유효한 관절 15/17, 코(0.87), 엉덩이(0.95, 0.94)
→ 해석: 15개 관절 감지, 신뢰도 양호
```

### 문제 진단

**"유효한 관절 5/17 미만"**
- 원인: 카메라 각도, 조명 부족
- 해결: 카메라 위치 조정, 조명 개선

**"낙상 감지 안 됨"**
- 해결: FALL_DETECTION_THRESHOLD 하향
- 확인: 관절 신뢰도 로그 확인

**"거짓 낙상 감지 많음"**
- 해결: 파라미터 상향 (ASPECT_RATIO, THRESHOLD)

---

## 🔄 상태 머신 흐름

```
[HOME] ──FALL_DETECTED──> [ALERT] ──TIMEOUT(20s)──> [EMERGENCY]
  ↑                          │                           ↓
  └──BUTTON_PRESSED──────────┴─────────────────────────→┘
  
  ├──(30분 미감지)──> [AWAY] ──(24시간 미복귀)──> [EMERGENCY]
  │                    │
  └────<PERSON_DETECTED
```

---

## 📋 체크리스트

- [ ] MoveNet 모델 파일 다운로드 (models/movenet_singlepose_lightning.tflite)
- [ ] Python 라이브러리 설치 (opencv-python, numpy, sounddevice, tflite-runtime)
- [ ] config.py 파라미터 확인 및 환경에 맞춰 조정
- [ ] 카메라 및 마이크 테스트
- [ ] 초기 실행 및 로그 확인
- [ ] 낙상 감지 정확도 테스트
- [ ] (선택) 오디오 모델 재학습 및 활성화

---

## 🆘 자주 묻는 질문

**Q. MoveNet 모델 버전은?**
A. MoveNet SinglePose Lightning (빠르고 경량)
   - Thunder 버전도 가능하나 더 느림

**Q. 처리 속도가 중요한가?**
A. 네, RPi 5는 리소스 제한이 있으므로 Lightning 권장

**Q. 낙상 외에 다른 사건도 감지 가능?**
A. 관절 좌표 분석으로 기울어짐, 누움 등 다양한 자세 감지 가능
   파라미터 조정으로 확장 가능

**Q. 여름/겨울 의류 변화는 문제 없나?**
A. MoveNet은 신체 관절만 추적하므로 의류 차이 영향 최소화

---

## 📞 기술 지원

마지막 업데이트: 2025년 11월
