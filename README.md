# care_system
# care_system

Mircoprocessor, Team Project

## 👵 독거노인 케어 시스템 (Elderly Care System)

이 프로젝트는 **Raspberry Pi 5** 환경에서 동작하는 독거노인 안전 모니터링 시스템입니다. 컴퓨터 비전(OpenCV)과 오디오 분석을 통해 낙상(쓰러짐), 응급 키워드, 장기 미활동 등을 실시간으로 감지하고 알림을 전송합니다.

### 핵심 요약

- 목표 플랫폼: Raspberry Pi 5 (64-bit)
- 권장 Python: 3.11 (가상환경 사용) — 예제 venv 경로: `/home/raspberry/my_venv_311`
- 모델(선택): `models/fall_detection.tflite` (TFLite 포맷)

### 폴더 구조 (간략)

`/home/pi/care_system/` 또는 프로젝트 루트
- `main.py`           : 시스템 진입점
- `config.py`         : 설정 (핀, 타이머, 모델 경로 등)
- `models/`           : `fall_detection.tflite`, `keyword_audio.tflite`
- `src/`              : 소스 코드 (하드웨어, 프로세서, 알림 등)

---

## 설치 및 실행 안내 (Raspberry Pi 5 / Python 3.11)

아래 예시는 가상환경을 `/home/raspberry/my_venv_311`에 만들고 사용하는 과정입니다.

1) 시스템 의존성 설치

```bash
sudo apt update
sudo apt install -y build-essential wget libsndfile1-dev libportaudio2 libatlas-base-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev pkg-config
```

2) (선택) Python 3.11 설치 — 이미 설치되어 있으면 생략

```bash
cd /tmp
wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
tar -xf Python-3.11.9.tgz
cd Python-3.11.9
./configure --enable-optimizations
sudo make altinstall
```

3) 가상환경 생성 및 활성화

```bash
/usr/local/bin/python3.11 -m venv /home/raspberry/my_venv_311
source /home/raspberry/my_venv_311/bin/activate
pip install --upgrade pip
```

4) 파이썬 패키지 설치

프로젝트 루트에서:

```bash
cd /home/raspberry/care_system  # 또는 실제 클론한 경로
pip install -r requirements.txt
```

주의: `tflite-runtime`은 플랫폼(아키텍처)과 파이썬 버전에 민감합니다. Raspberry Pi 5 (aarch64) + Python 3.11 용의 사전 빌드 휠(.whl)을 사용하는 것이 가장 안정적입니다. 예시:

```bash
# (예시) 정확한 파일명은 배포처/버전에 따라 달라집니다. 적절한 .whl 경로로 바꾸세요.
pip install https://example.com/path/to/tflite-runtime-<version>-cp311-cp311-linux_aarch64.whl
```

대안:
- `pip install tflite-runtime` 시도 (성공하지 않을 수 있음)
- 전체 `tensorflow`를 설치하여 `tensorflow.lite.Interpreter` 사용 (용량/메모리 부담 큼)

---

## 모델 설치 및 배치 (옵션)

- `models/fall_detection.tflite` 파일을 프로젝트 `models/` 경로에 위치시킵니다.
- 모델은 TFLite 형식이어야 하며, 입력 크기/채널(예: 224x224x3 등)과 출력 형식을 알아야 최적 전처리를 할 수 있습니다.

만약 TFLite 모델이 준비되어 있지 않다면, 이 저장소는 TFLite 없이 동작하는 **heuristic**(규칙 기반) 대체 모드를 제공합니다. 이 모드는 OpenCV의 HOG 사람 검출기를 사용해 단순한 쓰러짐(낙상) 징후를 탐지합니다. 정확도는 전문 모델보다 낮지만, 장비나 런타임 제약으로 모델을 사용할 수 없을 때 유용합니다.

---

## 실행 예시 (앱 모드)

프로젝트 루트에서 `run_app.py`가 제공됩니다. 두 가지 주요 모드가 있습니다.

- Heuristic fallback (모델 불필요, 빠르게 테스트 가능)

```bash
source /home/raspberry/my_venv_311/bin/activate
cd /home/raspberry/care_system
python run_app.py --mode heuristic --camera 0 --display
```

- TFLite 모델 사용

```bash
source /home/raspberry/my_venv_311/bin/activate
cd /home/raspberry/care_system
python run_app.py --mode tflite --model models/fall_detection.tflite --camera 0 --display
```

옵션 설명:
- `--camera 0`: 기본 카메라 장치(USB 또는 CSI). 필요시 인덱스 변경.
- `--display`: OpenCV 창으로 영상/결과 표시(모니터 사용 시).
- `--threshold`: TFLite 출력이 확률일 경우 쓰러짐 판정 임계값(기본 0.5).

---

## 통합 가이드

- `main.py`는 전체 멀티프로세스 기반 시스템의 진입점입니다. 빠른 독립 실행/검증을 위해 `run_app.py`(또는 `src/app.py`)를 사용해 모델 또는 휴리스틱 방식으로 동작을 확인하세요.
- 모델과 전처리가 모두 정상 작동하면 동일한 로직을 `src/processors.py`의 `VideoProcessor`로 통합해 메인 시스템과 함께 운영하세요.

## 문제 해결 팁

- TFLite 런타임 설치 실패: 정확한 아키텍처(aarch64)·파이썬 버전(cp311)에 맞는 `.whl`을 찾아 설치하세요.
- 카메라가 열리지 않음: `v4l2-ctl --list-devices`로 장치 확인. 권한 문제 시 `sudo usermod -a -G video $USER` 후 재로그인.
- 헤드리스 환경: `--display` 사용 금지. 로그 확인 또는 원격 스트리밍 사용.

---

더 상세한 개발 문서는 `Agent/` 디렉터리의 가이드를 참고하세요.
```



