# care_system
Mircoprocessor, Team Project
# 👵 독거노인 케어 시스템 (Elderly Care System)

이 프로젝트는 **Raspberry Pi 5** 환경에서 동작하는 독거노인 안전 모니터링 시스템입니다.
컴퓨터 비전(OpenCV)과 소리 감지(Audio Analysis) AI 모델을 활용하여 낙상, 응급 키워드, 장기 미활동 등을 실시간으로 감지하고 보호자에게 알림을 전송합니다.

## 🛠 폴더 구조

/home/pi/care_system/
├── main.py               # [Entry Point] 프로그램 시작점, 의존성 주입
├── config.py             # [Config] 핀 번호, 타이머, 모델 경로 설정
├── .env                  # [Secret] 민감 정보 (API Key 등)
├── models/               # [Model] AI 모델 파일 저장소
│   ├── fall_detection.tflite
│   └── keyword_audio.tflite
└── src/                  # [Source] 핵심 로직
    ├── __init__.py
    ├── interfaces.py     # [OCP] 알림 시스템 인터페이스
    ├── notifiers.py      # [Strategy] 알림 구현체 (Console, SMS 등)
    ├── hardware.py       # [Driver] RPi 5 하드웨어 제어 (gpiod)
    ├── processors.py     # [Process] AI 멀티프로세싱 로직
    └── states.py         # [State] 상태 패턴 기반 로직

## 🛠 하드웨어 요구사항

* **메인 보드:** Raspberry Pi 5 (필수, `gpiod` 라이브러리 사용)
* **카메라:** USB 웹캠 또는 라즈베리 파이 카메라 모듈
* **오디오:** USB 마이크 및 스피커 (또는 통합 모듈)
* **입력 장치:** 푸시 버튼 (GPIO 17번, GND 연결)

## ⚙️ 개발 환경

* **OS:** Raspberry Pi OS (Bookworm, 64-bit)
* **Python:** 3.11 (가상 환경 `venv` 권장)
* **Key Libraries:** `gpiod`, `multiprocessing`, `opencv-python-headless`, `tflite-runtime`, `sounddevice`

---

## 🚀 설치 가이드 (Installation)

라즈베리 파이 5의 기본 Python 버전(3.13+) 호환성 문제를 해결하기 위해 **Python 3.11**을 설치하고 가상 환경을 구축합니다.

### 1. 필수 빌드 도구 및 의존성 설치
터미널을 열고 다음 명령어를 순서대로 입력하세요.

```bash
sudo apt update
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev libportaudio2

cd /tmp
wget [https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz](https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz)
tar -xf Python-3.11.9.tgz
cd Python-3.11.9
./configure --enable-optimizations
sudo make altinstall

mkdir -p ~/care_system
cd ~/care_system

# Python 3.11 기반의 venv 생성
/usr/local/bin/python3.11 -m venv venv



# 가상 환경 진입
source venv/bin/activate

# pip 업그레이드 및 라이브러리 설치
pip install --upgrade pip
pip install numpy sounddevice gpiod opencv-python-headless

# TFLite Runtime 설치 (Linux aarch64 Python 3.11용)
# 만약 아래 명령어로 설치가 안 된다면, 호환되는 .whl 파일을 찾아 설치해야 합니다.
pip install tflite-runtime


