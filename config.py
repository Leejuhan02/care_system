# config.py
import os

# --- 경로 설정 ---
# __file__을 기준으로 절대 경로를 계산하여 경로 의존성 문제 해결
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MOVENET_MODEL_PATH = os.path.join(MODEL_DIR, "movenet_singlepose_lightning.tflite")
AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, "keyword_audio.tflite")

# --- 하드웨어 설정 (RPi 5) ---
BUTTON_PIN = 17         # BCM 17번 핀
GPIO_CHIP = 'gpiochip4' # RPi 5의 메인 GPIO 칩

# --- 타이머 설정 (단위: 초) ---
PERSON_TIMEOUT = 1800   # 30분간 사람 미감지 -> Away
ALERT_TIMEOUT = 20      # 낙상 감지 후 20초간 미응답 -> Emergency
AWAY_TIMEOUT = 86400    # 24시간 외출 미복귀 -> Emergency
WATCHDOG_TIMEOUT = 60   # 60초간 AI 응답 없음 -> 오류 감지

# --- MoveNet 모델 파라미터 ---
MOVENET_INPUT_SIZE = 192  # MoveNet Lightning 입력 크기 (192x192)
MOVENET_CONFIDENCE_THRESHOLD = 0.3  # 관절 감지 신뢰도 기준 (30%)

# --- 낙상 감지 파라미터 (관절 좌표 기반) ---
# MoveNet 17개 관절: [0]코, [1]왼쪽눈, [2]오른쪽눈, [3]왼쪽귀, [4]오른쪽귀,
#                   [5]왼쪽어깨, [6]오른쪽어깨, [7]왼쪽팔꿈치, [8]오른쪽팔꿈치,
#                   [9]왼쪽손목, [10]오른쪽손목, [11]왼쪽골반, [12]오른쪽골반,
#                   [13]왼쪽무릎, [14]오른쪽무릎, [15]왼쪽발목, [16]오른쪽발목

# 낙상 판정을 위한 조건들
FALL_DETECTION_THRESHOLD = 0.5  # 낙상 확률 기준 (50%)
FRAME_BUFFER_SIZE = 30  # 안정성을 위해 30프레임 누적 (약 1초)
ASPECT_RATIO_LOW = 0.3  # 너비/높이 비율이 낮을 때 낙상 의심 (누워있는 자세)
ASPECT_RATIO_HIGH = 0.8  # 너비/높이 비율이 높을 때 낙상 의심 (누워있는 자세)
HEAD_POSITION_THRESHOLD = 0.2  # 머리가 골반보다 낮을 때 낙상 의심

# --- 오디오 모델 파라미터 ---
# (재학습된 모델이 완성되면 이후에 활성화)
AUDIO_ENABLED = False  # 현재 오디오 모델 미완성 상태
KEYWORD_INDEX = 0  # 오디오 모델의 키워드 인덱스 (재학습 모델 완성 시)
AUDIO_CONFIDENCE_THRESHOLD = 0.6  # 60% 이상일 때만 인정
