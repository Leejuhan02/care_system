# src/processors.py
import multiprocessing
import time
import cv2
import numpy as np
import sounddevice as sd
from tflite_runtime.interpreter import Interpreter
import config
import threading
from collections import deque

class FallDetectionAnalyzer:
    """
    MoveNet 관절 좌표 기반 낙상 감지 분석기.
    신체 자세 변화를 분석하여 낙상 여부를 판단합니다.
    """
    
    def __init__(self):
        """분석기 초기화"""
        self.frame_buffer = deque(maxlen=config.FRAME_BUFFER_SIZE)
        self.last_fall_time = 0
        self.fall_debounce_time = 2.0  # 낙상 재감지 방지 (2초)
        
        # MoveNet 17개 관절 인덱스
        self.JOINTS = {
            'nose': 0,
            'left_eye': 1, 'right_eye': 2,
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }

    def calculate_body_aspect_ratio(self, keypoints):
        """
        신체 너비/높이 비율 계산.
        낮은 비율(0.3 이하) = 누워있는 자세 (낙상 가능성)
        높은 비율(0.8 이상) = 눕거나 쪼그린 자세 (낙상 가능성)
        """
        valid_points = [kp for kp in keypoints if kp[2] > config.MOVENET_CONFIDENCE_THRESHOLD]
        
        if len(valid_points) < 5:
            return None
        
        # x, y 좌표만 추출
        points = np.array([[kp[0], kp[1]] for kp in valid_points])
        
        # 너비와 높이 계산
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)
        
        if height < 1e-6:  # 0으로 나누기 방지
            return None
        
        aspect_ratio = width / height
        return aspect_ratio

    def calculate_head_position(self, keypoints):
        """
        머리 위치 계산 (y좌표 정규화).
        머리(코, 눈, 귀)와 골반(힙) 위치를 비교하여
        머리가 골반보다 훨씬 낮으면 낙상 의심.
        """
        head_joints = [
            keypoints[self.JOINTS['nose']],
            keypoints[self.JOINTS['left_eye']],
            keypoints[self.JOINTS['right_eye']]
        ]
        
        hip_joints = [
            keypoints[self.JOINTS['left_hip']],
            keypoints[self.JOINTS['right_hip']]
        ]
        
        # 신뢰도 기준 필터링
        valid_head = [kp for kp in head_joints if kp[2] > config.MOVENET_CONFIDENCE_THRESHOLD]
        valid_hip = [kp for kp in hip_joints if kp[2] > config.MOVENET_CONFIDENCE_THRESHOLD]
        
        if len(valid_head) < 1 or len(valid_hip) < 1:
            return None
        
        # 평균 y좌표
        avg_head_y = np.mean([kp[1] for kp in valid_head])
        avg_hip_y = np.mean([kp[1] for kp in valid_hip])
        
        # 머리가 골반보다 낮을 때 값이 커짐 (정규화된 좌표에서 y는 아래로 갈수록 커짐)
        # 머리가 골반보다 60% 이상 낮으면 낙상 의심
        head_position_diff = (avg_head_y - avg_hip_y) / (1 + np.abs(avg_hip_y))
        
        return head_position_diff

    def calculate_body_stability(self, keypoints):
        """
        신체 안정성 계산.
        양쪽 어깨와 엉덩이의 높이 차이 분석.
        기울어진 정도가 크면 불안정한 자세(낙상 중).
        """
        shoulder_joints = [
            keypoints[self.JOINTS['left_shoulder']],
            keypoints[self.JOINTS['right_shoulder']]
        ]
        
        hip_joints = [
            keypoints[self.JOINTS['left_hip']],
            keypoints[self.JOINTS['right_hip']]
        ]
        
        valid_shoulder = [kp for kp in shoulder_joints if kp[2] > config.MOVENET_CONFIDENCE_THRESHOLD]
        valid_hip = [kp for kp in hip_joints if kp[2] > config.MOVENET_CONFIDENCE_THRESHOLD]
        
        if len(valid_shoulder) < 2 or len(valid_hip) < 2:
            return None
        
        # 양쪽 어깨/엉덩이의 높이 차이 (기울기)
        shoulder_tilt = abs(valid_shoulder[0][1] - valid_shoulder[1][1])
        hip_tilt = abs(valid_hip[0][1] - valid_hip[1][1])
        
        # 정규화된 기울기 값 (0~1)
        body_tilt = max(shoulder_tilt, hip_tilt)
        
        return body_tilt

    def detect_fall(self, keypoints):
        """
        관절 좌표 기반 낙상 감지.
        여러 조건을 조합하여 낙상 판정.
        """
        if not isinstance(keypoints, np.ndarray) or len(keypoints) < 17:
            return False
        
        # 1. 체격 비율 분석 (누워있는 자세)
        aspect_ratio = self.calculate_body_aspect_ratio(keypoints)
        is_lying_down = False
        if aspect_ratio is not None:
            # 너무 낮거나 높은 비율 = 누워있거나 쪼그린 자세
            if aspect_ratio < config.ASPECT_RATIO_LOW or aspect_ratio > config.ASPECT_RATIO_HIGH:
                is_lying_down = True
        
        # 2. 머리 위치 분석 (머리가 골반보다 훨씬 낮음)
        head_diff = self.calculate_head_position(keypoints)
        is_head_low = False
        if head_diff is not None:
            if head_diff > config.HEAD_POSITION_THRESHOLD:
                is_head_low = True
        
        # 3. 신체 안정성 분석 (기울어진 자세)
        body_tilt = self.calculate_body_stability(keypoints)
        is_tilted = False
        if body_tilt is not None:
            if body_tilt > 0.15:  # 15% 이상 기울어짐
                is_tilted = True
        
        # [로직] 낙상 판정: 최소 2개 조건 만족 시 낙상으로 판정
        # - 쓰러질 때 앉아있는 모습: is_lying_down + is_tilted
        # - 웅크린 자세: is_lying_down (aspect_ratio가 낮음)
        # - 물건을 주는 자세: is_head_low (머리가 낮아짐)
        # - 눕는 동작: is_lying_down + is_head_low
        
        fall_score = 0
        if is_lying_down:
            fall_score += 1
        if is_head_low:
            fall_score += 1
        if is_tilted:
            fall_score += 1
        
        # 최소 2개 조건 만족 + 신뢰도 기준
        is_fall = fall_score >= 2
        
        return is_fall

    def process_frame(self, keypoints):
        """
        프레임별 낙상 분석 (버퍼를 이용한 안정성 향상).
        """
        # 버퍼에 추가
        self.frame_buffer.append(keypoints)
        
        # 버퍼가 충분하지 않으면 대기
        if len(self.frame_buffer) < config.FRAME_BUFFER_SIZE // 2:
            return False
        
        # 최근 프레임들에서 낙상 감지 빈도 확인
        fall_count = 0
        for kp in self.frame_buffer:
            if self.detect_fall(kp):
                fall_count += 1
        
        # 버퍼의 50% 이상에서 낙상이 감지되면 실제 낙상으로 판정
        fall_probability = fall_count / len(self.frame_buffer)
        
        # 디바운싱: 최근 2초 내에 낙상 감지 시 무시
        current_time = time.time()
        if fall_probability > config.FALL_DETECTION_THRESHOLD:
            if current_time - self.last_fall_time > self.fall_debounce_time:
                self.last_fall_time = current_time
                return True
        
        return False


class VideoProcessor(multiprocessing.Process):
    """
    MoveNet 모델을 사용한 관절 기반 낙상 감지.
    카메라에서 입력받은 영상으로부터 사람의 관절 좌표를 추출하고
    이를 기반으로 낙상을 감지합니다.
    """
    def __init__(self, queue: multiprocessing.Queue):
        super().__init__(daemon=True)
        self.queue = queue
        self.running = True

    def preprocess_image(self, frame):
        """
        이미지 전처리: MoveNet 입력 형식으로 변환.
        MoveNet Lightning: 192x192 RGB 입력 필요.
        """
        # 정사각형으로 변환 (종횡비 유지)
        h, w = frame.shape[:2]
        size = min(h, w)
        
        # 중앙에서 정사각형 크롭
        start_y = (h - size) // 2
        start_x = (w - size) // 2
        cropped = frame[start_y:start_y + size, start_x:start_x + size]
        
        # 192x192로 리사이즈
        resized = cv2.resize(cropped, (config.MOVENET_INPUT_SIZE, config.MOVENET_INPUT_SIZE))
        
        # RGB로 변환 (OpenCV는 BGR 사용)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 정규화 (0~1 범위로)
        normalized = rgb.astype(np.float32) / 255.0
        
        # 배치 차원 추가
        input_data = np.expand_dims(normalized, axis=0)
        
        return input_data

    def parse_output(self, output):
        """
        MoveNet 모델 출력 파싱.
        출력 형식: [1, 1, 17, 3] -> (y, x, confidence) for 17 joints
        반환: [17, 3] 배열 (각 관절의 y, x, confidence)
        """
        # 출력 shape에 따라 처리
        if len(output.shape) == 4:
            # [1, 1, 17, 3] 형태
            keypoints = output[0, 0, :, :]
        elif len(output.shape) == 3:
            # [1, 17, 3] 형태
            keypoints = output[0, :, :]
        else:
            # 다른 형태 대응
            keypoints = output.reshape(-1, 3)
        
        # x, y, confidence 순서로 재정렬 (MoveNet은 y, x 순서 반환)
        # [y, x, confidence] -> [x, y, confidence]
        if keypoints.shape[1] == 3:
            keypoints_reordered = keypoints.copy()
            # y와 x를 교환
            keypoints_reordered[:, 0], keypoints_reordered[:, 1] = keypoints[:, 1], keypoints[:, 0]
            return keypoints_reordered
        
        return keypoints

    def run(self):
        """비디오 프로세스 메인 루프"""
        print("[Video] MoveNet 프로세스 초기화 중...")
        interpreter = None
        cap = None
        fall_analyzer = None
        
        try:
            # [로직] MoveNet TensorFlow Lite 모델 로드
            interpreter = Interpreter(model_path=config.MOVENET_MODEL_PATH)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"[Video] 모델 입력 shape: {input_details[0]['shape']}")
            print(f"[Video] 모델 출력 shape: {output_details[0]['shape']}")
            
            # [로직] 카메라 초기화
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("카메라를 열 수 없습니다.")
            
            # RPi 5 카메라 설정 최적화
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 최소화
            cap.set(cv2.CAP_PROP_FPS, 30)  # 30fps 설정
            
            # 낙상 분석기 초기화
            fall_analyzer = FallDetectionAnalyzer()
            
            last_heartbeat = time.time()
            last_person_detection_time = 0
            frame_count = 0
            person_detected = False
            
            print("[Video] MoveNet 낙상 감지 모니터링 시작...")
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("[Video] 프레임 읽기 실패")
                    continue

                frame_count += 1
                
                try:
                    # [로직] 이미지 전처리
                    input_data = self.preprocess_image(frame)
                    
                    # [로직] 추론 수행
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    
                    # [로직] 출력 파싱
                    output = interpreter.get_tensor(output_details[0]['index'])
                    keypoints = self.parse_output(output)
                    
                    # [로직] 사람 감지 확인 (신뢰도 있는 관절이 충분한지)
                    valid_joints = np.sum(keypoints[:, 2] > config.MOVENET_CONFIDENCE_THRESHOLD)
                    
                    if valid_joints > 5:
                        person_detected = True
                        # 5초에 한 번 사람 감지 신호 전송
                        current_time = time.time()
                        if current_time - last_person_detection_time > 5:
                            self.queue.put("PERSON_DETECTED")
                            last_person_detection_time = current_time
                        
                        # [로직] 낙상 감지
                        if fall_analyzer.process_frame(keypoints):
                            self.queue.put("FALL_DETECTED")
                            print(f"[Video] 낙상 감지! (Frame {frame_count})")
                    else:
                        person_detected = False
                    
                    # [디버깅] 주요 관절 정보 출력 (10프레임마다)
                    if frame_count % 10 == 0 and valid_joints > 5:
                        nose = keypoints[0]
                        left_hip = keypoints[11]
                        right_hip = keypoints[12]
                        print(f"[Video] Frame {frame_count}: 유효한 관절 {valid_joints}/17, "
                              f"코({nose[2]:.2f}), 엉덩이({left_hip[2]:.2f}, {right_hip[2]:.2f})")
                    
                except Exception as e:
                    print(f"[Video] 추론 오류: {e}")
                    import traceback
                    traceback.print_exc()
                
                # [안정성] Watchdog을 위한 심박 신호 전송
                if time.time() - last_heartbeat > 10:
                    self.queue.put("HEARTBEAT")
                    last_heartbeat = time.time()
                
                # 약 30fps 제한
                time.sleep(0.03)

        except Exception as e:
            print(f"[Video] 치명적 오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 자원 정리
            if cap:
                cap.release()
            print("[Video] 프로세스 종료")


class AudioProcessor(multiprocessing.Process):
    """
    오디오 기반 비명/긴급 신호 감지.
    
    [현재 상태]
    - 오디오 모델은 재학습 중입니다.
    - 원본 모델의 비명 감지 정확도 문제로 인해 현재 비활성화 상태입니다.
    - 향후 커스텀 학습(비명, 도움 신호 등)을 완료한 후 재활성화됩니다.
    
    [재학습 완료 후 절차]
    1. 커스텀 재학습된 TFLite 모델을 models/ 디렉토리에 저장
    2. config.py에서 AUDIO_ENABLED를 True로 변경
    3. 아래 코드의 주석을 제거하고 활성화
    """
    def __init__(self, queue: multiprocessing.Queue):
        super().__init__(daemon=True)
        self.queue = queue
        self.running = True

    def run(self):
        """오디오 프로세스 메인 루프"""
        print("[Audio] 프로세스 초기화 중...")
        
        if not config.AUDIO_ENABLED:
            print("[Audio] 오디오 모델 비활성화 상태 (재학습 대기 중)")
            print("[Audio] 재학습 완료 후 config.AUDIO_ENABLED = True로 변경하세요")
            # 프로세스 유지
            while self.running:
                time.sleep(1)
            return
        
        interpreter = None
        
        try:
            # [로직] 재학습된 오디오 TensorFlow Lite 모델 로드
            interpreter = Interpreter(model_path=config.AUDIO_MODEL_PATH)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            input_idx = input_details[0]['index']
            output_idx = output_details[0]['index']
            input_shape = input_details[0]['shape']  # 예: [1, 16000]
            
            # [로직] 샘플 레이트 확인
            sample_rate = input_shape[1] if len(input_shape) > 1 else 16000
            
            print(f"[Audio] 프로세스 시작. 입력 샘플레이트: {sample_rate}Hz")
            print(f"[Audio] 모델 입력 shape: {input_shape}")

            def audio_callback(indata, frames, time_info, status):
                """오디오 콜백 함수: 재학습된 모델로 비명/긴급 신호 감지"""
                if status:
                    print(f"[Audio] 스트림 상태: {status}")
                
                try:
                    # [로직] 오디오 데이터 전처리
                    data = indata.flatten().astype(np.float32)
                    
                    # 입력 크기에 맞춰 패딩 또는 자르기
                    if len(data) < input_shape[1]:
                        data = np.pad(data, (0, input_shape[1] - len(data)), mode='constant')
                    else:
                        data = data[:input_shape[1]]
                    
                    data = np.expand_dims(data, axis=0)
                    
                    # [로직] 추론 수행
                    interpreter.set_tensor(input_idx, data)
                    interpreter.invoke()
                    output = interpreter.get_tensor(output_idx)
                    
                    # [로직] 키워드 감지 (재학습된 모델의 출력 인덱스에 맞춰 수정 필요)
                    if output[0][config.KEYWORD_INDEX] > config.AUDIO_CONFIDENCE_THRESHOLD:
                        self.queue.put("KEYWORD_DETECTED")
                        print("[Audio] 비명/긴급 신호 감지!")
                        
                except Exception as e:
                    print(f"[Audio] 콜백 오류: {e}")

            # [로직] 마이크 스트림 시작
            with sd.InputStream(
                channels=1,
                samplerate=sample_rate,
                blocksize=sample_rate,
                callback=audio_callback,
                device=None  # 기본 마이크 사용
            ):
                print("[Audio] 마이크 스트림 활성화")
                while self.running:
                    time.sleep(1)

        except Exception as e:
            print(f"[Audio] 치명적 오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[Audio] 프로세스 종료")
