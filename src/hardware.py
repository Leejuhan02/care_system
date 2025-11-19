# src/hardware.py
import gpiod
import threading
import time
import numpy as np
import sounddevice as sd
from multiprocessing import Queue
import config

class ButtonReader:
    """
    RPi 5의 gpiod 라이브러리를 사용하여 버튼 입력 감지.
    Falling Edge 감지로 안정적인 버튼 처리 구현.
    """
    def __init__(self, event_queue: Queue):
        self.queue = event_queue
        self.pin = config.BUTTON_PIN
        self.running = False
        self.chip = None
        self.line = None
        
        # [RPi 5 호환] gpiod 칩 초기화
        try:
            self.chip = gpiod.Chip(config.GPIO_CHIP)
            self.line = self.chip.get_line(self.pin)
            # 입력 모드, Falling Edge 감지, 내부 Pull-up 설정
            self.line.request(
                consumer="care_system",
                type=gpiod.LINE_REQ_EV_FALLING_EDGE,
                flags=gpiod.LINE_REQ_FLAG_BIAS_PULL_UP
            )
            print(f"[Hardware] GPIO 초기화 완료 (Pin {self.pin})")
        except Exception as e:
            print(f"[Hardware] GPIO 초기화 오류: {e}")
            self.line = None
            self.chip = None

    def start(self):
        """버튼 감시 스레드 시작"""
        if self.line and not self.running:
            self.running = True
            # 논블로킹을 위해 스레드 시작
            threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        """버튼 이벤트 대기 및 큐에 전송"""
        print(f"[Hardware] 버튼 감시 시작 (Pin {self.pin})")
        while self.running:
            try:
                # [로직] 이벤트 발생 대기 (타임아웃 1초)
                if self.line.wait_for_event(timeout_sec=1):
                    event = self.line.read_event()
                    if event.type == gpiod.LINE_EVENT_FALLING_EDGE:
                        self.queue.put("BUTTON_PRESSED")
                        time.sleep(0.3)  # 디바운싱
            except Exception as e:
                print(f"[Hardware] 버튼 에러: {e}")
                time.sleep(0.5)

    def cleanup(self):
        """GPIO 자원 해제"""
        self.running = False
        try:
            if self.line:
                self.line.release()
            if self.chip:
                self.chip.close()
            print("[Hardware] GPIO 정리 완료")
        except Exception as e:
            print(f"[Hardware] GPIO 정리 중 오류: {e}")


class Speaker:
    """
    RPi 5의 오디오 출력을 담당.
    스레드 기반으로 메인 루프 블로킹 방지.
    """
    def __init__(self):
        """Speaker 객체 초기화"""
        self.is_playing = False

    def play_alert(self):
        """
        경고 사운드 재생.
        880Hz 사인파 3회 반복 재생.
        """
        if not self.is_playing:
            threading.Thread(target=self._beep, daemon=True).start()

    def _beep(self):
        """실제 사운드 재생 로직"""
        self.is_playing = True
        try:
            fs = 44100  # 샘플링 레이트 (44.1kHz)
            duration = 0.5  # 0.5초
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            # 880Hz 사인파 생성
            wave = 0.5 * np.sin(2 * np.pi * 880 * t)
            
            # 3회 반복 재생
            for i in range(3):
                try:
                    sd.play(wave, fs)
                    sd.wait()
                    if i < 2:  # 마지막이 아니면 딜레이
                        time.sleep(0.2)
                except Exception as e:
                    print(f"[Hardware] 스피커 재생 오류 ({i+1}/3): {e}")
                    
        except Exception as e:
            print(f"[Hardware] 스피커 초기화 오류: {e}")
        finally:
            self.is_playing = False
