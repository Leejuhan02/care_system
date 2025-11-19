# src/states.py
import time
from abc import ABC, abstractmethod
from typing import List
from src.interfaces import INotifier
import config

# --- Abstract State ---
class State(ABC):
    """
    상태 머신의 기본 추상 클래스.
    모든 상태는 이 클래스를 상속받아 구현됩니다.
    """
    def __init__(self, context):
        self.context = context
        self.start_time = time.time()

    @abstractmethod
    def on_enter(self):
        """상태에 진입할 때 호출되는 메서드"""
        pass

    @abstractmethod
    def handle_event(self, event):
        """이벤트 처리 메서드"""
        pass

    @abstractmethod
    def update(self):
        """주기적 상태 업데이트 메서드"""
        pass


# --- Context (State Manager) ---
class Context:
    """
    상태 머신 컨텍스트.
    현재 상태를 관리하고 이벤트를 처리합니다.
    """
    def __init__(self, speaker, notifiers: List[INotifier]):
        self.speaker = speaker
        self.notifiers = notifiers
        
        self.last_person_time = time.time()
        self.last_heartbeat = time.time()
        
        # 초기 상태 설정
        self.state = HomeState(self)
        self.state.on_enter()

    def change_state(self, new_state_cls, reason=""):
        """상태 전환 메서드"""
        print(f"[State] 전환: {self.state.__class__.__name__} -> {new_state_cls.__name__} ({reason})")
        self.state = new_state_cls(self)
        self.state.on_enter()

    def notify_all(self, message):
        """
        [OCP] 주입된 모든 Notifier에게 메시지 전송 위임.
        새로운 Notifier 추가 시 이 메서드 수정 불필요.
        """
        for notifier in self.notifiers:
            notifier.send(message)

    def process_event(self, event):
        """
        이벤트 처리.
        Watchdog 처리 후 현재 상태에 위임.
        """
        # [로직] Watchdog 처리
        if event == "HEARTBEAT":
            self.last_heartbeat = time.time()
            return

        # [로직] 현재 상태에게 이벤트 처리 위임
        self.state.handle_event(event)

    def tick(self):
        """
        주기적 상태 업데이트 및 Watchdog 검사.
        메인 루프에서 매 프레임마다 호출됨.
        """
        # [로직] 현재 상태의 주기적 업데이트
        self.state.update()
        
        # [안정성] Watchdog: AI 시스템 응답 검사
        if time.time() - self.last_heartbeat > config.WATCHDOG_TIMEOUT:
            if not isinstance(self.state, EmergencyState):
                self.change_state(EmergencyState, "AI 시스템 응답 없음")


# --- Concrete States ---

class HomeState(State):
    """
    [Home] 정상 모니터링 상태.
    사람 감지 시 시간 기록, 낙상 감지 시 Alert로 전환.
    """
    def on_enter(self):
        print(">> [HOME] 모니터링 중...")

    def handle_event(self, event):
        """이벤트 처리"""
        if event == "PERSON_DETECTED":
            self.context.last_person_time = time.time()
        elif event == "FALL_DETECTED":
            self.context.change_state(AlertState, "낙상 감지")
        elif event == "KEYWORD_DETECTED":
            self.context.change_state(EmergencyState, "비상 키워드 감지")

    def update(self):
        """
        주기적 업데이트: 장시간 사람 미감지 시 Away로 전환.
        PERSON_TIMEOUT(30분) 이상 사람이 감지되지 않으면 외출로 판단.
        """
        if time.time() - self.context.last_person_time > config.PERSON_TIMEOUT:
            self.context.change_state(AwayState, "장시간 미감지")


class AlertState(State):
    """
    [Alert] 낙상 감지 후 사용자 응답 대기 상태.
    ALERT_TIMEOUT(20초) 내에 버튼 응답 필요.
    """
    def on_enter(self):
        print(">> [ALERT] 응답 대기 중...")
        self.context.speaker.play_alert()
        # [선택] 사용자에게 알림 전송 (실제 구현 시 활성화)
        # self.context.notify_all("낙상이 감지되었습니다. 괜찮으시면 버튼을 눌러주세요.")

    def handle_event(self, event):
        """이벤트 처리"""
        if event == "BUTTON_PRESSED":
            self.context.change_state(HomeState, "사용자 버튼 응답")
        elif event == "KEYWORD_DETECTED":
            self.context.change_state(EmergencyState, "비상 키워드")

    def update(self):
        """
        주기적 업데이트: 시간 내 미응답 시 Emergency로 전환.
        ALERT_TIMEOUT(20초) 이상 응답이 없으면 긴급 상황으로 판단.
        """
        if time.time() - self.start_time > config.ALERT_TIMEOUT:
            self.context.change_state(EmergencyState, "버튼 미응답")


class AwayState(State):
    """
    [Away] 외출 모드.
    사람이 감지되지 않은 상태.
    AWAY_TIMEOUT(24시간) 초과 시 Emergency로 전환.
    """
    def on_enter(self):
        print(">> [AWAY] 외출 모드")

    def handle_event(self, event):
        """이벤트 처리"""
        if event == "PERSON_DETECTED":
            self.context.change_state(HomeState, "사람 감지 (복귀)")

    def update(self):
        """
        주기적 업데이트: 장시간 미복귀 시 Emergency로 전환.
        AWAY_TIMEOUT(24시간) 이상 사람이 감지되지 않으면 긴급.
        """
        if time.time() - self.start_time > config.AWAY_TIMEOUT:
            self.context.change_state(EmergencyState, "장기 외출 미복귀")


class EmergencyState(State):
    """
    [Emergency] 비상 상황 상태.
    보호자에게 즉시 알림 발송.
    버튼 응답으로만 Home으로 복귀 가능.
    """
    def on_enter(self):
        print(">> [EMERGENCY] 비상 상황 발생!")
        # [로직] 등록된 모든 Notifier를 통해 알림 발송
        self.context.notify_all("비상 상황 발생! 즉시 확인 바랍니다.")

    def handle_event(self, event):
        """이벤트 처리"""
        if event == "BUTTON_PRESSED":
            self.context.change_state(HomeState, "비상 해제 (버튼)")

    def update(self):
        """
        주기적 업데이트: 현재는 수동 해제만 가능.
        (필요 시 자동 복귀 로직 추가 가능)
        """
        pass
