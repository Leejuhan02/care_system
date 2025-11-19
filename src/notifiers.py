# src/notifiers.py
from src.interfaces import INotifier

class ConsoleNotifier(INotifier):
    """
    SMS 기능을 대체하는 콘솔 출력 알림 구현체.
    실제 통신 비용 없이 로직을 검증할 수 있습니다.
    """
    def send(self, message: str) -> bool:
        # [로직] 실제 전송 대신 프롬프트 출력
        print("\n" + "="*50)
        print(f"[SMS 발송 시뮬레이션] >> {message}")
        print("="*50 + "\n")
        return True

# --- 확장 예시 (주석) ---
# class TwilioNotifier(INotifier):
#     def __init__(self, sid, token, to_num, from_num):
#         ...
#     def send(self, message: str) -> bool:
#         # Twilio API 호출 로직
#         ...
