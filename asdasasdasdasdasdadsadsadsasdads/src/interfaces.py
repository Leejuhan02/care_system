# src/interfaces.py
from abc import ABC, abstractmethod

class INotifier(ABC):
    """
    [OCP 적용] 알림 시스템을 위한 인터페이스.
    새로운 알림 방식(SMS, Email 등)을 추가할 때 기존 코드를 수정할 필요 없이
    이 클래스를 상속받아 구현하기만 하면 됩니다.
    """
    @abstractmethod
    def send(self, message: str) -> bool:
        """메시지를 발송하는 추상 메소드"""
        pass
