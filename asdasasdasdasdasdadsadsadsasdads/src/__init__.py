# src/__init__.py
"""
Care System Package

독거노인 케어 시스템 (RPi 5 기반)
낙상 감지, 응답 모니터링, 긴급 알림 시스템
"""

__version__ = "1.0.0"
__author__ = "Care System Team"

from src.hardware import ButtonReader, Speaker
from src.processors import VideoProcessor, AudioProcessor
from src.notifiers import ConsoleNotifier
from src.interfaces import INotifier
from src.states import Context, State, HomeState, AlertState, AwayState, EmergencyState

__all__ = [
    "ButtonReader",
    "Speaker",
    "VideoProcessor",
    "AudioProcessor",
    "ConsoleNotifier",
    "INotifier",
    "Context",
    "State",
    "HomeState",
    "AlertState",
    "AwayState",
    "EmergencyState",
]
