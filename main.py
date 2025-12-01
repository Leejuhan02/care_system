# main.py
import time
import multiprocessing
import sys
from src.hardware import ButtonReader, Speaker
from src.processors import VideoProcessor, AudioProcessor
from src.notifiers import ConsoleNotifier
from src.states import Context

def main():
    """
    독거노인 케어 시스템 메인 함수.
    RPi 5 환경에서 동작하는 멀티프로세싱 기반 모니터링 시스템.
    """
    print("=== 독거노인 케어 시스템 (RPi 5 / Python 3.11) ===")
    
    # 1. 통신용 큐 생성
    # 모든 이벤트는 이 하나의 큐로 모입니다.
    event_queue = multiprocessing.Queue()

    video_proc = None
    audio_proc = None
    button = None
    
    try:
        # 2. 하드웨어 및 알림 객체 초기화 (의존성 생성)
        speaker = Speaker()
        button = ButtonReader(event_queue)
        
        # [OCP] 알림 전략 주입. 나중에 TwilioNotifier() 등을 리스트에 추가하면 됨.
        notifiers = [ConsoleNotifier()]
        
        # 3. 상태 머신 초기화 (의존성 주입)
        context = Context(speaker, notifiers)

        # 4. AI 프로세스 생성 및 시작
        video_proc = VideoProcessor(event_queue)
        audio_proc = AudioProcessor(event_queue)
        
        video_proc.start()
        audio_proc.start()
        
        # 버튼 감시 시작
        if button:
            button.start()
        else:
            print("button false")

        print("[Main] 시스템 가동 시작. (Ctrl+C로 종료)")

        # 5. 메인 이벤트 루프
        while True:
            # 큐가 비어있지 않으면 이벤트 처리
            while not event_queue.empty():
                try:
                    event = event_queue.get(timeout=0.1)
                    context.process_event(event)
                except:
                    break
            
            # 상태 머신 틱(타이머 등) 업데이트
            context.tick()
            
            # CPU 과점유 방지
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[Main] 시스템 종료 요청...")
    except Exception as e:
        print(f"\n[Main] 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 자원 정리
        print("[Main] 자원 정리 중...")
        
        if video_proc is not None:
            try:
                video_proc.terminate()
                video_proc.join(timeout=2)
            except Exception as e:
                print(f"[Main] 비디오 프로세스 종료 오류: {e}")
        
        if audio_proc is not None:
            try:
                audio_proc.terminate()
                audio_proc.join(timeout=2)
            except Exception as e:
                print(f"[Main] 오디오 프로세스 종료 오류: {e}")
        
        if button is not None:
            try:
                button.cleanup()
            except Exception as e:
                print(f"[Main] 버튼 정리 오류: {e}")
        
        print("[Main] 종료 완료.")


if __name__ == "__main__":
    # [RPi 5 호환] 멀티프로세싱 안전 시작 방식 설정
    # spawn 방식은 Windows와 RPi 모두 호환 가능
    multiprocessing.set_start_method('spawn', force=True)
    main()
