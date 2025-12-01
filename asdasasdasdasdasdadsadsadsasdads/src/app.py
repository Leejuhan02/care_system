#!/usr/bin/env python3
"""
Standalone fall-detection application.

Modes:
 - tflite : use a TFLite model (`--model path/to/fall_detection.tflite`)
 - heuristic : use OpenCV HOG/person detector + simple temporal rules

This file is designed to run inside the Python 3.11 venv on Raspberry Pi 5.
"""
import argparse
import time
import sys
import collections
import numpy as np
import cv2


def load_tflite_interpreter(model_path):
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:
        try:
            from tensorflow.lite import Interpreter
        except Exception:
            raise RuntimeError("No TFLite runtime found. Install tflite-runtime or tensorflow.")

    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def tflite_process_loop(interpreter, camera_index=0, display=False, threshold=0.5):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_info = input_details[0]
    input_shape = input_info['shape']
    input_dtype = np.dtype(input_info['dtype'])

    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    print(f"TFLite model input shape {input_shape}, dtype {input_dtype}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Resize according to model
            if len(input_shape) == 4:
                _, h, w, c = input_shape
            elif len(input_shape) == 3:
                h, w, c = input_shape
            else:
                raise ValueError('Unsupported model input shape')

            if c == 1:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (w, h))
                inp = img.reshape((1, h, w, 1)).astype(input_dtype)
            else:
                img = cv2.resize(frame, (w, h))
                inp = img.reshape((1, h, w, c)).astype(input_dtype)

            if input_dtype == np.float32:
                inp = inp / 255.0

            interpreter.set_tensor(input_details[0]['index'], inp)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]['index'])

            fall = False
            score = None
            if out.size == 1:
                score = float(out.flatten()[0])
                fall = score >= threshold
            else:
                if out.ndim == 2:
                    cls = int(np.argmax(out[0]))
                else:
                    cls = int(np.argmax(out))
                fall = (cls == 1)

            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            if fall:
                print(f"[{ts}] FALL detected (score={score})")
            else:
                print(f"[{ts}] ok (score={score})", end='\r')

            if display:
                txt = 'FALL' if fall else 'OK'
                color = (0, 0, 255) if fall else (0, 255, 0)
                disp = frame.copy()
                cv2.putText(disp, f"{txt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                if score is not None:
                    cv2.putText(disp, f"{score:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.imshow('app', disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()


class HeuristicFallDetector:
    """
    Very simple heuristic fall detector:
    - Detect person using HOG person detector
    - Track bounding-box centroid Y across last N frames
    - If centroid Y drops sharply (person lying) or bbox aspect ratio changes, trigger fall
    This is intentionally simple and meant as a fallback for demo/testing.
    """
    def __init__(self, buffer_len=12, drop_thresh=0.12, ar_thresh=0.7):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.centers = collections.deque(maxlen=buffer_len)
        self.heights = collections.deque(maxlen=buffer_len)
        self.drop_thresh = drop_thresh
        self.ar_thresh = ar_thresh

    def detect(self, frame):
        # returns (fall_detected:bool, info:dict)
        gray = cv2.resize(frame, (640, 480))
        rects, weights = self.hog.detectMultiScale(gray, winStride=(8,8), padding=(8,8), scale=1.05)

        if len(rects) == 0:
            return False, {'reason': 'no_person'}

        # choose largest rect
        areas = [w*h for (x,y,w,h) in rects]
        idx = int(np.argmax(areas))
        x,y,w,h = rects[idx]

        center_y = (y + y + h) / 2.0 / gray.shape[0]
        ar = float(h) / float(w)  # height/width

        self.centers.append(center_y)
        self.heights.append(ar)

        fall = False
        reason = 'ok'

        if len(self.centers) >= 6:
            # centroid drop: compare average of older half vs recent half
            half = len(self.centers) // 2
            older = np.mean(list(self.centers)[:half])
            recent = np.mean(list(self.centers)[half:])
            drop = recent - older
            # when person falls, center_y usually increases (lower in frame)
            if drop > self.drop_thresh:
                fall = True
                reason = f'center_drop:{drop:.3f}'

            # aspect ratio change: lying person -> height reduced relative to width
            older_h = np.mean(list(self.heights)[:half])
            recent_h = np.mean(list(self.heights)[half:])
            if recent_h < older_h * self.ar_thresh:
                fall = True
                reason = f'ar_change:{recent_h:.3f}->{older_h:.3f}'

        info = {'center_y': center_y, 'aspect_ratio': ar, 'rect': (x,y,w,h), 'reason': reason}
        return fall, info


def heuristic_loop(camera_index=0, display=False):
    det = HeuristicFallDetector()
    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            fall, info = det.detect(frame)
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            if fall:
                print(f"[{ts}] FALL detected (info={info})")
            else:
                print(f"[{ts}] OK (info={info})", end='\r')

            if display and 'rect' in info:
                x,y,w,h = info['rect']
                disp = cv2.resize(frame, (640,480)).copy()
                cv2.rectangle(disp, (x,y), (x+w, y+h), (0,255,0), 2)
                if fall:
                    cv2.putText(disp, 'FALL', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                cv2.imshow('heuristic', disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description='Standalone fall-detection app (tflite or heuristic)')
    p.add_argument('--mode', choices=['tflite','heuristic'], default='heuristic', help='Detection mode')
    p.add_argument('--model', help='Path to .tflite model (required for tflite mode)')
    p.add_argument('--camera', default=0, help='Camera index (default 0)')
    p.add_argument('--display', action='store_true', help='Display camera window')
    p.add_argument('--threshold', type=float, default=0.5, help='Threshold for tflite probability models')
    args = p.parse_args()

    if args.mode == 'tflite':
        if not args.model:
            print('Error: --model is required for tflite mode', file=sys.stderr)
            sys.exit(2)
        interp = load_tflite_interpreter(args.model)
        tflite_process_loop(interp, camera_index=args.camera, display=args.display, threshold=args.threshold)
    else:
        heuristic_loop(camera_index=args.camera, display=args.display)


if __name__ == '__main__':
    main()
