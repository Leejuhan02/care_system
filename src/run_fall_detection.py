#!/usr/bin/env python3
"""
Simple real-time TFLite fall-detection runner using OpenCV camera input.

Usage example:
    python src/run_fall_detection.py --model ../models/fall_detection.tflite --camera 0 --display

Notes:
- The script will try to import `tflite_runtime.Interpreter` first; if unavailable it falls back to
  `tensorflow.lite.Interpreter` (requires full TensorFlow installation).
- Adjust preprocessing to match your model (normalization, channel order, etc.).
"""
import argparse
import time
import sys
import numpy as np
import cv2


def load_interpreter(model_path):
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:
        try:
            from tensorflow.lite import Interpreter
        except Exception:
            raise RuntimeError("No TFLite runtime found. Install tflite-runtime or tensorflow.")

    interp = Interpreter(model_path=model_path)
    return interp


def preprocess_frame(frame, input_shape, input_dtype):
    # input_shape expected like [1, h, w, c]
    if len(input_shape) == 4:
        _, h, w, c = input_shape
    elif len(input_shape) == 3:
        h, w, c = input_shape
    else:
        raise ValueError(f"Unsupported input shape: {input_shape}")

    if c == 1:
        frame_proc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_proc = cv2.resize(frame_proc, (w, h))
        frame_proc = frame_proc.reshape((h, w, 1))
    else:
        frame_proc = cv2.resize(frame, (w, h))

    arr = np.expand_dims(frame_proc, axis=0)

    if input_dtype == np.float32:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(input_dtype)

    return arr


def infer_loop(interpreter, camera_index=0, display=False, threshold=0.5):
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_info = input_details[0]
    input_shape = input_info['shape']
    input_dtype = np.dtype(input_info['dtype'])

    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    print(f"Model input shape: {input_shape}, dtype: {input_dtype}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            inp = preprocess_frame(frame, input_shape, input_dtype)
            interpreter.set_tensor(input_details[0]['index'], inp)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]['index'])

            # Interpret output: support single-prob or multi-class
            fall_detected = False
            score = None

            if out.size == 1:
                score = float(out.flatten()[0])
                fall_detected = score >= threshold
            else:
                # assume shape (1, N) -> argmax
                if out.ndim == 2:
                    cls = int(np.argmax(out[0]))
                else:
                    cls = int(np.argmax(out))
                score = None
                fall_detected = (cls == 1)

            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            if fall_detected:
                print(f"[{ts}] FALL detected! (score={score})")
            else:
                print(f"[{ts}] ok (score={score})", end='\r')

            if display:
                txt = 'FALL' if fall_detected else 'OK'
                color = (0, 0, 255) if fall_detected else (0, 255, 0)
                disp = frame.copy()
                cv2.putText(disp, f"{txt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                if score is not None:
                    cv2.putText(disp, f"{score:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.imshow('fall-detection', disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description='Run TFLite fall detection on camera')
    p.add_argument('--model', required=True, help='Path to .tflite model')
    p.add_argument('--camera', default=0, help='Camera index or path (default 0)')
    p.add_argument('--display', action='store_true', help='Show camera output window')
    p.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for fall detection')
    args = p.parse_args()

    interp = load_interpreter(args.model)
    try:
        infer_loop(interp, camera_index=args.camera, display=args.display, threshold=args.threshold)
    except KeyboardInterrupt:
        print('\nInterrupted by user')


if __name__ == '__main__':
    main()
