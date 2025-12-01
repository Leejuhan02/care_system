#!/usr/bin/env python3
"""
Launcher script for the standalone app.

Usage example:
    source /home/raspberry/my_venv_311/bin/activate
    python run_app.py --mode heuristic --camera 0 --display
    python run_app.py --mode tflite --model models/fall_detection.tflite --camera 0
"""
import sys
from src.app import main as app_main

if __name__ == '__main__':
    app_main()
