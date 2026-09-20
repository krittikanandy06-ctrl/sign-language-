#!/usr/bin/env python3
"""
Standalone Desktop OpenCV Inference Script for ASL 28-Class Recognition.
Runs directly in an OpenCV window without needing a web browser.
Press 'q' or ESC to exit.
"""

import cv2
import sys
from app.utils.real_time_recognition import get_recognizer


def main():
    print("=" * 60)
    print("  ASL 28-Class Real-Time Recognition (Standalone OpenCV Window)")
    print("  - Hold gesture ~0.5s to commit letter")
    print("  - 'space' sign commits word, 'del' sign backspaces")
    print("  - Press 'q' or ESC in video window to exit")
    print("=" * 60)

    recognizer = get_recognizer()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Ensure your camera is connected and not in use.")
        sys.exit(1)

    # Initialize MediaPipe Hands in recognizer
    recognizer.start_capture()
    print("[SUCCESS] Camera initialized. Show hand signs in front of the lens.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror frame horizontally for natural user experience
            frame = cv2.flip(frame, 1)

            # Process frame through MediaPipe + Deep MLP + Debouncer
            processed_frame = recognizer.process_frame(frame)

            # Display in OpenCV desktop window
            cv2.imshow("ASL Fingerspelling & Sentence Recognition", processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('c'):  # 'c' to clear
                recognizer.action_clear()
            elif key == ord(' '):  # spacebar
                recognizer.action_space()
            elif key == 8:  # backspace
                recognizer.action_backspace()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.stop_capture()
        print("\nSession ended. Goodbye!")


if __name__ == "__main__":
    main()
