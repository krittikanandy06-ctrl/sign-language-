#!/usr/bin/env python3
"""
Real-Time ASL Alphabet & Sentence Formation Engine.
Uses MediaPipe Hands and Deep MLP model with hold-to-commit debouncing,
word autocomplete suggestions, and speech synchronization.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import json
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
import threading
import time

from app.utils.autocomplete import AutocompleteEngine

warnings.filterwarnings("ignore")


class RealTimeSignLanguageRecognizer:
    def __init__(
        self,
        model_path="app/model/asl_mlp_model.keras",
        classes_path="app/model/asl_classes.json",
        cam_id=0,
    ):
        print("Loading ASL Alphabet model and classes...")
        self.model = load_model(model_path)
        with open(classes_path, "r") as f:
            self.classes = json.load(f)

        print(f"Loaded ASL model with {len(self.classes)} classes: {self.classes}")

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = None

        # Autocomplete Engine
        self.autocomplete = AutocompleteEngine()

        # Camera & Processing Parameters
        self.CAM_ID = cam_id
        self.cap = None
        self.is_running = False

        # Sentence Builder State
        self.sentence_words = []
        self.current_word = ""
        self.current_char = ""
        self.confirmed_char = ""
        self.char_progress = 0
        self.suggestions = []
        self.status = "Idle"
        self.handedness_str = "None"

        # Hold-to-commit debouncing mechanics
        self.MIN_HOLD_FRAMES = 12       # ~0.4-0.5s hold required to commit a letter
        self.COOLDOWN_FRAMES = 10       # Frames to wait after a commit before re-triggering
        self.consecutive_frames = 0
        self.candidate_char = None
        self.cooldown_counter = 0

        # Smoothing & Stats
        self.confidence = 0.0
        self.confidence_threshold = 0.60
        self.prediction_history = deque(maxlen=5)

        # FPS calculation
        self.frame_count = 0
        self.fps_timer = cv2.getTickCount()
        self.current_fps = 0.0

        # Thread synchronization
        self.lock = threading.Lock()

        print("ASL Recognizer initialization complete!")

    def _init_mediapipe(self):
        """Initialize MediaPipe Hands on camera start."""
        if self.hands is None:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )

    def start_capture(self):
        """Start webcam capture with Windows DirectShow backend."""
        if self.cap is None or not self.cap.isOpened():
            self._init_mediapipe()
            # Use CAP_DSHOW on Windows to eliminate camera initialization delays
            backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(self.CAM_ID, backend)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(self.CAM_ID)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.is_running = True
                self.frame_count = 0
                self.fps_timer = cv2.getTickCount()
                print("Webcam capture started successfully.")
                return True
            else:
                print("Failed to open webcam.")
                self.is_running = False
                return False
        return self.is_running

    def stop_capture(self):
        """Stop video capture and release camera resources."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.hands is not None:
            try:
                self.hands.close()
            except Exception:
                pass
            self.hands = None
        print("Webcam capture stopped.")

    def get_frame(self):
        """Read frame from webcam, process it, or return standby screen."""
        if not self.is_running or self.cap is None:
            # High-aesthetic dark standby screen
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            placeholder[:] = (20, 22, 28)
            cv2.putText(
                placeholder,
                "Camera on Standby",
                (185, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (180, 185, 195),
                2,
            )
            cv2.putText(
                placeholder,
                "Click 'Start Recognition' to begin ASL fingerspelling",
                (120, 265),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (110, 115, 130),
                1,
            )
            return placeholder

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        return self.process_frame(frame)

    def _update_fps(self):
        """Calculate running FPS."""
        self.frame_count += 1
        if self.frame_count >= 15:
            now = cv2.getTickCount()
            elapsed = (now - self.fps_timer) / cv2.getTickFrequency()
            self.current_fps = round(self.frame_count / elapsed, 1) if elapsed > 0 else 0.0
            self.fps_timer = now
            self.frame_count = 0

    def process_frame(self, frame):
        """Extract hand landmarks, run inference, and update sentence engine."""
        if frame is None:
            return frame

        # Horizontal mirror for natural interaction
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detected_sign = None
        conf = 0.0

        try:
            results = self.hands.process(rgb_frame)
        except Exception as e:
            # MediaPipe timestamp or lifecycle error guard
            print(f"MediaPipe process error: {e}")
            return frame

        if results and results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label
            self.handedness_str = handedness

            # Draw sleek landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Extract 21 landmark 3D coordinates
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            # Mirror x-coordinates for Right hand so both hands work with left-hand trained data
            if handedness == "Right":
                landmarks[:, 0] = 1.0 - landmarks[:, 0]

            # Flatten to 63-D vector
            input_vector = landmarks.flatten().reshape(1, -1)

            # Fast MLP inference (~2ms on CPU)
            predictions = self.model.predict(input_vector, verbose=0)[0]
            pred_idx = int(np.argmax(predictions))
            conf = float(predictions[pred_idx])

            if conf >= self.confidence_threshold:
                detected_sign = self.classes[pred_idx]
            else:
                detected_sign = None
        else:
            self.handedness_str = "None"
            detected_sign = None
            conf = 0.0

        # Update Sentence Debounce Engine
        self._update_sentence_engine(detected_sign, conf)

        # Draw HUD overlays on frame
        self._draw_overlay(frame)

        self._update_fps()
        return frame

    def _update_sentence_engine(self, detected_sign, conf):
        """Debounce candidate characters and commit confirmed letters/actions."""
        with self.lock:
            self.confidence = round(conf * 100, 1)

            # Handle cooldown after a character commit
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                self.char_progress = 0
                self.status = "Cooling down..."
                return

            if detected_sign is None:
                # No hand or confidence too low
                self.consecutive_frames = 0
                self.candidate_char = None
                self.char_progress = 0
                self.status = "Waiting for hand..." if self.handedness_str == "None" else "Uncertain gesture"
                return

            # Steady sign tracking
            if detected_sign == self.candidate_char:
                self.consecutive_frames += 1
            else:
                self.candidate_char = detected_sign
                self.consecutive_frames = 1

            self.current_char = self.candidate_char
            self.char_progress = min(100, int((self.consecutive_frames / self.MIN_HOLD_FRAMES) * 100))
            self.status = f"Holding sign: {self.current_char}"

            # If held steadily for MIN_HOLD_FRAMES, commit character!
            if self.consecutive_frames >= self.MIN_HOLD_FRAMES:
                self._commit_character(self.candidate_char)
                self.cooldown_counter = self.COOLDOWN_FRAMES
                self.consecutive_frames = 0
                self.candidate_char = None
                self.char_progress = 0

    def _commit_character(self, char):
        """Execute the confirmed letter or control command."""
        self.confirmed_char = char

        if char == "space":
            # Commit current word to the sentence
            if self.current_word:
                self.sentence_words.append(self.current_word)
                self.current_word = ""
                self.suggestions = []
        elif char == "del":
            # Backspace: remove last letter or last word
            if self.current_word:
                self.current_word = self.current_word[:-1]
                self.suggestions = self.autocomplete.suggest(self.current_word)
            elif self.sentence_words:
                self.current_word = self.sentence_words.pop()
                self.suggestions = self.autocomplete.suggest(self.current_word)
        else:
            # Regular Alphabet letter
            self.current_word += char
            self.suggestions = self.autocomplete.suggest(self.current_word)

    # --- External Action APIs (Buttons / User Interactions) ---

    def action_space(self):
        """Manually trigger space / word separation."""
        with self.lock:
            if self.current_word:
                self.sentence_words.append(self.current_word)
                self.current_word = ""
                self.suggestions = []

    def action_backspace(self):
        """Manually trigger backspace."""
        with self.lock:
            if self.current_word:
                self.current_word = self.current_word[:-1]
                self.suggestions = self.autocomplete.suggest(self.current_word)
            elif self.sentence_words:
                self.current_word = self.sentence_words.pop()
                self.suggestions = self.autocomplete.suggest(self.current_word)

    def action_clear(self):
        """Clear the entire sentence and current word."""
        with self.lock:
            self.sentence_words = []
            self.current_word = ""
            self.suggestions = []
            self.confirmed_char = ""

    def action_select_suggestion(self, word):
        """Accept an autocomplete suggestion."""
        with self.lock:
            word = word.strip().upper()
            if word:
                self.sentence_words.append(word)
                self.current_word = ""
                self.suggestions = []

    def get_full_sentence(self):
        """Return the complete assembled sentence string."""
        with self.lock:
            parts = list(self.sentence_words)
            if self.current_word:
                parts.append(self.current_word)
            return " ".join(parts)

    def get_state(self):
        """Return full state snapshot for API / UI polling."""
        with self.lock:
            full_sentence = " ".join(self.sentence_words)
            return {
                "is_running": self.is_running,
                "fps": self.current_fps,
                "status": self.status,
                "current_char": self.current_char or "--",
                "char_progress": self.char_progress,
                "confidence": self.confidence,
                "current_word": self.current_word,
                "sentence": full_sentence,
                "full_text": (full_sentence + (" " + self.current_word if self.current_word else "")).strip(),
                "suggestions": list(self.suggestions),
                "handedness": self.handedness_str,
            }

    def _draw_overlay(self, frame):
        """Draw sleek, non-distracting heads-up display on frame."""
        h, w, _ = frame.shape

        # Top status bar background
        cv2.rectangle(frame, (0, 0), (w, 45), (15, 17, 23), -1)

        # Character preview & progress
        char_text = f"Sign: {self.current_char or '--'} ({self.confidence:.0f}%)"
        cv2.putText(frame, char_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

        # Hold Progress Bar
        bar_w = 120
        bar_x = w - bar_w - 90
        fill_w = int(bar_w * (self.char_progress / 100.0))
        cv2.rectangle(frame, (bar_x, 15), (bar_x + bar_w, 30), (50, 54, 66), -1)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, 15), (bar_x + fill_w, 30), (0, 200, 115), -1)
        cv2.putText(frame, f"{self.char_progress}%", (bar_x + bar_w + 8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 185, 200), 1)

        # FPS indicator
        fps_text = f"{self.current_fps} FPS"
        cv2.putText(frame, fps_text, (w - 75, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 180, 255), 1)

        # Bottom Subtitle / Sentence Ribbon
        ribbon_h = 55
        cv2.rectangle(frame, (0, h - ribbon_h), (w, h), (15, 17, 23), -1)
        
        full_text = ((" ".join(self.sentence_words) + " " + self.current_word)).strip()
        if not full_text:
            display_text = "Spell words with signs (A-Z) | Hold ~0.5s to commit"
            color = (120, 125, 140)
        else:
            display_text = full_text
            color = (255, 255, 255)

        cv2.putText(frame, display_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


# Global singleton instance
_recognizer_instance = None


def get_recognizer():
    global _recognizer_instance
    if _recognizer_instance is None:
        _recognizer_instance = RealTimeSignLanguageRecognizer()
    return _recognizer_instance
