import time
import cv2
from flask import (
    render_template,
    request,
    jsonify,
    current_app as app,
    Response,
)

from app.utils.real_time_recognition import get_recognizer


@app.route("/")
def index():
    """Main ASL Fingerspelling & Sentence Studio dashboard"""
    return render_template("index.html")


# --- Video Streaming for Real-Time Recognition ---
def generate_frames():
    """Generate frames for video streaming with safe throttling and error handling"""
    recognizer = get_recognizer()

    while True:
        try:
            frame = recognizer.get_frame()
            if frame is None:
                time.sleep(0.04)
                continue

            # When camera is on standby, stream placeholder at ~10 FPS with minimal CPU/bandwidth
            if not recognizer.is_running:
                ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                    )
                time.sleep(0.1)
                continue

            # Live camera stream: encode at 80% JPEG quality, throttled to ~25-30 FPS
            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                time.sleep(0.02)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            time.sleep(0.033)
        except GeneratorExit:
            # Client closed stream (browser tab closed / navigated away)
            break
        except Exception:
            time.sleep(0.05)


@app.route("/video_feed")
def video_feed():
    """Video streaming route for live MediaPipe hand tracking overlay"""
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/current_predictions")
def current_predictions():
    """Get current real-time predictions and sentence engine state as JSON"""
    recognizer = get_recognizer()
    state = recognizer.get_state()

    if not recognizer.is_running:
        return jsonify(
            {
                "status": "standby",
                "current_prediction": "Standby",
                "current_char": "--",
                "char_progress": 0,
                "confidence": 0.0,
                "current_word": "",
                "sentence": "",
                "full_text": "",
                "suggestions": [],
                "fps": 0.0,
                "handedness": "None",
            }
        )

    return jsonify(
        {
            "status": "success",
            "current_prediction": state["current_char"],
            "current_char": state["current_char"],
            "char_progress": state["char_progress"],
            "confidence": state["confidence"],
            "current_word": state["current_word"],
            "sentence": state["sentence"],
            "full_text": state["full_text"],
            "suggestions": state["suggestions"],
            "fps": state["fps"],
            "engine_status": state["status"],
            "handedness": state["handedness"],
        }
    )


@app.route("/api/sentence/action", methods=["POST"])
def sentence_action():
    """Handle interactive sentence actions: space, backspace, clear, select suggestion"""
    data = request.get_json() or {}
    action = data.get("action")
    recognizer = get_recognizer()

    if action == "space":
        recognizer.action_space()
    elif action == "backspace":
        recognizer.action_backspace()
    elif action == "clear":
        recognizer.action_clear()
    elif action == "suggestion":
        word = data.get("word", "")
        recognizer.action_select_suggestion(word)
    else:
        return jsonify({"status": "error", "message": f"Unknown action {action}"}), 400

    return jsonify({"status": "success", "state": recognizer.get_state()})


@app.route("/toggle_recognition", methods=["POST"])
def toggle_recognition():
    """Start or stop real-time camera recognition"""
    action = request.json.get("action", "start")
    recognizer = get_recognizer()

    if action == "start":
        success = recognizer.start_capture()
        if success:
            return jsonify({"status": "started", "success": True})
        else:
            return jsonify(
                {"status": "error", "success": False, "message": "Failed to start camera"}
            )
    else:
        recognizer.stop_capture()
        return jsonify({"status": "stopped", "success": True})


@app.route("/recognition_status")
def recognition_status():
    """Get current recognition status"""
    recognizer = get_recognizer()
    return jsonify({"is_running": recognizer.is_running})
