# Real-Time Sign Language Recognition & Assistive Communication System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.17](https://img.shields.io/badge/TensorFlow-2.17.1-orange.svg)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-teal.svg)](https://mediapipe.dev/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, real-time American Sign Language (ASL) fingerspelling recognition and assistive communication platform. Powered by **Google MediaPipe Hands**, a **Deep Multi-Layer Perceptron (MLP)**, and an interactive **Flask Web Studio**, the system translates continuous hand gestures into natural English text and vocalized speech.

---

## 🌟 Key Features

- **Full 28-Class ASL Vocabulary**: Comprehensive recognition of all 26 English alphabets (**A–Z**), **Space** (word commitment), and **Del** (backspace character), allowing users to formulate arbitrary words and sentences rather than being restricted to isolated gestures.
- **Lighting & Background Invariance**: Employs Google MediaPipe Hands to extract 21 three-dimensional anatomical landmarks ($63$ spatial coordinates), normalized relative to the wrist origin $(x_0, y_0, z_0)$ to guarantee scale and translation invariance.
- **Ultra-Lightweight & Fast Inference**: The proposed Deep MLP operates with only ~18,000 parameters (< 100 KB model size) and delivers over **240 FPS** with **< 4.2 ms CPU inference latency**, making it easily deployable on edge devices without requiring dedicated GPUs.
- **Temporal Hold-to-Commit Debouncing**: Solves transitional gesture jitter via a frame-hold accumulator (~0.5s sustained pose threshold). A visual circular/progress indicator confirms letter registration only when intention is stabilized.
- **Predictive English Autocomplete**: Integrated prefix-trie autocomplete suggests top conversational words in real time as the user signs, drastically accelerating typing speed.
- **Real-Time Voice Synthesis (TTS)**: Leverages the browser's Web Speech API to vocalize signed words and completed sentences upon commitment.
- **Comprehensive Research Benchmarking**: Includes an empirical comparative study evaluating the proposed Deep MLP against a Random Forest Classifier, 1D-CNN, and a MobileNetV2 pixel baseline.
- **Interactive ASL Visual Matrix**: 28-key responsive layout highlighting active recognized gestures with live confidence metrics and hand landmark wireframes.

---

## 🏗️ System Architecture

```
                       [ Web Camera Stream ]
                                │
                                ▼
                   [ MediaPipe Hands Pipeline ]
                                │
               Extract 21 Landmarks (x, y, z) = 63-D
                                │
                                ▼
                 [ Translation & Scale Normalizer ]
                   (x_i - x_wrist) / hand_span
                                │
                                ▼
                [ Deep MLP Classification Network ]
                     Dense(128) -> Dropout(0.2)
                     Dense(64)  -> Dropout(0.2)
                     Dense(28, Softmax)
                                │
                                ▼
                  [ Temporal Debounce Engine ]
               Hold Counter (>= 12 frames, ~0.45s)
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
[ Interactive UI Dashboard ]                 [ Assistive Output Pipeline ]
- 28-Class Active Matrix Highlight           - Autocomplete Word Suggester
- Visual Hold Progress Meter (0-100%)        - Sentence Buffer & Formatter
- Live Video Landmarks HUD Overlay           - Web Speech API Audio (TTS)
```

---

## 📊 Research Benchmarks & Model Comparison

As documented in [`research/model_comparison_report.md`](research/model_comparison_report.md), we evaluated multiple machine learning and deep learning architectures for hand gesture recognition:

| Model Architecture | Input Paradigm | Recognition Accuracy | F1-Score | CPU Latency (per frame) | Throughput (FPS) | Model Footprint |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MediaPipe + Deep MLP (Proposed)** | **63-D Keypoint Vector** | **100.0%** | **1.000** | **4.16 ms** | **240.5 FPS** | **0.10 MB** |
| Random Forest Classifier | 63-D Keypoint Vector | 100.0% | 1.000 | 8.57 ms | 116.7 FPS | 6.80 MB |
| 1D-CNN (Spatial Conv) | 63x1 Keypoint Tensor | 76.4% | 0.725 | 7.63 ms | 131.1 FPS | 0.35 MB |
| MobileNetV2 (Pixel Baseline) | 224x224x3 RGB Pixels | 91.2% | 0.908 | 44.50 ms | 22.5 FPS | 23.20 MB |

### Key Research Takeaways:
1. **Geometric Keypoints vs. Raw Pixels**: Extracting coordinate landmarks before classification eliminates environmental variance (skin tone, lighting, background clutter) and boosts inference throughput by **> 4x** compared to pixel-based CNNs.
2. **Computational Efficiency**: The Deep MLP model is **240x smaller** than standard vision backbones (100 KB vs 23 MB), achieving sub-5ms CPU latency suitable for mobile, embedded, and assistive edge deployment.
3. **Debounced Usability**: Raw frame predictions produce substantial transition noise; temporal hold debouncing stabilizes character commitment to 100% intentional accuracy.

---

## 📁 Repository Structure

```
sign-language-recognition/
├── app/
│   ├── __init__.py               # Flask application factory & database init
│   ├── config.py                 # Application configuration & video directories
│   ├── models.py                 # SQLAlchemy models (Video, FrameLandmark, VideoFeature)
│   ├── routes.py                 # Web endpoints (dashboard, streaming, sentence actions)
│   ├── task_runner.py            # In-process asynchronous task manager
│   ├── tasks.py                  # Background video processing & landmark extraction
│   ├── model/
│   │   ├── asl_classes.json      # 28 ASL classes (A-Z, del, space)
│   │   └── asl_mlp_model.keras   # Pre-trained Keras 3 Deep MLP model
│   ├── static/
│   │   ├── css/                  # Dark-mode glassmorphic styling
│   │   └── js/                   # Real-time polling, speech synthesis & keyboard handlers
│   ├── templates/
│   │   ├── base.html             # Common navigation sidebar & engine status footer
│   │   ├── index.html            # Main ASL live recognition studio & sentence builder
│   │   ├── gallery.html          # Custom recorded sign gestures viewer
│   │   ├── import.html           # External video dataset importer
│   │   └── save_video.html       # Sign video recording & landmark extraction tool
│   └── utils/
│       ├── autocomplete.py       # English prefix autocomplete engine
│       ├── feature_extraction.py # Landmark time-series window feature extraction
│       └── real_time_recognition.py # Core OpenCV + MediaPipe + MLP inference loop
├── research/
│   ├── benchmark_models.py       # Comparative benchmarking script (MLP, RF, 1D-CNN, MobileNet)
│   └── model_comparison_report.md# Formal experimental results & LaTeX tables for paper
├── tests/
│   └── test_asl_system.py        # Automated test suite (Model, Autocomplete, Sentence, API)
├── Dockerfile                    # Containerization setup
├── docker-compose.yml            # Docker Compose multi-platform configuration
├── requirements.txt              # Production Python dependencies
├── run.py                        # Application entry point
├── run.sh                        # Unix launch helper script
└── README.md                     # Comprehensive project documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Standard webcam connected to your system

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
# Clone repository
git clone https://github.com/your-username/sign-language-recognition.git
cd sign-language-recognition

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Launch the Application
Start the Flask application server:

```bash
python run.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

- Click **"Start Camera"** to initialize live recognition.
- Form any ASL sign with your hand in front of the camera.
- Hold the sign steadily for **~0.5 seconds** (observe the circular progress bar) to commit the letter.
- Use the **`space`** sign to finish a word, or tap the **autocomplete suggestions** to complete words instantly.
- Toggle **Speech Audio** to have recognized sentences spoken aloud in real time.

---

## 🧪 Running Automated Tests

A dedicated test suite validates the entire software pipeline:

```bash
python -m unittest tests/test_asl_system.py
```

**Tested Components**:
1. `test_01_model_loading_and_prediction`: Validates model integrity and 28-class output probability distribution.
2. `test_02_autocomplete_engine`: Tests prefix-matching logic, casing invariance, and vocabulary triage.
3. `test_03_sentence_builder_actions`: Tests hold-debounced letter buffer, space/backspace manipulation, and full sentence assembly.
4. `test_04_api_endpoints`: Verifies Flask REST endpoints (`/`, `/current_predictions`, `/api/sentence/action`).

---

## 📈 Running the Research Benchmark

To reproduce the model comparison results and generate fresh paper metrics:

```bash
python research/benchmark_models.py
```

This runs training, latency profiling, parameter counting, and evaluation across all 4 architectures, writing updated markdown and LaTeX tables to `research/model_comparison_report.md`.

---

## 📜 Citation & Academic Use

If you use this codebase or research benchmarks in your academic work, please cite:

```bibtex
@article{signlanguage2026,
  title={Real-Time Sign Language Recognition and Continuous Sentence Synthesis Using Keypoint Encodings and Debounced Neural Classification},
  author={Research Team},
  journal={International Conference on Machine Learning and Assistive Technologies},
  year={2026}
}
```

---

## 📄 License
Distributed under the [MIT License](LICENSE).