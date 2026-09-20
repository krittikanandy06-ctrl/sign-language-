/**
 * ASL Fingerspelling & Sentence Engine (100% Client-Side Pure JavaScript)
 * Zero server dependencies, 0ms network latency.
 * Uses MediaPipe Hands via CDN + 28-Class MLP model weights from asl_model_weights.js.
 */

// Common English dictionary for instant predictive autocomplete
const COMMON_WORDS = [
  "I", "ME", "MY", "YOU", "YOUR", "HE", "SHE", "IT", "WE", "THEY", "THIS", "THAT",
  "HELLO", "HI", "HEY", "GOOD", "MORNING", "EVENING", "NIGHT", "BYE", "GOODBYE",
  "PLEASE", "THANK", "THANKS", "WELCOME", "SORRY", "EXCUSE", "HELP", "YES", "NO",
  "AM", "IS", "ARE", "WAS", "WERE", "BE", "HAVE", "HAS", "HAD", "DO", "DOES", "DID",
  "WANT", "NEED", "LIKE", "LOVE", "KNOW", "THINK", "SEE", "LOOK", "HEAR", "FEEL",
  "COME", "GO", "WALK", "RUN", "STOP", "START", "WAIT", "CALL", "ASK", "TELL", "SAY",
  "SPEAK", "TALK", "WRITE", "READ", "LEARN", "TEACH", "STUDY", "EAT", "DRINK", "SLEEP",
  "WHAT", "WHEN", "WHERE", "WHO", "WHY", "HOW", "WHICH",
  "NAME", "FRIEND", "FAMILY", "MOTHER", "FATHER", "BROTHER", "SISTER",
  "WATER", "FOOD", "TEA", "COFFEE", "BREAD", "RICE", "MEDICINE", "DOCTOR", "HOSPITAL",
  "DAY", "TODAY", "TOMORROW", "YESTERDAY", "TIME", "HOME", "SCHOOL", "WORK", "CAR",
  "HAPPY", "SAD", "TIRED", "SICK", "FINE", "OKAY", "GREAT", "READY", "SURE", "BUSY",
  "AND", "OR", "BUT", "BECAUSE", "IF", "WITH", "FOR", "FROM", "TO", "IN", "ON", "AT"
];

// Complete ASL 28-Sign Anatomy & Hand Posture Dictionary
const SIGN_DESCRIPTIONS = {
  "A": { title: "Letter A", emoji: "✊", desc: "Closed fist with thumb resting straight upright along the outer side of the index finger." },
  "B": { title: "Letter B", emoji: "✋", desc: "Four fingers held flat and upright together; thumb folded flat across the front of the palm." },
  "C": { title: "Letter C", emoji: "🤏", desc: "Curved fingers and thumb forming a 'C' arc shape, like gripping an open coffee mug." },
  "D": { title: "Letter D", emoji: "☝️", desc: "Index finger points straight up; other three fingers curl to touch the thumb tip, forming an 'O' loop." },
  "E": { title: "Letter E", emoji: "✊", desc: "All four fingertips curled tightly down, resting directly on top of the tucked thumb." },
  "F": { title: "Letter F", emoji: "👌", desc: "Index finger and thumb touch tips forming an 'O' circle ('OK' sign); middle, ring, and pinky stay spread upright." },
  "G": { title: "Letter G", emoji: "👉", desc: "Index finger and thumb extended horizontally forward parallel to each other like a pinch." },
  "H": { title: "Letter H", emoji: "✌️", desc: "Index and middle fingers extended horizontally forward held together; thumb tucked underneath." },
  "I": { title: "Letter I", emoji: "🤙", desc: "Pinky finger points straight up; other three fingers curled into a fist with thumb folded across them." },
  "J": { title: "Letter J", emoji: "🤙", desc: "Start in 'I' pose (pinky up) and swoop pinky down to trace a curved 'J' motion in the air." },
  "K": { title: "Letter K", emoji: "✌️", desc: "Index finger straight up, middle finger angled forward; thumb rests between them against middle finger knuckle." },
  "L": { title: "Letter L", emoji: "👆", desc: "Index finger points straight up, thumb extends out horizontally at 90° right angle forming an 'L'." },
  "M": { title: "Letter M", emoji: "✊", desc: "Fist with thumb tucked underneath the first three fingers (index, middle, and ring drape over thumb)." },
  "N": { title: "Letter N", emoji: "✊", desc: "Fist with thumb tucked underneath the first two fingers (index and middle drape over thumb)." },
  "O": { title: "Letter O", emoji: "👌", desc: "All fingers curved down to meet the thumb tip, forming a circular 'O' shape." },
  "P": { title: "Letter P", emoji: "👇", desc: "Downward 'K' sign: hand pointing downwards with index finger horizontal and middle finger pointing down." },
  "Q": { title: "Letter Q", emoji: "👇", desc: "Downward 'G' sign: index finger and thumb pointed straight down like picking up a coin." },
  "R": { title: "Letter R", emoji: "🤞", desc: "Index and middle fingers crossed over each other like wishing for 'good luck'." },
  "S": { title: "Letter S", emoji: "👊", desc: "Tight fist with thumb folded horizontally across the front of all curled fingers (unlike A)." },
  "T": { title: "Letter T", emoji: "✊", desc: "Fist with thumb tucked between index and middle fingers, thumb tip poking out between them." },
  "U": { title: "Letter U", emoji: "✌️", desc: "Index and middle fingers extended straight up, held tightly together (palm forward)." },
  "V": { title: "Letter V", emoji: "✌️", desc: "Peace sign: index and middle fingers extended straight up and spread apart in a 'V' shape." },
  "W": { title: "Letter W", emoji: "🖐️", desc: "Three fingers up: index, middle, and ring fingers extended upright and spread apart in a 'W' shape." },
  "X": { title: "Letter X", emoji: "☝️", desc: "Fist with index finger bent at the knuckle into a hooked shape (like a pirate's hook)." },
  "Y": { title: "Letter Y", emoji: "🤙", desc: "Hang loose / Shaka sign: thumb and pinky extended out sideways, middle three fingers folded." },
  "Z": { title: "Letter Z", emoji: "👉", desc: "Index finger extended forward, drawing the zigzag strokes of the letter 'Z' in the air." },
  "space": { title: "Space (SPC)", emoji: "🫲", desc: "Flat open palm held horizontally facing downwards, committing the current word into the sentence." },
  "del": { title: "Delete (DEL)", emoji: "🔙", desc: "Closed fist with thumb pointed down or horizontal wave, erasing the last typed character (Backspace)." }
};

// Application State
let isRunning = false;
let ttsEnabled = true;
let handsInstance = null;
let cameraInstance = null;
let lastActiveChar = null;

// Sentence Builder & Debouncing State
let currentWord = "";
let sentenceWords = [];
let candidateChar = null;
let consecutiveFrames = 0;
let cooldownCounter = 0;
const MIN_HOLD_FRAMES = 12;    // ~0.45s hold steady to commit
const COOLDOWN_FRAMES = 10;    // cooldown after commit to avoid accidental repeats
const CONFIDENCE_THRESHOLD = 0.60;

// Running FPS
let frameCount = 0;
let lastFpsTime = performance.now();
let currentFps = 0;

/**
 * Pure JavaScript Matrix Forward Pass (63-D -> 128 -> 64 -> 28)
 * Mathematically identical to the TensorFlow/Keras trained model.
 */
function predictASL(landmarks, handedness) {
  if (typeof ASL_WEIGHTS === 'undefined' || typeof ASL_CLASSES === 'undefined') {
    return { char: null, confidence: 0 };
  }

  // 1. Landmark Flattening & Mirroring
  // 21 landmarks * 3 coords = 63 inputs
  const x = new Float32Array(63);
  const isRightHand = (handedness === "Right");

  for (let i = 0; i < 21; i++) {
    const lm = landmarks[i];
    // Mirror x-coordinate if right hand, matching Python training data
    const xCoord = isRightHand ? (1.0 - lm.x) : lm.x;
    x[i * 3 + 0] = xCoord;
    x[i * 3 + 1] = lm.y;
    x[i * 3 + 2] = lm.z;
  }

  const { w1, b1, w2, b2, w3, b3 } = ASL_WEIGHTS;

  // Layer 1: Dense(63 -> 128) + ReLU
  const h1 = new Float32Array(128);
  for (let j = 0; j < 128; j++) {
    let sum = b1[j];
    for (let i = 0; i < 63; i++) {
      sum += x[i] * w1[i][j];
    }
    h1[j] = sum > 0 ? sum : 0; // ReLU
  }

  // Layer 2: Dense(128 -> 64) + ReLU
  const h2 = new Float32Array(64);
  for (let k = 0; k < 64; k++) {
    let sum = b2[k];
    for (let j = 0; j < 128; j++) {
      sum += h1[j] * w2[j][k];
    }
    h2[k] = sum > 0 ? sum : 0; // ReLU
  }

  // Layer 3: Dense(64 -> 28) + Softmax
  const logits = new Float32Array(28);
  let maxLogit = -Infinity;
  for (let m = 0; m < 28; m++) {
    let sum = b3[m];
    for (let k = 0; k < 64; k++) {
      sum += h2[k] * w3[k][m];
    }
    logits[m] = sum;
    if (sum > maxLogit) maxLogit = sum;
  }

  // Softmax with numerical stability
  let sumExp = 0.0;
  const probs = new Float32Array(28);
  for (let m = 0; m < 28; m++) {
    const e = Math.exp(logits[m] - maxLogit);
    probs[m] = e;
    sumExp += e;
  }

  let maxProb = 0.0;
  let bestIdx = 0;
  for (let m = 0; m < 28; m++) {
    probs[m] /= sumExp;
    if (probs[m] > maxProb) {
      maxProb = probs[m];
      bestIdx = m;
    }
  }

  return {
    char: ASL_CLASSES[bestIdx],
    confidence: maxProb
  };
}

/**
 * Autocomplete prefix search
 */
function getSuggestions(prefix) {
  if (!prefix || prefix.length === 0) return [];
  const upper = prefix.toUpperCase();
  const results = [];
  for (const word of COMMON_WORDS) {
    if (word.startsWith(upper) && word !== upper) {
      results.push(word);
      if (results.length >= 4) break;
    }
  }
  return results;
}

/**
 * Speech Synthesis (TTS)
 */
function speakSentence(textToSpeak) {
  if (!ttsEnabled || !('speechSynthesis' in window)) return;
  const text = textToSpeak || getFullSentenceText();
  if (!text || text.trim() === "") return;

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
}

function getFullSentenceText() {
  const sentence = sentenceWords.join(" ");
  if (currentWord) {
    return sentence ? `${sentence} ${currentWord}` : currentWord;
  }
  return sentence;
}

function toggleTTS() {
  ttsEnabled = !ttsEnabled;
  const icon = document.getElementById('ttsIcon');
  const text = document.getElementById('ttsStatusText');
  if (ttsEnabled) {
    icon.className = 'bx bx-volume-full';
    text.innerText = 'Audio: ON';
    speakSentence("Audio readout enabled");
  } else {
    icon.className = 'bx bx-volume-mute';
    text.innerText = 'Audio: OFF';
    window.speechSynthesis.cancel();
  }
}

function copySentence() {
  const text = getFullSentenceText();
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copyBtn');
    const originalText = btn.innerHTML;
    btn.innerHTML = `<i class='bx bx-check'></i> Copied!`;
    setTimeout(() => btn.innerHTML = originalText, 1500);
  });
}

/**
 * Debouncer and Character Commit Logic
 */
function handleSignDetection(detectedSign, confidence, handedness) {
  const charDisplay = document.getElementById('currentPrediction');
  const confidenceVal = document.getElementById('confidenceVal');
  const handLabel = document.getElementById('handLabel');
  const stabilityBadge = document.getElementById('stabilityBadge');
  const progressFill = document.getElementById('holdProgressFill');

  confidenceVal.innerText = `${Math.round(confidence * 100)}%`;
  handLabel.innerText = handedness || "None";

  // Handle post-commit cooldown
  if (cooldownCounter > 0) {
    cooldownCounter--;
    progressFill.style.width = "0%";
    stabilityBadge.innerText = "Cooling down...";
    stabilityBadge.className = "detection-state-tag";
    return;
  }

  if (!detectedSign) {
    consecutiveFrames = 0;
    candidateChar = null;
    progressFill.style.width = "0%";
    charDisplay.innerText = "--";
    stabilityBadge.innerText = handedness === "None" ? "Waiting for hand..." : "Uncertain sign";
    stabilityBadge.className = "detection-state-tag";
    highlightKey(null);
    return;
  }

  charDisplay.innerText = detectedSign;
  highlightKey(detectedSign);

  // Consecutive sign holding
  if (detectedSign === candidateChar) {
    consecutiveFrames++;
  } else {
    candidateChar = detectedSign;
    consecutiveFrames = 1;
  }

  const progressPercent = Math.min(100, Math.round((consecutiveFrames / MIN_HOLD_FRAMES) * 100));
  progressFill.style.width = `${progressPercent}%`;

  if (progressPercent >= 75) {
    stabilityBadge.innerText = "Committing...";
    stabilityBadge.className = "detection-state-tag tag-stable";
  } else if (progressPercent >= 25) {
    stabilityBadge.innerText = "Holding";
    stabilityBadge.className = "detection-state-tag";
  } else {
    stabilityBadge.innerText = "Tracking";
    stabilityBadge.className = "detection-state-tag";
  }

  // Commit on hold completion
  if (consecutiveFrames >= MIN_HOLD_FRAMES) {
    commitSign(candidateChar);
    cooldownCounter = COOLDOWN_FRAMES;
    consecutiveFrames = 0;
    candidateChar = null;
    progressFill.style.width = "0%";
  }
}

function commitSign(sign) {
  if (sign === "space") {
    sendAction('space');
  } else if (sign === "del") {
    sendAction('backspace');
  } else {
    // Normal letter
    currentWord += sign;
    updateSentenceUI();
  }
}

function sendAction(action, extra = {}) {
  if (action === 'space') {
    if (currentWord) {
      sentenceWords.push(currentWord);
      if (ttsEnabled) {
        speakSentence(currentWord);
      }
      currentWord = "";
    }
  } else if (action === 'backspace') {
    if (currentWord.length > 0) {
      currentWord = currentWord.slice(0, -1);
    } else if (sentenceWords.length > 0) {
      currentWord = sentenceWords.pop();
    }
  } else if (action === 'clear') {
    currentWord = "";
    sentenceWords = [];
  } else if (action === 'suggestion') {
    if (extra.word) {
      sentenceWords.push(extra.word);
      if (ttsEnabled) {
        speakSentence(extra.word);
      }
      currentWord = "";
    }
  }
  updateSentenceUI();
}

function selectSuggestion(word) {
  sendAction('suggestion', { word });
}

function updateSentenceUI() {
  // Word buffer
  const wordDisplay = document.getElementById('currentWordDisplay');
  wordDisplay.innerText = currentWord;

  // Sentence display
  const sentenceDisplay = document.getElementById('sentenceDisplay');
  const wordCount = document.getElementById('wordCount');
  const fullText = sentenceWords.join(" ");

  if (fullText) {
    sentenceDisplay.innerText = fullText;
    const count = sentenceWords.length;
    wordCount.innerText = `${count} word${count === 1 ? '' : 's'}`;
  } else {
    sentenceDisplay.innerHTML = '<span class="text-muted fst-italic" style="font-size: 13px;">Sentence appears here as words are completed...</span>';
    wordCount.innerText = "0 words";
  }

  // Autocomplete Suggestions
  const suggestionsContainer = document.getElementById('autocompleteList');
  const suggestions = getSuggestions(currentWord);

  if (suggestions.length > 0) {
    suggestionsContainer.innerHTML = suggestions.map(s =>
      `<span class="suggestion-chip" onclick="selectSuggestion('${s}')"><i class='bx bx-check'></i> ${s}</span>`
    ).join('');
  } else if (currentWord) {
    suggestionsContainer.innerHTML = `<span class="text-muted" style="font-size: 12px;">No suggestions for "${currentWord}"</span>`;
  } else {
    suggestionsContainer.innerHTML = `<span class="text-muted fst-italic" style="font-size: 12px;">Start spelling a word to see suggestions...</span>`;
  }
}

function highlightKey(char) {
  if (char === lastActiveChar) return;
  if (lastActiveChar) {
    const oldKey = document.querySelector(`.asl-key[data-char="${lastActiveChar}"]`);
    if (oldKey) oldKey.classList.remove('active-sign');
  }
  if (char && char !== "--") {
    const newKey = document.querySelector(`.asl-key[data-char="${char}"]`);
    if (newKey) newKey.classList.add('active-sign');
    lastActiveChar = char;
  } else {
    lastActiveChar = null;
  }
}

function inspectSign(char) {
  const info = SIGN_DESCRIPTIONS[char] || { title: `Sign ${char}`, emoji: "✋", desc: "Hold hand clearly in camera view." };
  const badge = document.getElementById('inspectorCharBadge');
  const desc = document.getElementById('inspectorDesc');
  if (badge) badge.innerText = `${info.emoji} ${info.title}`;
  if (desc) desc.innerHTML = `<strong>${info.emoji} ${info.title}:</strong> ${info.desc}`;
}

/**
 * MediaPipe Hands Frame Processing
 */
function onHandResults(results) {
  const canvas = document.getElementById('output_canvas');
  const ctx = canvas.getContext('2d');

  // Update running FPS
  frameCount++;
  const now = performance.now();
  if (frameCount >= 15) {
    const elapsed = (now - lastFpsTime) / 1000;
    currentFps = Math.round(frameCount / elapsed);
    document.getElementById('fpsChip').innerText = `${currentFps} FPS`;
    lastFpsTime = now;
    frameCount = 0;
  }

  // Draw current frame to canvas
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

  let detectedSign = null;
  let confidence = 0.0;
  let handednessLabel = "None";

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];
    handednessLabel = (results.multiHandedness && results.multiHandedness[0]?.label) || "Left";

    // Draw hand skeleton on canvas
    drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: 'rgba(37, 99, 235, 0.8)', lineWidth: 3 });
    drawLandmarks(ctx, landmarks, { color: '#10b981', lineWidth: 2, radius: 4 });

    // Run client-side inference
    const prediction = predictASL(landmarks, handednessLabel);
    if (prediction.confidence >= CONFIDENCE_THRESHOLD) {
      detectedSign = prediction.char;
      confidence = prediction.confidence;
    } else {
      confidence = prediction.confidence;
    }
  }

  ctx.restore();

  // Update Debouncer & UI
  handleSignDetection(detectedSign, confidence, handednessLabel);
}

/**
 * Camera Start / Stop Lifecycle
 */
async function toggleRecognition() {
  const btn = document.getElementById('toggleBtn');
  const badge = document.getElementById('statusBadge');
  const statusText = document.getElementById('statusText');
  const standby = document.getElementById('viewportStandby');
  const videoElement = document.getElementById('webcam');
  const canvasElement = document.getElementById('output_canvas');

  if (isRunning) {
    // Stop Camera
    isRunning = false;
    if (cameraInstance) {
      try { await cameraInstance.stop(); } catch (e) {}
      cameraInstance = null;
    }
    if (videoElement.srcObject) {
      videoElement.srcObject.getTracks().forEach(t => t.stop());
      videoElement.srcObject = null;
    }

    btn.className = 'btn-app btn-app-primary';
    btn.innerHTML = `<i class='bx bx-play'></i> <span>Start Recognition</span>`;
    badge.classList.remove('status-live');
    statusText.innerText = 'Standby';
    standby.classList.remove('hidden');

    // Reset displays
    document.getElementById('currentPrediction').innerText = "--";
    document.getElementById('confidenceVal').innerText = "0%";
    document.getElementById('handLabel').innerText = "None";
    document.getElementById('fpsChip').innerText = "0 FPS";
    document.getElementById('stabilityBadge').innerText = "Standby";
    highlightKey(null);
  } else {
    // Start Camera
    btn.disabled = true;
    btn.innerHTML = `<i class='bx bx-loader-alt bx-spin'></i> Starting Camera...`;

    try {
      // 1. Initialize MediaPipe Hands if not already done
      if (!handsInstance) {
        handsInstance = new Hands({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });
        handsInstance.setOptions({
          maxNumHands: 1,
          modelComplexity: 1,
          minDetectionConfidence: 0.6,
          minTrackingConfidence: 0.6
        });
        handsInstance.onResults(onHandResults);
      }

      // 2. Initialize Camera
      cameraInstance = new Camera(videoElement, {
        onFrame: async () => {
          if (isRunning) {
            await handsInstance.send({ image: videoElement });
          }
        },
        width: 640,
        height: 480
      });

      await cameraInstance.start();

      canvasElement.width = 640;
      canvasElement.height = 480;

      isRunning = true;
      standby.classList.add('hidden');
      badge.classList.add('status-live');
      statusText.innerText = 'Live Recognition';
      btn.className = 'btn-app btn-app-danger';
      btn.innerHTML = `<i class='bx bx-stop'></i> <span>Stop Recognition</span>`;
    } catch (err) {
      console.error("Camera access error:", err);
      alert("Could not access camera. Please make sure your webcam is connected and camera permissions are granted.");
      standby.classList.remove('hidden');
      badge.classList.remove('status-live');
      statusText.innerText = 'Standby';
      btn.className = 'btn-app btn-app-primary';
      btn.innerHTML = `<i class='bx bx-play'></i> <span>Start Recognition</span>`;
    } finally {
      btn.disabled = false;
    }
  }
}

/**
 * Modal Guide Helpers
 */
function populateModalGuide() {
  const grid = document.getElementById('modalSignGrid');
  if (!grid) return;

  let html = '';
  for (const [key, item] of Object.entries(SIGN_DESCRIPTIONS)) {
    html += `
      <div class="asl-sign-card">
        <div class="asl-badge-large">${key === 'space' ? 'SPC' : (key === 'del' ? 'DEL' : key)}</div>
        <div>
          <div class="d-flex align-items-center gap-2 mb-1">
            <strong class="text-white" style="font-size: 13.5px;">${item.emoji} ${item.title}</strong>
          </div>
          <p class="text-muted mb-0" style="font-size: 11.5px; line-height: 1.45;">${item.desc}</p>
        </div>
      </div>
    `;
  }
  grid.innerHTML = html;
}

function openGuideModal() {
  const modal = document.getElementById('guideModal');
  if (modal) modal.classList.add('show');
}

function closeGuideModal() {
  const modal = document.getElementById('guideModal');
  if (modal) modal.classList.remove('show');
}

// Global initialization
document.addEventListener('DOMContentLoaded', () => {
  populateModalGuide();
  inspectSign('A');
  updateSentenceUI();
});
