# 🖐️ Palm Reader AI (100% Client-Side WebAssembly)

> **Zero Backend Server. Zero Cloud API Calls. 100% Privacy-Preserving On-Device AI.**

A modern React application that performs **real-time hand landmark tracking** and **high-resolution palm line crease recognition** (Life Line, Heart Line, Head Line, Fate Line) completely inside the mobile or desktop web browser using **MediaPipe** and **ONNX Runtime Web (WASM)**.

---

## 🌟 Key Features

* **⚡ 100% In-Browser Inference**: Runs locally inside browser memory using WebAssembly. No Python backend, no Docker containers, and no external API calls required.
* **🔒 Total Privacy**: Images captured through the camera are processed directly in device RAM and never leave the user's phone or laptop.
* **📦 Pre-Bundled Model Weights**: Includes the complete, optimized 10.7 MB ONNX palm lines recognition model (`public/weights.onnx`). No training or weight downloading necessary.
* **🎯 Live MediaPipe Hand Guidance**: Real-time 21-point 3D hand tracking provides a visual skeleton guide (neon green bones with red joint nodes) to ensure proper hand alignment before capture.
* **📐 Intelligent Knuckle-to-Wrist Auto-Crop**: Automatically detects MCP knuckles (landmarks 5, 9, 13, 17) and wrist (landmark 0) to frame and extract a tight palm bed ROI.
* **✨ Smooth Dotted Keypoint Trajectory**: Renders smooth Catmull-Rom spline curves with glow and 10 tracked keypoint bead nodes along each crease line, matching professional biometric visualization standards.
* **💻 Desktop File Upload**: Automatically detects desktop devices and provides an intuitive Drag & Drop zone for high-res palm photo uploads (JPEG, PNG, WEBP), alongside webcam support.
* **📱 Mobile-Optimized**: Defaults to live camera with real-time MediaPipe overlay on smartphones, with front/rear toggling and a gallery upload shortcut.
* **🏷️ Separated & Labeled JSON Data**: Outputs cleanly labeled sections distinguishing `mediapipe_hand_tracking` (21 named anatomical joints with 3D normalized & pixel coords) from `palm_lines_detection` (neural network crease trajectories and confidences).

---

## 🧠 Detected Palm Lines

| Line | Anatomical Name | Color Code | Description |
| :--- | :--- | :---: | :--- |
| **Life Line** | Radial longitudinal crease | `#32CD32` (Lime Green) | Curves around the base of the thumb (Thenar eminence) |
| **Heart Line** | Distal transverse crease | `#FF3C3C` (Rose Red) | Traverses upper palm beneath knuckles |
| **Head Line** | Proximal transverse crease | `#00A5FF` (Electric Azure) | Extends across middle palm below heart line |
| **Fate Line** | Median longitudinal crease | `#FFD700` (Radiant Gold) | Runs vertically up the center of the palm toward middle finger |

---

## 🚀 Quick Start

### Prerequisites
* [Node.js](https://nodejs.org/) (v18.0 or newer)
* npm (bundled with Node.js)

### 1. Clone the Repository
```bash
git clone https://github.com/pranavv999/palm-reader-client.git
cd palm-reader-client
```

### 2. Run with One Command
Execute the included automated launcher:
```bash
./start.sh
```

*Or manually run standard npm commands:*
```bash
npm install
npm run dev
```

---

## 📱 Testing on Your Mobile Phone

Mobile browsers require **HTTPS** to grant camera permissions. The dev server includes `@vitejs/plugin-basic-ssl` to provide local HTTPS automatically.

1. Connect your mobile phone to the **same Wi-Fi network** as your computer.
2. In the terminal output, locate the **Network** URL, for example:
   ```text
   ➜  Network: https://192.168.1.105:5173/
   ```
3. Open that URL in **Safari (iOS)** or **Chrome (Android)** on your phone.
4. When the browser displays a self-signed certificate warning:
   * **iOS Safari:** Tap *Show Details* → *visit this website*.
   * **Android Chrome:** Tap *Advanced* → *Proceed to 192.168.x.x (unsafe)*.
5. Tap **Allow** when prompted for camera access.
6. Align your palm until the green skeleton snaps to your hand, then tap the capture shutter button.

---

## 🏗️ Architecture & Pipeline

```text
 ┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
 │   Camera Feed   │ ───►  │  MediaPipe Hand Track  │ ───►  │  Knuckle-to-Wrist    │
 │ (HTML5 <video>) │       │  (21 3D Landmarks)     │       │  Square ROI Crop     │
 └─────────────────┘       └────────────────────────┘       └──────────┬───────────┘
                                                                       │
 ┌─────────────────┐       ┌────────────────────────┐                  │
 │ Canvas Overlay  │ ◄───  │ 10-Point Keypoint Spline│ ◄────────────────┘
 │ & JSON Metadata │       │ ONNX Model (WASM)      │  Input: 640x640 Float32 Tensor
 └─────────────────┘       └────────────────────────┘  Output: 38x8400 Line Matrix
```

1. **Video Stream Acquisition**: Captures high-res camera feed via `navigator.mediaDevices.getUserMedia`.
2. **Landmark Detection**: `@mediapipe/tasks-vision` processes camera frames at ~30 FPS on device GPU/CPU.
3. **Palm Bed Isolation**: Crops square bounding box bounded superiorly by MCP knuckles and inferiorly by the wrist.
4. **ONNX WebAssembly Inference**: `onnxruntime-web` feeds the 640×640 normalized RGB tensor into `public/weights.onnx`.
5. **Post-Processing**: Decodes class confidences and 10 $(x, y, conf)$ keypoints per crease line, generating Catmull-Rom splines and coordinate overlays onto HTML5 canvas elements.

---

## 📂 Project Structure

```text
palm-reader-client/
├── public/
│   └── weights.onnx            # Pre-bundled 10.7 MB YOLO-pose palm model
├── src/
│   ├── utils/
│   │   └── palmDetector.js     # ONNX session loader, ROI cropper, Catmull-Rom renderer
│   ├── App.jsx                 # Camera UI, MediaPipe live loop, result viewer
│   ├── App.css                 # Dark theme responsive styling
│   └── main.jsx                # React root entry
├── index.html                  # HTML5 shell
├── package.json                # Project dependencies
├── vite.config.js              # Vite HTTPS and WASM optimization config
├── start.sh                    # One-command automated launch script
└── README.md                   # Documentation
```

---

## 📄 License
MIT License. Created by [Pranav Sonawane](https://github.com/pranavv999).
