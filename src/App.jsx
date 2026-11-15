import React, { useState, useEffect, useRef } from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import {
  Camera,
  RefreshCw,
  Sparkles,
  Copy,
  Check,
  Layers,
  Code,
  Split,
  ArrowLeft,
  Cpu,
  UploadCloud,
  FileImage,
  Monitor,
  Smartphone
} from 'lucide-react';
import {
  getOnnxSession,
  cropPalmFromSource,
  detectPalmLinesClient,
  renderOverlay,
  buildResultJson
} from './utils/palmDetector';
import { isSupportedImageFile, loadAndPrepareImage } from './utils/imageLoader';

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [5, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [9, 13], [13, 14], [14, 15], [15, 16],// Ring
  [13, 17], [17, 18], [18, 19], [19, 20],// Pinky
  [0, 17]                               // Palm base
];

const isMobileDevice = () => {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent || '';
  const mobileRegex = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i;
  const isTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
  return mobileRegex.test(ua) || (isTouch && window.innerWidth <= 820);
};

export default function App() {
  const [isMobile] = useState(isMobileDevice());
  const [inputMethod, setInputMethod] = useState(isMobile ? 'camera' : 'upload'); // 'upload' | 'camera'
  const [mode, setMode] = useState('input'); // 'input' | 'processing' | 'result'
  const [handDetected, setHandDetected] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Loading Client-Side AI Models...');
  const [facingMode, setFacingMode] = useState('environment'); // Default to rear camera
  const [results, setResults] = useState(null);
  const [activeTab, setActiveTab] = useState('overlay');
  const [copied, setCopied] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const landmarkerRef = useRef(null);
  const onnxSessionRef = useRef(null);
  const animationFrameRef = useRef(null);
  const currentLandmarksRef = useRef(null);
  const currentHandednessRef = useRef('Unknown');
  const fileInputRef = useRef(null);

  // 1. Initialize Both MediaPipe and ONNX in Browser Memory (Zero Backend)
  useEffect(() => {
    let isMounted = true;

    async function initClientEngines() {
      try {
        setStatusMessage('1/2: Loading MediaPipe Hand Tracker...');
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        );

        const handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numHands: 1,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        if (!isMounted) return;
        landmarkerRef.current = handLandmarker;

        // Load 10.7 MB ONNX model into WebAssembly memory
        setStatusMessage('2/2: Loading 10.7 MB Palm Line Model...');
        const session = await getOnnxSession();
        if (!isMounted) return;
        onnxSessionRef.current = session;

        setStatusMessage('AI Models Ready!');
        if (inputMethod === 'camera') {
          startCamera(facingMode);
        }
      } catch (err) {
        console.error('Model initialization error:', err);
        setStatusMessage('Init error: ' + err.message);
      }
    }

    initClientEngines();

    return () => {
      isMounted = false;
      stopCamera();
      if (landmarkerRef.current) landmarkerRef.current.close();
    };
  }, []);

  // Switch camera when inputMethod changes
  useEffect(() => {
    if (inputMethod === 'camera') {
      if (landmarkerRef.current && onnxSessionRef.current) {
        startCamera(facingMode);
      }
    } else {
      stopCamera();
    }
  }, [inputMethod]);

  // 2. Camera Management
  const startCamera = async (facing) => {
    stopCamera();
    try {
      const constraints = {
        video: {
          facingMode: facing,
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          startTrackingLoop();
        };
      }
    } catch (err) {
      console.error('Camera access error:', err);
      setStatusMessage('Camera error: Please allow camera permissions.');
    }
  };

  const stopCamera = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  const toggleCamera = () => {
    const nextFacing = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(nextFacing);
    startCamera(nextFacing);
  };

  // 3. Live MediaPipe Tracking Loop
  const startTrackingLoop = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;

    if (!video || !canvas || !landmarker) return;

    let lastVideoTime = -1;

    const render = () => {
      if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;

        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // MediaPipe in-browser detection
        const detections = landmarker.detectForVideo(video, performance.now());

        if (detections.landmarks && detections.landmarks.length > 0) {
          const landmarks = detections.landmarks[0];
          currentLandmarksRef.current = landmarks;
          if (detections.handednesses && detections.handednesses.length > 0 && detections.handednesses[0].length > 0) {
            currentHandednessRef.current = detections.handednesses[0][0].categoryName || 'Unknown';
          }
          setHandDetected(true);
          setStatusMessage('Palm Detected! Hold steady & tap capture');

          // Green skeleton bones
          ctx.strokeStyle = '#00FF00';
          ctx.lineWidth = 6;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';

          HAND_CONNECTIONS.forEach(([i, j]) => {
            const p1 = landmarks[i];
            const p2 = landmarks[j];
            ctx.beginPath();
            ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
            ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
            ctx.stroke();
          });

          // Red joint nodes
          landmarks.forEach((pt) => {
            ctx.fillStyle = '#FF0000';
            ctx.beginPath();
            ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 9, 0, 2 * Math.PI);
            ctx.fill();
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 1.5;
            ctx.stroke();
          });
        } else {
          currentLandmarksRef.current = null;
          setHandDetected(false);
          setStatusMessage('Position palm inside frame');
        }
      }

      animationFrameRef.current = requestAnimationFrame(render);
    };

    render();
  };

  // 4. Capture from Live Camera
  const handleCapture = async () => {
    const video = videoRef.current;
    const session = onnxSessionRef.current;
    if (!video || !session) return;

    setMode('processing');
    setStatusMessage('Analyzing palm creases 100% on device...');
    await new Promise((r) => setTimeout(r, 60));

    try {
      const landmarks = currentLandmarksRef.current;
      const handedness = currentHandednessRef.current;
      const { fullCanvas, cropCanvas, modelCanvas, cropBox } = cropPalmFromSource(video, landmarks);
      stopCamera();

      // Run ONNX Keypoint Model directly in WebAssembly
      const { predictions, inferenceMs } = await detectPalmLinesClient(
        session,
        modelCanvas,
        cropBox,
        fullCanvas.width,
        fullCanvas.height,
        0.20
      );

      // Render overlays
      const overlayCanvas = document.createElement('canvas');
      renderOverlay(overlayCanvas, fullCanvas, predictions, cropBox);

      const croppedCanvas = document.createElement('canvas');
      renderOverlay(croppedCanvas, cropCanvas, predictions, null);

      // Format complete, cleanly labeled JSON distinguishing MediaPipe from Line Detection
      const dataJson = buildResultJson({
        scanSource: 'camera_scan',
        deviceType: isMobile ? 'mobile' : 'desktop',
        imageWidth: fullCanvas.width,
        imageHeight: fullCanvas.height,
        inferenceMs,
        landmarks,
        handedness,
        cropBox,
        predictions
      });

      setResults({
        overlayImage: overlayCanvas.toDataURL('image/jpeg', 0.88),
        croppedImage: croppedCanvas.toDataURL('image/jpeg', 0.88),
        rawImage: fullCanvas.toDataURL('image/jpeg', 0.85),
        predictions,
        dataJson,
        cropBox,
        inferenceMs,
      });

      setMode('result');
    } catch (err) {
      console.error('Client-side processing error:', err);
      alert('Client execution error: ' + err.message);
      setMode('input');
      startCamera(facingMode);
    }
  };

  // 5. File Upload Handler (Desktop & Mobile)
  const processImageFile = async (file) => {
    if (!file) return;
    if (!isSupportedImageFile(file)) {
      alert('Please upload a valid image file (JPEG, PNG, WEBP, or iPhone HEIC).');
      return;
    }

    const session = onnxSessionRef.current;
    const landmarker = landmarkerRef.current;
    if (!session || !landmarker) {
      alert('AI models are still initializing. Please wait a few seconds and try again.');
      return;
    }

    setMode('processing');

    try {
      // 1. Prepare image: handles HEIC conversion, aspect-preserving downscaling
      const imgCanvas = await loadAndPrepareImage(file, 1600, (status) => setStatusMessage(status));

      setStatusMessage('Tracking palm bed with Google MediaPipe...');

      // 2. Safe MediaPipe detection on prepared canvas
      let landmarks = null;
      let handedness = 'Unknown';
      try {
        const detections = landmarker.detectForVideo(imgCanvas, performance.now());
        if (detections && detections.landmarks && detections.landmarks.length > 0) {
          landmarks = detections.landmarks[0];
          if (detections.handednesses && detections.handednesses.length > 0 && detections.handednesses[0].length > 0) {
            handedness = detections.handednesses[0][0].categoryName || 'Unknown';
          }
        }
      } catch (mpErr) {
        console.warn('MediaPipe tracking notice:', mpErr);
      }

      setStatusMessage('Detecting palm creases with ONNX WebAssembly...');

      // 3. Crop palm bed (anchored to knuckles and wrist, or center fallback)
      const { fullCanvas, cropCanvas, modelCanvas, cropBox } = cropPalmFromSource(imgCanvas, landmarks);

      // 4. Run ONNX Keypoint Model in WebAssembly
      const { predictions, inferenceMs } = await detectPalmLinesClient(
        session,
        modelCanvas,
        cropBox,
        fullCanvas.width,
        fullCanvas.height,
        0.20
      );

      // 5. Render overlays
      const overlayCanvas = document.createElement('canvas');
      renderOverlay(overlayCanvas, fullCanvas, predictions, cropBox);

      const croppedCanvas = document.createElement('canvas');
      renderOverlay(croppedCanvas, cropCanvas, predictions, null);

      // 6. Clean, labeled JSON distinguishing MediaPipe from Line Detection
      const dataJson = buildResultJson({
        scanSource: 'file_upload',
        deviceType: isMobile ? 'mobile' : 'desktop',
        imageWidth: fullCanvas.width,
        imageHeight: fullCanvas.height,
        inferenceMs,
        landmarks,
        handedness,
        cropBox,
        predictions
      });

      setResults({
        overlayImage: overlayCanvas.toDataURL('image/jpeg', 0.88),
        croppedImage: croppedCanvas.toDataURL('image/jpeg', 0.88),
        rawImage: fullCanvas.toDataURL('image/jpeg', 0.85),
        predictions,
        dataJson,
        cropBox,
        inferenceMs,
      });

      setMode('result');
    } catch (err) {
      console.error('File processing error:', err);
      alert('Error processing image: ' + err.message);
      setMode('input');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processImageFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processImageFile(e.target.files[0]);
    }
  };

  const handleRetake = () => {
    setResults(null);
    setMode('input');
    if (inputMethod === 'camera') {
      startCamera(facingMode);
    }
  };

  const copyJson = () => {
    if (!results || !results.dataJson) return;
    navigator.clipboard.writeText(JSON.stringify(results.dataJson, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="app-container">
      {/* ── App Header ─────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="logo-group">
          <div className="logo-badge">🖐️</div>
          <div>
            <h1 className="app-title">Palm Reader AI</h1>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className="model-tag" style={{ background: 'rgba(59, 130, 246, 0.15)', color: '#60a5fa', borderColor: 'rgba(59, 130, 246, 0.3)' }}>
            <Cpu size={12} style={{ display: 'inline', marginRight: 4 }} />
            100% Client-Side
          </span>
        </div>
      </header>

      {/* ── View 1: Input Screen (Camera OR File Upload) ───────────── */}
      {mode === 'input' && (
        <>
          {/* Method Selector Tabs */}
          <div className="input-toggle-bar">
            <button
              className={`toggle-tab ${inputMethod === 'upload' ? 'active' : ''}`}
              onClick={() => setInputMethod('upload')}
            >
              <FileImage size={15} style={{ marginRight: 6 }} />
              Upload Image {(!isMobile) && '(Desktop)'}
            </button>
            <button
              className={`toggle-tab ${inputMethod === 'camera' ? 'active' : ''}`}
              onClick={() => setInputMethod('camera')}
            >
              <Camera size={15} style={{ marginRight: 6 }} />
              Live Camera {(isMobile) && '(Mobile)'}
            </button>
          </div>

          {/* Mode A: Desktop & Mobile File Upload Dropzone */}
          {inputMethod === 'upload' && (
            <div className="upload-stage">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,.heic,.heif,.jpg,.jpeg,.png,.webp"
                style={{ display: 'none' }}
                onChange={handleFileInputChange}
              />
              <div
                className={`dropzone ${isDragging ? 'dragover' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current && fileInputRef.current.click()}
              >
                <div className="upload-icon-circle">
                  <UploadCloud size={36} color="#3b82f6" />
                </div>
                <h3 className="upload-title">Drop your palm photo here</h3>
                <p className="upload-subtitle">or click to browse from your device</p>
                <div className="upload-badges">
                  <span className="upload-badge">HEIC (iPhone)</span>
                  <span className="upload-badge">JPEG</span>
                  <span className="upload-badge">PNG</span>
                  <span className="upload-badge">WEBP</span>
                  <span className="upload-badge">Up to 25 MB</span>
                </div>
              </div>

              <div className="upload-instructions">
                <p className="instruction-heading">💡 Guidelines for best detection:</p>
                <ul className="instruction-list">
                  <li>Hold palm flat and spread fingers naturally.</li>
                  <li>Ensure bright, even lighting across the palm bed.</li>
                  <li>Capture from wrist crease to knuckles clearly.</li>
                </ul>
              </div>
            </div>
          )}

          {/* Mode B: Live Camera Stage with MediaPipe */}
          {inputMethod === 'camera' && (
            <>
              <div className="camera-stage">
                <video ref={videoRef} playsInline autoPlay muted className="video-feed" />
                <canvas ref={canvasRef} className="canvas-overlay" />

                {/* Target Alignment Box */}
                <div className={`palm-target-guide ${handDetected ? 'active' : ''}`} />

                {/* Live Status Pill */}
                <div className="status-pill">
                  <span className={`status-dot ${handDetected ? 'active' : ''}`} />
                  <span>{statusMessage}</span>
                </div>
              </div>

              {/* Bottom Camera Controls */}
              <div className="camera-controls">
                <button
                  className="icon-btn"
                  onClick={() => fileInputRef.current && fileInputRef.current.click()}
                  title="Upload image from device"
                >
                  <FileImage size={22} />
                </button>

                <button
                  className="shutter-btn"
                  onClick={handleCapture}
                  disabled={!onnxSessionRef.current}
                  title="Capture Palm"
                >
                  <div className="shutter-inner" />
                </button>

                <button className="icon-btn" onClick={toggleCamera} title="Switch Camera">
                  <RefreshCw size={22} />
                </button>
              </div>
            </>
          )}
        </>
      )}

      {/* ── View 2: Processing Spinner ───────────────────────────── */}
      {mode === 'processing' && (
        <div className="result-container" style={{ alignItems: 'center', justifyContent: 'center' }}>
          <Sparkles className="animate-spin" size={48} color="#3b82f6" />
          <h3 style={{ marginTop: 20, fontWeight: 700 }}>Processing 100% On-Device</h3>
          <p style={{ color: '#94a3b8', fontSize: 13, marginTop: 6, textAlign: 'center' }}>
            MediaPipe Hand Tracking + ONNX WebAssembly running in your browser...
          </p>
        </div>
      )}

      {/* ── View 3: Complete Results Viewer ───────────────────────── */}
      {mode === 'result' && results && (
        <div className="result-container">
          {/* Metadata & Tracking Banner */}
          <div style={{ background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.25)', borderRadius: 12, padding: '10px 14px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 12 }}>
            <span style={{ color: '#93c5fd', fontWeight: 600 }}>
              {results.dataJson?.mediapipe_hand_tracking?.hand_detected
                ? `🖐️ MediaPipe: ${results.dataJson.mediapipe_hand_tracking.handedness} Hand Tracked (21 Joints)`
                : '⚠️ Fallback Center Crop'}
            </span>
            <span style={{ fontFamily: 'monospace', color: '#60a5fa', fontWeight: 700 }}>
              {results.inferenceMs} ms
            </span>
          </div>

          {/* Navigation Tabs */}
          <div className="tab-nav">
            <button
              className={`tab-btn ${activeTab === 'overlay' ? 'active' : ''}`}
              onClick={() => setActiveTab('overlay')}
            >
              <Layers size={14} style={{ display: 'inline', marginRight: 4 }} />
              Lines Overlay
            </button>
            <button
              className={`tab-btn ${activeTab === 'compare' ? 'active' : ''}`}
              onClick={() => setActiveTab('compare')}
            >
              <Split size={14} style={{ display: 'inline', marginRight: 4 }} />
              Side-by-Side
            </button>
            <button
              className={`tab-btn ${activeTab === 'json' ? 'active' : ''}`}
              onClick={() => setActiveTab('json')}
            >
              <Code size={14} style={{ display: 'inline', marginRight: 4 }} />
              JSON Data
            </button>
          </div>

          {/* Tab 1: Full Hand Overlay */}
          {activeTab === 'overlay' && (
            <div className="display-card">
              <img src={results.overlayImage} alt="Palm Lines Overlay" className="display-image" />
            </div>
          )}

          {/* Tab 2: Side-by-Side Comparison */}
          {activeTab === 'compare' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div className="display-card">
                <p style={{ padding: '8px 12px', fontSize: 12, fontWeight: 700, color: '#94a3b8' }}>
                  1. Auto-Cropped Palm Bed (Input to Model)
                </p>
                <img src={results.croppedImage} alt="Cropped Palm" className="display-image" />
              </div>
              <div className="display-card">
                <p style={{ padding: '8px 12px', fontSize: 12, fontWeight: 700, color: '#94a3b8' }}>
                  2. Full Hand with Detected Creases
                </p>
                <img src={results.overlayImage} alt="Overlay" className="display-image" />
              </div>
            </div>
          )}

          {/* Tab 3: JSON Raw Data (Separated MediaPipe & Line Detection) */}
          {activeTab === 'json' && (
            <div style={{ position: 'relative' }}>
              <button
                className="icon-btn"
                onClick={copyJson}
                style={{ position: 'absolute', top: 10, right: 10, zIndex: 5, width: 36, height: 36 }}
                title="Copy JSON"
              >
                {copied ? <Check size={16} color="#10b981" /> : <Copy size={16} />}
              </button>
              <pre className="json-box">{JSON.stringify(results.dataJson, null, 2)}</pre>
            </div>
          )}

          {/* Lines Summary Cards */}
          <div className="lines-summary-grid">
            {results.predictions.map((p, idx) => (
              <div key={idx} className={`line-card ${p.class}`}>
                <div className="line-title">
                  <span>{p.name}</span>
                  <span className="line-conf" style={{ color: p.color }}>
                    {(p.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <span className="line-desc">{p.desc}</span>
                <span style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>
                  {p.keypoints.length} keypoints tracked
                </span>
              </div>
            ))}
          </div>

          {/* Retake Button */}
          <div className="result-actions">
            <button className="primary-btn" onClick={handleRetake}>
              <ArrowLeft size={18} />
              Analyze Another Palm
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
