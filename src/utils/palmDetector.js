import * as ort from 'onnxruntime-web';

// Configure ONNX Runtime to load matching 1.31.1 WASM assets directly from CDN
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
ort.env.wasm.numThreads = 1; // Single-thread mode avoids Safari SharedArrayBuffer COOP/COEP restrictions

const CLASS_CONFIGS = [
  { id: 0, class: 'fate',  name: 'Fate Line',  color: '#FFD700', glowColor: 'rgba(255, 235, 80, 0.45)', desc: 'Median longitudinal crease' },
  { id: 1, class: 'head',  name: 'Head Line',  color: '#00A5FF', glowColor: 'rgba(80, 200, 255, 0.45)', desc: 'Proximal transverse crease' },
  { id: 2, class: 'heart', name: 'Heart Line', color: '#FF3C3C', glowColor: 'rgba(255, 120, 120, 0.45)', desc: 'Distal transverse crease' },
  { id: 3, class: 'life',  name: 'Life Line',  color: '#32CD32', glowColor: 'rgba(100, 255, 100, 0.45)', desc: 'Radial longitudinal crease' },
];

let sessionInstance = null;

/**
 * Initializes the ONNX model session directly from public/weights.onnx
 */
export async function getOnnxSession() {
  if (sessionInstance) return sessionInstance;

  console.log('⏳ Loading ONNX model into browser memory (10.7 MB)...');
  try {
    sessionInstance = await ort.InferenceSession.create('/weights.onnx', {
      executionProviders: ['wasm'],
    });
    console.log('✅ ONNX model session ready in browser!');
    return sessionInstance;
  } catch (err) {
    console.error('Failed to create ONNX session:', err);
    throw err;
  }
}

// 21 Landmark Joint Names for Google MediaPipe Hand Landmarker
export const MEDIAPIPE_LANDMARK_NAMES = [
  'WRIST',
  'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
  'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
  'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
  'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
  'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
];

/**
 * Builds structured, cleanly labeled JSON distinguishing MediaPipe tracking from Line Detection.
 */
export function buildResultJson({
  scanSource = 'camera_scan',
  deviceType = 'mobile',
  imageWidth,
  imageHeight,
  inferenceMs = 0,
  landmarks = null,
  handedness = 'Unknown',
  cropBox = null,
  predictions = []
}) {
  const formattedLandmarks = (landmarks || []).map((pt, idx) => ({
    id: idx,
    joint_name: MEDIAPIPE_LANDMARK_NAMES[idx] || `LANDMARK_${idx}`,
    normalized_coordinates: {
      x: Math.round(pt.x * 10000) / 10000,
      y: Math.round(pt.y * 10000) / 10000,
      z: Math.round((pt.z || 0) * 10000) / 10000
    },
    pixel_coordinates: {
      x: Math.round(pt.x * imageWidth),
      y: Math.round(pt.y * imageHeight)
    }
  }));

  return {
    scan_metadata: {
      scan_source: scanSource, // 'file_upload' | 'camera_scan'
      device_type: deviceType, // 'desktop' | 'mobile'
      timestamp: new Date().toISOString(),
      image_resolution: {
        width: imageWidth,
        height: imageHeight
      },
      inference_time_ms: Math.round(inferenceMs)
    },
    mediapipe_hand_tracking: {
      hand_detected: !!(landmarks && landmarks.length > 0),
      handedness: handedness || 'Unknown',
      landmarks_count: formattedLandmarks.length,
      landmarks: formattedLandmarks,
      palm_bed_roi: cropBox ? {
        x1: cropBox.x1,
        y1: cropBox.y1,
        x2: cropBox.x2,
        y2: cropBox.y2,
        width: cropBox.width,
        height: cropBox.height
      } : null
    },
    palm_lines_detection: (predictions || []).map((p) => ({
      line_class: p.class,
      line_name: p.name,
      anatomical_name: p.desc,
      confidence_score: p.confidence,
      confidence_percentage: `${Math.round(p.confidence * 100)}%`,
      keypoints_count: (p.keypoints || []).length,
      color_hex: p.color,
      keypoints: p.keypoints || [],
      crop_keypoints: p.cropKeypoints || p.crop_keypoints || []
    }))
  };
}

/**
 * Crops the palm bed from any visual source (video, image, or canvas) based on MediaPipe landmarks.
 */
export function cropPalmFromSource(source, landmarks) {
  const vW = source.videoWidth || source.naturalWidth || source.width;
  const vH = source.videoHeight || source.naturalHeight || source.height;

  let x1, y1, x2, y2;

  if (landmarks && landmarks.length >= 18) {
    // MediaPipe landmarks: wrist 0, thumb 1-2, index MCP 5, middle MCP 9, ring MCP 13, pinky MCP 17
    const palmIndices = [0, 1, 2, 5, 9, 13, 17];
    const xs = palmIndices.map((i) => landmarks[i].x * vW);
    const ys = palmIndices.map((i) => landmarks[i].y * vH);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const bw = maxX - minX;
    const bh = maxY - minY;

    // Tight padding: keep top at knuckle line, bottom at wrist crease
    const padX = bw * 0.12;
    const padYTop = bh * 0.05;    // Keep knuckles right at the top
    const padYBottom = bh * 0.08; // Keep wrist at the bottom

    x1 = Math.max(0, Math.floor(minX - padX));
    y1 = Math.max(0, Math.floor(minY - padYTop));
    x2 = Math.min(vW, Math.ceil(maxX + padX));
    y2 = Math.min(vH, Math.ceil(maxY + padYBottom));

    // Make square centered on the palm bed
    const bwActual = x2 - x1;
    const bhActual = y2 - y1;
    const side = Math.max(bwActual, bhActual);
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;

    x1 = Math.max(0, Math.floor(cx - side / 2));
    y1 = Math.max(0, Math.floor(cy - side / 2));
    x2 = Math.min(vW, x1 + side);
    y2 = Math.min(vH, y1 + side);
  } else {
    // Center square fallback
    const side = Math.floor(Math.min(vW, vH) * 0.55);
    const cx = vW / 2;
    const cy = vH * 0.52;
    x1 = Math.max(0, Math.floor(cx - side / 2));
    y1 = Math.max(0, Math.floor(cy - side / 2));
    x2 = x1 + side;
    y2 = y1 + side;
  }

  const cropW = x2 - x1;
  const cropH = y2 - y1;

  // 1. Full-res snapshot canvas
  const fullCanvas = document.createElement('canvas');
  fullCanvas.width = vW;
  fullCanvas.height = vH;
  const fullCtx = fullCanvas.getContext('2d');
  fullCtx.drawImage(source, 0, 0, vW, vH);

  // 2. High-res cropped palm canvas (matches cropBox width & height 1:1)
  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = cropW;
  cropCanvas.height = cropH;
  const cropCtx = cropCanvas.getContext('2d');
  cropCtx.drawImage(source, x1, y1, cropW, cropH, 0, 0, cropW, cropH);

  // 3. 640x640 model input canvas (used for feeding ONNX tensor)
  const modelCanvas = document.createElement('canvas');
  modelCanvas.width = 640;
  modelCanvas.height = 640;
  const modelCtx = modelCanvas.getContext('2d');
  modelCtx.drawImage(cropCanvas, 0, 0, cropW, cropH, 0, 0, 640, 640);

  return {
    fullCanvas,
    cropCanvas,
    modelCanvas,
    cropBox: { x1, y1, x2, y2, width: cropW, height: cropH },
  };
}

export const cropPalmFromVideo = cropPalmFromSource;

/**
 * Runs client-side ONNX inference on the 640x640 canvas and decodes keypoints.
 */
export async function detectPalmLinesClient(session, modelCanvas, cropBox, fullWidth, fullHeight, minConf = 0.22) {
  const ctx = modelCanvas.getContext('2d');
  const imgData = ctx.getImageData(0, 0, 640, 640);
  const { data } = imgData; // RGBA uint8 array (length 640*640*4)

  // Construct planar Float32Array [1, 3, 640, 640] normalized to 0.0 - 1.0 (RGB)
  const floatData = new Float32Array(3 * 640 * 640);
  const planeSize = 640 * 640;

  for (let i = 0; i < planeSize; i++) {
    const r = data[i * 4] / 255.0;
    const g = data[i * 4 + 1] / 255.0;
    const b = data[i * 4 + 2] / 255.0;

    floatData[i] = r;                  // Red plane
    floatData[planeSize + i] = g;      // Green plane
    floatData[planeSize * 2 + i] = b;  // Blue plane
  }

  const inputTensor = new ort.Tensor('float32', floatData, [1, 3, 640, 640]);

  // Run the ONNX model on client CPU/WebGPU
  const startTime = performance.now();
  const feeds = { images: inputTensor };
  const results = await session.run(feeds);
  const inferenceMs = performance.now() - startTime;
  console.log(`⚡ Client-side ONNX inference completed in ${inferenceMs.toFixed(1)} ms`);

  // Output tensor output0 shape is [1, 38, 8400]
  const outData = results.output0.data;
  const numRows = 38;
  const numAnchors = 8400;

  const predictions = [];
  const scaleX = cropBox.width / 640.0;
  const scaleY = cropBox.height / 640.0;

  // For each of the 4 lines, find the anchor with the highest score
  CLASS_CONFIGS.forEach((cfg) => {
    const classRow = 4 + cfg.id; // Rows 4, 5, 6, 7 represent class scores
    let bestScore = -1;
    let bestAnchor = -1;

    for (let a = 0; a < numAnchors; a++) {
      const score = outData[classRow * numAnchors + a];
      if (score > bestScore) {
        bestScore = score;
        bestAnchor = a;
      }
    }

    if (bestScore >= minConf && bestAnchor >= 0) {
      // Extract the 10 keypoints (Rows 8 to 37)
      const cropKeypoints = [];
      const fullKeypoints = [];

      for (let k = 0; k < 10; k++) {
        const kx = outData[(8 + k * 3) * numAnchors + bestAnchor];
        const ky = outData[(8 + k * 3 + 1) * numAnchors + bestAnchor];
        const kc = outData[(8 + k * 3 + 2) * numAnchors + bestAnchor];

        // Map to cropped palm
        const cx = kx * scaleX;
        const cy = ky * scaleY;
        cropKeypoints.push({ x: Math.round(cx * 10) / 10, y: Math.round(cy * 10) / 10, confidence: kc });

        // Map to full high-res camera frame
        const fx = cx + cropBox.x1;
        const fy = cy + cropBox.y1;
        fullKeypoints.push({ x: Math.round(fx * 10) / 10, y: Math.round(fy * 10) / 10, confidence: kc });
      }

      predictions.push({
        class: cfg.class,
        name: cfg.name,
        color: cfg.color,
        desc: cfg.desc,
        confidence: Math.round(bestScore * 1000) / 1000,
        keypoints: fullKeypoints,
        cropKeypoints: cropKeypoints,
      });
    }
  });

  return {
    predictions,
    inferenceMs: Math.round(inferenceMs),
  };
}

/**
 * Generates Catmull-Rom spline points for smooth organic curve drawing.
 */
export function getCatmullRomSpline(points, numSegments = 16) {
  if (points.length < 2) return points;
  if (points.length === 2) return points;

  const result = [];
  const n = points.length;

  for (let i = 0; i < n - 1; i++) {
    const p0 = i > 0 ? points[i - 1] : points[i];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = i < n - 2 ? points[i + 2] : p2;

    for (let t = 0; t <= numSegments; t++) {
      const u = t / numSegments;
      const u2 = u * u;
      const u3 = u2 * u;

      const f0 = -0.5 * u3 + u2 - 0.5 * u;
      const f1 = 1.5 * u3 - 2.5 * u2 + 1.0;
      const f2 = -1.5 * u3 + 2.0 * u2 + 0.5 * u;
      const f3 = 0.5 * u3 - 0.5 * u2;

      const x = p0.x * f0 + p1.x * f1 + p2.x * f2 + p3.x * f3;
      const y = p0.y * f0 + p1.y * f1 + p2.y * f2 + p3.y * f3;
      result.push({ x, y });
    }
  }

  return result;
}

/**
 * Draws the HUD summary legend in the top-right corner (matching backend Python infer_local.py).
 */
function drawHudLegend(ctx, width, height, predictions, scale) {
  const pad = Math.round(16 * scale);
  const rowHeight = Math.round(26 * scale);
  const boxW = Math.round(260 * scale);
  const boxH = pad * 2 + Math.round(30 * scale) + predictions.length * rowHeight;

  const x2 = width - Math.round(20 * scale);
  const x1 = x2 - boxW;
  const y1 = Math.round(20 * scale);

  ctx.save();
  // Semi-transparent dark background
  ctx.fillStyle = 'rgba(18, 18, 22, 0.84)';
  ctx.fillRect(x1, y1, boxW, boxH);

  // Border
  ctx.strokeStyle = 'rgba(70, 70, 80, 0.85)';
  ctx.lineWidth = Math.max(1, Math.round(1.5 * scale));
  ctx.strokeRect(x1, y1, boxW, boxH);

  // Title
  ctx.fillStyle = '#FFFFFF';
  ctx.font = `bold ${Math.round(13 * scale)}px sans-serif`;
  ctx.fillText('PALM LINES DETECTED', x1 + pad, y1 + pad + Math.round(13 * scale));

  // Divider line
  const divY = y1 + pad + Math.round(22 * scale);
  ctx.strokeStyle = 'rgba(80, 80, 90, 0.8)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x1 + pad, divY);
  ctx.lineTo(x2 - pad, divY);
  ctx.stroke();

  // Rows
  let currY = divY + Math.round(21 * scale);
  const dotR = Math.max(3, Math.round(4.5 * scale));

  predictions.forEach((pred) => {
    const dotCenterY = currY - Math.round(4 * scale);
    // Node dot
    ctx.beginPath();
    ctx.arc(x1 + pad + dotR, dotCenterY, dotR + Math.round(1.5 * scale), 0, 2 * Math.PI);
    ctx.fillStyle = '#141419';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(x1 + pad + dotR, dotCenterY, dotR, 0, 2 * Math.PI);
    ctx.fillStyle = pred.color;
    ctx.fill();

    // Name
    ctx.fillStyle = '#E6E6E6';
    ctx.font = `600 ${Math.round(12 * scale)}px sans-serif`;
    ctx.fillText(pred.name, x1 + pad + dotR * 2 + Math.round(10 * scale), currY);

    // Confidence
    const confStr = `${(pred.confidence * 100).toFixed(1)}%`;
    ctx.font = `500 ${Math.round(12 * scale)}px sans-serif`;
    ctx.fillStyle = '#A0A0B0';
    const tw = ctx.measureText(confStr).width;
    ctx.fillText(confStr, x2 - pad - tw, currY);

    currY += rowHeight;
  });

  ctx.restore();
}

/**
 * Renders the visual overlay on a canvas matching the backend Python infer_local.py:
 * - Glowing curves + solid core in line's color
 * - 10 Keypoint node dots along each line (dotted appearance)
 * - NO floating tooltips/badges over the palm
 * - Top-right HUD summary legend on full overlay
 */
export function renderOverlay(canvas, sourceImageOrCanvas, predictions, cropBox = null) {
  const ctx = canvas.getContext('2d');
  canvas.width = sourceImageOrCanvas.width;
  canvas.height = sourceImageOrCanvas.height;

  // Draw base image
  ctx.drawImage(sourceImageOrCanvas, 0, 0);

  const scale = Math.max(0.65, Math.max(canvas.width, canvas.height) / 1600.0);
  const coreThickness = Math.max(2, Math.round(4 * scale));
  const glowThickness = Math.max(5, Math.round(11 * scale));
  const kpRadius = Math.max(3, Math.round(5 * scale));
  const kpBorder = Math.max(1, Math.round(1.5 * scale));

  // If cropBox provided, draw dashed ROI boundary on full hand
  if (cropBox) {
    ctx.save();
    ctx.strokeStyle = '#10B981';
    ctx.lineWidth = Math.max(2, Math.round(2.5 * scale));
    ctx.setLineDash([8, 8]);
    ctx.strokeRect(cropBox.x1, cropBox.y1, cropBox.width, cropBox.height);
    ctx.restore();
  }

  // 1. First Pass: Soft Glowing Curves
  predictions.forEach((pred) => {
    const kps = cropBox ? pred.keypoints : pred.cropKeypoints;
    if (!kps || kps.length < 2) return;

    const smoothPts = getCatmullRomSpline(kps, 16);

    ctx.save();
    ctx.strokeStyle = pred.color;
    ctx.shadowColor = pred.color;
    ctx.shadowBlur = Math.round(12 * scale);
    ctx.lineWidth = glowThickness;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(smoothPts[0].x, smoothPts[0].y);
    smoothPts.forEach((pt) => ctx.lineTo(pt.x, pt.y));
    ctx.stroke();
    ctx.restore();
  });

  // 2. Second Pass: Solid Crisp Core Lines & Keypoint Node Dots (Dotted effect)
  predictions.forEach((pred) => {
    const kps = cropBox ? pred.keypoints : pred.cropKeypoints;
    if (!kps || kps.length < 2) return;

    const smoothPts = getCatmullRomSpline(kps, 16);

    // Solid core line
    ctx.save();
    ctx.strokeStyle = pred.color;
    ctx.lineWidth = coreThickness;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(smoothPts[0].x, smoothPts[0].y);
    smoothPts.forEach((pt) => ctx.lineTo(pt.x, pt.y));
    ctx.stroke();
    ctx.restore();

    // Keypoint node dots along the entire crease line (matches backend assisted)
    kps.forEach((pt) => {
      // Dark outer ring
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, kpRadius + kpBorder, 0, 2 * Math.PI);
      ctx.fillStyle = '#141419';
      ctx.fill();

      // Colored inner dot
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, kpRadius, 0, 2 * Math.PI);
      ctx.fillStyle = pred.color;
      ctx.fill();
    });
  });

  // 3. Top-right HUD summary legend on full overlay (no tooltips over palm)
  if (cropBox && predictions.length > 0 && canvas.width >= 360) {
    drawHudLegend(ctx, canvas.width, canvas.height, predictions, scale);
  }
}
