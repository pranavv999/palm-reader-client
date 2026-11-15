import heic2any from 'heic2any';

/**
 * Validates whether a file is a supported image format (JPEG, PNG, WEBP, HEIC/HEIF).
 */
export function isSupportedImageFile(file) {
  if (!file) return false;
  if (file.type && file.type.startsWith('image/')) return true;
  const name = (file.name || '').toLowerCase();
  return (
    name.endsWith('.heic') ||
    name.endsWith('.heif') ||
    name.endsWith('.jpg') ||
    name.endsWith('.jpeg') ||
    name.endsWith('.png') ||
    name.endsWith('.webp')
  );
}

/**
 * Checks if a file is an Apple HEIC/HEIF format.
 */
export function isHeicFile(file) {
  if (!file) return false;
  const type = (file.type || '').toLowerCase();
  const name = (file.name || '').toLowerCase();
  return (
    type === 'image/heic' ||
    type === 'image/heif' ||
    name.endsWith('.heic') ||
    name.endsWith('.heif')
  );
}

/**
 * Loads any image file (including iPhone HEIC), converts HEIC to standard JPEG,
 * and scales down oversized images to maxDimension (preserving aspect ratio).
 *
 * @param {File|Blob} file
 * @param {number} maxDimension (default 1600)
 * @param {(status: string) => void} onStatusUpdate
 * @returns {Promise<HTMLCanvasElement>}
 */
export async function loadAndPrepareImage(file, maxDimension = 1600, onStatusUpdate = () => {}) {
  let blobToLoad = file;

  // Convert HEIC/HEIF from iPhones to JPEG
  if (isHeicFile(file)) {
    onStatusUpdate('Converting iPhone HEIC photo to JPEG...');
    try {
      const converted = await heic2any({
        blob: file,
        toType: 'image/jpeg',
        quality: 0.90,
      });
      blobToLoad = Array.isArray(converted) ? converted[0] : converted;
    } catch (heicErr) {
      console.error('HEIC conversion failed:', heicErr);
      throw new Error(
        'Unable to convert iPhone HEIC photo. Please take or export the photo as JPEG/PNG, or try another image.'
      );
    }
  }

  onStatusUpdate('Decoding image pixels...');

  const img = await new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(blobToLoad);
    const image = new Image();
    let settled = false;

    // 15-second safety timeout
    const timeout = setTimeout(() => {
      if (!settled) {
        settled = true;
        URL.revokeObjectURL(objectUrl);
        reject(new Error('Image decoding timed out. Please try a different image.'));
      }
    }, 15000);

    image.onload = () => {
      if (!settled) {
        settled = true;
        clearTimeout(timeout);
        URL.revokeObjectURL(objectUrl);
        resolve(image);
      }
    };

    image.onerror = (err) => {
      if (!settled) {
        settled = true;
        clearTimeout(timeout);
        URL.revokeObjectURL(objectUrl);
        reject(
          new Error('Failed to decode image file. Ensure it is a valid JPEG, PNG, WEBP, or HEIC image.')
        );
      }
    };

    image.src = objectUrl;
  });

  const origW = img.naturalWidth || img.width;
  const origH = img.naturalHeight || img.height;

  if (!origW || !origH) {
    throw new Error('Image has zero dimensions or could not be rendered.');
  }

  // Calculate aspect-preserving dimensions capped at maxDimension
  let targetW = origW;
  let targetH = origH;
  if (Math.max(origW, origH) > maxDimension) {
    const scale = maxDimension / Math.max(origW, origH);
    targetW = Math.round(origW * scale);
    targetH = Math.round(origH * scale);
  }

  const canvas = document.createElement('canvas');
  canvas.width = targetW;
  canvas.height = targetH;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, targetW, targetH);

  return canvas;
}
