import type { CropBox } from './api/client';

const finite = (value: number, fallback = 0) => Number.isFinite(value) ? value : fallback;
const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

/** Integer multiples of 3 × 4 keep the server's crop ratio exact. */
export function maxCropScale(width: number, height: number): number {
  if (!Number.isFinite(width) || !Number.isFinite(height)) return 0;
  return Math.max(0, Math.floor(Math.min(width / 3, height / 4)));
}

export function defaultCrop(width: number, height: number): CropBox | null {
  const max = maxCropScale(width, height);
  if (!max) return null;
  const scale = Math.max(1, Math.round(max * 0.8));
  const left = Math.round((width - scale * 3) / 2);
  const top = Math.round((height - scale * 4) / 2);
  return clampCrop(width, height, { left, top, right: left + scale * 3, bottom: top + scale * 4 });
}

export function clampCrop(width: number, height: number, crop: CropBox): CropBox | null {
  const max = maxCropScale(width, height);
  if (!max) return null;
  const scale = clamp(Math.round(finite(crop.right - crop.left, 3) / 3), 1, max);
  const left = clamp(Math.round(finite(crop.left)), 0, Math.floor(width) - scale * 3);
  const top = clamp(Math.round(finite(crop.top)), 0, Math.floor(height) - scale * 4);
  return { left, top, right: left + scale * 3, bottom: top + scale * 4 };
}

export function moveCrop(width: number, height: number, crop: CropBox, left: number, top: number): CropBox | null {
  const valid = clampCrop(width, height, crop);
  if (!valid) return null;
  left = finite(left, valid.left);
  top = finite(top, valid.top);
  return clampCrop(width, height, {
    left, top, right: left + valid.right - valid.left, bottom: top + valid.bottom - valid.top,
  });
}

export function resizeCrop(width: number, height: number, crop: CropBox, cropWidth: number): CropBox | null {
  const valid = clampCrop(width, height, crop);
  if (!valid) return null;
  const scale = clamp(Math.round(finite(cropWidth, valid.right - valid.left) / 3), 1, maxCropScale(width, height));
  const left = Math.round((valid.left + valid.right - scale * 3) / 2);
  const top = Math.round((valid.top + valid.bottom - scale * 4) / 2);
  return clampCrop(width, height, { left, top, right: left + scale * 3, bottom: top + scale * 4 });
}
