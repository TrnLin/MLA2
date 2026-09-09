import { useEffect, useId, useRef, useState } from 'react';
import type { KeyboardEvent, PointerEvent } from 'react';
import type { CropBox } from '../api/client';
import { clampCrop, defaultCrop, maxCropScale, moveCrop, resizeCrop } from '../crop';

export interface CropEditorProps {
  imageUrl: string;
  initialCrop?: CropBox | null;
  onApply: (crop: CropBox) => void;
  onWholeImage: () => void;
  onCancel?: () => void;
}

type ImageState = { url: string; width: number; height: number; crop: CropBox | null; error: boolean };
type Drag = { url: string; pointerId: number; x: number; y: number; width: number; height: number; crop: CropBox };

function PixelInput({ label, value, min, max, step = 1, onCommit }: {
  label: string; value: number; min: number; max: number; step?: number; onCommit: (value: number) => void;
}) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);
  return <label className="crop-pixel-field"><span>{label}</span><input type="number" inputMode="numeric"
    min={min} max={max} step={step} value={draft}
    onChange={event => setDraft(event.target.value)}
    onBlur={() => {
      const number = draft.trim() ? Number(draft) : NaN;
      if (Number.isFinite(number)) onCommit(number);
      setDraft(String(Number.isFinite(number) ? Math.min(max, Math.max(min, Math.round(number / step) * step)) : value));
    }}
    onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur(); }} />
  </label>;
}

export function CropEditor({ imageUrl, initialCrop, onApply, onWholeImage, onCancel }: CropEditorProps) {
  const [loaded, setLoaded] = useState<ImageState | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const drag = useRef<Drag | null>(null);
  const id = useId();
  const current = loaded?.url === imageUrl ? loaded : null;
  const crop = current?.crop ?? null;
  const width = current?.width ?? 0;
  const height = current?.height ?? 0;
  const cropWidth = crop ? crop.right - crop.left : 0;
  const cropHeight = crop ? crop.bottom - crop.top : 0;

  function update(next: CropBox | null) {
    setLoaded(previous => previous?.url === imageUrl ? { ...previous, crop: next } : previous);
  }

  function startDrag(event: PointerEvent<HTMLButtonElement>) {
    if (!crop || !stageRef.current || (event.pointerType === 'mouse' && event.button !== 0)) return;
    const bounds = stageRef.current.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    event.currentTarget.focus({ preventScroll: true });
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { url: imageUrl, pointerId: event.pointerId, x: event.clientX, y: event.clientY, width: bounds.width, height: bounds.height, crop };
    event.preventDefault();
  }

  function moveDrag(event: PointerEvent<HTMLButtonElement>) {
    const active = drag.current;
    if (!active || active.url !== imageUrl || active.pointerId !== event.pointerId) return;
    update(moveCrop(width, height, active.crop,
      active.crop.left + (event.clientX - active.x) * width / active.width,
      active.crop.top + (event.clientY - active.y) * height / active.height));
  }

  function moveByKeyboard(event: KeyboardEvent<HTMLButtonElement>) {
    if (!crop) return;
    const delta = event.shiftKey ? 10 : 1;
    const directions: Record<string, [number, number]> = {
      ArrowLeft: [-delta, 0], ArrowRight: [delta, 0], ArrowUp: [0, -delta], ArrowDown: [0, delta],
    };
    const offset = directions[event.key];
    if (!offset) return;
    event.preventDefault();
    update(moveCrop(width, height, crop, crop.left + offset[0], crop.top + offset[1]));
  }

  return <div className="crop-editor" tabIndex={-1} aria-labelledby={`${id}-title`}>
    <div className="crop-editor-intro">
      <h4 id={`${id}-title`}>Frame your item</h4>
      <p>Keep the whole item and about 15% space around it.</p>
    </div>
    <div className="crop-editor-body">
      <div className="crop-preview">
        <div ref={stageRef} className="crop-stage" style={width && height ? { width: `${Math.min(420, width / height * 300)}px` } : undefined}>
          <img key={imageUrl} ref={imageRef} src={imageUrl} alt="Original image for choosing a search crop" draggable={false}
            className="crop-image" onLoad={event => {
              const image = event.currentTarget;
              if (image !== imageRef.current || image.getAttribute('src') !== imageUrl) return;
              const nextWidth = image.naturalWidth;
              const nextHeight = image.naturalHeight;
              setLoaded({ url: imageUrl, width: nextWidth, height: nextHeight, error: false,
                crop: initialCrop ? clampCrop(nextWidth, nextHeight, initialCrop) : defaultCrop(nextWidth, nextHeight) });
            }} onError={event => {
              if (event.currentTarget !== imageRef.current || event.currentTarget.getAttribute('src') !== imageUrl) return;
              setLoaded({ url: imageUrl, width: 0, height: 0, crop: null, error: true });
            }} />
          {crop && <button type="button" className="crop-selection"
            aria-label={`Move crop. Left ${crop.left}, top ${crop.top}, width ${cropWidth}, height ${cropHeight} pixels`}
            aria-describedby={`${id}-move-help`} onKeyDown={moveByKeyboard}
            onPointerDown={startDrag} onPointerMove={moveDrag}
            onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}
            onLostPointerCapture={() => { drag.current = null; }}
            style={{ left: `${crop.left / width * 100}%`, top: `${crop.top / height * 100}%`, width: `${cropWidth / width * 100}%`, height: `${cropHeight / height * 100}%` }}>
            <span className="crop-crosshair" aria-hidden="true">+</span>
            <span className="crop-ratio" aria-hidden="true">3 : 4</span>
          </button>}
        </div>
        {!current && <p className="crop-status" role="status">Loading image…</p>}
        {current?.error && <p className="crop-status" role="alert">This image could not be shown. You can still use the whole image.</p>}
        {current && !current.error && !crop && <p className="crop-status" role="status">This image is too small for a 3:4 crop. Use the whole image.</p>}
      </div>
      {crop && <div className="crop-controls">
        <p id={`${id}-move-help`} className="crop-move-help">Drag the box to move it. Or focus it and use arrow keys. Hold Shift for bigger steps.</p>
        <label className="crop-size" htmlFor={`${id}-size`}><span>Crop size</span><output>{cropWidth} × {cropHeight} px</output></label>
        <input id={`${id}-size`} className="crop-slider" type="range" min="1" max={maxCropScale(width, height)} step="1"
          value={cropWidth / 3} aria-valuetext={`${cropWidth} by ${cropHeight} pixels`}
          onChange={event => update(resizeCrop(width, height, crop, Number(event.target.value) * 3))} />
        <div className="crop-size-ends" aria-hidden="true"><span>Smaller</span><span>Larger</span></div>
        <details className="crop-fine-tune"><summary>Fine tune</summary>
          <p>Original image: {width} × {height} px. The height follows the width to keep a 3:4 frame.</p>
          <div className="crop-pixel-fields">
            <PixelInput label="Left" value={crop.left} min={0} max={width - cropWidth}
              onCommit={left => update(moveCrop(width, height, crop, left, crop.top))} />
            <PixelInput label="Top" value={crop.top} min={0} max={height - cropHeight}
              onCommit={top => update(moveCrop(width, height, crop, crop.left, top))} />
            <PixelInput label="Width" value={cropWidth} min={3} max={maxCropScale(width, height) * 3} step={3}
              onCommit={nextWidth => update(resizeCrop(width, height, crop, nextWidth))} />
          </div>
        </details>
      </div>}
    </div>
    <div className="crop-actions">
      <button className="crop-apply" type="button" disabled={!crop} onClick={() => { if (crop) onApply(crop); }}>Search this crop <span aria-hidden="true">→</span></button>
      <button className="crop-whole" type="button" onClick={onWholeImage}>Use whole image</button>
      {onCancel && <button className="crop-cancel" type="button" onClick={onCancel}>Cancel</button>}
    </div>
  </div>;
}
