export type Target = 'articleType' | 'season' | 'gender' | 'usage';
export type ModelTarget = Target | 'retrieval';
export type Prediction = {
  label: string | null;
  probabilities: Record<string, number>;
  latency_ms: number;
  run_id: string;
  error: string | null;
  review_required?: boolean | null;
  cached?: boolean;
};
export type Analysis = {
  image_id: string;
  version: string;
  predictions: Record<Target, Prediction>;
  elapsed_ms: number;
};
export type ModelInfo = {
  status: 'ready' | 'error';
  name: string;
  run_id: string;
  width: number;
  height: number;
  parameters: number;
  error?: string;
};
export type Evaluation = {
  target: string;
  title: string;
  source: string;
  metrics: Record<string, number | string>;
  note: string;
};
export type Metadata = {
  version: string;
  models: Record<ModelTarget, ModelInfo>;
  evaluation: Evaluation[];
};
export type SimilarItem = {
  id: number;
  name: string;
  image: string;
  score: number;
  distance: number;
  rank: number;
};
export type CropBox = { left: number; top: number; right: number; bottom: number };
export type SearchResult = {
  image_id: string;
  version: string;
  items: SimilarItem[];
  latency_ms: number;
  gallery_size: number;
  method: string;
  cached?: boolean;
  crop: CropBox | null;
};

export class ModelVersionChanged extends Error {
  constructor() { super('The model server changed. Refreshing its settings…'); this.name = 'ModelVersionChanged'; }
}

export function requireVersion<T extends { version: string }>(response: T, version: string): T {
  if (response.version !== version) throw new ModelVersionChanged();
  return response;
}

async function request<T>(url: string, options: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(url, options); }
  catch (error) {
    if (options.signal?.aborted) throw error;
    throw new Error('Cannot reach the local model server. Check that it is running, then try again.');
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === 'string' ? body.detail : response.status >= 500
      ? 'The local model server is unavailable. Check that it is running, then try again.'
      : `The model server returned an error (${response.status}).`);
  }
  return response.json() as Promise<T>;
}

export async function imageIdentity(blob: Blob): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer());
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}

export const getMetadata = (signal: AbortSignal) => request<Metadata>('/api/metadata', { signal });
export const analyzeImage = (blob: Blob, signal: AbortSignal) => request<Analysis>('/api/analyze', {
  method: 'POST', body: blob, headers: { 'Content-Type': blob.type }, signal,
});
export function searchQueryKey(version: string | undefined, id: string | null, limit: number, crop: CropBox | null) {
  return ['similar-items', version, id, limit,
    crop ? [crop.left, crop.top, crop.right, crop.bottom] : null] as const;
}

export async function getSimilarItems(id: string, limit: number, signal: AbortSignal, crop: CropBox | null = null) {
  const parameters = new URLSearchParams({ limit: String(limit) });
  if (crop) {
    for (const field of ['left', 'top', 'right', 'bottom'] as const) {
      parameters.set(`crop_${field}`, String(crop[field]));
    }
  }
  const result = await request<SearchResult>(`/api/images/${encodeURIComponent(id)}/similar?${parameters}`, { signal });
  if (result.image_id !== id) throw new Error('The search returned a different image. Please try again.');
  const applied = result.crop ?? null;
  if (Boolean(applied) !== Boolean(crop) || (crop && applied &&
      (['left', 'top', 'right', 'bottom'] as const).some(field => applied[field] !== crop[field]))) {
    throw new Error('The server did not apply the selected crop. Please try the search again.');
  }
  return result;
}
