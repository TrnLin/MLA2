import assert from 'node:assert/strict';
import { test } from 'node:test';
import { imageIdentity, analyzeImage, getSimilarItems, requireVersion, searchQueryKey } from '../src/api/client.ts';

test('responses from a changed model version cannot enter an older cache key', () => {
  assert.throws(() => requireVersion({ version: 'new' }, 'old'), { name: 'ModelVersionChanged' });
  const response = { version: 'same' };
  assert.equal(requireVersion(response, 'same'), response);
});

test('image identity comes from original bytes, not filename or URL', async () => {
  const a = new Blob(['same bytes'], { type: 'image/png' });
  const b = new File(['same bytes'], 'different.png', { type: 'image/png' });
  assert.equal(await imageIdentity(a), await imageIdentity(b));
  assert.notEqual(await imageIdentity(a), await imageIdentity(new Blob(['other bytes'])));
});

test('analysis sends the original bytes and cancellation signal', async () => {
  const original = globalThis.fetch;
  const blob = new Blob(['original image bytes'], { type: 'image/webp' });
  const controller = new AbortController();
  try {
    globalThis.fetch = async (url, options) => {
      assert.equal(url, '/api/analyze');
      assert.equal(options.body, blob);
      assert.equal(options.signal, controller.signal);
      assert.equal(options.headers['Content-Type'], 'image/webp');
      return Response.json({ image_id: 'abc', version: 'v1', predictions: {} });
    };
    assert.equal((await analyzeImage(blob, controller.signal)).image_id, 'abc');
  } finally { globalThis.fetch = original; }
});

test('retrieval requests exact image identity and count, and surfaces API errors', async () => {
  const original = globalThis.fetch;
  const controller = new AbortController();
  try {
    globalThis.fetch = async (url, options) => {
      assert.equal(url, '/api/images/abc/similar?limit=10');
      assert.equal(options.signal, controller.signal);
      return Response.json({ detail: 'Search model is unavailable.' }, { status: 503 });
    };
    await assert.rejects(getSimilarItems('abc', 10, controller.signal), /Search model is unavailable/);
  } finally { globalThis.fetch = original; }
});

test('an aborted request stays aborted rather than becoming a server error', async () => {
  const original = globalThis.fetch;
  const controller = new AbortController();
  controller.abort();
  try {
    globalThis.fetch = async (_url, options) => { options.signal.throwIfAborted(); };
    await assert.rejects(analyzeImage(new Blob(['x']), controller.signal), { name: 'AbortError' });
  } finally { globalThis.fetch = original; }
});

test('crop coordinates are sent separately from original image identity', async () => {
  const original = globalThis.fetch;
  const controller = new AbortController();
  const crop = { left: 190, top: 130, right: 730, bottom: 850 };
  try {
    globalThis.fetch = async (url, options) => {
      assert.equal(url, '/api/images/original/similar?limit=5&crop_left=190&crop_top=130&crop_right=730&crop_bottom=850');
      assert.equal(options.signal, controller.signal);
      assert.equal(options.body, undefined);
      return Response.json({ image_id: 'original', crop, items: [] });
    };
    assert.deepEqual((await getSimilarItems('original', 5, controller.signal, crop)).crop, crop);
  } finally { globalThis.fetch = original; }
});

test('whole image and different crops never share a search cache key', () => {
  const first = { left: 190, top: 130, right: 730, bottom: 850 };
  const shifted = { ...first, left: 180, right: 720 };
  const key = searchQueryKey('v1', 'image', 5, first);
  assert.notDeepEqual(key, searchQueryKey('v1', 'image', 5, null));
  assert.notDeepEqual(key, searchQueryKey('v1', 'image', 5, shifted));
  assert.notDeepEqual(key, searchQueryKey('v1', 'other', 5, first));
  assert.notDeepEqual(key, searchQueryKey('v2', 'image', 5, first));
  assert.notDeepEqual(key, searchQueryKey('v1', 'image', 10, first));
  assert.deepEqual(key, searchQueryKey('v1', 'image', 5, { ...first }));
});

test('an unapplied or mismatched crop cannot appear as cropped results', async () => {
  const original = globalThis.fetch;
  const crop = { left: 190, top: 130, right: 730, bottom: 850 };
  try {
    globalThis.fetch = async () => Response.json({ image_id: 'original', crop: null, items: [] });
    await assert.rejects(getSimilarItems('original', 5, new AbortController().signal, crop), /crop/i);
    globalThis.fetch = async () => Response.json({ image_id: 'other', crop, items: [] });
    await assert.rejects(getSimilarItems('original', 5, new AbortController().signal, crop), /image/i);
  } finally { globalThis.fetch = original; }
});
