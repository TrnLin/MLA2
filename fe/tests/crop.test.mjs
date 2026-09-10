import assert from 'node:assert/strict';
import test from 'node:test';
import { defaultCrop, clampCrop, moveCrop, resizeCrop, maxCropScale } from '../src/crop.ts';

test('default crops stay centered in square, portrait and landscape images', () => {
  assert.deepEqual(defaultCrop(900, 900), { left: 180, top: 90, right: 720, bottom: 810 });
  assert.deepEqual(defaultCrop(300, 800), { left: 30, top: 240, right: 270, bottom: 560 });
  assert.deepEqual(defaultCrop(1200, 400), { left: 480, top: 40, right: 720, bottom: 360 });
});

test('the smallest valid image gives a whole 3 by 4 crop; smaller images disable cropping', () => {
  assert.deepEqual(defaultCrop(3, 4), { left: 0, top: 0, right: 3, bottom: 4 });
  for (const [width, height] of [[2, 4], [3, 3], [0, 10], [NaN, 10], [Infinity, 20]]) {
    assert.equal(defaultCrop(width, height), null);
    assert.equal(maxCropScale(width, height), 0);
  }
});

test('clamping keeps the exact requested crop and pulls out-of-bounds crops inside', () => {
  assert.deepEqual(clampCrop(900, 900, { left: 190, top: 130, right: 730, bottom: 850 }),
    { left: 190, top: 130, right: 730, bottom: 850 });
  assert.deepEqual(clampCrop(900, 900, { left: -30, top: 500, right: 510, bottom: 1220 }),
    { left: 0, top: 180, right: 540, bottom: 900 });
});

test('moving clamps all edges without changing crop size', () => {
  const crop = { left: 190, top: 130, right: 730, bottom: 850 };
  assert.deepEqual(moveCrop(900, 900, crop, 999, -20), { left: 360, top: 0, right: 900, bottom: 720 });
  assert.deepEqual(moveCrop(900, 900, crop, 10.6, 20.3), { left: 11, top: 20, right: 551, bottom: 740 });
});

test('resizing keeps the center where possible and caps the crop at the image bounds', () => {
  const crop = { left: 190, top: 130, right: 730, bottom: 850 };
  assert.deepEqual(resizeCrop(900, 900, crop, 300), { left: 310, top: 290, right: 610, bottom: 690 });
  assert.deepEqual(resizeCrop(900, 900, crop, 3000), { left: 123, top: 0, right: 798, bottom: 900 });
});

test('odd image bounds, invalid numbers and minimum sizes never emit non-integer or invalid crops', () => {
  for (const [width, height] of [[901, 899], [5, 7], [60, 80], [80, 60]]) {
    for (const desiredWidth of [-100, 0, 1, 7, 99999, NaN, Infinity]) {
      const crop = resizeCrop(width, height, defaultCrop(width, height), desiredWidth);
      assert.ok(Object.values(crop).every(Number.isInteger));
      assert.ok(crop.left >= 0 && crop.top >= 0 && crop.right <= width && crop.bottom <= height);
      assert.ok(crop.right > crop.left && crop.bottom > crop.top);
      assert.equal((crop.right - crop.left) * 4, (crop.bottom - crop.top) * 3);
    }
  }
  assert.deepEqual(clampCrop(60, 80, { left: NaN, top: Infinity, right: NaN, bottom: Infinity }),
    { left: 0, top: 0, right: 3, bottom: 4 });
  assert.equal(resizeCrop(2, 3, { left: 0, top: 0, right: 3, bottom: 4 }, 30), null);
});
