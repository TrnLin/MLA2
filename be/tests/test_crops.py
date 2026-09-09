"""Crop API geometry and parity with the saved R5 transform and gallery."""
import hashlib
import io
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

from fashion.config import ROOT
from fashion.demo import ModelRuntime, _retrieval, eligible_gallery_mask
from fashion_api.api import create_app


def retained_directory():
    directory = Path(__file__).resolve().parents[1] / 'tmp/demo-api/crop-tests' / uuid4().hex
    directory.mkdir(parents=True)
    return directory


def upload(client, raw, mime='image/png'):
    response = client.post('/api/analyze', content=raw, headers={'content-type': mime})
    assert response.status_code == 200
    return response.json()['image_id']


def crop_params(left=0, top=0, right=60, bottom=80):
    return dict(crop_left=left, crop_top=top, crop_right=right, crop_bottom=bottom)


@pytest.fixture(scope='module')
def retrieval_runtime():
    torch.set_num_threads(2)
    adapter, bundle, development = _retrieval(ROOT)
    return ModelRuntime(
        adapters={'retrieval': adapter}, search_bundle=bundle, development=development,
        gallery_records={int(row['id']): row for row in bundle.gallery.metadata.to_dict('records')},
    )


@pytest.mark.parametrize('params', [
    {'crop_left': 0},
    {'crop_left': 0, 'crop_top': 0, 'crop_right': 60},
    crop_params(left='no'), crop_params(left='0.0'), crop_params(right='60.5'),
    crop_params(left=-1), crop_params(top=-1),
    crop_params(right=0), crop_params(bottom=0), crop_params(left=61),
    crop_params(right=61), crop_params(left=60, right=120),
    crop_params(bottom=160, right=120), crop_params(top=4, bottom=84),
])
def test_invalid_crop_is_422_before_search(params):
    buffer = io.BytesIO()
    Image.new('RGB', (100, 80), 'red').save(buffer, format='PNG')
    client = TestClient(create_app(runtime=ModelRuntime(adapters={}),
                                   upload_dir=retained_directory()))
    image_id = upload(client, buffer.getvalue())
    response = client.get(f'/api/images/{image_id}/similar', params=params)
    assert response.status_code == 422, response.text


def test_crop_bounds_use_exif_oriented_pixels(retrieval_runtime):
    # Encoded landscape becomes 60 x 100 after EXIF rotation.
    image = Image.new('RGB', (100, 60), 'red')
    image.paste('blue', (0, 0, 50, 60))
    exif = Image.Exif()
    exif[274] = 6
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', exif=exif)
    raw = buffer.getvalue()
    directory = retained_directory()
    client = TestClient(create_app(runtime=retrieval_runtime, upload_dir=directory))
    image_id = upload(client, raw, 'image/jpeg')
    url = f'/api/images/{image_id}/similar'
    accepted = client.get(url, params=crop_params(0, 20, 60, 100))
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()['crop'] == {'left': 0, 'top': 20, 'right': 60, 'bottom': 100}
    # These fit the encoded width, but fall outside the oriented width.
    rejected = client.get(url, params=crop_params(55, 0, 100, 60))
    assert rejected.status_code == 422
    assert (directory / image_id).read_bytes() == raw


def test_real_crop_tensor_rankings_exclusion_and_cache(retrieval_runtime):
    from fashion.task4.preprocessing import normalize_for_model, preprocess_image
    from fashion.task4.probe import rank_single_embedding

    runtime = retrieval_runtime
    bundle = runtime.search_bundle
    source = ROOT / 'data/train/images_train/1163.jpg'
    raw = source.read_bytes()
    directory = retained_directory()
    client = TestClient(create_app(runtime=runtime, upload_dir=directory))
    image_id = upload(client, raw, 'image/jpeg')
    url = f'/api/images/{image_id}/similar'
    full = client.get(url).json()
    assert full['crop'] is None
    assert full['cached'] is False

    captured = []
    first_conv = next(module for module in bundle.model.modules()
                      if isinstance(module, torch.nn.Conv2d))
    handle = first_conv.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().cpu().clone()))
    try:
        response = client.get(url, params=crop_params(6, 8, 54, 72))
    finally:
        handle.remove()
    assert response.status_code == 200, response.text
    cropped = response.json()
    assert cropped['crop'] == {'left': 6, 'top': 8, 'right': 54, 'bottom': 72}
    assert cropped['cached'] is False
    assert tuple(captured[0].shape) == (1, 3, 320, 240)

    # Build the input directly from the chosen pixels, bypassing API/query preparation.
    with Image.open(source) as image:
        pixels = ImageOps.exif_transpose(image).crop((6, 8, 54, 72))
        transformed = preprocess_image(pixels, bundle.contract)
    normalized = normalize_for_model(transformed, mean=bundle.teacher_mean,
                                     std=bundle.teacher_std)
    expected_tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).unsqueeze(0)
    torch.testing.assert_close(captured[0], expected_tensor, rtol=0, atol=0)
    with torch.inference_mode():
        embedding = bundle.model.encode(expected_tensor).cpu().numpy()[0]
    mask = eligible_gallery_mask(bundle.gallery.metadata, runtime.development,
                                 hashlib.sha256(raw).hexdigest())
    ranked = rank_single_embedding(query_feature=embedding, gallery_ids=bundle.gallery.ids[mask],
                                   gallery_features=bundle.gallery.features[mask], max_k=5)
    assert [item['id'] for item in cropped['items']] == ranked['candidate_id'].tolist()
    np.testing.assert_allclose([item['distance'] for item in cropped['items']], ranked['distance'],
                               rtol=0, atol=1e-6)
    assert 1163 not in [item['id'] for item in cropped['items']]
    assert all(item['id'] in bundle.gallery.ids[mask] for item in cropped['items'])
    assert [item['id'] for item in cropped['items']] != [item['id'] for item in full['items']]

    for params, expected in [({}, full), (crop_params(6, 8, 54, 72), cropped)]:
        repeated = client.get(url, params=params).json()
        assert repeated['cached'] is True
        assert repeated['items'] == expected['items']
        assert repeated['crop'] == expected['crop']
    changed = client.get(url, params=crop_params(0, 0, 30, 40)).json()
    assert changed['cached'] is False
    assert changed['crop'] == {'left': 0, 'top': 0, 'right': 30, 'bottom': 40}
    ten = client.get(url, params={**crop_params(6, 8, 54, 72), 'limit': 10}).json()
    assert ten['cached'] is False
    assert len(ten['items']) == 10
    assert ten['items'][:5] == cropped['items']
    assert (directory / image_id).read_bytes() == raw
    assert source.read_bytes() == raw
