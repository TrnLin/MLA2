"""API boundaries: original bytes, task isolation, and safe retained uploads."""
import hashlib
import io
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image
from fastapi.testclient import TestClient


def image_bytes(size=(137, 91), color='red', format='PNG'):
    buffer = io.BytesIO()
    Image.new('RGB', size, color).save(buffer, format=format)
    return buffer.getvalue()


def retained_directory():
    directory = Path(__file__).resolve().parents[1] / 'tmp/demo-api/tests' / uuid4().hex
    directory.mkdir(parents=True)
    return directory


def make_client(*, failure=False, max_bytes=10 * 1024 * 1024, max_pixels=25_000_000):
    from fashion_api.api import create_app
    from fashion.demo import ModelAdapter, ModelRuntime

    raw = image_bytes()
    def predict(path):
        assert path.read_bytes() == raw
        with Image.open(path) as image:
            assert image.size == (137, 91)
        return {'label': 'Tshirts', 'probabilities': {'Tshirts': 0.75, 'Shirts': 0.25}}

    def fail(path):
        raise RuntimeError('season is unavailable')

    adapters = {
        target: ModelAdapter(target, 'test run', 60, 80, 123, 'identity-' + target,
                             fail if failure and target == 'season' else predict)
        for target in ('articleType', 'season', 'gender', 'usage')
    }
    runtime = ModelRuntime(adapters=adapters, errors={'retrieval': 'missing search'})
    directory = retained_directory()
    client = TestClient(create_app(runtime=runtime, upload_dir=directory,
                                   max_bytes=max_bytes, max_pixels=max_pixels))
    return client, raw, directory, runtime


def test_original_bytes_are_retained_and_every_classifier_reads_them():
    client, raw, directory, _ = make_client()
    response = client.post('/api/analyze', content=raw, headers={'content-type': 'image/png'})
    assert response.status_code == 200
    body = response.json()
    assert body['image_id'] == hashlib.sha256(raw).hexdigest()
    assert (directory / body['image_id']).read_bytes() == raw
    assert set(body['predictions']) == {'articleType', 'season', 'gender', 'usage'}
    assert all(value['label'] == 'Tshirts' and value['error'] is None
               for value in body['predictions'].values())
    assert all(value['latency_ms'] >= 0 for value in body['predictions'].values())
    assert body['version'] == client.get('/api/health').json()['version']


def test_failed_task_does_not_hide_other_tasks_or_get_cached():
    client, raw, _, runtime = make_client(failure=True)
    first = client.post('/api/analyze', content=raw, headers={'content-type': 'image/png'}).json()
    assert first['predictions']['season']['label'] is None
    assert first['predictions']['season']['error'] == 'season is unavailable'
    assert first['predictions']['articleType']['label'] == 'Tshirts'
    runtime.adapters['season'].predictor = runtime.adapters['gender'].predictor
    second = client.post('/api/analyze', content=raw, headers={'content-type': 'image/png'}).json()
    assert second['predictions']['season']['error'] is None


@pytest.mark.parametrize('content,mime,status', [
    (b'bad', 'image/png', 422), (b'bad', 'text/plain', 415),
    (image_bytes(format='GIF'), 'image/png', 415),
    (image_bytes(), 'image/jpeg', 415),
])
def test_invalid_images_are_rejected_before_any_file_is_saved(content, mime, status):
    client, _, directory, _ = make_client()
    assert client.post('/api/analyze', content=content,
                       headers={'content-type': mime}).status_code == status
    assert list(directory.iterdir()) == []


def test_byte_and_decoded_pixel_limits_are_enforced_before_storage():
    client, raw, directory, _ = make_client(max_bytes=20)
    assert client.post('/api/analyze', content=raw,
                       headers={'content-type': 'image/png'}).status_code == 413
    assert list(directory.iterdir()) == []
    client, raw, directory, _ = make_client(max_pixels=100)
    assert client.post('/api/analyze', content=raw,
                       headers={'content-type': 'image/png'}).status_code == 413
    assert list(directory.iterdir()) == []


def test_unknown_images_and_products_cannot_resolve_server_paths():
    client, raw, _, _ = make_client()
    assert client.get('/api/images/' + 'a' * 64 + '/similar').status_code == 404
    assert client.get('/api/gallery/99999999/image').status_code == 404
    body = client.post('/api/analyze', content=raw, headers={'content-type': 'image/png'}).json()
    assert client.get(f"/api/images/{body['image_id']}/similar?limit=5").status_code == 503
    assert client.get(f"/api/images/{body['image_id']}/similar?limit=8").status_code == 422


def test_degraded_health_and_version_include_model_identity():
    client, _, _, runtime = make_client()
    body = client.get('/api/health').json()
    assert body['status'] == 'degraded'
    assert body['models']['retrieval']['status'] == 'error'
    assert body['models']['articleType']['width'] == 60
    assert body['models']['retrieval']['height'] == 320
    version = runtime.version
    from fashion.demo import ModelRuntime
    changed = dict(runtime.adapters)
    from dataclasses import replace
    changed['gender'] = replace(changed['gender'], identity='different-checkpoint')
    assert ModelRuntime(adapters=changed, errors=runtime.errors).version != version
    assert client.get('/api/metadata').json()['evaluation'] == []


def test_repeated_upload_marks_saved_task_timings_without_mutating_first_result():
    client, raw, _, _ = make_client()
    first = client.post('/api/analyze', content=raw, headers={'content-type': 'image/png'}).json()
    second = client.post('/api/analyze', content=raw, headers={'content-type': 'image/png'}).json()
    assert all(value['cached'] is False for value in first['predictions'].values())
    assert all(value['cached'] is True for value in second['predictions'].values())


def test_gallery_serves_only_allowlisted_bytes_with_saved_hash():
    from fashion_api.api import create_app
    from fashion.demo import ModelRuntime
    directory = retained_directory()
    raw = image_bytes(format='JPEG')
    (directory / '1.jpg').write_bytes(raw)
    (directory / '2.jpg').write_bytes(raw)
    runtime = ModelRuntime(adapters={},
                           gallery_records={1: {'sha256': hashlib.sha256(raw).hexdigest()}},
                           image_directory=directory)
    client = TestClient(create_app(runtime=runtime, upload_dir=retained_directory()))
    assert client.get('/api/gallery/1/image').content == raw
    assert client.get('/api/gallery/2/image').status_code == 404
    (directory / '1.jpg').write_bytes(image_bytes(color='blue', format='JPEG'))
    assert client.get('/api/gallery/1/image').status_code == 503
