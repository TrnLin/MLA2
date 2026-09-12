"""Real local artifacts: saved transforms, direct scores, and gallery boundaries."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from fashion.config import ROOT


@pytest.fixture(scope='module')
def runtime():
    from fashion.demo import load_runtime
    return load_runtime(ROOT)


def test_real_adapters_use_each_saved_transform_and_match_direct_predictions(runtime):
    assert runtime.errors == {}
    image = ROOT / 'data/train/images_train/1163.jpg'
    original = image.read_bytes()
    shapes = {}
    handles = []
    try:
        for target, adapter in runtime.adapters.items():
            def observe(model, inputs, target=target):
                shapes[target] = tuple(inputs[0].shape)
            import torch
            first_conv = next(module for module in adapter.model.modules()
                              if isinstance(module, torch.nn.Conv2d))
            handles.append(first_conv.register_forward_pre_hook(observe))
        from fashion_api.api import create_app
        from fastapi.testclient import TestClient
        with TestClient(create_app(runtime=runtime)) as client:
            response = client.post('/api/analyze', content=original,
                                   headers={'content-type': 'image/jpeg'})
            assert response.status_code == 200
            body = response.json()
            for target in ('articleType', 'season', 'gender', 'usage'):
                direct = runtime.adapters[target].predictor(image)
                assert body['predictions'][target]['error'] is None
                assert body['predictions'][target]['label'] == direct['label']
                assert body['predictions'][target]['probabilities'] == direct['probabilities']
                assert shapes[target] == (1, 3, 80, 60)
            # R5.encode invokes the encoder directly, so observe that module too.
            def observe_r5(model, inputs):
                shapes['retrieval'] = tuple(inputs[0].shape)
            import torch
            first_conv = next(module for module in runtime.search_bundle.model.modules()
                              if isinstance(module, torch.nn.Conv2d))
            handles.append(first_conv.register_forward_pre_hook(observe_r5))
            matches = client.get(f"/api/images/{body['image_id']}/similar?limit=10")
            assert matches.status_code == 200, matches.text
            items = matches.json()['items']
            assert len(items) == 10
            assert shapes['retrieval'] == (1, 3, 320, 240)
            assert matches.json()['gallery_size'] == 26217
            assert 1163 not in [item['id'] for item in items]
            assert all(item['id'] in runtime.gallery_records for item in items)
            for item in items:
                assert item['score'] == pytest.approx(1 - item['distance'])
            allowed = client.get(f"/api/gallery/{items[0]['id']}/image")
            assert allowed.status_code == 200
            expected_hash = runtime.gallery_records[items[0]['id']]['sha256']
            assert hashlib.sha256(allowed.content).hexdigest() == expected_hash
    finally:
        for handle in handles:
            handle.remove()
    assert image.read_bytes() == original


def test_gallery_exclusion_removes_same_hash_duplicates_and_family():
    from fashion.demo import eligible_gallery_mask
    import pandas as pd
    metadata = pd.DataFrame([
        {'id': 1, 'sha256': 'abc', 'duplicate_group': 'd1', 'product_family_group': 'f1'},
        {'id': 2, 'sha256': 'def', 'duplicate_group': 'd1', 'product_family_group': 'f2'},
        {'id': 3, 'sha256': 'ghi', 'duplicate_group': 'd3', 'product_family_group': 'f1'},
        {'id': 4, 'sha256': 'jkl', 'duplicate_group': 'd4', 'product_family_group': 'f4'},
    ])
    assert np.array_equal(eligible_gallery_mask(metadata, metadata, 'abc'),
                          [False, False, False, True])


def test_metadata_uses_saved_scores_and_discloses_different_gallery(runtime):
    values = {item['target']: item for item in runtime.evaluation}
    assert values['articleType']['metrics']['Accuracy'] == '84.93%'
    assert values['season']['metrics']['Accuracy'] == '76.43%'
    assert values['gender']['metrics']['Accuracy'] == '90.46%'
    assert values['gender']['metrics']['Macro F1'] == '78.74%'
    assert values['usage']['metrics']['Macro F1'] == '42.26%'
    assert '32,773' in values['retrieval']['note']
    assert '26,217' in values['retrieval']['note']
    assert 'selected using development data' in values['gender']['note']


def test_gender_uses_selected_mixup_checkpoint(runtime):
    assert runtime.adapters['gender'].run_id == (
        't3_gender_name_truth_mixup_alpha020_refit_20260911T041436Z_3af0b94e')
    assert runtime.adapters['gender'].parameters == 390181


def test_gender_rejects_wrong_package_run(monkeypatch):
    from fashion import demo
    read_json = demo.read_json

    def wrong_run(path):
        data = read_json(path)
        if path == ROOT / 'model-weight/task3/gender_model/package_manifest.json':
            data['run_id'] = 'old-gender-run'
        return data

    monkeypatch.setattr(demo, 'read_json', wrong_run)
    with pytest.raises(ValueError, match='accepted run_id differs'):
        demo._task3(ROOT, 'gender')


def test_gender_omits_scores_from_another_run(runtime, monkeypatch):
    from fashion import demo
    read_json = demo.read_json

    def stale_scores(path):
        data = read_json(path)
        if path.name == 'evaluation.json':
            data['run_id'] = 'old-gender-run'
        return data

    monkeypatch.setattr(demo, 'read_json', stale_scores)
    assert demo.load_evaluation(ROOT, {'gender': runtime.adapters['gender']}) == []
