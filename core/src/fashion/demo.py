"""Adapters around the frozen predictors, with separate saved image transforms."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from fashion.config import ROOT

if TYPE_CHECKING:
    from fashion.task4.search import CropBox

CLASSIFIERS = ('articleType', 'season', 'gender', 'usage')
TARGETS = (*CLASSIFIERS, 'retrieval')


def digest(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


@dataclass
class ModelAdapter:
    name: str
    run_id: str
    width: int
    height: int
    parameters: int
    identity: str
    predictor: Callable[[Path], dict[str, Any]] | None
    model: Any = None


def eligible_gallery_mask(metadata, development, source_sha256):
    """Exclude the query's exact image, duplicate group and product family."""
    matches = development.loc[development['sha256'].eq(source_sha256)]
    mask = ~metadata['sha256'].eq(source_sha256)
    for column in ('id', 'duplicate_group', 'product_family_group'):
        values = matches[column].dropna()
        values = values.loc[values.astype(str).ne('')]
        mask &= ~metadata[column].isin(values)
    return mask.to_numpy(dtype=bool)


class ModelRuntime:
    def __init__(self, *, adapters, errors=None, search_bundle=None,
                 development=None, gallery_records=None, image_directory=None,
                 evaluation=None):
        self.adapters = adapters
        self.errors = errors or {}
        self.search_bundle = search_bundle
        self.development = development
        self.gallery_records = gallery_records or {}
        self.image_directory = image_directory
        self.evaluation = evaluation or []
        identities = {target: adapter.identity for target, adapter in adapters.items()}
        identities['errors'] = self.errors
        identities['api_contract'] = 2
        if search_bundle is not None:
            identities['gallery'] = search_bundle.gallery.identity_sha256
        self.version = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()

    @property
    def info(self):
        result = {}
        for target in TARGETS:
            adapter = self.adapters.get(target)
            result[target] = {
                'status': 'ready' if adapter else 'error',
                'name': adapter.name if adapter else target,
                'run_id': adapter.run_id if adapter else '',
                'width': adapter.width if adapter else (240 if target == 'retrieval' else 60),
                'height': adapter.height if adapter else (320 if target == 'retrieval' else 80),
                'parameters': adapter.parameters if adapter else 0,
            }
            if not adapter:
                result[target]['error'] = self.errors.get(target, 'Model unavailable.')
        return result

    @property
    def gallery_size(self):
        return len(self.gallery_records)

    def similar(self, path: Path, limit: int, crop: CropBox | None = None):
        if self.search_bundle is None:
            raise RuntimeError(self.errors.get('retrieval', 'Search unavailable.'))
        from fashion.task4.probe import rank_single_embedding
        from fashion.task4.search import _encode_query, _prepare_query

        bundle = self.search_bundle
        query = _prepare_query(bundle, image_path=path, query_id=None, crop=crop)
        embedding = _encode_query(bundle, query)
        mask = eligible_gallery_mask(bundle.gallery.metadata, self.development,
                                     query.source_sha256)
        ranked = rank_single_embedding(query_feature=embedding,
                                       gallery_ids=bundle.gallery.ids[mask],
                                       gallery_features=bundle.gallery.features[mask], max_k=limit)
        items = []
        for row in ranked.itertuples(index=False):
            product_id = int(row.candidate_id)
            metadata = self.gallery_records[product_id]
            items.append({'id': product_id, 'name': str(metadata['productDisplayName']),
                          'image': f'/api/gallery/{product_id}/image',
                          'score': 1.0 - float(row.distance), 'distance': float(row.distance),
                          'rank': int(row.rank)})
        return items

    def gallery_image(self, product_id: int) -> bytes:
        record = self.gallery_records[product_id]
        path = self.image_directory / f'{product_id}.jpg'
        if path.is_symlink() or path.resolve().parent != self.image_directory.resolve():
            raise ValueError('Invalid gallery path.')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Gallery image SHA-256 differs from saved gallery.')
        return data


def _task1(root):
    from fashion.task1.final_inference import load_task1_bundle, predict_article_type
    bundle = load_task1_bundle(root / 'models/task1_article_type.manifest.json',
                               registry_path=root / 'results/runs.csv', project_root=root,
                               device='cpu')
    def predict(path):
        result = predict_article_type(bundle, path)
        return {'label': result.predicted_label, 'probabilities': result.probabilities}
    return ModelAdapter('Task 1 · SmallCNN', bundle.run_id, 60, 80,
                        sum(p.numel() for p in bundle.model.parameters()),
                        bundle.manifest_sha256 + bundle.bundle_sha256, predict, bundle.model)


def _task2(root):
    from fashion.task2.inference import load_season_bundle, predict_season
    bundle = load_season_bundle(root / 'models/task2_season.manifest.json',
                                registry_path=(root / 'results/evidence/task2/final_handoff'
                                               / 'registry_snapshot.csv'),
                                project_root=root, device='cpu')
    def predict(path):
        result = predict_season(bundle, path)
        return {'label': result.predicted_label, 'probabilities': result.probabilities,
                'review_required': result.review_required}
    return ModelAdapter('Task 2 · Season CNN', bundle.run_id, 60, 80,
                        sum(p.numel() for p in bundle.model.parameters()),
                        bundle.manifest_sha256 + bundle.bundle_sha256, predict, bundle.model)


def _task3(root, target):
    from fashion.data.images import load_and_transform_image
    from fashion.train.config import Task3BaselineConfig
    from fashion.train.model import Task3BaselineCNN, Task3GeM3CNN
    from fashion.train.task3_gender_precision import ieee_precision

    package = root / 'model-weight/task3' / f'{target}_model'
    accepted_name = ('gender_final_sam25_refit_20260907' if target == 'gender'
                     else 'usage_final_e8_refit_20260907')
    accepted_path = root / 'reports/task3' / accepted_name / 'model_manifest.json'
    manifest_path = package / 'package_manifest.json'
    accepted = read_json(accepted_path)
    manifest = read_json(manifest_path)
    if (accepted['status'] != 'accepted_final' or accepted['scratch'] is not True
            or accepted['target'] != target or manifest['target'] != target):
        raise ValueError('Task 3 package is not the accepted scratch model.')
    for key in ('run_id', 'selected_epoch', 'class_names'):
        if manifest[key] != accepted[key]:
            raise ValueError(f'Task 3 accepted {key} differs from package.')
    expected_files = {'final_epoch.pt', 'config.json', 'normalization.json', 'class_names.json'}
    if set(manifest['files']) != expected_files:
        raise ValueError('Task 3 package file map differs.')
    for name, record in manifest['files'].items():
        path = package / name
        if digest(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError(f'Task 3 file identity failed: {name}')
    for key, filename in [('checkpoint', 'final_epoch.pt'), ('config', 'config.json'),
                          ('normalization', 'normalization.json')]:
        if accepted[key]['sha256'] != manifest['files'][filename]['sha256']:
            raise ValueError(f'Task 3 accepted {key} identity differs.')
    if digest(root / 'data/processed/splits.csv') != accepted['files']['data/processed/splits.csv']:
        raise ValueError('Canonical splits differ from accepted Task 3 model.')
    config = read_json(package / 'config.json')
    stats = read_json(package / 'normalization.json')
    classes = read_json(package / 'class_names.json')
    checkpoint = torch.load(package / 'final_epoch.pt', map_location='cpu', weights_only=True)
    if (classes != accepted['class_names'] or checkpoint['class_names'] != classes
            or checkpoint['run_id'] != accepted['run_id']
            or checkpoint['selected_epoch'] != accepted['selected_epoch']
            or checkpoint['config'] != config
            or any(stats.get(key) != value for key, value in checkpoint['normalization'].items())):
        raise ValueError('Task 3 checkpoint metadata differs from accepted package.')
    base = config if target == 'gender' else config['base_config']
    kwargs = {field.name: base[field.name] for field in fields(Task3BaselineConfig)}
    kwargs['channels'] = tuple(kwargs['channels'])
    if (kwargs['image_width'], kwargs['image_height']) != (60, 80):
        raise ValueError('Task 3 image size differs from frozen width 60, height 80.')
    model = (Task3GeM3CNN(Task3BaselineConfig(**kwargs),
                         classifier_dropout=config['classifier_dropout']) if target == 'gender'
             else Task3BaselineCNN(Task3BaselineConfig(**kwargs)))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != accepted['parameter_count']:
        raise ValueError('Task 3 parameter count differs from accepted model.')
    def predict(path):
        array = load_and_transform_image(path,
                                         image_size=(kwargs['image_height'], kwargs['image_width']),
                                         mean=stats['mean'], std=stats['std'])
        inputs = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)
        with ieee_precision(torch), torch.inference_mode():
            probabilities = model(inputs).softmax(dim=1)[0]
        if not torch.isfinite(probabilities).all():
            raise ValueError('Task 3 returned non-finite probabilities.')
        return {'label': classes[int(probabilities.argmax())],
                'probabilities': dict(zip(classes, probabilities.tolist(), strict=True))}
    return ModelAdapter('Task 3 · ' + ('Gender GeM CNN' if target == 'gender' else 'Usage E8 CNN'),
                        accepted['run_id'], 60, 80, parameters,
                        digest(manifest_path) + digest(accepted_path), predict, model)


def _retained_snapshot(source, destination, filenames):
    """Copy frozen bytes into a fresh retained directory; no temporary cleanup."""
    destination.mkdir(parents=True, exist_ok=False)
    for name in filenames:
        with (source / name).open('rb') as incoming, (destination / name).open('xb') as outgoing:
            while chunk := incoming.read(1024 * 1024):
                outgoing.write(chunk)
    return destination


def _retrieval(root):
    from fashion.data.dataset import load_splits
    from fashion.task4.gallery_artifact import _load_teacher_gallery_snapshot
    from fashion.task4.image_safety import reject_sealed_image_rows
    from fashion.task4.portable_model import load_r5_inference_package
    from fashion.task4.preprocessing import PreprocessingContract
    from fashion.task4.protocol import build_development_views
    from fashion.task4.search import SearchBundle, _checkpoint_sha256, _teacher_statistics

    snapshots = root / 'tmp/demo-api/snapshots' / uuid4().hex
    package = _retained_snapshot(root / 'model-weight/task4_r5', snapshots / 'r5',
                                  ('manifest.json', 'weights.pt'))
    gallery_source = root / 'models/task4_teacher_gallery'
    gallery_snapshot = _retained_snapshot(gallery_source, snapshots / 'gallery',
                                          ('manifest.json', 'metadata.csv', 'ids.npy',
                                           'features.npy', 'README.md'))
    manifest = read_json(package / 'manifest.json')
    model = load_r5_inference_package(package, device='cpu')
    contract = PreprocessingContract(240, 320)
    if manifest['input_contract'] != contract.to_dict():
        raise ValueError('R5 requires its saved RGB width 240, height 320 letterbox transform.')
    mean, std = _teacher_statistics(manifest)
    gallery = _load_teacher_gallery_snapshot(gallery_snapshot, artifact_directory=gallery_source,
                                             snapshot_owner=None)
    if (_checkpoint_sha256(gallery.manifest, label='gallery')
            != _checkpoint_sha256(manifest, label='model')
            or gallery.manifest['r5_checkpoint']['run_id']
            != manifest['source_checkpoint']['run_id']):
        raise ValueError('Gallery does not match the R5 checkpoint.')
    splits = load_splits(root / 'data/processed/splits.csv')
    primary, _ = build_development_views(splits, validation_fold=1)
    expected_ids = np.sort(pd.to_numeric(primary.gallery['id']).to_numpy(dtype=np.int64))
    if not np.array_equal(gallery.ids, expected_ids) or len(gallery.ids) != 26217:
        raise ValueError('Gallery differs from the fixed fold-1 teacher development gallery.')
    if gallery.manifest['fold'] != 1:
        raise ValueError('Gallery must be the fixed fold-1 gallery.')
    reject_sealed_image_rows(gallery.metadata, require_development=True)
    reject_sealed_image_rows(primary.queries, require_development=True)
    canonical = splits.set_index('id')
    metadata = gallery.metadata.set_index('id')
    for column in ('sha256', 'partition', 'cv_fold', 'duplicate_group', 'product_family_group'):
        expected = canonical.loc[metadata.index, column].astype(str)
        if not metadata[column].astype(str).equals(expected):
            raise ValueError(f'Gallery {column} differs from canonical splits.')
    bundle = SearchBundle(model=model, device=torch.device('cpu'), contract=contract,
                          teacher_mean=mean, teacher_std=std, model_manifest=manifest,
                          model_manifest_sha256=digest(package / 'manifest.json'), gallery=gallery,
                          query_catalogue=primary.queries.copy(),
                          splits=splits.loc[:, ['id', 'sha256', 'partition', 'cv_fold']].copy())
    adapter = ModelAdapter('Task 4 · R5 scratch autoencoder',
                            manifest['source_checkpoint']['run_id'],
                            240, 320, sum(p.numel() for p in model.parameters()),
                            bundle.model_manifest_sha256 + gallery.identity_sha256,
                            None, model)
    development = splits.loc[splits['partition'].eq('development'),
                             ['id', 'sha256', 'duplicate_group', 'product_family_group']].copy()
    return adapter, bundle, development


def load_runtime(root: Path = ROOT):
    root = Path(root).resolve()
    torch.set_num_threads(2)
    adapters, errors = {}, {}
    for target, loader in [('articleType', _task1), ('season', _task2),
                            ('gender', lambda p: _task3(p, 'gender')),
                            ('usage', lambda p: _task3(p, 'usage'))]:
        try:
            adapters[target] = loader(root)
        except Exception as error:
            errors[target] = str(error)
    bundle = development = None
    try:
        adapters['retrieval'], bundle, development = _retrieval(root)
    except Exception as error:
        errors['retrieval'] = str(error)
    records = ({} if bundle is None else
               {int(row['id']): row for row in bundle.gallery.metadata.to_dict('records')})
    return ModelRuntime(adapters=adapters, errors=errors, search_bundle=bundle,
                        development=development, gallery_records=records,
                        image_directory=root / 'data/train/images_train',
                        evaluation=load_evaluation(root, adapters))


def load_evaluation(root: Path, available_targets) -> list[dict]:
    """Read recorded evaluation summaries; absent evidence supplies no score."""
    result = []
    sources = {
        'articleType': 'results/evidence/task1/final_evaluation/metrics.csv',
        'season': 'results/evidence/task2/final_evaluation/holdout_metrics.json',
        'gender': 'reports/task3/gender_sam25_refit_holdout_20260907/evaluation.json',
        'usage': 'reports/task3/usage_e8_refit_holdout_20260907/evaluation.json',
        'retrieval': 'results/evidence/task4/final_evaluation/holdout_scorecard.csv',
    }
    notes = {
        'articleType': ('Saved reserved holdout results: 5,778 images. '
                        'Macro F1 gives each class equal weight.'),
        'season': ('Saved reserved holdout results: 5,778 images, '
                   'with the frozen temperature calibration.'),
        'gender': ('Accepted after holdout review; this was not a fresh blind selection. '
                   'Unisex remains weak.'),
        'usage': ('Accepted after holdout review; this was not a fresh blind selection. '
                  'Rare usage classes remain weak.'),
        'retrieval': ('Saved scores use a 32,773-product development gallery; '
                      'this demo uses 26,217. Relevance comes from catalogue tags, '
                      'not human similarity ratings.'),
    }
    titles = {
        'articleType': 'Article type — saved holdout scores',
        'season': 'Season — saved holdout scores',
        'gender': 'Gender — saved holdout scores',
        'usage': 'Usage — saved holdout scores',
        'retrieval': 'Visual search — saved holdout scores',
    }
    for target in available_targets:
        source = sources[target]
        try:
            path = root / source
            if target == 'articleType':
                row = pd.read_csv(path).iloc[0]
                scores = {'Accuracy': row['top1_accuracy'], 'Macro F1': row['macro_f1'],
                          'Top 5 accuracy': row['top5_accuracy']}
            elif target == 'retrieval':
                frame = pd.read_csv(path)
                row = frame.loc[frame['method'].eq('r5_scratch_autoencoder')
                                & frame['direction'].eq('teacher')].iloc[0]
                scores = {'nDCG at 10': row['ndcg_at_10_query_mean'],
                          'Any relevance at 10': row['precision_any_at_10']}
            else:
                data = read_json(path)
                if target == 'season':
                    row = data['I2_frozen_temperature']
                else:
                    key = 'Original labels: Single refit' if target == 'gender' else 'Single refit'
                    row = data['metrics'][key]
                scores = {'Accuracy': row['accuracy'], 'Macro F1': row['macro_f1']}
            if not all(np.isfinite(float(value)) for value in scores.values()):
                continue
            result.append({'target': target, 'title': titles[target], 'source': source,
                           'metrics': {key: f'{float(value):.2%}' for key, value in scores.items()},
                           'note': notes[target]})
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            continue
    return result
