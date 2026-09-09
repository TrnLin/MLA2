"""Run with: ./.venv/bin/python -m uvicorn fashion_api.api:app --host 127.0.0.1."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, Response
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool

from fashion.demo import CLASSIFIERS, ModelRuntime, load_runtime
from fashion.task4.search import CropBox
from fashion_api.storage import UploadStore
from fashion_api.config import UPLOAD_DIR


CropCoordinate = Annotated[str | None, Query(pattern=r'^[0-9]+$', max_length=10)]


def checked_crop(path: Path, coordinates: tuple[str | None, ...]) -> CropBox | None:
    if all(value is None for value in coordinates):
        return None
    if any(value is None for value in coordinates):
        raise HTTPException(422, 'Supply all four crop coordinates.')
    try:
        crop = CropBox(*(int(value) for value in coordinates))
        if 4 * (crop.right - crop.left) != 3 * (crop.bottom - crop.top):
            raise ValueError('Crop must have a 3:4 width-to-height ratio.')
        with Image.open(path) as image, ImageOps.exif_transpose(image) as oriented:
            if crop.right > oriented.width or crop.bottom > oriented.height:
                raise ValueError('Crop must fit inside the oriented image.')
    except (ValueError, OSError) as error:
        raise HTTPException(422, str(error)) from error
    return crop


def create_app(*, runtime: ModelRuntime | None = None, upload_dir: Path | None = None,
               max_bytes: int = 10 * 1024 * 1024, max_pixels: int = 25_000_000,
               cache_size: int = 128) -> FastAPI:
    store = UploadStore(upload_dir or UPLOAD_DIR,
                        max_bytes=max_bytes, max_pixels=max_pixels)
    lock = Lock()
    predictions: OrderedDict = OrderedDict()
    searches: OrderedDict = OrderedDict()

    @asynccontextmanager
    async def lifespan(application):
        if application.state.runtime is None:
            application.state.runtime = await run_in_threadpool(load_runtime)
        yield

    application = FastAPI(title='Fashion Intelligence local models', lifespan=lifespan)
    application.state.runtime = runtime

    def current():
        if application.state.runtime is None:
            raise HTTPException(503, 'Models are still loading.')
        return application.state.runtime

    def remember(cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > cache_size:
            cache.popitem(last=False)

    @application.get('/api/health')
    def health():
        models = current()
        info = models.info
        return {'status': 'ready' if all(x['status'] == 'ready' for x in info.values())
                else 'degraded', 'version': models.version, 'models': info}

    @application.get('/api/metadata')
    def metadata():
        models = current()
        return {'models': models.info, 'version': models.version,
                'evaluation': models.evaluation}

    def analyze_bytes(data, content_type):
        with lock:
            started = perf_counter()
            image_id, path = store.save(data, content_type)
            models = current()
            result = {}
            for target in CLASSIFIERS:
                key = (models.version, image_id, target)
                if key in predictions:
                    predictions.move_to_end(key)
                    result[target] = {**predictions[key], 'cached': True}
                    continue
                tick = perf_counter()
                adapter = models.adapters.get(target)
                value = {'label': None, 'probabilities': {}, 'latency_ms': 0,
                         'run_id': adapter.run_id if adapter else '',
                         'error': None, 'cached': False}
                try:
                    if adapter is None:
                        raise RuntimeError(models.errors.get(target, 'Model unavailable.'))
                    value.update(adapter.predictor(path))
                except Exception as error:
                    value.update(label=None, probabilities={}, error=str(error))
                value['latency_ms'] = (perf_counter() - tick) * 1000
                if value['error'] is None:
                    remember(predictions, key, value)
                result[target] = value
            return {'image_id': image_id, 'version': models.version, 'predictions': result,
                    'elapsed_ms': (perf_counter() - started) * 1000}

    @application.post('/api/analyze')
    async def analyze(request: Request):
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > max_bytes:
                raise HTTPException(413, 'Image must be at most 10 MiB.')
            body.extend(chunk)
        return await run_in_threadpool(analyze_bytes, bytes(body),
                                       request.headers.get('content-type', ''))

    @application.get('/api/images/{image_id}/similar')
    def similar(image_id: str, limit: int = Query(default=5),
                crop_left: CropCoordinate = None, crop_top: CropCoordinate = None,
                crop_right: CropCoordinate = None, crop_bottom: CropCoordinate = None):
        if limit not in (5, 10):
            raise HTTPException(422, 'Match count must be 5 or 10.')
        with lock:
            path = store.get(image_id)
            crop = checked_crop(path, (crop_left, crop_top, crop_right, crop_bottom))
            models = current()
            crop_key = None if crop is None else (crop.left, crop.top, crop.right, crop.bottom)
            key = (models.version, image_id, limit, crop_key)
            if key in searches:
                searches.move_to_end(key)
                return {**searches[key], 'cached': True}
            tick = perf_counter()
            try:
                items = models.similar(path, limit, crop=crop)
            except Exception as error:
                raise HTTPException(503, str(error)) from error
            value = {'image_id': image_id, 'version': models.version, 'items': items,
                     'crop': None if crop is None else crop.to_dict(),
                     'latency_ms': (perf_counter() - tick) * 1000,
                     'gallery_size': models.gallery_size,
                     'method': 'R5 · cosine distance', 'cached': False}
            remember(searches, key, value)
            return value

    @application.get('/api/gallery/{product_id}/image')
    def gallery_image(product_id: int):
        try:
            data = current().gallery_image(product_id)
        except KeyError as error:
            raise HTTPException(404, 'Unknown gallery product.') from error
        except (OSError, ValueError, RuntimeError) as error:
            raise HTTPException(503, 'Gallery image failed its integrity check.') from error
        return Response(data, media_type='image/jpeg',
                        headers={'Cache-Control': 'no-cache'})

    return application


app = create_app()
