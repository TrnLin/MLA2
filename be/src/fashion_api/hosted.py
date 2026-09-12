"""Serve the built website and model API from one origin."""
import os
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fashion_api.api import create_app


def create_hosted_app(*, runtime=None, upload_dir=None, static_dir=None):
    application = create_app(runtime=runtime, upload_dir=upload_dir)
    directory = Path(static_dir or os.environ.get('FASHION_STATIC_DIR', '/app/fe/dist'))
    if not (directory / 'index.html').is_file():
        raise RuntimeError('Build the frontend before starting the hosted app.')

    @application.get('/readyz', include_in_schema=False)
    def ready():
        models = application.state.runtime
        if models is None or any(item['status'] != 'ready' for item in models.info.values()):
            raise HTTPException(503, 'Models are not ready.')
        return {'status': 'ready', 'version': models.version}

    @application.get('/', include_in_schema=False)
    @application.get('/demo', include_in_schema=False)
    def website():
        return FileResponse(directory / 'index.html', headers={'Cache-Control': 'no-cache'})

    # Reserve API paths, including ones that do not exist, before the file mount.
    @application.api_route('/api/{path:path}', methods=['GET', 'POST'], include_in_schema=False)
    def unknown_api(path: str):
        raise HTTPException(404, 'Unknown API route.')

    application.mount('/', StaticFiles(directory=directory), name='website')
    return application
