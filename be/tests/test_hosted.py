from types import SimpleNamespace

from fashion_api.hosted import create_hosted_app
from fastapi.testclient import TestClient


def test_hosted_routes_and_readiness(tmp_path):
    static = tmp_path / 'dist'
    static.mkdir()
    (static / 'index.html').write_text('<html>Fashion demo</html>')
    (static / 'asset.js').write_text('const ready = true;')
    runtime = SimpleNamespace(info={'model': {'status': 'ready'}}, version='test')
    app = create_hosted_app(runtime=runtime, upload_dir=tmp_path / 'uploads', static_dir=static)
    with TestClient(app) as client:
        for route in ('/', '/demo'):
            response = client.get(route)
            assert response.status_code == 200
            assert response.text == '<html>Fashion demo</html>'
        assert client.get('/asset.js').status_code == 200
        assert client.get('/api/missing').status_code == 404
        assert client.get('/api/missing').json()['detail'] == 'Unknown API route.'
        assert client.get('/readyz').status_code == 200
        runtime.info['model']['status'] = 'error'
        assert client.get('/readyz').status_code == 503
