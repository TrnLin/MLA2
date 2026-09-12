"""Check the public deployment with one known fashion photo."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url')
    parser.add_argument('image', type=Path)
    args = parser.parse_args()
    base = args.url.rstrip('/')

    def get(path):
        with urllib.request.urlopen(base + path, timeout=60) as response:
            return response.read()

    assert json.loads(get('/readyz'))['status'] == 'ready'
    for path in ('/', '/demo'):
        assert b'<html' in get(path)
    raw = args.image.read_bytes()
    request = urllib.request.Request(base + '/api/analyze', data=raw,
                                     headers={'Content-Type': 'image/jpeg'})
    with urllib.request.urlopen(request, timeout=60) as response:
        analysis = json.load(response)
    assert analysis['image_id'] == hashlib.sha256(raw).hexdigest()
    assert len(analysis['predictions']) == 4
    assert all(p['label'] and p['error'] is None for p in analysis['predictions'].values())
    path = f"/api/images/{analysis['image_id']}/similar"
    search = json.loads(get(path + '?limit=5'))
    crop = json.loads(get(path + '?limit=5&crop_left=6&crop_top=8&crop_right=54&crop_bottom=72'))
    assert len(search['items']) == len(crop['items']) == 5
    assert crop['crop'] == dict(left=6, top=8, right=54, bottom=72)
    photo = get(f"/api/gallery/{search['items'][0]['id']}/image")
    assert photo[:2] == b'\xff\xd8'
    print(json.dumps({'predictions': analysis['predictions'],
                      'search_ids': [item['id'] for item in search['items']],
                      'crop_ids': [item['id'] for item in crop['items']],
                      'gallery_bytes': len(photo)}, indent=2))


if __name__ == '__main__':
    main()
