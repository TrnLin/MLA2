"""Build separate code and fixed-artifact archives without copying the dataset."""
import argparse
import csv
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRANCH = 'feat/demo-model-api'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_archive(destination, files, release):
    with tarfile.open(destination, 'x:gz', dereference=True) as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name, recursive=False)
        payload = json.dumps(release, indent=2).encode()
        entry = tarfile.TarInfo('release.json')
        entry.size = len(payload)
        entry.mode = 0o644
        archive.addfile(entry, io.BytesIO(payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code-only', action='store_true',
                        help='Reuse the already uploaded fixed artifacts.')
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    expected = subprocess.check_output(
        ['git', 'rev-parse', f'origin/{BRANCH}'], cwd=ROOT, text=True).strip()
    if commit != expected:
        raise SystemExit(f'Checkout must match fetched origin/{BRANCH}.')
    code, artifacts = {}, {}
    for line in (ROOT / 'HOSTING_RUNTIME_FILES.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        relative = Path(line)
        if relative.is_absolute() or '..' in relative.parts:
            raise SystemExit(f'Unsafe runtime path: {line}')
        is_artifact = line.startswith('model-weight/') or relative.suffix == '.pt'
        source = (args.artifact_root if is_artifact else ROOT / 'core') / relative
        if not source.is_file():
            raise SystemExit(f'Missing file: {source}')
        (artifacts if is_artifact else code)[f'core/{line}'] = source
    for folder in ('core/src', 'be/src', 'fe', 'deploy/oracle'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and not any(part in {'node_modules', 'dist', '__pycache__',
                                                   '.git'} for part in path.parts):
                code[str(path.relative_to(ROOT))] = path
    for name in ('core/pyproject.toml', 'be/pyproject.toml'):
        code[name] = ROOT / name
    with (ROOT / 'core/models/task4_teacher_gallery/metadata.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 26217:
        raise SystemExit('Expected the fixed 26,217-photo gallery.')
    for row in rows:
        name = f"data/train/images_train/{int(row['id'])}.jpg"
        source = args.artifact_root / name
        if sha256(source) != row['sha256']:
            raise SystemExit(f'Gallery hash mismatch: {name}')
        artifacts[f'core/{name}'] = source
    release = {'branch': BRANCH, 'base_commit': commit,
               'code_files': {name: sha256(path) for name, path in sorted(code.items())},
               'gallery_count': len(rows)}
    args.output.mkdir(parents=True, exist_ok=True)
    archives = [('code.tar.gz', code)]
    if not args.code_only:
        archives.append(('artifacts.tar.gz', artifacts))
    for name, files in archives:
        destination = args.output / name
        write_archive(destination, files, release)
        print(f'{name}: {destination.stat().st_size:,} bytes; sha256 {sha256(destination)}')
    (args.output / 'release.json').write_text(json.dumps(release, indent=2) + '\n')
    (args.output / 'SHA256SUMS').write_text(''.join(
        f'{sha256(args.output / name)}  {name}\n' for name, _ in archives))


if __name__ == '__main__':
    main()
