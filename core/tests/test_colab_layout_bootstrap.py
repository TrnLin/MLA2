"""Exercise the real checkout setup without network, training, or filesystem writes."""
import ast
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    path for path in (ROOT / 'notebooks/task3_training').glob('*.ipynb')
    if '"git", "clone"' in '\n'.join(
        ''.join(cell['source']) for cell in json.loads(path.read_text())['cells']
        if cell['cell_type'] == 'code'
    )
]


@pytest.mark.parametrize('notebook', NOTEBOOKS, ids=lambda path: path.stem)
@pytest.mark.parametrize('core_layout', [False, True], ids=['legacy', 'core'])
@pytest.mark.parametrize('existing', [False, True], ids=['fresh', 'existing'])
def test_checkout_setup_selects_model_root(notebook, core_layout, existing):
    dirs = set()
    calls = []
    commit = 'a' * 40

    class VirtualPath(PurePosixPath):
        def is_dir(self):
            return str(self) in dirs

        def exists(self):
            return self.is_dir()

        def rename(self, destination):
            old = str(self)
            dirs.update(str(destination) + name[len(old):] for name in tuple(dirs)
                        if name == old or name.startswith(old + '/'))
            return destination

    def install_checkout(path):
        dirs.update([str(path), str(path / '.git')])
        model_root = path / 'core' if core_layout else path
        dirs.update([str(model_root), str(model_root / 'src/fashion')])

    def run(command, **kwargs):
        command = [str(part) for part in command]
        calls.append((command, kwargs))
        if 'clone' in command:
            install_checkout(VirtualPath(command[-1]))
        elif 'merge' in command:
            location = command[command.index('-C') + 1] if '-C' in command else kwargs['cwd']
            install_checkout(VirtualPath(location))
        return SimpleNamespace(returncode=0)

    def output(command, **kwargs):
        calls.append(([str(part) for part in command], kwargs))
        if 'remote' in command:
            return 'https://github.com/TrnLin/MLA2.git\n'
        if 'status' in command:
            return ''
        return commit + '\n'

    class TemporaryCheckout:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return '/staging'

        def __exit__(self, *args):
            pass

    namespace = {
        'Path': VirtualPath,
        'subprocess': SimpleNamespace(run=run, check_output=output),
        'tempfile': SimpleNamespace(TemporaryDirectory=TemporaryCheckout),
        'run_checked': run,
        'BUNDLE_ROOT': VirtualPath('/bundle'),
        'CODE_COMMIT': commit,
    }
    code = '\n'.join(
        ''.join(cell['source']) for cell in json.loads(notebook.read_text())['cells']
        if cell['cell_type'] == 'code'
    )
    # Select complete checkout statements while excluding Drive/archive/training work.
    selected = []
    for node in ast.parse(code).body:
        source = ast.unparse(node)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {
                'REPO_URL', 'REPOSITORY', 'BRANCH', 'CHECKOUT_DIR', 'REPO_DIR', 'LOCAL_REGISTRY'
            }:
                if name == 'REPO_DIR' and 'bundle_digest' in source:
                    continue
                if name == 'REPO_DIR' and 'find_repo_root' in source:
                    continue
                selected.append(node)
        elif isinstance(node, ast.If) and '.git' in ast.unparse(node.test):
            selected.append(node)
        elif isinstance(node, ast.With) and "'git', 'clone'" in source:
            selected.append(node)
        if selected and isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == 'LOCAL_REGISTRY' and any(
                isinstance(n, (ast.If, ast.With)) for n in selected
            ):
                break
    for candidate in [VirtualPath('/content/MLA2'),
                      VirtualPath('/bundle-code-' + commit[:12]),
                      VirtualPath('/content/MLA2-usage-v3-' + commit[:12])]:
        if existing:
            install_checkout(candidate)
    setup = compile(ast.Module(body=selected, type_ignores=[]), str(notebook), 'exec')
    exec(setup, namespace)
    checkout = namespace.get('CHECKOUT_DIR', namespace['REPO_DIR'])
    expected = checkout / 'core' if core_layout else checkout
    assert namespace['REPO_DIR'] == expected
    if 'LOCAL_REGISTRY' in namespace:
        assert namespace['LOCAL_REGISTRY'].is_relative_to(expected)
    for command, kwargs in calls:
        if 'clone' not in command and ('-C' in command or kwargs.get('cwd') is not None):
            location = command[command.index('-C') + 1] if '-C' in command else kwargs['cwd']
            assert not str(location).endswith('/core'), command
