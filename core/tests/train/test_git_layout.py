"""Model source hashes and evaluation gates survive a nested project root."""
import subprocess
from uuid import uuid4

import pytest

from fashion.config import ROOT
from fashion.train.cache import (
    implementation_sha256,
    implementation_sha256_at_commit,
    verify_implementation_at_head,
)

def test_source_hashes_resolve_before_and_after_core_move():
    # Retain this tiny test repository; no automatic fixture cleanup.
    checkout = ROOT / 'tmp/layout-tests' / uuid4().hex
    (checkout / 'src').mkdir(parents=True)
    (checkout / 'src/model.py').write_text('VALUE = 1\n')

    def git(*args):
        return subprocess.check_output(
            ['git', '-c', 'user.name=Layout Test', '-c', 'user.email=layout@example.invalid',
             *args], cwd=checkout, text=True).strip()

    git('init', '-q')
    git('add', 'src/model.py')
    git('commit', '-q', '-m', 'Original model layout')
    old_commit = git('rev-parse', 'HEAD')
    before = implementation_sha256('src/model.py', root=checkout)
    model_root = checkout / 'core'
    model_root.mkdir()
    (checkout / 'src').rename(model_root / 'src')
    git('add', '-A')
    git('commit', '-q', '-m', 'Group model work in core')
    new_commit = git('rev-parse', 'HEAD')
    assert implementation_sha256('src/model.py', root=model_root) == before
    assert implementation_sha256_at_commit(
        'src/model.py', commit=old_commit, root=model_root) == before
    assert implementation_sha256_at_commit(
        'src/model.py', commit=new_commit, root=model_root) == before
    assert verify_implementation_at_head('src/model.py', root=model_root) == ('src/model.py',)


@pytest.mark.parametrize('outside_change', [False, True])
def test_nested_scoring_gate_keeps_outside_core_changes_visible(outside_change):
    from fashion.task4_evaluation.score import _verify_post_blind_git_state

    checkout = ROOT / 'tmp/layout-tests' / uuid4().hex
    model_root = checkout / 'core'
    model_root.mkdir(parents=True)
    (model_root / 'source.py').write_text('VALUE = 1\n')

    def git(*args):
        return subprocess.check_output(
            ['git', '-c', 'user.name=Layout Test', '-c', 'user.email=layout@example.invalid',
             *args], cwd=checkout, text=True).strip()

    git('init', '-q')
    git('add', 'core/source.py')
    git('commit', '-q', '-m', 'Synthetic source')
    source_commit = git('rev-parse', 'HEAD')
    (model_root / 'results').mkdir()
    (model_root / 'results/frozen.csv').write_text('id\n1\n')
    receipt_path = model_root / 'results/receipt.json'
    receipt_path.write_text('{}\n')
    if outside_change:
        (checkout / 'fe').mkdir()
        (checkout / 'fe/change.js').write_text('const changed = true;\n')
    git('add', '.')
    git('commit', '-q', '-m', 'Synthetic evidence')
    git('tag', 'task4-holdout-scoring-approved-v1')
    receipt = {'git': {'commit': source_commit},
               'artifacts': {'rankings': {'path': 'results/frozen.csv'}}}
    if outside_change:
        with pytest.raises(RuntimeError, match='fe/change.js'):
            _verify_post_blind_git_state(root=model_root, receipt=receipt,
                                        receipt_path=receipt_path)
    else:
        result = _verify_post_blind_git_state(root=model_root, receipt=receipt,
                                             receipt_path=receipt_path)
        assert result[2] == ['core/results/frozen.csv', 'core/results/receipt.json']
