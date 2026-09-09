"""Data archives cannot replace GitHub code or escape the local training checkout."""

import stat
from zipfile import ZipFile, ZipInfo

import pytest

from fashion.train.task3_usage_v3_setup import copy_archive, extract_data


def test_archive_installs_only_allowed_data_and_reuses_the_local_copy(tmp_path):
    archive = tmp_path / "existing.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("data/images/one.png", b"fixture image")
        bundle.writestr("src/fashion/train/model.py", b"stale bundled code")
    cached, digest = copy_archive(archive, tmp_path / "cache")
    assert len(digest) == 64
    assert copy_archive(archive, tmp_path / "cache") == (cached, digest)
    root = tmp_path / "checkout"
    assert extract_data(cached, root, prefixes=("data/images/",)) == 1
    assert (root / "data/images/one.png").read_bytes() == b"fixture image"
    assert not (root / "src").exists()
    assert extract_data(cached, root, prefixes=("data/images/",)) == 1


@pytest.mark.parametrize("name", ["../escape.png", "/tmp/escape.png", "data\\escape.png"])
def test_unsafe_member_fails_before_any_extraction(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("data/good.png", b"good")
        bundle.writestr(name, b"bad")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        extract_data(archive, tmp_path / "checkout", prefixes=("data/",))
    assert not (tmp_path / "checkout/data/good.png").exists()


def test_symlink_member_is_rejected(tmp_path):
    archive = tmp_path / "link.zip"
    entry = ZipInfo("data/link")
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(archive, "w") as bundle:
        bundle.writestr(entry, "../../escape")
    with pytest.raises(ValueError, match="symlink"):
        extract_data(archive, tmp_path / "checkout", prefixes=("data/",))
