from pathlib import Path

import pytest

from r_elegans.data import (
    DATA_ROOT_ENV,
    DataRootNotConfigured,
    data_path,
    get_data_root,
    initialize_data_root,
    sha256_file,
    verify_sha256,
)
from r_elegans.data.paths import DATA_DIRECTORIES, repository_root


def test_data_root_requires_explicit_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)

    with pytest.raises(DataRootNotConfigured):
        get_data_root()


def test_external_data_layout_is_initialized(tmp_path: Path) -> None:
    root = initialize_data_root(tmp_path / "r-elegans-data")

    assert all((root / relative).is_dir() for relative in DATA_DIRECTORIES)


def test_repository_local_data_root_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside the Git repository"):
        get_data_root(repository_root() / "data", must_exist=False)


def test_data_path_cannot_escape_root(tmp_path: Path) -> None:
    root = initialize_data_root(tmp_path / "r-elegans-data")

    with pytest.raises(ValueError, match="beneath the data root"):
        data_path("..", "elsewhere", root=root, must_exist=False)


def test_sha256_verification(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"r-elegans")
    expected = "0913d42e6cec16c97e00fd5f02ee22b2e8cd77296619d200815556375eeaa90c"

    assert sha256_file(artifact) == expected
    verify_sha256(artifact, expected.upper())
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256(artifact, "0" * 64)
