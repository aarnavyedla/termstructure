import shutil
from collections.abc import Generator
from pathlib import Path

import pytest

_SVENSSON_PARAMS = Path("data/processed/svensson_params.parquet")


@pytest.fixture
def backup_svensson_params(tmp_path: Path) -> Generator[None, None, None]:
    """Save svensson_params.parquet before test and restore it afterward."""
    backup = tmp_path / "svensson_params.parquet"
    if _SVENSSON_PARAMS.exists():
        shutil.copy(_SVENSSON_PARAMS, backup)
    yield
    if backup.exists():
        shutil.copy(backup, _SVENSSON_PARAMS)
