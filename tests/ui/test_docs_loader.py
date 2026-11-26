"""Behavior tests for the help-page content loader and its screenshots."""

import os
import re
from pathlib import Path

import pytest

from neuroclamp_ui import docs_loader


def _shipped_names() -> list[str]:
    """Return the names of the screenshots shipped with the package.

    Returns:
        Sorted file names.
    """
    return sorted(path.name for path in docs_loader.SCREENSHOTS_DIR.glob("*.png"))


def test_every_help_image_ships_with_the_package() -> None:
    """Each ``/screenshots/...`` image in the help pages exists in the package."""
    referenced = {
        name
        for text in docs_loader.PROSE.values()
        for name in re.findall(r"\]\(/screenshots/([^)\s]+)\)", text)
    }

    assert referenced
    assert referenced <= set(_shipped_names())


def test_install_screenshots_copies_them_into_assets(tmp_path: Path) -> None:
    """A fresh app directory gets every screenshot under ``assets/screenshots``."""
    docs_loader.install_screenshots(tmp_path / "assets")

    installed = tmp_path / "assets" / "screenshots"
    assert sorted(path.name for path in installed.iterdir()) == _shipped_names()
    name = _shipped_names()[0]
    assert (installed / name).read_bytes() == (
        docs_loader.SCREENSHOTS_DIR / name
    ).read_bytes()


def test_install_screenshots_leaves_unchanged_files_alone(tmp_path: Path) -> None:
    """An up-to-date screenshot is not rewritten, so file watchers see no change."""
    docs_loader.install_screenshots(tmp_path)
    copy = tmp_path / "screenshots" / _shipped_names()[0]
    os.utime(copy, ns=(1_000_000_000, 1_000_000_000))

    docs_loader.install_screenshots(tmp_path)

    assert copy.stat().st_mtime_ns == 1_000_000_000


def test_install_screenshots_restores_a_modified_file(tmp_path: Path) -> None:
    """A screenshot that differs from the package copy is replaced."""
    docs_loader.install_screenshots(tmp_path)
    name = _shipped_names()[0]
    (tmp_path / "screenshots" / name).write_bytes(b"stale")

    docs_loader.install_screenshots(tmp_path)

    assert (tmp_path / "screenshots" / name).read_bytes() == (
        docs_loader.SCREENSHOTS_DIR / name
    ).read_bytes()


def test_install_screenshots_removes_ones_no_longer_shipped(tmp_path: Path) -> None:
    """Screenshots dropped from the package do not linger in the assets."""
    installed = tmp_path / "screenshots"
    installed.mkdir()
    (installed / "retired.png").write_bytes(b"png")

    docs_loader.install_screenshots(tmp_path)

    assert sorted(path.name for path in installed.iterdir()) == _shipped_names()


def test_install_screenshots_warns_instead_of_failing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unwritable assets directory is logged and does not stop the app.

    Args:
        tmp_path: Pytest temporary directory.
        caplog: Pytest log-capture fixture.
    """
    blocker = tmp_path / "a-file"
    blocker.write_text("")

    docs_loader.install_screenshots(blocker / "assets")

    (record,) = caplog.records
    assert record.levelname == "WARNING"
    assert "help screenshots" in record.getMessage()
