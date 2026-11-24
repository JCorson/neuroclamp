"""Behavior tests for the ``patch-sim ui`` working-directory launcher."""

import os
import socket
from pathlib import Path

import pytest

from patch_sim_ui import launcher


def _make_screenshots(root: Path, names: list[str]) -> Path:
    """Create a fake screenshots directory holding the named files.

    Args:
        root: Parent directory to create it in.
        names: File names to create.

    Returns:
        The screenshots directory.
    """
    screenshots = root / "screenshots-src"
    screenshots.mkdir(exist_ok=True)
    for existing in screenshots.iterdir():
        existing.unlink()
    for name in names:
        (screenshots / name).write_bytes(b"png")
    return screenshots


def test_prepare_workdir_writes_config_stub_and_screenshots(tmp_path: Path) -> None:
    """A fresh working directory gets the rxconfig stub and the screenshots."""
    screenshots = _make_screenshots(tmp_path, ["overview.png"])
    workdir = tmp_path / "work"

    launcher.prepare_workdir(workdir, screenshots)

    assert (workdir / "rxconfig.py").read_text() == launcher.RXCONFIG_STUB
    assert (workdir / "assets" / "screenshots" / "overview.png").is_file()


def test_prepare_workdir_leaves_unchanged_config_untouched(tmp_path: Path) -> None:
    """An up-to-date rxconfig.py is not rewritten, so Reflex sees no change."""
    workdir = tmp_path / "work"
    launcher.prepare_workdir(workdir, None)
    rxconfig = workdir / "rxconfig.py"
    os.utime(rxconfig, ns=(1_000_000_000, 1_000_000_000))

    launcher.prepare_workdir(workdir, None)

    assert rxconfig.stat().st_mtime_ns == 1_000_000_000


def test_prepare_workdir_rewrites_edited_config(tmp_path: Path) -> None:
    """A modified rxconfig.py is restored to the stub."""
    workdir = tmp_path / "work"
    launcher.prepare_workdir(workdir, None)
    (workdir / "rxconfig.py").write_text("config = None\n")

    launcher.prepare_workdir(workdir, None)

    assert (workdir / "rxconfig.py").read_text() == launcher.RXCONFIG_STUB


def test_prepare_workdir_drops_screenshots_removed_from_the_package(
    tmp_path: Path,
) -> None:
    """Screenshots no longer shipped do not linger in the working directory."""
    workdir = tmp_path / "work"
    launcher.prepare_workdir(
        workdir, _make_screenshots(tmp_path, ["keep.png", "old.png"])
    )

    launcher.prepare_workdir(workdir, _make_screenshots(tmp_path, ["keep.png"]))

    shipped = workdir / "assets" / "screenshots"
    assert sorted(p.name for p in shipped.iterdir()) == ["keep.png"]


def test_prepare_workdir_keeps_frontend_build_unless_reset(tmp_path: Path) -> None:
    """The .web build survives a normal restart and is deleted by reset."""
    workdir = tmp_path / "work"
    launcher.prepare_workdir(workdir, None)
    marker = workdir / ".web" / "marker"
    marker.parent.mkdir()
    marker.write_text("built")

    launcher.prepare_workdir(workdir, None)
    assert marker.is_file()

    launcher.prepare_workdir(workdir, None, reset=True)
    assert not marker.exists()
    assert (workdir / "rxconfig.py").is_file()


def test_prepare_workdir_refuses_a_directory_it_did_not_create(
    tmp_path: Path,
) -> None:
    """A non-empty directory without the marker is left untouched."""
    project = tmp_path / "project"
    (project / "assets").mkdir(parents=True)
    (project / "assets" / "logo.png").write_bytes(b"png")
    (project / "rxconfig.py").write_text("config = 'mine'\n")

    for reset in (False, True):
        with pytest.raises(launcher.LauncherError, match="not a patch-sim"):
            launcher.prepare_workdir(project, None, reset=reset)

    assert (project / "assets" / "logo.png").is_file()
    assert (project / "rxconfig.py").read_text() == "config = 'mine'\n"


def test_prepare_workdir_accepts_an_existing_empty_directory(tmp_path: Path) -> None:
    """An empty directory is adopted and marked as a working directory."""
    launcher.prepare_workdir(tmp_path, None)

    assert (tmp_path / launcher.WORKDIR_MARKER).is_file()
    assert (tmp_path / "rxconfig.py").is_file()


def test_prepare_workdir_reports_unwritable_location(tmp_path: Path) -> None:
    """A working directory that cannot be created raises a readable error."""
    blocker = tmp_path / "a-file"
    blocker.write_text("")

    with pytest.raises(launcher.LauncherError, match="--workdir"):
        launcher.prepare_workdir(blocker / "work", None)


def test_generated_config_loads_the_patch_sim_ui_app(tmp_path: Path) -> None:
    """Reflex reads the stub as a config for the installed patch_sim_ui app."""
    pytest.importorskip("reflex")
    from reflex_base.config import _get_config

    launcher.prepare_workdir(tmp_path, None)
    config = _get_config(tmp_path)

    assert config.app_name == "patch_sim_ui"
    assert config.module == "patch_sim_ui.patch_sim_ui"


def test_find_screenshots_dir_locates_the_help_screenshots() -> None:
    """The help screenshots are found from a checkout or an installed wheel."""
    screenshots = launcher.find_screenshots_dir()

    assert screenshots is not None
    assert (screenshots / "overview.png").is_file()


def test_default_workdir_is_keyed_by_version() -> None:
    """Each package version gets its own working directory."""
    pytest.importorskip("platformdirs")

    first = launcher.default_workdir("1.2.3")
    second = launcher.default_workdir("1.2.4")

    assert first.name == "1.2.3"
    assert first.parent == second.parent
    assert "patch-sim" in first.parts


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", "http://localhost:3000"),
        ("::1", "http://localhost:3000"),
        ("0.0.0.0", "http://localhost:3000"),
        ("::", "http://localhost:3000"),
        ("192.168.1.5", "http://192.168.1.5:3000"),
        ("fe80::1", "http://[fe80::1]:3000"),
    ],
)
def test_browser_url_opens_local_hosts_as_localhost(host: str, expected: str) -> None:
    """Loopback and wildcard hosts open as localhost, matching the API URL.

    The figure and export requests go to ``http://localhost:<port>``, so a page
    opened at ``127.0.0.1`` would make them cross-origin and they would fail.

    Args:
        host: Bind address passed to the launcher.
        expected: URL the browser should open.
    """
    assert launcher.browser_url(host, 3000) == expected


def test_port_in_use_detects_a_listening_socket() -> None:
    """A port held by a listening socket is reported busy, and free once closed."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        assert launcher.port_in_use(port)
    assert not launcher.port_in_use(port)


def test_check_host_rejects_an_invalid_address() -> None:
    """An unusable --host is reported as a host problem, not a busy port."""
    with pytest.raises(launcher.LauncherError, match="--host"):
        launcher.check_host("x" * 64 + ".example")


def test_child_env_disables_telemetry_unless_the_user_sets_it() -> None:
    """Telemetry and update checks default off; explicit user values win."""
    env = launcher.build_child_env({"PATH": "/bin"})
    assert env["REFLEX_TELEMETRY_ENABLED"] == "false"
    assert env["REFLEX_CHECK_LATEST_VERSION"] == "false"
    assert env["PATH"] == "/bin"

    env = launcher.build_child_env({"REFLEX_TELEMETRY_ENABLED": "true"})
    assert env["REFLEX_TELEMETRY_ENABLED"] == "true"


def test_browser_opens_once_the_server_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The browser opens only after the health check succeeds."""
    checks = iter([False, False, True])
    monkeypatch.setattr(launcher, "_server_ready", lambda url: next(checks))
    opened: list[str] = []
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)

    thread = launcher.open_browser_when_ready(
        "http://localhost:3000", lambda: True, poll_interval=0.0
    )
    thread.join(timeout=5)

    assert opened == ["http://localhost:3000"]


def test_browser_wait_stops_when_the_server_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No browser opens if the server process exits before it is ready."""
    monkeypatch.setattr(launcher, "_server_ready", lambda url: False)
    opened: list[str] = []
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)

    thread = launcher.open_browser_when_ready(
        "http://localhost:3000", lambda: False, poll_interval=0.0
    )
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert opened == []


def test_browser_wait_warns_when_the_server_never_answers(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A server that stays up but never answers gets a warning with the URL.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        caplog: Pytest log-capture fixture.
    """
    monkeypatch.setattr(launcher, "_server_ready", lambda url: False)
    monkeypatch.setattr(launcher.webbrowser, "open", _fail_open)

    thread = launcher.open_browser_when_ready(
        "http://localhost:3000", lambda: True, timeout=0.0
    )
    thread.join(timeout=5)

    (record,) = caplog.records
    assert record.levelname == "WARNING"
    assert "http://localhost:3000" in record.getMessage()


def _fail_open(url: str) -> bool:
    """Fail the test if a browser is opened.

    Args:
        url: URL that would have been opened.

    Raises:
        AssertionError: Always.
    """
    raise AssertionError(f"browser opened for {url}")
