"""Behavior tests for the ``neuroclamp ui`` working-directory launcher."""

import http.server
import logging
import os
import socket
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from neuroclamp_ui import launcher


def test_prepare_workdir_writes_the_config_stub(tmp_path: Path) -> None:
    """A fresh working directory gets the rxconfig stub and the marker."""
    workdir = tmp_path / "work"

    launcher.prepare_workdir(workdir)

    assert (workdir / "rxconfig.py").read_text() == launcher.RXCONFIG_STUB
    assert (workdir / launcher.WORKDIR_MARKER).is_file()


def test_prepare_workdir_leaves_unchanged_config_untouched(tmp_path: Path) -> None:
    """An up-to-date rxconfig.py is not rewritten, so Reflex sees no change."""
    workdir = tmp_path / "work"
    launcher.prepare_workdir(workdir)
    rxconfig = workdir / "rxconfig.py"
    os.utime(rxconfig, ns=(1_000_000_000, 1_000_000_000))

    launcher.prepare_workdir(workdir)

    assert rxconfig.stat().st_mtime_ns == 1_000_000_000


def test_prepare_workdir_rewrites_edited_config(tmp_path: Path) -> None:
    """A modified rxconfig.py is restored to the stub."""
    workdir = tmp_path / "work"
    launcher.prepare_workdir(workdir)
    (workdir / "rxconfig.py").write_text("config = None\n")

    launcher.prepare_workdir(workdir)

    assert (workdir / "rxconfig.py").read_text() == launcher.RXCONFIG_STUB


def test_prepare_workdir_keeps_frontend_build_unless_reset(tmp_path: Path) -> None:
    """The .web build survives a normal restart and is deleted by reset."""
    workdir = tmp_path / "work"
    launcher.prepare_workdir(workdir)
    marker = workdir / ".web" / "marker"
    marker.parent.mkdir()
    marker.write_text("built")

    launcher.prepare_workdir(workdir)
    assert marker.is_file()

    launcher.prepare_workdir(workdir, reset=True)
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
        with pytest.raises(launcher.LauncherError, match="not a neuroclamp"):
            launcher.prepare_workdir(project, reset=reset)

    assert (project / "assets" / "logo.png").is_file()
    assert (project / "rxconfig.py").read_text() == "config = 'mine'\n"


def test_prepare_workdir_accepts_an_existing_empty_directory(tmp_path: Path) -> None:
    """An empty directory is adopted and marked as a working directory."""
    launcher.prepare_workdir(tmp_path)

    assert (tmp_path / launcher.WORKDIR_MARKER).is_file()
    assert (tmp_path / "rxconfig.py").is_file()


def test_prepare_workdir_reports_unwritable_location(tmp_path: Path) -> None:
    """A working directory that cannot be created raises a readable error."""
    blocker = tmp_path / "a-file"
    blocker.write_text("")

    with pytest.raises(launcher.LauncherError, match="--workdir"):
        launcher.prepare_workdir(blocker / "work")


def test_generated_config_loads_the_neuroclamp_ui_app(tmp_path: Path) -> None:
    """Reflex reads the stub as a config for the installed neuroclamp_ui app."""
    pytest.importorskip("reflex")
    from reflex_base.config import _get_config

    launcher.prepare_workdir(tmp_path)
    config = _get_config(tmp_path)

    assert config.app_name == "neuroclamp_ui"
    assert config.module == "neuroclamp_ui.neuroclamp_ui"


def test_default_workdir_is_keyed_by_version() -> None:
    """Each package version gets its own working directory."""
    pytest.importorskip("platformdirs")

    first = launcher.default_workdir("1.2.3")
    second = launcher.default_workdir("1.2.4")

    assert first.name == "1.2.3"
    assert first.parent == second.parent
    assert "neuroclamp" in first.parts


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


class _PingHandler(http.server.BaseHTTPRequestHandler):
    """Answer the health check the way the Reflex backend does."""

    def do_GET(self) -> None:
        """Reply 200 to ``/ping`` and 404 to anything else."""
        self.send_response(200 if self.path == "/ping" else 404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        """Keep request logging out of the test output.

        Args:
            format: Log format string.
            *args: Values for the format string.
        """


@pytest.fixture
def live_server() -> Iterator[str]:
    """Serve ``/ping`` on a local port, like a running app.

    Yields:
        The server's base URL.
    """
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _PingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def _closed_url() -> str:
    """Return the URL of a local port with nothing listening on it.

    Returns:
        A base URL whose health check will fail.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


def test_browser_opens_once_the_server_answers(
    live_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The app opens in the browser when the server answers its health check.

    Args:
        live_server: Base URL of a server that answers ``/ping``.
        monkeypatch: Pytest monkeypatch fixture.
    """
    opened: list[str] = []
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)

    thread = launcher.open_browser_when_ready(live_server, lambda: True)
    thread.join(timeout=10)

    assert opened == [live_server]


def test_browser_wait_stops_when_the_server_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No browser opens if the server process exits before it is ready.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    opened: list[str] = []
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)

    thread = launcher.open_browser_when_ready(_closed_url(), lambda: False)
    thread.join(timeout=10)

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
    opened: list[str] = []
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    url = _closed_url()

    thread = launcher.open_browser_when_ready(url, lambda: True, timeout=0.5)
    thread.join(timeout=10)

    assert opened == []
    # Count only warnings: the launcher also logs at DEBUG, which caplog
    # captures whenever an earlier test lowered the logger's level.
    (record,) = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert record.levelname == "WARNING"
    assert url in record.getMessage()
