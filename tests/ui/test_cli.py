"""Behavior tests for the ``patch-sim`` command line.

The Reflex server is never started: ``subprocess.Popen`` is replaced with a
recorder, so the tests check what the launcher would run and where.
"""

import socket
import subprocess
from pathlib import Path
from typing import Any

import pytest

from patch_sim_ui import cli, launcher


class _FakeServer:
    """Stand-in for the Reflex child process that records how it was started."""

    calls: list[dict[str, Any]] = []
    exit_code = 0

    def __init__(self, cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
        """Record the launch arguments.

        Args:
            cmd: Command the launcher would run.
            cwd: Working directory for the command.
            env: Environment for the command.
        """
        _FakeServer.calls.append({"cmd": cmd, "cwd": Path(cwd), "env": env})

    def poll(self) -> int:
        """Report the process as already exited.

        Returns:
            The configured exit code.
        """
        return _FakeServer.exit_code

    def wait(self, timeout: float | None = None) -> int:
        """Return immediately, as if the server stopped.

        Args:
            timeout: Ignored.

        Returns:
            The configured exit code.
        """
        return _FakeServer.exit_code


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> type[_FakeServer]:
    """Replace the Reflex child process and the browser with recorders.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        The fake server class, whose ``calls`` list records each launch.
    """
    _FakeServer.calls = []
    _FakeServer.exit_code = 0
    monkeypatch.setattr(launcher.subprocess, "Popen", _FakeServer)
    monkeypatch.setattr(launcher.webbrowser, "open", _fail_browser_open)
    monkeypatch.delenv(cli.WORKDIR_ENV_VAR, raising=False)
    return _FakeServer


def _fail_browser_open(url: str) -> bool:
    """Fail the test if the launcher tries to open a browser.

    Args:
        url: URL the launcher tried to open.

    Raises:
        AssertionError: Always.
    """
    raise AssertionError(f"browser opened for {url}")


def _free_port() -> int:
    """Return a TCP port on 127.0.0.1 that is currently free.

    Returns:
        The port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_no_command_prints_usage_and_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Running ``patch-sim`` with no subcommand shows usage and exits with 2."""
    assert cli.main([]) == 2
    assert "usage: patch-sim" in capsys.readouterr().err


def test_version_flag_prints_the_package_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``patch-sim --version`` prints the version and exits successfully."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])

    assert excinfo.value.code == 0
    assert capsys.readouterr().out.startswith("patch-sim ")


def test_ui_defaults_to_localhost_port_3000() -> None:
    """``patch-sim ui`` binds to this computer only, on port 3000, by default."""
    args = cli.build_parser().parse_args(["ui"])

    assert (args.host, args.port) == ("127.0.0.1", 3000)
    assert not args.no_browser
    assert not args.reset


def test_ui_without_extra_prints_install_hint(
    fake_server: type[_FakeServer],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Without the ui extra, the command explains how to install it."""
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)

    assert cli.main(["ui", "--workdir", str(tmp_path)]) == 1

    err = capsys.readouterr().err
    assert err.strip() == cli.UI_EXTRA_MESSAGE
    assert "Traceback" not in err
    assert fake_server.calls == []


def test_ui_runs_reflex_in_production_mode_from_the_workdir(
    fake_server: type[_FakeServer], tmp_path: Path
) -> None:
    """The server runs single-port prod mode in the prepared working directory."""
    port = _free_port()

    code = cli.main(
        ["ui", "--no-browser", "--workdir", str(tmp_path), "--port", str(port)]
    )

    assert code == 0
    (call,) = fake_server.calls
    assert call["cwd"] == tmp_path.resolve()
    assert call["cmd"][1:4] == ["-m", "reflex", "run"]
    cmd = " ".join(call["cmd"])
    assert f"--env prod --single-port --frontend-port {port}" in cmd
    assert "--backend-host 127.0.0.1" in cmd
    assert call["env"]["REFLEX_TELEMETRY_ENABLED"] == "false"
    assert (tmp_path / "rxconfig.py").read_text() == launcher.RXCONFIG_STUB


def test_ui_returns_the_server_exit_code(
    fake_server: type[_FakeServer], tmp_path: Path
) -> None:
    """A failing server's exit code is passed through to the shell."""
    fake_server.exit_code = 3

    code = cli.main(
        ["ui", "--no-browser", "--workdir", str(tmp_path), "--port", str(_free_port())]
    )

    assert code == 3


def test_ui_reads_workdir_from_environment(
    fake_server: type[_FakeServer],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``PATCH_SIM_WORKDIR`` selects the working directory when no flag is given."""
    monkeypatch.setenv(cli.WORKDIR_ENV_VAR, str(tmp_path))

    cli.main(["ui", "--no-browser", "--port", str(_free_port())])

    assert fake_server.calls[0]["cwd"] == tmp_path.resolve()


def test_ui_refuses_a_port_in_use(
    fake_server: type[_FakeServer],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """A busy port is reported with a hint, and the server is not started."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]

        code = cli.main(["ui", "--workdir", str(tmp_path), "--port", str(port)])

    assert code == 1
    err = capsys.readouterr().err
    assert f"Port {port}" in err
    assert "--port" in err
    assert "Traceback" not in err
    assert fake_server.calls == []


def test_ui_reports_an_unwritable_workdir(
    fake_server: type[_FakeServer],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """A working directory that cannot be created gives a readable error."""
    blocker = tmp_path / "a-file"
    blocker.write_text("")

    code = cli.main(
        [
            "ui",
            "--no-browser",
            "--workdir",
            str(blocker / "work"),
            "--port",
            str(_free_port()),
        ]
    )

    assert code == 1
    assert "Cannot prepare the working directory" in capsys.readouterr().err
    assert fake_server.calls == []


def test_ui_opens_the_browser_once_the_server_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_python_dash_m_runs_the_same_command() -> None:
    """``python -m patch_sim_ui --help`` prints the patch-sim usage."""
    result = subprocess.run(
        [cli.sys.executable, "-m", "patch_sim_ui", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage: patch-sim" in result.stdout
