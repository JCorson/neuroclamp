"""Behavior tests for the ``patch-sim`` command line.

The Reflex server is never started: ``subprocess.Popen`` is replaced with a
fake process, so the tests check what the launcher would run, where, and how
it shuts the server down.
"""

import os
import signal
import socket
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from patch_sim_ui import cli, launcher


class _FakeServer:
    """Stand-in for the Reflex child process.

    Each launch is recorded in ``calls``. ``wait_effects`` scripts successive
    ``wait()`` calls: an exception instance is raised, a callable is called,
    and once the list is empty ``wait()`` returns ``exit_code``. While effects
    remain the process reports itself as running.
    """

    pid = 4242
    calls: list[dict[str, Any]] = []
    instances: list["_FakeServer"] = []
    exit_code = 0
    wait_effects: list[BaseException | Callable[[], None]] = []

    def __init__(self, cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
        """Record the launch arguments.

        Args:
            cmd: Command the launcher would run.
            cwd: Working directory for the command.
            env: Environment for the command.
        """
        _FakeServer.calls.append({"cmd": cmd, "cwd": Path(cwd), "env": env})
        _FakeServer.instances.append(self)
        self.alive = bool(_FakeServer.wait_effects)
        self.signals: list[str] = []

    def poll(self) -> int | None:
        """Report whether the process has exited.

        Returns:
            ``None`` while running, otherwise the configured exit code.
        """
        return None if self.alive else _FakeServer.exit_code

    def wait(self, timeout: float | None = None) -> int:
        """Apply the next scripted effect, or exit.

        Args:
            timeout: Ignored.

        Returns:
            The configured exit code.
        """
        if _FakeServer.wait_effects:
            effect = _FakeServer.wait_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
            effect()
        self.alive = False
        return _FakeServer.exit_code

    def terminate(self) -> None:
        """Record a SIGTERM sent by the launcher."""
        self.signals.append("terminate")

    def kill(self) -> None:
        """Record a kill sent by the launcher and mark the process exited."""
        self.signals.append("kill")
        self.alive = False


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> type[_FakeServer]:
    """Replace the Reflex child process and the browser with fakes.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        The fake server class, whose ``calls`` list records each launch.
    """
    _FakeServer.calls = []
    _FakeServer.instances = []
    _FakeServer.exit_code = 0
    _FakeServer.wait_effects = []
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
    """Return a TCP port that is currently free.

    Returns:
        The port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_ui(workdir: Path, *extra: str) -> int:
    """Run ``patch-sim ui --no-browser`` on a free port.

    Args:
        workdir: Working directory to pass with ``--workdir``.
        *extra: Additional command-line arguments.

    Returns:
        The command's exit code.
    """
    return cli.main(
        [
            "ui",
            "--no-browser",
            "--workdir",
            str(workdir),
            "--port",
            str(_free_port()),
            *extra,
        ]
    )


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


@pytest.mark.parametrize("value", ["0", "70000", "abc"])
def test_ui_rejects_an_invalid_port(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Out-of-range or non-numeric ports are usage errors, not tracebacks.

    Args:
        value: The ``--port`` value under test.
        capsys: Pytest output-capture fixture.
    """
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["ui", "--port", value])

    assert excinfo.value.code == 2
    assert "1-65535" in capsys.readouterr().err


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
    assert err.strip() == f"Error: {cli.UI_EXTRA_MESSAGE}"
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

    assert _run_ui(tmp_path) == 3


def test_ui_announces_the_frontend_build_only_when_needed(
    fake_server: type[_FakeServer],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The slow-first-launch notice appears without a build, or with --reset."""
    _run_ui(tmp_path)
    assert "First launch builds the web frontend" in capsys.readouterr().err

    (tmp_path / ".web").mkdir()
    _run_ui(tmp_path)
    assert "First launch" not in capsys.readouterr().err

    _run_ui(tmp_path, "--reset")
    assert "First launch" in capsys.readouterr().err


def test_ui_verbose_shows_the_server_command(
    fake_server: type[_FakeServer],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """``--verbose`` adds the exact Reflex command; the default output omits it.

    Args:
        fake_server: The fake Reflex process.
        capsys: Pytest output-capture fixture.
        tmp_path: Pytest temporary directory.
    """
    _run_ui(tmp_path)
    quiet = capsys.readouterr().err
    assert "Starting patch-sim at http://localhost:" in quiet
    assert "reflex run" not in quiet

    _run_ui(tmp_path, "--verbose")
    verbose = capsys.readouterr().err
    assert "-m reflex run --env prod" in verbose
    assert verbose.count("Starting patch-sim") == 1


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

    assert _run_ui(blocker / "work") == 1
    assert "Cannot prepare the working directory" in capsys.readouterr().err
    assert fake_server.calls == []


def test_ui_refuses_to_reset_a_directory_it_did_not_create(
    fake_server: type[_FakeServer],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """``--workdir <user dir> --reset`` deletes nothing and starts nothing."""
    keep = tmp_path / "notes.txt"
    keep.write_text("important")

    assert _run_ui(tmp_path, "--reset") == 1

    assert keep.read_text() == "important"
    assert "not a patch-sim working directory" in capsys.readouterr().err
    assert fake_server.calls == []


def test_ui_ctrl_c_waits_for_the_server_and_returns_130(
    fake_server: type[_FakeServer], tmp_path: Path
) -> None:
    """Ctrl+C lets the already-interrupted server exit without re-signaling it."""
    fake_server.wait_effects = [KeyboardInterrupt()]

    assert _run_ui(tmp_path) == 130

    (server,) = fake_server.instances
    assert server.poll() is not None
    assert server.signals == []


def test_ui_kills_a_server_that_ignores_ctrl_c(
    fake_server: type[_FakeServer], tmp_path: Path
) -> None:
    """A server still running after the grace period is killed."""
    fake_server.wait_effects = [
        KeyboardInterrupt(),
        subprocess.TimeoutExpired("reflex", launcher.SHUTDOWN_GRACE_SECONDS),
    ]

    assert _run_ui(tmp_path) == 130

    (server,) = fake_server.instances
    assert server.signals == ["kill"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
def test_ui_sigterm_stops_the_server(
    fake_server: type[_FakeServer], tmp_path: Path
) -> None:
    """SIGTERM to the launcher terminates the server instead of orphaning it."""
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers need the main thread")
    previous = signal.getsignal(signal.SIGTERM)
    fake_server.wait_effects = [lambda: os.kill(os.getpid(), signal.SIGTERM)]

    assert _run_ui(tmp_path) == 143

    (server,) = fake_server.instances
    assert server.signals == ["terminate"]
    assert signal.getsignal(signal.SIGTERM) == previous


def test_ui_skips_the_browser_without_a_display(
    fake_server: type[_FakeServer],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """On headless Linux, no console browser takes over the terminal."""
    monkeypatch.setattr(launcher.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    started: list[str] = []
    monkeypatch.setattr(
        launcher,
        "open_browser_when_ready",
        lambda url, proc_alive: started.append(url),
    )

    code = cli.main(["ui", "--workdir", str(tmp_path), "--port", str(_free_port())])

    assert code == 0
    assert started == []
    assert "No display found; open http://localhost:" in capsys.readouterr().err


def test_python_dash_m_runs_the_same_command() -> None:
    """``python -m patch_sim_ui --help`` prints the patch-sim usage."""
    result = subprocess.run(
        [sys.executable, "-m", "patch_sim_ui", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage: patch-sim" in result.stdout
