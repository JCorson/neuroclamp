"""Behavior tests for the ``patch-sim`` command line.

The Reflex server is never started: ``subprocess.Popen`` is replaced with an
autospec mock, so every call the launcher makes is checked against the real
``Popen`` signatures. The tests check what the launcher would run, where, and
how it shuts the server down.
"""

import os
import signal
import socket
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import pytest

from patch_sim_ui import cli, launcher


def _script_server(
    proc: mock.NonCallableMagicMock,
    *wait_effects: BaseException | Callable[[], None],
    exit_code: int = 0,
) -> None:
    """Script how the mocked server process behaves.

    Successive ``wait()`` calls apply ``wait_effects`` in order: an exception
    is raised, a callable is called. Once they run out, ``wait()`` returns
    ``exit_code``. The process reports itself as running (``poll()`` returns
    ``None``) until ``wait()`` returns or ``kill()`` is called.

    Args:
        proc: The mocked ``Popen`` instance.
        *wait_effects: Effects of successive ``wait()`` calls.
        exit_code: Exit code of the process.
    """
    pending = list(wait_effects)
    running = [bool(pending)]

    def poll() -> int | None:
        """Report the scripted exit status.

        Returns:
            ``None`` while running, otherwise ``exit_code``.
        """
        return None if running[0] else exit_code

    def wait(timeout: float | None = None) -> int:
        """Apply the next scripted effect, or exit.

        Args:
            timeout: Ignored.

        Returns:
            ``exit_code``.
        """
        if pending:
            effect = pending.pop(0)
            if isinstance(effect, BaseException):
                raise effect
            effect()
        running[0] = False
        return exit_code

    def kill() -> None:
        """Mark the process exited."""
        running[0] = False

    proc.poll.side_effect = poll
    proc.wait.side_effect = wait
    proc.kill.side_effect = kill


@pytest.fixture
def popen(monkeypatch: pytest.MonkeyPatch) -> mock.MagicMock:
    """Replace the Reflex child process and the browser with mocks.

    The server exits with code 0 as soon as the launcher waits on it; tests
    change that with :func:`_script_server` on ``popen.return_value``.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        The autospec mock of ``subprocess.Popen``.
    """
    popen = mock.create_autospec(subprocess.Popen)
    # ``pid`` is set by Popen.__init__, so it is not part of the class spec.
    popen.return_value.pid = 4242
    _script_server(popen.return_value)
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.webbrowser, "open", _fail_browser_open)
    monkeypatch.delenv(cli.WORKDIR_ENV_VAR, raising=False)
    return popen


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
    popen: mock.MagicMock,
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
    popen.assert_not_called()


def test_ui_runs_reflex_in_production_mode_from_the_workdir(
    popen: mock.MagicMock, tmp_path: Path
) -> None:
    """The server runs single-port prod mode in the prepared working directory."""
    port = _free_port()

    code = cli.main(
        ["ui", "--no-browser", "--workdir", str(tmp_path), "--port", str(port)]
    )

    assert code == 0
    popen.assert_called_once()
    (command,), kwargs = popen.call_args
    assert kwargs["cwd"] == tmp_path.resolve()
    assert command[1:4] == ["-m", "reflex", "run"]
    cmd = " ".join(command)
    assert f"--env prod --single-port --frontend-port {port}" in cmd
    assert "--backend-host 127.0.0.1" in cmd
    assert kwargs["env"]["REFLEX_TELEMETRY_ENABLED"] == "false"
    assert (tmp_path / "rxconfig.py").read_text() == launcher.RXCONFIG_STUB


def test_ui_returns_the_server_exit_code(popen: mock.MagicMock, tmp_path: Path) -> None:
    """A failing server's exit code is passed through to the shell."""
    _script_server(popen.return_value, exit_code=3)

    assert _run_ui(tmp_path) == 3


def test_ui_announces_the_frontend_build_only_when_needed(
    popen: mock.MagicMock,
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
    popen: mock.MagicMock,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """``--verbose`` adds the exact Reflex command; the default output omits it.

    Args:
        popen: The mocked ``subprocess.Popen``.
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
    popen: mock.MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``PATCH_SIM_WORKDIR`` selects the working directory when no flag is given."""
    monkeypatch.setenv(cli.WORKDIR_ENV_VAR, str(tmp_path))

    cli.main(["ui", "--no-browser", "--port", str(_free_port())])

    assert popen.call_args.kwargs["cwd"] == tmp_path.resolve()


def test_ui_refuses_a_port_in_use(
    popen: mock.MagicMock,
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
    popen.assert_not_called()


def test_ui_reports_an_unwritable_workdir(
    popen: mock.MagicMock,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """A working directory that cannot be created gives a readable error."""
    blocker = tmp_path / "a-file"
    blocker.write_text("")

    assert _run_ui(blocker / "work") == 1
    assert "Cannot prepare the working directory" in capsys.readouterr().err
    popen.assert_not_called()


def test_ui_refuses_to_reset_a_directory_it_did_not_create(
    popen: mock.MagicMock,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """``--workdir <user dir> --reset`` deletes nothing and starts nothing."""
    keep = tmp_path / "notes.txt"
    keep.write_text("important")

    assert _run_ui(tmp_path, "--reset") == 1

    assert keep.read_text() == "important"
    assert "not a patch-sim working directory" in capsys.readouterr().err
    popen.assert_not_called()


def test_ui_ctrl_c_waits_for_the_server_and_returns_130(
    popen: mock.MagicMock, tmp_path: Path
) -> None:
    """Ctrl+C lets the already-interrupted server exit without re-signaling it."""
    server = popen.return_value
    _script_server(server, KeyboardInterrupt())

    assert _run_ui(tmp_path) == 130

    assert server.poll() is not None
    server.terminate.assert_not_called()
    server.kill.assert_not_called()


def test_ui_kills_a_server_that_ignores_ctrl_c(
    popen: mock.MagicMock, tmp_path: Path
) -> None:
    """A server still running after the grace period is killed."""
    server = popen.return_value
    _script_server(
        server,
        KeyboardInterrupt(),
        subprocess.TimeoutExpired("reflex", launcher.SHUTDOWN_GRACE_SECONDS),
    )

    assert _run_ui(tmp_path) == 130

    server.terminate.assert_not_called()
    server.kill.assert_called_once_with()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
def test_ui_sigterm_stops_the_server(popen: mock.MagicMock, tmp_path: Path) -> None:
    """SIGTERM to the launcher terminates the server instead of orphaning it."""
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers need the main thread")
    previous = signal.getsignal(signal.SIGTERM)
    server = popen.return_value
    _script_server(server, lambda: os.kill(os.getpid(), signal.SIGTERM))

    assert _run_ui(tmp_path) == 143

    server.terminate.assert_called_once_with()
    server.kill.assert_not_called()
    assert signal.getsignal(signal.SIGTERM) == previous


def test_ui_skips_the_browser_without_a_display(
    popen: mock.MagicMock,
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
