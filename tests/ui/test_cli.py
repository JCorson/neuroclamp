"""Behavior tests for the ``patch-sim`` command line.

``patch-sim ui`` runs as a real process, the way a user starts it. The only
stand-in is the server it launches: a small fake ``reflex`` package placed
first on ``PYTHONPATH``, so ``python -m reflex`` in the child process runs the
fake instead of building and serving the app. The fake records where it ran,
then exits with a chosen code or keeps running until it is signaled, which
lets the tests check exit codes and shutdown with real processes and real
signals.
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from patch_sim_ui import cli

#: The fake ``python -m reflex``. ``FAKE_REFLEX_RECORD`` names a JSON file
#: that receives the working directory and arguments once the fake is ready.
#: With ``FAKE_REFLEX_SERVE`` set it runs until SIGINT or SIGTERM and writes
#: the signal name to ``<record>.stopped``; otherwise it exits with
#: ``FAKE_REFLEX_EXIT`` (default 0).
_FAKE_REFLEX_MAIN = """
import json
import os
import signal
import sys
import time
from pathlib import Path

record = Path(os.environ["FAKE_REFLEX_RECORD"])


def _stop(signum, frame):
    record.with_suffix(".stopped").write_text(signal.Signals(signum).name)
    raise SystemExit(0)


if os.environ.get("FAKE_REFLEX_SERVE"):
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
record.write_text(json.dumps({"cwd": os.getcwd(), "argv": sys.argv[1:]}))
if os.environ.get("FAKE_REFLEX_SERVE"):
    while True:
        time.sleep(0.05)
raise SystemExit(int(os.environ.get("FAKE_REFLEX_EXIT", "0")))
"""

#: The ``patch-sim`` console script installed alongside this interpreter.
_PATCH_SIM = shutil.which("patch-sim", path=str(Path(sys.executable).parent))


@pytest.fixture
def fake_env(tmp_path: Path) -> dict[str, str]:
    """Build an environment in which ``patch-sim ui`` launches the fake server.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Environment variables for a ``patch-sim`` process. ``FAKE_REFLEX_RECORD``
        points at the file the fake server writes when it starts.
    """
    package = tmp_path / "fake-site" / "reflex"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(_FAKE_REFLEX_MAIN)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in (cli.WORKDIR_ENV_VAR, "DISPLAY", "WAYLAND_DISPLAY")
    }
    env["PYTHONPATH"] = str(package.parent)
    env["FAKE_REFLEX_RECORD"] = str(tmp_path / "server.json")
    return env


def _free_port() -> int:
    """Return a TCP port that is currently free.

    Returns:
        The port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _ui_command(workdir: Path | None, *extra: str) -> list[str]:
    """Return a ``patch-sim ui --no-browser`` command line on a free port.

    Args:
        workdir: Value for ``--workdir``, or ``None`` to leave it out.
        *extra: Further command-line arguments.

    Returns:
        The argument list.
    """
    assert _PATCH_SIM is not None, "the patch-sim console script is not installed"
    command = [_PATCH_SIM, "ui", "--no-browser", "--port", str(_free_port())]
    if workdir is not None:
        command += ["--workdir", str(workdir)]
    return [*command, *extra]


def _run_ui(
    env: dict[str, str], workdir: Path | None, *extra: str
) -> subprocess.CompletedProcess[str]:
    """Run ``patch-sim ui`` to completion against the fake server.

    Args:
        env: Environment from the ``fake_env`` fixture.
        workdir: Value for ``--workdir``, or ``None`` to leave it out.
        *extra: Further command-line arguments.

    Returns:
        The finished process, with its output captured as text.
    """
    return subprocess.run(
        _ui_command(workdir, *extra),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _server_record(env: dict[str, str]) -> dict[str, object] | None:
    """Return what the fake server recorded, if it was started.

    Args:
        env: Environment from the ``fake_env`` fixture.

    Returns:
        The recorded working directory and arguments, or ``None`` if the
        server never started.
    """
    record = Path(env["FAKE_REFLEX_RECORD"])
    return json.loads(record.read_text()) if record.is_file() else None


def _start_serving(env: dict[str, str], workdir: Path) -> subprocess.Popen[str]:
    """Start ``patch-sim ui`` with a fake server that runs until signaled.

    The launcher gets its own process group, like a job started from a
    terminal, so a test can deliver Ctrl+C to the launcher and server
    together.

    Args:
        env: Environment from the ``fake_env`` fixture.
        workdir: Working directory for the launcher.

    Returns:
        The running launcher process, once the fake server is up.
    """
    proc = subprocess.Popen(
        _ui_command(workdir),
        env={**env, "FAKE_REFLEX_SERVE": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 30
    while _server_record(env) is None:
        if proc.poll() is not None or time.monotonic() > deadline:
            proc.kill()
            output, _ = proc.communicate()
            pytest.fail(f"the server did not start:\n{output}")
        time.sleep(0.05)
    return proc


def test_no_command_prints_usage_and_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Running ``patch-sim`` with no subcommand shows usage and exits with 2.

    Args:
        capsys: Pytest output-capture fixture.
    """
    assert cli.main([]) == 2
    assert "usage: patch-sim" in capsys.readouterr().err


def test_version_flag_prints_the_package_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``patch-sim --version`` prints the version and exits successfully.

    Args:
        capsys: Pytest output-capture fixture.
    """
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])

    assert excinfo.value.code == 0
    assert capsys.readouterr().out.startswith("patch-sim ")


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


def test_ui_serves_from_the_workdir_and_returns_the_server_exit_code(
    fake_env: dict[str, str], tmp_path: Path
) -> None:
    """The server runs in the working directory, and its exit code is returned.

    Args:
        fake_env: Environment that launches the fake server.
        tmp_path: Pytest temporary directory.
    """
    workdir = tmp_path / "work"

    result = _run_ui({**fake_env, "FAKE_REFLEX_EXIT": "3"}, workdir)

    assert result.returncode == 3, result.stderr
    record = _server_record(fake_env)
    assert record is not None
    assert Path(str(record["cwd"])) == workdir
    assert "Starting patch-sim at http://localhost:" in result.stderr


def test_ui_reads_workdir_from_environment(
    fake_env: dict[str, str], tmp_path: Path
) -> None:
    """``PATCH_SIM_WORKDIR`` selects the working directory when no flag is given.

    Args:
        fake_env: Environment that launches the fake server.
        tmp_path: Pytest temporary directory.
    """
    workdir = tmp_path / "from-env"

    result = _run_ui({**fake_env, cli.WORKDIR_ENV_VAR: str(workdir)}, None)

    assert result.returncode == 0, result.stderr
    record = _server_record(fake_env)
    assert record is not None
    assert Path(str(record["cwd"])) == workdir


def test_ui_announces_the_frontend_build_only_when_needed(
    fake_env: dict[str, str], tmp_path: Path
) -> None:
    """The slow-first-launch notice appears without a build, or with --reset.

    Args:
        fake_env: Environment that launches the fake server.
        tmp_path: Pytest temporary directory.
    """
    workdir = tmp_path / "work"
    assert "First launch" in _run_ui(fake_env, workdir).stderr

    (workdir / ".web").mkdir()
    assert "First launch" not in _run_ui(fake_env, workdir).stderr

    assert "First launch" in _run_ui(fake_env, workdir, "--reset").stderr


def test_ui_verbose_shows_the_server_command(
    fake_env: dict[str, str], tmp_path: Path
) -> None:
    """``--verbose`` adds the command that runs the server.

    Args:
        fake_env: Environment that launches the fake server.
        tmp_path: Pytest temporary directory.
    """
    quiet = _run_ui(fake_env, tmp_path / "work").stderr
    verbose = _run_ui(fake_env, tmp_path / "work", "--verbose").stderr

    assert "reflex run" not in quiet
    assert "-m reflex run" in verbose


def test_ui_refuses_a_port_in_use(fake_env: dict[str, str], tmp_path: Path) -> None:
    """A busy port is reported with a hint, and the server is not started.

    Args:
        fake_env: Environment that launches the fake server.
        tmp_path: Pytest temporary directory.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]

        result = _run_ui(fake_env, tmp_path / "work", "--port", str(port))

    assert result.returncode == 1
    assert f"Error: Port {port} is already in use" in result.stderr
    assert "Traceback" not in result.stderr
    assert _server_record(fake_env) is None


def test_ui_reports_an_unwritable_workdir(
    fake_env: dict[str, str], tmp_path: Path
) -> None:
    """A working directory that cannot be created gives a readable error.

    Args:
        fake_env: Environment that launches the fake server.
        tmp_path: Pytest temporary directory.
    """
    blocker = tmp_path / "a-file"
    blocker.write_text("")

    result = _run_ui(fake_env, blocker / "work")

    assert result.returncode == 1
    assert "Cannot prepare the working directory" in result.stderr
    assert "Traceback" not in result.stderr
    assert _server_record(fake_env) is None


def test_ui_refuses_to_reset_a_directory_it_did_not_create(
    fake_env: dict[str, str], tmp_path: Path
) -> None:
    """``--workdir <user dir> --reset`` deletes nothing and starts nothing.

    Args:
        fake_env: Environment that launches the fake server.
        tmp_path: Pytest temporary directory.
    """
    user_dir = tmp_path / "notes"
    user_dir.mkdir()
    keep = user_dir / "notes.txt"
    keep.write_text("important")

    result = _run_ui(fake_env, user_dir, "--reset")

    assert result.returncode == 1
    assert keep.read_text() == "important"
    assert "not a patch-sim working directory" in result.stderr
    assert _server_record(fake_env) is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_ui_ctrl_c_stops_the_server_and_returns_130(
    fake_env: dict[str, str], tmp_path: Path
) -> None:
    """Ctrl+C stops the server, and the launcher exits with 130.

    Args:
        fake_env: Environment that launches the fake server.
        tmp_path: Pytest temporary directory.
    """
    proc = _start_serving(fake_env, tmp_path / "work")

    # A terminal delivers Ctrl+C to every process in the foreground group.
    os.killpg(proc.pid, signal.SIGINT)
    proc.communicate(timeout=30)

    assert proc.returncode == 130
    stopped = Path(fake_env["FAKE_REFLEX_RECORD"]).with_suffix(".stopped")
    assert stopped.read_text() == "SIGINT"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_ui_sigterm_stops_the_server_and_returns_143(
    fake_env: dict[str, str], tmp_path: Path
) -> None:
    """SIGTERM to the launcher alone stops the server instead of orphaning it.

    Args:
        fake_env: Environment that launches the fake server.
        tmp_path: Pytest temporary directory.
    """
    proc = _start_serving(fake_env, tmp_path / "work")

    proc.terminate()
    proc.communicate(timeout=60)

    assert proc.returncode == 143
    stopped = Path(fake_env["FAKE_REFLEX_RECORD"]).with_suffix(".stopped")
    assert stopped.read_text() == "SIGTERM"


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
