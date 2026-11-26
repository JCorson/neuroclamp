"""The ``patch-sim`` command-line interface.

``patch-sim ui`` starts the web UI from any directory. The command has
subcommands so other front ends can be added alongside ``ui`` later.

This module uses only the standard library at import time, so ``patch-sim
--help`` works, and the "install the ui extra" message prints cleanly, when
only the core library is installed.

Progress and errors are reported through :mod:`logging` on the
``patch_sim_ui`` logger, which :func:`main` sends to stderr: plain messages at
INFO, ``Warning:`` and ``Error:`` prefixes above it, and DEBUG detail with
``--verbose``.
"""

import argparse
import importlib.metadata
import importlib.util
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

#: Printed when ``patch-sim ui`` runs without the ``ui`` extra installed.
UI_EXTRA_MESSAGE = 'The web UI requires the ui extra: pip install "patch-sim[ui]"'

#: Environment variable that overrides the default working directory.
WORKDIR_ENV_VAR = "PATCH_SIM_WORKDIR"

#: Modules the ``ui`` extra provides.
_UI_MODULES: tuple[str, ...] = ("reflex", "plotly", "platformdirs")

logger = logging.getLogger(__name__)


class _ConsoleFormatter(logging.Formatter):
    """Format log records as console messages.

    INFO and DEBUG records print as the bare message; warnings and errors are
    prefixed with their level, as in ``Error: Port 3000 is already in use``.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format one record.

        Args:
            record: The log record.

        Returns:
            The message, prefixed with ``Warning:`` or ``Error:`` at those
            levels.
        """
        message = super().format(record)
        if record.levelno >= logging.ERROR:
            return f"Error: {message}"
        if record.levelno >= logging.WARNING:
            return f"Warning: {message}"
        return message


class _ConsoleHandler(logging.StreamHandler):
    """Stream handler that :func:`configure_logging` installs and replaces."""


def configure_logging(verbose: bool = False) -> None:
    """Send the ``patch_sim_ui`` loggers' messages to stderr.

    Calling it again replaces the previous console handler, so running
    :func:`main` more than once in a process does not duplicate output.

    Args:
        verbose: Include DEBUG messages.
    """
    package_logger = logging.getLogger("patch_sim_ui")
    for handler in list(package_logger.handlers):
        if isinstance(handler, _ConsoleHandler):
            package_logger.removeHandler(handler)
    handler = _ConsoleHandler(sys.stderr)
    handler.setFormatter(_ConsoleFormatter("%(message)s"))
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.DEBUG if verbose else logging.INFO)


def _package_version() -> str:
    """Return the installed ``patch-sim`` version.

    Returns:
        The distribution version, or ``"0+unknown"`` if the package metadata
        is unavailable (for example, when running from an uninstalled
        checkout).
    """
    try:
        return importlib.metadata.version("patch-sim")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


def _ui_extra_installed() -> bool:
    """Report whether the ``ui`` extra's modules are importable.

    The modules are located without being imported, which keeps the check
    fast.

    Returns:
        ``True`` if every module the web UI needs is installed.
    """
    return all(importlib.util.find_spec(name) is not None for name in _UI_MODULES)


def _port(value: str) -> int:
    """Parse a TCP port number for argparse.

    Args:
        value: The command-line value.

    Returns:
        The port number.

    Raises:
        argparse.ArgumentTypeError: If the value is not an integer from 1 to
            65535.
    """
    message = f"{value!r} is not a port number (1-65535)"
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(message) from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(message)
    return port


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``patch-sim`` command.

    Returns:
        The configured parser, with a ``ui`` subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="patch-sim",
        description="Patch clamp experiment simulator.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {_package_version()}"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    ui = subparsers.add_parser(
        "ui",
        help="start the web UI",
        description=(
            "Start the patch-sim web UI. The first launch builds the web "
            "frontend, which takes a few minutes and needs internet access."
        ),
    )
    ui.add_argument(
        "--port", type=_port, default=3000, help="port to serve on (default: 3000)"
    )
    ui.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind to (default: 127.0.0.1, this computer only)",
    )
    ui.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open a browser once the server is ready",
    )
    ui.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help=(
            "directory for the generated config and frontend build (default: "
            f"${WORKDIR_ENV_VAR}, else a per-user cache directory)"
        ),
    )
    ui.add_argument(
        "--reset",
        action="store_true",
        help="delete the working directory first and rebuild the frontend",
    )
    ui.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show detailed progress, including the command that runs the server",
    )
    return parser


def _run_ui(args: argparse.Namespace) -> int:
    """Run the ``ui`` subcommand.

    Args:
        args: Parsed ``ui`` arguments.

    Returns:
        The process exit code.
    """
    if not _ui_extra_installed():
        logger.error(UI_EXTRA_MESSAGE)
        return 1

    from patch_sim_ui import launcher

    workdir = args.workdir
    if workdir is None and os.environ.get(WORKDIR_ENV_VAR):
        workdir = Path(os.environ[WORKDIR_ENV_VAR])
    if workdir is None:
        workdir = launcher.default_workdir(_package_version())

    try:
        return launcher.run_ui(
            workdir=workdir.expanduser().resolve(),
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
            reset=args.reset,
        )
    except launcher.LauncherError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        # Ctrl+C while preparing the working directory, before the server
        # starts; once it runs, the launcher handles Ctrl+C itself.
        return 130


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``patch-sim`` command.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ui":
        configure_logging(verbose=args.verbose)
        return _run_ui(args)
    parser.print_help(sys.stderr)
    return 2
