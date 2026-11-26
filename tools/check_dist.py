"""Verify the contents of the built wheel and sdist before publishing.

Usage::

    uv build
    uv run --no-project python tools/check_dist.py dist

The checks guard what an installed ``neuroclamp`` needs at runtime (help pages,
JS templates, screenshots) and what must stay out of it (a top-level
``rxconfig`` module, tests, build caches). The script uses only the standard
library, so it runs without the project environment. It exits non-zero and
lists every failure if any check fails.
"""

import argparse
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

#: Files that must be present in the wheel, as glob patterns with the minimum
#: number of matches each.
WHEEL_REQUIRED: dict[str, int] = {
    "neuroclamp/__init__.py": 1,
    "neuroclamp/py.typed": 1,
    "neuroclamp/presets/*.py": 1,
    "neuroclamp_ui/__init__.py": 1,
    "neuroclamp_ui/help_content/*.md": 5,
    "neuroclamp_ui/js/*.js": 2,
    "neuroclamp_ui/help_content/screenshots/*.png": 7,
}

#: Top-level names that must not appear in either artifact.
FORBIDDEN_TOP_LEVEL: tuple[str, ...] = (
    "rxconfig.py",
    "tests",
    "scratch",
    "tools",
    "docs",
    ".web",
    "assets",
)

#: Distributions that may only be required under the ``ui`` extra.
UI_ONLY_REQUIREMENTS: tuple[str, ...] = ("reflex", "plotly", "platformdirs")


def _single(dist_dir: Path, pattern: str) -> Path:
    """Return the single file in ``dist_dir`` matching ``pattern``.

    Args:
        dist_dir: Directory produced by ``uv build``.
        pattern: Glob pattern for the artifact.

    Returns:
        The path of the one matching artifact.

    Raises:
        SystemExit: If there is not exactly one match.
    """
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected one {pattern} in {dist_dir}, found {matches}")
    return matches[0]


def _requirement_name(requirement: str) -> str:
    """Return the normalized distribution name of a ``Requires-Dist`` value.

    Args:
        requirement: A ``Requires-Dist`` header value, such as
            ``reflex[pydantic]>=0.11.0; extra == "ui"``.

    Returns:
        The lower-cased distribution name.
    """
    name = requirement
    for separator in ("[", ";", "<", ">", "=", "!", "~", " "):
        name = name.split(separator, 1)[0]
    return name.strip().lower().replace("_", "-")


def check_wheel(wheel: Path) -> list[str]:
    """Check the wheel's file list and dependency metadata.

    Args:
        wheel: Path to the built ``.whl`` file.

    Returns:
        A list of failure messages; empty when the wheel is valid.
    """
    failures: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_name).decode())

    paths = [Path(name) for name in names]
    for pattern, minimum in WHEEL_REQUIRED.items():
        count = sum(path.match(pattern) for path in paths)
        if count < minimum:
            failures.append(f"wheel: {pattern} matched {count}, expected {minimum}+")

    top_level = {path.parts[0] for path in paths}
    for name in FORBIDDEN_TOP_LEVEL:
        if name in top_level:
            failures.append(f"wheel: forbidden top-level entry {name}")
    if any("__pycache__" in path.parts for path in paths):
        failures.append("wheel: contains __pycache__")

    for requirement in metadata.get_all("Requires-Dist") or []:
        name = _requirement_name(requirement)
        marker = requirement.partition(";")[2].replace("'", '"').replace(" ", "")
        if name in UI_ONLY_REQUIREMENTS and marker != 'extra=="ui"':
            failures.append(f"wheel: {name} is required outside the ui extra")
    if "ui" not in (metadata.get_all("Provides-Extra") or []):
        failures.append("wheel: missing the ui extra")
    return failures


def check_sdist(sdist: Path) -> list[str]:
    """Check that the sdist carries the sources and leaves out development files.

    Args:
        sdist: Path to the built ``.tar.gz`` file.

    Returns:
        A list of failure messages; empty when the sdist is valid.
    """
    failures: list[str] = []
    with tarfile.open(sdist) as archive:
        members = [Path(name) for name in archive.getnames()]
    # Entries are prefixed with "<name>-<version>/".
    relative = [Path(*m.parts[1:]) for m in members if len(m.parts) > 1]
    top_level = {path.parts[0] for path in relative}
    for name in FORBIDDEN_TOP_LEVEL:
        if name in top_level:
            failures.append(f"sdist: forbidden top-level entry {name}")
    if not any(
        path.match("neuroclamp_ui/help_content/screenshots/*.png") for path in relative
    ):
        failures.append("sdist: missing the help-page screenshots")
    return failures


def main(argv: list[str] | None = None) -> int:
    """Run all distribution checks and report failures.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 when every check passes, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dist_dir", type=Path, help="directory produced by uv build")
    args = parser.parse_args(argv)

    wheel = _single(args.dist_dir, "*.whl")
    sdist = _single(args.dist_dir, "*.tar.gz")
    failures = check_wheel(wheel) + check_sdist(sdist)
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    if failures:
        return 1
    print(f"OK {wheel.name} and {sdist.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
