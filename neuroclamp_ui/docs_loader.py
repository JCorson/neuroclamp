"""Load the application-documentation pages for the in-app ``/help`` route.

The Markdown files under ``neuroclamp_ui/help_content/`` are the user guide for
the web application: how to configure a neuron, set up a protocol, run a
simulation, and read the analysis. They are read once at import and rendered by
``rx.markdown`` on the ``/help`` route.

This is the *application* documentation track. The *library* documentation (the
mkdocs API reference) lives separately under the repository-root ``docs/`` and
is published as a static site; the help page links out to it.

Help content lives inside the package, so it resolves relative to this module
regardless of the working directory. That includes the screenshots the prose
embeds as ``/screenshots/<name>.png``: :func:`install_screenshots` copies them
into ``assets/screenshots/`` in the directory Reflex runs from, which Reflex
serves at ``/screenshots/``.
"""

import filecmp
import logging
import pathlib
import re
import shutil
import uuid

logger = logging.getLogger(__name__)

_DOCS_DIR = pathlib.Path(__file__).parent / "help_content"

#: Help-page screenshots shipped with the package.
SCREENSHOTS_DIR: pathlib.Path = _DOCS_DIR / "screenshots"

#: URL of the deployed mkdocs API-reference site, linked from the help page.
API_DOCS_URL: str = "https://jcorson.github.io/neuroclamp/"

#: Ordered ``(slug, label, filename)`` triples driving the help topic nav.
_TOPICS: list[tuple[str, str, str]] = [
    ("getting-started", "Getting started", "getting-started.md"),
    ("neuron", "Configuring the neuron", "neuron.md"),
    ("protocol", "Setting up a protocol", "protocol.md"),
    ("running", "Running & reading results", "running.md"),
    ("analysis", "Analysis panels", "analysis.md"),
]

#: Ordered ``(slug, label)`` pairs for building the topic nav.
HELP_TOPICS: list[tuple[str, str]] = [(slug, label) for slug, label, _ in _TOPICS]

#: Default topic shown when the help page first loads.
DEFAULT_TOPIC: str = _TOPICS[0][0]

# Markdown links whose target is not an absolute http(s) URL — i.e. the
# cross-topic links between help pages. They have no meaning on the help page
# (which switches topics by state, not by URL), so they are reduced to their
# link text. The ``(?<!\!)`` lookbehind leaves image syntax (``![alt](src)``)
# untouched so screenshots still render.
_INTERNAL_LINK = re.compile(r"(?<!\!)\[([^\]]+)\]\((?!https?://)[^)]*\)")


def _load(filename: str) -> str:
    """Read one help page and flatten its cross-topic links.

    Args:
        filename: Name of the Markdown file under ``help_content/``.

    Returns:
        The file's contents with internal cross-topic links flattened to plain
        text; image references and external (http/https) links are left intact.
    """
    text = (_DOCS_DIR / filename).read_text(encoding="utf-8")
    return _INTERNAL_LINK.sub(r"\1", text)


#: Topic slug -> rendered Markdown body, loaded once at import.
PROSE: dict[str, str] = {slug: _load(filename) for slug, _, filename in _TOPICS}


def install_screenshots(assets_dir: pathlib.Path) -> None:
    """Copy the help-page screenshots into a Reflex app's assets directory.

    Reflex serves static files only from ``assets/`` in its working directory,
    and that directory differs between a source checkout, the ``neuroclamp ui``
    working directory and the Docker image. Copying from the package at app
    import covers all three from one source. ``rx.asset(shared=True)`` does the
    same with symlinks, but creating a symlink on Windows requires Developer
    Mode or administrator rights.

    A file is written only when its contents differ, through a temporary file
    renamed into place, so concurrent imports never read a partial file and an
    unchanged screenshot is left alone for the dev server's file watcher.
    Screenshots no longer in the package are removed. A failure to write is
    logged rather than raised: the app still runs, with broken help images.

    Args:
        assets_dir: The app's ``assets`` directory, usually
            ``Path.cwd() / "assets"``.
    """
    dest = assets_dir / "screenshots"
    shipped = {path.name: path for path in SCREENSHOTS_DIR.glob("*.png")}
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for name, src in shipped.items():
            target = dest / name
            if target.is_file() and filecmp.cmp(src, target, shallow=False):
                continue
            staged = dest / f".{uuid.uuid4().hex}.tmp"
            try:
                shutil.copyfile(src, staged)
                staged.replace(target)
            finally:
                staged.unlink(missing_ok=True)
        for stale in dest.glob("*.png"):
            if stale.name not in shipped:
                stale.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not copy the help screenshots into %s: %s", dest, exc)
