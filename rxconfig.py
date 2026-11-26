"""Reflex entry point for running the web UI from a source checkout.

``uv run reflex run`` reads this module from the repository root. It is not
part of the distributed package: the configuration lives in
``neuroclamp_ui._rxconfig``, and ``neuroclamp ui`` writes an equivalent stub into
its own working directory.
"""

from neuroclamp_ui._rxconfig import config

__all__ = ["config"]
