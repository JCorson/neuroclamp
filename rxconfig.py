"""Reflex entry point for running the web UI from a source checkout.

``uv run reflex run`` reads this module from the repository root. It is not
part of the distributed package: the configuration lives in
``patch_sim_ui._rxconfig``, and ``patch-sim ui`` writes an equivalent stub into
its own working directory.
"""

from patch_sim_ui._rxconfig import config

__all__ = ["config"]
