"""Hatchling build hook that packages the help-page screenshots.

The screenshots live in the repository-root ``assets/screenshots/`` directory,
where Reflex serves them during development. The standard wheel carries a copy
at ``patch_sim_ui/_web_assets/screenshots/`` for the ``patch-sim ui`` launcher
to place in its working directory. Editable installs skip the copy: they read
the screenshots straight from the checkout, and a copy would leave a stray
``patch_sim_ui`` directory in site-packages.
"""

from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class ScreenshotsBuildHook(BuildHookInterface):
    """Force-include the screenshots in standard (non-editable) wheels."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Add the screenshots to the wheel being built.

        Args:
            version: The wheel build version, ``"standard"`` or ``"editable"``.
            build_data: Mutable build data shared with the wheel builder.
        """
        if self.target_name == "wheel" and version == "standard":
            build_data["force_include"]["assets/screenshots"] = (
                "patch_sim_ui/_web_assets/screenshots"
            )
