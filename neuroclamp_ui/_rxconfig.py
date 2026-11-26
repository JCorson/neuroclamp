"""Reflex configuration for the neuroclamp web UI.

Reflex loads its configuration from an ``rxconfig.py`` module in the working
directory. The configuration itself lives here, inside the package, so that
it ships with the wheel without installing a top-level ``rxconfig`` module.
The repository-root ``rxconfig.py`` (development) and the stub that
``neuroclamp ui`` writes into its working directory both re-export ``config``
from this module.
"""

import reflex as rx
from reflex.plugins.sitemap import SitemapPlugin

config = rx.Config(
    app_name="neuroclamp_ui",
    disable_plugins=[SitemapPlugin],
    plugins=[
        rx.plugins.RadixThemesPlugin(
            theme=rx.theme(
                appearance="inherit",
                accent_color="blue",
                radius="medium",
            ),
        ),
    ],
)
