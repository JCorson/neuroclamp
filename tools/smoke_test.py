"""Smoke-test a running instance of the web UI in a headless browser.

Checks that the app works end to end, not just that it serves pages: the
browser connects to the Reflex backend, a simulation runs and its trace plot
renders (which exercises the figure-fetch API), and Export CSV returns a CSV
(which exercises the export API). Both APIs are reached through the app's
configured backend URL, so a wrong ``REFLEX_API_URL`` fails here.

CI runs this against the Docker image and against ``neuroclamp ui`` started
from the built wheel; it works against any running app.

Usage:
    1. Start the app, e.g. ``docker run --rm -p 3000:3000 neuroclamp``.
    2. ``uv run --frozen --group browser python tools/smoke_test.py``

The target URL defaults to ``http://localhost:3000`` and can be overridden with
the ``NEUROCLAMP_URL`` environment variable. Exits non-zero on the first failed
check.
"""

import os
import sys

from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeout

BASE_URL = os.environ.get("NEUROCLAMP_URL", "http://localhost:3000")

#: JS expression for the rendered Plotly graph div, or null before the first
#: render. Plotly.react() adds ``js-plotly-plot`` to ``#ps-trace-plot`` itself.
_GRAPH_DIV = (
    "(() => { const e = document.querySelector('#ps-trace-plot');"
    " return e && (e.classList.contains('js-plotly-plot')"
    " ? e : e.querySelector('.js-plotly-plot')); })()"
)

#: JS predicate: the trace plot holds at least one trace with samples. Plotly
#: serializes numpy arrays as ``{dtype, bdata}`` (base64) objects, which
#: ``graphDiv.data`` keeps as-is, so accept either encoding.
_TRACE_HAS_DATA = (
    f"() => {{ const g = {_GRAPH_DIV};"
    " return !!(g && g.data && g.data.some(d => d.x"
    " && (d.x.length > 0 || (d.x.bdata && d.x.bdata.length > 0)))); }"
)

#: JS that summarizes the trace plot's state for a failure message.
_DESCRIBE_PLOT = (
    f"() => {{ const g = {_GRAPH_DIV};"
    " if (!g) return 'no rendered Plotly graph in #ps-trace-plot';"
    " return JSON.stringify((g.data || []).map(d => ({type: d.type,"
    " x: d.x ? Object.keys(d.x).slice(0, 3) : null}))); }"
)


class SmokeTestError(Exception):
    """A smoke-test check failed."""


def _connect(page: Page) -> None:
    """Load the app and wait for the browser to connect to the backend.

    Args:
        page: The Playwright page driving the app.

    Raises:
        SmokeTestError: If the sidebar never renders or the Reflex event
            websocket never opens.
    """
    try:
        with page.expect_websocket(
            lambda ws: "/_event/" in ws.url, timeout=30_000
        ) as socket:
            page.goto(BASE_URL, wait_until="networkidle")
        page.locator("#ps-sidebar").wait_for(state="visible", timeout=30_000)
    except PlaywrightTimeout as exc:
        raise SmokeTestError(
            "the page did not connect to the backend; if the app is published "
            "on a different host port, set REFLEX_API_URL"
        ) from exc
    print(f"ok: page loaded and connected to {socket.value.url}")


def _run_simulation(page: Page) -> None:
    """Click Run and wait for the trace plot to show simulated data.

    Retries the click, because a click that lands before the socket.io
    handshake completes is dropped.

    Args:
        page: The Playwright page driving the app.

    Raises:
        SmokeTestError: If the trace never renders.
    """
    for _ in range(3):
        run_btn = page.get_by_role("button", name="Run", exact=True)
        if run_btn.count() > 0 and run_btn.first.is_visible():
            run_btn.first.click()
        try:
            page.wait_for_function(_TRACE_HAS_DATA, timeout=20_000)
        except PlaywrightTimeout:
            continue
        print("ok: simulation ran and the trace plot rendered")
        return
    raise SmokeTestError(
        "the trace plot did not render after clicking Run; plot state: "
        f"{page.evaluate(_DESCRIBE_PLOT)}"
    )


def _export_csv(page: Page) -> None:
    """Click Export CSV and check that the export API returns a CSV file.

    Args:
        page: The Playwright page driving the app.

    Raises:
        SmokeTestError: If the export request fails or returns no CSV rows.
    """
    try:
        with page.expect_response(
            lambda r: "/api/export/" in r.url, timeout=15_000
        ) as info:
            page.get_by_role("button", name="Export CSV").click()
    except PlaywrightTimeout as exc:
        raise SmokeTestError("Export CSV made no request to the export API") from exc
    response = info.value
    if response.status != 200:
        raise SmokeTestError(f"export API returned HTTP {response.status}")
    # The download response's body is not exposed to the page, so fetch the
    # same token again (exports are not consumed on read).
    # The export opens with ``# key: value`` metadata lines, then a header row
    # starting with ``time_ms``, then one row per sample.
    lines = page.request.get(response.url).text().strip().splitlines()
    rows = [line for line in lines if not line.startswith("#")]
    if len(rows) < 2 or not rows[0].startswith("time_ms,"):
        raise SmokeTestError(f"export is not a CSV with data rows: {rows[:2]!r}")
    print(f"ok: Export CSV returned {len(rows) - 1} data rows")


def main() -> int:
    """Run every smoke-test check against ``BASE_URL``.

    Returns:
        Process exit code: 0 when all checks pass, 1 otherwise.
    """
    print(f"Smoke-testing {BASE_URL}")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            _connect(page)
            _run_simulation(page)
            _export_csv(page)
        except SmokeTestError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        finally:
            browser.close()
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
