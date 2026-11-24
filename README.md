## patch-sim: Patch Clamp Simulator

`patch_sim` is a Python library for simulating neuronal electrophysiology using the
[Hodgkin-Huxley model](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model).
It models how action potentials arise at a neuron's membrane by simulating the dynamics
of voltage-gated ion channels over time. The core model includes the classic sodium,
potassium, and leak channels, and can be extended with 10 additional channel types
(Ih, IKa, INaP, INaR, IM, IKir, IKCa, ICaL, ICaT, ICaN) via a pluggable channel
architecture.

Key features:

- **Hodgkin-Huxley neuron model** with configurable conductances, ion concentrations,
  and temperature. Reversal potentials are derived via the Nernst equation, with
  Goldman-Hodgkin-Katz support for mixed-ion channels (e.g. Ih). Optional intracellular
  calcium dynamics are coupled to calcium-permeable channels.
- **Current clamp** simulations: inject a stimulus current and record the resulting
  membrane voltage response.
- **Voltage clamp** simulations: hold the membrane at a commanded voltage and record the
  resulting ionic currents.
- A library of **stimulation protocols**: current clamp (step, ramp, pulse train,
  sinusoidal, chirp, noise) and voltage clamp (step, ramp, pulse train, I-V curve).

---

## Web UI

`patch_sim` includes an interactive web interface inspired by pClamp electrophysiology software.

- Configure neuron parameters and enable/disable additional ion channels with conductance sliders
- Run current clamp and voltage clamp protocols
- Sweep overlay with click/hover/keyboard selection
- Continuous simulation mode for oscilloscope-like real-time recording
- 6 built-in presets: Action Potential, Subthreshold Response, Repetitive Firing, I-V Curve, Na+ Channel Activation, Frequency Response

### Installing from PyPI

```
pip install patch-sim           # core library only
pip install "patch-sim[ui]"     # library plus the web UI
```

Start the web UI from any directory:

```
patch-sim ui
```

The app opens in your browser at `http://localhost:3000` once it is ready.
Options:

- `--port PORT`: serve on a different port (default 3000).
- `--host HOST`: interface to bind to (default `127.0.0.1`, this computer only).
  If you bind to `0.0.0.0` and open the app from another machine, also set
  `REFLEX_API_URL` to the URL the browser uses, for example
  `REFLEX_API_URL=http://myhost:3000`.
- `--no-browser`: do not open a browser.
- `--workdir PATH`: where the generated config and frontend build are kept
  (default: `$PATCH_SIM_WORKDIR`, else a per-user cache directory such as
  `~/.cache/patch-sim/<version>/` on Linux).
- `--reset`: delete the working directory first and rebuild the frontend.
  The launcher only writes to, or deletes, a directory it created (or an empty
  one), so `--workdir` cannot overwrite an existing folder.
- `-v`, `--verbose`: show detailed progress, including the command that runs
  the server.

The first launch downloads a JavaScript runtime and builds the web frontend.
This takes a few minutes and needs internet access; later launches reuse the
build. On minimal Linux systems the download needs `curl` and `unzip`.

### Running from a source checkout

```
uv sync --frozen --extra=ui
uv run reflex run
```

Then open `http://localhost:3000` in your browser.

### Running with Docker

You can run the web UI without installing uv, Python, or Node; only
[Docker](https://docs.docker.com/get-started/get-docker/) is required. From the
repository root:

```
docker build -t patch-sim .
docker run --rm -p 3000:3000 patch-sim
```

or, with Docker Compose:

```
docker compose up --build
```

Then open `http://localhost:3000`. The container runs Reflex in production
mode and serves both the UI and the backend on port 3000. Each start rebuilds
the frontend bundle, which takes a few seconds before the app responds.

If the browser reaches the app at any URL other than `http://localhost:3000`,
pass that URL as `REFLEX_API_URL`:

```
docker run --rm -p 8080:3000 -e REFLEX_API_URL=http://localhost:8080 patch-sim
```

- **Different host port** (e.g. `-p 8080:3000`): without `REFLEX_API_URL` the
  page cannot reach its backend at all, so nothing responds.
- **Another machine on port 3000** (e.g. `http://192.168.1.20:3000`): without
  `REFLEX_API_URL` the UI connects, but trace plots and **Export CSV** fail.

The container runs as a non-root user (uid 1000) and needs a writable
filesystem, because Reflex writes its build and session state at startup.
Reflex telemetry is disabled in the image.

---

## Getting started with development

### Prerequisites

You should have the following installed:

- [uv](https://docs.astral.sh/uv/getting-started/installation/): Python package manager


### Development tooling

Common development tasks are run directly via `uv`. The commands below assume
you are at the top level of the repository.

#### Installing the development environment

```
uv sync --frozen --group=dev
```

This creates a Python `venv` virtual environment under the name `.venv`
in the top-level directory of the repository.

#### Managing dependencies

To regenerate and update project dependencies, run:

```
uv lock --upgrade
```

#### Unit tests

The full unit test suite can be run using:

```
uv run --frozen -m pytest --verbose
```

#### Test coverage

To see a line-by-line coverage report for the core library:

```
uv run --frozen -m pytest --cov=patch_sim --cov-report=term-missing
```

To generate an HTML report (written to `htmlcov/`):

```
uv run --frozen -m pytest --cov=patch_sim --cov-report=html
```

#### Linting and code style

Python code is styled using Ruff. Ruff is configured to cover typical flake8,
black and isort behavior.

Formatting fixes can be applied using:

```
uvx ruff format .
uvx ruff check . --fix-only
```

Conformance to the styling rules can be checked with:

```
uvx ruff check .
uvx ruff format --check .
```

#### Type checking

```
uvx ty check
```

---

## License

This project is licensed under the [GNU General Public License v3.0 or later](https://github.com/JCorson/patch-sim/blob/main/LICENSE).
