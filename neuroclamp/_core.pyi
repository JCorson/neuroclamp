"""Type stub for the compiled Rust core (``rust/src/python.rs``).

The extension integrates current- and voltage-clamp simulations of a neuron
lowered to a model spec by :func:`neuroclamp._spec.lower`.  Output columns, in
order, are the membrane voltage, one current per channel, the total current,
one value per gate, and ``ca_i`` when the model has Ca²⁺ dynamics.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

class Model:
    """A model compiled from the JSON form of a lowered neuron spec."""

    def __init__(self, spec_json: str) -> None:
        """Parse and validate a model spec.

        Args:
            spec_json: ``json.dumps(lower(neuron))``.

        Raises:
            ValueError: If the JSON is malformed or the spec is invalid.
        """

    @property
    def n_columns(self) -> int:
        """Number of output columns per time step."""

    def simulate_current_clamp(
        self,
        i_ext: NDArray[np.float64],
        v0: float,
        gates0: Sequence[float],
        ca0: float,
        dt: float,
    ) -> NDArray[np.float64]:
        """Run one current-clamp simulation.

        Args:
            i_ext: Injected current per time step in µA/cm².
            v0: Initial membrane voltage in mV.
            gates0: Initial gate values, in spec order.
            ca0: Initial intracellular Ca²⁺ in mM.
            dt: Time step in ms.

        Returns:
            An ``(n_steps, n_columns)`` array.

        Raises:
            ValueError: If ``i_ext`` is empty or ``gates0`` has the wrong length.
        """

    def simulate_current_clamp_batch(
        self,
        i_ext: Sequence[NDArray[np.float64]],
        v0: float,
        gates0: Sequence[float],
        ca0: float,
        dt: float,
    ) -> list[NDArray[np.float64]]:
        """Run one sweep per array in ``i_ext`` in parallel from the same state.

        Every fresh current-clamp run starts from rest, so the sweeps share one
        initial state.  Sweeps may differ in length.

        Args:
            i_ext: Injected current per time step in µA/cm², one array per
                sweep.
            v0: Initial membrane voltage in mV.
            gates0: Initial gate values, in spec order.
            ca0: Initial intracellular Ca²⁺ in mM.
            dt: Time step in ms.

        Returns:
            One ``(n_steps, n_columns)`` array per sweep, in order.

        Raises:
            ValueError: If ``i_ext`` or any sweep is empty, or ``gates0`` has
                the wrong length.
        """

    def simulate_voltage_clamp(
        self,
        v_protocol: NDArray[np.float64],
        gates0: Sequence[float],
        ca0: float,
        dt: float,
    ) -> NDArray[np.float64]:
        """Run one voltage-clamp simulation.

        Args:
            v_protocol: Prescribed membrane voltage per time step in mV.
            gates0: Initial gate values, in spec order.
            ca0: Initial intracellular Ca²⁺ in mM.
            dt: Time step in ms.

        Returns:
            An ``(n_steps, n_columns)`` array.

        Raises:
            ValueError: If ``v_protocol`` is empty or ``gates0`` has the wrong
                length.
        """

    def simulate_voltage_clamp_batch(
        self,
        v_protocol: Sequence[NDArray[np.float64]],
        gates0: Sequence[Sequence[float]],
        ca0: Sequence[float],
        dt: float,
    ) -> list[NDArray[np.float64]]:
        """Run one voltage-clamp sweep per array in ``v_protocol`` in parallel.

        Each sweep starts from its own state, since a voltage-clamp run starts
        from the steady state at its protocol's first voltage.  Sweeps may
        differ in length.

        Args:
            v_protocol: Prescribed voltage per time step in mV, one array per
                sweep.
            gates0: Initial gate values, one sequence per sweep, in spec order.
            ca0: Initial intracellular Ca²⁺ in mM, one value per sweep.
            dt: Time step in ms.

        Returns:
            One ``(n_steps, n_columns)`` array per sweep, in order.

        Raises:
            ValueError: If ``v_protocol`` or any sweep is empty, or ``gates0`` or
                ``ca0`` does not have one entry of the right length per sweep.
        """
