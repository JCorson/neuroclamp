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
        i_ext: NDArray[np.float64],
        v0: float,
        gates0: Sequence[float],
        ca0: float,
        dt: float,
    ) -> NDArray[np.float64]:
        """Run one sweep per row of ``i_ext`` in parallel from the same state.

        Args:
            i_ext: ``(n_sweeps, n_steps)`` injected currents in µA/cm².
            v0: Initial membrane voltage in mV.
            gates0: Initial gate values, in spec order.
            ca0: Initial intracellular Ca²⁺ in mM.
            dt: Time step in ms.

        Returns:
            An ``(n_sweeps, n_steps, n_columns)`` array.

        Raises:
            ValueError: If ``i_ext`` is empty or ``gates0`` has the wrong length.
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
        v_protocol: NDArray[np.float64],
        gates0: Sequence[float],
        ca0: float,
        dt: float,
    ) -> NDArray[np.float64]:
        """Run one voltage-clamp sweep per row of ``v_protocol`` in parallel.

        Args:
            v_protocol: ``(n_sweeps, n_steps)`` prescribed voltages in mV.
            gates0: Initial gate values, in spec order.
            ca0: Initial intracellular Ca²⁺ in mM.
            dt: Time step in ms.

        Returns:
            An ``(n_sweeps, n_steps, n_columns)`` array.

        Raises:
            ValueError: If ``v_protocol`` is empty or ``gates0`` has the wrong
                length.
        """
