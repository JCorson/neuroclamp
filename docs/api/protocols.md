# Protocols

Stimulus builders for current- and voltage-clamp simulations. The dispatchers
(`build_current_protocol`, `build_voltage_protocol`) select a generator by name;
the generators can also be called directly for single-sweep stimuli.

## Dispatchers

::: neuroclamp.build_current_protocol

::: neuroclamp.build_voltage_protocol

## Current-clamp generators

::: neuroclamp.step_current

::: neuroclamp.ramp_current

::: neuroclamp.pulse_train

::: neuroclamp.sinusoidal_current

::: neuroclamp.chirp_current

::: neuroclamp.noise_current

## Voltage-clamp generators

::: neuroclamp.step_voltage

::: neuroclamp.ramp_voltage

::: neuroclamp.pulse_train_voltage
