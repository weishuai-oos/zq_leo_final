"""Dynamics-randomized ImpactV2 continuation task implementations.

Modules are intentionally imported lazily: :mod:`mdp` is CPU-testable without
an IsaacLab installation, while env/config modules are loaded by Gym only when
the simulator task is selected.
"""
