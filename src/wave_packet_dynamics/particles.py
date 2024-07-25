"""Particles module."""

from typing import Protocol


class Particle(Protocol):
    """Particle protocol."""

    charge: int
    mass: int


class Electron:
    r"""Electron.

    Notes
    -----
    $$q = -1e$$

    $$m = 1m_e$$
    """

    charge = -1
    mass = 1


class Proton:
    r"""Proton.

    Notes
    -----
    $$q = +1e$$

    $$m = 1836 m_e$$
    """

    charge = 1
    mass = 1836
