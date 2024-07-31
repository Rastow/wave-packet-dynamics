"""Particles module."""


class Particle:
    """Arbitrary particle.

    Parameters
    ----------
    charge : int
        Charge of the particle in atomic units.
    mass : int
        Mass of the particle in atomic units.
    """

    def __init__(self, charge: int, mass: int) -> None:
        self.charge = charge
        self.mass = mass


class Electron(Particle):
    r"""Electron.

    Notes
    -----
    $$q = -1e$$

    $$m = 1m_e$$
    """

    def __init__(self) -> None:
        super().__init__(charge=-1, mass=1)


class Proton(Particle):
    r"""Proton.

    Notes
    -----
    $$q = +1e$$

    $$m = 1836 m_e$$
    """

    def __init__(self) -> None:
        super().__init__(charge=1, mass=1836)
