"""Simulation module.

Provides a class for the simulation of one-dimensional single particle
quantum dynamics in a time-independent external potential. The wave
packet is propagated in time by applying the time evolution operator at
each time step. While the simulation is running for a specified number
of time steps, wave function, probability density and expectation values
are saved at specified intervals.

Examples
--------
To set up a simulation, a grid, a particle, its initial wave function
and a potential function are required.
>>> from functools import partial
>>> import wave_packet_dynamics as wpd
>>> simulation = wpd.Simulation(
...     grid=wpd.Grid((0, 10), 1000),
...     particle=wpd.particles.Electron(),
...     wave_function=partial(
...         wpd.wave_function.gaussian, sigma=1, x0=5, k0=10
...     ),
...     potential=wpd.potential.zero,
... )

Creating a simulation object also normalizes the wave function.
>>> simulation.expectation_value("total_density")
np.float64(1.0000005734276094)

At any point of the simulation, expectation values can be calculated.
>>> simulation.expectation_value("position")
np.float64(5.000002867138047)
>>> simulation.expectation_value("momentum")
np.float64(9.983189000651)
>>> simulation.expectation_value("kinetic_energy")
np.float64(50.08273993919016)
>>> simulation.expectation_value("potential_energy")
np.float64(0.0)

Calling the simulation object will run the simulation for a specified
number of time steps with a given time interval. Additionally,
simulation data is saved at specified intervals.
>>> simulation(
...     time_interval=0.01,
...     total_time_steps=1000,
...     save_step=1001,
...     save_directory="output",
... )  # doctest: +SKIP
>>> simulation.time  # doctest: +SKIP
9.999999999999831
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scipy import integrate

from wave_packet_dynamics.operators import HamiltonianOperator
from wave_packet_dynamics.operators import IdentityOperator
from wave_packet_dynamics.operators import KineticEnergyOperator
from wave_packet_dynamics.operators import MomentumOperator
from wave_packet_dynamics.operators import PositionOperator
from wave_packet_dynamics.operators import PotentialEnergyOperator
from wave_packet_dynamics.operators import TimeEvolutionOperator


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from wave_packet_dynamics import Grid
    from wave_packet_dynamics.particles import Particle
    from wave_packet_dynamics.potential import Potential
    from wave_packet_dynamics.wave_function import WaveFunction


class Simulation:
    """Simulation of a single particle in an electrostatic potential.

    Parameters
    ----------
    grid : Grid
        Grid on which the simulation is performed.
    particle : Particle
        The particle to simulate.
    wave_function : typing.Callable[[NDArray[np.float64]], NDArray[np.complex128]]
        The initial wave function of the particle.
    potential : typing.Callable[[NDArray[np.float64]], NDArray[np.float64]]
        The electrostatic potential acting on the particle.

    Attributes
    ----------
    time : float
        The current time of the simulation.

    """  # noqa: W505

    def __init__(
        self, *, grid: Grid, particle: Particle, wave_function: WaveFunction, potential: Potential
    ) -> None:
        self.time = 0.0
        self.grid = grid
        self.particle = particle
        self.wave_function = wave_function(grid.coordinates)
        self.potential = potential(grid.coordinates)
        self.normalize()

        # initialize operators
        self.operators = {
            "position": PositionOperator(self),
            "momentum": MomentumOperator(self),
            "kinetic_energy": KineticEnergyOperator(self),
            "potential_energy": PotentialEnergyOperator(self),
            "total_energy": HamiltonianOperator(self),
            "total_density": IdentityOperator(self),
        }

    def __call__(
        self,
        time_interval: float,
        total_time_steps: int,
        save_step: int = 10,
        save_directory: str | Path = "simulation",
    ) -> None:
        """Run the simulation.

        Parameters
        ----------
        time_interval : float
            Time interval between two consecutive time steps.
        total_time_steps : int
            Total number of time steps.
        save_step : int, optional
            Save interval in time steps.
        save_directory : str or Path, optional
            Output directory.

        Raises
        ------
        FileExistsError
            If the output directory already exists.
        """
        # create output directory and time evolution operator
        save_directory = Path(save_directory)
        save_directory.mkdir()
        u = TimeEvolutionOperator(self, time_interval)

        # start the simulation
        for time_step in range(total_time_steps):
            # save data and observables at specified intervals
            if time_step % save_step == 0:
                with (save_directory / "time.txt").open("a") as file:
                    file.write(f"{self.time}\n")
                for data in ["wave_function", "density"]:
                    np.savetxt(save_directory / f"{data}.txt", [getattr(self, data)])
                for observable in self.operators:
                    with (save_directory / f"{observable}.txt").open("a") as file:
                        file.write(f"{self.expectation_value(observable)}\n")
            # evolve the wave function and update the time
            self.wave_function = u @ self.wave_function
            self.time += time_interval

        # always save grid, potential and output wave function
        np.savetxt(save_directory / "grid.txt", [self.grid.coordinates])
        np.savetxt(save_directory / "potential.txt", [self.potential])
        np.savetxt(save_directory / "output.txt", [self.wave_function])

    @property
    def density(self) -> NDArray[np.float64]:
        r"""Calculate the probability density.

        Notes
        -----
        The probability density is defined as the absolute square of the
        wave function.

        $$|\psi(x)|^{2} = \psi^{\ast}(x) \psi(x) \quad \text{with} \quad
        |\psi(x)|^{2} \in \mathbb{R}_{\geq 0}$$
        """
        return np.real(self.wave_function.conjugate() * self.wave_function)

    def normalize(self) -> None:
        """Normalize the wave function."""
        self.wave_function /= integrate.trapezoid(self.density, dx=self.grid.spacing)

    def expectation_value(self, observable: str) -> float:
        r"""Calculate the expectation value of an observable.

        Notes
        -----
        The expectation value of an observable ($A$) is calculated using
        the corresponding self-adjoint linear operator ($\hat{A}$).

        $$\langle A \rangle_\psi = \langle \psi | \hat{A} | \psi \rangle
        = \int_{-\infty}^{\infty} \psi^{\ast}(x) \hat{A} \psi(x) \, dx$$
        """
        try:
            operator = self.operators[observable]
        except KeyError as error:
            msg = "Unknown observable."
            raise ValueError(msg) from error
        return integrate.trapezoid(  # type: ignore[no-any-return]
            np.real(self.wave_function.conjugate() * (operator @ self.wave_function)),
            dx=self.grid.spacing,
        )
