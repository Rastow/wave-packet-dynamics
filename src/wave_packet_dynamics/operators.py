"""Operators module."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from scipy.linalg import lapack
from scipy.sparse.linalg import LinearOperator


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from wave_packet_dynamics import Simulation


class SelfAdjointOperator(LinearOperator, ABC):  # type: ignore[misc]
    """Self-adjoint operator."""

    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation
        super().__init__(
            shape=(self.simulation.grid.points, self.simulation.grid.points), dtype=np.float64
        )

    @abstractmethod
    def _matvec(self, x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Matrix-vector product."""
        ...

    def _rmatvec(self, x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Adjoint matrix-vector product."""
        return self._matvec(x)


class IdentityOperator(SelfAdjointOperator):
    """Identity operator."""

    def _matvec(self, x: NDArray[np.complex128]) -> NDArray[np.complex128]:  # noqa: PLR6301
        return x


class PositionOperator(SelfAdjointOperator):
    """Position operator."""

    def _matvec(self, x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return self.simulation.grid.coordinates * x


class MomentumOperator(SelfAdjointOperator):
    """Momentum operator."""

    def _matvec(self, x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        y = np.zeros_like(x)
        f = -1j / self.simulation.grid.spacing
        y[0] = f * x[1]
        y[1:-1] = 0.5 * f * (x[2:] - x[:-2])
        y[-1] = f * -1 * x[-2]
        return y


class KineticEnergyOperator(SelfAdjointOperator):
    """Kinetic energy operator."""

    def _matvec(self, x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        y = np.zeros_like(x)
        f = -1 / (2 * self.simulation.particle.mass * self.simulation.grid.spacing**2)
        y[0] = f * (-2 * x[0] + x[1])
        y[1:-1] = f * (x[:-2] - 2 * x[1:-1] + x[2:])
        y[-1] = f * (x[-2] - 2 * x[-1])
        return y


class PotentialEnergyOperator(SelfAdjointOperator):
    """Potential energy operator."""

    def _matvec(self, x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return -self.simulation.particle.charge * self.simulation.potential * x


class HamiltonianOperator(SelfAdjointOperator):
    """Hamiltonian operator."""

    def _matvec(self, x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        y = np.zeros_like(x)
        f = -1 / (2 * self.simulation.particle.mass * self.simulation.grid.spacing**2)
        y[0] = f * (-2 * x[0] + x[1]) + self.simulation.potential[0] * x[0]
        y[1:-1] = f * (x[:-2] - 2 * x[1:-1] + x[2:]) + self.simulation.potential[1:-1] * x[1:-1]
        y[-1] = f * (x[-2] - 2 * x[-1]) + self.simulation.potential[-1] * x[-1]
        return y


class TimeEvolutionOperator(SelfAdjointOperator):
    """Time evolution operator.

    Attributes
    ----------
    a : NDArray[np.complex128]
        Lower diagonal of the tridiagonal matrix.
    b : NDArray[np.complex128]
        Main diagonal of the tridiagonal matrix.
    c : NDArray[np.complex128]
        Upper diagonal of the tridiagonal matrix.
    """

    def __init__(self, simulation: Simulation, time_interval: float) -> None:
        self.simulation = simulation
        self.time_interval = time_interval
        self.b = 2 * (
            self.simulation.particle.mass
            * self.simulation.grid.spacing**2
            * (
                2j / self.time_interval
                - -self.simulation.particle.charge * self.simulation.potential
            )
            - 1
        )
        self.a = np.ones((self.simulation.grid.points - 1,), dtype=np.complex128)
        self.c = np.ones((self.simulation.grid.points - 1,), dtype=np.complex128)
        super().__init__(simulation)

    def _matvec(self, x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Matrix-vector product.

        First calculates the right-hand-side vector `d` and then solves
        the tridiagonal system of equations using the LAPACK routine
        `zgtsv`.
        """
        d = np.zeros_like(x)
        d[1:-1] = (
            -x[:-2]
            + 2
            * (
                self.simulation.particle.mass
                * self.simulation.grid.spacing**2
                * (
                    2j / self.time_interval
                    + -self.simulation.particle.charge * self.simulation.potential[1:-1]
                )
                + 1
            )
            * x[1:-1]
            - x[2:]
        )
        return lapack.zgtsv(self.a, self.b, self.c, d)[3]  # type: ignore[no-any-return]
