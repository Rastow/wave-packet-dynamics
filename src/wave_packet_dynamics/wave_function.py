r"""Wave function module.

This module contains continuous functions representing wave packets
suitable for simulation of quantum dynamics. The wave functions take
a coordinate array as input and return the corresponding discretized
complex-valued wave functions.

Examples
--------
>>> import wave_packet_dynamics as wpd
>>> x = wpd.Grid((-5, 5), 101).coordinates
>>> psi = wpd.wave_function.gaussian(x, sigma=1, x0=0, k0=10)
>>> x[50], psi[50]
(np.float64(0.0), np.complex128(0.6316187777460647+0j))

All wave functions are normalized.
>>> from scipy import integrate
>>> norm = integrate.trapezoid(np.real(psi.conj() * psi), x)
>>> norm
np.float64(0.9999994143527634)
"""

from collections.abc import Callable
from typing import TypeAlias

import numpy as np

from numpy.typing import NDArray


WaveFunction: TypeAlias = Callable[[NDArray[np.float64]], NDArray[np.complex128]]


def gaussian(
    x: NDArray[np.float64], *, sigma: float, x0: float = 0.0, k0: float = 0.0
) -> NDArray[np.complex128]:
    r"""Gaussian wave function.

    Parameters
    ----------
    x : NDArray[np.float64]
        Coordinate array.
    sigma : float
        Standard deviation of the Gaussian distribution.
    x0 : float
        Initial most probable coordinate.
    k0 : float
        Initial wave number of the particle.

    Returns
    -------
    NDArray[np.complex128]
        Discretized Gaussian wave function.

    Notes
    -----
    $$\Psi(x) = \left( 2 \pi \sigma ^2 \right) ^{-1/4}
    \cdot e^{ -\left( x-x_0 \right) ^2 / 4 \sigma ^2 }
    \cdot e^{ \text{i} k_0 \left( x-x_0 \right) }$$
    """
    return (  # type: ignore[no-any-return]
        (2 * np.pi * sigma**2) ** -0.25
        * np.exp(-((x - x0) ** 2) / (4 * sigma**2))
        * np.exp(1j * k0 * (x - x0))
    )
