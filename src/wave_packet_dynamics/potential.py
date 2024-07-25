"""Potential module.

This module contains predefined potential functions that can be used to
generate electrostatic potentials for the simulation. The potential
functions take a coordinate array as input and return the corresponding
potential energy values.

Examples
--------
>>> import wave_packet_dynamics as wpd
>>> x = wpd.Grid((-1, 1), 11).coordinates
>>> V = wpd.potential.harmonic(x, k=5, x0=0)
>>> V
array([2.5, 1.6, 0.9, 0.4, 0.1, 0. , 0.1, 0.4, 0.9, 1.6, 2.5])

With the help of `functools.partial`, the potential function can be
converted into a `Callable` that only takes the coordinate array as an
argument.
>>> from functools import partial
>>> V = partial(wpd.potential.barrier, w=0.5, h=1)
>>> V(x)
array([0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0.])
"""

from collections.abc import Callable
from typing import TypeAlias

import numpy as np

from numpy.typing import NDArray


Potential: TypeAlias = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def zero(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Zero potential function.

    Parameters
    ----------
    x : NDArray[np.float64]
        Coordinate array.

    Returns
    -------
    NDArray[np.float64]
        Array of zeros.

    Notes
    -----
    $$V\left( x \right) = 0$$
    """
    return np.zeros_like(x)


def harmonic(x: NDArray[np.float64], *, k: float, x0: float = 0.0) -> NDArray[np.float64]:
    r"""Harmonic potential function.

    Parameters
    ----------
    x : NDArray[np.float64]
        Coordinate array.
    k : float
        Force constant of the harmonic potential.
    x0 : float
        Coordinate of the potential minimum.

    Returns
    -------
    NDArray[np.float64]
        Discretized harmonic potential.

    Notes
    -----
    $$V(x) = 0.5k \left( x-x_0 \right) ^2$$
    """
    return 0.5 * k * (x - x0) ** 2


def barrier(x: NDArray[np.float64], *, h: float, w: float, x0: float = 0.0) -> NDArray[np.float64]:
    r"""Rectangular barrier potential function.

    Parameters
    ----------
    x : NDArray[np.float64]
        Coordinate array.
    h : float
        Height of the potential.
    w : float
        Width of the potential.
    x0 : float
        Center of the potential.

    Returns
    -------
    NDArray[np.float64]
        Discretized rectangular barrier potential.

    Notes
    -----
    $$V(x)= \begin{cases}
    0 &\text{for} & &x &\le x_0 - w / 2 \\
    h &\text{for} &x_0 - w / 2 \le &x &\le x_0 + w / 2 \\
    0 &\text{for} &x_0 + w / 2 \le &x &
    \end{cases}$$
    """
    return np.where((x0 - w / 2 <= x) & (x <= x0 + w / 2), h, 0.0)


def step(x: NDArray[np.float64], *, h: float, x0: float = 0.0) -> NDArray[np.float64]:
    r"""Step potential function.

    Parameters
    ----------
    x : NDArray[np.float64]
        Coordinate array.
    h : float
        Height of the potential.
    x0 : float
        Coordinate of the step.

    Returns
    -------
    NDArray[np.float64]
        Discretized step potential.

    Notes
    -----
    $$V(x)= \begin{cases}
    0 &\text{for} & &x &\le x_0\\
    h &\text{for} &x_0 \le &x &
    \end{cases}$$
    """
    return np.where(x <= x0, h, 0.0)
