"""Submodule with predefined potential functions."""

import numpy as np


def harmonic(x: np.ndarray, k: float, x0: float) -> np.ndarray:
    """
    Represents a harmonic (parabola) potential function.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Coordinate array for computation of function values.
    k : :class:`float`
        Force constant of the harmonic potential.
    x0 : :class:`float`
        Coordinate of the potential valley (zero point).

    Returns
    -------
    v : :class:`numpy.ndarray`
        Array with function values of the harmonic potential.

    Notes
    -----
    .. math:: V\\left( x \\right) = 0.5k \\, \\left( x-x_0 \\right) ^2
    """
    return 0.5 * k * (x - x0) ** 2


def wall(x: np.ndarray, height: float, width: float, x0: float) -> np.ndarray:
    r"""
    Represents a (rectangular) hard wall potential step function.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Coordinate array for computation of function values.
    height : :class:`float`
        Height :math:`h` of the potential step.
    width : :class:`float`
        Width :math:`w` of the potential step.
    x0 : :class:`float`
        Coordinate of the potential center.

    Returns
    -------
    v : :class:`numpy.ndarray`
        Array with function values of the wall potential.

    Notes
    -----
    .. math::

        V \left( x \right) =
        \begin{cases}
        0 & \text{for} & x_0 - w/2 > & x & \\
        h & \text{for} & x_0 - w/2 < & x & < x_0 + w/2 \\
        0 & \text{for} &             & x & > x_0 + w/2 \\
        \end{cases}

    """
    v = np.empty_like(x)
    for index, coord in enumerate(x):
        if x0 - width / 2 < coord < x0 + width / 2:
            v[index] = height
        else:
            v[index] = 0
    return v
