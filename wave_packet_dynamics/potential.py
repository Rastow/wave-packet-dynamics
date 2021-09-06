"""Submodule with predefined potential functions."""
import numpy as np


def harmonic(x: np.ndarray, k: float, x0: float) -> np.ndarray:
    """
    Represents a harmonic (parabola) potential.

    .. math::

        V\\left( x \\right) = 0.5k \\, \\left( x-x_0 \\right) ^2

    :param x: Coordinate array for computation of function values.
    :param k: Force constant of the harmonic potential.
    :param x0: Coordinate of the potential valley (zero point).

    :return: Array with function values of the harmonic potential.
    """
    return 0.5 * k * (x - x0) ** 2


def wall(x: np.ndarray, height: float, width: float, x0: float) -> np.ndarray:
    r"""
    Represents a (rectangular) hard wall potential step.

    .. math::

        V \left( x \right) =
        \begin{cases}
        0 & \text{for} & x_0 - w/2 > & x & \\
        h & \text{for} & x_0 - w/2 < & x & < x_0 + w/2 \\
        0 & \text{for} &             & x & > x_0 + w/2 \\
        \end{cases}

    :param x: Coordinate array for computation of function values.
    :param height: Height :math:`h` of the potential step.
    :param width: Width :math:`w` of the potential step.
    :param x0: Coordinate of the potential center.

    :return: Array with function values of the wall potential.
    """
    y = np.empty_like(x)
    for index, coord in enumerate(x):
        if x0 - width / 2 < coord < x0 + width / 2:
            y[index] = height
        else:
            y[index] = 0
    return y
