"""Submodule with predefined potential functions."""
import numpy as np


def harmonic(x: np.ndarray, k: float, x0: float) -> np.ndarray:
    """
    Represents a harmonic (parabola) potential.

    :param numpy.ndarray x: coordinate array for computation of function values
    :param float k: force constant
    :param float x0: coordinate of the potential valley (zero point)

    :return: array with function values of the harmonic potential
    :rtype: numpy.ndarray
    """
    return 0.5 * k * (x - x0) ** 2


def wall(x: np.ndarray, height: float, width: float, x0: float) -> np.ndarray:
    """
    Represents a (rectangular) hard wall potential step.

    :param numpy.ndarray x: coordinate array for computation of function values
    :param float height: height of the potential step
    :param float width: width of the potential step
    :param float x0: coordinate of the potential center

    :return: array with function values of the wall potential
    :rtype: numpy.ndarray
    """
    y = np.empty_like(x)
    for index, coord in enumerate(x):
        if x0 - width / 2 < coord < x0 + width / 2:
            y[index] = height
        else:
            y[index] = 0
    return y
