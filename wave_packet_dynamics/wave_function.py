"""Submodule with predefined wave functions."""
import numpy as np


def gaussian(x: np.ndarray, sigma: float, x0: float, k0: float) -> np.ndarray:
    """
    Represents a wave packet with gaussian distribution and initial momentum.

    .. math::

        f\left( x \right)=\left( 2\pi\sigma^{ 2 } \right)^{ -1/4 }
        \cdot e^{ -\left( x-x_{ 0 } \right) ^{ 2 }/4
        \cdot \sigma^{ 2 }}\cdot e^{i\cdot k_{ 0 } x}

    :param numpy.ndarray x: coordinate array for computation of function values
    :param float sigma: full width at half maximum of the gaussian distribution
    :param float x0: initial most probable coordinate
    :param float k0: initial wave number of the matter wave

    :return: array with function values of the gaussian wave packet
    :rtype: numpy.ndarray
    """
    return (2 * np.pi * sigma ** 2) ** -0.25 * np.exp(-(x - x0) ** 2 / (4 * sigma ** 2)) * np.exp(1j * k0 * x)
