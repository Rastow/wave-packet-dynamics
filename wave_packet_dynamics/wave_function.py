"""Submodule with predefined wave functions."""
import numpy as np


def gaussian(x: np.ndarray, sigma: float, x0: float, k0: float) -> np.ndarray:
    r"""
    Represents a wave packet with gaussian distribution and initial momentum.

    .. math::

        \Psi \left( x \right) = \left( 2 \pi \sigma ^2 \right) ^{-1/4}
        \cdot e^{ -\left( x-x_0 \right) ^2 / 4 \sigma ^2 }
        \cdot e^{ \text{i} k_0 x }

    :param x: Coordinate array for computation of function values.
    :param sigma: Full width at half maximum of the gaussian distribution.
    :param x0: Initial most probable coordinate.
    :param k0: Initial wave number of the matter wave.

    :return: Array with function values of the gaussian wave packet.
    """
    return (2 * np.pi * sigma ** 2) ** -0.25 * np.exp(-(x - x0) ** 2 / (4 * sigma ** 2)) * np.exp(1j * k0 * x)
