"""Submodule with predefined wave functions."""

import numpy as np


def gaussian(x: np.ndarray, sigma: float, x0: float, k0: float) -> np.ndarray:
    r"""
    Represents a wave packet with gaussian distribution and initial momentum.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Coordinate array for computation of function values.
    sigma : :class:`float`
        Full width at half maximum of the gaussian distribution.
    x0 : :class:`float`
        Initial most probable coordinate.
    k0 : :class:`float`
        Initial wave number of the matter wave.

    Returns
    -------
    psi : :class:`numpy.ndarray`
        Array with function values of the gaussian wave packet.

    Notes
    -----
    .. math::

        \Psi \left( x \right) = \left( 2 \pi \sigma ^2 \right) ^{-1/4}
        \cdot e^{ -\left( x-x_0 \right) ^2 / 4 \sigma ^2 }
        \cdot e^{ \text{i} k_0 x }

    """
    psi = (2 * np.pi * sigma ** 2) ** -0.25 * np.exp(-(x - x0) ** 2 / (4 * sigma ** 2)) * np.exp(1j * k0 * x)
    return psi
