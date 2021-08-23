"""Module with predefined wave functions."""
import numpy as np


def gaussian(x: np.ndarray, sigma: float, x0: float, k0: float) -> np.ndarray:
    """
    Represents a wave packet with gaussian distribution and initial momentum.

    :param np.ndarray x: initial coordinate array which should be used for computing the function values
    :param float sigma: full width at half maximum of the gaussian distribution
    :param float x0: initial most probable coordinate
    :param float k0: initial wave number of the matter wave

    :return: array with function values of the gaussian wave packet
    :rtype: np.ndarray
    """
    return (2 * np.pi * sigma ** 2) ** -0.25 * np.exp(-(x - x0) ** 2 / (4 * sigma ** 2)) * np.exp(1j * k0 * x)
