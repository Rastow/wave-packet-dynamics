"""Grid module.

This module provides a class for creation of one-dimensional uniform
grids for later discretization of continuous spatial functions.

Examples
--------
>>> import wave_packet_dynamics as wpd
>>> grid = wpd.Grid(bounds=(0.0, 1.0), points=5)
>>> grid.coordinates
array([0.  , 0.25, 0.5 , 0.75, 1.  ])
>>> grid.spacing
0.25
"""

import numpy as np

from numpy.typing import NDArray


class Grid:
    """One-dimensional uniform grid.

    Parameters
    ----------
    bounds : tuple[float, float]
        Grid boundaries.
    points : int
        Number of grid points.
    """

    def __init__(self, bounds: tuple[float, float], points: int) -> None:
        self.bounds = bounds
        self.points = points

    @property
    def coordinates(self) -> NDArray[np.float64]:
        """Spatial coordinates of the grid points."""
        return np.linspace(self.bounds[0], self.bounds[1], self.points)

    @property
    def spacing(self) -> float:
        r"""Spacing between grid points.

        Notes
        -----
        $$\Delta x = \frac{ x_{\text{max}}-x_{\text{min}} }
        {N_\text{points}-1} $$
        """
        return (self.bounds[1] - self.bounds[0]) / (self.points - 1)
