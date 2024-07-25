"""Visualization module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from mpl_toolkits.mplot3d.art3d import Line3D
    from mpl_toolkits.mplot3d.axes3d import Axes3D


class Animation:
    """Helper class for animating the simulation data.

    Parameters
    ----------
    directory : str or Path
        Directory containing the simulation data.


    Attributes
    ----------
    fig : Figure
        Matplotlib figure.
    ax : Axes3D
        Three-dimensional axes.
    lines : dict[str, Line3D]
        Dictionary of three-dimensional line artists.
    text : Text
        Matplotlib text artist.
    animation : FuncAnimation
        Matplotlib animation object.
    """

    def __init__(self, directory: str | Path) -> None:
        # attempt to load required simulation data
        directory = Path(directory)
        try:
            self.grid = np.loadtxt(directory / "grid.txt", dtype=np.float64)
            self.time = np.loadtxt(directory / "time.txt", dtype=np.float64)
            self.potential = np.loadtxt(directory / "potential.txt", dtype=np.float64)
            self.wave_function = np.loadtxt(directory / "wave_function.txt", dtype=np.complex128)
            self.density = np.loadtxt(directory / "density.txt", dtype=np.float64)
        except FileNotFoundError as error:
            msg = "Directory contains incomplete simulation data."
            raise FileNotFoundError(msg) from error

        # calculate reasonable axes limit, rescale the potential
        self.limit = np.max(np.abs(self.density))
        max_potential = np.max(np.abs(self.potential))
        if self.limit < max_potential < np.inf:
            self.potential *= self.limit / max_potential

        # initialize the figure, axes and artists
        self.fig: Figure = plt.figure()
        self.ax: Axes3D = self.fig.add_subplot(projection="3d")
        self.text: Text = self.ax.text2D(
            0.1, 0.9, f"t = {self.time[0]:.2f}", transform=self.ax.transAxes
        )
        self.lines: dict[str, Line3D] = {
            "wave_function": self.ax.plot([], [], [], label=r"$\Psi$")[0],
            "density": self.ax.plot([], [], [], label=r"$|\Psi|^2$")[0],
            "potential": self.ax.plot([], [], [], label=r"$V$")[0],
        }

        self.animation = FuncAnimation(
            self.fig,
            func=self._update,
            init_func=self._initialize,
            save_count=len(self.time),
            blit=True,
        )

    def _initialize(self) -> list[Line3D | Text]:
        """Initialize the artists."""
        self.ax.set_xlim(self.grid[0], self.grid[-1])
        self.ax.set_ylim(-self.limit, self.limit)
        self.ax.set_zlim(-self.limit, self.limit)

        self.ax.set_xlabel(r"$x$")
        self.ax.set_ylabel(r"$\mathfrak{Im}$")
        self.ax.set_zlabel(r"$\mathfrak{Re}$")
        self.ax.legend()

        for name in ["potential"]:
            self.lines[name].set_xdata(self.grid)
            self.lines[name].set_ydata(getattr(self, name).imag)
            self.lines[name].set_3d_properties(getattr(self, name).real)
        return [*list(self.lines.values()), self.text]

    def _update(self, frame: int) -> list[Line3D | Text]:
        """Update the artists."""
        self.text.set_text(f"t={self.time[frame]:.2f}")
        for name in ["wave_function", "density"]:
            self.lines[name].set_xdata(self.grid)
            self.lines[name].set_ydata(getattr(self, name)[frame].imag)
            self.lines[name].set_3d_properties(getattr(self, name)[frame].real)
        return [*list(self.lines.values()), self.text]
