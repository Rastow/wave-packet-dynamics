"""Visualization module.

After running calculations using the `Simulation` class, visualization
of the results is often desired. This module provides two classes for
plotting the simulation data. The `Animation` class creates an animated
three-dimensional plot of the complex-valued wave function alongside
the probability density. The `Plot` class creates a series of
two-dimensional plots for the recorded expectation values.

Examples
--------
Use the following commands to create an animation of data in a
directory `simulation/`. You have full control over figure and axes,
enabling you to set the figure size and the viewing angle for example.
>>> animation = Animation("simulation/")  # doctest: +SKIP
>>> animation.fig.set_size_inches(10, 10)  # doctest: +SKIP
>>> animation.ax.view_init(elev=50, azim=30)  # doctest: +SKIP
>>> animation.animation.save("animation.mp4")  # doctest: +SKIP

Run the following commands to create a series of plots of the
expectation values in a directory `simulation/`. Again, you can
customize the figure and axes to your liking. For example, you can
add vertical lines to indicate specific points in time or change the
color of specific lines.
>>> plot = Plot("simulation/")  # doctest: +SKIP
>>> plot.axs["position"].axvline(x=5, linestyle="--")  # doctest: +SKIP
>>> plot.lines["total_energy"].set_color("red")  # doctest: +SKIP
>>> plot.fig.savefig("plot.png")  # doctest: +SKIP
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from mpl_toolkits.mplot3d.art3d import Line3D
    from mpl_toolkits.mplot3d.axes3d import Axes3D


class Animation:
    """Helper class for animating simulation data.

    Animates the time evolution of the complex-valued wave function and
    the real-valued probability density in a three-dimensional
    coordinate system. Additionally, the simulation time is displayed
    and the (rescaled) electrostatic potential is plotted as a guide
    for the eye.

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
        Matplotlib text artist. Displays the current time.
    animation : FuncAnimation
        Matplotlib animation object.
    """

    labels: ClassVar[dict[str, str]] = {
        "wave_function": r"$\Psi$",
        "density": r"$|\Psi|^2$",
        "potential": r"$V$",
    }

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
            name: self.ax.plot([], [], [], label=label)[0] for name, label in self.labels.items()
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


class Plot:
    """Helper class for plotting expectation values.

    Plots the expectation values of the total density, position,
    momentum, kinetic energy, potential energy, and total energy as
    functions of the simulation time. Each expectation value is
    displayed in a separate subplot.

    Parameters
    ----------
    directory : str or Path
        Directory containing the simulation data.

    Attributes
    ----------
    fig : Figure
        Matplotlib figure.
    axs : dict[str, Axes]
        Dictionary of two-dimensional axes.
    lines: list[Line2D]
        Dictionary of two-dimensional line artists.
    """

    labels: ClassVar[dict[str, str]] = {
        "total_density": r"$\langle \Psi | \Psi \rangle$",
        "position": r"$\langle x \rangle_\Psi$",
        "momentum": r"$\langle p \rangle_\Psi$",
        "kinetic_energy": r"$\langle T \rangle_\Psi$",
        "potential_energy": r"$\langle V \rangle_\Psi$",
        "total_energy": r"$\langle H \rangle_\Psi$",
    }

    def __init__(self, directory: str | Path) -> None:
        # attempt to load required simulation data
        directory = Path(directory)
        try:
            self.time = np.loadtxt(directory / "time.txt", dtype=np.float64)
            self.expectation_values = {
                observable: np.loadtxt(directory / f"{observable}.txt", dtype=np.float64)
                for observable in self.labels
            }
        except FileNotFoundError as error:
            msg = "Directory contains incomplete simulation data."
            raise FileNotFoundError(msg) from error

        # plot the expectation values in different subplots
        self.fig: Figure = plt.figure()
        self.axs: dict[str, Axes] = {}
        self.lines: dict[str, Line2D] = {}
        for index, (name, values) in enumerate(self.expectation_values.items(), start=1):
            self.axs[name] = self.fig.add_subplot(2, 3, index)
            self.axs[name].plot(self.time, values)
            self.axs[name].set_xlabel(r"$t$")
            self.axs[name].set_ylabel(self.labels[name])
            self.axs[name].ticklabel_format(useMathText=True)

        # set reasonable axis limits
        max_energy = np.max(self.expectation_values["total_energy"])
        for name in ["potential_energy", "kinetic_energy", "total_energy"]:
            self.axs[name].set_ylim(bottom=0, top=float(max_energy) * 1.1)

        # adjust layout to prevent overlapping labels
        self.fig.tight_layout()
