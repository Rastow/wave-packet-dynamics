"""Wave packet dynamics package.

This package provides a framework for simulating the time evolution of
a single wave packet in an arbitrary time-independent potential.
Currently, only one-dimensional simulations are supported.
"""

from wave_packet_dynamics import operators
from wave_packet_dynamics import particles
from wave_packet_dynamics import potential
from wave_packet_dynamics import wave_function
from wave_packet_dynamics.grid import Grid
from wave_packet_dynamics.simulation import Simulation
from wave_packet_dynamics.visualization import Animation
from wave_packet_dynamics.visualization import Plot


__all__ = [
    "Animation",
    "Grid",
    "Plot",
    "Simulation",
    "operators",
    "particles",
    "potential",
    "wave_function",
]
__version__ = "2.0.0"
