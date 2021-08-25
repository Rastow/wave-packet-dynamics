"""Main module"""
import os
from typing import Callable, Iterator, List, Tuple, Union
from datetime import datetime
import numpy as np
from scipy.sparse import dia_matrix, identity, diags
from scipy.linalg import lapack
from matplotlib import pyplot as plt
from matplotlib import animation


class Grid:
    """
    Representation of a 1-dimensional grid.

    Attributes
    ----------
    dimension : `int`
        Grid dimension.
    bounds : `tuple` of `int`
        Tuple containing the upper and lower bounds of the grid.
    points : `int`
        Number of grid points used for discretization.
    """

    def __init__(self, bounds: Tuple[float, float], points: int):
        """
        Constructor of the `Grid` object. The dimension is set to 1 because higher dimensions are not yet supported.

        Parameters
        ----------
        bounds : `tuple` of `int`
            Tuple containing the upper and lower bounds of the grid.
        points : `int`
            Number of grid points used for discretization.
        """
        self.dimension = 1
        self.bounds = bounds
        self.points = points

    @property
    def coordinates(self) -> np.ndarray:
        """
        Coordinates of the grid points which are computed with numpy's linspace function.

        Returns
        -------
        coordinates : `numpy.ndarray`
            Coordinates of the grid points.
        """
        return np.linspace(*self.bounds, num=self.points)

    @property
    def spacing(self) -> float:
        """
        Spacing between the grid points which is computed with the grid bounds and the number of grid points.

        Returns
        -------
        spacing : `float`
            Spacing between the grid points.
        """
        return (self.bounds[1] - self.bounds[0]) / (self.points - 1)


class WaveFunction:
    """
    Representation of a particle wave function.

    Attributes
    ----------
    grid: `Grid`
        A `Grid` instance required for discretization of the function values.
    function: callable
        Function that acts on the `Grid` coordinate array to produce function values.
        - ``x`` : coordinate array (`numpy.ndarray`).
    mass : `int`
        Mass of the particle in atomic units.
    values: `numpy.ndarray`
        Array with discretized function values.
    """

    def __init__(self, grid: Grid, function: Callable, mass: float = 1):
        """
        Constructor of the `WaveFunction` object.

        Parameters
        ----------
        grid: `Grid`
            A `Grid` instance required for discretization of the function values.
        function: callable
            Function that acts on the `Grid` coordinate array to produce function values.
            - ``x`` : coordinate array (`numpy.ndarray`).
        mass : `int`, optional
            Mass of the particle in atomic units (default is 1, which is the electron mass).
        """
        self.grid = grid
        self.function = function
        self.mass = mass
        self.values = function(grid.coordinates)

    @property
    def probability_density(self):
        return np.real(self.values.conjugate() * self.values)

    def normalize(self):
        integral = integrate(self.probability_density, self.grid.spacing)
        self.values /= integral

    def expectation_value(self, operator) -> float:
        expectation_value = integrate(self.values.conjugate() * operator.dot(self.values), self.grid.spacing).real
        return expectation_value


class Potential:

    def __init__(self, grid: Grid, function: Callable):
        self.grid = grid
        self.function = function
        self.values = function(grid.coordinates)


class LinearOperator:

    def __init__(self, grid: Grid, matrix: dia_matrix):
        self.grid = grid
        self.matrix = matrix

    def dot(self, vector: np.ndarray) -> np.ndarray:
        return self.matrix.dot(vector)

    @property
    def shape(self) -> Tuple[int, int]:
        shape = (self.grid.points, self.grid.points)
        return shape

    @classmethod
    def scalar(cls, grid: Grid, scalar: Union[int, float, complex, np.ndarray]):
        shape = (grid.points, grid.points)
        matrix = diags(scalar, 0, shape=shape)
        return cls(grid, matrix)

    @classmethod
    def derivative(cls, grid: Grid):
        diagonals = np.array([-1, 1]) / (2 * grid.spacing)
        offsets = [-1, 1]
        shape = (grid.points, grid.points)
        matrix = diags(diagonals, offsets, shape=shape)
        return cls(grid, matrix)

    @classmethod
    def momentum(cls, grid: Grid):
        matrix = -1j * LinearOperator.derivative(grid).matrix
        return cls(grid, matrix)

    @classmethod
    def second_derivative(cls, grid: Grid):
        diagonals = np.array([1, -2, 1]) / (grid.spacing ** 2)
        offsets = [-1, 0, 1]
        shape = (grid.points, grid.points)
        matrix = diags(diagonals, offsets, shape=shape)
        return cls(grid, matrix)

    @classmethod
    def kinetic_energy(cls, grid: Grid, mass: float):
        matrix = -1 * np.reciprocal(2. * mass) * LinearOperator.second_derivative(grid).matrix
        return cls(grid, matrix)

    @classmethod
    def hamilton(cls, grid: Grid, mass: float, potential: np.ndarray):
        matrix = LinearOperator.kinetic_energy(grid, mass).matrix + LinearOperator.potential(grid, potential).matrix
        return cls(grid, matrix)

    @classmethod
    def time_evolution_lhs(cls, grid: Grid, time_increment: float, mass: float, potential: np.ndarray):
        matrix = identity(grid.points) + 0.5j * time_increment * LinearOperator.hamilton(grid, mass, potential).matrix
        return cls(grid, matrix)

    @classmethod
    def time_evolution_rhs(cls, grid: Grid, time_increment: float, mass: float, potential: np.ndarray):
        matrix = identity(grid.points) - 0.5j * time_increment * LinearOperator.hamilton(grid, mass, potential).matrix
        return cls(grid, matrix)

    # operator aliases
    potential = scalar
    position = scalar
    integrated_density = scalar


class Simulation:

    def __init__(self, grid: Grid, wave_function: WaveFunction, potential: Potential, time_increment: float,
                 name: str = 'simulation'):
        self.grid = grid
        self.wave_function = wave_function
        self.potential = potential
        self.time_increment = time_increment
        self.name = name

        # create a dictionary which contains linear operators associated to their expectation value
        self.operator_dictionary = {
            "potential_energy": LinearOperator.potential(self.grid, self.potential.values),
            "kinetic_energy": LinearOperator.kinetic_energy(self.grid, self.wave_function.mass),
            "total_energy": LinearOperator.hamilton(self.grid, self.wave_function.mass, self.potential.values),
            "momentum": LinearOperator.momentum(self.grid),
            "position": LinearOperator.position(self.grid, self.grid.coordinates),
            "integrated_density": LinearOperator.integrated_density(self.grid, np.ones_like(self.wave_function.values)),
        }

    def __call__(self, total_time_steps: int, **kwargs):
        # create a new directory
        time_stamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        directory_name = '_'.join([self.name, time_stamp])
        os.mkdir(directory_name)

        # change working directory to new directory
        working_directory = os.getcwd()
        os.chdir(os.path.join(working_directory, directory_name))

        # start the matplotlib animation
        print("Starting simulation:")
        try:
            self.animate(total_time_steps, **kwargs)
        finally:
            print("Simulation finished!")
            # change working directory back to original directory
            os.chdir("..")

    def frame_sequence(self, total_time_steps, data_to_write: List[str], value_to_write: List[str],
                       write_step: int, animate_step: int) -> Iterator[int]:
        # create operators needed for time evolution
        args = [self.grid, self.time_increment, self.wave_function.mass, self.potential.values]
        lhs_operator = LinearOperator.time_evolution_lhs(*args).matrix
        rhs_operator = LinearOperator.time_evolution_rhs(*args).matrix

        # extract les coefficients a, b, c
        a = lhs_operator.diagonal(-1)
        b = lhs_operator.diagonal(0)
        c = lhs_operator.diagonal(1)

        # iterate until the last time step is reached
        for time_step in range(1, total_time_steps + 1):
            # update les coefficient d
            d = rhs_operator.dot(self.wave_function.values)

            # pass les coefficients to thomas algorithm to evolve wave_function
            self.wave_function.values = thomas_algorithm(a, b, c, d)

            # check if data needs to be written to a file
            if time_step % write_step == 0:
                print(f"Writing data from time step {time_step} to file")
                for item in data_to_write:
                    filename = f"{item}.csv"
                    data = self.get_data(item)
                    with open(filename, "a") as file:
                        np.savetxt(file, data, fmt='%s', delimiter=',', newline='')
                        file.write("\n")
                for item in value_to_write:
                    filename = f"{item}.csv"
                    value = self.get_value(item)
                    with open(filename, "a") as file:
                        file.write(f"{value}\n")

            # only yield time steps which should be animated
            if time_step % animate_step == 0:
                print(f"Adding data from time step {time_step} to animation")
                yield time_step

    def animate(self, total_time_steps: int, **kwargs):
        # process keyword arguments
        data_to_animate = kwargs.pop('data_to_animate', [])
        animate_step = kwargs.pop('animate_step', total_time_steps)
        data_to_write = kwargs.pop('data_to_write', [])
        value_to_write = kwargs.pop('value_to_write', [])
        write_step = kwargs.pop('write_step', total_time_steps)

        x_ticks = kwargs.pop('x_ticks', ())
        y_ticks = kwargs.pop('y_ticks', ())
        z_ticks = kwargs.pop('z_ticks', ())

        x_limits = kwargs.pop('x_limits', self.grid.bounds)
        y_limits = kwargs.pop('y_limits', (-1, 1))
        z_limits = kwargs.pop('z_limits', (-1, 1))

        fps = kwargs.pop('fps', 20)
        dpi = kwargs.pop('dpi', 200)
        bitrate = kwargs.pop('bitrate', 2500)

        # set up a generator for matplotlib animation
        frames = self.frame_sequence(total_time_steps, data_to_write, value_to_write, write_step, animate_step)

        # set up figure and axes
        fig = plt.figure()
        fig.set_size_inches(8, 4.5)
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

        # set view direction of 3 dimensional plot
        ax.view_init(elev=11, azim=102)

        # set axes properties: labels, ticks, dimensions
        ax.set_xlabel('x')
        ax.set_ylabel('Im')
        ax.set_zlabel('Re')

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_zticks(z_ticks)

        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_zlim(z_limits)

        # generate line objects
        lines = []
        for item in data_to_animate:
            new_line = ax.plot([], [], [], lw=2, label=item)[0]
            lines.append(new_line)

        # initialisation: set up base-line canvas
        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        # function to update the line objects
        def update():

            x = self.grid.coordinates
            y = []
            z = []

            for item in data_to_animate:
                data = self.get_data(item)
                y.append(data.imag)
                z.append(data.real)

            for index, line in enumerate(lines):
                line.set_data(x, y[index])
                line.set_3d_properties(z[index])

            plt.legend()

            return lines

        # create the animation
        # frames which are supposed to be animated are 'lazily' generated by a given generator
        # overwrite the unreasonable save_count of 100 frames when using a generator
        anim = animation.FuncAnimation(fig=fig,
                                       func=update,
                                       frames=frames,
                                       init_func=init,
                                       save_count=total_time_steps,
                                       blit=True,
                                       repeat=False)

        # save the animation as mp4 (ffmpeg has to be installed)
        anim.save(f"animation.mp4", writer="ffmpeg", fps=fps, dpi=dpi, bitrate=bitrate)

        # delete the animation if it contains one or no frames
        if total_time_steps / animate_step <= 1:
            print("Deleting animation!")
            os.remove("animation.mp4")

    def get_data(self, name: str):
        if name == 'wave_function':
            return self.wave_function.values
        elif name == 'probability_density':
            return self.wave_function.probability_density
        elif name == 'potential':
            return self.potential.values
        else:
            raise ValueError(f"Unknown quantity '{name}'")

    def get_value(self, name: str):
        try:
            operator = self.operator_dictionary[name]
        except KeyError as error:
            raise ValueError(f"Unknown expectation value '{name}'") from error
        else:
            exp_val = self.wave_function.expectation_value(operator)
            return exp_val


def integrate(function_values: np.ndarray, grid_spacing: float) -> float:
    return np.sum(function_values) * grid_spacing


def thomas_algorithm(a, b, c, d) -> np.ndarray:
    return lapack.cgtsv(a, b, c, d, 1, 1, 1, 1)[3]
