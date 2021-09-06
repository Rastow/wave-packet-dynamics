"""Main module with most functionality. The contents of this submodule are loaded when importing the package."""
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

    :var dimension: The dimension is set to 1 because higher dimensions are not yet supported.
    :vartype dimension: :class:`int`

    :param bounds: Tuple containing the upper and lower bounds of the grid.
    :param points: Number of grid points used for discretization.
    """

    dimension = 1

    def __init__(self, bounds: Tuple[float, float], points: int):
        """
        Constructor of the :class:`Grid` object.

        :param bounds: Tuple containing the upper and lower bounds of the grid.
        :param points: Number of grid points used for discretization.
        """
        self.bounds = bounds
        self.points = points

    @property
    def coordinates(self) -> np.ndarray:
        """
        Coordinates of the grid points.

        The coordinates are provided as an array which is computed with :func:`numpy.linspace`, resulting in linear \
        spacing between the grid points. The endpoint is included.

        :return: Coordinates of the grid points.
        """
        return np.linspace(*self.bounds, num=self.points)

    @property
    def spacing(self) -> float:
        """
        Spacing between the grid points.

        The spacing between the grid points is linear and the endpoint is included.

        :return: Spacing between the grid points.
        """
        return (self.bounds[1] - self.bounds[0]) / (self.points - 1)


class WaveFunction:
    """
    Representation of a particle wave function.

    :param grid: :class:`Grid` instance required for discretization of the function values.
    :param function: Function that acts on the :class:`Grid` coordinates to produce function values.
    :param mass: Mass of the particle in atomic units, default is 1 which is the mass of an electron.

    :var values: Array with discretized function values.
    :vartype values: :class:`numpy.ndarray`
    """

    def __init__(self, grid: Grid, function: Callable, mass: float = 1):
        """
        Constructor of the :class:`WaveFunction` object. Function values are calculated.

        :param grid: :class:`Grid` instance required for discretization of the function values.
        :param function: Function that acts on the :class:`Grid` instance's coordinate array to produce function \
            values.
        :param mass: Mass of the particle in atomic units, default is 1 which is the mass of an electron.
        """
        self.grid = grid
        self.function = function
        self.mass = mass
        self.values = function(grid.coordinates)

    @property
    def probability_density(self) -> np.ndarray:
        """
        Probability density of the particle.

        The probability density is computed by multiplying the conjugate wave function values with the ordinary wave \
        function values. Additionally, the imaginary part of the result is discarded, because in theory it should be \
        zero.

        :return: Spatial probability distribution of the particle.
        """
        return np.real(self.values.conjugate() * self.values)

    def normalize(self):
        """
        Normalizes the wave function.

        First, the integral over all space is computed and then the wave function values are divided by the integral \
        value. Uses the function :func:`integrate` to compute the integral.
        """
        integral = integrate(self.probability_density, self.grid.spacing)
        self.values /= integral

    def expectation_value(self, operator) -> float:
        """
        Calculates the expectation value of an observable.

        The observable :math:`A` is obtained by evaluating the matrix element :math:`\\langle \\Psi | \\hat{A} | \\Psi \
        \\rangle`. :math:`\\hat{A}` is the :class:`LinearOperator` associated to the observable, it is provided as a \
        parameter. Precisely, the dot product of the linear operator matrix with the wave function values are \
        computed, then they are multiplied with the conjugate wave function values from the left and lastly \
        integrated over all space.

        :param operator: :class:`LinearOperator` instance associated to the observable which should be determined. The \
            operator matrix must match the array with the wave function values.
        :type operator: :class:`LinearOperator`

        :return: Expectation value of the specified quantum operator.
        """
        expectation_value = integrate(self.values.conjugate() * operator.dot(self.values), self.grid.spacing).real
        return expectation_value


class Potential:
    """
    Representation of a time-independent potential function.

    :param grid: :class:`Grid` instance required for discretization of the function values.
    :param function: Function that acts on the :class:`Grid` coordinates to produce function values.

    :var values: Array with discretized function values.
    :vartype values: :class:`numpy.ndarray`
    """

    def __init__(self, grid: Grid, function: Callable):
        """
        Constructor of the :class:`Potential` object. Function values are calculated.

        :param grid: :class:`Grid` instance required for discretization of the function values.
        :param function: Function that acts on the :class:`Grid` instance's coordinate array to produce function \
            values.
        """
        self.grid = grid
        self.function = function
        self.values = function(grid.coordinates)


class LinearOperator:
    """
    Representation of a linear operator which should act on discretized :class:`WaveFunction` values.

    :param grid: :class:`Grid` instance required for construction of the matrix representation.
    :param matrix: Sparse diagonal matrix which represents the linear operator.
    :type matrix: :class:`scipy.sparse.dia_matrix`
    """

    def __init__(self, grid: Grid, matrix: dia_matrix):
        """
        Constructor of the :class:`Potential` object. Function values are calculated.

        :param grid: :class:`Grid` instance required for construction of the matrix representation.
        :param matrix: Sparse diagonal matrix which represents the linear operator.
        :type matrix: :class:`scipy.sparse.dia_matrix`
        """
        self.grid = grid
        self.matrix = matrix

    def dot(self, vector: np.ndarray) -> np.ndarray:
        """
        Computes the action of the :class:`LinearOperator` on discretized :class:`WaveFunction` values.

        Uses :func:`scipy.sparse.dia_matrix.dot` to calculate this matrix vector dot product.

        :param vector: Coordinate array with function values, the vector.
        :return: Matrix vector product.
        """
        return self.matrix.dot(vector)

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Shape of the :class:`LinearOperator` matrix.

        :return: Tuple containing the two matrix dimensions of the :class:`LinearOperator` matrix representation.
        """
        shape = (self.grid.points, self.grid.points)
        return shape

    @classmethod
    def scalar(cls, grid: Grid, scalar: Union[int, float, complex, np.ndarray]):
        """
        Creates a :class:`LinearOperator` instance whose matrix representation scales function values.

        The matrix only contains elements on the main diagonal. Uses :func:`scipy.sparse.diags` to create the matrix,
        meaning when single values are specified as a scalar, they will be broadcast along the main diagonal.

        :param grid: :class:`Grid` instance.
        :param scalar: Scalar which scales the function values.
        :return: Scalar :class:`LinearOperator`.
        """
        shape = (grid.points, grid.points)
        matrix = diags(scalar, 0, shape=shape)
        return cls(grid, matrix)

    @classmethod
    def derivative(cls, grid: Grid):
        """
        Creates a :class:`LinearOperator` instance whose matrix representation derives function values.

        It is a finite difference matrix which contains the central finite difference coefficients on the two \
        diagonals above and below the main diagonal. The derivative is of order 1 and the accuracy is of order 2. \
        Beware that the edges are not of higher order, boundary effects can occur. Therefore it is important \
        for the function values to be zero at the edges.

        :param grid: :class:`Grid` instance.
        :return: First derivative :class:`LinearOperator`.
        """
        diagonals = np.array([-1, 1]) / (2 * grid.spacing)
        offsets = [-1, 1]
        shape = (grid.points, grid.points)
        matrix = diags(diagonals, offsets, shape=shape)
        return cls(grid, matrix)

    @classmethod
    def momentum(cls, grid: Grid):
        """
        Creates a :class:`LinearOperator` instance which resembles the quantum momentum operator.

        The matrix representation is generated analogous to the equation :math:`\\hat{p} = -\\text{i} \\hbar \\nabla`.

        See also :func:`LinearOperator.derivative`.

        :param grid: :class:`Grid` instance.
        :return: Momentum :class:`LinearOperator`.
        """
        matrix = -1j * LinearOperator.derivative(grid).matrix
        return cls(grid, matrix)

    @classmethod
    def second_derivative(cls, grid: Grid):
        """
        Creates a :class:`LinearOperator` instance whose matrix representation calculates the second derivative \
        of function values when acting on them.

        It is a finite difference matrix which contains the central finite difference coefficients on the main \
        diagonals and the two diagonals above and below the main diagonal. The derivative is of order 2 and the \
        accuracy is of order 2. Beware that the edges are not of higher order, boundary effects can occur. Therefore \
        it is important for the function values to be zero at the edges.

        :param grid: :class:`Grid` instance.
        :return: Second derivative :class:`LinearOperator`.
        """
        diagonals = np.array([1, -2, 1]) / (grid.spacing ** 2)
        offsets = [-1, 0, 1]
        shape = (grid.points, grid.points)
        matrix = diags(diagonals, offsets, shape=shape)
        return cls(grid, matrix)

    @classmethod
    def kinetic_energy(cls, grid: Grid, mass: float):
        """
        Creates a :class:`LinearOperator` instance which resembles the quantum kinetic energy operator.

        The matrix representation is generated analogous to the equation :math:`\\hat{T} = - \\frac{ \\hbar ^2 } \
        { 2m } \\nabla ^2`.

        See also :func:`LinearOperator.second_derivative`.

        :param grid: :class:`Grid` instance.
        :param mass: Mass of the particle described by the wave function.
        :return: Kinetic energy :class:`LinearOperator`.
        """
        matrix = -1 * np.reciprocal(2. * mass) * LinearOperator.second_derivative(grid).matrix
        return cls(grid, matrix)

    @classmethod
    def hamilton(cls, grid: Grid, mass: float, potential: np.ndarray):
        """
        Creates a :class:`LinearOperator` instance which resembles the quantum hamilton operator.

        The matrix representation is generated analogous to the equation :math:`\\hat{H} = - \\frac{ \\hbar ^2 } \
        { 2m } \\nabla ^2 + \\hat{V}`.

        See also :func:`LinearOperator.kinetic_energy` and :func:`LinearOperator.potential`.

        :param grid: :class:`Grid` instance.
        :param mass: Mass of the particle described by the wave function.
        :param potential: Discretized time-independent potential acting on the particle.
        :return: Hamilton :class:`LinearOperator`.
        """
        matrix = LinearOperator.kinetic_energy(grid, mass).matrix + LinearOperator.potential(grid, potential).matrix
        return cls(grid, matrix)

    @classmethod
    def time_evolution_lhs(cls, grid: Grid, time_increment: float, mass: float, potential: np.ndarray):
        """
        Creates a :class:`LinearOperator` instance which resembles the left hand side time evolution operator.

        The matrix representation is generated analogous to the equation :math:`\\hat{U}_{ \\text{LHS} } = \\hat{1} + \
        \\frac{ \\text{i} \\hat{H} \\Delta t }{ 2 \\hbar }`.

        See also :func:`LinearOperator.hamilton`.

        :param grid: :class:`Grid` instance.
        :param time_increment: Time increment specified for the time evolution steps.
        :param mass: Mass of the particle described by the wave function.
        :param potential: Discretized time-independent potential acting on the particle.
        :return: Left hand side time evolution operator :class:`LinearOperator`.
        """
        matrix = identity(grid.points) + 0.5j * time_increment * LinearOperator.hamilton(grid, mass, potential).matrix
        return cls(grid, matrix)

    @classmethod
    def time_evolution_rhs(cls, grid: Grid, time_increment: float, mass: float, potential: np.ndarray):
        """
        Creates a :class:`LinearOperator` instance which resembles the right hand side time evolution operator.

        The matrix representation is generated analogous to the equation :math:`\\hat{U}_{ \\text{RHS} } = \\hat{1} - \
        \\frac{ \\text{i} \\hat{H} \\Delta t }{ 2 \\hbar }`.

        See also :func:`LinearOperator.hamilton`.

        :param grid: :class:`Grid` instance.
        :param time_increment: Time increment specified for the time evolution steps.
        :param mass: Mass of the particle described by the wave function.
        :param potential: Discretized time-independent potential acting on the particle.
        :return: Right hand side time evolution operator :class:`LinearOperator`.
        """
        matrix = identity(grid.points) - 0.5j * time_increment * LinearOperator.hamilton(grid, mass, potential).matrix
        return cls(grid, matrix)

    @classmethod
    def potential(cls, grid: Grid, potential: np.ndarray):
        """
        Creates a :class:`LinearOperator` instance which resembles the quantum potential energy operator.

        When the matrix acts on function values it scales them by the value of the potential.

        See also :func:`LinearOperator.scalar`.

        :param grid: :class:`Grid` instance.
        :param potential: Discretized time-independent potential acting on the particle.
        :return: Potential :class:`LinearOperator`.
        """
        matrix = LinearOperator.scalar(grid, potential).matrix
        return cls(grid, matrix)

    @classmethod
    def position(cls, grid: Grid):
        """
        Creates a :class:`LinearOperator` instance which resembles the quantum position operator.

        When the matrix acts on function values it scales them by the grid coordinate values.

        See also :func:`LinearOperator.scalar`.

        :param grid: :class:`Grid` instance.
        :return: Position :class:`LinearOperator`.
        """
        matrix = LinearOperator.scalar(grid, grid.coordinates).matrix
        return cls(grid, matrix)

    @classmethod
    def integrated_density(cls, grid: Grid):
        """
        Creates an identity :class:`LinearOperator` instance.

        It is used for calculating the probability of finding the particle somewhere in space. For this an identity \
        matrix is generated with :func:`scipy.sparse.identity`.

        :param grid: :class:`Grid` instance.
        :return: Identity :class:`LinearOperator`.
        """
        matrix = identity(grid.points)
        return cls(grid, matrix)


class Simulation:
    """
    Class for simulation of the wave packet dynamics.

    Creating an instance of this class by providing a :class:`Grid`, :class:`WaveFunction` and :class:`Potential` \
    object alongside a desired :data:`time_increment` primes the simulation. The simulation is executed by calling \
    the instance, see :meth:`Simulation.__call__`. This animates the wave packet propagation and writes data to a file.

    :param grid: :class:`Grid` required for creating linear operators.
    :param wave_function: The :class:`WaveFunction` which should be evolved.
    :param potential: The time-independent :class:`Potential` acting on the particle.
    :param time_increment: Desired time increment for each simulation step in atomic units.
    :param name: Name of the simulation, defaults to 'simulation'.
    """

    def __init__(self, grid: Grid, wave_function: WaveFunction, potential: Potential, time_increment: float,
                 name: str = 'simulation'):
        """
        Constructor of the :class:`Simulation` object.

        :param grid: :class:`Grid` required for creating linear operators.
        :param wave_function: The :class:`WaveFunction` which should be evolved.
        :param potential: The time-independent :class:`Potential` acting on the particle.
        :param time_increment: Desired time increment for each simulation step in atomic units.
        :param name: Name of the simulation, defaults to 'simulation'.
        """
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
            "position": LinearOperator.position(self.grid),
            "integrated_density": LinearOperator.integrated_density(self.grid),
        }

    def __call__(self, total_time_steps: int, **kwargs):
        """
        Start the simulation.

        The simulation is executed by calling the instance and providing the run time in form of an integer \
        :data:`total_time_steps`. A unique folder will be created in the working directory when executing the \
        simulation, which will contain all associated files. Specifying additional keyword arguments when calling \
        the simulation changes what/how data will be represented in the animation and what data will be written to \
        files.

        :param total_time_steps: Run time of the animation provided as a number of total time steps.
        :param kwargs: Optional keyword arguments.
        """
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
        # preform clean-up duties
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
        def update(time_step):

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
