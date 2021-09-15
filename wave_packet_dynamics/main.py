"""Main module with most functionality.

The contents of this submodule are loaded when importing the package.
"""
import os
from typing import Callable, List, Tuple, Union
from datetime import datetime
from abc import abstractmethod
from scipy.sparse import csr_matrix, dia_matrix, diags, identity
from scipy.sparse.linalg import spsolve
import numpy as np
from findiff import FinDiff
from timeit import default_timer
from functools import cached_property


class Grid:
    """Class representation of a 1-dimensional grid.

    Parameters
    ----------
    bounds : :class:`Tuple` of :class:`float`
        Tuple containing the upper and lower bounds of the grid.
    points : :class:`int`
        Number of grid points used for discretization.
    """

    def __init__(self, bounds: Tuple[float, float], points: int):
        self.bounds = bounds
        self.points = points

    @property
    def coordinates(self) -> np.ndarray:
        """Coordinates of the grid points.

        The coordinates are provided as an array computed with :func:`numpy.linspace`, resulting in linear \
        spacing between the grid points. The endpoint is included.

        Returns
        -------
        coordinates : :class:`numpy.ndarray`
            Coordinates of the grid points.
        """
        return np.linspace(*self.bounds, num=self.points)

    @property
    def spacing(self) -> float:
        """Spacing between the grid points.

        The spacing between the grid points is linear and the endpoint is included.

        Returns
        -------
        spacing : :class:`float`
            Spacing between the grid points.

        Notes
        -----
        If :math:`x_{ \\text{min} }` is the lower bound, :math:`x_{ \\text{max} }` is the upper bound and :math:`N` \
        is the total number of grid points, the grid spacing :math:`\\Delta x` is calculated with following equation:

        .. math:: \\Delta x = \\frac{ x_{ \\text{max} } - x_{ \\text{min} } }{ N - 1 }
        """
        return (self.bounds[1] - self.bounds[0]) / (self.points - 1)


class WaveFunction:
    """Class representation of a particle wave function.

    Parameters
    ----------
    grid : :class:`Grid`
        :class:`Grid` instance required for discretization of the function values.
    function : :class:`Callable`
        Function that acts on the :attr:`Grid.coordinates` to produce function values.
    mass : :class:`float`, default=1
        Mass of the particle in atomic units, default being 1 which is the mass of an electron.

    Attributes
    ----------
    values : :class:`numpy.ndarray`
        Array with discretized function values.
    """

    def __init__(self, grid: Grid, function: Callable, mass: float = 1):
        self.grid = grid
        self.function = function
        self.mass = mass
        self.values = function(grid.coordinates)

    @property
    def probability_density(self) -> np.ndarray:
        """Probability density :math:`\\rho \\left( x \\right)` of the particle.

        Returns
        -------
        probability_density : :class:`numpy.ndarray`
            Spatial probability distribution of the particle.

        Notes
        -----
        The probability density is computed analogous to the following equation:

        .. math:: \\rho \\left( x \\right) = \\left| \\Psi \\left( x \\right) \\right| ^2 = \\Psi ^{ \\ast } \\Psi

        The imaginary part of the result is discarded, because in theory it should be zero.
        """
        return np.real(self.values.conjugate() * self.values)

    def normalize(self):
        """Normalizes the wave function.

        First, the integral over all space is computed with :func:`integrate`. \
        Then the wave function values are divided by the integral value.
        """
        integral = integrate(self.probability_density, self.grid.spacing)
        self.values /= integral

    def expectation_value(self, operator: 'LinearOperator') -> float:
        """Calculates the expectation value :math:`\\langle A \\rangle` of an observable :math:`A`.

        Precisely, the matrix vector product of the linear operator's matrix representation and the state vector \
        (discretized wave function values) is computed. Then, the product is multiplied with the conjugate wave \
        function values from the left. The expectation value is obtained by integrating over all space.

        Parameters
        ----------
        operator : :class:`LinearOperator`
            Quantum operator associated to the observable which should be determined. The operator's matrix \
            representation must match the state vector (wave function).

        Returns
        -------
        expectation_value : :class:`float`
            Expectation value of the specified observable.

        Notes
        -----
        The expectation value :math:`\\langle A \\rangle` of an observable :math:`A` is obtained by evaluating \
        following matrix element:

        .. math::

            \\langle A \\rangle = \\langle \\Psi | \\hat{A} | \\Psi \\rangle =
            \\int _{ - \\infty } ^{ + \\infty } \\Psi ^{ \\ast } \\hat{A} \\Psi \\, d \\tau

        :math:`\\hat{A}` is the quantum operator associated to the observable :math:`A`. It must be a \
        :class:`LinearOperator`. In order to obtain real eigenvalues, the operator must also be hermitian.
        """
        expectation_value = integrate(self.values.conjugate() * operator.map(self).values, self.grid.spacing).real
        return expectation_value


class Potential:
    """Class representation of a time-independent potential function.

    Parameters
    ----------
    grid : :class:`Grid`
        :class:`Grid` instance required for discretization of the function values.
    function : :class:`Callable`
        Function that acts on the :attr:`Grid.coordinates` to produce function values.

    Attributes
    ----------
    values : :class:`numpy.ndarray`
        Array with discretized function values.
    """

    def __init__(self, grid: Grid, function: Callable):
        self.grid = grid
        self.function = function
        self.values = function(grid.coordinates)


class LinearOperator:
    """Class representation of a linear operator.

    Quantum operators inherit most methods from this class.

    Parameters
    ----------
    grid : :class:`Grid`
        The grid defines the basis of the linear operator. It determines the physical states (wave functions) the \
        operator may act on.
    """

    def __init__(self, grid: Grid):
        self.grid = grid

    def map(self, vector: WaveFunction) -> WaveFunction:
        """Applies the linear operator to a state vector (wave function).

        First, compatability is asserted. Then, the matrix vector product is calculated.

        Parameters
        ----------
        vector : :class:`WaveFunction`
            Physical state to be mapped.

        Returns
        -------
        vector : :class:`WaveFunction`
            Linear transformation of the state vector.

        Raises
        ------
        ValueError
            If the wave function is not compatible with the :class:`LinearOperator` instance.
        """
        if self.assert_compatibility(vector) is False:
            raise ValueError("Grid of vector and linear operator do not match!")
        vector.values = self.matrix.dot(vector.values)
        return vector

    def assert_compatibility(self, vector: Union[WaveFunction, Potential]) -> bool:
        """
        Checks if a vector is compatible with the linear operator.

        Uses :func:`numpy.array_equal` to verify that the grid coordinate arrays are equal.

        Parameters
        ----------
        vector : :class:`WaveFunction` or :class:`Potential`
            Class with discretized function values, must also contain the underlying :class:`Grid` instance as an \
            attribute.

        Returns
        -------
        compatibility : :class:`bool`
            Returned value is ``True`` if the grid coordinates match and ``False`` otherwise.
        """
        if np.array_equal(vector.grid.coordinates, self.grid.coordinates):
            return True
        else:
            return False

    @property
    @abstractmethod
    def matrix(self):
        """Matrix representation of the linear operator.

        .. note::
            This is an abstract method. No default implementation is provided because the matrix representation \
            depends on the underlying quantum operator.

        Raises
        ------
        NotImplementedError
            If this method is not implemented by a subclass.
        """
        raise NotImplementedError


class PositionOperator(LinearOperator):
    """Class representation of the position operator :math:`\\hat{x}`."""

    @property
    def matrix(self) -> dia_matrix:
        """Matrix representation of the position operator :math:`\\hat{x}`.

        Returns
        -------
        matrix : :class:`scipy.sparse.dia.dia_matrix`
            Sparse matrix containing the grid coordinate values on the main diagonal.

        Notes
        -----
        Uses :func:`diags` from :mod:`scipy.sparse` to generate the scalar matrix.
        """
        shape = (self.grid.points, self.grid.points)
        matrix = diags(self.grid.coordinates, 0, shape=shape)
        return matrix


class PotentialEnergyOperator(LinearOperator):
    """Class representation of the potential energy operator :math:`\\hat{V}`.

    Parameters
    -----------------
    potential : :class:`Potential`
        Time-independent external potential, required for the matrix representation.
    """

    def __init__(self, grid: Grid, potential: Potential):
        super().__init__(grid)
        self.potential = potential

    @property
    def matrix(self) -> dia_matrix:
        """Matrix representation of the potential energy operator :math:`\\hat{V}`.

        Returns
        -------
        matrix : :class:`scipy.sparse.dia.dia_matrix`
            Sparse matrix containing the values of the potential on the main diagonal.

        Raises
        ------
        ValueError
            If the grid of the potential doesn't match the grid of the :class:`LinearOperator` instance.

        Notes
        -----
        Uses :func:`diags` from :mod:`scipy.sparse` to generate the scalar matrix.
        """
        if self.assert_compatibility(self.potential) is False:
            raise ValueError("Grids of potential and linear operator do not match!")
        shape = (self.grid.points, self.grid.points)
        matrix = diags(self.potential.values, 0, shape=shape)
        return matrix


class MomentumOperator(LinearOperator):
    """Class representation of the momentum operator :math:`\\hat{p}`.

    Other Parameters
    ----------------
    accuracy : :class:`int`, default=2
        Order of accuracy in the grid spacing of the finite difference scheme. By default, :mod:`findiff` uses second \
        order accuracy.

    Notes
    -----
    .. math:: \\hat{p} = -\\text{i} \\hbar \\nabla
    """

    def __init__(self, grid: Grid, **kwargs):
        super().__init__(grid)
        self.accuracy = kwargs.get("accuracy", 2)

    @property
    def matrix(self) -> csr_matrix:
        """Matrix representation of the momentum operator :math:`\\hat{p}`.

        Returns
        -------
        matrix : :class:`scipy.sparse.csr.csr_matrix`
            Sparse matrix containing the first derivative finite difference coefficients multiplied with \
            :math:`- \\text{i}`.

        Notes
        -----
        Uses :class:`FinDiff` from :mod:`findiff` to create the necessary matrix with finite difference coefficients, \
        assumes a homogeneous grid with even spacing. For further information refer to the :mod:`findiff` package and \
        its documentation.
        """
        first_derivative = FinDiff(0, self.grid.spacing, 1, acc=self.accuracy).matrix(self.grid.coordinates.shape)
        matrix = -1j * first_derivative
        return matrix


class KineticEnergyOperator(LinearOperator):
    """Class representation of the kinetic energy operator :math:`\\hat{T}`.

    Parameters
    ----------
    mass : :class:`float`
        Mass of the particle, required for the matrix representation.

    Other Parameters
    ----------------
    accuracy : :class:`int`, default=2
        Order of accuracy in the grid spacing of the finite difference scheme. By default, :mod:`findiff` uses second \
        order accuracy.

    Notes
    -----
    .. math:: \\hat{T} = - \\frac{ \\hbar ^2 }{ 2m } \\nabla ^2
    """

    def __init__(self, grid: Grid, mass: float, **kwargs):
        super().__init__(grid)
        self.mass = mass
        self.accuracy = kwargs.get("accuracy", 2)

    @property
    def matrix(self) -> csr_matrix:
        """Matrix representation of the kinetic energy operator :math:`\\hat{T}`.

        Returns
        -------
        matrix : :class:`scipy.sparse.csr.csr_matrix`
            Sparse matrix containing the second derivative finite difference coefficients multiplied with \
            :math:`- \\frac{ \\hbar ^2 }{ 2m }`.

        Notes
        -----
        Uses :class:`FinDiff` from :mod:`findiff` to create the necessary matrix with finite difference coefficients, \
        assumes a homogeneous grid with even spacing. For further information refer to the :mod:`findiff` package and \
        its documentation.
        """
        second_derivative = FinDiff(0, self.grid.spacing, 2, acc=self.accuracy).matrix(self.grid.coordinates.shape)
        matrix = -1 * np.reciprocal(2. * self.mass) * second_derivative
        return matrix


class Hamiltonian(LinearOperator):
    """Class representation of the Hamiltonian :math:`\\hat{H}`.

    Parameters
    ----------
    mass : :class:`float`
        Mass of the particle, required for the matrix representation.
    potential : :class:`Potential`
        Time-independent external potential, required for the matrix representation.

    Other Parameters
    ----------------
    accuracy : :class:`int`, default=2
        Order of accuracy in the grid spacing of the finite difference scheme. By default, :mod:`findiff` uses second \
        order accuracy.

    Notes
    -----
    .. math:: \\hat{H} = \\hat{T} + \\hat{V} = - \\frac{ \\hbar ^2 }{ 2m } \\nabla ^2 + \\hat{V}
    """

    def __init__(self, grid: Grid, mass: float, potential: Potential, **kwargs):
        super().__init__(grid)
        self.mass = mass
        self.potential = potential
        self.accuracy = kwargs.get("accuracy", 2)

    @property
    def matrix(self) -> csr_matrix:
        """Matrix representation of the Hamiltonian :math:`\\hat{H}`.

        Returns
        -------
        matrix : :class:`scipy.sparse.csr.csr_matrix`
            Sparse matrix containing the second derivative finite difference coefficients multiplied with \
            :math:`- \\frac{ \\hbar ^2 }{ 2m }`, the values of the external potential :math:`V` are added on top \
            of the coefficients of the main diagonal.

        Notes
        -----
        Uses :class:`KineticEnergyOperator` and :class:`PotentialEnergyOperator` to create the necessary matrix \
        representations of these operators.
        """
        matrix = KineticEnergyOperator(self.grid, self.mass, accuracy=self.accuracy).matrix \
            + PotentialEnergyOperator(self.grid, self.potential).matrix
        return matrix


class IdentityOperator(LinearOperator):
    """Class representation of the identity operator :math:`\\hat{1}`."""

    @property
    def matrix(self) -> dia_matrix:
        """Matrix representation of the identity operator :math:`\\hat{1}`.

        Returns
        -------
        matrix : :class:`scipy.sparse.dia.dia_matrix`
            Sparse matrix containing :math:`1` on the main diagonal.

        Notes
        -----
        Uses :func:`identity` from :mod:`scipy.sparse` to generate the identity matrix.
        """
        matrix = identity(self.grid.points)
        return matrix


class TimeEvolutionOperator(LinearOperator):
    """Class representation of the time evolution operator :math:`\\hat{U} \\left( \\Delta t \\right)`.

    The exponential operator :math:`e^{ - \\text{i} \\hat{H} \\Delta t / \\hbar }` is replaced by its diagonal \
    :math:`\\left[ 1 / 1 \\right]` Padé approximant. The resulting operator is unitary and conserves wave function \
    normalization and time reversibility.

    Parameters
    ----------
    mass : :class:`float`
        Mass of the particle, required for the matrix representation.
    potential : :class:`Potential`
        Time-independent external potential, required for the matrix representation.
    time_increment : :class:`float` or :class:`complex`
        Time interval between simulation steps in atomic units.

    Other Parameters
    ----------------
    accuracy : :class:`int`, default=2
        Order of accuracy in the grid spacing of the finite difference scheme. By default, :mod:`findiff` uses second \
        order accuracy.

    Notes
    -----

    .. math::

        \\hat{U} \\left( \\Delta t \\right) = e^{ - \\text{i} \\hat{H} \\Delta t / \\hbar }
        \\approx \\frac{ \\hat{1} - \\text{i} \\hat{H} \\Delta t / 2 \\hbar }
        { \\hat{1} + \\text{i} \\hat{H} \\Delta t / 2 \\hbar } + \\mathcal{O} \\left( \\Delta t \\right) ^3

    """

    def __init__(self, grid: Grid, mass: float, potential: Potential, time_increment: Union[float, complex], **kwargs):
        super().__init__(grid)
        self.mass = mass
        self.potential = potential
        self.time_increment = time_increment
        self.accuracy = kwargs.get("accuracy", 2)

    def map(self, vector: WaveFunction) -> WaveFunction:
        r"""Applies the time evolution operator to a state vector.

        The linear transformation cannot be computed through a simple matrix vector product because the time \
        evolution operator's matrix exponential is replaced by its diagonal :math:`\left[ 1 / 1 \right]` Padé \
        approximant.

        .. warning:: Does not assert compatability!

        Parameters
        ----------
        vector : :class:`WaveFunction`
            Initial state vector (wave function).

        Returns
        -------
        vector : :class:`WaveFunction`
            Time evolved wave function.

        Notes
        -----
        The linear mapping is achieved by solving following linear equation system.

        .. math::

            \Psi \left( x, t + \Delta t \right) &= \hat{U} \left( \Delta t \right) \Psi \left( x, t \right) \\
            \hat{U}_{ \text{Denominator} } \, \Psi \left( x, t + \Delta t \right) &=
            \hat{U}_{ \text{Numerator} } \, \Psi \left( x, t \right)

        """
        # matrix a is already known
        a = self.matrix[1]

        # vector b is computed via the ordinary matrix vector product
        b = self.matrix[0].dot(vector.values)

        # solve the linear equation system Ax=b using scipy
        vector.values = spsolve(a, b)
        return vector

    @property
    def matrix(self) -> Tuple[csr_matrix, csr_matrix]:
        r"""Matrix representation of the time evolution operator :math:`\hat{U} \left( \Delta t \right)`.

        Precisely, the matrix representations of the numerator and the denominator of the time evolution operator's \
        diagonal Padé approximant are returned because the inverse of the Hamiltonian cannot be calculated.

        Returns
        -------
        numerator_matrix, denominator_matrix : :class:`tuple` of :class:`scipy.sparse.csr.csr_matrix`
            Sparse matrices containing the matrix representations of the numerator and the denominator of the \
            approximated time evolution operator.

        Notes
        -----

        .. math::

            \hat{U} _{ \text{Numerator} } = \hat{1} - \frac{ \text{i} \hat{H} \Delta t }{ 2 \hbar } \\
            \hat{U} _{ \text{Denominator} } = \hat{1} + \frac{ \text{i} \hat{H} \Delta t }{ 2 \hbar }

        Uses :class:`Hamiltonian` to create the necessary matrix representation of the time-independent Hamiltonian \
        operator. Additionally uses :func:`identity` from :mod:`scipy.sparse` to generate the identity matrix.
        """
        numerator_matrix = identity(self.grid.points) - 0.5j * self.time_increment \
            * Hamiltonian(self.grid, self.mass, self.potential, accuracy=self.accuracy).matrix
        denominator_matrix = identity(self.grid.points) + 0.5j * self.time_increment \
            * Hamiltonian(self.grid, self.mass, self.potential, accuracy=self.accuracy).matrix
        return numerator_matrix, denominator_matrix


class Simulation:

    def __init__(self, grid: Grid, wave_function: WaveFunction, potential: Potential, time_increment: float, **kwargs):
        self.grid = grid
        self.wave_function = wave_function
        self.potential = potential
        self.time_increment = time_increment

        # process keyword arguments
        self.accuracy = kwargs.get("accuracy", 2)

        self.operators = {
            "total_density": IdentityOperator(self.grid),
            "position": PositionOperator(self.grid),
            "potential_energy": PotentialEnergyOperator(self.grid, self.potential),
            "momentum": MomentumOperator(self.grid, accuracy=self.accuracy),
            "kinetic_energy": KineticEnergyOperator(self.grid, self.wave_function.mass, accuracy=self.accuracy),
            "total_energy": Hamiltonian(self.grid, self.wave_function.mass, self.potential, accuracy=self.accuracy),
        }

    def __call__(self, total_time_steps: int, **kwargs):
        # process optional keyword arguments
        name = kwargs.get("name", "simulation")
        write_step = kwargs.get("write_step", total_time_steps)
        data_objects = kwargs.get("data_objects", None)
        expectation_values = kwargs.get("expectation_values", None)

        # create a new directory whose name includes a unique time stamp
        time_stamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        directory_name = '_'.join([name, time_stamp])
        os.mkdir(directory_name)

        # change the working directory to the new directory
        working_directory = os.getcwd()
        os.chdir(os.path.join(working_directory, directory_name))

        # create the time evolution operator
        operator_args = [self.grid, self.wave_function.mass, self.potential, self.time_increment]
        time_evo_op = TimeEvolutionOperator(*operator_args, accuracy=self.accuracy)

        # start the simulation
        print("Starting simulation...")
        start = default_timer()
        time = 0
        try:
            for time_step in range(0, total_time_steps):
                # check if something needs to be written to a file
                if time_step % write_step == 0:
                    self._write_to_file(time, data_objects, expectation_values)

                # evolve the wave function
                self.wave_function = time_evo_op.map(self.wave_function)

                # update the simulation time
                time += self.time_increment
        # perform clean up duties even if the simulation fails
        finally:
            end = default_timer()
            elapsed = round(end - start, 5)
            print(f"Simulation finished after {elapsed} seconds!")
            # change working directory back to original directory
            os.chdir("..")

    def _get_data(self, item: str) -> np.ndarray:
        available_data = {
            "wave_function": self.wave_function.values,
            "wave_function_real": self.wave_function.values.real,
            "wave_function_imag": self.wave_function.values.imag,
            "probability_density": self.wave_function.probability_density,
            "potential": self.potential.values,
        }
        try:
            data = available_data[item]
        except KeyError as error:
            raise ValueError(f"Cannot find reference to '{item}'!") from error
        else:
            return data

    def _get_expectation_value(self, observable: str) -> float:
        try:
            operator = self.operators[observable]
        except KeyError as error:
            raise ValueError(f"Cannot find operator corresponding to '{observable}'!") from error
        else:
            expectation_value = self.wave_function.expectation_value(operator)
            return expectation_value

    def _write_to_file(self, time: complex, data_objects: List[str], expectation_values: List[str]):
        # append current simulation time in atomic units to the time file
        with open("time.txt", "a") as file:
            file.write(f"{time}\n")

        # append all data to their corresponding files
        for item in data_objects:
            filename = f"{item}.txt"
            data = self._get_data(item)
            with open(filename, "ab") as file:
                np.savetxt(file, [data], fmt="%.3e", delimiter=",")

        # append all expectation values to their corresponding files
        for observable in expectation_values:
            filename = f"{observable}.txt"
            exp_val = self._get_expectation_value(observable)
            with open(filename, "a") as file:
                file.write(f"{exp_val}\n")


def integrate(function_values: np.ndarray, grid_spacing: float) -> float:
    return np.sum(function_values) * grid_spacing
