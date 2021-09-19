"""Main module with most functionality.

The contents of this submodule are loaded when importing the package.
"""
from typing import Callable, Dict, List, Tuple, Union
from abc import abstractmethod

from scipy.sparse import csr_matrix, dia_matrix, diags, identity
from scipy.sparse.linalg import spsolve
import numpy as np
from findiff import FinDiff

import os
from datetime import datetime
from timeit import default_timer


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
        self.values: np.ndarray = function(grid.coordinates)

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
        (wave function) is computed. Then, the product is multiplied with the complex conjugate wave function values \
        from the left. The expectation value is obtained by integrating over all space.

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
        expectation_value = integrate(self.values.conjugate() * operator.map(self), self.grid.spacing).real
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

    def map(self, vector: WaveFunction) -> np.ndarray:
        """Applies the linear operator to a state vector (wave function).

        First, compatability is asserted. Then, the matrix vector product is calculated.

        .. note::

            This method expects a :class:`WaveFunction` object instead of a function value array since compatability \
            has to be asserted. However, keep in mind that an array containing the mapped function values is returned.

        Parameters
        ----------
        vector : :class:`WaveFunction`
            Physical state to be mapped.

        Returns
        -------
        transformed_vector : :class:`numpy.ndarray`
            Linear transformation of the state vector.

        Raises
        ------
        ValueError
            If the wave function is not compatible with the :class:`LinearOperator` instance.
        """
        if self.assert_compatibility(vector) is False:
            raise ValueError("Grid of vector and linear operator do not match!")
        transformed_vector = self.matrix.dot(vector.values)
        return transformed_vector

    def assert_compatibility(self, vector: Union[WaveFunction, Potential]) -> bool:
        """
        Checks if a vector is compatible with the linear operator.

        Uses :func:`numpy.array_equal` to verify that the grid coordinate arrays are equal.

        Parameters
        ----------
        vector : :class:`WaveFunction` or :class:`Potential`
            Class with discretized function values, must also contain the underlying :class:`Grid` as an \
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

        Uses :func:`diags` from :mod:`scipy.sparse` to generate the scalar matrix.

        Returns
        -------
        matrix : :class:`scipy.sparse.dia.dia_matrix`
            Sparse matrix containing the grid coordinate values on the main diagonal.
        """
        shape = (self.grid.points, self.grid.points)
        matrix = diags(self.grid.coordinates, 0, shape=shape)
        return matrix


class PotentialEnergyOperator(LinearOperator):
    """Class representation of the potential energy operator :math:`\\hat{V}`.

    Parameters
    -----------------
    potential : :class:`Potential`
        Time-independent external potential.
    """

    def __init__(self, grid: Grid, potential: Potential):
        super().__init__(grid)
        self.potential = potential

    @property
    def matrix(self) -> dia_matrix:
        """Matrix representation of the potential energy operator :math:`\\hat{V}`.

        Uses :func:`diags` from :mod:`scipy.sparse` to generate the scalar matrix.

        Returns
        -------
        matrix : :class:`scipy.sparse.dia.dia_matrix`
            Sparse matrix containing the function values of the potential on the main diagonal.

        Raises
        ------
        ValueError
            If the grid of the potential doesn't match the grid of the :class:`LinearOperator` instance.
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

        Uses :class:`FinDiff` from :mod:`findiff` to create the necessary matrix with finite difference coefficients, \
        assumes a homogeneous grid with even spacing. For further information refer to the :mod:`findiff` package and \
        its documentation.

        Returns
        -------
        matrix : :class:`scipy.sparse.csr.csr_matrix`
            Sparse matrix containing the first derivative finite difference coefficients multiplied with \
            :math:`- \\text{i}`.
        """
        first_derivative = FinDiff(0, self.grid.spacing, 1, acc=self.accuracy).matrix(self.grid.coordinates.shape)
        matrix = -1j * first_derivative
        return matrix


class KineticEnergyOperator(LinearOperator):
    """Class representation of the kinetic energy operator :math:`\\hat{T}`.

    Parameters
    ----------
    mass : :class:`float`
        Mass of the particle.

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

        Uses :class:`FinDiff` from :mod:`findiff` to create the necessary matrix with finite difference coefficients, \
        assumes a homogeneous grid with even spacing. For further information refer to the :mod:`findiff` package and \
        its documentation.

        Returns
        -------
        matrix : :class:`scipy.sparse.csr.csr_matrix`
            Sparse matrix containing the second derivative finite difference coefficients multiplied with \
            :math:`- \\frac{ \\hbar ^2 }{ 2m }`.
        """
        second_derivative = FinDiff(0, self.grid.spacing, 2, acc=self.accuracy).matrix(self.grid.coordinates.shape)
        matrix = -1 * np.reciprocal(2. * self.mass) * second_derivative
        return matrix


class Hamiltonian(LinearOperator):
    """Class representation of the Hamiltonian :math:`\\hat{H}`.

    Parameters
    ----------
    mass : :class:`float`
        Mass of the particle.
    potential : :class:`Potential`
        Time-independent external potential.

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

        Uses :class:`KineticEnergyOperator` and :class:`PotentialEnergyOperator` to create the necessary matrix \
        representations of these operators.

        Returns
        -------
        matrix : :class:`scipy.sparse.csr.csr_matrix`
            Sparse matrix containing the second derivative finite difference coefficients multiplied with \
            :math:`- \\frac{ \\hbar ^2 }{ 2m }`, the function values of the external potential :math:`V` are added \
            to the main diagonal.
        """
        matrix = KineticEnergyOperator(self.grid, self.mass, accuracy=self.accuracy).matrix \
            + PotentialEnergyOperator(self.grid, self.potential).matrix
        return matrix


class IdentityOperator(LinearOperator):
    """Class representation of the identity operator :math:`\\hat{1}`."""

    @property
    def matrix(self) -> dia_matrix:
        """Matrix representation of the identity operator :math:`\\hat{1}`.

        Uses :func:`identity` from :mod:`scipy.sparse` to generate the identity matrix.

        Returns
        -------
        matrix : :class:`scipy.sparse.dia.dia_matrix`
            Sparse matrix containing :math:`1` on the main diagonal.
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
        Mass of the particle.
    potential : :class:`Potential`
        Time-independent external potential.
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

    def map(self, vector: WaveFunction):
        r"""Applies the time evolution operator to a state vector.

        The linear transformation cannot be computed through a simple matrix vector product because the time \
        evolution operator's matrix exponential is replaced by its diagonal :math:`\left[ 1 / 1 \right]` Padé \
        approximant. Instead, the linear mapping is achieved by solving a linear equation system.

        Parameters
        ----------
        vector : :class:`WaveFunction`
            Initial wave function.

        Returns
        ----------
        transformed_vector : :class:`numpy.ndarray`
            Evolved wave function.

        Raises
        ------
        ValueError
            If the wave function is not compatible with the :class:`LinearOperator` instance.

        Notes
        -----

        .. math::

            \Psi \left( x, t + \Delta t \right) &= \hat{U} \left( \Delta t \right) & \Psi \left( x, t \right) \\
            \hat{U}_{ \text{Denominator} } \, \Psi \left( x, t + \Delta t \right) &=
            \hat{U}_{ \text{Numerator} } \, & \Psi \left( x, t \right)

        """
        if self.assert_compatibility(vector) is False:
            raise ValueError("Grid of vector and linear operator do not match!")

        # matrix a is already known
        a = self.matrix[1]

        # vector b is computed via the ordinary matrix vector product
        b = self.matrix[0].dot(vector.values)

        # solve the linear equation system Ax=b using scipy sparse linalg solver
        transformed_vector = spsolve(a, b)
        return transformed_vector

    @property
    def matrix(self) -> Tuple[csr_matrix, csr_matrix]:
        r"""Matrix representation of the time evolution operator :math:`\hat{U} \left( \Delta t \right)`.

        Precisely, the matrix representations of the numerator and the denominator of the time evolution operator's \
        diagonal :math:`\left[ 1 / 1 \right]` Padé approximant are returned because the inverse of the Hamiltonian \
        cannot be calculated.

        Uses :func:`identity` from :mod:`scipy.sparse` to generate the identity matrix. Additionally uses \
        :class:`Hamiltonian` to create the matrix representation of the time-independent Hamiltonian.

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

        """
        numerator_matrix = identity(self.grid.points) - 0.5j * self.time_increment \
            * Hamiltonian(self.grid, self.mass, self.potential, accuracy=self.accuracy).matrix
        denominator_matrix = identity(self.grid.points) + 0.5j * self.time_increment \
            * Hamiltonian(self.grid, self.mass, self.potential, accuracy=self.accuracy).matrix
        return numerator_matrix, denominator_matrix


class Simulation:

    _operator_class_dict = {
        "total_density": IdentityOperator,
        "position": PositionOperator,
        "momentum": MomentumOperator,
        "potential_energy": PotentialEnergyOperator,
        "kinetic_energy": KineticEnergyOperator,
        "total_energy": Hamiltonian,
    }

    def __init__(self, wave_function: WaveFunction, potential: Potential, time_increment: float, **kwargs):
        self.wave_function = wave_function
        self.potential = potential

        self.time_increment = time_increment
        self.accuracy_grid: int = kwargs.get("accuracy_grid", 2)
        self.accuracy_time: int = kwargs.get("accuracy_time", 2)

        self._operator_instance_dict: Dict[str, LinearOperator] = {}
        self._valid = False

        self.time_step: int = 0
        self.time: Union[float, complex] = 0.

    def __call__(self, total_time_steps: int, **kwargs):
        # process optional keyword arguments
        name: str = kwargs.get("name", "simulation")
        write_step: int = kwargs.get("write_step", total_time_steps)
        data_objects: List[str] = kwargs.get("data_objects", None)
        expectation_values: List[str] = kwargs.get("expectation_values", None)

        # create a new directory whose name includes a unique time stamp
        time_stamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        directory_name = '_'.join([name, time_stamp])
        os.mkdir(directory_name)

        # change the working directory to the new directory
        working_directory = os.getcwd()
        os.chdir(os.path.join(working_directory, directory_name))

        # make sure all operators are up-to-date
        for observable in expectation_values:
            self._update_operator(observable)
        self._valid = True

        # create the time evolution operator
        time_evo_op = TimeEvolutionOperator(**self._operator_args)

        # start the simulation
        print("Starting simulation...")
        start = default_timer()
        try:
            for time_step in range(0, total_time_steps):
                # check if something needs to be written to a file
                if time_step % write_step == 0:
                    # write the simulation time
                    self._write_time()

                    # write the data objects
                    for item in data_objects:
                        self._write_data(item)

                    # write the expectation values
                    for observable in expectation_values:
                        self._write_expectation_value(observable)

                # evolve the wave function
                self.wave_function.values = time_evo_op.map(self.wave_function)

                # update the simulation time
                self.time_step += 1
                self.time += self.time_increment

        # perform clean up duties even if the simulation fails
        finally:
            # end the simulation
            end = default_timer()
            elapsed = round(end - start, 5)
            print(f"Simulation finished after {elapsed} seconds!")

            # change working directory back to original directory
            os.chdir("..")

            # operators are no longer valid and they will be updated
            self._valid = False

    def _write_parameters(self, **kwargs):
        pass

    def _write_time(self):
        with open("time.txt", "a") as file:
            file.write("%.5e \n" % self.time)

    @property
    def _operator_args(self):
        operator_args = {
            "grid": self.wave_function.grid,
            "potential": self.potential,
            "mass": self.wave_function.mass,
            "time_increment": self.time_increment,
            "accuracy_grid": self.accuracy_grid,
            "accuracy_time": self.accuracy_time,
        }
        return operator_args

    def _register_operator(self, observable: str):
        try:
            operator_class = self._operator_class_dict[observable]
        except KeyError as error:
            raise ValueError(f"Cannot find operator class corresponding to '{observable}'!") from error
        else:
            operator_instance = operator_class(**self._operator_args)
            self._operator_instance_dict[observable] = operator_instance

    def _unregister_operator(self, observable: str):
        try:
            self._operator_instance_dict.pop(observable)
        except KeyError:
            print(f"Cannot find operator instance corresponding to '{observable}'!")

    def _update_operator(self, observable: str):
        try:
            operator = self._operator_instance_dict[observable]
        except KeyError:
            self._register_operator(observable)
        else:
            operator.valid = False

    def get_expectation_value(self, observable: str) -> float:
        if not self._valid:
            self._update_operator(observable)
        operator = self._operator_instance_dict[observable]
        expectation_value = self.wave_function.expectation_value(operator)
        return expectation_value

    def _write_expectation_value(self, observable: str):
        filename = f"{observable}.txt"
        exp_val = self.get_expectation_value(observable)
        with open(filename, "a") as file:
            file.write("%.5e \n" % exp_val)

    @property
    def _data_dict(self) -> Dict[str, np.ndarray]:
        data_dict = {
            "wave_function": self.wave_function.values,
            "probability_density": self.wave_function.probability_density,
            "potential": self.potential.values,
        }
        return data_dict

    def get_data(self, item: str) -> np.ndarray:
        try:
            data = self._data_dict[item]
        except KeyError as error:
            raise ValueError(f"Cannot find reference to data '{item}'!") from error
        else:
            return data

    def _write_data(self, identifier: str):
        filename = f"{identifier}.txt"
        data = self.get_data(identifier)
        with open(filename, "ab") as file:
            np.savetxt(file, [data], fmt="%.3e", delimiter=",")


def integrate(function_values: np.ndarray, grid_spacing: float) -> float:
    return np.sum(function_values) * grid_spacing
