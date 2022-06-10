"""Main module with most functionality. It is loaded when importing the package."""

from typing import Any, Callable, Dict, List, Tuple, Union
from abc import abstractmethod
import math
from scipy.sparse import csr_matrix, dia_matrix, diags, identity
from scipy.sparse.linalg import spsolve
import numpy as np
from findiff import FinDiff
import os
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

    def __init__(self, grid: Grid, **kwargs):
        self.grid = grid
        self._matrix = None

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

    def assert_compatibility(self, vector: WaveFunction) -> bool:
        """
        Checks if a state vector is compatible with the linear operator.

        Uses :func:`numpy.array_equal` to verify that the grid coordinate arrays are equal.

        Parameters
        ----------
        vector : :class:`WaveFunction`
            State vector.

        Returns
        -------
        compatibility : :class:`bool`
            Returned value is ``True`` if the grid coordinates match and ``False`` otherwise.
        """
        if np.array_equal(self.grid.coordinates, vector.grid.coordinates):
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
        if self._matrix is None:
            raise NotImplementedError
        return self._matrix


class PositionOperator(LinearOperator):
    """Class representation of the position operator :math:`\\hat{x}`."""

    def __init__(self, grid: Grid, **kwargs):
        super().__init__(grid)

    @property
    def matrix(self) -> dia_matrix:
        """Matrix representation of the position operator :math:`\\hat{x}`.

        Returns
        -------
        matrix : :class:`scipy.sparse.dia.dia_matrix`
            Sparse matrix containing the grid coordinate values on the main diagonal.

        Notes
        -----
        Uses :func:`scipy.sparse.diags` to generate the scalar matrix.
        """
        if self._matrix is None:
            shape = (self.grid.points, self.grid.points)
            matrix = diags(self.grid.coordinates, 0, shape=shape)
            self._matrix = matrix
        return self._matrix


class PotentialEnergyOperator(LinearOperator):
    """Class representation of the potential energy operator :math:`\\hat{V}`.

    Parameters
    -----------------
    potential : :class:`Potential`
        Time-independent external potential.
    """

    def __init__(self, grid: Grid, potential: Potential, **kwargs):
        super().__init__(grid)
        self.potential = potential

    @property
    def matrix(self) -> dia_matrix:
        """Matrix representation of the potential energy operator :math:`\\hat{V}`.

        Returns
        -------
        matrix : :class:`scipy.sparse.dia.dia_matrix`
            Sparse matrix containing the function values of the potential on the main diagonal.

        Raises
        ------
        ValueError
            If the grid of the potential doesn't match the grid of the :class:`LinearOperator` instance.

        Notes
        -----
        Uses :func:`scipy.sparse.diags` to generate the scalar matrix.
        """
        if self._matrix is None:
            if np.array_equal(self.grid.coordinates, self.potential.grid.coordinates) is False:
                raise ValueError("Grids of potential and linear operator do not match!")
            else:
                shape = (self.grid.points, self.grid.points)
                matrix = diags(self.potential.values, 0, shape=shape)
                self._matrix = matrix
        return self._matrix


class MomentumOperator(LinearOperator):
    """Class representation of the momentum operator :math:`\\hat{p}`.

    Other Parameters
    ----------------
    accuracy_grid : :class:`int`, default=2
        Order of accuracy in the grid spacing of the finite difference scheme. By default, :mod:`findiff` uses second \
        order accuracy.

    Notes
    -----
    .. math:: \\hat{p} = -\\text{i} \\hbar \\nabla
    """

    def __init__(self, grid: Grid, **kwargs):
        super().__init__(grid)
        self.accuracy_grid = kwargs.get("accuracy_grid", 2)

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
        Uses :class:`findiff.FinDiff` to create the necessary matrix with finite difference coefficients, \
        assumes a homogeneous grid with even spacing. For further information refer to the :mod:`findiff` package and \
        its documentation.
        """
        if self._matrix is None:
            first_derivative = FinDiff(0, self.grid.spacing, 1, acc=self.accuracy_grid)\
                .matrix(self.grid.coordinates.shape)
            matrix = -1j * first_derivative
            self._matrix = matrix
        return self._matrix


class KineticEnergyOperator(LinearOperator):
    """Class representation of the kinetic energy operator :math:`\\hat{T}`.

    Parameters
    ----------
    mass : :class:`float`
        Mass of the particle.

    Other Parameters
    ----------------
    accuracy_grid : :class:`int`, default=2
        Order of accuracy in the grid spacing of the finite difference scheme. By default, :mod:`findiff` uses second \
        order accuracy.

    Notes
    -----
    .. math:: \\hat{T} = - \\frac{ \\hbar ^2 }{ 2m } \\nabla ^2
    """

    def __init__(self, grid: Grid, mass: float, **kwargs):
        super().__init__(grid)
        self.mass = mass
        self.accuracy_grid = kwargs.get("accuracy_grid", 2)

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
        Uses :class:`findiff.FinDiff` to create the necessary matrix with finite difference coefficients, \
        assumes a homogeneous grid with even spacing. For further information refer to the :mod:`findiff` package and \
        its documentation.
        """
        if self._matrix is None:
            second_derivative = FinDiff(0, self.grid.spacing, 2, acc=self.accuracy_grid)\
                .matrix(self.grid.coordinates.shape)
            matrix = -1 * np.reciprocal(2. * self.mass) * second_derivative
            self._matrix = matrix
        return self._matrix


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
    accuracy_grid : :class:`int`, default=2
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
        self.accuracy_grid = kwargs.get("accuracy_grid", 2)

    @property
    def matrix(self) -> csr_matrix:
        """Matrix representation of the Hamiltonian :math:`\\hat{H}`.

        Returns
        -------
        matrix : :class:`scipy.sparse.csr.csr_matrix`
            Sparse matrix containing the second derivative finite difference coefficients multiplied with \
            :math:`- \\frac{ \\hbar ^2 }{ 2m }`, the function values of the external potential :math:`V` are added \
            to the main diagonal.

        Notes
        -----
        Uses :class:`KineticEnergyOperator` and :class:`PotentialEnergyOperator` to create the necessary matrix \
        representations of these operators.
        """
        if self._matrix is None:
            matrix = KineticEnergyOperator(self.grid, self.mass, accuracy_grid=self.accuracy_grid).matrix \
                     + PotentialEnergyOperator(self.grid, self.potential).matrix
            self._matrix = matrix
        return self._matrix


class IdentityOperator(LinearOperator):
    """Class representation of the identity operator :math:`\\hat{1}`."""

    def __init__(self, grid: Grid, **kwargs):
        super().__init__(grid)

    @property
    def matrix(self) -> dia_matrix:
        """Matrix representation of the identity operator :math:`\\hat{1}`.

        Returns
        -------
        matrix : :class:`scipy.sparse.dia.dia_matrix`
            Sparse matrix containing :math:`1` on the main diagonal.

        Notes
        -----
        Uses :func:`scipy.sparse.identity` to generate the identity matrix.
        """
        if self._matrix is None:
            matrix = identity(self.grid.points)
            self._matrix = matrix
        return self._matrix


class TimeEvolutionOperator(LinearOperator):
    r"""Class representation of the time evolution operator :math:`\hat{U} \left( \Delta t \right)`.

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
    accuracy_grid : :class:`int`, default=2
        Order of accuracy in the grid spacing of the finite difference scheme. By default, :mod:`findiff` uses second \
        order accuracy.
    accuracy_time : :class:`int`, default=3
        Order of accuracy in the time increment. An uneven order of accuracy in the time increment results in a \
        diagonal :math:`\left[ m / m \right]` Padé approximant which provides higher numerical stability and \
        precision. By default, the :math:`\left[ 1 / 1 \right]` Padé approximant with third order accuracy in the \
        time increment is employed.

    Notes
    -----
    The exponential operator :math:`e^{ - \text{i} \hat{H} \Delta t / \hbar }` is replaced by its \
    :math:`\left[ m / n \right]` Padé approximant. The resulting operator is unitary and conserves wave function \
    normalization and time reversibility. [1]_

    .. math::

        \hat{U} \left( \Delta t \right) = e^{ - \text{i} \hat{H} \Delta t / \hbar }
        \approx \frac{ \prod _{ s = 1 } ^m \left( \hat{1} + \text{i} \hat{H} \Delta t / z _s ^m \hbar \right) }
        { \prod _{ s = 1 } ^n \left( \hat{1} - \text{i} \hat{H} \Delta t / z _s ^n \hbar \right) }
        + \mathcal{O} \left( \Delta t ^{ m + n + 1 } \right)

    References
    ----------
    .. [1]  W. van Dijk and F. M. Toyama, Phys. Rev. E **2007**, *75*, 036707, DOI: `10.1103/PhysRevE.75.036707 \
            <https://doi.org/10.1103/PhysRevE.75.036707>`_.
    """

    def __init__(self, grid: Grid, mass: float, potential: Potential, time_increment: Union[float, complex], **kwargs):
        super().__init__(grid)
        self.mass = mass
        self.potential = potential
        self.time_increment = time_increment
        self.accuracy_grid = kwargs.get("accuracy_grid", 2)
        self.accuracy_time = kwargs.get("accuracy_time", 3)

    def map(self, vector: WaveFunction):
        r"""Applies the time evolution operator to a state vector.

        The linear transformation cannot be computed through a simple matrix vector product because the time \
        evolution operator is replaced by its diagonal :math:`\left[ m / n \right]` Padé approximant. \
        Instead, the linear mapping is achieved by solving the arising system of linear equations with \
        :func:`scipy.sparse.linalg.spsolve`.

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

            \hat{U} \left( \Delta t \right) _{ \text{Denominator} } \, \Psi \left( x, t + \Delta t \right) =
            \hat{U} \left( \Delta t \right) _{ \text{Numerator} } \, \Psi \left( x, t \right)

        """
        if self.assert_compatibility(vector) is False:
            raise ValueError("Grid of vector and linear operator do not match!")

        # matrix A is already known
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
        diagonal :math:`\left[ m / n \right]` Padé approximant are returned because the inverse of the Hamiltonian \
        cannot be calculated.

        Returns
        -------
        numerator_matrix, denominator_matrix : :class:`tuple` of :class:`scipy.sparse.csr.csr_matrix`
            Sparse matrices containing the matrix representations of the numerator and the denominator of the Padé \
            approximated time evolution operator.

        Notes
        -----
        First, determines the polynomial coefficients :math:`a_k` and :math:`b_k` of the :math:`\left[ m / n \right]` \
        Padé approximation's numerator and denominator polynomials through known recursive relations. [1]_ Then, \
        calculates the roots of the polynomials :math:`z _s ^m` and :math:`z _s ^n` for the polynomials' factored \
        form. Lastly, generates the matrix representations of the denominator and numerator polynomials. \
        Uses :func:`scipy.sparse.identity` to generate the identity matrix. Additionally, uses \
        :class:`Hamiltonian` to create the matrix representation of the time-independent Hamiltonian.

        .. math::

            \hat{U} \left( \Delta t \right) _{ \text{Numerator} }
            = \prod _{ s = 1 } ^m \left( \hat{1} + \text{i} \hat{H} \Delta t / z _s ^m \hbar \right) \\
            \hat{U} \left( \Delta t \right) _{ \text{Denominator} }
            = \prod _{ s = 1 } ^n \left( \hat{1} - \text{i} \hat{H} \Delta t / z _s ^n \hbar \right)

        References
        ----------
        .. [1]  Nicholas J. Higham, *SIAM J. Matrix Anal. Appl.* **2005**, *26* (4), 1179–1193, DOI: \
                `10.1137/04061101X <https://doi.org/10.1137/04061101X>`_.
        """
        if self._matrix is None:
            # order of Padé approximation
            if self.accuracy_time % 2 == 0:
                m = n = int((self.accuracy_time - 1) / 2)
            else:
                m = int(math.floor((self.accuracy_time - 1) / 2))
                n = int(math.ceil((self.accuracy_time - 1) / 2))

            # polynomial coefficients stored in a list
            a = []
            b = []

            # coefficients obtained through recursive relations
            fac = math.factorial
            for k in range(m + 1):
                a.append(fac(m + n - k) * fac(m) / (fac(m + n) * fac(k) * fac(m - k)))
            for k in range(n + 1):
                b.append((-1) ** k * fac(m + n - k) * fac(n) / (fac(m + n) * fac(k) * fac(n - k)))

            # find the roots of the numerator p and denominator q for factored form
            roots_p = np.roots(np.flip(a))
            roots_q = np.roots(np.flip(b))

            # create the matrix representations for p and q recursively with help of the calculated roots
            p = 1
            q = 1
            for root in roots_p:
                p *= (identity(self.grid.points) + 1.j * self.time_increment
                      * Hamiltonian(self.grid, self.mass, self.potential, accuracy_grid=self.accuracy_grid).matrix
                      / root)
            for root in roots_q:
                q *= (identity(self.grid.points) + 1.j * self.time_increment
                      * Hamiltonian(self.grid, self.mass, self.potential, accuracy_grid=self.accuracy_grid).matrix
                      / root)

            self._matrix = p, q
        return self._matrix


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
        # wave function and potential are private because changes to them later on are not supported
        self._wave_function = wave_function
        self._potential = potential

        # arguments controlling the accuracy of the simulation are public and can be changed later on
        self.time_increment = time_increment
        self.accuracy_grid: int = kwargs.get("accuracy_grid", 2)
        self.accuracy_time: int = kwargs.get("accuracy_time", 3)

        # simulation time should not be public to avoid duplicate times
        self._time_step: int = 0
        self._time: Union[float, complex] = 0.

    def __call__(self, total_time_steps: int, **kwargs):
        # process optional keyword arguments
        name: str = kwargs.get("name", "simulation")
        write_step: int = kwargs.get("write_step", total_time_steps)
        data_objects: List[str] = kwargs.get("data_objects", None)
        expectation_values: List[str] = kwargs.get("expectation_values", None)

        # create a new directory whose name is unique
        directory_name = name
        i = 1
        while True:
            try:
                os.mkdir(directory_name)
            except FileExistsError:
                directory_name = '_'.join([name, str(i)])
                i += 1
                continue
            else:
                break

        # change the working directory to the new directory
        working_directory = os.getcwd()
        os.chdir(os.path.join(working_directory, directory_name))

        # start the timer
        print("Starting simulation...")
        start = default_timer()

        try:
            # create all required operator instances
            operator_dict: Dict[str, LinearOperator] = {}
            for observable in expectation_values:
                operator_dict[observable] = self.create_operator(observable)

            # create the time evolution operator
            time_evo_op = TimeEvolutionOperator(**self._operator_kwargs)

            # iterate over all time steps
            for time_step in range(0, total_time_steps):
                # check if something needs to be written to a file
                if time_step % write_step == 0:
                    # write the simulation time
                    self._write_time()

                    # write all data objects
                    for item in data_objects:
                        self._write_data(item)

                    # write all expectation values
                    for observable, operator in operator_dict.items():
                        self._write_expectation_value(observable, operator)

                # evolve the wave function
                self._wave_function.values = time_evo_op.map(self._wave_function)

                # update the simulation time
                self._time_step += 1
                self._time += self.time_increment

        # perform clean up duties even if the simulation is interrupted
        finally:
            # end the simulation
            end = default_timer()
            elapsed = round(end - start, 5)
            print(f"Simulation finished after {elapsed} seconds!")

            # change working directory back to original directory
            os.chdir("..")

    def _write_time(self):
        with open("time.txt", "a") as file:
            file.write(f"{self._time_step}, {self._time:.5f}, \n")

    @property
    def _operator_kwargs(self) -> Dict[str, Any]:
        operator_kwargs = {
            "grid": self._wave_function.grid,
            "potential": self._potential,
            "mass": self._wave_function.mass,
            "time_increment": self.time_increment,
            "accuracy_grid": self.accuracy_grid,
            "accuracy_time": self.accuracy_time,
        }
        return operator_kwargs

    def create_operator(self, observable: str) -> LinearOperator:
        try:
            operator_class = self._operator_class_dict[observable]
        except KeyError as error:
            raise ValueError(f"Cannot find operator corresponding to '{observable}'") from error
        else:
            operator_instance = operator_class(**self._operator_kwargs)
            return operator_instance

    def _write_expectation_value(self, observable: str, operator: LinearOperator):
        filename = f"{observable}.txt"
        exp_val = self._wave_function.expectation_value(operator)
        with open(filename, "a") as file:
            file.write(f"{self._time_step}, {exp_val:.5f}, \n")

    @property
    def _data_dict(self) -> Dict[str, np.ndarray]:
        data_dict = {
            "wave_function": self._wave_function.values,
            "probability_density": self._wave_function.probability_density,
            "potential": self._potential.values,
        }
        return data_dict

    def _get_data(self, identifier: str) -> np.ndarray:
        try:
            data = self._data_dict[identifier]
        except KeyError as error:
            raise ValueError(f"Cannot find reference to data labelled '{identifier}'") from error
        else:
            return data

    def _write_data(self, identifier: str):
        filename = f"{identifier}.txt"
        data = self._get_data(identifier)
        with open(filename, "a") as file:
            file.write(f"{self._time_step}, ")
        with open(filename, "ab") as file:
            np.savetxt(file, [data], fmt="%.3e", delimiter=", ")


def integrate(function_values: np.ndarray, grid_spacing: float) -> float:
    return np.sum(function_values) * grid_spacing
