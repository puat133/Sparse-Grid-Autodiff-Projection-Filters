from inspect import getfullargspec
from typing import Callable, Dict, Tuple

import sympy as sp
import jax.numpy as jnp
from jax import partial

from projection_filter import OneDExponentialFamily, SStarProjectionFilter
from symbolic import SDE, backward_diffusion_one_D, column_polynomials_coefficients_one_D, \
    column_polynomials_maximum_degree_one_D, lamdify, sympy_matrix_to_jax


class OneDimensionalSStarProjectionFilter(SStarProjectionFilter):
    def __init__(self, dynamic_sde: SDE,
                 measurement_sde: SDE,
                 natural_statistics_symbolic: sp.MutableDenseMatrix,
                 constants: Dict,
                 initial_condition: jnp.ndarray,
                 measurement_record: jnp.ndarray,
                 delta_t: float,
                 nodes_number: int = 10,
                 bijection: Callable[[jnp.ndarray], jnp.ndarray] = jnp.arctanh):
        """
        A Class that encapsulates the S Bullet class of exponential family projection filter. The assumption is that
        the natural statistics span the measurement sde drift functions. Here we also assume that
        the sdes correspond to the dynamic and the measurement are polynomial functions with the same variable.
        The measurement numerical value is assumed to be close to the limit so that it can be treated as a continuous
        measurements. The filtering dynamics is solved using Euler-Maruyama scheme.
        See [1]

        Parameters
        ----------
        dynamic_sde : SDE
            SDE for the dynamic.

        measurement_sde : SDE
            SDE for the measurement.

        natural_statistics_symbolic : MutableDenseMatrix
            Natural statistics symbolic expression, at the moment, it only supports polynomial functions.

        constants : Dict
            Some constants to be passed to the matrix expression.

        Returns
        -------
        out : OneDimensionalSStarProjectionFilter

        References
        ----------
        [1]

        """
        super().__init__(dynamic_sde,
                         measurement_sde,
                         natural_statistics_symbolic,
                         constants,
                         initial_condition,
                         measurement_record,
                         delta_t,
                         bijection)

        #       the statistics are assumed to be free of additional symbolic parameters
        self._natural_statistics, _ = \
            sympy_matrix_to_jax(self._natural_statistics_symbolic, [self._dynamic_sde.variables[0], ])

        _L, _A, _b, lamda = self._get_projection_filter_matrices()
        #   since M_0, m_h, lamda starts with coefficient of x^0, then we need to remove the first column/entry
        self._L_0 = _L[:, 1:]
        self._ell_0 = _L[:, 0]
        self._A_0 = _A[:, 1:]
        self._b_h = _b[0, 1:]
        self._a_0 = _A[:, 0]
        self._b_0 = _b[0, 0]
        self._lamda = lamda[:, 1:].T
        self._lambda_0 = lamda[:, 0]

        self._remaining_statistics, _ = \
            sympy_matrix_to_jax(self._construct_remaining_statistics(), [self._dynamic_sde.variables[0], ])
        self._exponential_density = OneDExponentialFamily(nodes_number, bijection, self._natural_statistics,
                                                          self._remaining_statistics)

        if initial_condition.shape[0] != self._exponential_density.params_num:
            raise Exception("Wrong initial condition shape!, expected {} "
                            "given {}".format(initial_condition.shape[0], self._exponential_density.params_num))
        self._current_state = initial_condition
        self._state_history = self._current_state[jnp.newaxis, :]

    @property
    def natural_statistics(self):
        return self._exponential_density.natural_statistics

    @property
    def remaining_statistics(self):
        return self._exponential_density.remaining_statistics

    @property
    def exponential_density(self):
        return self._exponential_density

    def get_density_values(self, grid_limits, nb_of_points):
        grid_limits = grid_limits.squeeze()
        x_ = jnp.linspace(grid_limits[0], grid_limits[1], nb_of_points[0], endpoint=True)
        c_ = self._exponential_density.natural_statistics(x_)

        @partial(jnp.vectorize, signature='(n)->(m)')
        def _evalulate_density(theta_):
            psi_ = self._exponential_density.log_partition(theta_)
            density = jnp.exp(c_ @ theta_ - psi_)
            return density

        density_history_ = _evalulate_density(self.state_history)
        return x_, density_history_

    def _construct_remaining_statistics(self):
        natural_statistics_max_polynomial_degree = column_polynomials_maximum_degree_one_D(
            self._natural_statistics_symbolic,
            self._dynamic_sde.variables[0])
        M_0_max_degree_plus_one = self._A_0.shape[1] + 1
        temp = []
        for i in range(natural_statistics_max_polynomial_degree + 1, M_0_max_degree_plus_one):
            temp.append(self._dynamic_sde.variables[0] ** i)

        remaining_statistics_symbolic = sp.Matrix(temp)
        return remaining_statistics_symbolic

    def _get_projection_filter_matrices(self) -> Tuple:
        """
        Get matrices related to a projection filter with an exponential family densities.

        Returns
        -------
        matrices: Tuple
            List of matrices containing M_0, m_h, and lamda
        """
        Lc = backward_diffusion_one_D(self._natural_statistics_symbolic, self._dynamic_sde)

        #   squared measurement drift times natural statistics
        absh2c = self._natural_statistics_symbolic * self._measurement_sde.drifts[0] ** 2

        L_sym = column_polynomials_coefficients_one_D(Lc, self._dynamic_sde.variables[0])

        A_sym = column_polynomials_coefficients_one_D(Lc - absh2c / 2, self._dynamic_sde.variables[0])

        #   this would be the maximum degree in M_0, m_h, and lamda
        A_max_degree = column_polynomials_maximum_degree_one_D(Lc - absh2c / 2, self._dynamic_sde.variables[0])

        b_sym = column_polynomials_coefficients_one_D(sp.Matrix([self._measurement_sde.drifts[0] ** 2 / 2]),
                                                      self._dynamic_sde.variables[0])

        lamda_sym = column_polynomials_coefficients_one_D(sp.Matrix([self._measurement_sde.drifts[0]]),
                                                          self._dynamic_sde.variables[0])

        matrices = []
        expression_list = [L_sym, A_sym, b_sym, lamda_sym]

        #   this will convert the symbolic expression to ndarrays
        for i, expression in zip(range(len(expression_list)), expression_list):
            vector_fun = lamdify(expression)
            arguments_length = len(expression.free_symbols)
            if arguments_length == 0:
                matrix = jnp.array(vector_fun(), dtype=jnp.float32)
            else:
                full_arg_spec = getfullargspec(vector_fun)
                arg_names = full_arg_spec.args
                arguments_list = [self._constants[key] for key in arg_names]
                matrix = jnp.array(vector_fun(*arguments_list),
                                   dtype=jnp.float32)  # there could be a better way to implement this.

            if (matrix.shape[1] != A_max_degree + 1) and i < 3:
                temp = jnp.pad(matrix, ((0, 0), (0, A_max_degree + 1 - matrix.shape[1])))
                matrix = temp

            # for lamda
            if i == 3 and (matrix.shape[1] < self._natural_statistics_symbolic.shape[0] + 1):
                temp = jnp.pad(matrix, ((0, 0), (0, self._natural_statistics_symbolic.shape[0] + 1 - matrix.shape[1])))
                matrix = temp

            matrices.append(matrix)

        return tuple(matrices)
