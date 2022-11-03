from itertools import chain
from typing import Callable, Dict, List, Tuple

import numpy as onp
import sympy as sp
import jax.numpy as jnp
from jax import partial

from projection_filter import MultiDimensionalExponentialFamilyQMC, MultiDimensionalExponentialFamilySPG, \
    SStarProjectionFilter
from symbolic import SDE, sympy_matrix_to_jax
from symbolic.n_d import backward_diffusion, column_polynomials_coefficients, from_tuple_to_symbolic_monom, \
    get_monomial_degree_set


class MultiDimensionalSStarProjectionFilter(SStarProjectionFilter):
    def __init__(self, dynamic_sde: SDE,
                 measurement_sde: SDE,
                 natural_statistics_symbolic: sp.MutableDenseMatrix,
                 constants: Dict,
                 initial_condition: jnp.ndarray,
                 measurement_record: jnp.ndarray,
                 delta_t: float,
                 nodes_number: int = 1000,
                 integrator: str = 'spg',
                 level: int = 5,
                 epsilon: float = 1e-7,
                 bijection: Callable[[jnp.ndarray], jnp.ndarray] = jnp.arctanh,
                 ode_solver: str = 'RK',
                 sRule: str = "clenshaw-curtis"):
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
        out : OneDimensionalSBulletProjectionFilter

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
                         bijection,
                         ode_solver=ode_solver)
        self._level = level
        self._nodes_number = nodes_number
        #       the statistics are assumed to be free of additional symbolic parameters
        self._natural_statistics, _ = \
            sympy_matrix_to_jax(
                self._natural_statistics_symbolic, self._dynamic_sde.variables)

        (L_mat, A_mat, b_mat,
         lambda_mat, monom_list_,
         remaining_monom_list_) = self._get_projection_filter_matrices()
        #   since M_0, m_h, lamda starts with coefficient of x^0, then we need to remove the first column/entry
        self._L_0 = L_mat[:, 1:]
        self._ell_0 = L_mat[:, 0]
        self._A_0 = A_mat[:, 1:]
        self._b_h = b_mat[0, 1:]
        self._a_0 = A_mat[:, 0]
        self._b_0 = b_mat[0, 0]
        self._lamda = lambda_mat.T
        self._monom_list = monom_list_
        self._remaining_monom_list = remaining_monom_list_

        self._remaining_statistics, _ = \
            sympy_matrix_to_jax(
                self._construct_remaining_statistics(), self._dynamic_sde.variables)

        self._integrator = integrator

        if integrator.lower() == 'spg':
            self._exponential_density = MultiDimensionalExponentialFamilySPG(sample_space_dimension=self._sample_space_dimension,
                                                                             sparse_grid_level=self._level,
                                                                             bijection=bijection,
                                                                             statistics=self._natural_statistics,
                                                                             remaining_statistics=self._remaining_statistics,
                                                                             epsilon=epsilon,
                                                                             sRule=sRule
                                                                             )
            self._integrator = integrator
        elif integrator.lower() == 'qmc':
            self._exponential_density = MultiDimensionalExponentialFamilyQMC(sample_space_dimension=self._sample_space_dimension,
                                                                             nodes_number=self._nodes_number,
                                                                             bijection=bijection,
                                                                             statistics=self._natural_statistics,
                                                                             remaining_statistics=self._remaining_statistics
                                                                             )
            self._integrator = integrator
        else:
            raise ValueError('Integrator not recognized. At the moment, it should be either SPG for sparse grid,'
                             'or QMC for quasi Monte Carlo')

        if initial_condition.shape[0] != self._exponential_density.params_num:
            raise Exception("Wrong initial condition shape!, expected {} "
                            "given {}".format(initial_condition.shape[0], self._exponential_density.params_num))
        self._current_state = initial_condition
        self._state_history = self._current_state[jnp.newaxis, :]

    @property
    def level(self):
        return self._level

    @property
    def nodes_number(self):
        return self.exponential_density.nodes_number

    @property
    def integrator_type(self):
        return self._integrator

    @property
    def monom_list(self):
        return self._monom_list

    @property
    def natural_statistics(self):
        return self._exponential_density.natural_statistics

    @property
    def remaining_statistics(self):
        return self._exponential_density.remaining_statistics

    @property
    def exponential_density(self):
        return self._exponential_density

    def get_density_values(self, grid_limits: jnp.ndarray, nb_of_points: jnp.ndarray):
        x_ = []
        for i in range(self._exponential_density.sample_space_dim):
            temp_ = jnp.linspace(
                grid_limits[i, 0], grid_limits[i, 1], nb_of_points[i], endpoint=True)
            x_.append(temp_)
        grids = jnp.meshgrid(*x_, indexing='xy')
        grids = jnp.stack(grids, axis=-1)
        return self.get_density_values_from_grids(grids)

    def get_density_values_from_grids(self, grids):
        c_ = self.natural_statistics(grids)

        if self._exponential_density.sample_space_dim == 1:
            signature = '(n)->(m)'
        elif self._exponential_density.sample_space_dim == 2:
            signature = '(n)->(l,m)'
        elif self._exponential_density.sample_space_dim == 3:
            signature = '(n)->(k,l,m)'
        else:
            raise NotImplementedError

        @partial(jnp.vectorize, signature=signature)
        def _evalulate_density(theta_):
            psi_ = self._exponential_density.log_partition(theta_)
            density = jnp.exp(c_ @ theta_ - psi_)
            return density

        density_history = _evalulate_density(self.state_history)
        return grids, density_history

    def _construct_remaining_statistics(self):
        remaining_monoms = [from_tuple_to_symbolic_monom(self._dynamic_sde.variables, monomial_degree)
                            for monomial_degree in self._remaining_monom_list]
        remaining_statistics_symbolic = sp.Matrix(remaining_monoms)
        return remaining_statistics_symbolic

    def _get_projection_filter_matrices(self) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray, onp.ndarray,
                                                       List[Tuple[int, int]],
                                                       List[Tuple[int, int]]]:
        """
        Get matrices related to a projection filter with an exponential family densities.

        Returns
        -------
        matrices: Tuple
            List of matrices containing M_0, m_h, and lamda
        """
        Lc = backward_diffusion(
            self._natural_statistics_symbolic, self._dynamic_sde)

        h_T_h = self._measurement_sde.drifts.transpose() * self._measurement_sde.drifts
        #   squared measurement drift times natural statistics
        h_T_h_per_2_times_c = h_T_h[0] * \
            self._natural_statistics_symbolic
        Lc_min_h_T_h_times_c_per_2 = Lc - h_T_h_per_2_times_c / 2

        l_monom_set = get_monomial_degree_set(Lc, self._dynamic_sde.variables)

        a_monom_set = get_monomial_degree_set(
            Lc_min_h_T_h_times_c_per_2, self._dynamic_sde.variables)

        b_monom_set = get_monomial_degree_set(
            h_T_h / 2, self._dynamic_sde.variables)

        c_monom_set = get_monomial_degree_set(
            self._measurement_sde.drifts, self._dynamic_sde.variables)

        natural_monom_set = get_monomial_degree_set(
            self._natural_statistics_symbolic, self._dynamic_sde.variables)

        monom_set = natural_monom_set.union(a_monom_set).union(
            b_monom_set).union(c_monom_set).union(l_monom_set)

        remaining_monoms_set = monom_set.difference(natural_monom_set)
        constant_monom = (0, 0)
        constant_monom_list = [constant_monom, ]
        if constant_monom in remaining_monoms_set:
            remaining_monoms_set.remove(constant_monom)

        natural_monom_list = list(natural_monom_set)
        natural_monom_list.sort()
        remaining_monoms_list = list(remaining_monoms_set)
        remaining_monoms_list.sort()
        monom_list = list(chain.from_iterable(
            [constant_monom_list, natural_monom_list, remaining_monoms_list]))

        monoms_list_symbol_L, L_matrix = column_polynomials_coefficients(
            Lc, self._dynamic_sde.variables, monom_list)

        monoms_list_symbol_A, A_matrix = column_polynomials_coefficients(Lc_min_h_T_h_times_c_per_2,
                                                                         self._dynamic_sde.variables,
                                                                         monom_list)

        monoms_list_symbol_b, b_matrix = column_polynomials_coefficients(h_T_h / 2, self._dynamic_sde.variables,
                                                                         monom_list)

        monoms_list_symbol_lam, lambda_matrix = column_polynomials_coefficients(self._measurement_sde.drifts,
                                                                                self._dynamic_sde.variables,
                                                                                natural_monom_list)

        return L_matrix, A_matrix, b_matrix, lambda_matrix, monom_list, remaining_monoms_list
