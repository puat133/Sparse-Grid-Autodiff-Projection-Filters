from typing import Callable, Tuple
from LowDiscrepancy import halton
import jax.numpy as jnp
from jax import partial, jit
from jax.tree_util import register_pytree_node_class
from abc import ABC, abstractmethod
from projection_filter import ExponentialFamily


@register_pytree_node_class
class MultiDimensionalExponentialFamily(ExponentialFamily, ABC):
    def __init__(self, sample_space_dimension: int,
                 bijection: Callable[[jnp.ndarray], jnp.ndarray],
                 statistics: Callable[[jnp.ndarray], jnp.ndarray],
                 remaining_statistics: [[jnp.ndarray], jnp.ndarray] = None,
                 statistics_vectorization_signature: str = '(d)->(n)',
                 partition_vectorization_signature: str = '(m)->()',
                 partition_integrand_vectorization_signature: str = '(m),(d)->()',
                 ):
        """
        Exponential family for sample space with dimension `d` >=1, and parameter space with dimension `m` where
        of log partition function is solved via `n` Quasi Monte Carlo points (Halton low discrepancy points).

        Parameters
        ----------
        sample_space_dimension
        bijection
        statistics
        remaining_statistics
        statistics_vectorization_signature
        partition_vectorization_signature
        partition_integrand_vectorization_signature
        """

        ExponentialFamily.__init__(self, sample_space_dimension=sample_space_dimension,
                                   bijection=bijection,
                                   statistics=statistics,
                                   remaining_statistics=remaining_statistics,
                                   statistics_vectorization_signature=statistics_vectorization_signature,
                                   partition_vectorization_signature=partition_vectorization_signature,
                                   partition_integrand_vectorization_signature=
                                   partition_integrand_vectorization_signature)

        # default value to be realized in the implemented class
        self._quadrature_points = jnp.empty((1,), dtype=jnp.float32)
        self._bijected_points = jnp.empty((1,), dtype=jnp.float32)
        self._bijected_volume = jnp.empty((1,), dtype=jnp.float32)

    @property
    @abstractmethod
    def bijected_points(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def quadrature_points(self):
        raise NotImplementedError

    @abstractmethod
    def numerical_integration(self, numerical_values: jnp.ndarray, axis=None):
        raise NotImplementedError

    @partial(jit, static_argnums=[0, ])
    def integrate_partition(self, theta: jnp.ndarray) -> jnp.ndarray:
        x = self._quadrature_points
        res = self.numerical_integration(self._partition_integrand(theta, x))
        return res

    @partial(jit, static_argnums=[0, ])
    def integrate_partition_extended(self, theta_extended: jnp.ndarray) -> jnp.ndarray:
        x = self._quadrature_points
        res = self.numerical_integration(self._partition_integrand_extended(theta_extended, x))
        return res

    def get_density_values(self, grid_limits: jnp.ndarray, theta: jnp.ndarray, nb_of_points: jnp.ndarray) -> \
            Tuple[jnp.ndarray, jnp.ndarray]:
        x_ = []
        for i in range(self._sample_space_dim):
            temp_ = jnp.linspace(grid_limits[i, 0], grid_limits[i, 1], nb_of_points[i], endpoint=True)
            x_.append(temp_)
        grids = jnp.meshgrid(*x_, indexing='ij')
        grids = jnp.stack(grids, axis=-1)

        return self.get_density_values_from_grids(grids, theta)

    def get_density_values_from_grids(self, grids, theta):
        c_ = self.natural_statistics(grids)

        @jit
        def _evalulate_density(theta_):
            psi_ = self.log_partition(theta_)
            density_ = jnp.exp(c_ @ theta_ - psi_)
            return density_

        density = _evalulate_density(theta)
        return grids, density

    @abstractmethod
    def _compute_D_part_for_fisher_metric(self, c: jnp.ndarray, exp_c_theta: jnp.ndarray):
        raise NotImplementedError

    def direct_fisher_metric(self, theta: jnp.ndarray):
        """
        Compute Fisher metric without using autodiff. This will produces exactly the same result with
        self.fisher_metric.

        Parameters
        ----------
        theta

        Returns
        -------

        """
        c = self._natural_statistics(self._bijected_points)
        exp_c_theta = jnp.reshape(jnp.exp(c @ theta), (self._quadrature_points.shape[0], 1))
        expectation_of_exp_c_theta_dv = self.numerical_integration(exp_c_theta * self._bijected_volume)
        expectation_of_exp_c_theta_c_dv = self.numerical_integration(exp_c_theta * c * self._bijected_volume, axis=0)
        D = self._compute_D_part_for_fisher_metric(c, exp_c_theta)
        return (1 / expectation_of_exp_c_theta_dv) * ((-1 / expectation_of_exp_c_theta_dv) *
                                                      jnp.outer(expectation_of_exp_c_theta_c_dv,
                                                                expectation_of_exp_c_theta_c_dv) + D)

    def direct_natural_statistics_expectation(self, theta: jnp.ndarray):
        """
        Compute natural statistics expectation without using autodiff. This will produces exactly the same result with
        self.natural_statistics_expectation.
        Parameters
        ----------
        theta

        Returns
        -------

        """
        c = self._natural_statistics(self._bijected_points)
        exp_c_theta = jnp.reshape(jnp.exp(c @ theta), (self._quadrature_points.shape[0], 1))
        expectation_of_exp_c_theta_dv = self.numerical_integration(exp_c_theta * self._bijected_volume)
        expectation_of_exp_c_theta_c_dv = self.numerical_integration(exp_c_theta * c * self._bijected_volume, axis=0)
        return expectation_of_exp_c_theta_c_dv / expectation_of_exp_c_theta_dv

    def direct_extended_statistics_expectation(self, theta_extended: jnp.ndarray):
        """
        Compute extended statistics expectation without using autodiff. This will produces exactly the same result with
        self.extended_statistics_expectation.
        Parameters
        ----------
        theta_extended

        Returns
        -------

        """
        c = self._extended_statistics(self._bijected_points)
        exp_c_theta = jnp.reshape(jnp.exp(c @ theta_extended), (self._quadrature_points.shape[0], 1))
        expectation_of_exp_c_theta_dv = self.numerical_integration(exp_c_theta * self._bijected_volume)
        expectation_of_exp_c_theta_c_dv = self.numerical_integration(exp_c_theta * c * self._bijected_volume, axis=0)
        return expectation_of_exp_c_theta_c_dv / expectation_of_exp_c_theta_dv

    def tree_flatten(self):
        auxiliaries = None
        return (self._sample_space_dim, self._nodes_num, self._params_num), auxiliaries

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
