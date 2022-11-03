from typing import Callable, Tuple

import jax.numpy as jnp
from jax import partial
from jax.tree_util import register_pytree_node_class

from projection_filter import ExponentialFamily


@register_pytree_node_class
class OneDExponentialFamily(ExponentialFamily):
    def __init__(self, nodes_number: int,
                 bijection: Callable[[jnp.ndarray], jnp.ndarray],
                 statistics: Callable[[jnp.ndarray], jnp.ndarray],
                 remaining_statistics: [[jnp.ndarray], jnp.ndarray] = None):
        self._nodes_num = nodes_number

        # This is the Chebyshev node positions
        self._absisca = jnp.cos(
            jnp.pi * (jnp.arange(self._nodes_num) + 0.5) / self._nodes_num)

        super().__init__(sample_space_dimension=1,
                         bijection=bijection,
                         statistics=statistics,
                         remaining_statistics=remaining_statistics,
                         statistics_vectorization_signature='()->(n)',
                         partition_vectorization_signature='(n)->()',
                         partition_integrand_vectorization_signature='(n),()->()'
                         )

    @property
    def nodes_number(self):
        return self._nodes_num

    def get_density_values(self, grid_limits: jnp.ndarray, theta: jnp.ndarray, nb_of_points: jnp.ndarray) -> \
            Tuple[jnp.ndarray, jnp.ndarray]:
        grid_limits = grid_limits.squeeze()
        grid = jnp.linspace(
            grid_limits[0], grid_limits[1], nb_of_points[0], endpoint=True)
        c_ = self._natural_statistics(grid)

        def _evalulate_density(theta_):
            psi_ = self.log_partition(theta_)
            density = jnp.exp(c_ @ theta_ - psi_)
            return density

        density = _evalulate_density(theta)
        return grid, density

    def integrate_partition(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Perform integration of the log partition function using Chebyshev rule

        Parameters
        ----------
        theta: ndarray
            Parameter of the extended exponential family

        Returns
        -------
        res: ndarray
            Integration result

        """
        x = self._absisca

        # This is the Chebyshev
        res = jnp.sum(self._partition_integrand(theta, x) *
                      jnp.sqrt(1 - x * x)) * jnp.pi / self._nodes_num

        return res

    def integrate_partition_extended(self, theta_extended: jnp.ndarray) -> jnp.ndarray:
        """
        Perform integration of the log partition extended function using Chebyshev rule

        Parameters
        ----------
        theta_extended: ndarray
            Parameter of the extended exponential family

        Returns
        -------
        res: ndarray
            Integration result

        """
        x = self._absisca

        # This is the Chebyshev
        res = jnp.sum(self._partition_integrand_extended(theta_extended, x) *
                      jnp.sqrt(1 - x * x)) * jnp.pi / self._nodes_num

        return res

    def tree_flatten(self):
        auxiliaries = None
        return (self._sample_space_dim, self._nodes_num, self._params_num), auxiliaries

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
