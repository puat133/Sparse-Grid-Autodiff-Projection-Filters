from typing import Callable

import Tasmanian
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from projection_filter import MultiDimensionalExponentialFamily


@register_pytree_node_class
class MultiDimensionalExponentialFamilySPG(MultiDimensionalExponentialFamily):
    def __init__(self, sample_space_dimension: int,
                 sparse_grid_level: int,
                 bijection: Callable[[jnp.ndarray], jnp.ndarray],
                 statistics: Callable[[jnp.ndarray], jnp.ndarray],
                 remaining_statistics: [[jnp.ndarray], jnp.ndarray] = None,
                 epsilon: float = 1e-7,
                 sRule: str = "clenshaw-curtis"
                 ):
        """
        Exponential family for sample space with dimension `d` >=1, and parameter space with dimension `m` where
        of log partition function is solved via `n` Quasi Monte Carlo points (Halton low discrepancy points).

        Parameters
        ----------
        sample_space_dimension
        sparse_grid_level
        bijection
        statistics
        remaining_statistics
        statistics_vectorization_signature
        partition_vectorization_signature
        partition_integrand_vectorization_signature
        """

        super().__init__(sample_space_dimension=sample_space_dimension,
                         bijection=bijection,
                         statistics=statistics,
                         remaining_statistics=remaining_statistics)
        self._epsilon = epsilon
        self._spg_level = sparse_grid_level
        self._sRule = sRule
        if self._sRule.lower() in ["clenshaw-curtis", "gauss-patterson"]:
            if self._sRule == "gauss-patterson" and self._spg_level > 9:
                self._spg_level = 9
        else:
            self._sRule = "clenshaw-curtis"

        tasmanian_grid = Tasmanian.makeGlobalGrid(iDimension=self._sample_space_dim,
                                                  iOutputs=0, iDepth=self._spg_level,
                                                  sType="level", sRule=self._sRule)

        # points, weights = sparse_quadrature(dim=self._sample_space_dim, order=self._spg_level)
        points = tasmanian_grid.getPoints()
        weights = tasmanian_grid.getQuadratureWeights()
        points = jnp.atleast_2d(points)

        #   To avoid nan in bijection result
        mask = jnp.linalg.norm(points, ord=jnp.inf, axis=-1) < 1 - self._epsilon
        self._spg_weights = jnp.asarray(weights[mask], dtype=jnp.float32)
        self._quadrature_points = jnp.asarray(points[mask], dtype=jnp.float32)
        self._nodes_num = self._spg_weights.shape[0]
        self._bijected_points = self._bijection(self._quadrature_points)

        # This is OK since the bijection is assumed to be vectorized one dimensional
        self._bijected_volume = jnp.reshape(jnp.prod(self._dbijection(self._quadrature_points), axis=-1),
                                            (self._quadrature_points.shape[0], 1))

    @property
    def spg_weights(self):
        return self._spg_weights

    @property
    def nodes_number(self):
        return self._nodes_num

    @property
    def bijected_points(self):
        return self._bijected_points

    @property
    def quadrature_points(self):
        return self._quadrature_points

    @property
    def sparse_grid_level(self):
        return self._spg_level

    def numerical_integration(self, numerical_values: jnp.ndarray, axis=None):
        if not axis:
            result = self._spg_weights @ numerical_values
        else:
            result = jnp.mean(self._spg_weights @ numerical_values, axis=axis)
        return result

    def _compute_D_part_for_fisher_metric(self, c: jnp.ndarray, exp_c_theta: jnp.ndarray):
        D = jnp.einsum('k,ki,kj,k', exp_c_theta.ravel(), c, c, self._bijected_volume.ravel() * self._spg_weights)
        return D

    def tree_flatten(self):
        auxiliaries = None
        return (self._sample_space_dim, self._nodes_num, self._params_num), auxiliaries

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
