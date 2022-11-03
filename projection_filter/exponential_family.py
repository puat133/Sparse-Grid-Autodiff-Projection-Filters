from abc import ABC, abstractmethod
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import jit, partial


class ExponentialFamily(ABC):
    """
    This class encapsulate the exponential family concept, where the log_paritition of this family is computed
    via Numerical Integration and bijection.
    """

    def __init__(self,
                 sample_space_dimension: int,
                 bijection: Callable[[jnp.ndarray], jnp.ndarray],
                 statistics: Callable[[jnp.ndarray], jnp.ndarray],
                 remaining_statistics: [[jnp.ndarray], jnp.ndarray] = None,
                 statistics_vectorization_signature: str = '()->(n)',
                 partition_vectorization_signature: str = '(n)->()',
                 partition_integrand_vectorization_signature: str = '(n),(m)->(m)'):

        self._sample_space_dim = sample_space_dimension
        self._bijection = jnp.vectorize(bijection)
        self._dbijection = jnp.vectorize(jax.grad(bijection))

        self._stats_vect_sign = statistics_vectorization_signature
        self._par_integrand_vect_sign = partition_integrand_vectorization_signature
        self._par_vect_sign = partition_vectorization_signature

        self._natural_statistics = jit(jnp.vectorize(statistics, signature=self._stats_vect_sign))

        if sample_space_dimension == 1:
            temp = self._natural_statistics(1.)
        else:
            temp = self._natural_statistics(jnp.zeros(self._sample_space_dim))

        self._params_num = temp.shape[0]
        self._remaining_moments_num = 0

        if remaining_statistics:
            self._remaining_moments = jit(jnp.vectorize(remaining_statistics, signature=self._stats_vect_sign))

            def extended_statistics(x):
                return jnp.concatenate((self._natural_statistics(x), self._remaining_moments(x)))

            self._extended_statistics = jit(jnp.vectorize(extended_statistics, signature=self._stats_vect_sign))

            if sample_space_dimension == 1:
                temp = self._remaining_moments(1.)
            else:
                temp = self._remaining_moments(jnp.ones(self._sample_space_dim))

            self._remaining_moments_num = temp.shape[0]
        else:
            self._extended_statistics = self._natural_statistics
            self._remaining_moments_num = 0

        def partition_integrand(theta, x):
            return jnp.prod(self._dbijection(x)) * jnp.exp(self._natural_statistics(self._bijection(x)) @ theta)

        def partition_integrand_extended(theta_extended, x):
            return jnp.prod(self._dbijection(x)) * jnp.exp(
                self._extended_statistics(self._bijection(x)) @ theta_extended)

        self._partition_integrand = jit(jnp.vectorize(partition_integrand,
                                                      signature=self._par_integrand_vect_sign))
        self._partition_integrand_extended = jit(jnp.vectorize(partition_integrand_extended,
                                                               signature=self._par_integrand_vect_sign))

        def log_partition(theta):
            return jnp.log(self.integrate_partition(theta))

        def log_partition_extended(theta_extended):
            """
            Formula 8.1
            """
            return jnp.log(self.integrate_partition_extended(theta_extended))

        self._log_partition = jit(jnp.vectorize(log_partition, signature=self._par_vect_sign))
        self._log_partition_jac = jax.jacobian(self._log_partition)
        self._log_partition_hess = jax.hessian(self._log_partition)

        self._log_partition_extended = jit(jnp.vectorize(log_partition_extended,
                                                         signature=self._par_vect_sign))
        self._log_partition_extended_jac = jax.jacobian(self._log_partition_extended)
        self._log_partition_extended_hess = jax.hessian(self._log_partition_extended)

    @property
    def sample_space_dim(self):
        return self._sample_space_dim

    @property
    def params_num(self):
        return self._params_num

    @property
    def remaining_moments_num(self):
        return self._remaining_moments_num

    @property
    def bijection(self):
        return self._bijection

    @property
    def natural_statistics(self):
        return self._natural_statistics

    @property
    def higher_moments(self):
        return self._remaining_moments

    @property
    def partition_integrand(self):
        return self._partition_integrand

    @property
    def partition_extended_integrand(self):
        return self._partition_integrand_extended

    @property
    def log_partition(self):
        return self._log_partition

    @property
    def log_partition_extended(self):
        return self._log_partition_extended

    @partial(jnp.vectorize, signature='(m)->(n)', excluded=[0, ])
    @partial(jit, static_argnums=[0, ])
    def natural_statistics_expectation(self, theta):
        return self._log_partition_jac(theta)

    @partial(jnp.vectorize, signature='(m)->(n)', excluded=[0, ])
    @partial(jit, static_argnums=[0, ])
    def extended_statistics_expectation(self, theta):
        return self._log_partition_extended_jac(jnp.pad(theta, (0, self._remaining_moments_num)))

    @partial(jnp.vectorize, signature='(m)->(m,m)', excluded=[0, ])
    @partial(jit, static_argnums=[0, ])
    def fisher_metric(self, theta):
        return self._log_partition_hess(theta)

    @abstractmethod
    def get_density_values(self, grid_limits: jnp.ndarray, theta: jnp.ndarray, nb_of_points: jnp.ndarray) -> \
            Tuple[jnp.ndarray, jnp.ndarray]:
        """
        get density values for given `theta` on `grid_limits` as an array of `N_d x 2` and `nb_of_points` as an
        array of `N_d` integers

        Parameters
        ----------
        grid_limits : Tuple
        theta : ndarray
        nb_of_points    : int

        Returns
        -------
        result  : ndarray, ndarray
            x_grid and the density result.
        """
        raise NotImplementedError

    @abstractmethod
    def integrate_partition(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        This is where the children class need to implement their numerical integration to
        obtain the integration of log partition function at theta in R^d

        Parameters
        ----------
        theta: ndarray
            exponential family natural parameter

        Returns
        -------
        res: ndarray
            integration result
        """
        raise NotImplementedError

    @abstractmethod
    def integrate_partition_extended(self, theta_extended: jnp.ndarray) -> jnp.ndarray:
        """
        This is where the children class need to implement their numerical integration to
        obtain the integration of log partition extended function at theta in R^d

        Parameters
        ----------
        theta_extended: ndarray
            extended exponential family natural parameter

        Returns
        -------
        res: ndarray
            integration result
        """
        raise NotImplementedError
