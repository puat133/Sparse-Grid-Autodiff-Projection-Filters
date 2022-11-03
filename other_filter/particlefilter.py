from other_filter.basefilter import FilterBase
from typing import Callable, NamedTuple, Tuple
from other_filter.resampling import log_gaussian_density, log_mean_exp, systematic
import jax.numpy as jnp
from jax import jit, lax, partial, pmap, vmap
import jax.random as jrandom


# The jitted function ___ includes a pmap. Using jit-of-pmap can lead to inefficient data
# movement, as the outer jit does not preserve sharded data representations and instead collects input and output
# arrays onto a single device. Consider removing the outer jit unless you know what you're doing

class ParticleFilter(FilterBase):
    def __init__(self,
                 n_devices: int,
                 n_particle_per_device: int,
                 state_init: jnp.ndarray,
                 measurement_history: jnp.ndarray,
                 transition_fun: Callable,
                 output_fun: Callable,
                 constraint: Callable,
                 process_cov: jnp.ndarray,
                 meas_cov: jnp.ndarray,
                 prng_key: jnp.ndarray):
        """
            Particle filter implementation with systematic resampling accross multiple-core via `jax.pmap`.
            Both process and measurement noises are assumed to be Gaussians. If the constraint is specified, then the
            constraint will be evaluated so that all weights that resulted from the calculation of the likelihood function
            :math: `p(y_k| x[k|k])` will be assigned value 0 if the :math: `x[k|k]` lies outside the constraint [1].

            Parameters
            ----------

            n_devices   : int
                how many device. Should not be more than the length of `jax.devices()`
            n_particle_per_device: int
                how many particle to be per device
            state_init  : (N_state) np.ndarray
                initial state
            input_history   : (N_time x N_input) np.ndarray
                input record
            measurement_history : (N_time x N_measurement) np.ndarray
                measurement record
            transition_fun  : Callable
                transition function
            output_fun  : Callable
                output function
            constraint  : Callable
                constraint evaluation on the particles state. Should receive (N_state) array and outputs boolean.
            dynamic_parameters  : NamedTuple
                parameters for dynamics
            process_cov : (N_state x N_state) np.ndarray
                process error covariance
            meas_cov    : (N_measurement x N_measurement) np.ndarray
                measurement error covariance
            prng_key    :  np.ndarray
                jax.random.PRNGkey
            Returns
            -------
            result : Tuple
                neg_log_likelihood_end, state_end, x_particle_history, neg_likelihood_history, log_weights_history, \
                   estimated_state_history

            References
            ----------
            [1] A Particle Filtering Approach To Constrained Motion Estimation In Tracking Multiple Targets

        """
        super().__init__(measurement_history=measurement_history,
                         transition_fun=transition_fun,
                         output_fun=output_fun,
                         constraint=constraint
                         )
        self._state_init = state_init
        self._process_cov = process_cov
        self._meas_cov = meas_cov
        self._prng_key = prng_key
        self._n_devices = n_devices
        self._n_particle_per_device = n_particle_per_device
        self._prng_key, subkey = jrandom.split(self._prng_key)

        self._q_particle = jnp.sqrt(jnp.diag(self._process_cov))*jrandom.normal(subkey,
                                                                                (self._measurement_history.shape[0],
                                                                                 self._n_devices,
                                                                                 self._n_particle_per_device,
                                                                                 self._state_init.shape[-1]))

        self._prng_key, subkey = jrandom.split(self._prng_key)
        self._uniforms = jrandom.uniform(subkey, (self._measurement_history.shape[0],))

        self._meas_cov_inv = jnp.linalg.solve(self._meas_cov, jnp.eye(self._meas_cov.shape[0]))

        @jit
        @partial(jnp.vectorize, signature='(n),(n)->(n),()', excluded=[2])
        def _parallelized_routine(x_particle_, q_particle_, y):
            x_particle_ = self._transition_fun(x_particle_)
            x_particle_ += q_particle_
            out = self._output_fun(x_particle_)
            log_weights_ = log_gaussian_density(y - out, self._meas_cov_inv)
            return x_particle_, log_weights_

        # @jit
        def _particle_filter_body(carry_: Tuple, inputs_: Tuple):
            x_particle_, x_, log_weights_, neg_likelihood_ = carry_
            y_, q_particle_, uni_ = inputs_
            # log_weights_ = np.where(constraint(x_particle_), log_weights_, -np.inf)
            x_particle_resampled_ = systematic(x_particle_, log_weights_, uni_)
            x_ = jnp.mean(x_particle_resampled_, axis=(0, 1))  # take the mean over here.
            x_particle_, log_weights_ = _parallelized_routine(x_particle_resampled_, q_particle_, y_)
            # x_particle_, log_weights_ = pmap(_parallelized_routine,
            #                                  static_broadcasted_argnums=[2, ],
            #                                  axis_name='device_axis')(x_particle_resampled_, q_particle_, y_)

            neg_likelihood_ -= log_mean_exp(log_weights_.ravel())
            return (x_particle_, x_, log_weights_, neg_likelihood_), (
                x_particle_resampled_, log_weights_, x_, neg_likelihood_)

        self._particle_filter_body = _particle_filter_body

    def run(self):
        if self._state_init.ndim == 1:
            x_particle_init = self._state_init + self._q_particle[0]
        elif self._state_init.shape == self._q_particle[0].shape:
            x_particle_init = self._state_init
            self._state_init = jnp.mean(self._state_init, axis=(0, 1))
        # no need to be normalized
        log_weigths_init = jnp.zeros((self._n_devices, self._n_particle_per_device), dtype=jnp.float32)

        neg_likelihood_init = 0

        carry = (x_particle_init, self._state_init, log_weigths_init, neg_likelihood_init)

        inputs = (self._measurement_history, self._q_particle, self._uniforms)

        (x_particle_end, state_end, log_weights_end, neg_log_likelihood_end), res = lax.scan(self._particle_filter_body,
                                                                                             carry,
                                                                                             inputs)

        x_particle_history, log_weights_history, estimated_state_history, neg_likelihood_history = res

        return neg_log_likelihood_end, state_end, x_particle_history, neg_likelihood_history, log_weights_history, \
               estimated_state_history

    @property
    def prngkey(self):
        return self._prng_key
