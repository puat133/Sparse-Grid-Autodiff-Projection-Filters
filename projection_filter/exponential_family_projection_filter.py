from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple

import numpy as onp
import sympy as sp
from jax import jit
from jax import lax
from jax import partial
import jax.numpy as jnp

from projection_filter import ExponentialFamily
from symbolic import SDE


class SStarProjectionFilter(ABC):
    def __init__(self,
                 dynamic_sde: SDE,
                 measurement_sde: SDE,
                 natural_statistics_symbolic: sp.MutableDenseMatrix,
                 constants: Dict,
                 initial_condition: jnp.ndarray,
                 measurement_record: jnp.ndarray,
                 delta_t: float,
                 bijection: Callable[[jnp.ndarray], jnp.ndarray] = jnp.arctanh,
                 ode_solver: str = 'RK'):
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

        self._dynamic_sde = dynamic_sde
        self._measurement_sde = measurement_sde
        self._natural_statistics_symbolic = natural_statistics_symbolic
        self._constants = constants
        self._measurement_record = measurement_record
        self._dt = delta_t
        self._ode_solver = ode_solver
        self._dy = jnp.diff(self._measurement_record, axis=0)
        self._re_scale_measurement()

        if self._dy.ndim != 2:
            self._dy = self._dy[:, jnp.newaxis]
        self._t = jnp.arange(self._measurement_record.shape[0]) * self._dt
        self._sample_space_dimension = len(self._dynamic_sde.variables)
        self._current_state = initial_condition
        self._state_history = self._current_state[jnp.newaxis, :]
        self._bijection = bijection
        self._exponential_density: ExponentialFamily = None
        self._L_0: jnp.ndarray = None
        self._ell_0: jnp.ndarray = None
        self._A_0: jnp.ndarray = None
        self._a_0: jnp.ndarray = None
        self._b_0: jnp.ndarray = None
        self._b_h: jnp.ndarray = None
        self._lamda: jnp.ndarray = None
        self._lambda_0: jnp.ndarray = None

        if self._ode_solver.lower() == 'rk':
            self._one_step_fokker_planck = self._runge_kutta
        elif self._ode_solver.lower() == 'euler':
            self._one_step_fokker_planck = self._euler
        else:
            self._ode_solver = 'euler'
            self._one_step_fokker_planck = self._euler

    def _re_scale_measurement(self):
        """
        The original projection filter is written for assumption that :math:`dV` is a standard Brownian

        .. math:: dy = h(x)dt + dV

        To Allow the measurement, with the same dV and non identity diffusion matrix :math:`G`

        .. math:: dy = h(x) dt + G dV

        then we need to use the scaled version: i.e

        .. math:: d \tilde{y} = \tilde{h} dt + \tilde{G} dV

        with :math:`\tilde{G}\tilde{G}^\top = I`. This is achieved by setting

        .. math:: d \tilde{y} = chol((GG^T)^{-1})
        Returns
        -------

        """
        g = self._measurement_sde.diffusions
        gg_T_inv = (g * g.transpose()).inv()
        gg_T_chol = gg_T_inv.cholesky()
        scaled_diffusion = gg_T_chol * g
        scaled_drift = gg_T_chol * self._measurement_sde.drifts
        scaled_measurement_SDE = SDE(drifts=scaled_drift, diffusions=scaled_diffusion, time=self._measurement_sde.time,
                                     variables=self._measurement_sde.variables,
                                     brownians=self._measurement_sde.brownians)
        gg_T_chol_np = jnp.atleast_2d(jnp.array(onp.array(gg_T_chol).astype(onp.float32)))
        if self._dy.ndim == 1:
            self._dy = gg_T_chol_np[0, 0] * self._dy
        elif self._dy.ndim == 2:
            self._dy = gg_T_chol_np[jnp.newaxis, :, :] @ self._dy[:, :, jnp.newaxis]
        self._dy = self._dy.squeeze()
        self._measurement_sde = scaled_measurement_SDE

    @partial(jit, static_argnums=[0, ])
    def _fokker_planck(self, theta_: jnp.ndarray, t: float):
        fisher_ = self._exponential_density.fisher_metric(theta_)
        eta_tilde_ = self._exponential_density.extended_statistics_expectation(theta_)
        dtheta_dt = jnp.linalg.solve(fisher_, self._ell_0 + self._L_0 @ eta_tilde_)
        return dtheta_dt

    @partial(jit, static_argnums=[0, ])
    def _runge_kutta(self, theta_: jnp.ndarray, t: float):
        k1 = self._fokker_planck(theta_, t)
        k2 = self._fokker_planck(theta_ + 0.5 * self._dt * k1, t + 0.5 * self._dt)
        k3 = self._fokker_planck(theta_ + 0.5 * self._dt * k2, t + 0.5 * self._dt)
        k4 = self._fokker_planck(theta_ + self._dt * k3, t + self._dt)
        new_theta = theta_ + self._dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return new_theta

    @partial(jit, static_argnums=[0, ])
    def _euler(self, theta_: jnp.ndarray, t: float):
        return theta_ + self._dt * self._fokker_planck(theta_, t)

    @property
    def projection_filter_matrices(self):
        return self._ell_0, self._L_0, self._a_0, self._A_0, self._b_h, self._lamda

    @property
    def ode_solver(self):
        return self._ode_solver

    @ode_solver.setter
    def ode_solver(self, value):
        if value.lower() == 'rk':
            self._ode_solver = value
            self._one_step_fokker_planck = self._runge_kutta
        elif value.lower() == 'euler':
            self._ode_solver = value
            self._one_step_fokker_planck = self._euler

    @property
    @abstractmethod
    def natural_statistics(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def remaining_statistics(self):
        raise NotImplementedError

    @property
    def current_state(self):
        return self._current_state

    @property
    def state_history(self):
        return self._state_history

    @property
    @abstractmethod
    def exponential_density(self):
        raise NotImplementedError

    @abstractmethod
    def get_density_values(self, grid_limits: jnp.ndarray, nb_of_points: jnp.ndarray):
        """
        get density values for given `grid_limits` as an array of `N_d x 2` and `nb_of_points` as an
        array of `N_d` integers
        Parameters
        ----------
        grid_limits
        nb_of_points

        Returns
        -------

        """
        raise NotImplementedError

    def propagate(self):
        self._propagate_filter()

    def solve_Fokker_Planck(self):
        self._propagate_FK()

    def _propagate_FK(self):
        @jit
        def integrator_loop(carry_, inputs_):
            t_ = inputs_
            theta_ = carry_

            theta_ = self._one_step_fokker_planck(theta_, t_)

            carry_ = theta_
            return carry_, carry_

        self._current_state, _history = lax.scan(integrator_loop, self._current_state, self._t[1:])
        self._state_history = _history

    def _propagate_filter(self):
        @jit
        def integrator_loop(carry_, inputs_):
            t_, dy_ = inputs_
            theta_ = carry_
            fisher_ = self._exponential_density.fisher_metric(theta_)
            eta_tilde_ = self._exponential_density.extended_statistics_expectation(theta_)
            eta_ = eta_tilde_[:self._exponential_density.params_num]
            dtheta_ = jnp.linalg.solve(fisher_, self._a_0 + self._b_0 * eta_ +
                                       (self._A_0 + jnp.outer(eta_, self._b_h)) @ eta_tilde_) * self._dt \
                      + self._lamda @ dy_
            carry_ = theta_ + dtheta_
            return carry_, carry_

        self._current_state, _history = lax.scan(integrator_loop, self._current_state, [self._t[1:],
                                                                                        self._dy])
        self._state_history = _history

    def discrete_propagate(self):
        """
        This is used in the case that the measurement is a discrete process
        Returns
        -------

        """
        self._propagate_filter_two_step()

    def _propagate_filter_two_step(self):
        @jit
        def integrator_loop(carry_, inputs_):
            t_, dy_ = inputs_
            theta_ = carry_

            # predictive update
            theta_ = self._one_step_fokker_planck(theta_, t_)
            # bayesian update
            theta_ += self._lamda @ dy_ - self._lambda_0

            carry_ = theta_
            return carry_, carry_

        self._current_state, _history = lax.scan(integrator_loop, self._current_state, [self._t[1:],
                                                                                        self._dy])
        self._state_history = _history

    @abstractmethod
    def _construct_remaining_statistics(self):
        raise NotImplementedError

    @abstractmethod
    def _get_projection_filter_matrices(self):
        """
        Get matrices related to a projection filter with an exponential family densities.

        Returns
        -------
        matrices: Tuple
            List of matrices containing M_0, m_h, and lamda
        """
        raise NotImplementedError
