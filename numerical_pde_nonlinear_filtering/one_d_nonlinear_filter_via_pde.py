from typing import Callable

import jax.numpy as jnp
from jax import jit, partial
from jax.lax import scan

from numerical_pde_nonlinear_filtering import NonLinearFilterPDE


class OneDNonLinearFilterPDE(NonLinearFilterPDE):
    def __init__(self, grids: jnp.ndarray,
                 dynamic_drift: Callable[[jnp.ndarray, float], jnp.ndarray],
                 dynamic_diffusion: Callable[[jnp.ndarray, float], jnp.ndarray],
                 measurement_drift: Callable[[jnp.ndarray, float], jnp.ndarray],
                 initial_condition: jnp.ndarray,
                 delta_t: float,
                 measurement_record: jnp.ndarray,
                 measurement_stdev: float = 1.):
        """
        Class that encapsulates a one dimensional optimal filtering problem PDE solution (Zakai - equation)
        using finite difference scheme [1].
        The grid is assumed to be equidistance, and the finite difference is order one center.
        Only one dimensional measurement is supported at the moment.


        Parameters
        ----------
        grids : array
        dynamic_drift: Callable
        dynamic_diffusion: Callable
        measurement_drift: Callable
        initial_condition: array
        delta_t: float
        measurement_record: array
        measurement_stdev: float

        References
        ----------
        [1] Alan Bain , Dan Crisan, Fundamentals of Stochastic Filtering
        """

        dx_ = jnp.array([grids[1] - grids[0]])
        super().__init__(grids=grids,
                         dx=dx_,
                         dynamic_drift=dynamic_drift,
                         dynamic_diffusion=dynamic_diffusion,
                         measurement_drift=measurement_drift,
                         initial_condition=initial_condition,
                         delta_t=delta_t,
                         measurement_record=measurement_record,
                         measurement_stdev=measurement_stdev
                         )

        if measurement_record.ndim != 1:
            raise NotImplementedError("Having multiple measurement is not supported yet!")
        else:
            self._measurement_record = measurement_record
            self._dy = jnp.diff(self._measurement_record, append=self._measurement_record[-1, jnp.newaxis])

        # self._next_nonnormalized_predictive_density = _euler

    @partial(jit, static_argnums=(0,))
    def normalize_density(self, non_normalized_density: jnp.ndarray)-> jnp.ndarray:
        return non_normalized_density / jnp.trapz(non_normalized_density, dx=self._dx)

    @partial(jit, static_argnums=(0,))
    def _fokker_planck(self, p: jnp.ndarray, t: float):
        f_eval = jnp.squeeze(self._dynamic_drift(self._grids, t))
        sigma_eval = jnp.squeeze(self._dynamic_diffusion(self._grids, t))
        fp = f_eval * p
        sigma_p_sigma = sigma_eval * sigma_eval * p
        dd_sigma_p_sigma_mid = (sigma_p_sigma[:-2] - 2 * sigma_p_sigma[1:-1] + sigma_p_sigma[2:]) / (
                self._dx * self._dx)
        dd_sigma_p_sigma = jnp.concatenate((dd_sigma_p_sigma_mid[0, jnp.newaxis],
                                            dd_sigma_p_sigma_mid,
                                            dd_sigma_p_sigma_mid[-1, jnp.newaxis]))
        temp = - jnp.gradient(fp, self._dx[0], axis=0)
        temp += 0.5 * dd_sigma_p_sigma  # TODO: be check again

        return temp

    def propagate_zakai(self):
        t_end = self._measurement_record.shape[0] * self._dt
        t = jnp.linspace(self._dt, t_end, self._measurement_record.shape[0], endpoint=True)
        inputs = t, self._dy

        @jit
        def _to_be_scanned(car, inp):
            p_ = car
            t_, dy_ = inp
            # not using Kalliajnpur-Striebel
            # h_ = jnp.squeeze(self._measurement_drift(self._grid[0], t_))
            # dp_ = self._fokker_planck(p_, t_) * self._dt + p_ * h_ * dy_ / self._meas_stdev
            # p_ = p_ + dp_
            # p_ =  p_ / jnp.trapz(p_, dx=self._dx)

            #   use Kalliajnpur–Striebel formula
            p_ = self._one_step_fokker_planck(p_, t_)
            # p_ = self.normalize_density(p_)

            h_ = jnp.squeeze(self._measurement_drift(self._grids, t_))
            psi_ = jnp.exp(-jnp.square((dy_ - h_ * self._dt)) / (2 * self._meas_stdev * self._meas_stdev * self._dt))
            p_ = p_ * psi_

            p_ = self.normalize_density(p_)

            return p_, p_

        _, density_history = scan(_to_be_scanned, self._initial_condition, inputs)

        #   final normalizing
        # density_history = density_history / jnp.sum(density_history * self._dx, axis=1)[:, jnp.newaxis]
        return density_history
