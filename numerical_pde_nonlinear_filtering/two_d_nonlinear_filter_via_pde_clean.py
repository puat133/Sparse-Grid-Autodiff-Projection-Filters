from typing import Callable, Tuple
import jax.numpy as jnp
from jax import jit, partial
from jax.lax import scan

from numerical_pde_nonlinear_filtering import NonLinearFilterPDE


class TwoDNonLinearFilterPDE(NonLinearFilterPDE):
    def __init__(self,
                 one_d_grids: Tuple[jnp.ndarray, jnp.ndarray],
                 dynamic_drift: Callable[[jnp.ndarray, float], jnp.ndarray],
                 dynamic_diffusion: Callable[[jnp.ndarray, float], jnp.ndarray],
                 measurement_drift: Callable[[jnp.ndarray, float], jnp.ndarray],
                 initial_condition: jnp.ndarray,
                 delta_t: float,
                 measurement_record: jnp.ndarray,
                 measurement_stdev: float = 1.,
                 ode_solver: str = 'EULER'
                 ):
        """Class that encapsulates a two dimensional optimal filtering problem PDE solution (Zakai - equation)
        using finite difference scheme [1].
        The grid is assumed to be equidistance, and the finite difference is order one center.
        

        Parameters
        ----------
        one_d_grids : Tuple[jnp.ndarray, jnp.ndarray]
            tuples of d one dimensional grids, each has shape [Ng]
        dynamic_drift : Callable[[jnp.ndarray, float], jnp.ndarray]
            Drift part of the state dynamic sde
        dynamic_diffusion : Callable[[jnp.ndarray, float], jnp.ndarray]
            Diffusion part of the state dynamic sde
        measurement_drift : Callable[[jnp.ndarray, float], jnp.ndarray]
            Drift part of the measurement sde
        initial_condition : jnp.ndarray
            Initial condition, the array should match the shape of the grids
        delta_t : float
            Time step
        measurement_record : jnp.ndarray
            Measurement records array of shape [Nt x m]
        measurement_stdev : float, optional
            measurement standard deviation, by default 1.
        ode_solver : str, optional
            ODE solver option, by default 'EULER'

        References
        ----------
        [1] Alan Bain , Dan Crisan, Fundamentals of Stochastic Filtering
        """
        dx, grids = TwoDNonLinearFilterPDE._create_two_dimensional_grid(
            one_d_grids)

        super().__init__(grids=grids,
                         dx=dx,
                         dynamic_drift=dynamic_drift,
                         dynamic_diffusion=dynamic_diffusion,
                         measurement_drift=measurement_drift,
                         initial_condition=initial_condition,
                         delta_t=delta_t,
                         measurement_record=measurement_record,
                         measurement_stdev=measurement_stdev,
                         ode_solver=ode_solver
                         )

        self._measurement_record = measurement_record
        self._dy = jnp.diff(self._measurement_record, axis=0)

    @staticmethod
    def _create_two_dimensional_grid(one_d_grids: Tuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ create two dimensional grid

        Parameters
        ----------
        one_d_grids : Tuple
            tuples of d one dimensional grids, each has shape [Ng]

        Returns
        -------
        Tuple[jnp.ndarray,jnp.ndarray]
            grid_size, and grids of shape [Ng x Ng x 2]
        """
        grid_1, grid_2 = one_d_grids
        dx1 = grid_1[1] - grid_1[0]
        dx2 = grid_2[1] - grid_2[0]
        dx = jnp.array([dx1, dx2])
        grid_limits = jnp.array(
            [[grid_1[0], grid_1[-1]], [grid_2[0], grid_2[-1]]])
        num_points = jnp.array(
            [grid_1.shape[0], grid_2.shape[0]], dtype=jnp.int32)
        # create a meshgrid
        x_ = []
        for i in range(2):
            temp_ = jnp.linspace(
                grid_limits[i, 0], grid_limits[i, 1], num_points[i], endpoint=True)
            x_.append(temp_)
        grids = jnp.meshgrid(*x_, indexing='ij')
        grids = jnp.stack(grids, axis=-1)
        return dx, grids

    @partial(jit, static_argnums=(0,))
    def normalize_density(self, non_normalized_density: jnp.ndarray) -> jnp.ndarray:
        """Normalize a probability density

        Parameters
        ----------
        non_normalized_density : jnp.ndarray
            Non normalized density [Ng x Ng]

        Returns
        -------
        jnp.ndarray
            Normalized density [Ng x Ng]
        """
        return non_normalized_density / jnp.trapz(jnp.trapz(non_normalized_density,
                                                            dx=self.dx[0], axis=0), dx=self.dx[1])

    @partial(jit, static_argnums=(0,))
    def _fokker_planck(self, p: jnp.ndarray, t: float) -> jnp.ndarray:
        """Compute :math: `dp/dt` of the Fokker-Planck equation for two dimensional problem. 

        Parameters
        ----------
        p : jnp.ndarray
            probability density at the moment
        t : float
            time at the moment

        Returns
        -------
        jnp.ndarray
            :math: `dp/dt`
        """
        f_eval = self._dynamic_drift(self._grids, t)
        g_eval = self._dynamic_diffusion(self._grids, t)
        D_eval = 0.5 * jnp.einsum('ijkl,ijml->ijkm', g_eval, g_eval)
        Dp_eval = D_eval * p[:, :, jnp.newaxis, jnp.newaxis]
        fp_eval = f_eval * p[:, :, jnp.newaxis]

        dfpdx0 = jnp.gradient(fp_eval[:, :, 0], self._dx[0], axis=0)
        dfpdx1 = jnp.gradient(fp_eval[:, :, 1], self._dx[1], axis=1)

        dd_Dp_dx0dx1 = jnp.gradient(jnp.gradient(Dp_eval[:, :, 0, 1], self._dx[0], axis=0),
                                    self._dx[1], axis=1)
        dd_Dp_dx1dx0 = jnp.gradient(jnp.gradient(Dp_eval[:, :, 1, 0], self._dx[1], axis=1),
                                    self._dx[0], axis=0)

        mid_result = (Dp_eval[:-2, :, 0, 0] - 2 * Dp_eval[1:-1, :, 0, 0] + Dp_eval[2:, :, 0, 0]) / (self._dx[0] *
                                                                                                    self._dx[0])
        dd_Dp_dx0dx0 = jnp.block(
            [[mid_result[0, :]], [mid_result], [mid_result[-1, :]]])

        mid_result = (Dp_eval[:, :-2, 1, 1] - 2 * Dp_eval[:, 1:-1, 1, 1] + Dp_eval[:, 2:, 1, 1]) / (self._dx[1] *
                                                                                                    self._dx[1])
        dd_Dp_dx1dx1 = jnp.block(
            [[mid_result[:, 0, jnp.newaxis], mid_result, mid_result[:, -1, jnp.newaxis]]])

        result = -(dfpdx0 + dfpdx1) + (dd_Dp_dx0dx0 +
                                       dd_Dp_dx1dx1 + dd_Dp_dx0dx1 + dd_Dp_dx1dx0)

        return result

    def propagate_zakai(self) -> jnp.ndarray:
        """Realization of `propagate_zakai` for two dimensional problem

        Returns
        -------
        jnp.ndarray
            Next normalized probability density.
        """
        t_end = self._measurement_record.shape[0] * self._dt
        t = jnp.linspace(self._dt, t_end, self._dy.shape[0], endpoint=True)
        inputs = t, self._dy

        @jit
        def scan_body_ks(carry, inp):
            p_ = carry
            t_, dy_ = inp

            #   use Kalliajnpur–Striebel formula
            p_ = self._one_step_fokker_planck(p_, t_)

            h_ = jnp.squeeze(self._measurement_drift(self._grids, t_))
            psi_ = jnp.exp(-jnp.square((dy_ - h_ * self._dt)) /
                           (2 * self._meas_stdev * self._meas_stdev * self._dt))
            p_ = p_ * psi_

            p_ = self.normalize_density(p_)
            return p_, p_

        _, all_ps = scan(scan_body_ks, self._initial_condition, inputs)

        return all_ps


