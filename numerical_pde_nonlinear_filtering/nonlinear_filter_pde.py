import jax.numpy as jnp
from jax import jit
from typing import Callable, Tuple, List, NoReturn
from jax.lax import scan
from abc import ABC, abstractmethod
from collections import namedtuple

SDECoeffs = namedtuple('SDECoeffs', ('a', 'b', 'x0'))


class NonLinearFilterPDE(ABC):
    def __init__(self,
                 grids: jnp.ndarray,
                 dx: jnp.ndarray,
                 dynamic_drift: Callable[[jnp.ndarray, float], jnp.ndarray],
                 dynamic_diffusion: Callable[[jnp.ndarray, float], jnp.ndarray],
                 measurement_drift: Callable[[jnp.ndarray, float], jnp.ndarray],
                 initial_condition: jnp.ndarray,
                 delta_t: float,
                 measurement_record: jnp.ndarray,
                 measurement_stdev: float = 1.,
                 ode_solver: str = 'EULER'):
        """
        Abstract class for finite-difference based solver of Zakai equation.

        Parameters
        ----------
        grids : jnp.ndarray
            Grid 
        dx : jnp.ndarray
            Grid step for each dimension array of shape [d]
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

        Raises
        ------
        ValueError
            [If the initial condition shape does not match the grid shape]
        """

        self._grids = grids
        self._dynamic_drift = dynamic_drift
        self._dynamic_diffusion = dynamic_diffusion
        self._measurement_drift = measurement_drift
        self._dt = delta_t
        self._dx = dx
        self._meas_stdev = measurement_stdev
        self._ode_solver = ode_solver
        if self._ode_solver.lower() == 'rk':
            self._one_step_fokker_planck = self._runge_kutta
        elif self._ode_solver.lower() == 'euler':
            self._one_step_fokker_planck = self._euler
        else:
            self._ode_solver = 'euler'
            self._one_step_fokker_planck = self._euler

        self._dimension = self._dx.shape[0]
        for dim in range(self._dimension):
            if grids.shape[dim] != initial_condition.shape[dim]:
                raise ValueError("Initial_condition.shape {} at dim {} does not match with grid.shape = {}".format(
                    initial_condition.shape[dim],
                    dim,
                    grids.shape[dim]))

        # self._initial_condition = initial_condition / jnp.sum(
        #     initial_condition * jnp.prod(self._dx))  # do normalizing
        self._initial_condition = self.normalize_density(initial_condition)

        self._measurement_record = measurement_record

    @abstractmethod
    def normalize_density(self, non_normalized_density: jnp.ndarray) -> jnp.ndarray:
        pass

    @property
    def dx(self) -> jnp.ndarray:
        return self._dx

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def measurement_stdev(self) -> float:
        return self._meas_stdev

    @property
    def measurement_record(self) -> jnp.ndarray:
        return self._measurement_record

    @property
    def grids(self) -> jnp.ndarray:
        return self._grids

    def propagate_fokker_plank(self):
        #   declare time
        t_end = self._measurement_record.shape[0] * self._dt
        t = jnp.linspace(self._dt, t_end, int(t_end / self._dt), endpoint=True)

        @jit
        def _to_be_scanned(car, inp):
            p_ = car
            t_ = inp
            p_ = self._one_step_fokker_planck(p_, t_)
            return p_, p_

        _, density_history = scan(_to_be_scanned, self._initial_condition, t)
        return density_history

    @abstractmethod
    def propagate_zakai(self) -> jnp.ndarray:
        """Propagate Zakai equation. To be implemented in the inherited class.

        Returns
        -------
        jnp.ndarray
            The evolution of density
        """
        pass

    @abstractmethod
    def _fokker_planck(self, p: jnp.ndarray, t: float) -> jnp.ndarray:
        """Compute :math: `dp/dt` for the Fokker-Planck equation. To be implemented in the inherited class.

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
        pass

    def _euler(self, p: jnp.ndarray, t: float) -> jnp.ndarray:
        """Euler's method
        """
        return p + self.dt * self._fokker_planck(p, t)

    def _runge_kutta(self, p: jnp.ndarray, t: float) -> jnp.ndarray:
        r"""4-th order Runge--Kutta

        .. math::

            dp/dt = f(p, t) dt

        Parameters
        ----------
        p : jnp.ndarray (n1, n2)
            Initial condition in matrix/vector.
        t : float
            Initial time.
        Returns
        -------
        p : jnp.ndarray (n1, n2)
            RK4 integrated ODE solution at :math:`t + \Delta t`.
        """
        k1 = self._fokker_planck(p, t)
        k2 = self._fokker_planck(p + 0.5 * self.dt * k1, t + 0.5 * self.dt)
        k3 = self._fokker_planck(p + 0.5 * self.dt * k2, t + 0.5 * self.dt)
        k4 = self._fokker_planck(p + self.dt * k3, t + self.dt)
        return p + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
