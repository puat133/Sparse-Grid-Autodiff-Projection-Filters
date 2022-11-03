import jax
import math
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax import jit
from jax.lax import scan
from functools import partial
from typing import Callable, Tuple, List
from .nonlinear_filter_pde import NonLinearFilterPDE, SDECoeffs


class TwoDNonLinearFilterPDE():
    r"""Non-linear filtering by solving Zakai's equation in 2D

    .. math::

        dx = a(x, t) dt + b(x, t) dW,

        dy = h(x, t) dt + r dB,

    where x \in \R^2. Aim is to estimate the unnormlised density :math:`p(x, t | y_t)` satisfying Zakai's equation

    .. math::

        d p = (-\sum^2_i \frac{\partial}{\partial x_i} [a_i p]
        + 1/2 \sum^2_{ij} \frac{\partial^2}{\partial x_i \partial x_j} [b b^T p]) dt
        + p h^T R^{-1} dz

    Then normalise p(x, t | y_t) by numerical integration, or use Kallianpur–Striebel formula.
    """
    def __init__(self,
                 grids: Tuple[jnp.ndarray, jnp.ndarray],
                 dynamic_drift: Callable[[jnp.ndarray, float], jnp.ndarray],
                 dynamic_diffusion: Callable[[jnp.ndarray, float], jnp.ndarray],
                 measurement_drift: Callable[[jnp.ndarray, float], jnp.ndarray],
                 initial_condition: jnp.ndarray,
                 delta_t: float,
                 measurement_record: jnp.ndarray,
                 measurement_stdev: float = 1.,
                 true_x: jnp.ndarray = None):
        r"""Init

        Parameters
        ----------
        grids : Tuple[ndarray] ((n1, ), (n2, ))
            Tuple of spatial grids.
        dynamic_drift : Callable
            SDE drift function. Shape follows ((n1, n2, 2), float) -> (n1, n2, 2)
        dynamic_diffusion : Callable
            SDE dispersion function. Shape follows ((n1, n2, 2), float) -> (n1, n2, 2, 2)
        measurement_drift : Callable
            Measurement SDE drift function. Shape follows ((n1, n2, 2), float) -> (n1, n2, 2)
        initial_condition : ndarray (n1, n2)
            Probability density of the initial condition at the grid. Assumed to be normalised.
        delta_t : float
            :math:`\Delta t`.
        measurement_record : ndarray (m+1, n_y)
            A sequence of measurements of y(t) (with the first element as the initial measurement value).
        measurement_stdev : float
            :math:`r`.
        """
        self.sde_x = SDECoeffs(dynamic_drift, dynamic_diffusion, None)
        self.sde_y = SDECoeffs(measurement_drift, measurement_stdev, None)
        self.init_p = initial_condition

        self.y = measurement_record
        self.dy = jnp.diff(measurement_record, axis=0)
        self.measurement_length = measurement_record.shape[0] - 1

        self.dt = delta_t
        self.ts = jnp.concatenate([jnp.array([0.]),
                                   jnp.cumsum(delta_t * jnp.ones((self.measurement_length-1, )))])

        self.dx1 = jnp.mean(jnp.diff(grids[0]))
        self.dx2 = jnp.mean(jnp.diff(grids[1]))
        self.meshgrid = jnp.meshgrid(grids[0], grids[1], indexing='ij') # Do note the xy or ij order
        self.grid = self.gen_2d_grid(grids[0], grids[1])
        self.grid_pad = self.gen_2d_grid(jnp.pad(grids[0], pad_width=(1, 1),
                                                 mode='constant',
                                                 constant_values=(grids[0][0] - self.dx1,
                                                                  grids[0][-1] + self.dx1)),
                                         jnp.pad(grids[1], pad_width=(1, 1),
                                                 mode='constant',
                                                 constant_values=(grids[1][0] - self.dx2,
                                                                  grids[1][-1] + self.dx2)))

        self.true_x = true_x

    @partial(jit, static_argnums=(0, ))
    def fpk(self, p: jnp.ndarray, t: float) -> jnp.ndarray:
        r"""Fokker--Planck--Kolmogorov equation

        .. math::

            A(p) = -d/dx1 (a1 p) - d/dx2 (a2 p)
            + 1/2 (d^2/dx1x1 ((b b^T)_{11} p) + d^2/dx1x2 ((b b^T)_{12} p)
                              + d^2/dx2x1 ((b b^T)_{21} p) + d^2/dx2x2 ((b b^T)_{22} p)

        Parameters
        ----------
        p : jnp.ndarray (n1, n2)
            Previous :code:`p(x, t)` at the grid and time :code:`t`.
        t : float
            Previous time instance.

        Returns
        -------
        Ap : jnp.ndarray (n1, n2)
            Approximated :math:`A(p)(x, t)` at the grid and time :code:`t`.
        """
        ax = self.sde_x.a(self.grid_pad, t) # Out shape: (n1+2, n2+2, 2)
        bx = self.sde_x.b(self.grid_pad, t) # Out shape: (n1+2, n2+2, 2, nw)
        bbx = bx @ jnp.transpose(bx, axes=(0, 1, 3, 2)) # Out shape: (n1+2, n2+2, 2, 2)

        # Pad p boundaries with zero   Out shape: (n1+2, n2+2)
        # Note: Here the padding of
        p = jnp.pad(p, pad_width=(1, 1), mode='edge')
        # p = jnp.pad(p, pad_width=(1, 1), mode='constant', constant_values=(0., 0.))
        # p = self.extrapolate_2d(p)

        return - self.fd_2d(ax[:, :, 0] * p, (self.dx1, self.dx2), mode='d/dx1') \
               - self.fd_2d(ax[:, :, 1] * p, (self.dx1, self.dx2), mode='d/dx2') \
               + 0.5 * (self.fd_2d(bbx[:, :, 0, 0] * p, (self.dx1, self.dx2), mode='d^2/dx1^2')
                        + self.fd_2d(bbx[:, :, 1, 1] * p, (self.dx1, self.dx2), mode='d^2/dx2^2')
                        + 2 * self.fd_2d(bbx[:, :, 1, 0] * p, (self.dx1, self.dx2), mode='d^2/dx1dx2')
                        )

    def propagate_zakai(self,
                        int_steps: int,
                        int_method: str = 'euler',
                        mode: str = 'k-s') -> jnp.ndarray:
        """Solve Zakai's equation.

        Parameters
        ----------
        int_steps : int
            Number of integration steps of FPK. If int_steps = 10 and dt = 1, then the program will do 10 times FPK
            with dt = 0.1 for each measurement.
        int_method : str, default='rk4'
            Temporal integration method. Options are 'euler' and 'rk4' (default).
        mode : str, default='k-s'
            Set :code:`k-s` to use Kallianpur–Striebel formula, otherwise use default method.

        Returns
        -------
        all_ps : jnp.ndarray (m, n1, n2)
            History of p(x, t) for all t at measurement points.

        References
        ----------
        Alan Bain , Dan Crisan, Fundamentals of Stochastic Filtering, pp. 207
        """
        if int_method == 'rk4':
            solver = self.rk4
        else:
            solver = self.euler

        @jit
        def scan_body_ks(carry, elem):
            p = carry
            t, dy = elem

            res = jnp.sum((self.sde_y.a(self.grid, t) * self.dt - dy) ** 2, axis=-1)
            psi = jnp.exp(-res / (2 * self.sde_y.b ** 2 * self.dt))
            p = psi * self.propogate_fpk(p, t, self.dt / int_steps, int_steps, solver)
            p = p / jnp.trapz(jnp.trapz(p, dx=self.dx1, axis=0), dx=self.dx2)
            return p, p

        _, all_ps = scan(scan_body_ks, self.init_p, (self.ts, self.dy))

        return all_ps

    def propagate_zakai_debug(self,
                        t_int_method: str = 'euler',
                        mode: str = 'k-s') -> jnp.ndarray:
        """For debug/test use only."""
        if t_int_method == 'rk4':
            solver = self.rk4
        else:
            solver = self.euler

        p = self.init_p
        if mode == 'k-s':
            for idx in range(self.measurement_length):
                t = self.ts[idx]
                dy = self.dy[idx]
                res = jnp.sum((self.sde_y.a(self.grid, t) * self.dt - dy) ** 2, axis=-1)
                psi = jnp.exp(-res / (2 * self.sde_y.b ** 2 * self.dt))
                p = psi * solver(self.fpk, p, t, self.dt)
                p = p / jnp.trapz(jnp.trapz(p, dx=self.dx1, axis=0), dx=self.dx2)

                if idx % 1000 == 0:
                    plt.contour(self.meshgrid[0], self.meshgrid[1], p, levels=20)
                    plt.scatter(self.true_x[idx, 0], self.true_x[idx, 1], s=100, marker='x')
                    plt.xlabel('x1')
                    plt.ylabel('x2')
                    plt.savefig(f'test_{idx}.jpg')
                    plt.close()

        else:
            for idx in range(self.measurement_length):
                t = self.ts[idx]
                dy = self.dy[idx]
                p = solver(self.fpk, p, t, self.dt) + p * (self.sde_y.a(self.grid, t) @ dy) / self.sde_y.b ** 2
                p = p / jnp.trapz(jnp.trapz(p, dx=self.dx1, axis=0), dx=self.dx2)

                if idx % 1000 == 0:
                    plt.contour(self.meshgrid[0], self.meshgrid[1], p, levels=20)
                    plt.scatter(self.true_x[idx, 0], self.true_x[idx, 1], s=100, marker='x')
                    plt.xlabel('x1')
                    plt.ylabel('x2')
                    plt.savefig(f'test_{idx}.jpg')
                    plt.close()

        return p

    @partial(jit, static_argnums=(0, 4, 5, ))
    def propogate_fpk(self, p: jnp.ndarray, t: float, dt: float, n: int, int_method: Callable):
        """Solve FPK.

        Parameters
        ----------
        p : jnp.ndarray (n1, n2)
            InitialP PDF.
        t : float
            Time. (Note that time-inhomogenous case is not implemented)
        dt: float
            Time interval.
        n: int
            Number of steps.
        int_method : Callable
            self.euler or self.rk4

        Returns
        -------
        p : jnp.ndarray (n1, n2)
            FPK solution after n steps.
        """
        @jit
        def scan_body(carry, elem):
            p = carry
            _ = elem

            p = int_method(self.fpk, p, t, dt)
            p = p / jnp.trapz(jnp.trapz(p, dx=self.dx1, axis=0), dx=self.dx2)
            return p, p

        p, _ = scan(scan_body, p, jnp.arange(n))
        return p

    def debug_fpk(self, t_int_method: str = 'rk4') -> jnp.ndarray:
        """For debug/test only."""
        if t_int_method == 'rk4':
            solver = self.rk4
        else:
            solver = self.euler

        # all_ps = jnp.zeros(shape=(1, *self.init_p.shape))

        p = self.init_p
        for idx in range(self.measurement_length):
            t = self.ts[idx]
            p = solver(self.fpk, p, t, self.dt)
            p = p / jnp.trapz(jnp.trapz(p, dx=self.dx1, axis=0), dx=self.dx2)

            # all_ps = jnp.vstack((all_ps, p[None, :, :]))
        # return all_ps
        return p

    @staticmethod
    def gen_2d_grid(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """Generate 2D grid given spatial domains in each argument

        Parameters
        ----------
        x1 : jnp.ndarray (n1, )
            Grid of the first argument
        x2 : jnp.ndarray (n2, )
            Grid of the first argument

        Returns
        -------
        jnp.ndarray (n1, n2, 2)
        """
        return jnp.moveaxis(jnp.array(jnp.meshgrid(x1, x2, indexing='ij')), 0, 2) # type: jnp.ndarray

    @staticmethod
    @partial(jit, static_argnums=(2, ))
    def fd_2d(f: jnp.ndarray,
              dxs: Tuple[float, float],
              mode: str) -> jnp.ndarray:
        r"""Finite (central) difference method for f \colon \R^2 \to \R.

        Parameters
        ----------
        f : jnp.ndarray (n1+2, n2+2)
            Function evaluated at a (padded) 2D grid. Note that n1, n2 >= 3.
        dxs : Tuple[float, float]
            Differences along each axis.
        mode : str
            Options are, 'd/dx1', 'd/dx2', 'd^2/dx1^2', 'd^2/dx2^2', 'd^2/dx1dx2', and 'd^2/dx2dx1'

        Returns
        -------
        jnp.ndarray (n1, n2)

        Notes
        -----
        Unit test shows that it somehow does not work for float32

        References
        ----------
        Table of finite difference: https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119083405.app1
        """
        if mode == 'd/dx1':
            return (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * dxs[0])
        elif mode == 'd/dx2':
            return (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * dxs[1])
        elif mode == 'd^2/dx1^2':
            return (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / (dxs[0] ** 2)
        elif mode == 'd^2/dx2^2':
            return (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / (dxs[1] ** 2)
        elif mode == 'd^2/dx1dx2':
            return (f[2:, 2:] - f[:-2, 2:] - f[2:, :-2] + f[:-2, :-2]) / (4 * dxs[0] * dxs[1])
        elif mode == 'd^2/dx2dx1':
            return (f[2:, 2:] - f[:-2, 2:] - f[2:, :-2] + f[:-2, :-2]) / (4 * dxs[0] * dxs[1])
        else:
            raise ValueError('Wrong derivative.')

    @staticmethod
    def euler(f: Callable,
              p: jnp.ndarray,
              t: float, dt: float) -> jnp.ndarray:
        """Euler's method
        """
        return p + dt * f(p, t)

    @staticmethod
    def rk4(f: Callable,
            p: jnp.ndarray,
            t: float, dt: float) -> jnp.ndarray:
        r"""4-th order Runge--Kutta

        .. math::

            dp/dt = f(p, t) dt

        Parameters
        ----------
        f : Callable
            Callable function taking :code:`p` and :code:`t` as arguments.
        p : jnp.ndarray (n1, n2)
            Initial condition in matrix/vector.
        t : float
            Initial time.
        dt : float
            Time interval.

        Returns
        -------
        p : jnp.ndarray (n1, n2)
            RK4 integrated ODE solution at :math:`t + \Delta t`.
        """
        k1 = f(p, t)
        k2 = f(p + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(p + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(p + dt * k3, t + dt)
        return p + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    @staticmethod
    def normal_pdf(x: jnp.ndarray, mean=jnp.zeros((2,)), varr=0.125) -> jnp.ndarray:
        """Standard multivariate Normal probability density function.

        Parameters
        ----------
        x : jnp.ndarray (n1, n2, 2)
        """
        mean = jnp.broadcast_to(mean, (*x.shape[:2], 2))
        return jnp.exp(-0.5 * jnp.sum((x - mean) ** 2, axis=-1) / varr) \
               / (jnp.sqrt((2 * math.pi) ** x.shape[-1] * varr))

    @staticmethod
    @jit
    def extrapolate_2d(p: jnp.ndarray) -> jnp.ndarray:
        """Extrapolate a 2D matrix linearly with width 1
        """
        x_start = 2 * p[0] - p[1]
        x_end = 2 * p[-1] - p[-2]

        lu = 2 * p[0, 0] - p[1, 1]
        ru = 2 * p[0, -1] - p[1, -2]
        ll = 2 * p[-1, 0] - p[-2, 1]
        rl = 2 * p[-1, -1] - p[-2, -2]

        y_start = jnp.pad(2 * p[:, 0] - p[:, 1],
                          pad_width=(1, 1),
                          mode='constant',
                          constant_values=(lu, ll))
        y_end = jnp.pad(2 * p[:, -1] - p[:, -2],
                        pad_width=(1, 1),
                        mode='constant',
                        constant_values=(ru, rl))

        # if assume_positive:
        #     jax.ops.index_update(x_start, x_start < 0, 0.)
        #     jax.ops.index_update(x_end, x_end < 0, 0.)
        #     jax.ops.index_update(y_start, y_start < 0, 0.)
        #     jax.ops.index_update(y_end, y_end < 0, 0.)

        p = jnp.vstack((x_start, p))
        p = jnp.vstack((p, x_end))
        p = jnp.column_stack((y_start, p))
        p = jnp.column_stack((p, y_end))

        return p