import jax.numpy as np
from jax import jvp, lax
from typing import Callable
from sde.buthcers import SRK2W1, SRK2Wm, SRA3Wm, ButcherTable
from sde.ito import ito_1_wm, ito_2_wm


def ito_euler_increment(f: Callable[[np.ndarray, float], np.ndarray],
                        g: Callable[[np.ndarray, float], np.ndarray],
                        x: np.ndarray,
                        t: float,
                        dt: float,
                        stochastic_increments: np.ndarray) -> np.ndarray:
    """
    Compute Euler Maruyama increment with possibly vector brownian increment.

    Parameters
    ----------
    f       : Callable[[np.ndarray, float], np.ndarray]
        the deterministic part of the sde (d x 1  array) and is a function
        of time and (scalar or  d x 1  array).
    g       : Callable[[np.ndarray, float], np.ndarray]
        the stochastic part of the sde (d x m  array) and is a function
        of time and (d x 1  array).
    x       : np.ndarray
        the array of current iteration.
    t       : float
        current time
    dt      : float
        delta t
    stochastic_increments      : np.ndarray
        current instance of Brownian increment (dW), each has variance dt.

    Returns
    -------
    output  : np.ndarray
        next iteration
    """

    return f(x, t) * dt + g(x, t) @ stochastic_increments


def ito_euler_scalar_noise_increment(f: Callable[[np.ndarray, float], np.ndarray],
                                     g: Callable[[np.ndarray, float], np.ndarray],
                                     x: np.ndarray, t: float, dt: float,
                                     stochastic_increments: float) -> np.ndarray:
    """
    Compute Euler Maruyama increment, with scalar brownian increment.

    Parameters
    ----------
    f       : Callable[[np.ndarray, float], np.ndarray]
        the deterministic part of the sde (d x 1  array) and is a function
        of time and (scalar or  d x 1  array).
    g       : Callable[[np.ndarray, float], np.ndarray]
        the stochastic part of the sde (d x m  array) and is a function
        of time and (d x 1  array).
    x       : np.ndarray
        the array of current iteration.
    t       : float
        current time
    dt      : float
        delta t
    stochastic_increments      : np.ndarray
        current instance of Brownian increment (dW), each has variance dt.

    Returns
    -------
    output  : np.ndarray
        dx
    """

    return f(x, t) * dt + g(x, t) * stochastic_increments


def ito_milstein_scalar_noise_increment(f: Callable[[np.ndarray, float], np.ndarray],
                                        g: Callable[[np.ndarray, float], np.ndarray],
                                        x: np.ndarray, t: float, dt: float,
                                        stochastic_increments: np.ndarray) -> np.ndarray:
    """
    Compute Milstein increment with scalar brownian increment.

    Parameters
    ----------
    f       : Callable[[np.ndarray, float], np.ndarray]
        the deterministic part of the sde (d x 1  array) and is a function
        of time and (scalar or  d x 1  array).
    g       : Callable[[np.ndarray, float], np.ndarray]
        the stochastic part of the sde (d x 1  array) and is a function
        of time and (d x 1  array).
    x       : np.ndarray
        the array of current iteration.
    t       : float
        current time
    dt      : float
        delta t
    stochastic_increments      : np.ndarray
        current instance of Brownian increment (dW), each has variance dt.

    Returns
    -------
    output  : np.ndarray
        dx
    """

    g_ = g(x, t)
    f_ = f(x, t)
    _, dgdx_g_ = jvp(g, (x, t), (g_, 0.))
    return f_ * dt + g_ * stochastic_increments + (dgdx_g_ * (stochastic_increments * stochastic_increments - dt))


def strongSRKW1(f: Callable[[np.ndarray, float], np.ndarray],
                g: Callable[[np.ndarray, float], np.ndarray],
                x: np.ndarray, t: float, dt: float,
                stochastic_increments: np.ndarray,
                scheme: ButcherTable = SRK2W1) -> np.ndarray:
    """
    Compute Stochastic Runge-Kutta increment with scalar scalar brownian increment.
    The valid scheme for this integration is SRK2W1 and SRK1w1

    Parameters
    ----------
    f       : Callable[[np.ndarray, float], np.ndarray]
        the deterministic part of the sde (d x 1  array) and is a function
        of time and (scalar or  d x 1  array).
    g       : Callable[[np.ndarray, float], np.ndarray]
        the stochastic part of the sde (d x 1  array) and is a function
        of time and (d x 1  array).
    x       : np.ndarray
        the array of current iteration.
    t       : float
        current time
    dt      : float
        delta t
    stochastic_increments : np.ndarray
        set of ito integrals I1, I2, and I3
    scheme  : ButcherTable (optional)
        SRK scheme for scalar Brownian increment. Default is SRK2W1


    Returns
    -------
    output  : np.ndarray
        dx

    References
    ----------
    [1] RUNGE–KUTTA METHODS FOR THE STRONG APPROXIMATION OF SOLUTIONS OF STOCHASTIC DIFFERENTIAL EQUATIONS
    """
    i1, i2, i3 = stochastic_increments
    sdt = np.sqrt(dt)
    # there are four stages and we will write them manually
    # here we use i = 0, .. , s-1, for the index in equation (6.10) p. 934 of [1]
    H0_0 = x
    H1_0 = x

    f_0 = f(H0_0, t)
    g_0 = g(H1_0, t)
    H0_1 = x + scheme.A[0][1, 0] * f_0 * dt \
           + scheme.B[0][1, 0] * g_0 * i2[1, 0] / dt
    H1_1 = x + scheme.A[1][1, 0] * f_0 * dt \
           + scheme.B[1][1, 0] * g_0 * sdt

    f_1 = f(H0_1, t + scheme.c[0, 1] * dt)
    g_1 = g(H1_1, t + scheme.c[1, 1] * dt)
    H0_2 = x + (scheme.A[0][2, 0] * f_0
                + scheme.A[0][2, 1] * f_1) * dt \
           + (scheme.B[0][2, 0] * g_0 * i2[1, 0]
              + scheme.B[0][2, 1] * g_1 * i2[1, 0]) / dt
    H1_2 = x + (scheme.A[1][2, 0] * f_0
                + scheme.A[1][2, 1] * f_1) * dt \
           + (scheme.B[1][2, 0] * g_0
              + scheme.B[1][2, 1] * g_1) * sdt

    f_2 = f(H0_2, t + scheme.c[0, 2] * dt)
    g_2 = g(H1_2, t + scheme.c[1, 2] * dt)
    H0_3 = x + (scheme.A[0][3, 0] * f_0
                + scheme.A[0][3, 1] * f_1
                + scheme.A[0][3, 2] * f_2) * dt \
           + (scheme.B[0][3, 0] * g_0 * i2[1, 0]
              + scheme.B[0][3, 1] * g_1 * i2[1, 0]
              + scheme.B[0][3, 2] * g_2 * i2[1, 0]) / dt
    H1_3 = x + (scheme.A[1][3, 0] * f_0
                + scheme.A[1][3, 1] * f_1
                + scheme.A[1][3, 2] * f_2) * dt \
           + (scheme.B[1][3, 0] * g_0
              + scheme.B[1][3, 1] * g_1
              + scheme.B[1][3, 2] * g_2) * sdt

    f_3 = f(H0_3, t + scheme.c[0, 3] * dt)
    g_3 = g(H1_3, t + scheme.c[1, 3] * dt)

    fs = np.stack((f_0, f_1, f_2, f_3))
    gs = np.stack((g_0, g_1, g_2, g_3))

    itos = np.array([i1, i2[1, 1] / sdt, i2[1, 0] / dt, i3 / dt])

    dx = (scheme.a @ fs) * dt + ((scheme.b.T @ itos) @ gs)
    return dx


def strongSRKWm(f: Callable[[np.ndarray, float], np.ndarray],
                g: Callable[[np.ndarray, float], np.ndarray],
                x: np.ndarray, t: float, dt: float,
                stochastic_increments: np.ndarray,
                scheme: ButcherTable = SRK2Wm) -> np.ndarray:
    """
    Compute Stochastic Runge-Kutta increment with scalar m-dimensional brownian increment.
    The valid scheme for this integration is SRA1,SRA2 and SRA3

    Parameters
    ----------
    f       : Callable[[np.ndarray, float], np.ndarray]
        the deterministic part of the sde (d x 1  array) and is a function
        of time and (scalar or  d x 1  array).
    g       : Callable[[np.ndarray, float], np.ndarray]
        the stochastic part of the sde (d x 1  array) and is a function
        of time and (d x 1  array). The function g should be vectorized.
    x       : np.ndarray
        the array of current iteration.
    t       : float
        current time
    dt      : float
        delta t
    stochastic_increments : np.ndarray
        set of ito integrals I1, I2, and I3
    scheme  : ButcherTable (optional)
        SRK scheme for scalar Brownian increment. Default is SRK2W1


    Returns
    -------
    output  : np.ndarray
        dx

    References
    ----------
    [1] RUNGE–KUTTA METHODS FOR THE STRONG APPROXIMATION OF SOLUTIONS OF STOCHASTIC DIFFERENTIAL EQUATIONS
    """
    i1, i2 = stochastic_increments
    sdt = np.sqrt(dt)
    # there are four stages and we will write them manually
    # here we use i = 0, .. , s-1, for the index in equation (6.10) p. 934 of [1]
    H0_0 = x
    H1_0 = x

    f_0 = f(H0_0, t)
    g_0 = g(H1_0, t)
    H0_1 = x + scheme.A[0][1, 0] * f_0 * dt  # <-- this is one dimensional array
    Hm_1 = x + scheme.A[1][1, 0] * f_0 * dt \
           + scheme.B[1][1, 0] * g_0 @ i2 / sdt  # <-- this is two dimensional array

    f_1 = f(H0_1, t + scheme.c[0, 1] * dt)
    g_1 = g(Hm_1.T, t + scheme.c[1, 1] * dt)
    # H0_2 = x
    Hm_2 = x + scheme.A[1][2, 0] * f_0 * dt \
           + scheme.B[1][2, 0] * g_0 @ i2 / sdt

    g_2 = g(Hm_2.T, t + scheme.c[1, 2] * dt)

    dx = (scheme.a[0] * f_0 + scheme.a[1] * f_1) * dt \
         + g_0 @ (scheme.b[0, 0] * i1) \
         + np.trace(g_1 * scheme.b[1, 1] + g_2 * scheme.b[1, 2], axis1=0, axis2=2) * sdt

    return dx


def strongSRA_Wm(f: Callable[[np.ndarray, float], np.ndarray],
                 g: Callable[[np.ndarray, float], np.ndarray],
                 x: np.ndarray, t: float, dt: float,
                 stochastic_increments: np.ndarray,
                 scheme: ButcherTable = SRA3Wm) -> np.ndarray:
    """
    Compute Stochastic Runge-Kutta increment with m-dimensional brownian increment.
    Corresponds to Section 6.5. Order 1.5 strong SRK methods for SDEs with additive noise of [1].

    Parameters
    ----------
    f       : Callable[[np.ndarray, float], np.ndarray]
        the deterministic part of the sde (d x 1  array) and is a function
        of time and (scalar or  d x 1  array).
    g       : Callable[[np.ndarray, float], np.ndarray]
        the stochastic part of the sde (d x 1  array) and is a function
        of time and (d x 1  array). The function g should be vectorized.
    x       : np.ndarray
        the array of current iteration.
    t       : float
        current time
    dt      : float
        delta t
    stochastic_increments : np.ndarray
        set of ito integrals I1, and I2
    scheme  : ButcherTable (optional)
        SRK scheme for scalar Brownian increment. Default is SRA3Wm


    Returns
    -------
    output  : np.ndarray
        dx

    References
    ----------
    [1] RUNGE–KUTTA METHODS FOR THE STRONG APPROXIMATION OF SOLUTIONS OF STOCHASTIC DIFFERENTIAL EQUATIONS
    """
    i1, i2 = stochastic_increments
    sdt = np.sqrt(dt)
    # there are four stages and we will write them manually
    # here we use i = 0, .. , s-1, for the index in equation (6.10) p. 934 of [1]
    H0_0 = x
    f_0 = f(H0_0, t)
    g_0 = g(x, t + scheme.c[1, 0] * dt)

    H0_1 = x + scheme.A[0][1, 0] * f_0 * dt  # <-- this is one dimensional array
    f_1 = f(H0_1, t + scheme.c[0, 1] * dt)
    g_1 = g(x, t + scheme.c[1, 1] * dt)

    H0_2 = x + (scheme.A[0][2, 0] * f_0 + scheme.A[0][2, 1] * f_1) * dt \
           + (scheme.B[0][2, 0] * g_0 + scheme.B[0][2, 1] * g_1) @ i2[:, 0] / dt
    f_2 = f(H0_2, t + scheme.c[0, 2] * dt)
    # g_2 = g(x, t + scheme.c[1, 2] * dt)   # --> Not needed

    dx = (scheme.a[0] * f_0 + scheme.a[1] * f_1 + scheme.a[2] * f_2) * dt \
         + g_0 @ (scheme.b[0, 0] * i1) \
         + (g_0 * scheme.b[1, 0] + g_1 * scheme.b[1, 1]) @ i2[:, 0] / dt

    return dx
