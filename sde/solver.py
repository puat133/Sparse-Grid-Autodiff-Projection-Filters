"""
Numerical integration algorithms for Ito and Stratonovich stochastic
ordinary differential equations. This is the JaX version of a package
written by Matthew Aburn https://github.com/mattja/sdeint

Usage:
    itoint(f, g, x0, tspan)  for Ito equation dy = f dt + g dW
    stratint(f, g, x0, tspan)  for Stratonovich equation dy = f dt + g \circ dW

    x0 is the initial vector
    tspan is an array of time values (currently these must be equally spaced)
    function f is the deterministic part of the system (d x 1  vector)
    and is a function of time and (d x 1  vector).
    function g is the stochastic part of the system (d x m matrix)
    and is a function of time and (d x 1  vector)

sdeint will choose an algorithm for you. Or you can choose one explicitly:
"""

import jax.numpy as np
import jax.lax as lax
import jax.random as jrandom
from jax import jit, vmap
from sde import SDEValueError
from sde import SDESolverTypes
from sde.ito import ito_2_w1, ito_3_w1, ito_2_wm
from typing import Callable, Tuple
from functools import partial
from sde.buthcers import SRK2W1, SRK1W1, SRK2Wm, SRK1Wm, SRA3Wm
from sde.strong import ito_euler_increment, \
    ito_milstein_scalar_noise_increment, \
    ito_euler_scalar_noise_increment, \
    strongSRKW1, strongSRKWm, strongSRA_Wm


def _check_args(f: Callable[[np.ndarray, float], np.ndarray], g: Callable[[np.ndarray, float], np.ndarray],
                x0: np.ndarray, tspan: np.ndarray,
                dW: np.ndarray) -> Tuple[int, int, Callable[[np.ndarray, float], np.ndarray],
                                         Callable[[np.ndarray, float], np.ndarray], np.ndarray,
                                         np.ndarray, np.ndarray]:
    """
    Do some validation common to all algorithms. Find dimension n and number
    of Wiener processes m.

    Parameters
    ----------
    f       : Callable[[np.ndarray, float], np.ndarray]
        the deterministic part of the sde (n x 1  array) and is a function
        of time and (scalar or  n x 1  array).
    g       : Callable[[np.ndarray, float], np.ndarray]
        the stochastic part of the sde (n x m  array) and is a function
        of time and (d x 1  array).
    x0      : np.ndarray
        the initial vector.
    tspan   : np.ndarray
        an array of time values (currently these must be equally spaced).
    dW      : np.ndarray
        a set of brownian array with dimension equals to ( T x m).

    Returns
    -------
    out: Tuple
        d, m, f, g, x0, tspan, dW
    """

    if not np.isclose(min(np.diff(tspan)), max(np.diff(tspan)), rtol=1e-2):
        raise SDEValueError('Currently time steps must be equally spaced.')

    # determine dimension d of the system
    n = len(x0)
    if len(f(x0, tspan[0])) != n:
        raise SDEValueError('x0 and f have incompatible shapes.')
    message = """x0 has length {0:d}. So g must either be a single function
              returning an array of shape ({0:d},) or 
              a matrix of shape ({0:d}, m)""".format(n)

    gtest = g(x0, tspan[0])
    if gtest.shape[0] != n:
        raise SDEValueError(message)
    elif gtest.ndim not in (1, 2):
        raise SDEValueError(message)

    # determine number of independent Wiener processes m
    if gtest.ndim == 1:
        m = 1
    elif gtest.ndim == 2:
        m = gtest.shape[1]

    message = """From function g, it seems m=={0:d}. If present, the optional
              parameter dW must be an array of shape (len(tspan)-1, {0:d}) giving
              m independent Wiener increments for each time interval.""".format(m)
    if m == 1 and dW.shape[0] != tspan.shape[0]:
        raise SDEValueError(message)
    elif m > 1 and dW.shape != (tspan.shape[0], m):
        raise SDEValueError(message)

    return n, m, f, g, x0, tspan, dW


def sde_solver(f: Callable[[np.ndarray, float], np.ndarray],
               g: Callable[[np.ndarray, float], np.ndarray],
               x0: np.ndarray,
               tspan: np.ndarray,
               dw: np.ndarray,
               solver_type: SDESolverTypes,
               seed: int = 0) -> np.ndarray:
    """
    A general outer method to solve the following SDE
    dx = f(x,t)dt + G(x,t) dW(t)

    where x is the n-dimensional state vector, f is a vector-valued function,
    G is an d x m matrix-valued function giving the noise coefficients and
    dW(t) = (dW_1, dW_2, ... dW_m) is a vector of independent Wiener increments

    the loop to solve the SDE is implemented via routines in Jax.lax such as
    scan, while, foriloop, etc, so that the graph is much more efficient.

    Parameters
    ----------
    solver_type : SDESolverTypes
        type of SDE solver.
    f       : Callable[[np.ndarray, float], np.ndarray]
        the deterministic part of the sde (n x 1  array) and is a function
        of time and (scalar or  n x 1  array).
    g       : Callable[[np.ndarray, float], np.ndarray]
        the stochastic part of the sde (n x m  array) and is a function
        of time and (n x 1  array).
    x0      : np.ndarray
        the initial vector.
    tspan   : np.ndarray
        an array of time values (currently these must be equally spaced).
    dw      : np.ndarray
        a set of brownian array with dimension equals to ( T x m).
    seed    : int (optional)
        seed number to be put in jax.random.PRNGKey function, default is 0.
    Returns
    -------
    x       : np.ndarray
         sde integration results of dimension ( T x n)

    Raises
    ------
      SDEValueError

    References
    ----------
    Kloeden and Platen (1999) Numerical Solution of Differential Equations
    """
    # check arguments
    (d, m, f, g, x0, tspan, dw) = _check_args(f, g, x0, tspan, dw)

    dt = tspan[1] - tspan[0]

    if solver_type == SDESolverTypes.ItoEulerMaruyama:
        if m == 1:
            increment = jit(partial(ito_euler_scalar_noise_increment, f=f, g=g, dt=dt))
        else:
            increment = jit(partial(ito_euler_increment, f=f, g=g, dt=dt))
        carry = x0
        inputs = [tspan, dw]

    elif solver_type == SDESolverTypes.ItoMilstein:
        if m == 1:
            increment = jit(partial(ito_milstein_scalar_noise_increment, f=f, g=g, dt=dt))
            carry = x0
            inputs = [tspan, dw]
        else:
            message = "only one dimensional noise is supported for Milstein scheme"
            raise SDEValueError(message)

    elif solver_type in (SDESolverTypes.ItoSRK1W1, SDESolverTypes.ItoSRK2W1):
        if m == 1:
            if solver_type == SDESolverTypes.ItoSRK2W1:
                increment = jit(partial(strongSRKW1, f=f, g=g, dt=dt, scheme=SRK2W1))
            else:
                increment = jit(partial(strongSRKW1, f=f, g=g, dt=dt, scheme=SRK1W1))
            carry = x0
            ito_integrals = [dw, ito_2_w1(dw, dt, jrandom.PRNGKey(seed)), ito_3_w1(dw, dt)]
            inputs = [tspan, ito_integrals]
        else:
            message = "only one dimensional noise is supported for ItoSRK1W1 or ItoSRK2W1 schemes"
            raise SDEValueError(message)

    elif solver_type in (SDESolverTypes.ItoSRK1Wm, SDESolverTypes.ItoSRK2Wm, SDESolverTypes.ItoSRA3Wm):
        gv = partial(np.vectorize, signature='(n)->(n,m)', excluded=(1,))(g)    # <- the g function need to be
        # parallelized
        if solver_type == SDESolverTypes.ItoSRK1Wm:
            increment = jit(partial(strongSRKWm, f=f, g=gv, dt=dt, scheme=SRK1Wm))
        elif solver_type == SDESolverTypes.ItoSRK2Wm:
            increment = jit(partial(strongSRKWm, f=f, g=gv, dt=dt, scheme=SRK2Wm))
        else:
            increment = jit(partial(strongSRA_Wm, f=f, g=gv, dt=dt, scheme=SRA3Wm))
        carry = x0
        i2 = ito_2_wm(dw, dt, jrandom.PRNGKey(seed))
        ito_integrals = [dw, i2[:, 1:, 1:]]
        inputs = [tspan, ito_integrals]


    else:
        increment = jit(partial(ito_euler_increment, f=f, g=g, dt=dt))
        carry = x0
        inputs = [tspan, dw]

    @jit
    def integrator_loop(carry_, inputs_):
        t_, stochastic_increments_ = inputs_
        x_ = carry_
        dx_ = increment(x=x_, t=t_, stochastic_increments=stochastic_increments_)
        carry_ = x_ + dx_
        return carry_, carry_

    x_last, x = lax.scan(integrator_loop, carry, inputs)
    return x
