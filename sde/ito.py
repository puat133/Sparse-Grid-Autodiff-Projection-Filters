from functools import partial

import jax.numpy as np
import jax.random as jrandom
from jax.ops import index, index_update

from sde.wiener import multidimensional_wiener_process, scalar_wiener_process


@partial(np.vectorize, signature='(n),(n)->(n,n)')
def outer(avec: np.ndarray, bvec: np.ndarray) -> np.ndarray:
    """
    vectorized version of numpy outer product

    Parameters
    ----------
    avec
    bvec

    Returns
    -------

    """
    return np.outer(avec, bvec)


# ------------------------------------------------- ----------------------------
# Functions for generating multiple Ito integrals
# ------------------------------------------------- ----------------------------
def ito_1_w1(dw):
    """
    Single Ito integral from 1 to dW for scalar Wiener process
    
    Parameters
    ----------
    dw      :np.ndarray
        brownian increment

    Returns
    -------
    output  :np.ndarray
        dw
    """

    return dw


def ito_2_w1(dw: np.ndarray, dt: float, prngkey: np.ndarray):
    """
    Generation of double integral values Ito for the scalar Wiener process

    Parameters
    ----------
    dw      : np.ndarray
        brownian increment
    dt      : float
        delta t
    prngkey :np.ndarray
        jax random key

    Returns
    -------

    """

    dzeta = scalar_wiener_process(dw.shape[0], dt, prngkey) + dt

    ito_1_1 = 0.5 * (dw ** 2 - dt)
    ito_1_0 = 0.5 * dt * (dw + dzeta / np.sqrt(3))
    ito_0_1 = 0.5 * dt * (dw - dzeta / np.sqrt(3))
    ito_0_0 = dt * np.ones(dw.shape[0])
    ito = np.array([[ito_0_0, ito_0_1], [ito_1_0, ito_1_1]])
    return np.swapaxes(ito, 0, 2)


def levi_integral(dw: np.ndarray, dt: float, prngkey: np.ndarray):
    """
    The Lévy integral for calculating the approximation of the Ito integral
    num --- the number of members of the series

    Parameters
    ----------
    prngkey : np.ndarray
        jax random key
    dw      : ndarray
        Brwonian increment with array (nt, dim)
    dt      : float
        delta time

    Returns
    -------
    out     : np.ndarray
        LeviIntegration array with shape (nt, dim, dim)
    """
    (nt, dim) = dw.shape
    key, subkey = jrandom.split(prngkey)
    x = jrandom.normal(subkey, (nt, nt, dim))

    _, subkey = jrandom.split(key)
    y = jrandom.normal(subkey, (nt, nt, dim))

    y_tilde = (y + np.sqrt(2 / dt) * dw[np.newaxis, :, :])
    a1 = outer(x, y_tilde)
    a2 = outer(y_tilde, x)
    t = np.arange(nt)
    factor = (1.0 / (t + 1))
    scaled_difference = (a1 - a2) * factor[:, np.newaxis, np.newaxis, np.newaxis]
    a = (dt / np.pi) * np.sum(scaled_difference, axis=0)

    return a


def ito_1_wm(dw, dt):
    """
    Generation of single Ito integral values for multidimensional
    Wiener process

    Parameters
    ----------
    dw      : ndarray
        Brwonian increment with shape (nt, dim)
    dt      : float
        delta time

    Returns
    -------
    output  : np.ndarray
        ito integration with shape (nt, dim+ 1)
    """
    ito = np.hstack([np.ones((dw.shape[0], 1)) * dt, dw])
    return ito


def ito_2_wm(dw: np.ndarray, dt: float, prngkey: np.ndarray):
    """
    Calculation of the Ito integral(by exact formula or by approximation)
    Returns
    Parameters
    ----------
    dw          : np.ndarray
        multi dimensional brownian increment, with shape (nt, dim)
    dt          : float
        delta t
    prngkey    : np.ndarray
        jax random key

    Returns
    -------
    out : np.ndarray
        a list of matrices I(i, j) of shape (nt, dim, dim)
    """

    (nt, dim) = dw.shape
    e = np.identity(dim)

    key, subkey = jrandom.split(prngkey)
    dzeta = multidimensional_wiener_process((nt, dim), dt, subkey)  # np.random.normal(loc=0, scale=dt, size=(n, m))

    _, subkey = jrandom.split(subkey)
    ito_0_0 = dt * np.ones(nt)
    ito_0_1 = 0.5 * dt * (dw - dzeta / np.sqrt(3))
    ito_1_0 = 0.5 * dt * (dw + dzeta / np.sqrt(3))
    ito_1_1 = outer(dw, dw) - dt * e + levi_integral(dw, dt, subkey)

    ito_0 = np.block([[ito_0_0[:, np.newaxis, np.newaxis], ito_0_1[:, np.newaxis, :]]])
    ito_1 = np.block([[ito_1_0[:, :, np.newaxis], ito_1_1]])
    return np.block([[ito_0], [ito_1]])


def ito_3_w1(dw: np.ndarray, dt: float) -> np.ndarray:
    """
    Generation of the triple Ito integral
    for the scalar Wiener process

    Parameters
    ----------
    dw      : ndarray
        Brwonian increment
    dt      : float
        delta time

    Returns
    -------
    output  : np.ndarray
        ito integration
    """

    ito_integration = 0.5 * (dw ** 2 - dt)
    ito_integration = (1.0 / 6.0) * dt * (ito_integration ** 3 - 3.0 * dt * dw)
    return ito_integration
