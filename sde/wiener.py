import jax.numpy as np
import jax.random as jrandom
from typing import Tuple, List
from sde.common import time



def scalar_wiener_process(nt: int, dt: float, prngkey: np.ndarray) -> np.ndarray:
    """
    The function generates the trajectory of the scalar Wiener process
    Parameters
    ----------
    nt       :int
        time partition
    dt      :float
        delta t
    prngkey :ndarray
        jax random seed

    Returns
    -------
    out     : ndarray
        dw
    """

    #
    dw = np.sqrt(dt) * jrandom.normal(prngkey, (nt,))

    return dw


def multidimensional_wiener_process(shape: Tuple[int, int], dt: float, prngkey: np.ndarray) -> np.ndarray:
    """
    The function generates the trajectory of the scalar Wiener process
    Parameters
    ----------
    shape   : Tuple
        shape of the output array
    dt      : float
        delta t
    prngkey :np.ndarray
        jax random seed

    Returns
    -------
    out     :np.ndarray
        dw
    """

    dw = np.sqrt(dt) * jrandom.normal(prngkey, shape)

    return dw


def cov_multidimensional_wiener_process(nt: int, dim: int, dt: float, cov: np.ndarray,
                                        prngkey: np.ndarray) -> np.ndarray:
    """
    The function generates a multidimensional(dimension dim) Wiener process
    using multivariate_normal function. The processes W1, W2, ... Wn can
    correlate if the covariance matrix cov is not diagonal
    Parameters
    ----------
    nt       :int
        time partition
    dim     :int
        dimension
    dt      :float
        delta t
    cov     :np.ndarray
        covariance/dt of dW
    prngkey :ndarray
        jax random seed

    Returns
    -------
    out     : np.ndarray
        dw
    """
    dw = jrandom.multivariate_normal(prngkey, np.zeros(dim), dt * cov, (nt, dim))
    return dw


def wiener_process(prngkey: np.ndarray, nt: int, dim: int, interval: Tuple = (0.0, 1.0)) -> \
        Tuple[float, np.ndarray, np.ndarray]:
    """
    A wrapper function for generating a wiener process

    Parameters
    ----------
    nt       :int
        time partition
    dim     :int
        dimension
    prngkey :ndarray
        jax random seed
    interval: Tuple (optional)
        t_0, t_end default (0.,1.)


    Returns
    -------
    out     :Tuple
        dt, t, dw
    """

    dt, t = time(nt, interval)
    size = len(t)
    if dim <= 1:
        dw = scalar_wiener_process(size, dt, prngkey)
    else:
        dw = multidimensional_wiener_process((size, dim), dt, prngkey)

    return dt, t, dw
