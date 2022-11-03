import jax.numpy as np
from jax import lax, jit, partial
from typing import Tuple


@partial(np.vectorize, signature='(n),(n,n)->()')
def log_gaussian_density(x: np.ndarray, cov_inv: np.ndarray) -> float:
    """
    Evaluate log of gaussian density with zero mean and an inverse covariance matrix `cov_inv`.
    Parameters
    ----------
    x       : (N,) np.ndarray
        vector
    cov_inv : (N,N) np.ndarray
        inverse covariance
    Returns
    -------
    out: float
        log gaussian density
    """
    return -0.5 * ((x @ cov_inv @ x) + cov_inv.shape[0] * np.log(2 * np.pi) - np.linalg.slogdet(cov_inv)[1])


@partial(np.vectorize, signature='(n),(n)->(n,n)')
def outer(x: np.ndarray, y: np.ndarray):
    return np.outer(x, y)


@partial(np.vectorize, signature='(n)->()')
def log_sum_exp(log_weights: np.ndarray) -> float:
    """
    Evaluate log of sum of exponential of a vector of log_weights.

    Parameters
    ----------
    log_weights: (N,) np.ndarray
        an array.

    Returns
    -------
    res : float
        result
    """
    max_log_weights = np.max(log_weights)
    return max_log_weights + np.log(np.sum(np.exp(log_weights - max_log_weights)))


@partial(np.vectorize, signature='(n)->()')
def log_mean_exp(log_weights: np.ndarray) -> float:
    """
    Evaluate log of mean of exponential of a vector of log_weights.

    Parameters
    ----------
    log_weights: (N,) np.ndarray
        an array.

    Returns
    -------
    res : float
        result
    """
    max_log_weights = np.max(log_weights)
    return max_log_weights + np.log(np.mean(np.exp(log_weights - max_log_weights)))


def normalize_log_weights(log_weights: np.ndarray):
    return log_weights - log_sum_exp(log_weights.ravel())


@jit
def inverse_cdf(su, w):
    """Inverse CDF algorithm for a finite distribution.

        Parameters
        ----------
        su: (M,) ndarray
            M sorted uniform variates (i.e. M ordered points in [0,1]).
        w: (N,) ndarray
            a vector of N normalized weights (>=0 and sum to one)

        Returns
        -------
        flat_indices: (M,) ndarray
            a vector of M indices in range 0, ..., N-1
    """

    #     j = 0
    #     s = W[0]
    #     M = su.shape[0]
    #     A = np.empty(M, dtype=np.int64)
    #     for n in range(M):
    #         while su[n] > s:
    #             j += 1
    #             s += W[j]
    #         A[n] = j

    def body_scan(carry, inputs):
        def body_while(val):
            j_, s_ = val
            j_ += 1
            s_ += w[j_]
            return j_, s_

        n, j, s = carry
        j, s = lax.while_loop(lambda val: su[n] > val[1], body_while, (j, s))
        n += 1
        return (n, j, s), j

    _, flat_indices = lax.scan(body_scan, (0, 0, w[0]), np.ones(su.shape[0]))
    return flat_indices


@jit
def resample(indices: Tuple[np.ndarray, np.ndarray], particles: np.ndarray) -> np.ndarray:
    """ A util function to apply indices to state

    Parameters
    ----------
    indices: Tuple of two (N_devices x N_samples_per_devices) array
        The indexing of state
    particles: (N_devices x N_samples_per_devices x N_state) array
        Particles


    Returns
    -------
    out: particles
        Rasampled particles

    """
    dev_indices, sample_indices = indices
    n_devices = particles.shape[0]
    n_samples_perdevices = particles.shape[1]

    #   make sure that the indices are in the limit
    sample_indices = np.clip(sample_indices, 0, n_samples_perdevices - 1)
    dev_indices = np.clip(dev_indices, 0, n_devices - 1)

    return particles[dev_indices, sample_indices, :].reshape((n_devices, n_samples_perdevices, -1))


@jit
def systematic(particles: np.ndarray, log_weights: np.ndarray, uniform: float) -> np.ndarray:
    """ Applies systematic resampling to the state

    Parameters
    ----------
    particles: (N_devices x N_samples_per_devices x N_state) array
        Particles
    log_weights: (N_devices x N_samples_per_devices) array
        The log weights
    uniform: float
        a sample from uniform distribution

    Returns
    -------

        Resampled particles
    """
    n_devices = log_weights.shape[0]
    n_samples_perdevice = log_weights.shape[1]
    n_particles = n_devices * n_samples_perdevice
    log_weights = normalize_log_weights(log_weights)
    linspace = np.arange(n_particles, dtype=log_weights.dtype)
    flat_indices = inverse_cdf((uniform + linspace) / n_particles, np.exp(log_weights.ravel()))
    indices = (flat_indices // n_samples_perdevice, flat_indices % n_samples_perdevice)
    return resample(indices, particles)

# def stratified(backend: Backend, state: ImportanceWeightedSample, uniform):
#     """ Applies stratified resampling to the state
#
#     Parameters
#     ----------
#     backend: jaxter.backend.base.Backend
#         calculation backend
#     state: ImportanceWeightedSample
#         State to be resampled, particles have shape (K, N)
#     uniform: (N) array
#         Uniforms used to resample the state
#
#     Returns
#     -------
#     out: ImportanceWeightedSample
#         Resampled state
#     """
#
#     return systematic(backend, state, uniform)
