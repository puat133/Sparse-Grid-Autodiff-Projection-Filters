import jax.numpy as jnp
from functools import partial


@partial(jnp.vectorize, signature='(n),(n),(n,n),(n,n)->()')
def hellinger_distance_between_two_gaussians(mean1: jnp.ndarray, mean2: jnp.ndarray,
                                             var1: jnp.ndarray, var2: jnp.ndarray):
    """
    Compute hellinger distance between two Gaussian distribution with means equal to `mean1` and
    `mean2` and variances equal to `var1` and `var2`.

    Parameters
    ----------
    mean1
    mean2
    var1
    var2

    Returns
    -------

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> mean1 = jnp.array([1.,1.])
    >>> mean2 = mean1
    >>> var1 = jnp.eye(2)
    >>> var2 = var1
    >>> hellinger_distance_between_two_gaussians(mean1,mean2,var1,var2)
    0.
    """
    delta_mean = mean1 - mean2
    term1 = jnp.power(jnp.linalg.det(var1) * jnp.linalg.det(var2), 0.25) / jnp.power(jnp.linalg.det(0.5 * (var1 + var2))
                                                                                     , 0.5)
    term2 = delta_mean @ (jnp.linalg.solve(0.5 * (var1 + var2), delta_mean))
    hell = 1 - term1 * jnp.exp(-term2 / 8)
    return hell
