import jax.numpy as jnp
import numpy as onp


def sieve(n: int) -> jnp.ndarray:
    """
    Computes a sequence of primes.

    Parameters
    ----------
    n   : int

    Returns
    -------
    res : onp.ndarray
        numpy array containing first `n` primes
    References
    ----------
    [1]https://code.activestate.com/recipes/117119-sieve-of-eratosthenes/

    """
    z = {2}  # set of primes
    N = {x for x in range(3, n + 1, 2)}  # natural numbers in range.
    size = len(N)
    while size > 0:  # stop when N is exhausted.
        sn = min(N)  # sn: smallest number in N
        for i in z:
            if sn % i == 0:  # it is divisible with a prime and cannot be a prime.
                m = {n for n in range(sn, n + 1, sn)}  # m is set of multiples of sn
                N = N - m  # using deduction of sets to update N
            elif sn < i * i:  # match-point has been passed.
                z.add(sn)
                m = {n for n in range(sn, n + 1, sn)}  # m is set of multiples of sn
                N = N - m  # using deduction of sets to update N
                break
        size = len(N)
    prime_numbers = onp.sort(onp.fromiter(z, int))
    return jnp.array(prime_numbers)
