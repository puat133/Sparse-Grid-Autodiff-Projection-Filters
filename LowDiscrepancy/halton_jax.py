import jax.numpy as jnp
from jax import lax, partial
from typing import Tuple
import platform
import numpy as onp
from LowDiscrepancy import sieve

PRIME_NUMBERS = sieve(onp.iinfo(onp.uint16).max)


def _halton_condition(val: Tuple):
    r, t, prime_inv, primes = val
    return 0 < jnp.sum(t)


def _halton_body(val: Tuple):
    r, t, prime_inv, primes = val
    d = jnp.mod(t, primes)
    r = r + d * prime_inv
    prime_inv = prime_inv / primes
    t = jnp.floor_divide(t, primes)
    return r, t, prime_inv, primes


@partial(jnp.vectorize, signature='(),()->(m)')
def halton(i: int, m: int):
    """
    HALTON computes an element of a Halton sequence.
    Parameters
    ----------
    i   : int
        the index of the element of the sequence.
    m   : int
        integer M, the spatial dimension.

    Returns
    -------
    res : np.ndarray
        real R(M), the element of the sequence with index i.
    """
    t = i * jnp.ones(m, dtype=jnp.int32)
    primes = PRIME_NUMBERS[:m]
    prime_inv = 1.0 / primes
    r = jnp.zeros(m)
    r, t, prime_inv, primes = lax.while_loop(_halton_condition, _halton_body, (r, t, prime_inv, primes))
    return r


def halton_test():
    print('')
    print('HALTON_TEST')
    print('  Python version: %s' % (platform.python_version()))
    print('  HALTON returns the I-th element of an M-dimensional')
    print('  Halton sequence.')
    print('')
    print('    I          HALTON(I)')

    for m in range(4, 10):
        print('')
        print('  Use M = %d' % m)
        print('')

        i_ = jnp.arange(0, 11)
        r_ = halton(i_, m)
        for i in range(0, 11):
            # r = halton(i, m)
            r = r_[i]
            print('  %3d' % i, end='')
            for j in range(0, m):
                print('  %14.8f' % (r[j]), end='')
            print('')
    #
    #  Terminate.
    #
    print('')
    print('HALTON_TEST')
    print('  Normal end of execution.')
    return


if __name__ == '__main__':
    halton_test()
