import jax.numpy as jnp
from jax import lax, partial
from typing import Tuple
import platform
import numpy as onp
from LowDiscrepancy import sieve


def i4_bit_hi1(n: int):
    """
    i4_bit_hi1 returns the position of the high 1 bit base 2 in an I4.

    Parameters
    ----------
    n   : int
        a number

    Returns
    -------

    """
    if n == 0:
        res = 0
    else:
        res = len(onp.binary_repr(0))

    return res


def i4_bit_lo0(n):
    """
    i4_bit_lo0 returns the position of the low 0 bit base 2 in an I4.

    Parameters
    ----------
    n   : int
        a number

    Returns
    -------
    res : int
        integer BIT, the position of the low 1 bit.
    """
    binary_rep = onp.binary_repr(n)
    return len(binary_rep) - binary_rep.rfind('0')


def tau_sobol(dim_num):
    """

    Parameters
    ----------
    dim_num

    Returns
    -------

    """
    dim_max = 13

    tau_table = [0, 0, 1, 3, 5, 8, 11, 15, 19, 23, 27, 31, 35]

    if 1 <= dim_num <= dim_max:
        tau = tau_table[dim_num]
    else:
        tau = - 1

    return tau


def assign_v_matrix(atmost: int, dim_max: int, log_max: int):
    """

    Parameters
    ----------
    atmost
    dim_max
    log_max

    Returns
    -------

    """
    v = onp.zeros((dim_max, log_max), dtype=onp.int32)
    v[0:40, 0] = onp.array([
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    v[2:40, 1] = onp.array([
        1, 3, 1, 3, 1, 3, 3, 1,
        3, 1, 3, 1, 3, 1, 1, 3, 1, 3,
        1, 3, 1, 3, 3, 1, 3, 1, 3, 1,
        3, 1, 1, 3, 1, 3, 1, 3, 1, 3])

    v[3:40, 2] = onp.array([
        7, 5, 1, 3, 3, 7, 5,
        5, 7, 7, 1, 3, 3, 7, 5, 1, 1,
        5, 3, 3, 1, 7, 5, 1, 3, 3, 7,
        5, 1, 1, 5, 7, 7, 5, 1, 3, 3])

    v[5:40, 3] = onp.array([
        1, 7, 9, 13, 11,
        1, 3, 7, 9, 5, 13, 13, 11, 3, 15,
        5, 3, 15, 7, 9, 13, 9, 1, 11, 7,
        5, 15, 1, 15, 11, 5, 3, 1, 7, 9])

    v[7:40, 4] = onp.array([
        9, 3, 27,
        15, 29, 21, 23, 19, 11, 25, 7, 13, 17,
        1, 25, 29, 3, 31, 11, 5, 23, 27, 19,
        21, 5, 1, 17, 13, 7, 15, 9, 31, 9])

    v[13:40, 5] = onp.array([
        37, 33, 7, 5, 11, 39, 63,
        27, 17, 15, 23, 29, 3, 21, 13, 31, 25,
        9, 49, 33, 19, 29, 11, 19, 27, 15, 25])

    v[19:40, 6] = onp.array([
        13,
        33, 115, 41, 79, 17, 29, 119, 75, 73, 105,
        7, 59, 65, 21, 3, 113, 61, 89, 45, 107])

    v[37:40, 7] = onp.array([
        7, 23, 39])

    max_col = i4_bit_hi1(atmost)
    #   Initialize row 1 of v
    v[0, 0:max_col] = 1
    return v


class Sobol(object):
    def __init__(self):
        self._dim_max = 40
        self._current_dim_num = 2
        self._log_max = 30
        self._atmost = 2 ** self._log_max - 1
        self._max_col = i4_bit_hi1(self._atmost)
        self._seed_save = -1
        self._prime_numbers = sieve(onp.iinfo(onp.uint16).max)
        self._v = assign_v_matrix(self._atmost, self._dim_max, self._log_max)

        # initial value of recipd and lastq
        self._recipd = 1.0
        self._lastq = onp.zeros(self._current_dim_num)

        self._poly = jnp.array([
            1, 3, 7, 11, 13, 19, 25, 37, 59, 47,
            61, 55, 41, 67, 97, 91, 109, 103, 115, 131,
            193, 137, 145, 143, 241, 157, 185, 167, 229, 171,
            213, 191, 253, 203, 211, 239, 247, 285, 369, 299], dtype=jnp.int32)

    def i4_sobol(self, dim_num: int, seed: int):
        if self._current_dim_num != dim_num:
            if dim_num < 1 or self._dim_max < dim_num:
                raise ValueError(" The spatial dimension DIM_NUM should satisfy: 1 <= DIM_NUM <= {}, "
                                 " But this input value is DIM_NUM = {}".format(dim_num, self._dim_max))
            else:
                self._current_dim_num = dim_num
                for i in range(2, self._current_dim_num + 1):
                    # The bits of the integer POLY(I) gives the form of polynomial I.
                    # Find the degree of polynomial I from binary encoding.

                    j = self._poly[i - 1]
                    binary_rep = onp.binary_repr(j)
                    m = len(binary_rep) - 1
                    # ORIGINAL
                    # m = 0
                    # while True:
                    #     j = onp.floor_divide(j, 2)
                    #     if j <= 0:
                    #         break
                    #     m = m + 1

                    # Expand this bit pattern to separate components of the logical array INCLUD.
                    j = self._poly[i - 1]
                    res = onp.unpackbits(onp.array([int(binary_rep[1:], 2)], dtype=onp.uint8))
                    includ = res[-m:]
                    # ORIGINAL
                    # includ = onp.zeros(m)
                    # for k in range(m, 0, -1):
                    #     j2 = onp.floor_divide(j, 2)
                    #     includ[k - 1] = (j != 2 * j2)
                    #     j = j2

                    # Calculate the remaining elements of row I as explained
                    # in Bratley and Fox, section 2.

                    for j in range(m + 1, self._max_col + 1):
                        newv = self._v[i - 1, j - m - 1]
                        ell = 1
                        for k in range(1, m + 1):
                            ell = 2 * ell
                            if includ[k - 1]:
                                newv = onp.bitwise_xor(newv, ell * self._v[i - 1, j - k - 1])
                        self._v[i - 1, j - 1] = newv

                # Multiply columns of V by appropriate power of 2.

                ell = 1
                for j in range(self._max_col - 1, 0, -1):
                    ell = 2 * ell
                    self._v[0:self._current_dim_num, j - 1] = self._v[0:self._current_dim_num, j - 1] * ell

                # RECIPD is 1/(common denominator of the elements in V).

                self._recipd = 1.0 / (2 * ell)
                self._lastq = onp.zeros(self._current_dim_num)

        if seed < 0:
            raise ValueError("Seed cannot be less than zero!")

        elif seed == 0:
            ell = 1
            lastq = onp.zeros(self._current_dim_num)

        elif seed == self._seed_save + 1:

            # Find the position of the right-hand zero in SEED.

            ell = i4_bit_lo0(seed)

        elif seed <= self._seed_save:

            seed_save = 0
            # ell = 1
            self._lastq = onp.zeros(self._current_dim_num, dtype=onp.int32)

            for seed_temp in range(int(seed_save), int(seed)):
                ell = i4_bit_lo0(seed_temp)
                for i in range(1, self._current_dim_num + 1):
                    self._lastq[i - 1] = onp.bitwise_xor(self._lastq[i - 1], self._v[i - 1, ell - 1])

            ell = i4_bit_lo0(seed)

        elif self._seed_save + 1 < seed:

            for seed_temp in range(self._seed_save + 1, seed):
                ell = i4_bit_lo0(seed_temp)
                for i in range(1, dim_num + 1):
                    self._lastq[i - 1] = onp.bitwise_xor(self._lastq[i - 1], self._v[i - 1, ell - 1])

            ell = i4_bit_lo0(seed)

        # Check that the user is not calling too many times!

        if self._max_col < ell:
            print('I4_SOBOL - Fatal error!')
            print('	Too many calls!')
            print('	MAXCOL = {:d}\n'.format(self._max_col))
            print('	L =			{:d}\n'.format(ell))
            return

        # Calculate the new components of QUASI.

        quasi = onp.zeros(dim_num)
        for i in range(1, dim_num + 1):
            quasi[i - 1] = self._lastq[i - 1] * self._recipd
            lastq[i - 1] = onp.bitwise_xor(self._lastq[i - 1], int(self._v[i - 1, ell - 1]))

        self._seed_save = seed
        seed = seed + 1

        return [quasi, seed]

    def i4_sobol_generate(self, m, n, skip):
        r = onp.zeros((m, n))
        for j in range(1, n + 1):
            seed = skip + j - 2
            [r[0:m, j - 1], seed] = self.i4_sobol(m, seed)
        return r
