import jax.numpy as np
from sde.ito import ito_1_w1, ito_2_w1, ito_3_w1, ito_1_wm, ito_2_wm
from typing import Callable, Tuple, List
import jax.random as jrandom
import jax.lax as lax


def __strong_method_selector(name):
    """
    Filling arrays of the Butcher table for a specific method
    Parameters
    ----------
    name

    Returns
    -------

    """
    if name == 'SRK1W1':
        s = 4
        a = np.array([1. / 3., 2. / 3., 0., 0.])
        b1 = np.array([- 1., 4. / 3., 2. / 3., 0.])
        b2 = np.array([- 1., 4. / 3., -1. / 3., 0.])
        b3 = np.array([2., -4. / 3., -2. / 3., 0.])
        b4 = np.array([- 2., 5. / 3., -2. / 3., 1.])

        c0 = np.array([0., 3. / 4., 0., 0.])
        c1 = np.array([0., 1. / 4., 1., 1. / 4.])

        A0 = np.array([[0., 0., 0., 0.],
                       [3. / 4., 0., 0., 0.],
                       [0., 0., 0., 0.],
                       [0., 0., 0., 0.]])
        A1 = np.array([[0., 0., 0., 0.],
                       [1. / 4., 0., 0., 0.],
                       [1., 0., 0., 0.],
                       [0., 0., 1. / 4., 0.]])
        B0 = np.array([[0., 0., 0., 0.],
                       [3. / 2., 0., 0., 0.],
                       [0., 0., 0., 0.],
                       [0., 0., 0., 0.]])
        B1 = np.array([[0., 0., 0., 0.],
                       [0.5, 0., 0., 0.],
                       [-1., 0., 0., 0.],
                       [-5., 3., 0.5, 0.]])
    elif name == 'SRK2W1':
        s = 4
        a = np.array([1. / 6., 1. / 6., 2. / 3., 0.])
        b1 = np.array([- 1., + 4. / 3., + 2. / 3., 0.])
        b2 = np.array([+ 1., -4. / 3., + 1. / 3., 0.])
        b3 = np.array([+ 2., -4. / 3., -2. / 3., 0.])
        b4 = np.array([- 2., + 5. / 3., -2. / 3., 1.])

        c0 = np.array([0., 1., 0.5, 0.])
        c1 = np.array([0., 0.25, 1., 0.25])

        A0 = np.array([[0., 0., 0., 0.],
                       [1., 0., 0., 0.],
                       [1. / 4., 1. / 4., 0., 0.],
                       [0., 0., 0., 0.]])
        A1 = np.array([[0., 0., 0., 0.],
                       [1. / 4., 0., 0., 0.],
                       [1., 0., 0., 0.],
                       [0., 0., 1. / 4., 0.]])
        B0 = np.array([[0., 0., 0., 0.],
                       [0., 0., 0., 0.],
                       [1., 0.5, 0., 0.],
                       [0., 0., 0., 0.]])
        B1 = np.array([[0., 0., 0., 0.],
                       [-0.5, 0., 0., 0.],
                       [+1., 0., 0., 0.],
                       [+2., -1., 0.5, 0.]])
    elif name == 'KlPl':
        # Kloeden an Platen
        s = 2
        a = np.array([1., 0.])
        b1 = np.array([1., 0.])
        b2 = np.array([- 1., 1.])
        b3 = np.array([0., 0.])
        b4 = np.array([0., 0.])
        c0 = np.array([0., 0.])
        c1 = np.array([0., 0.])

        A0 = np.array([[0., 0.],
                       [0., 0.]])
        A1 = np.array([[0., 0.],
                       [1., 0.]])
        B0 = np.array([[0., 0.],
                       [0., 0.]])
        B1 = np.array([[0., 0.],
                       [1., 0.]])
    elif name == 'SRK1Wm':
        s = 3
        a = np.array([1., 0., 0.])
        b1 = np.array([1., 0., 0.])
        b2 = np.array([0., 0.5, -0.5])

        c0 = np.array([0., 0., 0.])
        c1 = np.array([0., 0., 0.])

        A0 = np.array([[0., 0., 0.],
                       [0., 0., 0.],
                       [0., 0., 0.]])
        A1 = np.array([[0., 0., 0.],
                       [0., 0., 0.],
                       [0., 0., 0.]])
        B0 = np.array([[0., 0., 0.],
                       [0., 0., 0.],
                       [0., 0., 0.]])
        B1 = np.array([[0., 0., 0.],
                       [+1., 0., 0.],
                       [-1., 0., 0.]])
    elif name == 'SRK2Wm':
        s = 3
        a = np.array([0.5, 0.5, 0.])
        b1 = np.array([1., 0., 0.])
        b2 = np.array([0., 0.5, -0.5])

        c0 = np.array([0., 1., 0.])
        c1 = np.array([0., 1., 1.])

        A0 = np.array([[0., 0., 0.],
                       [1., 0., 0.],
                       [0., 0., 0.]])
        A1 = np.array([[0., 0., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.]])
        B0 = np.array([[0., 0., 0.],
                       [0., 0., 0.],
                       [0., 0., 0.]])
        B1 = np.array([[0., 0., 0.],
                       [+1., 0., 0.],
                       [-1., 0., 0.]])
    # end if
    if name in {'SRK1W1', 'SRK2W1', 'KlPl'}:
        return s, a, b1, b2, b3, b4, c0, c1, A0, A1, B0, B1
    elif name in {'SRK1Wm', 'SRK2Wm'}:
        return s, a, b1, b2, c0, c1, A0, A1, B0, B1


def strongSRKW1(f: Callable[[float, np.ndarray], np.ndarray], g: Callable[[float, np.ndarray], np.ndarray],
                tspan: np.ndarray, dt: float, Y0: np.ndarray, dw: np.ndarray,
                prngkey: np.ndarray, name='SRK2W1'):
    """
    Stochastic Runge-Kutta method of strong order p = 1.5
    for the scalar Wiener process
    """

    if not isinstance(name, str):
        name = 'SRK2W1'
        print(" Argument `name` is not string! Using `name = 'SRK2W1'`")

    (s, a, b1, b2, b3, b4,
     c0, c1, A0, A1, B0, B1) = __strong_method_selector(name)

    sqdt = np.sqrt(dt)
    # x = []
    # x_tmp = Y0
    #
    # X0 = np.zeros(s)
    # X1 = np.zeros(s)

    i1 = ito_1_w1(dw)
    key, subkey = jrandom.split(prngkey)
    i2 = ito_2_w1(dw, dt, subkey)
    i3 = ito_3_w1(dw, dt)

    def SRK_H_stages(index, value):
        H0, H1 = value
        H0 = A0

    inputs = [tspan, i1, i2, i3]

    H0 = np.tile(Y0, (s, 1))
    H1 = np.tile(Y0, (s, 1))

    carry = (Y0, H0, H1)
    x, carry = lax.scan(SRK_W1_loop, carry, inputs)

    # for t, i1, i2, i3 in zip(tspan, i1, i2, i3):
    #     for i in range(0, s, 1):
    #         X0[i] = x_tmp + np.dot(A0[i, :], f(t + c0[i] * dt, X0)) * dt
    #         X0[i] += np.dot(B0[i, :], g(t + c1[i] * dt, X1)) * i2[1, 0] / dt
    #         #
    #         X1[i] = x_tmp + np.dot(A1[i, :], f(t + c0[i] * dt, X0)) * dt
    #         X1[i] += np.dot(B1[i, :], g(t + c1[i] * dt, X1)) * sqdt
    #     # end for
    #     x_tmp = x_tmp + np.dot(a, f(X0)) * dt + np.dot(
    #         (b1 * i1 + b2 * i2[1, 1] / sqdt + (b3 * i2[1, 0] + b4 * i3) / dt), g(X1))
    #     x.append(x_tmp)
    # end for
    return x


# def strongSRKp1Wm(f, G, dt, Y0, dw, name='SRK1Wm'):
#     """Stochastic Runge-Kutta method of strong order p = 1.
#     for the multidimensional Wiener process """
#
#     (nt, dim) = dw.shape
#     (s, a, b1, b2, c0, c1, A0, A1, B0, B1) = __strong_method_selector(name)
#
#     sqh = np.sqrt(dt)
#
#     x_num = []
#     x_tmp = Y0
#     i1 = Ito1Wm(dw, dt)
#     i2 = Ito2Wm(dw, dt)
#
#     X = np.zeros(shape=(dim + 1, s, dim))
#     # Here the index a stands for \ alpha
#     for i1, i2 in zip(i1, i2):
#         for i in range(s):
#             _f = np.array([f(var) for var in X[0]])
#             _G = np.array([[G(var2) for var2 in var1] for var1 in X[1:]])
#
#             X[0, i, :] = x_tmp + np.einsum("j, ja", A0[i], _f * dt)
#             X[0, i, :] += np.einsum("j, ljal, l", B0[i], _G, i1[1:])
#
#             for k in range(1, dim + 1, 1):
#                 _f = np.array([f(var) for var in X[0]])
#                 _G = np.array([[G(var2) for var2 in var1] for var1 in X[1:]])
#
#                 X[k, i, :] = x_tmp + np.einsum("j, ja", A1[i], _f * dt)
#                 X[k, i, :] += np.einsum("j, ljal, l", B1[i], _G, i2[1:, k]) / sqh
#
#         _f = np.array([f(var) for var in X[0]])
#         _G = np.array([[G(var2) for var2 in var1] for var1 in X[1:]])
#
#         x_tmp = x_tmp + np.einsum("i, ia", a, _f * dt)
#         x_tmp += np.einsum("i, k, kiak", b1, i1[1:], _G)
#         x_tmp += np.einsum("i, kiak", b2 * sqh, _G)
#
#         x_num.append(x_tmp)
#
#     return np.asarray(x_num)
#
#
# def strongSRKp1Wm1(f, g, h, Y0, dw):
#     """A special case of the strong-order stochastic Runge-Kutta method
#     for p = 1.and a multidimensional Wiener process. This function is needed for
#     speed """
#
#     (nt, dim) = dw.shape
#     s = 3
#
#     sqh = np.sqrt(h)
#
#     x_num = []
#     x_tmp = np.asarray(Y0)
#     I1 = Ito1Wm(dw, h)
#     I2 = Ito2Wm(dw, h, n=10)
#
#     X = np.zeros(shape=(dim + 1, s, dim))
#     for i1, i2 in zip(I1, I2):
#         for i in range(s):
#             X[0, i, :] = x_tmp
#         for k in range(1, dim + 1):
#             X[k, 0, :] = x_tmp
#
#         _G = g(x_tmp)
#         for k in range(1, dim + 1):
#             X[k, 1, :] = x_tmp + np.tensordot(_G, i2[1:, k] / sqh, axes=(1, 0))
#             X[k, 2, :] = x_tmp - np.tensordot(_G, i2[1:, k] / sqh, axes=(1, 0))
#
#         _G = np.array([[g(var2) for var2 in var1] for var1 in X[1:]])
#         x_tmp = x_tmp + f(x_tmp) * h + np.einsum("k, kak", i1[1:], _G[:, 0, :, :])
#         x_tmp += 0.5 * sqh * np.einsum("kak", _G[:, 1, :, :])
#         x_tmp -= 0.5 * sqh * np.einsum("kak", _G[:, 2, :, :])
#
#         # if x_tmp.min() <0.:
#         # break
#         x_num.append(x_tmp)
#
#     return np.asarray(x_num)
