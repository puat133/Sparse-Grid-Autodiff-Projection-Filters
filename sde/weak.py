# import jax.numpy as np
#
#
# def __weak_method_selector(name):
#     """Filling arrays of the Butcher table for a specific method"""
#     if name == 'SRK1Wm':
#         s = 3
#         a = np.array([0.1, 3.0 / 14.0, 24.0 / 35.0])
#         b1 = np.array([1.0, -1.0, -1.0])
#         b2 = np.array([0.0, 1.0, -1.0])
#         b3 = np.array([0.5, -0.25, -0.25])
#         b4 = np.array([0.0, 0.5, -0.5])
#         c0 = np.array([0.0, 1.0, 5.0 / 12.0])
#         c1 = np.array([0.0, 0.25, 0.25])
#         c2 = np.array([0.0, 0.0, 0.0])
#         A0 = np.array([[0.0, 0.0, 0.0],
#                        [1.0, 0.0, 0.0],
#                        [25.0 / 144.0, 35.0 / 144.0, 0.0]])
#         A1 = np.array([[0.0, 0.0, 0.0],
#                        [0.25, 0.0, 0.0],
#                        [0.25, 0.0, 0.0]])
#         A2 = np.array([[0.0, 0.0, 0.0],
#                        [0.0, 0.0, 0.0],
#                        [0.0, 0.0, 0.0]])
#         B0 = np.array([[0.0, 0.0, 0.0],
#                        [1.0 / 3.0, 0.0, 0.0],
#                        [-5.0 / 6.0, 0.0, 0.0]])
#         B1 = np.array([[0.0, 0.0, 0.0],
#                        [0.5, 0.0, 0.0],
#                        [-0.5, 0.0, 0.0]])
#         B2 = np.array([[0.0, 0.0, 0.0],
#                        [1.0, 0.0, 0.0],
#                        [-1.0, 0.0, 0.0]])
#     elif name == 'SRK2Wm':
#         s = 3
#         a = np.array([0.5, 0.5, 0.0])
#         b1 = np.array([0.5, 0.25, 0.25])
#         b2 = np.array([0.0, 0.5, -0.5])
#         b3 = np.array([- 0.5, 0.25, 0.25])
#         b4 = np.array([0.0, 0.5, -0.5])
#         c0 = np.array([0.0, 1.0, 0.0])
#         c1 = np.array([0.0, 1.0, 1.0])
#         c2 = np.array([0.0, 0.0, 0.0])
#         A0 = np.array([[0.0, 0.0, 0.0],
#                        [1.0, 0.0, 0.0],
#                        [0.0, 0.0, 0.0]])
#         A1 = np.array([[0.0, 0.0, 0.0],
#                        [1.0, 0.0, 0.0],
#                        [1.0, 0.0, 0.0]])
#         A2 = np.array([[0.0, 0.0, 0.0],
#                        [0.0, 0.0, 0.0],
#                        [0.0, 0.0, 0.0]])
#         B0 = np.array([[0.0, 0.0, 0.0],
#                        [1.0, 0.0, 0.0],
#                        [0.0, 0.0, 0.0]])
#         B1 = np.array([[0.0, 0.0, 0.0],
#                        [1.0, 0.0, 0.0],
#                        [-1.0, 0.0, 0.0]])
#         B2 = np.array([[0.0, 0.0, 0.0],
#                        [1.0, 0.0, 0.0],
#                        [-1.0, 0.0, 0.0]])
#     # end if
#     return s, a, b1, b2, b3, b4, c0, c1, c2, A0, A1, A2, B0, B1, B2
#
#
# # ------------------------------------------------- ---------------------------
# # Numerical methods with weak convergence
# # ------------------------------------------------- ---------------------------
# def n_point_distribution(values, probabilities, shape):
#     """n-point distribution"""
#     index = np.arange(len(values))
#     n_point = st.rv_discrete(name='n_point', values=(index, probabilities))
#     num = np.prod(shape)
#     res = np.asarray(values)[n_point.rvs(size=num)].reshape(shape)
#     return res
#
#
# def weakIto(h, dim, N):
#     """The function calculates approximations for the Ito integrals in the case of a weak
#     convergence.
#     h --- step
#     dim --- SDE dimension
#     N --- the number of points on the time interval
#     """
#     # Two point distribution
#     x = (-np.sqrt(h), np.sqrt(h))
#     p = (0.5, 0.5)
#     I_tilde = n_point_distribution(x, p, shape=(N, dim))
#
#     # Three point distribution
#     x = (-np.sqrt(3.0 * h), 0.0, np.sqrt(3.0 * h))
#     p = (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)
#     I_hat = n_point_distribution(x, p, shape=(N, dim))
#
#     I_ = np.empty(shape=(N, dim, dim))
#
#     it = np.nditer(I_, flags=['multi_index'], op_flags=['writeonly'])
#     while not it.finished:
#         i = it.multi_index[0]
#         k = it.multi_index[1]
#         ell = it.multi_index[2]
#         if k < ell:
#             it[0] = 0.5 * (I_hat[i, k] * I_hat[i, ell] - np.sqrt(h) * I_tilde[i, k])
#         elif ell < k:
#             it[0] = 0.5 * (I_hat[i, k] * I_hat[i, ell] + np.sqrt(h) * I_tilde[i, ell])
#         elif ell == k:
#             it[0] = 0.5 * (I_hat[i, k] ** 2 - h)
#         it.iternext()
#
#     return I_hat, I_
#
#
# def weakSRKp2Wm(f, G, h, x_0, dW, name='SRK1Wm'):
#     """Stochastic Runge-Kutta method of weak order p = 2.0
#     for the multidimensional Wiener process """
#
#     (N, dim) = dW.shape
#     (s, a, b1, b2, b3, b4, c0, c1,
#      c2, A0, A1, A2, B0, B1, B2) = __weak_method_selector(name)
#
#     sqh = np.sqrt(h)
#
#     x_num = []
#     x_tmp = x_0
#     (I1, I2) = weakIto(h, dim, N)
#
#     X = np.zeros(shape=(dim + 1, s, dim))
#     X_hat = np.zeros(shape=(dim, s, dim))
#     # Here the index a stands for \ alpha
#     for i1, i2 in zip(I1, I2):
#         for i in range(s):
#             _f = np.array([f(var) for var in X[0]])
#             _G = np.array([[G(var2) for var2 in var1] for var1 in X[1:]])
#
#             X[0, i, :] = x_tmp + np.einsum("j, ja", A0[i], _f * h)
#             X[0, i, :] += np.einsum("j, ljal, l", B0[i], _G, i1)
#
#             for k in range(1, dim + 1, 1):
#                 _f = np.array([f(var) for var in X[0]])
#                 _G = np.array([[G(var2) for var2 in var1] for var1 in X[1:]])
#
#                 X[k, i, :] = x_tmp + np.einsum("j, ja", A1[i], _f * h)
#                 X[k, i, :] += np.einsum("j, ja", B1[i], _G[k, :, :, k]) * sqh
#
#                 X_hat[k, i, :] = x_tmp + np.einsum("j, ja", A2[i], _f * h)
#                 # we must exclude the case l = k from the sum, so we have to
#                 # sum in cycles, not einsum
#                 X_hat[k, i, :] += np.array([np.sum([
#                     [B2[i, j] * _G[ell, j, alpha, ell] * i2[k - 1, ell] / sqh for ell in range(dim) if ell != k - 1]
#                     for j in range(s)]) for alpha in range(dim)])
#
#         _f = np.array([f(var) for var in X[0]])
#         _G = np.array([[G(var2) for var2 in var1] for var1 in X[1:]])
#
#         x_tmp = x_tmp + np.einsum("i, ia", a, _f) * h
#         x_tmp += np.einsum("i, k, kiak", b1, i1, _G)
#         x_tmp += np.einsum("i, kiak, kk", b2, _G, i2) / sqh
#         x_tmp += np.einsum("i, kiak, k", b3, _G, i1)
#         x_tmp += np.einsum("i, kiak", b4, _G) * sqh
#
#         x_num.append(x_tmp)
#
#     return np.asarray(x_num)
