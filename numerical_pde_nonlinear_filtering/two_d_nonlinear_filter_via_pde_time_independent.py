from typing import Callable, Tuple

import jax.numpy as jnp
import numpy as onp
import numba as nb
import jax.experimental.sparse_ops as jsp
import scipy.sparse as ssp
import scipy.sparse.linalg as ssla

from numerical_pde_nonlinear_filtering.two_d_nonlinear_filter_via_pde_clean import TwoDNonLinearFilterPDE


@nb.njit(fastmath=True)
def crop_index(an_index: int, max_limit: int) -> int:
    return min(max(0, an_index), max_limit)


@nb.njit(fastmath=True)
def generate_crank_nicolson_coo_matrix(f_eval: onp.ndarray, D_eval: onp.ndarray, grids: onp.ndarray, dx: onp.ndarray,
                                       dt: float):
    # matrix_n_plus_one = onp.zeros((grids.shape[0] * grids.shape[1], grids.shape[0] *
    #                                grids.shape[1]), dtype=onp.float32)
    diagonal_default_num = grids.shape[0] * grids.shape[1]
    non_zero_count = diagonal_default_num + 2 * (diagonal_default_num - 1) + \
                     2 * (diagonal_default_num - grids.shape[1]) + 2 * (diagonal_default_num - grids.shape[1] - 1) + \
                     2 * (diagonal_default_num - grids.shape[1] + 1)
    matrix_n_plus_one_entries = onp.zeros((non_zero_count,), dtype=onp.float32)
    row_index = onp.zeros((non_zero_count,), dtype=onp.int32)
    col_index = onp.zeros((non_zero_count,), dtype=onp.int32)
    temp_entries = onp.zeros((9,), dtype=onp.float32)
    temp_col_indices = onp.zeros((9,), dtype=onp.int32)
    current_entry_index = -1

    f1_scaled = f_eval[:, :, 0] / (2 * dx[0])
    f2_scaled = f_eval[:, :, 1] / (2 * dx[1])
    d11_scaled = D_eval[:, :, 0, 0] / (dx[0] * dx[0])
    d22_scaled = D_eval[:, :, 1, 1] / (dx[1] * dx[1])
    d12_scaled = (D_eval[:, :, 0, 1] + D_eval[:, :, 1, 0]) / (dx[1] * dx[0] * 4)

    for i in range(grids.shape[0]):
        for j in range(grids.shape[1]):
            i_plus = crop_index(i + 1, grids.shape[0] - 1)
            i_min = crop_index(i - 1, grids.shape[0] - 1)
            j_plus = crop_index(j + 1, grids.shape[1] - 1)
            j_min = crop_index(j - 1, grids.shape[1] - 1)

            f1_i_plus_j = f1_scaled[i_plus, j]
            f1_i_min_j = f1_scaled[i_min, j]

            f2_i_j_plus = f2_scaled[i, j_plus]
            f2_i_j_min = f2_scaled[i, j_min]

            d11_i_min_j = d11_scaled[i_min, j]
            d11_i_j = d11_scaled[i, j]
            d11_i_plus_j = d11_scaled[i_plus, j]

            d22_i_j_min = d22_scaled[i, j_min]
            d22_i_j = d22_scaled[i, j]
            d22_i_j_plus = d22_scaled[i, j_plus]

            d12_i_plus_j_plus = d12_scaled[i_plus, j_plus]
            d12_i_plus_j_min = d12_scaled[i_plus, j_min]
            d12_i_min_j_plus = d12_scaled[i_min, j_plus]
            d12_i_min_j_min = d12_scaled[i_min, j_min]

            temp_col_indices[0] = i_min * grids.shape[1] + j_min
            temp_col_indices[1] = i_min * grids.shape[1] + j
            temp_col_indices[2] = i_min * grids.shape[1] + j_plus
            temp_col_indices[3] = i * grids.shape[1] + j_min
            temp_col_indices[4] = i * grids.shape[1] + j
            temp_col_indices[5] = i * grids.shape[1] + j_plus
            temp_col_indices[6] = i_plus * grids.shape[1] + j_min
            temp_col_indices[7] = i_plus * grids.shape[1] + j
            temp_col_indices[8] = i_plus * grids.shape[1] + j_plus

            temp_entries[0] = - 0.5 * dt * d12_i_min_j_min
            temp_entries[1] = - 0.5 * dt * (f1_i_min_j + d11_i_min_j)
            temp_entries[2] = 0.5 * dt * (- d12_i_min_j_plus)

            temp_entries[3] = - 0.5 * dt * (f2_i_j_min + d22_i_j_min)
            temp_entries[4] = 1 - 0.5 * dt * (-2 * d11_i_j - 2 * d22_i_j)
            temp_entries[5] = 0.5 * dt * (f2_i_j_plus - d22_i_j_plus)

            temp_entries[6] = - 0.5 * dt * (- d12_i_plus_j_min)
            temp_entries[7] = - 0.5 * dt * (- f1_i_plus_j + d11_i_plus_j)
            temp_entries[8] = - 0.5 * dt * d12_i_plus_j_plus

            current_row_index = temp_col_indices[4]
            current_col_index = temp_col_indices[0]
            row_index[current_entry_index] = current_row_index
            col_index[current_entry_index] = current_col_index

            current_entry_index += 1
            sort_indices = onp.argsort(temp_col_indices)
            temp_col_indices.sort()




            matrix_n_plus_one_entries[current_entry_index] = temp_entries[sort_indices[0]]

            for k in range(1, temp_col_indices.shape[0]):
                if temp_col_indices[k] == current_col_index:
                    matrix_n_plus_one_entries[current_entry_index] += temp_entries[sort_indices[k]]
                    continue
                if temp_col_indices[k] != current_col_index:
                    current_entry_index += 1
                    current_col_index = temp_col_indices[k]
                    row_index[current_entry_index] = current_row_index
                    col_index[current_entry_index] = current_col_index
                    matrix_n_plus_one_entries[current_entry_index] = temp_entries[sort_indices[k]]

            # matrix_n_plus_one[index_ij, index_i_min_j_min] += - 0.5 * dt * d12_i_min_j_min
            # matrix_n_plus_one[index_ij, index_i_min_j] += - 0.5 * dt * (f1_i_min_j + d11_i_min_j)
            # matrix_n_plus_one[index_ij, index_i_min_j_plus] += 0.5 * dt * (- d12_i_min_j_plus)
            #
            # matrix_n_plus_one[index_ij, index_ij_min] += - 0.5 * dt * (f2_i_j_min + d22_i_j_min)
            # matrix_n_plus_one[index_ij, index_ij] += 1 - 0.5 * dt * (-2 * d11_i_j - 2 * d22_i_j)
            # matrix_n_plus_one[index_ij, index_ij_plus] += 0.5 * dt * (f2_i_j_plus - d22_i_j_plus)
            #
            # matrix_n_plus_one[index_ij, index_i_plus_j_min] += - 0.5 * dt * (- d12_i_plus_j_min)
            # matrix_n_plus_one[index_ij, index_i_plus_j] += - 0.5 * dt * (- f1_i_plus_j + d11_i_plus_j)
            # matrix_n_plus_one[index_ij, index_i_plus_j_plus] += - 0.5 * dt * d12_i_plus_j_plus

    # matrix_n = - matrix_n_plus_one + 2 * onp.eye(matrix_n_plus_one.shape[0], dtype=onp.float32)
    # return matrix_n_plus_one, matrix_n
    return matrix_n_plus_one_entries[:current_entry_index+1], row_index[:current_entry_index+1], \
           col_index[:current_entry_index+1]


class TimeIndependentTwoDNonLinearFilterPDE(TwoDNonLinearFilterPDE):
    def __init__(self,
                 one_d_grids: Tuple[jnp.ndarray, jnp.ndarray],
                 dynamic_drift: Callable[[jnp.ndarray, float], jnp.ndarray],
                 dynamic_diffusion: Callable[[jnp.ndarray, float], jnp.ndarray],
                 measurement_drift: Callable[[jnp.ndarray, float], jnp.ndarray],
                 initial_condition: jnp.ndarray,
                 delta_t: float,
                 measurement_record: jnp.ndarray,
                 measurement_stdev: float = 1.,
                 ode_solver: str = 'EULER'
                 ):
        """Class that encapsulates a two dimensional optimal filtering problem PDE solution (Zakai - equation)
        using finite difference scheme [1].
        The grid is assumed to be equidistance, and the finite difference is order one center.
        

        Parameters
        ----------
        one_d_grids : Tuple[jnp.ndarray, jnp.ndarray]
            tuples of d one dimensional grids, each has shape [Ng]
        dynamic_drift : Callable[[jnp.ndarray, float], jnp.ndarray]
            Drift part of the state dynamic sde
        dynamic_diffusion : Callable[[jnp.ndarray, float], jnp.ndarray]
            Diffusion part of the state dynamic sde
        measurement_drift : Callable[[jnp.ndarray, float], jnp.ndarray]
            Drift part of the measurement sde
        initial_condition : jnp.ndarray
            Initial condition, the array should match the shape of the grids
        delta_t : float
            Time step
        measurement_record : jnp.ndarray
            Measurement records array of shape [Nt x m]
        measurement_stdev : float, optional
            measurement standard deviation, by default 1.
        ode_solver : str, optional
            ODE solver option, by default 'EULER'

        References
        ----------
        [1] Alan Bain , Dan Crisan, Fundamentals of Stochastic Filtering
        """
        super().__init__(one_d_grids=one_d_grids,
                         dynamic_drift=dynamic_drift,
                         dynamic_diffusion=dynamic_diffusion,
                         measurement_drift=measurement_drift,
                         initial_condition=initial_condition,
                         delta_t=delta_t,
                         measurement_record=measurement_record,
                         measurement_stdev=measurement_stdev,
                         ode_solver=ode_solver
                         )

        # if dynamical_drif, and dynamical_diffusion are not function of time
        self._f_eval = self._dynamic_drift(self._grids, 0)
        self._g_eval = self._dynamic_diffusion(self._grids, 0)
        self._D_eval = 0.5 * jnp.einsum('ijkl,ijml->ijkm', self._g_eval, self._g_eval)
        matrix_n_plus_one_entries, row_index, col_index = self._generate_crank_nicolson_matrices()
        _matrix_n_plus_one_sparse = ssp.coo_matrix((matrix_n_plus_one_entries, (row_index, col_index)),
                                                   shape=(self._grids.shape[0] * self._grids.shape[1],
                                                          self._grids.shape[0] * self._grids.shape[1]))
        eye = ssp.eye(_matrix_n_plus_one_sparse.shape[0], dtype=onp.float32)
        # _matrix_n = - _matrix_n_plus_one_sparse + 2 * ssp.eye(_matrix_n_plus_one_sparse.shape[0],
        #                                                       dtype=onp.float32)
        _crank_nicolson_matrix_sparse = - eye + 2*ssla.spsolve(_matrix_n_plus_one_sparse, eye)
        _crank_nicolson_matrix_sparse = _crank_nicolson_matrix_sparse.tocoo()

        self._crank_nicolson_matrix = jsp.COO((_crank_nicolson_matrix_sparse.data,
                                               _crank_nicolson_matrix_sparse.row,
                                               _crank_nicolson_matrix_sparse.col),
                                              shape=_matrix_n_plus_one_sparse.shape)
        self._ode_solver = 'crank_nicholson'
        self._one_step_fokker_planck = self._crank_nicolson

    def _generate_crank_nicolson_matrices(self):
        return generate_crank_nicolson_coo_matrix(onp.asarray(self._f_eval),
                                                  onp.asarray(self._D_eval),
                                                  onp.asarray(self._grids),
                                                  onp.asarray(self._dx),
                                                  self._dt)

    def _crank_nicolson(self, p: jnp.ndarray, t: float):
        # p_flatten = jnp.linalg.solve(self._matrix_n_plus_one, self._matrix_n @ p.flatten())
        # p_flatten = self._matrix_n_plus_one_inverse@self._matrix_n @ p.flatten()
        p_flatten = self._crank_nicolson_matrix.matvec(p.flatten())
        return p_flatten.reshape((self._grids.shape[0], self._grids.shape[1]))
