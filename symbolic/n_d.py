from typing import Tuple, List, Set
import sympy as sp
import numpy as onp

from symbolic import SDE


def backward_diffusion(fun_array: sp.MutableDenseMatrix, sde: SDE) \
        -> sp.MutableDenseMatrix:
    """
    Compute backward diffusion operator of a given function for a given sde. The function and sde
    should be function of the same symbolic variable.

    Parameters
    ----------
    fun_array : Callable[[sympy.Symbol], sympy.Symbol]
    sde : SDE

    Returns
    -------
    res : sympy.matrices.dense.MutableDenseMatrix
    """
    jac = fun_array.jacobian(sde.variables)
    res = jac * sde.drifts
    for i in range(jac.shape[0]):
        res[i] += sp.trace(sde.diffusions.transpose() * jac[i, :].jacobian(sde.variables) * sde.diffusions) / 2
    return res


def get_monomial_degree_set(fun_array: sp.MutableDenseMatrix, variables: Tuple[sp.Symbol]) -> Set[Tuple[int]]:
    """
    Get coefficients of monomials from an array of polynomials given in terms of variables in a tuple.

    Parameters
    ----------
    fun_array
    variables

    Returns
    -------

    """
    monomial_degree_set = set()
    for an_entry in fun_array:
        monoms = an_entry.as_poly(variables).monoms()
        for monom in monoms:
            monomial_degree_set.add(monom)

    # res = list(monomial_degree_set)
    # res.sort()
    return monomial_degree_set


def from_tuple_to_symbolic_monom(variables: Tuple[sp.Symbol], degree_tuple: Tuple[int]) -> sp.Symbol:
    """
    Convert

    Parameters
    ----------
    variables
    degree_tuple

    Returns
    -------

    """
    res = 1
    for i in range(len(degree_tuple)):
        res *= variables[i] ** degree_tuple[i]
    return res


def column_polynomials_coefficients(col: sp.MutableDenseMatrix, variables: Tuple[sp.Symbol],
                                    monomial_degree_list: List[Tuple[int, int]] = None) -> Tuple[List[sp.Symbol],
                                                                                                 onp.ndarray]:
    """

    Parameters
    ----------
    monomial_degree_list
    col
    variables

    Returns
    -------

    """
    if not monomial_degree_list:
        monomial_degree_list = get_monomial_degree_set(col, variables)

    monoms = [from_tuple_to_symbolic_monom(variables, monomial_degree) for monomial_degree in monomial_degree_list]
    coefficients = onp.zeros((col.shape[0], len(monoms)))
    for i in range(col.shape[0]):
        for j in range(len(monoms)):
            coefficients[i, j] = col[i].as_poly(variables).coeff_monomial(monoms[j])

    return monoms, coefficients
