from collections import namedtuple

import sympy as sp

# from sympy import Matrix, MutableDenseMatrix, Symbol, diff

"""
Named tuple that encapsulate a stochastic differential equation symbolically
"""
SDE = namedtuple('sde', ['drifts', 'diffusions', 'time', 'variables', 'brownians'])


def backward_diffusion_one_D(fun_array: sp.MutableDenseMatrix, sde: SDE) \
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

    res = sde.drifts[0] * sp.diff(fun_array, sde.variables[0])
    res += (sde.diffusions[0] * sde.diffusions[0]) * sp.diff(fun_array, sde.variables[0], 2) / 2
    return res


def column_polynomials_maximum_degree_one_D(col: sp.MutableDenseMatrix, variable: sp.Symbol) -> int:
    """
    Compute maximum polynomial degree in a vector with polynomial elements.
    Parameters
    ----------
    col : MutableDenseMatrix
        A vector with entries are symbolic functions (in sympy).

    variable : Symbol
        Sympy Symbol instance.

    Returns
    -------
    max_degree : int
        Maximum degree of polynomials in terms variable in the vector input.
    """
    max_degree = 0
    for entry in col:
        degree = entry.as_poly(variable).degree()
        if degree > max_degree:
            max_degree = degree
    return max_degree


def column_polynomials_coefficients_one_D(col: sp.MutableDenseMatrix, variable: sp.Symbol) -> sp.MutableDenseMatrix:
    """
    Compute a matrix with elements are coefficients of each element in a column vector with polynomial expressions.

    Parameters
    ----------
    col : MutableDenseMatrix
    variable : Symbol

    Returns
    -------
    res : MutableDenseMatrix
    """
    max_degree = column_polynomials_maximum_degree_one_D(col, variable)
    coeffs = [term.as_poly(variable).all_coeffs() for term in col]
    for coeff in coeffs:
        coeff.reverse()
        if len(coeff) - 1 < max_degree:
            for i in range(max_degree + 1 - len(coeff)):
                coeff.append(0)
    # # reverse back
    # for coeff in coeffs:
    #     coeff.reverse()

    return sp.Matrix(coeffs)
