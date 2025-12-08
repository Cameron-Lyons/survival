
# cython: language_level=3
from libc.math cimport isfinite
import numpy as np
cimport numpy as cnp

def cholesky2(cnp.ndarray[cnp.float64_t, ndim=2] matrix, int n, double toler):
    cdef double temp
    cdef int i, j, k
    cdef double eps, pivot
    cdef int rank
    cdef int nonneg

    nonneg = 1
    eps = 0
    for i in range(n):
        if matrix[i, i] > eps:
            eps = matrix[i, i]
        for j in range(i + 1, n):
            matrix[j, i] = matrix[i, j]
    if eps == 0:
        eps = toler
    else:
        eps *= toler

    rank = 0
    for i in range(n):
        pivot = matrix[i, i]
        if not isfinite(pivot) or pivot < eps:
            matrix[i, i] = 0
            if pivot < -8 * eps:
                nonneg = -1
        else:
            rank += 1
            for j in range(i + 1, n):
                temp = matrix[j, i] / pivot
                matrix[j, i] = temp
                matrix[j, j] -= temp * temp * pivot
                for k in range(j + 1, n):
                    matrix[k, j] -= temp * matrix[k, i]

    return rank * nonneg
