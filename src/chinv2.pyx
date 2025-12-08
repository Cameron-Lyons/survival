# cython: language_level=3
import numpy as np
cimport numpy as cnp

def chinv2(cnp.ndarray[cnp.float64_t, ndim=2] matrix, int n):
    cdef double temp
    cdef int i, j, k

    for i in range(n):
        if matrix[i, i] > 0:
            matrix[i, i] = 1 / matrix[i, i]
            for j in range(i + 1, n):
                matrix[j, i] = -matrix[j, i]
                for k in range(i):
                    matrix[j, k] += matrix[j, i] * matrix[i, k]

    for i in range(n):
        if matrix[i, i] == 0:
            for j in range(i):
                matrix[j, i] = 0
            for j in range(i, n):
                matrix[i, j] = 0
        else:
            for j in range(i + 1, n):
                temp = matrix[j, i] * matrix[j, j]
                if j != i:
                    matrix[i, j] = temp
                for k in range(i, j):
                    matrix[i, k] += temp * matrix[j, k]
