
# cython: language_level=3
import numpy as np
cimport numpy as cnp

def chsolve2(cnp.ndarray[cnp.float64_t, ndim=2] matrix, int n, cnp.ndarray[cnp.float64_t, ndim=1] y):
    cdef int i, j
    cdef double temp

    for i in range(n):
        temp = y[i]
        for j in range(i):
            temp -= y[j] * matrix[i, j]
        y[i] = temp

    for i in range(n - 1, -1, -1):
        if matrix[i, i] == 0:
            y[i] = 0
        else:
            temp = y[i] / matrix[i, i]
            for j in range(i + 1, n):
                temp -= y[j] * matrix[j, i]
            y[i] = temp
