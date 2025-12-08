# cython: language_level=3
cimport numpy as cnp
import numpy as np

cdef int maxval, minval
cdef int firsttime, depth

def init_doloop(int min, int max):
    global firsttime, minval, maxval, depth
    firsttime = 1
    minval = min
    maxval = max
    depth = 1

def doloop(int nloops, cnp.ndarray[cnp.int_t, ndim=1] index):
    cdef int i

    global firsttime, minval, maxval, depth

    if firsttime == 1:
        for i in range(nloops):
            index[i] = minval + i
        firsttime = 0
        if maxval >= (minval + i):
            return minval + i - 1
        else:
            return minval - 1

    nloops -= 1
    index[nloops] += 1

    if index[nloops] <= (maxval - depth):
        return index[nloops]
    elif nloops == 0:
        return minval - depth
    else:
        depth += 1
        index[nloops] = doloop(nloops, index) + 1
        depth -= 1
        return index[nloops]
