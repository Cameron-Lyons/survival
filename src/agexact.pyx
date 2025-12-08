from libc.math cimport exp, log
import numpy as np
cimport numpy as cnp

cdef extern from "survS.h":
    double** dmatrix(double* data, int nrows, int ncols)

from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp

cpdef void agexact(int* maxiter, int* nusedx, int* nvarx, double* start, 
                   double* stop, int* event, double* covar2, double* offset, 
                   int* strata, double* means, double* beta, double* u, 
                   double* imat2, double loglik[2], int* flag, double* work, 
                   int* work2, double* eps, double* tol_chol, 
                   double* sctest, int* nocenter):
    cdef:
        int i, j, k, person
        int iter
        int n = nusedx[0]
        int nvar = nvarx[0]
        double **covar, **imat
        double *a, *newbeta
        double temp
        double *score
        double **cmat
        double *newvar
        int *index
        int deaths, nrisk
        int halving = 0

    covar = <double **>malloc(n * sizeof(double *))
    for i in range(n):
        covar[i] = <double *>malloc(nvar * sizeof(double))

    imat = <double **>malloc(nvar * sizeof(double *))
    for i in range(nvar):
        imat[i] = <double *>malloc(nvar * sizeof(double))
    
    cmat = <double **>malloc(nvar * sizeof(double *))
    for i in range(nvar):
        cmat[i] = <double *>malloc(nvar * sizeof(double))
    
    a = <double *>malloc(nvar * sizeof(double))
    newbeta = <double *>malloc(nvar * sizeof(double))
    score = <double *>malloc(nvar * sizeof(double))
    newvar = <double *>malloc(nvar * sizeof(double))
    index = <int *>malloc(n * sizeof(int))

    for i in range(n):
        free(covar[i])
    free(covar)

    for i in range(nvar):
        free(imat[i])
    free(imat)

    for i in range(nvar):
        free(cmat[i])
    free(cmat)
    
    free(a)
    free(newbeta)
    free(score)
    free(newvar)
    free(index)

