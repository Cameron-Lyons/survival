from libc.math cimport exp, log
import numpy as np
cimport numpy as cnp

cdef extern from "survS.h":
    double** dmatrix(double* data, int nrows, int ncols)

cdef extern from "survproto.h":
    int cholesky2(double** mat, int n, double tol)
    void chsolve2(double** mat, int n, double* b)
    void chinv2(double** mat, int n)
    int init_doloop(int start, int end)
    int doloop(int size, int* index)

cpdef void agexact(int* maxiter, int* nusedx, int* nvarx, double* start, 
                   double* stop, int* event, double* covar2, double* offset, 
                   int* strata, double* means, double* beta, double* u, 
                   double* imat2, double loglik[2], int* flag, double* work, 
                   int* work2, double* eps, double* tol_chol, 
                   double* sctest, int* nocenter):
    cdef:
        int i, j, k, l, person
        int iter
        int n = nusedx[0]
        int nvar = nvarx[0]
        double **covar, **cmat, **imat
        double *a, *newbeta
        double *score, *newvar
        double denom, zbeta, weight
        double time
        double temp
        double newlk = 0
        int halving
        int nrisk, deaths
        int *index, *atrisk

