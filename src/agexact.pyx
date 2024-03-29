cdef extern from "math.h":
    double exp(double x)
    double log(double x)

cdef extern from "survS.h":

cdef extern from "survproto.h":


cdef void agexact(int* maxiter, int* nusedx, int* nvarx, double* start, 
                  double* stop, int* event, double* covar2, double* offset, 
                  int* strata, double* means, double* beta, double* u, 
                  double* imat2, double loglik[2], int* flag, double* work, 
                  int* work2, double* eps, double* tol_chol, 
                  double* sctest, int* nocenter):
    cdef:
        int i, j, k, l, person
        int iter
        int n, nvar
        double **covar, **cmat, **imat  # Ragged array versions
        double *a, *newbeta
        double *score, *newvar
        double denom, zbeta, weight
        double time
        double temp
        double newlk = 0
        int halving
        int nrisk, deaths
        int *index, *atrisk

