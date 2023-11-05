/*
** Anderson-Gill formulation of the cox Model
**   Do an exact calculation of the partial likelihood. (CPU city!)
**
**  the input parameters are
**
**       maxiter      :number of iterations
**       nused        :number of people
**       nvar         :number of covariates
**       start(n)     :each row covers the time interval (start,stop]
**       stop(n)      :
**       event(n)     :was there an event at 'stop':1=dead , 0=censored
**       covar(nv,n)  :covariates for person i.
**                        Note that S sends this in column major order.
**       strata(n)    :marks the strata.  Will be 1 if this person is the
**                       last one in a strata.  If there are no strata, the
**                       vector can be identically zero, since the nth person's
**                       value is always assumed to be = to 1.
**       offset(n)    :linear offset
**       eps          :tolerance for convergence.  Iteration continues until
**                       the percent change in loglikelihood is <= eps.
**       tol_chol     : tolerance for the Cholesky routine
**
**  returned parameters
**       means(nv)    :column means of the X matrix
**       beta(nv)     :the vector of answers (at start contains initial est)
**       u            :the first derivative vector at solution
**       imat(nv,nv)  :the variance matrix at beta=final, also a ragged array
**                      if flag<0, imat is undefined upon return
**       loglik(2)    :loglik at beta=initial values, at beta=final
**       sctest       :the score test at beta=initial
**       flag         :success flag  1000  did not converge
**                                   1 to nvar: rank of the solution
**       maxiter      :actual number of iterations used
**
**  work arrays
**       score(n)              the score exp(beta*z)
**       a(nvar)
**       cmat(nvar,nvar)       ragged array
**       newbeta(nvar)         always contains the "next iteration"
**
**  the 4 arrays score, a, cmat, and newbeta are passed as a single
**    vector of storage, and then broken out.
**
**  calls functions:  cholesky2, chsolve2, chinv2
**
**  the data must be sorted by ascending time within strata, deaths before
**          living within tied times.
*/

struct CoxModel {
    maxiter: usize,
    nused: usize,
    nvar: usize,
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<u8>,
    covar: Vec<Vec<f64>>,
    strata: Vec<u8>,
    offset: Vec<f64>,
    eps: f64,
    tol_chol: f64,
    // returned parameters
    means: Vec<f64>,
    beta: Vec<f64>,
    u: Vec<f64>,
    imat: Vec<Vec<f64>>,
    loglik: [f64; 2],
    sctest: f64,
    flag: i32,
    iter_used: usize,
    work: Vec<f64>,
}

impl CoxModel {
    pub fn new(
        maxiter: usize,
        nused: usize,
        nvar: usize,
        start: Vec<f64>,
        stop: Vec<f64>,
        event: Vec<u8>,
        covar: Vec<Vec<f64>>,
        strata: Vec<u8>,
        offset: Vec<f64>,
        eps: f64,
        tol_chol: f64,
        initial_beta: Vec<f64>,
    ) -> CoxModel {
        let means = vec![0.0; nvar];
        let imat = vec![vec![0.0; nvar]; nvar];
        let u = vec![0.0; nvar];

        let loglik = [0.0, 0.0];
        let sctest = 0.0;
        let flag = 0;

        // Work arrays
        let work_len = nvar * nvar + nvar + nvar; 
        let work = vec![0.0; work_len];

        CoxModel {
            maxiter,
            nused,
            nvar,
            start,
            stop,
            event,
            covar,
            strata,
            offset,
            eps,
            tol_chol,
            means,
            beta: initial_beta,
            u,
            imat,
            loglik,
            sctest,
            flag,
            iter_used: 0,
            work,
        }
    }

    pub fn compute(&mut self) {
        for iter in 0..self.maxiter {
            self.score_and_info();

            let delta_beta = self.solve_system();

            for i in 0..self.nvar {
                self.beta[i] += delta_beta[i];
            }

            if self.has_converged(&delta_beta) {
                self.iter_used = iter + 1;
                self.flag = 0; // Indicating successful convergence
                break;
            }
        if self.flag != 0 {
            self.flag = 1000;
        }
        self.finalize_statistics();
    }
}
