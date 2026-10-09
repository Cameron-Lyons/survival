# Rank-deficient AFT designs

`survreg` keeps the complete fitted design when a column is redundant. The
native Newton/Fisher iteration uses R survival's generalized Cholesky solve;
the full fit marks the aliased location coefficient as `NaN` (R's `NA`) and
keeps its covariance row and column at zero. The compact `survreg_fit` result
retains the solver's numeric coefficient, normally zero with automatic
starting values, before full-model marking. Degrees of freedom retain R's
parameter-count convention, including redundant columns and estimated scales.

Constant numeric columns remain unscaled, including constants other than zero
or one. Previously a constant Helmert contrast of `-1`, produced by a declared
but unobserved fourth factor level, was divided by its zero sample standard
deviation. This changed an otherwise finite design into `NaN` before the
optimizer could identify the redundancy. Leaving that column on its original
scale permits the same native likelihood and solver to fit the identified
part of the model. Nonconstant numeric columns still receive the existing
centering and scaling when the design starts with an intercept and no initial
estimates are supplied.

Training linear predictors, their standard errors, and predicted quantiles
remain finite and agree with the equivalent identified model. Covariance
comparisons include its location and estimated-scale block, with robust
covariance and the model-based covariance retained separately. Prediction
from explicit new data and term predictions retain stock `predict.survreg`'s
behavior: multiplying the aliased `NA` coefficient makes those affected
values missing. The fit does not identify an effect for a previously
unobserved factor level.

The correction deliberately differs from survival 3.8-12's automatic
constant-column rescaling. Stock R can fail with a coefficient-name error or
return a nonfinite fit. Supplying initial estimates bypasses that rescaling;
the stock native solver then returns the same identified fit, singular
coefficient, and generalized covariance as the port.

The [reference generator](../scripts/generate_aft_rank_deficient_reference.R)
records 33 cases from R 4.5.3 and survival 3.8-12. They cover unused Helmert
levels, a nonbinary numeric constant, an affine dependent column, ordinary
and clustered weighted fits, scale strata, and Weibull, lognormal, Gaussian
and exponential distributions. Each reference retains stock R's failure or
warnings. For those failures the generator changes only the constant-column
scaling predicate in `survreg.fit`; the installed likelihood, optimizer,
generalized inverse, and methods stay unchanged. Every case is independently
checked against a stock identified fit and a stock full-design fit initialized
from that identified estimate.

Primary implementations:
[survreg.fit](https://github.com/therneau/survival/blob/master/R/survreg.fit.R),
[survreg6](https://github.com/therneau/survival/blob/master/src/survreg6.c),
[survreg's singular marking](https://github.com/therneau/survival/blob/master/R/survreg.R),
and [predict.survreg](https://github.com/therneau/survival/blob/master/R/predict.survreg.R).
