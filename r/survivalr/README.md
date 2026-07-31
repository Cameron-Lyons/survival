# survivalr

`survivalr` is an experimental R facade for this repository's Rust-backed
Python package. It exposes familiar R entry points such as `Surv`, `survfit`,
`coxph`, `aareg`, `survdiff`, `survreg`, `basehaz`, `cox.zph`, and `concordance`, then
delegates computation to `survival.r_api` through `reticulate`.
Simple Python return values are converted back into R objects when possible;
fitted model objects stay wrapped so S3 methods can dispatch to the Python API.
Bridged models support standard R generics including `coef`, `vcov`, `confint`,
`logLik`, `nobs`, `extractAIC`, `fitted`, `summary`, `predict`, `residuals`,
`model.matrix`, `model.frame`, and `anova`.
Common result objects such as `survfit`, `basehaz`, `survdiff`, `concordance`,
`cox.zph`, `coxph.detail`, and `anova` outputs can also be converted with
`as.data.frame`.
Multi-state `survfit` objects with retained model frames support native-shaped
influence residuals and pseudo-values for state probabilities, cumulative
transition hazards, and integrated state occupancy.
`cch` fits use the Rust case-cohort kernel for Prentice, Self-Prentice,
Lin--Ying, I.Borgan, and II.Borgan estimation. This includes right-censored and
counting-process responses, sampling-stratified population sizes,
factor-expanded formulas, phase-two variance, optimal allocation fractions,
and the optional robust Lin--Ying variance.
`aareg` uses the Rust risk-set kernel for right-censored and counting-process
fits, including weights, tapering, clustered influence arrays, and standard R
model retention options.
`coxph` formula fits support `tt(...)` terms for right-censored and
counting-process responses, with either the default O'Brien transform or a
custom four-argument transform function.
Low-level `coxsurv.fit` and `survfitcoxph.fit` calls use a Rust risk-set sweep
for weighted, stratified, tied-event, and counting-process baselines. Their R
assembly preserves multiple prediction rows, standard errors, and individual
time-dependent trajectories.

Time-dependent data construction is also native: `tmerge` evaluates the
familiar `tdc`, `cumtdc`, `event`, and `cumevent` expressions locally, uses the
Rust-backed sweep kernels, and returns an ordinary R `tmerge` object with
`tm.retain` and `tcount` attributes preserved across repeated calls.

```r
baseline <- data.frame(id = 1:2, group = c("control", "treated"))
spans <- data.frame(id = 1:2, stop = c(10, 8))
updates <- data.frame(
  id = c(1, 1, 2),
  time = c(2, 6, 4),
  dose = c(5, 3, 4),
  status = c(0, 1, 1)
)

timeline <- tmerge(baseline, spans, id = id, tstop = stop)
timeline <- tmerge(
  timeline,
  updates,
  id = id,
  dose = tdc(time, dose, init = 0),
  total_dose = cumtdc(time, dose, init = 0),
  endpoint = event(time, status)
)
```

The package is intentionally named `survivalr` rather than `survival` so it can
coexist with CRAN's upstream `survival` package while this bridge matures.

Install the Python package first, then install this R package from the
`r/survivalr` directory:

```r
install.packages(c("reticulate", "remotes"))
remotes::install_local("r/survivalr")
```

If `reticulate` does not discover the intended Python environment, set it before
loading the package:

```r
reticulate::use_python("/path/to/python", required = TRUE)
library(survivalr)
```
