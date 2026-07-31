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
Unstratified `cch` fits use the Rust case-cohort kernel for Prentice,
Self-Prentice, and Lin--Ying estimation, including right-censored and
counting-process responses, factor-expanded formulas, phase-two variance, and
the optional robust Lin--Ying variance.
`aareg` uses the Rust risk-set kernel for right-censored and counting-process
fits, including weights, tapering, clustered influence arrays, and standard R
model retention options.

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
