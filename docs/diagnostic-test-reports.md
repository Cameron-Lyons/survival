# Diagnostic and test reports

The R-style API returns `ModelPrint` objects for case-cohort models,
conditional logistic regression, proportional-hazards diagnostics,
concordance and survival-difference tests. Reports reuse existing fits and
numerical results. Constructing them does not write to stdout.

```python
from survival import datasets, r

lung = datasets.load_lung()
fit = r.coxph("Surv(time, status) ~ age + sex", lung)
print(r.print_cox_zph(r.cox_zph(fit)))
print(r.print_concordance(r.concordance(fit)))
report = r.print_survdiff(r.survdiff("Surv(time, status) ~ sex", lung))
print(report)
columns = r.as_data_frame(report)
```

`lines` contains formatted text; `str(report)` appends a final newline.
`tables` contains independent `NamedMatrix` values at full precision, and
`statistics` holds scalar results and report metadata. `as_data_frame` returns
independent columns from `tables[report.primary_table]`, with row names under
`report.row_label` when present. R call expressions and trailing whitespace
are omitted. Options are explicit; no global print state changes.

| Function | Primary table | Default precision | Other content |
| --- | --- | --- | --- |
| `print_cch(fit)` | `coefficients` | Seven significant figures | Cohort/subcohort sizes |
| `print_summary_cch(model_summary(fit))` | `coefficients` | Three decimal places, then seven-digit layout | Hazard ratios and 1.96-SE confidence limits |
| `print_clogit(fit)` | `coefficients` | Four significant figures | Existing Cox model footer |
| `print_cox_zph(result)` | `tests` | Three significant figures | Term and optional global tests |
| `print_concordance(result)` | `concordance` | Four significant figures for scalar estimates; four decimal places for multiple estimates | Standard errors, pair counts, sample size |
| `print_survConcordance(result)` | `counts` | Seven significant figures | Legacy estimates, counts and sample size |
| `print_survdiff(result)` | `test` | Three significant figures | Chi-square, degrees of freedom, p-value |

Every function accepts `width=80` for column-block wrapping. Supported widths
are 10–10,000. `print_clogit`, `print_cox_zph`, `print_concordance` and
`print_survdiff` accept `digits` from 1 to 22. The first two also accept
`signif_stars=False`. Coefficient and p-value formatting follows each R
method's precision rules; it does not simply round every value alike.

Case-cohort reports include all five estimators and robust Lin–Ying fits.
`tables["sizes"]` retains named strata. Direct reports show the coefficient,
standard error, absolute z statistic and p-value. Summary reports add hazard
ratios and confidence limits, preserving overflow as infinity. R's summary
printer ignores its `digits` argument; the Python method likewise validates
but ignores that argument and uses three decimal places. The text preserves
R's literal `x$method,` heading and its single-coefficient row labels (`[1,]`
for the fit and `Value` for the summary). Numeric tables retain the actual
coefficient name. Summary calculation copies the covariance matrix once,
removing a repeated full-matrix copy for each coefficient.

Modern concordance reports retain `tables["counts"]` with one row or separate
predictor/stratum rows. Displayed counts round to two decimal places before
R's seven-digit numeric formatting; stored weighted counts stay unrounded.
Standard errors come from the variance or its diagonal. Missing variance omits
the standard error. An absence of comparable pairs displays `NaN`.
Formula results preserve `na_action`; concordance computed from a fitted
model does not invent an omission record from that model's training data.
The legacy report function is explicit because R 3.8-12 contains
`print.survConcordance` but no longer registers it for automatic dispatch.
Calling the Python legacy estimator retains its existing deprecation warning;
rendering its result adds no warning.

Survival-difference reports cover log-rank, G-rho, stratified and one-sample
expected-survival tests. Stratified observed and expected events sum across
strata for display. Group contributions with zero denominators retain R's
`NaN`/infinity semantics. One-sample tests show observed events, expected
events, the signed z statistic and the p-value. Omission records preserve the
rows and omission policy used by both formula and direct-response inputs.

## Reference checks

`scripts/generate_diagnostic_report_reference.R` generates 46 cases from
R 4.5.3 and survival 3.8-12. The fixture covers all five case-cohort estimators,
robust and single-coefficient fits, ordinary and spline diagnostics, conditional
logistic regression, scalar/multiple/stratified/weighted concordance, omitted
rows, unavailable variances, no comparable pairs, grouped/stratified/G-rho and
one-sample survival tests, zero expected events, precision changes, significance
stars and narrow layouts.

For the multiple-predictor concordance report without variance, the reference
removes the variance from an existing fitted result: R 3.8-12's
`concordancefit(..., std.err=FALSE)` fails for multiple predictors. Python's
existing fitter supports that input directly. Tests compare native reports
with R text and numerical values, and independently render the R snapshots.
They also check ownership, serialization, data-frame labels, missing-row
metadata and invalid options.
