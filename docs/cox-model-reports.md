# Cox model reports

`survival.r` exposes Cox model and summary reports as structured `ModelPrint`
objects. They use the fitted Rust model and existing summary calculations;
printing never refits a model or expands its observation-level arrays.

```python
from survival import datasets, r

fit = r.coxph("Surv(time, status) ~ age + sex", datasets.load_lung())
report = r.print_coxph(fit)
print(report)
columns = r.as_data_frame(report)

summary = r.model_summary(fit, conf_int=0.90)
detailed = r.print_summary_coxph(summary, signif_stars=False)
print(detailed)
intervals = detailed.tables["conf_int"]
```

`tables["coefficients"]` contains a `NamedMatrix` with full-precision values.
Summary reports also retain `tables["conf_int"]` when intervals were requested.
`statistics` contains sample sizes, likelihoods, tests, concordance and other
available numerical footer fields. `lines` contains the formatted text;
`str(report)` joins it with a final newline. Constructing a report does not
write to stdout, and editing report tables or statistics does not mutate the
fit or summary. `as_data_frame(report)` returns independent coefficient columns
and a `term` column with their labels; a null report returns an empty mapping.

## Ordinary, robust and null fits

`print_coxph(fit, digits=None, signif_stars=False, width=80)` shows coefficients,
hazard ratios, standard errors, Wald z statistics and p-values. Robust fits
include both model-based and robust standard errors. The footer contains the
likelihood-ratio test, sample size, event count and missing-row notice.

`print_summary_coxph(summary, digits=None, signif_stars=True, expand=False,
width=80)` adds fitted confidence intervals, concordance, likelihood-ratio,
Wald and score tests. Clustered fits retain the robust score test and R's
explanation of the independence assumptions used by the different tests.
The summary's `conf_level` preserves the requested interval labels, including
90% and 97.5% intervals. Older summary mappings without this field use 95%.

Both methods normally default to four significant digits. `width` wraps tables
into column blocks. The shared formatter follows R's coefficient/standard-error
precision, separate test-statistic precision, vector p-value formatting,
significance cutoffs and legend. Missing coefficients remain `NaN` in numeric
tables and appear as `NA` in ordinary reports.

A null fit dispatches to `print_coxph_null`, which reports its log likelihood
and sample size. As in R, the null printer ignores the display precision
argument and uses seven digits. `model_summary(null_fit)` returns the fit
itself, so it can be passed directly to `print_summary_coxph`.

## Penalized fits

`print_coxph` dispatches penalized fits automatically. The explicit method
`print_coxph_penal(fit, terms=False, maxlabel=25, digits=None, width=80)` exposes
additional options. It defaults to three digits and reports per-term
coefficients, both standard errors, chi-square tests, effective degrees of
freedom and p-values. Spline terms split into linear and nonlinear rows;
frailty terms report random-effect tests. Sparse and dense frailties, including
fits consisting only of a frailty term, use the existing native summaries.

`terms=True` combines ordinary multicolumn terms into a single Wald-test row.
`maxlabel` truncates displayed coefficient labels while preserving complete
labels in the numeric table. Iteration counts, penalty history and effective
degrees of freedom appear beneath the table.

`print_summary_coxph_penal(summary, digits=None, signif_stars=True,
maxlabel=25, width=80)` defaults to four digits and uses R's character-table
layout, followed by confidence intervals, fitting history and concordance.
R accepts but ignores `signif_stars` for this method. Confidence-interval
labels remain untruncated, matching R's behavior. A penalized summary can also
be passed to `print_summary_coxph` for automatic dispatch.

## Multistate fits

`print_coxph` groups identical coefficient maps under their shared transition
headings, annotates proportional baseline coefficients, and shows state names
and the number of subjects. As in R, the ordinary `se(coef)` column is omitted
from this display; robust standard errors are retained.

`print_summary_coxph` normally shows the complete coefficient table. With
`expand=True`, it groups coefficients and confidence intervals by transition,
disables significance marks, and lists state names. Numeric report tables keep
the original complete rows without duplicating coefficients shared between
transitions.

## Compatibility and validation

The reports omit original R call expressions because the Python fit does not
store them. They return data objects instead of R's invisible return value and
remove trailing spaces from text lines. Options are explicit arguments; no
global R print options or locale settings are read. Coefficient labels retain
the spelling stored by the Python formula interface, so different whitespace
inside a formula expression can produce different labels than R's deparser.

`scripts/generate_cox_report_reference.R` regenerates the committed references
with R 4.5.3 and survival 3.8-12: 55 model reports and 82 matrix-format cases.
The 172 focused tests compare native fitted models and R summary
snapshots independently. The cases cover ordinary, robust, stratified, aliased,
missing-data, null, ridge, spline, sparse/dense frailty and multistate fits;
scaled and omitted intervals; grouped term tests; significance marks;
proportional baselines; long transition labels; and narrow column layouts.
Separate matrix cases exercise tiny p-values, missing estimates, precision
changes and exact wrapping boundaries.
