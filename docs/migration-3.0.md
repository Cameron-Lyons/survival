# Migrating to survival 3.0

Version 3.0 makes the breaking clean-ups collected in the 3.0 plan: one R API at
the package root and in `survival.r`, the raw native kernels in
`survival.kernels`, R's names and meanings for result components, and a Rust
crate whose public API is its typed functions rather than its Python bindings.
Numerical results are unchanged; only names, result types and module locations
move.

Each section below lists what changed and what to write instead.

## Python

### `survival.r_api` is removed

`survival.r_api` was a second name for `survival.r`. Import `survival.r` (or use
the package root for R's exported functions):

| 2.x | 3.0 |
| --- | --- |
| `from survival import r_api` | `from survival import r` |
| `survival.r_api.coxph(...)` | `survival.r.coxph(...)` |

The private helpers the R bridge used through `r_api` are attributes of
`survival.pybridge`. `survivalr` 0.2.0 requires survival 3.0 and checks the
Python package version when it first loads it.

### Native kernels move to `survival.kernels`

`survival.kernels` holds every core kernel. A kernel that shares a name with an
R function exists only there, so the R function of that name is unambiguous:

| 2.x | 3.0 |
| --- | --- |
| `survival.surv_analysis.survdiff(time, status, group)` | `survival.kernels.survdiff(...)` (or `survival.survdiff(formula, data)` for R's) |
| `survival.regression.coxph_fit(...)`, `survreg_fit`, `survpenal_fit`, `cox_zph`, `coxph_detail`, `coxph_wtest`, `finegray`, `dsurvreg`/`psurvreg`/`qsurvreg`/`rsurvreg` | `survival.kernels.<name>` |
| `survival.population.pyears`, `survexp`, `summary_pyears`, `summary_survexp`, `match_ratetable`, `is_ratetable`, `survexp_us`/`usr`/`mn`, `PyearsResult`, `PyearsSummary`, `SurvExpResult` | `survival.kernels.<name>` |
| `survival.surv_analysis.aggregate_survfit`, `coxsurv_fit`, `pseudo`, `statefig`, `survfit0`, `survfit_confint`, `summary_survfit`, `quantile_survfit`, `SurvDiffResult`, `SurvfitKMResult`, `StateFigResult` | `survival.kernels.<name>` |
| `survival.validation.brier`, `cipoisson`, `royston`, `survcheck`, `survobrien`, `yates` and their result classes | `survival.kernels.<name>` |
| `survival.data_prep.cluster`, `strata`, `lvcf`, `neardate`, `nostutter`, `rttright`, `survcondense`, `tcut`, `totimeline` | `survival.kernels.<name>` |
| `survival.core.concordancefit`, `nsk` | `survival.kernels.<name>` |
| `survival.residuals.agmart`, `SurvregResiduals` | `survival.kernels.<name>` |

The other kernels stay in their domain modules and are also in
`survival.kernels`. `survival.core.CoxRegressionInput` moves to
`survival.regression`, beside the fitters that take it.

### `survival.residuals` is R's `residuals`

`survival.residuals` used to be a module of three kernels. It is now R's
`residuals` generic, as in `survival.residuals(fit, type="deviance")`. Its
`agmart` kernel and the `SurvregResiduals` container are in `survival.kernels`.

### Removed duplicate and internal kernels

| 2.x | 3.0 |
| --- | --- |
| `survival.validation.logrank_test(time, status, group, entry_times=)` | `survival.kernels.survdiff(time, status, group, start=)`; `statistic`/`p_value` are `chisq`/`pvalue` |
| `survival.surv_analysis.nelson_aalen(time, status)` | `survival.surv_analysis.survfitkm(time, status, stype=2, ctype=1)` (`cumhaz`) |
| `survival.core.coxscore2`, `agscore3`, `schoenfeld_residuals_counting`, `CoxMartInput`, `survival.residuals.coxmart`, `survival.validation.lrt_test`, `score_test`, `survival.pybridge.cox_callback` | removed from the public modules; fitted models provide these through their methods (`CoxPHFit.score_residuals()`, `schoenfeld_residuals()`, `martingale_residuals()`) and `survival.r.anova` |

### Non-R helpers

| 2.x | 3.0 |
| --- | --- |
| `survival.totimeline(start, stop, status, states=, id=)`, `survival.r.totimeline` | `survival.kernels.totimeline(id, time1, time2, status, istate)`; `totimeline` is an unexported draft in R |
| `survival.r.Timeline` | removed with `totimeline` |
| `survival.r.predict_terms_constant(fit)` | `survival.predict(fit, type="terms").constant` |
| `survival.survfit_residuals(fit, times=)` | `survival.residuals_survfit(fit, times=)` or `survival.residuals(fit, times=)` |

`predict(coxfit, type="terms")` returns a `TermsPrediction`, a list of rows with
R's `constant` attribute. `constant` is None where R drops the attribute (rows
padded for `na.exclude`, `collapse` sums). Terms predictions of other models
have no `constant`, as in R.

### `pspline` and unknown keywords

`pspline` takes R's `Boundary.knots` as `Boundary_knots` or the dotted keyword
`Boundary.knots`; the `boundary_knots` spelling is removed. `coxph` and `clogit`
raise `TypeError` for an unknown keyword, as the other functions do; the message
is still R's "Argument ... not matched".

### Result class names

Result classes follow one rule: the R function's name with each dotted part
capitalized, then `Result`, or `Model` for a fitted model. `CoxSurvFitResult`
(the `coxsurv.fit` result) and `CoxSurvfitResult` (`survfit.coxph`) differed
only in case; the first is now `CoxsurvRawResult`. The classes are the same, so
only imports, annotations and `isinstance` checks change, and pickles made by
2.x still load.

| 2.x | 3.0 |
| --- | --- |
| `AaregModelResult`, `CchModelResult`, `SurvregModelResult` | `AaregModel`, `CchModel`, `SurvregModel` |
| `YatesModel` (an input adapter, not a fit) | `YatesFitAdapter` |
| `CoxSurvFitResult`, `CoxSurvFitList` | `CoxsurvRawResult`, `CoxsurvRawList` |
| `CoxFitResult` | `CoxphFitResult` |
| `CoxPHDetailResult`, `CoxPHWTestResult`, `CoxZPHResult` | `CoxphDetailResult`, `CoxphWtestResult`, `CoxZphResult` |
| `CoxBaseHazardResult` | `BasehazResult` |
| `SurvDiffResult`, `SurvExpResult`, `SurvCheckResult`, `SurvCheckProblem`, `StateFigResult` | `SurvdiffResult`, `SurvexpResult`, `SurvcheckResult`, `SurvcheckProblem`, `StatefigResult` |
| `SurvExpSummary`, `PyearsSummary`, `RateTableSummary` | `SummarySurvexpResult`, `SummaryPyearsResult`, `SummaryRatetableResult` |
| `SurvfitQuantileResult`, `SurvfitResidualsResult` | `QuantileSurvfitResult`, `ResidualsSurvfitResult` |
| `RateTableMatch`, `RateTablePrint` | `RatetableMatch`, `RatetablePrint` |
| `TMergeFrame`, `TMergeOperation`, `FineGrayFrame` | `TmergeFrame`, `TmergeOperation`, `FinegrayFrame` |

The native kernel classes in `survival.kernels` keep their names.

### Summary results

`summary` (`model_summary`) of a coxph, survreg, aareg or cch fit returns a
frozen result: `SummaryCoxphResult`, `SummaryCoxphPenalResult`,
`SummarySurvregResult`, `SummaryAaregResult` or `SummaryCchResult`. Each one
supports attribute access and read-only `summary["component"]`; R's dotted
spelling (`summary["conf.int"]`) works too. A component that R leaves out is
None as an attribute and missing as a key. The components keep R's names and
meanings, and table rows use R's column names:

| 2.x | 3.0 |
| --- | --- |
| Cox rows `exp_coef`, `se`, `p` (`naive_se`/`robust_se` when robust) | `exp(coef)`, `se(coef)`, `Pr(>\|z\|)`, plus `robust se`; as in R, `se(coef)` of a robust fit is the naive standard error |
| Cox `conf_int` rows `lower`, `upper`, and `conf_level` | `lower .95`, `upper .95`, named for the level as in R |
| Cox `loglik` (final) and `null_loglik` | `loglik`: R's pair (null, final) |
| Cox `df`, `coefficient_names`, `coefficient_columns`, `model_type` | `logtest["df"]`, the rows' `name`, `columns`, `isinstance` |
| Penalized Cox rows `se`, `chisq`, `df` | `se(coef)`, `Chisq`, `DF` |
| survreg `coefficients` (the rows) | `table`; `coefficients` is R's named coefficient vector, as a dict |
| survreg rows `coef`, `se`, `naive_se`, `robust_se` | `Value`, `Std. Error` (robust: `Std. Err`, `(Naive SE)`) |
| survreg `scales`, `scale_names`, `location_coefficients`, `location_coefficient_names` | `scale` (a dict by stratum when there are several) and `coefficients` |
| aareg rows `se`, `robust_se`, and `columns` | `se(coef)`, `robust se` |
| cch rows `coef`, `se`, `z` | `Value`, `SE`, `Z` |

The summaries are read-only; `dict(summary)` gives a plain dict.

### `anova` tables

`anova` of Cox models and of survreg models returns one type, `AnovaResult`. It
replaces `AnovaCoxphResult` and `SurvregAnovaResult`. `columns` holds the table
under R's column names, `row_names` the row names and `heading` R's heading.
`result["Pr(>|Chi|)"]` reads a column, and `frame()` returns the columns.

| 2.x | 3.0 |
| --- | --- |
| `anova(coxfit).rows[i].loglik`, `.chisq`, `.df`, `.p_value`, `.name` | `result.loglik`, `result["Chisq"]`, `result["Df"]`, `result["Pr(>\|Chi\|)"]`, `result.row_names` |
| survreg `table.loglik` (which held -2 log-likelihood) | `result.neg2loglik` (the `-2*LL` column) |
| survreg `table.terms`, `df`, `deviance`, `resid_df`, `p`, `test_labels` | `result.row_names`, `result["Df"]`, `["Deviance"]`, `["Resid. Df"]`, `["Pr(>Chi)"]`, `["Test"]` |
| `as_data_frame(anova(coxfit))["model"]` | `anova(coxfit).row_names` |
| `anova(coxfit, test="none")` raised ValueError | omits the p-values for one model; a list of models has p-values for any `test` but None, as in R |

### Penalized Cox components

On a penalized Cox fit, `pterms`, `assign2` and `history` are dicts keyed by
term, like R's named vectors and lists. `assign2` is new. On a fit without a
penalty, `df`, `var2`, `frail`, `fvar`, `penalty`, `pterms`, `assign2` and
`history` are None (R's `NULL`); in 2.x they raised AttributeError. survreg
fits already behaved this way.

| 2.x | 3.0 |
| --- | --- |
| `fit.pterms == [1, 0]` | `fit.pterms == {"pspline(age, df=3)": 1, "sex": 0}` |
| `fit.history[0]` | `fit.history["pspline(age, df=3)"]` |
| `hasattr(fit, "penalty")` | `fit.penalty is not None` |
| `coxfit.coef_names`, `cchfit.coef_names` | `coefficient_names`, as on every model |

## R bridge (survivalr)

- `cox_zph()` and `coxph_detail()` are removed; call R's `cox.zph()` and
  `coxph.detail()`.
- `totimeline()` is no longer exported, as in R (`survivalr:::totimeline`).
