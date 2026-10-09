# Changelog

## Unreleased

- Validate mutable public Rust inputs at Kaplan-Meier, Aalen-Johansen, and
  log-rank fitting boundaries, returning input errors before accessing rows.
  Release the Python GIL during convenience and one-sample log-rank tests.
- Follow R's factor comparison methods in formulas, including declared ordered
  levels, incompatible-level errors, and unordered-factor warnings. Preserve
  ordered metadata in pandas inputs, row subsets, and the R bridge.
  Fit ordered covariates with polynomial contrasts, retain explicit bases and
  fitted prediction coding, and apply standard named contrasts to prepared
  model frames. Preserve contrast attributes through R data preparation and
  factor-valued time-transform callbacks.
- Reuse native subject/time ordering for carry-forward initialization, fixing
  missing-time ordering and retaining missing initial factor values. Remove the
  second Python sort; see [carry-forward validation and timings](docs/carry-forward.md).
- Compute grouped curve medians by selecting the middle values, with a reusable
  contiguous group buffer. Check mutable grouping codes and preserve observed
  group ordering when the product of declared factor levels overflows.
  See [aggregate curve validation and timings](docs/aggregate-survival-curves.md).
- Accept three-dimensional NumPy state probabilities directly for curve
  aggregation, including strided views, with an owned copy before releasing
  the GIL. Avoid intermediate nested lists in multistate curve aggregation.
  Align NumPy storage before constructing Rust views at typed input boundaries.
  Normalize NumPy boolean bytes without borrowing them as Rust `bool` values.
- Keep constant AFT covariate columns unscaled so the existing rank-aware
  solver can identify aliases and return finite fitted values. Validate
  identifiable results against stock R fits; see
  [rank-deficient AFT designs](docs/aft-rank-deficient-designs.md).
- Use R's rank-limited polynomial contrast construction when high powers
  lose numerical rank, with frozen base-R references through 24 levels and
  documented high-degree platform sensitivity.
- Match stock Cox terms prediction attributes across centering references,
  missing-row restoration, grouped predictions, and sparse frailty paths.

- Evaluate restricted formula expressions with R precedence and three-valued
  logical operators. Preserve nested logical and factor identities, logical
  offsets, fitted contrasts, and empty or missing prediction types. Check
  expressions and full Cox/AFT fits against independent stock-R references;
  preserve recovered nullable logical sources through the R bridge.
  Preserve integer arithmetic and overflow missingness, and decode supported
  R string escapes with canonical term labels.
- Compute exact Cox tie moments from the smaller of the death subset and its
  complement, reducing work and memory for nearly complete risk-set ties.
  Preserve tiny covariance contributions when log risks differ sharply, with
  exhaustive subset checks and an independent R fitting reference. Revalidate
  mutable Rust inputs at the counting-process exact fitter boundary.
  Sweep untied counting-process risk sets with growing accumulators or stable
  blocked moment trees, removing repeated full-stratum scans. Preserve moderate
  means across extreme covariate contrasts and dominant risk removal.
- Retain survival-curve missing-row metadata and restore excluded observation
  rows in residual and pseudo-value arrays. Preserve original observation labels
  after omission, with 96 stock-R differential cases across KM and multistate fits.
- Make benchmark execution and malformed coverage reports fail CI, compare
  measured Divan cases between PR and base revisions, and test CI helpers without
  building the extension. Run external Rust API integration tests in CI and
  remove a stale Cox-dispatch test skip. Test numerical results from installed
  wheels on macOS and Windows and regenerate current-stock references in CI.
  Reject logical-to-numeric reference changes and check the built R source
  archive with no package warnings or notes.

- Prepare grouped survival residuals and pseudo-values with one pass over input
  rows, and check curve endpoints with cumulative offsets. Preserve weighted,
  clustered and counting-process results while removing repeated scans per stratum.

- Compute independent competing-risk Aalen–Johansen uncertainty from small
  influence moment factors, avoiding subject scans at every reporting time.
  Reuse the earliest entry time across grouped initial-state estimates.
  Grouped multistate tables assemble state columns by position, preserving
  repeated state selections and avoiding repeated label scans.

- Select requested population-survival times with one forward scan, preserving
  duplicates, groups and follow-up breakpoints without quadratic searches.
  Expected cumulative hazards preserve undefined survival probabilities as NaN.

- Reuse stratum data, event-time weights and time ordering across concordance
  predictors, retaining independent rank trees and joint covariance. Prevent
  zero-weight events with an empty weighted risk set from producing a `NaN`
  Cox variance numerator.
- Classify plain numeric formula arrays from their dtype, avoiding repeated
  Python scalar scans while preserving logical and categorical contrasts.
- Size interval-splitting outputs with binary searches over sorted cutpoints,
  removing a scan of every cutpoint for every input interval.
- Add a runnable Rust Kaplan-Meier and Cox example alongside the Python examples.

- Preserve sparse and dense frailty penalties returned by Cox time transforms,
  including controller histories, summaries and saved models. Python callbacks
  accept `CoxPenaltyBasis` and direct `pspline()` results; stored model matrices
  retain sparse group columns without expanding the native fitting matrix.

- Add person-years, survival-data consistency and marginal-means reports.
  Retain built-in rate-table origins and matched-population summaries, and
  preserve categorical status metadata in `factor()`/`as.factor()` survival
  responses. Person-years totals traverse grouped cells without flattening them.

- Add case-cohort, conditional-logistic, proportional-hazards diagnostic,
  concordance and survival-difference reports with full-precision tables.
  Retain formula omission records and case-cohort stratum labels, and avoid
  repeated covariance copies in case-cohort summaries.

- Add AFT and Aalen model/summary reports, including scales, coefficient
  correlations, missing-row notices and full-precision tables. Preserve robust
  Aalen covariance when changing summary cutoffs or test weights, with bounded
  influence conversion memory. Add width and data-frame support to penalized
  AFT reports.

- Add Cox model and summary reports for ordinary, robust, null, penalized and
  multistate fits, with full-precision tables and statistics, R-style coefficient
  formatting, confidence intervals, term tests and transition grouping.

- Add detailed survival-summary and expected-survival reports with grouped
  tables, confidence columns, state probabilities and full-precision data-frame
  conversion. Preserve conditional cutoff units when formatting scaled summaries.
- Add compact survival reports for Kaplan–Meier, Turnbull, Cox and multistate
  curves, including counts, medians, restricted means and R-style text formatting.
  Reports use the Rust summary kernels without expanding event-time arrays and
  expose full-precision columns through `as_data_frame`.
- Add raw-response plots, expected-survival overlays, and model-specific
  `plot`, `lines`, and `points` dispatch in `survival.plotting`. Expected curves
  preserve group names and R's straight-line overlay default without inferring
  confidence bands or censor counts.
- Add Aalen cumulative-coefficient plots and overlays, with tied-event handling,
  ordinary or influence-based confidence bands, and bounded temporary memory
  when reducing stored influences. Numerical plot data works without Matplotlib.
- Add Cox proportional-hazards diagnostic plots with natural-spline curves,
  two-standard-error bands, scaled residuals, and hazard-ratio views. The Rust
  smoother shares factorizations across terms and releases the Python GIL.
- Add optional Matplotlib survival graphics for Kaplan–Meier, Turnbull, Cox,
  and multistate curves, including confidence bands, censor marks, transformations,
  and overlays. Numerical plot data is available without a renderer, and constant
  runs are compressed before drawing. See [survival graphics](docs/survival-plotting.md).
- Add R-style `survfit.matrix` through `survfit` on a square matrix of KM or
  Cox curves, with discrete and matrix-exponential updates, stratified curves,
  prediction columns, starting distributions and conditional start times.
  The Rust kernel scans the transition curves once and releases the Python
  GIL; its results support the existing multistate summaries and curve methods.
- Preserve tiny transition probabilities in the matrix exponential's closed
  forms using `expm1`, including the shared multistate Cox prediction path.
- Add `CoxPHFit::predict_survival_at` and its Python method to evaluate only
  requested times, returning a time-by-observation matrix. Cox estimator
  survival predictions and R-style Brier scores use this path to avoid full
  curve expansion and Python list conversions.
- Cache censoring-survival lookups in Brier scoring, removing binary searches
  from the subject-by-time loop. Its Python binding accepts NumPy arrays and
  matrices through the shared checked input types.

## 2.0.0

2.0 turns `survival.r` into a faithful port of R's survival 3.8: the formula
functions follow R's defaults, argument rules, result shapes and names, and
the Rust kernels are ports of the corresponding R and C routines. Many results
change to R's values as a consequence. Where R itself is wrong, the port
returns the intended result; those cases, and every other known difference,
are listed in [R compatibility](docs/r-compatibility.md).

### Added

- Multi-state Cox models: `coxph` on a factor-status response with `id=`,
  formula lists, `istate=` and `statedata=` returns a `CoxphmsModel` (with
  `CoxphmsShare`), with `coef`/`vcov(matrix=True)`, `predict`, `residuals`,
  `cox_zph`, `coxph_detail`, `survfit` (R's `survfit.coxphms`, with `p0`),
  `summary_survfit`, `survfit0` and `aggregate_survfit`.
- Penalized `survreg`: `ridge()` and `pspline()` terms are fitted through a
  port of R's `survpenal.fit` (`SurvpenalFit`, with `inner_failures`, `n_eff`
  and `frail_index`), with `print_survreg_penal`, `summary`, penalized `anova`
  refits and fractional `logLik` df. `SurvregControl` takes `outer_max`.
- Penalized Cox fits get R's `summary.coxph.penal`, fractional-df `anova`,
  frailty terms in `predict(type="terms")` and `survfit(id=)` curves;
  `cox_zph` accepts penalized fits.
- `survfit.coxph` follows R's `type=`, `start.time`, `individual=`, newdata and
  curve names; Turnbull and Cox curves support `survfit0`, `summary` and
  `quantile`; residuals and pseudo values work on `start.time` fits.
- `yates` supports R's default fits, aliased coefficients, estimability and
  `predict="survival"`; `summary_pyears` is reachable from `survival.r`.
- Timeline data: `fromtimeline` with R's formula interface, and `Surv2`
  responses in `coxph`, `survfit`, `survcheck` and `survSplit`.
- The formula language gains R's `cut()`, `tcut()` arguments, literal vectors,
  comparisons, `%in%` and rmap expressions.
- `survival.r` exports the result classes and the R helpers that were
  missing, including `cluster`, and is typed inline.
- The core result classes pickle, copy and deepcopy.
- The PyO3 bindings read NumPy arrays in one copy and release the GIL in the
  heavy kernels; survfit influence matrices are shared, read-only NumPy views.
- `rsurvreg(seed=s)` reproduces R's `set.seed(s)` stream.

### Breaking changes

#### Removed and renamed

- `survival.regression.aareg`, `AaregOptions`, `AaregResult`,
  `AaregConfidenceInterval`, `AaregFitDetails` and `AaregDiagnostics` are
  removed; use `survival.aareg(formula, data, ...)` or
  `survival.regression.aareg_fit(AaregData, AaregOptions)`.
- The positional `survival.regression.survreg` / `survival._survival.survreg`
  is removed; use `survival.regression.survreg_fit(SurvregData(...),
  SurvregDistribution(name), control=SurvregControl(...))` or
  `survival.survreg(formula, data)`.
- The legacy Cox-baseline builders are removed from `survival.surv_analysis`
  and `survival._survival`: `agsurv4`, `agsurv5`,
  `compute_baseline_survival_steps`, `compute_tied_baseline_summaries`,
  `cox_expected_baseline_by_stratum`, `cox_survfit_from_baseline`,
  `condition_cox_survfit_curves`, `step_matrix_values_at`, `basehaz` (use
  `survival.basehaz`), `survfit_from_hazard`, `survfit_from_cumhaz`,
  `survfit_from_matrix`, `survfit_multistate` and `SurvfitMatrixResult`.
- `survival.surv_analysis.norisk` (and the Rust `norisk_flags`) is removed.
- `survival.Surv2data`, `survival.r.Surv2data` and `survival.r.Surv2Data` are
  removed; use `fromtimeline` or `survival.data_prep.surv2counting`.
  `fromtimeline(time, status, *, id, states, repeated)` is replaced by R's
  `fromtimeline(formula, data, subset, id, repeated, lvcf, yname)`, which
  returns a column mapping.
- Bridge-only helpers are private: `survival.r` / `survival` no longer export
  `aggregate_survfit_result`, `survfitkm_influence`,
  `survfitkm_counting_influence` or `survexp_individual`; use
  `aggregate_survfit(x, by=)`, `survfit(..., influence=)` and
  `survexp(..., cohort=False)`.
- `python/survival/r_api.pyi` and `python/survival/r/__init__.pyi` are removed;
  type checkers read the inline annotations. `r.survfit`, `r.survfit0` and
  `r.survcheck` are typed as unions.
- `survival.r.SurvfitInfluence` is now `SurvfitInfluenceMatrix`;
  `_types.TurnbullSurvfitResult` is removed (use `SurvfitResult`).
- `survcheck()` takes a formula only (the `time1`/`time2`/`status` keywords
  are gone).
- The `cuda` Cargo feature and fourteen unused `survival::constants` items are
  removed.

#### Data and missing values

- `datasets.load_*()` no longer include `_nrow`/`_ncol`; use
  `len(data["time"])` and `list(data)`. Data columns whose names start with
  `_` are no longer dropped.
- `coxph`, `clogit`, `survreg`, `concordance`, `survConcordance`, `aareg`,
  `model_frame` and `rttright` default to `na_action="na.omit"` (was `"fail"`,
  or na.pass for `model_frame`/`rttright`). `"na.exclude"` pads residuals and
  predictions with NaN at the removed rows, and a collapse vector must then
  have the data's length. `ModelFrame.na_action` is a `NaAction` record.
- `predict` on coxph and survreg fits takes `na_action` (default `"na.pass"`)
  and returns NaN for incomplete newdata rows instead of raising;
  `survfit(<coxph>, newdata=)` and `basehaz(newdata=)` leave such rows out and
  raise "all rows of newdata have missing values" when none is left.
- Formula arithmetic follows R: `log`/`sqrt` of negative values and `0/0` are
  NaN (missing, with a "NaNs produced" warning), `x/0` and `log(0)` are ±Inf,
  and a numeric covariate with a missing value stays numeric under na.pass.
  coxph and survreg raise "data contains an infinite predictor".
- Counting-process rows with `start >= stop` are missing in model frames
  (dropped with R's warning under na.omit, an error under na.fail);
  `core.CountingProcessData` rejects `stop == start` with "Stop time must be >
  start time (row i: s >= t)".
- After `subset=` or an na.action, `ModelFrame.data` holds only the formula's
  variables, and a formula variable of the wrong length raises "variable
  lengths differ".
- A factor or Categorical column keeps its unused levels, as R's
  `model.frame` does: coxph, survreg and concordance get an aliased (NaN)
  coefficient for such a level, `yates` an NA row, `aareg` raises the nmin
  error and `cch(method="Prentice")` stops.
- A formula with missing strata values under na.pass raises ValueError
  ("missing values in the strata").

#### Formula interface

- `%in%` interaction coefficients are named in R's variable order; unquoted
  names containing `= < > ! & |` raise ValueError.
- An ordered comparison with a string or factor operand raises.
- `as.numeric(f)` of a factor inside a formula expression is the factor's
  codes; `tcut()` honours `scale=` and positional labels; `survSplit(Surv(time,
  status == 2) ~ .)` names its event column "event".
- rmap strings are evaluated as R code when they read data columns or are
  expressions; a bare word is a label, and a constant-only expression such as
  "60 * 365.25" is a label (pass the number).
- `pyears` `cut()` terms get R's labels and intervals, rows outside the breaks
  are dropped (new `PyearsResult.na_action`), factor levels keep their declared
  order with unused levels as zero cells, and `tcut()`/`cut()` breaks naming a
  column are read from the whole data.
- Python digit separators (`1_000`) are no longer numbers in formulas.
- `model_frame(formula_str, data, ...)` returns a column dict (not a
  `ModelFrame`), `model_frame(coxph fit)` rebuilds the frame, and mappings
  with a `.model` attribute are refused.

#### Cox models

- `coxph()` ignores the individual control keywords when `control=` is given
  (R's rule); the aliases `max_iter`, `tol_chol`, `toler`, `time.fix` and
  `time_fix` are gone, an unknown control entry raises TypeError, and
  `coxph_control()` accepts fractional `iter.max`/`outer.max`, raises
  ValueError for infinite ones and TypeError with R's message for non-numeric
  options. `CoxphModel` has no `no_events` field.
- (start, stop] Breslow/Efron fits follow `agfit4.c`: flag is the rank or 1000
  (never -2), iteration counts change for fits that used to stop mid-halving,
  non-converged right-censored fits report `iter = iter_max + 1`, aliased
  coefficients are NaN, `coxph_fit` raises "exp overflow due to covariates"
  where R stops, and messages say "beta may be infinite". Robust fits emit R's
  convergence warning. `CoxPHFit.info` is new.
- `predict(fit, newdata, type="lp"|"risk")` no longer centres the new offset
  unless `se.fit` is set, the fit is stratified with `reference="strata"`, or
  `reference="zero"` with non-zero means.
- `fitted(coxph_fit)` ignores `type`, `se_fit` and `reference` and is not
  NaN-padded; `model_matrix(fit)["assign"]` counts strata terms and has a
  "strata" entry.
- `residuals(type="schoenfeld"|"scaledsch")` returns a
  `CoxSchoenfeldResiduals` (the old list is its `.values`). Null-model
  residuals other than martingale and deviance raise ValueError.
- Summary dicts drop `n_event`, `score_test` and `robust`; read `nevent`,
  `sctest["test"]` and `used_robust`.
- `survfit`, `basehaz` and `survexp(ratetable=)` refuse a Cox model whose
  interaction lacks a lower-order term, as R does.
- `survfit(<coxph>, type=)` maps R's old-style types (an unknown type raises);
  `stype` and `individual` default to None and `individual` warns; `start.time`
  returns curves; curves of a newdata with `id=` or row names are named by
  them. `cox_survfit_baseline` returns an `AgsurvCurve`.
- `agexact.fit` rejects case weights other than 1.
- `cox.zph`: `CoxZphTest.df` and the table's df are floats; the transform
  defaults to None (the km transform) and may be a callable or values; the
  `penalized` keyword is gone (pass the `CoxpenalFit` as `fit`).
- `survival.validation.anova_coxph`: `AnovaRow.df` is `float | None`, df
  decreases are accepted (p NaN) and non-finite df raise.

#### Penalized models

- Penalized coefficients change to R's whenever `subset=` or the na.action
  drops rows (pspline knots, ridge scaling, the frailty df search and sparse
  default, a dense frailty's columns).
- Dense gaussian/t frailty coefficients are named `gauss:<level>` and
  `t:<level>`; `pspline(intercept=TRUE)` coefficients `ps(x)1..k`.
- `frailty(distribution=)` matching is case sensitive and ambiguous prefixes
  raise; invalid `pspline` `Boundary.knots` raise "Invalid values for
  Boundary.knots"; a penalty option of length one is a scalar.
- Penalized Cox `summary()`: `model_type` "coxph.penal", one row per term with
  keys `name, coef, se, se2, chisq, df, p`, a `print2` entry, `df` per term and
  `iter` as `[outer, inner]`; the coefficient_names, sctest, waldtest, rsq,
  robust and method keys are gone. `anova` uses `sum(fit$df)`. The df of an
  aliased single-column penalized term is NaN (was 0).
- `survreg()` with `ridge()`/`pspline()` returns a fit; `degrees_freedom`,
  `df_residual`, anova Df and summary `chi_df` are floats for penalized fits
  (ints for unpenalized ones); `SurvregFit.df` and `df_residual` are floats;
  `SurvregModelResult.df` is a per-term list and `iter` a pair for penalized
  fits. `survreg_control()` refuses `outer_max < 1`.
- Penalty terms inside interactions are refused.

#### Multi-state Cox models

- `coxph()` on a multi-state response returns a `CoxphmsModel` and needs `id`;
  `coef()` and `vcov()` take `matrix=`; `coxph_control()` rejects an unknown
  `survcheckallow` flag; `NamedMatrix.values` may hold ints;
  `CoxphModel.strata` is `list[str | None]`.
- Under `na_action="pass"` a missing categorical value gives NaN design
  entries instead of an NA level.
- `predict`, `residuals`, `cox_zph`, `coxph_detail`, `fitted`, `model_matrix`,
  `model_term_names` and `model_weights` work on multi-state fits; `anova`
  raises R's "anova not yet available for multistate" and
  `predict_terms_constant` raises ValueError. `survfit` returns curves;
  `survfit_coxph`'s `se_fit` defaults to None.

#### Parametric models (`survreg`)

- `vcov(fit, complete=False)` is R's matrix (scale rows kept, aliased rows
  dropped); `coef_names(complete=True)` names stratum scales
  "Log(scale[<stratum>])"; `model_summary(fit)["loglik"]` is
  `[intercept-only, full]`, and the dict gains `var` and `correlation`.
- `SurvregModelResult` no longer forwards attributes to `.fit`;
  `case_weights`, `x_matrix`, `y_response`, `model_frame` and `score_values`
  are `weights`, `x`, `y`, `model` and `score` (None unless `score=True`);
  several derived fields are removed; `strata_columns` is replaced by
  `strata_terms`.
- Distribution names follow R's `match.arg`: `SurvregDistribution(name)`,
  `AFTEstimator(distribution=)` and `cv_survreg_loglik` are case sensitive,
  accept unique prefixes of the exact names and drop the "extreme_value"
  aliases and whitespace trimming; `dsurvreg`/`psurvreg`/`qsurvreg`/
  `rsurvreg` find the exact name in any case and raise "Distribution not
  found" otherwise. `predict_type`/`residual_type` are matched the same way.
- `rsurvreg(seed=s)` returns R's `set.seed(s)` stream; seeds are R's 32-bit
  integers ([-2^31 + 1, 2^31 - 1]).
- `survreg(<Surv>, x=..., na_action="omit")` is accepted and leaves the design
  as given; survreg raises "data contains an infinite predictor" for a
  non-finite design value, and the typed `SurvregFit.predict` reports newdata
  of the wrong width as "newdata must be n x p, got a x b".

#### Survival curves

- Influence matrices (`SurvfitInfluence.values`, `SurvfitAJInfluence.values`
  and `.i0`) are read-only, column-major NumPy arrays and the classes are
  frozen; `SurvfitResult.influence_surv`/`influence_chaz` hold R's row names
  (cluster or id values, or 1..n) instead of 0-based codes.
- `SurvfitResult` has no `start_time`; `time0` is False for KM and Turnbull
  fits (only `survfit0` results set it).
- A split stratum is a whole unstratified result; `fit[, states]` keeps the
  selected states' influence and drops counts;
  `SurvfitMultiStateResult.transitions` may be None.
- Turnbull fits: `std.err` follows survfitKM's robust rule, `cluster` raises,
  `start.time` keeps interval rows by their right end with `t0 = min(0,
  time)`; `_survival.turnbull` takes `se_fit`/`robust` and
  `quantile_survfit` takes `start_time`.
- `summary_survfit(fit, censored=True)` no longer returns influence matrices;
  `survfitresid` and `pseudo` take a `call_stype` argument; invalid intervals
  are reported as "(row i: s >= t)".

#### Concordance

- `ConcordanceResult.ranks` is a dict of columns (time, rank, timewt,
  casewt); clustered dfbeta rows are in sorted cluster order; several fits
  with different strata are accepted and `influence` selects dfbeta or
  influence.
- `concordance(fit, ...)` raises TypeError for data, weights, subset, strata,
  scores, reverse and na_action; a non-Surv response ignores `timewt`; a
  character response or a factor with more than two levels raises.
- Raw strata vectors order their count rows by sorted level (numeric levels
  named "1", not "1.0"); missing strata values and values outside declared
  categories raise.
- `survConcordance` returns `SurvConcordanceResult` (R's statistics) and
  refuses several predictors; `survConcordance_fit` returns R's tied.time and
  std(c-d), per stratum.
- `survival.r._concordance.concordancefit`'s `strata_levels`/`formula`
  keywords are private.

#### Data preparation and other functions

- `survSplit` puts rows with a missing time in episode 2.
- `tmerge` raises for string arguments that are not columns of `data2`,
  recycles only `tstart`, keeps the values' Python type (bool events, int
  values), types a new tdc variable as R's assignment does, and raises for a
  missing `cumevent` increment at an event.
- `finegray` names counts with R's `make.names`; its non-finite errors name
  the value.
- `AeqSurvResult.changed` is removed.
- `survobrien` returns `I()` terms' variables untransformed and names columns
  with R's `make.names` (`log.z.`, `identity.z.`) and `make.unique`.
- `cch`: Borgan `sc` has one row per id in id order (`CchModelResult.sc_ids`);
  the Prentice fit reports R's point estimate; a missing stratum raises
  ValueError.
- `brier(newdata=)` refuses newdata with missing model values or without the
  strata columns.
- `yates`: `YatesContrast.df` is `int | None`, predict values R rejects raise
  ValueError, a `YatesModel` with other predictions returns linear-predictor
  means silently, and `YatesResult` gains `summary`.
- `population.summary_pyears(pyears, n, event=None, expected=None, dims=[],
  ...)` replaces the old signature; `cipoisson` returns NaN for a missing k;
  rate-table date values in non-date dimensions raise R's error;
  `survexp` with a Cox rate table and a conditional/hakulinen method returns
  NaN after a group empties.
- `as_data_frame(pyears table)` uses the `data.frame = TRUE` layout; the reprs
  of the survfit, cox.zph and survreg results omit bulky fields.
- Result classes (`AnovaRow`, `YatesContrast`, `SurvCheckFlags`,
  `TcutResult`, ...) report `__module__` `survival._survival`.
- An id that is not an int, float or str raises TypeError; status errors no
  longer contain "values must be 0 or 1"; an all-zero matrix is singular in
  column 0 and matrices with rcond ≥ eps factorise.
- Binding inputs accept integral floats for integer arrays; a ragged nested
  list fails with "row i length mismatch" and a 1-D matrix argument is one
  column; `pyears`' `categories_data` defaults to None.

#### Rust crate API

- `aareg_fit(&AaregData, &AaregOptions)`, `agexact_fit(CoxphData, options)`
  (`AgexactData` removed), `CoxpenalData::try_new(CoxphData, terms)`,
  `finegray` returning `SurvivalResult<FineGrayOutput>`, `cox_zph` taking
  `impl Into<ZphFit>`, `validation::yates_simulate`, `BrierInput` by value,
  `regression::CoxResidualType` / `surv_analysis::PseudoResidualType`,
  `SurvfitOptions.start_time`, `agsurv::step_at`'s fourth argument,
  `PystepResult` without `index2`/`weight`, and `SurvregFit.covariates` as
  `Array2<f64>`. `coxsurv_fit`, the public `survreg` and `cch_fit`/
  `cch_borgan_fit` (use `cch`/`cch_borgan`) are gone. serde and bincode are
  regular dependencies.

#### R bridge (`r/survivalr`)

- `Surv2data` is no longer exported; the fitters' `na.action` default is
  `getOption("na.action")`; `survConcordance` returns a survConcordance list;
  survreg `residuals()` defaults to `type = "response"` and Cox `fitted()`
  ignores `type` and `se.fit`; `rsurvreg` uses R's random number stream.
