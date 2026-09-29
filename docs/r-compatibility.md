# R survival compatibility

The Python formula interface is `survival.r`; its numerical routines run in
Rust and are also exposed through the Rust crate's domain modules. The checked-in
reference fixtures were generated with R survival **3.8.11**. They cover model
fits, covariance matrices, predictions, residuals, survival curves and utilities.
They establish compatibility for the tested inputs, rather than a claim that
every R expression, plotting method or extension package is supported.
Regression tests added since then hard-code values from R survival 3.8-12
(R 4.5.3); the few places where 3.8-12 changed behaviour are listed under
[Versions](#versions).

This page records every known difference from R:

- [Reference differences retained deliberately](#reference-differences-retained-deliberately):
  the fixture checks that are expected to fail;
- [Deliberate fixes of R defects](#deliberate-fixes-of-r-defects-no-fixture):
  inputs where R errors, crashes or returns wrong numbers and the port returns
  the intended result;
- [Other deliberate differences](#other-deliberate-differences): versions,
  numerics, messages, labels, sort orders and Python-specific representations;
- [Not yet implemented](#not-yet-implemented): R entry points without a port.

## Formula and population interfaces

Penalized Cox formulas support `ridge`, `pspline`, `frailty`, `frailty.gamma`,
`frailty.gaussian` and `frailty.t`; penalized `survreg` formulas support
`ridge` and `pspline` (R's `survreg` refuses frailty terms, and so does the
port). Basis construction and penalty optimization run in Rust. Prediction
reuses training knots and factor levels. Formula options are parsed as
literals; Python or R code is never evaluated.

Custom `survreg` distribution dictionaries can supply density, initialization,
quantile, deviance, variance, and response-transform callbacks. Ordinary and
penalized fitting, prediction, residuals, distribution functions, and Python
pickle retain these callbacks. Native Rust callers implement
`regression::SurvregCallbacks`. See [AFT distribution callbacks](survreg-density.md)
for the batch contract, examples, and serialization requirements.

Cox and AFT formulas support stratum-specific effects such as
`age * strata(sex)`, `age:strata(sex)`, and interactions with compound strata
or categorical covariates. The strata still determine the Cox baseline hazards
or AFT scales. Cox constructs the full model matrix before dropping the strata
main effects; AFT removes those main effects before assigning contrasts. The
port preserves this difference, including factor order and column labels.
Cox survival curves for these models require `newdata`, as in R.

`scripts/generate_strata_interaction_reference.R` checks coefficients,
covariances, design matrices, predictions, residuals, curves, and proportional
hazards diagnostics. Its cases include delayed entry, Breslow ties, case
weights, combined strata, and AFT scale strata. R's `predict.coxph` can remove
the wrong columns when a strata interaction follows a multi-column factor:
linear prediction errors and term prediction can silently use different
columns. The fixture retains those R results and separately calculates the
intended predictions from R's correct `model.matrix.coxph` output; the port
uses that model matrix consistently.

R's curve prediction fails for the same factor models, so the reference
generator verifies their curves against equivalent separate fits within each
stratum. R's proportional-hazards test also fails for redundant full factor
indicators; the port's global test agrees with the equivalent treatment-coded
model. The fixture records these R errors alongside the working references.

`model_matrix(survreg_fit, data)` builds prediction rows with the fitted
contrasts and drops incomplete rows. R's method can fail while rebuilding
the terms of a model with strata interactions. Yates factorial and SAS
populations retain compound strata as evaluated factors; their labels are
not parsed back into the original source columns.

Formula expansion also handles `offset()` and `cluster()` inside products,
nesting, and powers. Each distinct offset contributes once even if it appears
in a removed term, matching R's `terms()` behavior. Cluster main terms supply
robust-variance groups; their interaction columns remain in the design. R's
checks for multiple clusters and missing interaction margins are preserved.

Variables removed by formula subtraction remain in the model frame and take
part in training-row omission. For new-data predictions, `na.pass` allows a
missing unused variable without losing the prediction; `na.omit`,
`na.exclude`, and `na.fail` still check it. Cox and AFT cluster extraction
rebuilds the formula and drops such unused variables, as R does.

`scripts/generate_formula_special_reference.R` checks these cases against R,
including numeric and categorical clusters and transformed missing values.
It retains R's raw offset predictions alongside the intended values: the port
includes AFT new-data offsets and consistently centers Cox offsets even for
models without coefficients. For models without covariate terms, term
prediction returns an empty-column matrix where R's method errors.

```python
from survival import datasets, r

lung = datasets.load_lung()
fit = r.coxph("Surv(time, status) ~ pspline(age, df=4) + sex", lung)
prediction = r.predict(fit, {"age": [50, 70], "sex": [1, 2]}, type="lp")
effective_df = r.degrees_freedom(fit)

cox = r.coxph("Surv(time, status) ~ age + sex", lung, model=True)
expected = r.survexp("~ sex", lung, ratetable=cox, times=[0, 100, 365])
```

Multi-state Cox models (`coxph` on a factor-status `Surv` with `id=`, formula
lists, `istate=` and `statedata=`) return a `CoxphmsModel` with R's `cmap`,
`smap`, `states` and `transitions`, and support `coef`/`vcov(matrix=True)`,
`predict`, `residuals`, `cox_zph`, `coxph_detail` and `survfit` (R's
`survfit.coxphms`, with `p0`, `summary_survfit`, `survfit0` and
`aggregate_survfit`).

Multistate `summary_survfit` returns probabilities, state-specific counts,
confidence limits and a table of restricted mean time in each state. Requested
times are sorted and deduplicated; events and censors accumulate between them.
`scale`, `extend`, `censored` and the common, individual, numeric and omitted
restricted-mean options are supported. Rust callers can use
`surv_analysis::{summary_survfit_aj, survmean_aj}`.

`survfit` also accepts a square matrix of ordinary KM or Cox curves, with
`None` for missing transitions (`survfit.matrix`). It supports both `discrete`
and `matexp`, vector or per-curve `p0`, grouped curves, multiple Cox prediction
columns, and `start_time`. The default is `discrete`, matching R's actual
dispatch for both kinds of curve. The computation is available directly to
Rust callers as `surv_analysis::survfit_matrix`. See
[transition-curve matrices](survfit-matrix.md) for the representation and
the additional R 3.8-12 differential fixtures.

`blogit`, `bprobit`, `bcloglog` and `blog` accept `inverse=True` for the inverse
link, with scalar or vector input.

`yates(..., predict="risk", nsim=200, options={"seed": 123})` uses Rust simulation
with R's default Mersenne-Twister/inversion normal stream. An explicit seed gives
reproducible covariance estimates without altering a global RNG. The default
seed is zero. External linear models can supply their coefficients and covariance
through `r.YatesModel(formula, data, coefficients, variance, sigma2=None)`;
coefficients must follow the formula's columns, including its intercept. R's
`lm` is part of the separate `stats` package; the adapter does not refit it.

## Reference differences retained deliberately

The Python fixture suite (`python/tests/test_r_fixtures.py`) has **37 expected
numerical differences**, with no missing-feature or API-error exemptions. None
of the reference JSON files were changed. The differences are:

| Checks | Fixture family | Reason |
| ---: | --- | --- |
| 7 | Aalen regression, veteran data | The reference continues through rank-deficient risk sets and contains missing coefficients and values around 10¹⁵. The port truncates at the last nonsingular risk set and returns finite estimates. |
| 6 | Cox concordance | Exact linear-predictor ties depend on floating-point/BLAS rounding. The reference and this platform classify a few tied pairs differently. |
| 1 | Exact counting-process Cox deviance residuals | The reference's unclassed fit dispatch returns martingale residuals for a deviance request. The port computes deviance residuals. |
| 18 | Interval-censored AFT residual derivatives | The reference has opposite signs in some scale derivatives. The port's likelihood derivatives are checked against finite differences for every censoring type and distribution family. |
| 4 | AFT prediction with a new-data offset | R's `predict.survreg` sets the new-data offset to 0 when the model has one scale (`predict.survreg.R:82`) and keeps it when there are several. The port includes it in the linear predictor and the transformed predictions in both cases. |
| 1 | `survcondense` with no rows to merge | The reference produces zero rows through negative indexing of an empty index. The port preserves all input rows. |

These cases remain explicit `xfail(strict=True)` entries (`KNOWN_FAILURES`), so
a future change in behavior requires review. They are not counted as passing
comparisons. The Rust fixture adapter also records cases it cannot construct;
those harness limitations are separate from the Python tests of the same Rust
engines.

Two fixture transport details are handled by the adapters: CSV factor columns
recover the recorded R levels, and the near-tie test recovers the generator's
IEEE floating-point expressions before fitting. Output time comparisons allow
only the JSON format's 15-significant-digit rounding. Counts and the number of
distinct times remain checked separately.

## Deliberate fixes of R defects (no fixture)

In each case below R errors, crashes, reads uninitialised memory or returns a
result that contradicts its own documentation or code comments, and the port
returns what R intends. No fixture covers these inputs, since R's output there
is not a usable reference; regression tests pin the port's values, usually
next to R's own output.

### Cox models

- **Non-converged variance** (`coxfit6.c`, `coxexact.c`): when iterations run
  out with `iter.max > 1`, R hands the recomputed information matrix to `chinv2`
  without `cholesky2`, so its variance is not the inverse information. The port
  factors it. Lung `age + sex + ph.ecog`, `iter.max = 2`: R's `var[1, 1]` is
  8.330972e-05 (Efron), the port's 8.588522e-05.
- **agfit4 centre moves**: a centre move of more than 709 raises "exp overflow
  due to covariates" whenever the running sums are positive. R tests only
  `denom` and carries on with the deaths already added: two tied deaths alone
  at risk with offsets 0 and 1419 give R a log likelihood of -709.5 (the true
  value is -1419), where the port raises; with offsets 0 and 1418 R returns
  -709 and the port -1418.
- **coxph.detail**: a stratum whose largest linear predictor lies beyond ±200
  has its risk scores shifted by it; R computes `exp(lp)` and overflows. The
  outputs agree wherever R's are finite.
- **predict(type = "expected" | "survival") with new-data offsets**: the port
  uses `chaz * exp(x beta + offset)`; R's `predict.coxph` computes
  `chaz * (exp(x beta) + offset)` (`newrisk <- exp(newx %*% coef) + newoffset`).
  Lung `age + offset(sex)`, newdata age = (60, 70), sex = (0, 2), time = 300:
  port 0.12554, 1.15698; R 0.12554, 0.42161.
- **predict at an extreme offset**: `predict(type = "expected" | "survival",
  se.fit = TRUE)` at an offset whose `exp()` overflows or underflows (ovarian
  with offsets 707 / -740) returns finite values; R stops ("NA/NaN/Inf in
  foreign function call") or returns NaN standard errors.
- **cox.zph with a ridge penalty**: a diagonal `coxlist2$second` is used as
  `diag(second)`. R's `matrix(second, nvar)` recycles it into
  `pmat[i, j] = second[i]`: lung `~ ridge(age, sex, theta = 1)` gives 2.514 on
  1.986 df here and 3.548 in R (2.514 with R patched to `diag()`).
- **cox.zph with a sparse frailty that is not the last term**: R stops with
  "subscript out of bounds"; each term gets its own df here.
- **Collapsed residuals of an na.exclude fit**: `residuals(collapse = TRUE)` and
  `predict(collapse =)` sum the fit's rows; R fails in `rowsum` on vectors of
  different lengths.
- **Stratified scaled Schoenfeld residuals** with several covariates keep
  their strata attribute; R loses it as a side effect of `%*%`.

### Penalized Cox models (`coxpenal.fit`)

- **Hazard between strata**: `coxfit5_c` never resets the hazard at a stratum
  boundary, so R's martingale residuals of a stratified penalized fit
  (`pspline(age, df = 3) + strata(sex)`) are wrong after the first stratum.
  The coefficients agree; the port computes each stratum's own residuals.
- **Efron weights in agfit5b**: `agfit5.c:475` adds `risk * weights[p]` to the
  Efron sum, weighting the tied deaths twice. Weighted `(start, stop]` pspline
  fit: R 0.325129, port 0.325615.
- **Frailty diagonal in coxfit5**: `coxfit5.c:403` leaves the frailty diagonal
  of the information unweighted. With case weights, `age + frailty(inst)` gives
  0.0179368 in R's sparse fit; the port gives R's dense fit, 0.0179446, for
  both.
- **nocenter with a sparse frailty first**: `coxpenal.fit.R:290` computes the
  0/1 column flags on `x` but applies them to `xx`, so R's means and
  `survfit(fit)` depend on the order of the terms (`frailty(inst) + age +
  female`: means (62.46, 0.4038), `survfit(fit)$surv[10]` 0.94156; with the
  frailty last: (62.46, 0) and 0.92841). The port gives R's frailty-last result
  for every order.
- **predict(type = "terms") with a sparse frailty** puts the frailty column at
  its own place for any term order (R returns a zero column after a frailty in
  the middle), and supports `terms =` and `collapse` with a frailty (R errors).

### Multi-state Cox models (`coxph` on multi-state data, `survfit.coxphms`)

Fit (`coxph.R`, `stacker.R`, `parsecovar.R`):

- A formula list's first-level missing-value drop also removes rows with a
  missing `istate` or cluster; R computes those conditions and discards them,
  then `survcheck2` fails with "missing value where TRUE/FALSE needed".
- After that drop the cluster and id codes are subset with the other vectors;
  R extracts them before the drop, so its robust variance uses misaligned
  clusters.
- `smap`'s strata rows are chosen by the strata terms' positions among the
  model terms; R indexes by the special's position among the variables, which
  fails or picks the wrong row when an `offset()` or a variable without its own
  term comes before `strata()`.
- With strata terms and shared baselines, a block is stratified by the strata
  of its own first transition; R reads `smap` column `b` for block `b`, another
  transition when blocks and transitions do not line up.
- A `cluster()` term in the first formula of a list becomes the cluster and the
  rest of the list is kept; R drops every covariate line. In a later line it
  raises "cluster() terms are only allowed in the first formula of a list"; R
  gives `cluster(x)` a coefficient.
- A shared baseline whose reference transition is not observed raises "a shared
  baseline hazard has no observed reference transition"; R fails with "missing
  value where TRUE/FALSE needed".
- No events: "a multi-state coxph model needs at least one event" (R crashes in
  `parsecovar2`). No covariates: "a multi-state coxph model needs at least one
  covariate" (R: "incorrect number of dimensions" or "invalid 'nrow' value").
- `n`, `n.id` and the stored frame are recomputed after every drop; R keeps the
  pre-drop values after a first-level drop alone, and `survfit` then fails.
- `survcheckallow` must name survcheck flags (overlap, gap, jump, teleport); R
  silently turns every check off when no name matches. An empty value keeps
  them all.
- `vcov(matrix = TRUE)` fills each transition's block as R's documentation
  describes (a dict of `NamedMatrix` keyed by transition; the R bridge returns
  R's 3-D array); R 3.8's own array recycles an index.
- `concordance(fit)` and `brier(fit)` refuse a multi-state fit with explicit
  messages; R fails with unrelated errors.

Methods (`predict.coxphms`, `residuals.coxphms`, `cox.zph`, `coxph.detail`):

- `predict` drops the `ph()` rows of `coef(fit, matrix = TRUE)`, which scale
  baseline hazards; R stops with "non-conformable arguments" for every model
  with proportional baselines.
- Martingale residual columns follow the transition of each stacked row, one
  column per transition. R places them by baseline block into the
  `colSums(cmap) > 0` columns, so with a shared baseline one column holds every
  transition and the other is 0, and a covariate-less transition stacked
  through a common baseline overwrites its partner.
- Weighted uncollapsed martingale residuals are `weights * rr` (padded under
  na.exclude); R stops with "argument is of length zero" when the fit has no
  `na.action`. The collapsed or weighted matrix has the same columns as the
  plain one; R allocates every `cmap` column and fills only the first blocks.
- Weighted score, dfbeta and dfbetas residuals multiply each stacked row by its
  own source row's weight. R indexes the stacked weights again by data row
  (`residuals.coxphms.R:238`); on myeloid with weight 2 for `id %% 3 == 0`, the
  crossprod of R's collapsed dfbeta misses `var` by 4.8e-3.
- Weighted Schoenfeld residuals use each event's own weight; R's
  `weights[deaths]` (`residuals.coxphms.R:164`) applies a sorted-order mask to
  stacked-order weights. The Schoenfeld "transition" names each event's own
  transition; R labels by the block's first transition.
- Data rows in no stacked block get zero score/dfbeta rows (R fails setting row
  names); a one-coefficient model gives one-column matrices (R: "object 'rr'
  not found", "'dimnames' applied to non-array").
- A collapse vector with missing values gives a last group labelled "NA"; a
  factor collapse vector with unused levels labels only the levels present (R
  fails setting dimnames in both cases). Groups follow first appearance, also
  for a factor id or cluster.
- `anova` raises R's intended "anova not yet available for multistate"; R
  reaches only "$ operator is invalid for atomic vectors" or "argument must be
  the fit of a multistate hazard  model".
- `cox.zph` with `ph()` coefficients works with `terms = TRUE` (R: "length of
  'dimnames' [2] not equal to array extent"); `cox.zph` and `coxph.detail` work
  with `strata()` terms (R: "subscript out of bounds", from
  `coxph.getdata.R:84-86`). The stacked strata are the strings "1", "2", ...
  where R returns integers.

Curves (`survfit.coxphms`, `coxsurv1-4.c`):

- Transitions are counted one at a time. `coxsurv2.c` walks rows in sorted order
  but reads their transition in data order, so a shared baseline of
  non-contiguous transitions (`smap` 1 2 1) gets hazard 0 everywhere in R.
- A transition with no rows after `start.time` gets hazard 0; R's
  `dim(cn) <- c(ntime, ntrans, ...)` misaligns the transitions.
- Offsets enter the risk scores (data offsets, and new-data offsets); R
  computes `offset.mean` and `offset2` and never uses them.
- The `survmean2` table of stratified curves with several newdata rows reports
  each stratum's own event counts; R's `rep(c(nevent), each = ndata)`
  misaligns them.
- `se_fit=True` warns "se.fit not yet implemented for multistate coxph models";
  R's warning is dead code and it drops the request silently.
- Aliased (NaN) coefficients count as 0, as in `survfit.coxph`; R propagates NA
  into every hazard.
- `na_action="na.pass"` with an incomplete newdata row raises a ValueError; R
  segfaults in `coxsurv1`.
- Curves work after a formula list's first-level missing-value drop, where R
  fails with "Failed to reconstruct the original data set". Rows with a missing
  user-strata value belong to no curve and are left out of the counts.
- Subsetting keeps `n_id` and every `n_transition` column (R drops `n.id` and
  keeps only the first column); an unknown state name or an out-of-range index
  is an error (R returns curves filled with NA).

### Parametric regression (`survreg`)

- **Loglogistic with a zero lower bound**: `survreg.R:130` lists
  `"Log logisit"`, so R skips the zero-lower-bound conversion for the
  loglogistic family and errors "Invalid survival times for this distribution"
  on interval data with `l = 0`. The port converts, as for the other log-time
  families.
- **t-family deviance**: R's `center` is `rowMeans(y)`, which averages the
  status column into interval rows, and its log likelihood takes the log of a
  negative number (`survreg.distributions.R:147-148`). The port centres on the
  interval and evaluates the likelihood; response residuals of interval rows
  differ, and R's deviance residuals there are NaN ("NaNs produced").
- **Negative deviance terms** from rounding are clamped to 0 before the square
  root; R returns NaN.
- **vcov(complete = FALSE)** keeps every Log(scale) row and drops the aliased
  coefficients. R's `var[keep, keep]` recycles the location-length NA pattern
  over the scale rows: it usually stops in `dimnames<-` (`~ age + age2 +
  strata(grp)` with three strata), and when one scale row is left it returns
  that row labelled "Log(scale)" (`~ one + age + strata(sex)` loses sex = 2's
  scale).
- **summary(correlation = TRUE)** with an aliased coefficient returns the
  correlation of the coefficients that are not NA (R stops in `dimnames<-`);
  with a single coefficient it returns `[[1]]` (R's `diag(1/stds)` builds an
  identity of size `1/se` and stops with "non-conformable arguments").
- **Penalty terms in interactions** (survreg and coxph) are refused with R's
  intended "Penalty terms cannot be in an interaction". R never reaches that
  stop: `pspline(age, df = 3):sex` fails with "missing value where TRUE/FALSE
  needed", and `pspline(age, df = 3) * sex` or `ridge(age, theta = 1) * sex`
  fit the interaction columns unpenalized (`survreg.R:213-218`,
  `coxph.R:575`). In survreg the check runs before the frailty check, so
  `frailty(inst):sex` gets the interaction message where R says "survreg does
  not support frailty terms".
- **Frailty terms** are refused by penalty kind; R greps "frailty" in every
  model-frame column name, so it also refuses a ridge/pspline model with a
  covariate named like `frailty_score`.

Penalized survreg (`survpenal.fit`, `survreg7.c`):

- Non-sparse `linear.predictors` include the offset; `survpenal.fit.R:597`
  drops it, although the likelihood, R's sparse branch and `survreg.fit`
  include it.
- Linear predictors count an aliased coefficient as 0 (R's are all NA);
  robust and cluster variances of such a fit are therefore computed, where R
  fails with "missing value where TRUE/FALSE needed".
- A one-column diagonal penalty with a fixed scale uses the 1x1 penalty; R's
  `diag(c(second))` fails with "non-conformable arguments".
- A penalized term after a sparse frailty fits (R: "object 'pcol' not found",
  `survpenal.fit.R:166`); a numeric `init` of length `nvar` with a fixed scale
  and a sparse term gets the frailty zeros prepended where R misaligns it.
- A robust variance with a sparse term is refused ("robust variance is not
  available with a sparse frailty term"); R fails with "non-conformable
  arguments".
- Penalized t fits with interval-censored rows work: `survpenal.fit.R:85` drops
  `parms` from the upper-endpoint density, and `survreg7.c:220` sizes the
  callback buffer at `n` entries, which overflows the heap.
- A sparse frailty with a callback distribution (t) is evaluated with correct
  indexing; `survregc2.c` adds `beta[i]` instead of `beta[i + nf]` to eta and
  reads a multi-strata scale at `beta[strata + nvar]` instead of
  `beta[strata + nvar + nf]`.
- A model without dense columns, a sparse term and an estimated scale starts
  from the intercept-only log scales; R's init fails in `rep(0, nvar - 1)`.
- `inner_failures` carries R's intended `iterfail` list, which R never fills
  because `survreg7.c:460` overwrites the flag with 1000; like R, the fit does
  not warn.

### Concordance

- `concordancefit(std.err = FALSE)` with two or more strata returns the
  per-stratum counts. R reshapes the five counts per stratum into rows of six,
  misaligning its counts and concordance (0.4948 instead of 0.1905 for
  ovarian's `rx`), with a "data length is not a sub-multiple" warning.
- `ranks = TRUE` with strata works (R: "number of items to replace is not a
  multiple of replacement length"); several predictors with strata give one
  pooled value per predictor (R: "length of dimnames [1] not equal to array
  extent").
- Zero-weight deaths with `timewt` other than "n": R's `fastkm.c` fills
  `etime` for every death time but counts only positive-weight ones, and
  overruns its arrays.
- `timefix = FALSE` is honoured. R's `concordance.formula` and `cord.work` call
  `concordancefit` with its default `timefix = TRUE`, so near-tied times are
  merged either way.
- More than ten strata with `keepstrata = FALSE` and `timewt` "S", "S/G" or
  "n/G2": R errors in `colSums` ("x must be an array of at least two
  dimensions"); lung `strata(inst)` with `timewt = "S"` gives 0.4258949 here.
- With merged strata (more than ten, `timewt` "n" or "I"), R's time shift
  shows in its ranks, and `ymax` is compared with the shifted times, so a
  `ymax` below the largest time gives every stratum after the first weight zero
  (R returns NaN with all-zero counts). The port reports real times and
  compares `ymax` with them.
- Several fits: `var` is `crossprod(dfbeta)`, whose diagonal is each fit's own
  variance. R's `cord.work` computes `t(wt * dfbeta) %*% dfbeta`, applying the
  case weights twice, and fails with weights plus a cluster.
- A missing strata or cluster value raises ("strata contains missing values",
  "cluster contains missing values"); R either errors ("NAs are not allowed in
  subscripted assignments") or, with `std.err = FALSE`, misaligns its counts,
  and its `rowsum` makes a missing cluster a group of its own. A factor-like
  strata value outside its declared levels raises (R's `factor()` makes it NA,
  which then fails as above).
- `concordance(fit, newdata)` with a row missing only its response drops the
  row from both the frame and the predictions; R drops it from one and stops
  with "x and y are not the same length".

### Survival curves

- `summary(fit, times)` of a multi-state curve with a time before its first
  time reports `p0` there. R's `findInterval` index is 0 for that time, so its
  `pstate` has one row fewer than its `time`.
- `survfit0` adds the zero influence column only to the curves that get the t0
  row. R's `lapply(x$influence.surv, addcol)` runs over every curve as soon as
  one needs a row, so a curve already at t0 gets one column more than it has
  times.
- `aggregate_survfit` drops `std.chaz` with the other components that do not
  collapse. R's list names `"std.cumhaz"`, which no survfit object has, so R
  keeps the uncollapsed `std.chaz`.
- The id/cluster check warns whenever an id appears in two clusters. R's
  `.Call(Ctwoclust, id, cluster, order(id))` passes 1-based positions to 0-based
  C code and misses cases, and with `start.time` compares trimmed ids with the
  untrimmed cluster vector. With cluster and `start.time`, the clusters of the
  kept rows are numbered; R's `survfitAJ` keeps the untrimmed cluster vector.
- `fit[, states]` keeps only the selected states' columns of `n.censor`,
  `n.enter`, `se0` and `i0` (R keeps every old column of the first two and
  drops the others); a stratum after one that `start.time` emptied reports its
  own `n` and `n.id` (R's `[.survfit` indexes `n` by the curve's position among
  the fitted curves).
- `survfit(coxfit, newdata, start.time =)` where `start.time` empties a stratum
  gives each newdata row its own curve; R's `split()` drops the empty stratum
  and hands a row the curve of the row before it. With `id =` and a subject
  whose intervals all end before `start.time`, the subject gets an empty curve
  (R stops in `apply(dt, 2, cumsum)`), and `id` names a newdata column (R
  evaluates it outside newdata: "object 'pid' not found").
- Turnbull (interval-censored) curves: `cluster` raises "cluster is not
  supported for interval-censored data" (R hands the full cluster vector to the
  `survfitKM` fits of the pseudo-observations, which reads out of bounds). For
  exact, left- and right-censored rows `start.time` compares their time; R's
  rule compares the placeholder 1 in `y[, 2]` and drops them for any
  `start.time > 1`.
- `survexp` with a coxph rate table and an individual method returns values
  when `subset =` or the na.action removes rows (R errors), and a row whose Cox
  response is missing is dropped with the others under `na.omit` (R returns NA
  for it when no other row is removed). `na.exclude` restores removed rows as
  NaN at positions relative to the selected subset.

Turnbull curves follow R's EM (`survfitTurnbull.R`: Aitken acceleration every
fifth step, stopping at max |change| < 5e-5). That rule is sensitive to
floating-point order, so on larger data the curves agree with R only to that
tolerance amplified: up to about 0.02 in survival at n = 500 and 0.006 at
n = 2000, with different iteration counts. The fixtures use small data where
this does not show.

### Data preparation

- `survSplit`: a row with a missing endpoint is episode 2 as in R, keeps its
  status and gets `added = FALSE`; `survsplit.c` leaves its censor flag
  uninitialised, so R's status and `added` there depend on memory contents.
- `survcondense` on data not sorted by id and time: R's two special-case
  branches (a single merged block) index `Y` by sorted position without
  `index[]` and move the wrong start time. For `data.frame(id = c(2, 1, 1),
  t1 = c(1, 0, 3), t2 = c(5, 3, 8), st = c(1, 0, 1), x = 1)` R returns (3, 8]
  for id 1; the port returns (0, 8].
- `rttright` at reporting times on counting-process data uses the status. R
  tests `Y[, 2] > 0` (the stop time, `rttright.R:186`), so a subject's censored
  last row keeps a weight: with `times = c(3, 8.5)` R's weights sum to 1.467 at
  8.5, the port's to 1.
- `finegray` raises "censoring probability is zero before a selected event"
  where R's weights become Inf/NaN (delayed entry can drive G to 0 before a
  later event).
- `finegray` computes the delayed-entry flag from each subject's first row; R
  reads unsorted `Y` at sorted positions, so its output depends on the row
  order. The port equals R on data sorted by id.
- `tmerge`: on a first call without `tstop`, a first event whose time has the
  wrong length raises "argument <name> is not the same length as id"; R indexes
  the short vector by subject and stops with "missing time value, when that
  variable defines the span".
- `Surv2` timeline data: a given `istate` must equal the derived current state
  at each interval's start row; R's check compares vectors of different
  lengths.

### Other routines

- `survobrien` with strata: R's right-censored branch compares the status column
  with time (`y[, 2] >= temp[x, 1]`), and the counting branch selects the other
  strata (`!strata.keep == ...`). The port implements the intended comparisons.
- `survobrien` takes the kept variables from the data after subset and
  na.action, aligned with the expanded rows; R indexes the full data by
  model-frame row, which misaligns them when rows were dropped.
- `survcheck`'s events table keeps every state with events. R's
  `novisit <- rowSums(events[, -1]) == 0` assumes the first count column is 0,
  so when every subject visits at least one state it drops real states (ids 1
  and 2 each going a then b give only the "(any)" row).
- `yates` on a Cox fit with `strata()`: R builds its population rows with
  `model.matrix(Terms)`, which includes the strata column that the fit's design
  leaves out, so Cmat is misaligned with the coefficients (with an aliased
  coefficient R stops in `qr.resid`). The port uses the fit's own design.
- `yates(predict = "survival")`: R's summary keeps the time-0 row of the curve,
  so its `surv` has one row more than its time vector and every row sits one
  time early, and it leaves `cumhaz` at the baseline's (it assigns
  `-log(surv)` to a misspelt local). Here row i is the curve at `time[i]` and
  `cumhaz` is `-log(surv)` per level.
- `yates` with no estimable level, or with an `rmean` at or before the first
  time: R 3.8-12 stops inside `gsolve` ("requires numeric/complex
  matrix/vector arguments"); the port returns NA tests, or a test of 0 on 0 df
  (the result R's branch computes).
- `brier` on a stratified Cox fit returns scores; R 3.8-12 errors in
  `summary.survfit`. `brier(fit, newdata)` on the fit's own data equals
  `brier(fit)` (lung reversed, `~ age + strata(sex)`: 0.2078394407,
  0.2325654809).
- `aareg` on a design with an unused factor level raises "the nmin threshold is
  too high; no Aalen model can be fit"; R 3.8-12 does not return (still running
  after 90 s on lung).
- `pyears(expect = "pyears")` with a zero rate uses the limit of
  `(1 - exp(-lambda t)) / lambda` at `lambda = 0`, which is `t`; R's C code
  divides by zero.
- `summary_pyears` of a result without terms returns scalar tables (R's
  `summary.pyears` fails while printing); with `totals=True` it raises where
  R's `pytot` fails with "dim(X) must have a positive length".

## Other deliberate differences

### Versions

- The fixtures and the R bridge's CI use survival 3.8-11. `Surv2` and
  `surv2counting` follow 3.8-12: a missing 0/1 status stays NA (the interval is
  then removed by the na.action) and a missing factor outcome is censored
  without 3.8-11's level shift. The multi-state `parsecovar2` and `survfitAJ`
  paths follow 3.8-12 as well.
- `Surv` and `Surv2` retain the censoring factor's first level in `clabel`
  and format it as `:label`, following 3.8-12. Legacy objects without `clabel`
  keep the `+` marker. The old formatting fixture reconstructs that legacy
  metadata; the vector-operation fixtures check current labels against 3.8-12.
- The censoring column of a transitions table is labelled "(censored)" (3.8-11)
  where 3.8-12 prints "(<first level>)", e.g. "(censor)".
- `rsurvreg(seed=s)` reproduces R's `set.seed(s)` stream (Mersenne-Twister,
  inversion); without a seed it draws from a clock-seeded generator, since
  there is no R session whose RNG state it could share. Seeds are R's 32-bit
  integers in [-2^31 + 1, 2^31 - 1].

### Values and numerics

- (start, stop] Cox fits (`agfit4.c`): when R moves the centre after a tied
  death was added at the same time, it leaves that death on the old centre;
  here the deaths of a death time use its final centre. Rows with tied entry
  times leave the risk set in stop-time order rather than data order, and
  Efron's j-th terms are formed as `sum - j/d * deaths`. Results agree to
  rounding whenever the centre does not move mid death time.
- `core.coxph_fit` (and the R bridge's `coxph.fit`, `agreg.fit` and
  `agexact.fit` wrappers) follows `coxph()` rather than the bare fitters: it
  centres the offset, so `agreg.fit`'s info rescale count differs for a large
  offset (0 instead of 5 for heart with offset -750; coefficients and loglik
  agree), and zero-event data get `coxph()`'s skeleton where R's `coxph.fit`
  iterates to coefficient 0 with "Ran out of iterations".
- A Cox fit's `offset` is the offset as given; R's `fit$offset` exists only for
  `x = TRUE` and is the centred offset.
- Linear algebra: R's `solve()` estimates the 1-norm of the inverse with
  `dgecon` (a lower bound); the port computes it exactly, so a matrix right at
  the `eps` boundary that R accepts can be reported singular. The singular
  message reads "LU factorisation (reciprocal condition number = X): system is
  computationally singular" with six significant digits.
- `signif` beyond |x| of about 1e306 (or below 1e-306) in the penalized survreg
  print can differ from R in the last bits.
- The t family's `n.eff` evaluates `variance(sigma^2)` as R does and can be
  negative (-111.527 on log lung time); only AIC-method psplines with
  `dist = "t"` use it.

### Missing values and newdata

- Under na.pass (predict's default) and na.exclude, a newdata row with a
  missing value in any variable the prediction reads predicts NaN in every
  column, where R lets the NA reach only what uses it: `predict(type =
  "terms")` keeps the terms whose variables are present in R (lung, age = NA,
  sex = 2: R's sex term is -0.3106); `predict.survreg` computes lp, response and
  terms for a row with a missing stratum (R: 6.215335, 6.090643);
  `predict.coxph(type = "expected")` gives 0 for a missing stratum and ignores
  a missing status of a right-censored response (R: 0.140304, 0.372894 for
  status c(1, NA); here 0.1403, NaN). Under na.omit both drop such rows.
- A newdata row with an infinite covariate (from `log(0)` or `x/0`) raises
  "newdata contains non-finite value"; R predicts ±Inf.
- A response made infinite by arithmetic (`Surv(time/z, status)` at `z = 0`)
  stops with "time contains non-finite value inf"; R's coxph stops in
  `coxph.fit` and survreg with "Invalid survival times for this distribution".
- aareg and concordance refuse a numeric covariate with NaN under na.pass with
  the engine's message; R's aareg stops in the foreign call and its
  concordance returns an object with no concordance.
- `survexp` with a Cox rate table and an incomplete na.fail newdata row raises
  "missing values in newdata" (R: "non-conformable arguments").
- `core.CountingProcessData` rejects `stop <= start` where R's `Surv` makes the
  row NA with a warning; the formula interface's model frames treat such rows
  as missing, as R does.
- `survfit` on `Surv2` data applies an explicit `na_action` only after the
  timeline conversion (Python cannot tell an explicit "na.omit" from the
  default); R also applies a user-given na.action before it.
- `survConcordance_fit` with a missing x or y raises; R returns NULL.

### Formulas and arguments

- Term labels are the formula's own text; R deparses them (R names
  `I(age*1e-3)` "I(age * 0.001)", `pspline(age, df=4)` "pspline(age, df = 4)",
  and the `pspline(penalty = FALSE)` columns after the deparsed call). They
  agree when the formula is written with R's spacing.
- In a comparison `T` and `F` are column names (R falls back to TRUE/FALSE when
  the data has no such column); a bare word in an rmap string is a label
  (`race = "white"`), where R's `rmap = list(race = white)` is a variable
  lookup.
- An ordered comparison (`<`, `<=`, `>`, `>=`) with a string or factor operand
  raises; R orders strings by the locale's collation and gives NA for a factor.
- `cut()` refuses an argument it does not know; R's `cut.default` swallows it.
- Last value carried forward in timeline data runs over the data columns the
  formula reads, not R's evaluated model-frame variables: `I(x + z)` carries
  each column forward on its own.
- `fromtimeline`'s `Surv(time, status)` form accepts right and mright data only;
  with one interval per subject and a time argument that is not a variable name
  the default response names are tstart/status, where R fails.
  `model_frame()` refuses a `Surv2` response (R returns the Surv2 matrix), and
  `predict(type = "expected", newdata)` on a Surv2 fit asks for the response
  columns. survcheck reports problem rows of Surv2 data as rows of the
  converted (counting-process) data.
- A formula list refuses a bare unquoted state name on its left-hand side (R
  evaluates it as a variable); `coxph(strata=vector)` is refused for
  multi-state fits.
- `coxph.control`: a non-finite `iter.max` or `outer.max`, or one ≥ 2^31,
  raises coxph.control's message (R's `as.integer` gives NA with a warning and
  the fit then fails); the names in a `control=` mapping are matched exactly
  (R's `do.call(coxph.control, control)` also matches partial names).
  `survreg_control(outer_max < 1)` raises "invalid value for outer.max", also
  for unpenalized fits, where R never checks it.
- `dsurvreg`/`psurvreg`/`qsurvreg`/`rsurvreg` keep `survreg()`'s `parms` rules:
  the t family defaults to df = 4 (R stops on the missing parms), requires
  a finite df ≥ 3 (R evaluates `dt`/`pt`/`qt` with it, e.g.
  `dsurvreg(1, 0, 1, "t", parms = 1)` = 0.1591549431), and `parms` given to
  another family are an error (R ignores them).
- `survreg(offset=)` (a column name or vector) and `survSplit(id=)` with a
  character id are Python extensions; R uses `offset()` in the formula and
  errors when survSplit must add an id to a subset.
- `concordancefit` keeps a keyword-only `names` for the columns of x, standing
  in for R's `colnames(x)`.
- `survobrien` keeps plain string columns untransformed like factors (a Python
  list has no factor/character distinction; R ≥ 4 rank-transforms a character
  vector).
- A logical or factor response expression in `concordance` is evaluated into a
  column, and the data columns it reads are removed before the model frame; a
  right-hand side, weights or cluster argument naming one of them fails.
- `clogit`'s response is named `Surv(rep(1, length(case)), case)` in the model
  frame (R: `Surv(1 + 0 * case, case)`).
- `start.time` in `survfit(coxfit)` must be finite; R also accepts -Inf and Inf.
- anova refits of survreg fits use the rows of the original fit; R's `update()`
  re-evaluates the reduced formula on the full data, so dropping a term with
  missing values gains rows in R.

### Results and labels

- `survfit.matrix` supports omitted `p0`, unstratified curves and empty event
  grids (R 3.8-12 errors on these inputs). A start parameter earlier than the
  curves' shared `start_time` warns and uses the latter; R warns but still uses
  the earlier parameter. Sample sizes are repeated for each Cox prediction
  column to align with the output curves; R leaves `n` at its original length.
  The result retains the input transition cumulative hazards and transition
  event counts, supplies zero censor counts, and leaves Python `n_id` unset.
  These extra fields make the ordinary multistate curve methods available;
  no censoring histories or unique subject counts are reconstructed. State
  names default to strings or come from a named `p0` mapping or `states=`.
  The `p0` sum tolerance is `1e-8` instead of R's exact equality check; values
  are not renormalized. Curves and state names are checked for finite numeric
  inputs, nondecreasing hazards and unique, nonempty labels.
- Character strata, ids and cluster labels are sorted by code point, with
  numeric-looking strings as numbers ("9" before "10", "B" before "a"); R's
  `as.factor` uses the locale's collation. Numeric, logical and factor values
  sort as in R.
- `cch`'s `sc_ids` holds the id values themselves; R's rownames are their
  character form.
- `survfit.coxph` curve names: ids that differ but print alike are kept apart
  with `make.unique` ("0.3", "0.3.1"; R repeats the name); a newdata index with
  repeated or missing labels names the curves 1..m; a 0-based integer index
  names them by its values ("0", "2" where R's row names are "1", "3"); the
  columns of an unstratified newdata matrix are unnamed.
- `quantile()` of a stratified `survfit.coxph` object with several curves has
  one row per (stratum, curve), ordered as `summary()`'s table; R returns a
  stratum x curve x probability array. Quantiles and medians always return a
  `SurvfitQuantileResult`, including single-response and single-probability
  calls. Tolerance-induced interpolation ties use R's averaged indices
  without emitting its "collapsing to unique x values" warning.
- A Turnbull fit reports `cumhaz` and `std.chaz` (R's object has them only after
  `survfit0`).
- `aggregate_survfit` without `by` keeps a data axis of length 1 (the package
  convention); R returns a time x state matrix. The `newdata` of multi-state
  Cox curves holds the rows actually used; R stores the user's data frame.
- Residuals on multi-state Cox curves raise TypeError; R returns Aalen-Johansen
  residuals of the data that ignore the Cox model.
- `summary_pyears` returns the tables instead of printing them (R's print
  switches are not taken); for a `data_frame=True` result the restored table
  keeps every level the pyears call used.
- `model_term_names(aareg)` returns the formula's term labels (R's
  `labels.aareg` returns NULL); `model_frame(survreg)` still needs
  `model=TRUE`.
- A `cox.zph` function transform is labelled by its name, or "user" without
  one, not by R's deparsed expression.
- The penalized Cox summary keeps a scalar `loglik` plus `null_loglik`;
  `anova.coxph.penal`, which R does not register, is not ported (anova of one
  penalized model follows R's reachable `anova.coxph`). Residuals of a null Cox
  model other than martingale and deviance raise "... residuals are not defined
  for a null model".
- `print_survreg_penal` prints the Call block as `survreg(formula = ...)`; R
  `dput()`s the whole call. Its `terms=TRUE` Wald rows keep R's p-value on 1 df
  while the DF column shows the term's df, as `summary_coxph_penal` does.
- The typed `SurvpenalFit` returns `penalty` as `c(0, P)` for every fit (R: a
  scalar for a sparse fit) and dense column means for sparse fits (R includes
  the group-code column).

### Messages

- A `match.arg` miss reads `'arg' should be one of "response", "link", ...` with
  ASCII quotes; R in a UTF-8 locale prints curly quotes.
- tmerge's not-found message adds "in data2" to R's "object 'x' not found"; a
  missing cumevent increment at an event raises a dedicated message (R: "NAs
  are not allowed in subscripted assignments"); a logical tdc default is
  inserted as given (R's assignment turns the column numeric).
- Residuals and pseudo values of a KM fit whose `start.time` empties a stratum
  refuse with survfitAJ's "start.time has removed all the observations from at
  least one curve" (R: "strata k not matched").

### R bridge (`r/survivalr`)

- A penalized survreg fit keeps the bridge's `survival_py_survreg` class rather
  than R's `c("survreg.penal", "survreg")`, so survival's own methods do not
  dispatch on it.
- The `summary.survfitms` list of `survfit.coxphms` curves carries time, the
  counts, pstate, cumhaz, states, table, rmean.endtime, strata and newdata, not
  R's n, n.id, p0, transitions or call.

## SAS-style Yates tests

`yates(method="sgtt")` computes SAS-style type III tests for selected
main-effect variables in a treatment-coded model. It defaults to `population="sas"`
and requires linear predictions. Marginal estimates and their covariance
match the direct SAS-population calculation; the test uses the estimable
hypotheses from the full indicator design. `YatesResult.sas`, `sas_names`,
and `sas_row_names` expose that matrix and its labels.

The Rust `validation::yates_sgtt` kernel takes a `YatesSgttInput` with the
indicator design, term assignments and adjustments, fitted coefficients,
and covariance. It reuses one LINPACK-style QR to solve for all design
columns and one per adjustment block. Residual-only QR callers do not
retain the triangular factor needed by coefficient solves.

For a Cox model, the baseline intercept is removed before testing the
fitted coefficients. R 3.8-12 leaves that extra dimension in the SGTT
hypothesis matrix and can fail with "non-conformable arguments"; the port
supports this case. The external `YatesModel` adapter uses treatment
contrasts; arbitrary contrast matrices are not exposed by this adapter.

Joint requests such as `term="a + b"` or `term="a:b"` compare the Cartesian
product of both variables' levels, with the first variable varying fastest.
A scalar or vector of one-based fitted term numbers also selects variables,
including the variables of an interaction term.
A `levels={"a": [...], "b": [...]}` mapping supplies per-variable values;
omitted categorical variables use their fitted levels. Direct tests compare
all combinations jointly, or pairwise when requested. SGTT returns one type
III test per selected main-effect variable, in the requested order.

Three R 3.8-12 defects are corrected here: reversing the requested factor order
can apply the wrong factor levels in R, and joint nonlinear global tests
fail when R assigns several names to a single test row. The port preserves
the requested variable order and names a joint global test `global`. Numeric
term selection includes the last fitted term, which R accidentally rejects
by using an exclusive upper bound.

`scripts/generate_yates_sgtt_reference.R` records linear-model estimates,
covariances, SAS matrices, and tests for additive, interaction, unbalanced,
weighted, missing-cell, and no-intercept models.
`scripts/generate_yates_joint_reference.R` adds joint categorical and numeric
requests, partial level mappings, population choices, and Cox risk predictions.

The native factorial benchmark (nine indicator columns, four estimable
coefficients, 15 samples) measured median times of 39 microseconds for 1,000
rows, 0.59 milliseconds for 10,000, and 13.8 milliseconds for 100,000 on the
development machine. Reproduce with
`cargo bench --bench survival_benchmarks --offline -- yates_sgtt_bench --sample-count 15 --sample-size 1`.

All Yates Python kernels release the GIL and accept NumPy arrays through the
typed input converters. Nested matrices use `FloatRows`, which reads arrays
directly into row buffers and keeps list inputs without a flatten/rebuild
cycle. In `scripts/benchmark_yates_sgtt.py`, the 100,000-row NumPy call fell
from 55.3 to 13.4 milliseconds (median of nine calls); list inputs measured
18.3 milliseconds before and 17.7 after. Fortran and strided arrays measured
13.2 and 14.2 milliseconds. These include Python argument conversion and
result construction, unlike the native benchmark above.

## Survival response vector operations

`concat_surv`, `rep_surv`, `rev_surv`, `unique_surv`, `duplicated_surv`,
`transpose_surv`, `levels_surv`, `head_surv`, and `tail_surv` provide R's
row operations through `survival.r` and the package root. Concatenation
requires matching censoring types and multistate levels. Row operations
preserve normalized status codes, state levels, and the censoring label;
they never recode a subset containing only status 2 as an ordinary 1/2
event response.

Repetition supports scalar or per-entry `times`, `each`, and `length_out`
(`length.out`); the length takes precedence over times. Duplicate detection
and uniqueness support `from_last` (`fromLast`) and compare complete rows,
including missing values in matching columns. They use a hash set with
expected linear scan time and storage proportional to distinct rows.
Python responses represent R's numeric NA and NaN by the same NaN, so
these are treated as a single missing value. Empty responses remain empty
under repetition, reversal, head, and tail; R's `1:nrow(x)` in those methods
can instead produce an out-of-bounds error.

`as_character_surv` returns labels without extra right padding;
`format_surv` pads them to a common width. Both retain the common numeric
precision and leading padding of the time columns. `transpose_surv`
returns a plain matrix with one row per response column, and `levels_surv`
returns event-state names or `None` for an ordinary response.

`scripts/generate_surv_vector_reference.R` records 144 operations across
right, left, counting-process, interval, and multistate responses, plus
concatenations. Run `scripts/benchmark_surv_vectors.py` to measure duplicate
detection and uniqueness on distinct and repeated rows.
On Linux x86-64 with Python 3.14.7 (median of 11 calls, excluding response
construction), duplicate detection took 0.22/2.42/30.25 ms for
1,000/10,000/100,000 distinct rows. At 100,000 rows, uniqueness took
35.61 ms for distinct rows and 26.59 ms when each row occurred ten times.

## Population model components and summaries

Native calendar conversion rejects nonfinite day counts and dates outside
the `CalendarDate` range (years representable by `i32`). Calendar cutpoints,
input dates, and derived birth dates receive the same checks; malformed
table dimensions also reject zero lengths and size overflow. Rust
`days_to_date` and `start_of_year` return `SurvivalResult`; Python raises
`ValueError`. R can retain and print nonfinite or larger numeric `Date`
values, which cannot be represented by the native calendar object.

`pyears` and cohort `survexp` honor `model`, `x`, and `y`. `model=True`
retains the evaluated formula columns, original source columns referenced by
`rmap` expressions, and supplied weights. Factors retain their level order
and unused levels, and `tcut` columns retain cutpoints and labels. Subsetting
and missing-row removal apply consistently to every retained component.
`model=True` takes precedence over `x` and `y`, as in R.

With `model=False`, `pyears(x=True)` keeps a row-major matrix of one-based
category codes and raw scaled `tcut` times; `survexp(x=True)` keeps a
`StrataFactor` with zero-based codes, labels, and counts. Without grouping
terms, either function retains a vector of ones. `pyears(y=True)` keeps the
`Surv` response or a one-column numeric matrix. `survexp(y=True)` keeps
numeric follow-up times; without a response, a rate-table call uses the
maximum requested time before output scaling, and a Cox-reference call
keeps `None`.

Both results support `model_formula` and `model_term_names`, plus
`model_frame` when made with `model=True`. The latter returns plain columns
and expands a `Surv` response into its time/status columns; `.model`
preserves the richer column objects. The default flags retain no row data.
Individual `survexp` methods return a plain vector and ignore the retention
flags. With `na.exclude`, they restore excluded rows as NaN.

`scripts/generate_population_retention_reference.R` records these components
and numerical outputs against R, including repeated subset rows, missing
values, expression mappings, factor order, and both rate-table and Cox
references.

`summary_survexp`, `summary_ratetable`, and `summary_tmerge` are available
through `survival.r` and the `model_summary` generic.

Expected-survival summaries return a `SurvExpSummary`: a vector for one
curve or a time-by-curve matrix otherwise. The `strata` labels name columns,
as on `SurvExpResult`. Requested times are sorted with duplicates retained;
missing times and values outside the observed range are dropped. Survival
uses the preceding observation (1 before the first), and risk counts use
the next observation. Omitted times keep the original rows. Scalar `scale`
divides the output times, including R's zero and nonfinite arithmetic.
Repeated source times use R's averaged interpolation indices without its
tie-collapse warning. R call expressions and manually attached `na.action`
attributes are not represented in these Python result containers.

Rust callers use `population::summary_survexp` with a `SurvExpResult`; the
Python native entry point accepts its time vector and row-major matrices.
The native selector validates source shapes and time order, sorts the
requested times when necessary, then uses a single sweep. Its selection
work is linear in source rows plus requested rows after sorting, with output
copying proportional to the number of selected cells. The
`expected_survival_summary_bench` group measures sparse and dense requests
on two-curve inputs of 1,000–100,000 rows.
On a local Linux x86-64 release build, the median of 15 samples at 100,000
source rows was 0.309 ms for 25 requested times and 3.80 ms for 100,000
requested times, excluding source-curve construction. Reproduce with
`cargo bench --bench survival_benchmarks -- expected_survival_summary_bench --sample-count 15 --sample-size 1`.

`summary_ratetable` returns a `RateTableSummary` containing the canonical
attributes, a dimension table, and native summary text available through
`str(result)`. Factor dimensions have levels; other dimensions have bounds
in their native units, with calendar bounds formatted as ISO dates.
`summary_tmerge` returns a column-oriented count table with one row per
operation. These methods return structured data without printing. Expected
curves and summaries, rate tables and their summaries, and the merged-data
count table all support `as_data_frame`.

`scripts/generate_population_summary_reference.R` regenerates curve-selection
edge cases, population-method examples, all three bundled rate-table
summaries, and merged-data counts from R survival 3.8-12.

## P-spline prediction

`predict(pspline(x), newx)` and `predict_pspline(basis, newx)` evaluate the
existing Rust basis on new values, preserving the original boundaries,
degree, number of terms, intercept and combined columns. Values beyond the
boundaries use linear extrapolation. The result is a `PsplineResult` with
`penalty=False`; its matrix is in `.basis`. Omitting new values returns the
original object. The generic accepts either `newdata` or R's `newx` keyword.

The shared penalty builder accumulates the three nonzero entries in each
second-difference row directly, replacing a cubic dense multiplication.
It still returns the same dense matrix, requiring quadratic storage and
initialization. In a local release run with ten prediction rows, median
times over nine calls fell from 0.110 to 0.020 ms at 10 terms, from 33.1 to
0.154 ms at 100 terms, and from 764 to 0.751 ms at 300 terms. Basis and penalty
values were identical. `scripts/bench_pspline_basis.py` measures the shared
construction path with fixed prediction boundaries.

`scripts/generate_pspline_prediction_reference.R` regenerates ten R survival
3.8-12 cases covering degrees, extrapolation, missing rows, combined columns,
intercepts and smoothing methods, including their penalty matrices.

## Response quantiles

`quantile` and `median` accept raw `Surv` responses and fitted KM, Turnbull,
and Cox survival curves. `quantile_surv`, `median_surv`, and `median_survfit`
are the explicit method names. Raw responses use the existing Rust curve
fitters before inversion; they reject missing rows unless `na_rm=True` and
refuse multiple-endpoint responses as R does. `median(Surv(...))` includes
confidence bounds by default, while `median(survfit(...))` returns point
estimates. Both preserve the curve-by-probability result layout of
`quantile_survfit`.

The native quantile routine handles non-monotone confidence bands, duplicate
levels, missing band entries, and tolerance-induced ties using R's sorting
and index-averaging rules. With the default tolerance, an entirely censored
curve returns an undefined quantile even at probability zero, matching R's
terminal-flat rule. The common monotone path uses adjacent comparisons
instead of a hash table and shares one interpolation buffer across the
two tolerance shifts.

`scripts/generate_surv_quantile_reference.R` regenerates the response,
Cox/KM curve, and interpolation references from R survival 3.8-12.
`scripts/bench_surv_quantiles.py` measures quantile evaluation with curve
construction excluded, and can compare saved release libraries using
`--extension`.

Local release measurements on Linux x86-64 with Python 3.14.7 used the median
of 101 calls. All 24 monotone benchmark combinations returned identical
estimates and bounds before and after the change. Representative timings
with confidence bounds enabled were:

| Curve points | Probabilities | Repeated levels | Before (ms) | After (ms) |
| ---: | ---: | :---: | ---: | ---: |
| 1,000 | 3 | no | 0.043 | 0.019 |
| 10,000 | 3 | no | 0.623 | 0.183 |
| 100,000 | 3 | no | 7.550 | 2.072 |
| 100,000 | 1,001 | no | 7.745 | 2.348 |
| 1,000 | 1,001 | no | 0.118 | 0.123 |
| 1,000 | 1,001 | yes | 0.090 | 0.101 |

Across curves with and without repeated levels or confidence bounds,
quartiles were 2.1–3.8 times faster. Dense grids on 10,000–100,000 points
were 1.6–3.4 times faster. On 1,000-point curves, requesting 1,001
probabilities was 4–16% slower (about 3–12 microseconds): the shared buffer
trades additional arithmetic during searches for less allocation and copying.

## Not yet implemented

These R entry points have no port. Calls that reach them raise an explicit
error; none silently falls back to other behaviour.

- **R-style print and format methods**: `survival.r` returns data objects and
  `as_data_frame` tables instead of printed output. Only
  `print_survreg_penal` ports R's printed table, and `format_surv` formats a
  `Surv`; `str(summary_ratetable(...))` exposes the native rate-table text.
  The sparse (frailty) branch of `print.survreg.penal` is not ported.
- **R operator groups**: `Surv` arithmetic, comparisons and reductions do
  not dispatch through R's `Math`/`Ops`/`Summary` groups (which reject all
  operations). Python's ordinary object equality remains structural.
- **Low-level R exports** that `survival.r` does not re-export: `coxph.fit`,
  `agreg.fit`, `agexact.fit`, `survreg.fit`, `survpenal.fit`, `survfitKM`,
  `coxsurv.fit`, `survfitcoxph.fit`, `attrassign`, `untangle.specials`,
  `yates_setup` and `match.ratetable`. Their computations are reachable through
  the formula functions and the domain modules (`survival.regression.coxph_fit`,
  `survreg_fit`, `survival.surv_analysis.survfitkm`,
  `survival.population.match_ratetable`, ...); the R bridge provides R-named
  wrappers for several of them.
- **Graphics**: `plot`, `lines`, `points` and the other plotting methods.

Features R itself does not implement stay refused with R's message: anova on
multi-state fits; `predict.coxphms` types expected, survival and terms and
`reference = "strata"`; `basehaz`, `royston` and `yates` on multi-state fits;
`survexp`/`pyears` with a multi-state rate table ("Invalid rate table").

`pyears(ratetable=cox_fit)` also remains explicitly unsupported. R 3.8-12
recognizes the fit initially but then tries to coerce it to a numeric rate
table and fails with "'list' object cannot be coerced to type 'double'".
`survexp(ratetable=cox_fit)` is supported.

## Reproduce validation

```sh
cargo test --lib --no-default-features --offline
PYO3_PYTHON=$PWD/.venv/bin/python PATH=$PWD/.venv/bin:$PATH \
  cargo test --lib --all-features --offline
PYTHONPATH=python .venv/bin/python -m pytest python/tests -q
python3 scripts/generate_binding_manifest.py --check
PYTHONPATH=python .venv/bin/python scripts/generate_stubs.py --check
```

The normal fixture run includes the documented expected differences. Add
`--runxfail` to display their full discrepancies. R itself is needed to regenerate
the reference data; see [the fixture workflow](../test/r/README.md).

For curve complexity, benchmark inputs and measured timings, see
[Kaplan–Meier performance](kaplan-meier-performance.md).
