# R survival compatibility

The Python formula interface is `survival.r`; its numerical routines run in
Rust and are also exposed through the Rust crate's domain modules. The original
reference fixtures were generated with R survival **3.8.11**. They cover model
fits, covariance matrices, predictions, residuals, survival curves and utilities.
They establish compatibility for the tested inputs, rather than a claim that
every R expression, plotting method or extension package is supported.
Newer regression tests and generated fixtures use values from R survival
3.8-12 (R 4.5.3); the few places where 3.8-12 changed behavior are listed under
[Versions](#versions).

This page records every known difference from R:

- [API coverage](#api-coverage): exported entry points;
- [Deliberate fixes of R defects](#deliberate-fixes-of-r-defects): inputs where R
  errors, crashes or returns wrong numbers and the port returns the intended
  result;
- [Other deliberate differences](#other-deliberate-differences): versions,
  numerics, messages, labels, sort orders and Python-specific representations;
- [R behavior reproduced deliberately](#r-behavior-reproduced-deliberately): R
  quirks the port keeps on purpose;
- [Reference differences retained deliberately](#reference-differences-retained-deliberately):
  the fixture checks that are expected to fail;
- [Reference limitations](#reference-limitations): operations refused by R itself.

The interfaces themselves are described in the topic guides:
[formula interface](formula-interface.md), [predictions](prediction.md),
[low-level fitting](low-level-api.md), [survival curves](curves.md),
[printing and graphics](printing-and-graphics.md),
[population calculations](population.md), [data preparation](data-preparation.md),
[concordance](concordance.md), [marginal means](yates.md),
[R bridge](r-bridge.md) and [performance](performance.md).

## API coverage

The exported API inventory covers all 80 exports in stock survival 3.8-12.
That inventory checks entry points; the independent references and deliberate
differences below describe the tested argument and numerical behavior. R's
`uspop2` population array (age by sex by year, 2000–2020) is the one data object
that is not shipped.

## Deliberate fixes of R defects

In each case below R errors, crashes, reads uninitialized memory or returns a
result that contradicts its own documentation or code comments, and the port
returns what R intends. Regression tests pin the intended values. Where a
current-stock reference records defective output, it retains that output
alongside the corrected oracle.

### Formulas and model matrices

- **Strata interactions after a multi-column factor**: R's `predict.coxph` can
  remove the wrong columns when a strata interaction follows a multi-column
  factor, so linear prediction errors and term predictions can silently use
  different columns. The port uses R's correct `model.matrix.coxph` design
  consistently. R's curve prediction fails for the same factor models (the port's
  curves agree with separate fits within each stratum), and R's
  proportional-hazards test fails for redundant full factor indicators (the
  port's global test agrees with the equivalent treatment-coded model).
- **`model.matrix.survreg` with strata interactions**: R's method can fail while
  rebuilding the terms. `model_matrix(survreg_fit, data)` builds the prediction
  rows with the fitted contrasts and drops incomplete rows.
- **Cox model matrices**: the strata-removal branch of `model.matrix.coxph`
  misspells `contrasts.arg`, so a rebuilt dense frailty loses its first indicator;
  the port keeps the fitted full contrasts. `makepredictcall.pspline` drops the
  original `df` and `penalty = FALSE` options, so a valid small unpenalized basis
  fails under the default `df = 4`; the port keeps the original options and
  fitted boundary knots. R's spline constructor fails on empty input; the port
  returns the fitted columns and assignments with zero rows.
- **Interaction-only time transforms**: `coxph` counts main-effect terms rather
  than distinct transform variables, so a `tt()` term that occurs only in an
  interaction can fail; the port runs each distinct transform once.

### Predictions and residuals

- **Offsets in predictions**: the port includes AFT new-data offsets in the linear
  predictor (see the [retained reference
  differences](#reference-differences-retained-deliberately)) and consistently
  centers Cox offsets, even for models without coefficients. For models without
  covariate terms, term prediction returns an empty-column matrix where R's method
  errors.
- **Empty AFT term predictions**: stock AFT term prediction fails on empty new data,
  when omission leaves zero rows (including entirely omitted new data, in
  `attrassign()`) and for intercept-only models (in the empty term loop). The port
  returns a matrix with the selected column names and width and the retained row
  count, or missing-value padding under `na.exclude`.
- **All spline inputs missing**: stock prediction raises an empty-derivatives error;
  the port returns the empty or padded output.
- **Penalized Cox survival predictions**: the stock penalized wrapper refuses `type
  = "survival"`, before validating a term selector; the port computes the prediction
  from the fitted state.
- **AFT term labels with standalone strata**: stock AFT prediction passes the full
  formula terms to `attrassign()` after its model matrix has removed standalone
  strata columns. When strata precede covariates, the remaining columns get
  displaced labels (a `flag` contribution can be labeled `strata(g)`) and a named
  selection can choose the wrong contribution. The port names prediction columns
  from the fitted coefficient assignments.
- **Grouped predictions and residuals**: stock sparse Cox predictions can be
  collapsed both in the delegated prediction method and in the penalized method;
  sparse term reconstruction and partial residuals can fail for a one-column dense
  design; and Cox residuals reject logical group vectors through a scalar logical
  condition. The port groups each prediction or residual once and accepts logical
  groups (excluded rows are covered under [Cox models](#cox-models)). The R wrapper
  keeps the group labels even where a stock method loses them while dropping a
  one-column matrix.
- **Null Cox models**: stock expected and survival predictions for a null Cox model
  fail when multiplying by `NULL` coefficients; the port returns the baseline hazard
  and its exponential survival.
- **Empty AFT probability requests**: stock Gaussian, lognormal and t predictions
  reject `p = NULL`, and estimated-scale empty predictions with `se.fit = TRUE`
  fail; some fixed-scale predictions keep a nonempty error vector beside an empty
  fit. The port returns empty fit and error matrices of the same width.

### Cox models

- **Non-converged variance** (`coxfit6.c`, `coxexact.c`): when iterations run
  out with `iter.max > 1`, R hands the recomputed information matrix to `chinv2`
  without `cholesky2`, so its variance is not the inverse information. The port
  factors it. Lung `age + sex + ph.ecog`, `iter.max = 2`: R's `var[1, 1]` is
  8.330972e-05 (Efron), the port's 8.588522e-05.
- **agfit4 center moves**: a center move of more than 709 raises "exp overflow
  due to covariates" whenever the running sums are positive. R tests only
  `denom` and carries on with the deaths already added: two tied deaths alone
  at risk with offsets 0 and 1419 give R a log likelihood of -709.5 (the true
  value is -1419), where the port raises; with offsets 0 and 1418 R returns
  -709 and the port -1418.
- **Exact counting-process fits with a large common offset**: R's front end
  rejects a common offset of +1,000; the port fits it, giving the fit obtained
  after removing the common shift.
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
  `predict(collapse =)` pad the excluded rows and sum the fit's rows; R checks a
  padded group vector against the unpadded response and fails in `rowsum` on
  vectors of different lengths.
- **Stratified scaled Schoenfeld residuals** with several covariates keep
  their strata attribute; R loses it as a side effect of `%*%`.
- **Tied deaths across strata in `Ccoxcount2`** (`coxcount1.c`): R indexes the
  sorted stratum boundaries through original row indices while walking tied
  deaths, which affects counting-process `tt()` expansions with tied stop times.
  The reference checks use distinct stop times, so this case has no R reference.

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
  both. This affects the covariance, frailty variance and degrees of freedom of
  weighted sparse fits, including time-transform frailties.
- **nocenter with a sparse frailty first**: `coxpenal.fit.R:290` computes the
  0/1 column flags on `x` but applies them to `xx`, so R's means and
  `survfit(fit)` depend on the order of the terms (`frailty(inst) + age +
  female`: means (62.46, 0.4038), `survfit(fit)$surv[10]` 0.94156; with the
  frailty last: (62.46, 0) and 0.92841). The port gives R's frailty-last result
  for every order.
- **predict(type = "terms") with a sparse frailty** puts the frailty column at
  its own place for any term order (R returns a zero column after a frailty in
  the middle, and its wrapper can return malformed term dimensions), adds the
  fitted frailty contributions and errors (zeros for new data), and supports
  `terms =` and `collapse` with a frailty (R errors).
- **Penalized time-transform martingale residuals**: R's cleanup can violate the
  risk-set sum identities (nonzero sums within one-event risk sets for
  transformed splines); the port computes the residuals with the ordinary Cox
  kernel at the fitted linear predictors.
- **vcov of a sparse-only fit**: R's `vcov()` fails because the dense covariance
  is `NULL`; the port returns an empty dense matrix.

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
- A collapse vector with missing values gives a last group labeled "NA"; a
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
  is an error (R returns curves filled with NA). After a state selection,
  `n_censor` follows the selected state columns (R 3.8-12 keeps the original
  columns), and event-only summaries keep per-state censor counts where R drops
  the matrix to one column when joining strata.

### Parametric regression (`survreg`)

- **Loglogistic with a zero lower bound**: `survreg.R:130` lists
  `"Log logisit"`, so R skips the zero-lower-bound conversion for the
  loglogistic family and errors "Invalid survival times for this distribution"
  on interval data with `l = 0`. The port converts, as for the other log-time
  families.
- **t-family deviance**: R's `center` is `rowMeans(y)`, which averages the
  status column into interval rows, and its log likelihood takes the log of a
  negative number (`survreg.distributions.R:147-148`). The port centers on the
  interval and evaluates the likelihood; response residuals of interval rows
  differ, and R's deviance residuals there are NaN ("NaNs produced").
- **Negative deviance terms** from rounding are clamped to 0 before the square
  root; R returns NaN.
- **vcov(complete = FALSE)** keeps every Log(scale) row and drops the aliased
  coefficients. R's `var[keep, keep]` recycles the location-length NA pattern
  over the scale rows: it usually stops in `dimnames<-` (`~ age + age2 +
  strata(grp)` with three strata), and when one scale row is left it returns
  that row labeled "Log(scale)" (`~ one + age + strata(sex)` loses sex = 2's
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
- **Empty scale strata**: AFT keeps scale strata after subset or missing-row
  removal, including zero covariance rows for empty groups. This matches R for
  empty interior strata. A trailing empty level makes R fail assigning scale
  names; R's quantile standard errors and influence residuals infer their column
  count from the largest observed code, omitting trailing scale positions; and
  penalized fitting tries to invert the intercept-only covariance including the
  empty scale's zero row and column. The port keeps every scale position and
  computes the penalized effective sample size from the observed scales only.
- **Interval-censored callback fits**: fits with custom callback distributions
  and interval-censored rows avoid R's unsafe callback C path; the reference
  checks compare them with R's equivalent built-in families.
- **Constant design columns**: AFT leaves constant columns unscaled, allowing
  nonbinary constants and unused Helmert contrasts to reach the rank-aware solver.
  The fit keeps the design, marks singular coefficients as NA and returns the
  identified fitted values and covariance. R's automatic zero-standard-deviation
  scaling can fail with a coefficient-name error or return a nonfinite fit;
  initialized R fits give the same identified result as the port.

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
  callback buffer at `n` entries, while `survregc2.c` also writes every
  interval's upper endpoint, which overflows the heap. The same callback path
  serves custom distributions.
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
  ovarian's `rx`), with a "data length is not a sub-multiple" warning. With
  several predictors, R 3.8-12's `concordancefit(std.err = FALSE)` fails; the
  port supports it.
- `ranks = TRUE` with strata works (R: "number of items to replace is not a
  multiple of replacement length"); several predictors with strata give one
  pooled value per predictor (R: "length of dimnames [1] not equal to array
  extent").
- Zero-weight deaths with `timewt` other than "n": R's `fastkm.c` fills
  `etime` for every death time but counts only positive-weight ones, and
  overruns its arrays. Here, event times with no weighted observations at risk
  contribute zero to the Cox variance numerator, so they do not turn otherwise
  valid `cvar` results into `NaN`.
- `timefix = FALSE` is honored. R's `concordance.formula`, `cord.work` and its
  model dispatcher call `concordancefit` with its default `timefix = TRUE`, so
  near-tied times are merged either way.
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
- Joint concordance aligns influence rows by cluster membership, preserving
  the first model's factor order. Renaming groups or changing their declared
  order leaves covariance unchanged; genuinely different partitions are
  refused. R compares only the number of groups and pairs their influence
  rows by position, which can reverse a covariance's sign. Fitted Cox models
  retain categorical cluster levels, and plain character cluster labels use
  lexical order. See [concordance](concordance.md#fitted-models).
- A missing strata or cluster value raises ("strata contains missing values",
  "cluster contains missing values"); R either errors ("NAs are not allowed in
  subscripted assignments") or, with `std.err = FALSE`, misaligns its counts,
  and its `rowsum` makes a missing cluster a group of its own. A factor-like
  strata value outside its declared levels raises (R's `factor()` makes it NA,
  which then fails as above).
- AFT new-data concordance includes formula offsets in the predictions; see
  [offsets in predictions](#predictions-and-residuals).
- `concordance(fit, newdata)`, including for external linear models, with a row
  missing only its response drops the row from both the frame and the
  predictions; R drops it from one and stops with "x and y are not the same
  length".

### Survival curves

- Formula `cluster()` terms select robust-variance groups and warn about the
  deprecated syntax, following the intended `survfit.formula` branch. R 3.8-12
  reuses model-frame terms that have no special-term metadata, so the branch
  is bypassed and cluster values become extra curve groups. The supplemental
  formula-special reference retains those raw results and compares the port
  with R's explicit `cluster=` argument.
- `summary(fit, times)` of a multi-state curve with a time before its first
  time reports `p0` there. R's `findInterval` index is 0 for that time, so its
  `pstate` has one row fewer than its `time`.
- Without requested times, R's grouped multi-state summaries index the count
  matrices as vectors and lose the state and transition columns; the port
  accumulates each column within its stratum.
- A summary whose first stratum selects no times keeps the fitted vectors of the
  other strata (R can lose them), and an entirely empty summary keeps
  numeric-vector shapes.
- `survfit0` adds the zero influence column only to the curves that get the t0
  row. R's `lapply(x$influence.surv, addcol)` runs over every curve as soon as
  one needs a row, so a curve already at t0 gets one column more than it has
  times. `survfit0` also inserts the unweighted count rows, which R leaves at
  their old row count.
- `aggregate_survfit` drops `std.chaz` with the other components that do not
  collapse. R's list names `"std.cumhaz"`, which no survfit object has, so R
  keeps the uncollapsed `std.chaz`: in a four-prediction example, aggregating to
  two groups changes `surv` from 12 by 4 to 12 by 2 while `std.chaz` stays 12 by
  4, and selecting an aggregate group can return another group's
  cumulative-hazard error.
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
- Counting-process KM curves retain every event time, including events on
  internal intervals of a subject. R 3.8-12's grouped branch omits those
  times from its grid, unlike its single-curve branch, so it can lose events
  and produce incorrect pseudo-values. The bridge's recurrent-event test
  uses a local R reference with that one grid condition corrected and also
  compares separate single-group fits. Neither the installed R namespace nor
  its numerical kernels are changed. Initial interval boundaries without
  events or terminal censoring are omitted, following the 3.8-12 grid rule.
- **Infinite endpoints and near ties**: once R's `aeqSurv` finds a near tie it
  applies finite cutpoints to every endpoint. Positive infinity then becomes the
  final finite time, and a negative-infinity zero index drops a value before R
  recycles or reconstructs columns, which can change censoring times, scramble
  response rows or give a dimension error. The port leaves nonfinite endpoints and
  their statuses in their own rows. Infinite-origin summary tables keep R's
  undefined arithmetic.
- `survfitKM`: default weights and absent `id`/`cluster` are resolved before
  `start_time` filtering (R's lazy default weights can be evaluated after
  filtering, producing the wrong length, and R can access missing `id` or
  `cluster` arguments), and unused levels get empty influence components (some
  R combinations try to set row names on a missing matrix and fail).
- Grouped Aalen–Johansen residual and pseudo-value tables work for a single
  requested time. R 3.8-12 calls `col()` on a three-dimensional residual array
  and errors; the port agrees with the same rows of R's successful multi-time
  table. Excluded multi-time residual arrays retain their state and row
  metadata, which R loses while reinserting missing rows.
- Wholly censored multistate curves return their initial-state probabilities
  with zero-column cumulative-hazard and transition matrices. R 3.8-12's
  `survfitAJ` errors with "subscript out of bounds" in `grabit1` when there
  are no observed transitions, including valid contiguous subject histories.
- `survfit(coxfit, newdata, start.time =)` where `start.time` empties a stratum
  gives each newdata row its own curve; R's `split()` drops the empty stratum
  and hands a row the curve of the row before it. With `id =` and a subject
  whose intervals all end before `start.time`, the subject gets an empty curve
  (R stops in `apply(dt, 2, cumsum)`), and `id` names a newdata column (R
  evaluates it outside newdata: "object 'pid' not found").
- `coxsurv.fit` Kalbfleisch–Prentice curves: after a terminal death in which the
  full weighted risk set dies, R's risk-set subtraction can leave a negative
  rounding residue before a fractional power and return NaN; the port bounds the
  survival increment to [0, 1], giving survival zero.
- Turnbull (interval-censored) curves: `cluster` raises "cluster is not
  supported for interval-censored data" (R hands the full cluster vector to the
  `survfitKM` fits of the pseudo-observations, which reads out of bounds). For
  exact, left- and right-censored rows `start.time` compares their time; R's
  rule compares the placeholder 1 in `y[, 2]` and drops them for any
  `start.time > 1`.

Turnbull curves follow R's EM (`survfitTurnbull.R`: Aitken acceleration every
fifth step, stopping at max |change| < 5e-5). That rule is sensitive to
floating-point order, so on larger data the curves agree with R only to that
tolerance amplified: up to about 0.02 in survival at n = 500 and 0.006 at
n = 2000, with different iteration counts. The fixtures use small data where
this does not show.

### Population calculations

- **Scalar rate mappings**: stock `survexp` passes literal scalar rate entries
  to its native kernel without expanding them for multiple rows, which produced
  incorrect values and a crash. The port expands scalar entries to the row count
  before integration.
- **Repeated requested times**: rate-table `survexp` without a response retains
  duplicate requested times. R 3.8.11 deduplicates the integration grid but
  selects from it using indices in the original requested grid; repeated
  earlier requests can select incorrect rows or produce "subscript out of
  bounds" for grouped fits. Selection here always uses the fitted grid.
- `survexp` with a coxph rate table and an individual method returns values
  when `subset =` or the na.action removes rows (R errors), and a row whose Cox
  response is missing is dropped with the others under `na.omit` (R returns NA
  for it when no other row is removed). `na.exclude` restores removed rows as
  NaN at positions relative to the selected subset.
- **Cox rate models in the R bridge**: individual offsets enter inside the
  exponential, and source rows stay aligned after subset and missing-value
  handling, as in Python. Stratified conditional and Hakulinen cohorts work where
  R 3.8-12 errors while indexing its hazard matrix. A singleton Ederer population
  reports one person at risk instead of R's training-sample count. Baselines
  without events return survival one at the requested times instead of failing
  in interpolation. A null Cox model keeps the population row count when its
  rate-data frame has no columns, where R drops those rows and errors before
  prediction.
- A `ratetable(age = ..., sex = ..., year = ...)` term in a `survexp` or
  `pyears` formula, as older survival code wrote it, is read as the equivalent
  `rmap`. R 3.8 ignores the term's arguments and looks for columns named after
  the rate-table dimensions, so it agrees only when those names match and
  otherwise fails ("object 'age' not found"); `pyears` fails outright.
- `pyears(expect = "pyears")` with a zero rate uses the limit of
  `(1 - exp(-lambda t)) / lambda` at `lambda = 0`, which is `t`; R's C code
  divides by zero.
- A `pyears` formula without terms can return a data frame with person-years
  and counts; R's no-term `data.frame = TRUE` branch fails while assembling it.
- `summary_pyears` of a result without terms returns scalar tables (R's
  `summary.pyears` fails while printing); with `totals=True` it raises where
  R's `pytot` fails with "dim(X) must have a positive length".

### Data preparation

- `tmerge` takes numeric times; R also accepts `Date` times and returns `Date`
  `tstart`/`tstop`. Convert dates to day counts first. `event(censor = )` sets
  the value of censored rows, which R's `tmerge` ignores (a character event
  censors at `""` there).
- `survSplit`: a row with a missing endpoint is episode 2 as in R, keeps its
  status and gets `added = FALSE`; `survsplit.c` leaves its censor flag
  uninitialized, so R's status and `added` there depend on memory contents.
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
- `neardate` with POSIXct or POSIXlt dates: R calls
  `methods::as(value, class(other))` with the two-element class vector and fails
  ("length(class2) == 1L is not TRUE"); passing only the first class also fails
  for some conversions. The R facade uses `as.POSIXct`/`as.POSIXlt` and matches
  the intended numeric epoch result.

### Marginal means

- `yates` on a Cox fit with `strata()`: R builds its population rows with
  `model.matrix(Terms)`, which includes the strata column that the fit's design
  leaves out, so Cmat is misaligned with the coefficients (with an aliased
  coefficient R stops in `qr.resid`). The port uses the fit's own design.
- SAS type III tests on a Cox model: R 3.8-12 leaves the baseline intercept in
  the SGTT hypothesis matrix and can fail with "non-conformable arguments"; the
  port removes it before testing.
- `yates(predict = "survival")`: R's summary keeps the time-0 row of the curve,
  so its `surv` has one row more than its time vector and every row sits one
  time early, and it leaves `cumhaz` at the baseline's (it assigns
  `-log(surv)` to a misspelled local). Here row i is the curve at `time[i]` and
  `cumhaz` is `-log(surv)` per level. `yates_setup` summaries share this
  correction.
- `yates` with no estimable level, or with an `rmean` at or before the first
  time: R 3.8-12 stops inside `gsolve` ("requires numeric/complex
  matrix/vector arguments"); the port returns NA tests, or a test of 0 on 0 df
  (the result R's branch computes). An empty-event survival simulation with no
  estimable coefficients returns zero restricted means and standard errors; R
  fails eigendecomposing a zero-by-zero covariance.
- Reversing the requested factor order can apply the wrong factor levels in R,
  and joint nonlinear global tests fail when R assigns several names to one test
  row; the port keeps the requested order and names a joint global test
  `global`. Numeric term selection includes the last fitted term, which R
  rejects through an exclusive upper bound.
- The R `yates` wrapper accepts a levels matrix for joint settings, empty
  adjustment sets and matrix-valued SAS adjusters, and explicit populations keep
  custom contrasts, where stock R's preparation fails.
- When all observations share a single censored time, the `yates_setup` baseline
  keeps its training sample count; R 3.8-12 removes it while filtering out
  censored times.

### Printing and graphics

- Summary objects keep a conditional cutoff in the units of their scaled times.
  R 3.8-12's report filtering compares scaled times with an unscaled cutoff and
  can drop all valid rows. Requested times before a multi-state conditional start
  are handled by the summary selector, where R can produce mismatched time and
  probability lengths and fail while printing.
- An entirely event-free summary raises a clear error instead of R's failed
  `NULL`-to-matrix conversion.
- An empty `Surv` prints `character(0)`; R 4.5.3 / survival 3.8-12 fails while
  converting it.
- Curve plots: R sets an internal log-axis flag for `fun = "log"` and
  `"logpct"` without updating the plotting call; the port uses the requested log
  probability axis without applying a second logarithm to log state
  probabilities. R's multi-state point method can index a time vector with a
  flattened event matrix; the port marks event times across states. Out-of-range
  NA censor marks left by R after `xmax` are omitted.
- Aalen plots: retaining only the first event can drop R's matrix dimensions, a
  cutoff before that event can include it through `1:0` indexing, and robust
  bands can misalign when an event occurs at zero. The port keeps matrix
  dimensions, returns only the zero origin before the first event and removes a
  duplicate zero origin from every array.

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
- `brier` on a stratified Cox fit returns scores; R 3.8-12 errors in
  `summary.survfit`. `brier(fit, newdata)` on the fit's own data equals
  `brier(fit)` (lung reversed, `~ age + strata(sex)`: 0.2078394407,
  0.2325654809).
- Clustered `aareg` covariance uses event-sorted cluster identifiers. R 3.8-12
  groups sorted test influences with the original cluster order, so its result
  changes when input rows are reordered. Native results match R on event-sorted
  input; the AFT/Aalen report fixtures retain both raw and aligned R outputs.
  Truncated or reweighted summaries also retain robust covariance, computed
  from stored group influences.
- `aareg` on a design with an unused factor level raises "the nmin threshold is
  too high; no Aalen model can be fit"; R 3.8-12 does not return (still running
  after 90 s on lung).
- Penalty searches: a df search with no upper point at an exact boundary returns
  an error where R can return an unusable value, and the shared fitter avoids an
  out-of-bounds access in that case.

## Other deliberate differences

### Versions

- Historical numerical fixtures use survival 3.8-11; the R bridge's CI and
  current response/label references use 3.8-12. `Surv2` and
  `surv2counting` follow 3.8-12: a missing 0/1 status stays NA (the interval is
  then removed by the na.action) and a missing factor outcome is censored
  without 3.8-11's level shift. The multi-state `parsecovar2` and `survfitAJ`
  paths follow 3.8-12 as well.
  With `entry=True`, multistate reporting times include actual subject entries,
  transitions and final exits, omitting censored continuation boundaries.
- `Surv` and `Surv2` retain the censoring factor's first level in `clabel`
  and format it as `:label`, following 3.8-12. Legacy objects without `clabel`
  keep the `+` marker. The old formatting fixture reconstructs that legacy
  metadata; the vector-operation fixtures check current labels against 3.8-12.
- Transition tables in multi-state curves, Cox models and `print_survcheck`
  output use the response's censoring level, e.g. "(censor)" or "(lost)",
  following 3.8-12; printed `survcheck` tables of numeric responses use
  "(censor)". The raw `survcheck` result keeps the column label "(censored)". Curve transformations
  and serialization preserve the label. Historical fixtures adapt only their
  fixed "(censored)" column name; all counts remain checked unchanged.
- `rsurvreg(seed=s)` reproduces R's `set.seed(s)` stream (Mersenne-Twister,
  inversion); without a seed it draws from a clock-seeded generator, since
  there is no R session whose RNG state it could share. Seeds are R's 32-bit
  integers in [-2^31 + 1, 2^31 - 1].

### Values and numerics

- (start, stop] Cox fits (`agfit4.c`): when R moves the center after a tied
  death was added at the same time, it leaves that death on the old center;
  here the deaths of a death time use its final center. Rows with tied entry
  times leave the risk set in stop-time order rather than data order, and
  Efron's j-th terms are formed as `sum - j/d * deaths`. Results agree to
  rounding whenever the center does not move mid death time.
- `kernels.coxph_fit` (and the R bridge's `coxph.fit`, `agreg.fit` and
  `agexact.fit` wrappers) follows `coxph()` rather than the bare fitters: it
  centers the offset, so `agreg.fit`'s info rescale count differs for a large
  offset (0 instead of 5 for heart with offset -750; coefficients and loglik
  agree), and zero-event data get `coxph()`'s skeleton where R's `coxph.fit`
  iterates to coefficient 0 with "Ran out of iterations". The Python
  `r.coxph_fit`, `r.agreg_fit` and `r.agexact_fit` keep raw offsets, as R's
  bare fitters do.
- A Cox fit's `offset` is the offset as given; R's `fit$offset` exists only for
  `x = TRUE` and is the centered offset.
- Linear algebra: R's `solve()` estimates the 1-norm of the inverse with
  `dgecon` (a lower bound); the port computes it exactly, so a matrix right at
  the `eps` boundary that R accepts can be reported singular. The singular
  message reads "LU factorisation (reciprocal condition number = X): system is
  computationally singular" with six significant digits.
- `signif` beyond |x| of about 1e306 (or below 1e-306) in the penalized survreg
  print can differ from R in the last bits.
- Ordered-factor polynomial contrasts use R's rank-limited QR construction.
  Very high degree columns are ill conditioned and sensitive to floating-point
  and BLAS differences, including their signs; exact cross-platform coefficient
  parity is not guaranteed for those columns.
- `aggregate_survfit` mean and sum use compensated arithmetic for ordinary values
  and replay R's sequential extended-precision arithmetic for exceptional groups.
  Ordinary results may differ from R in the final bits, and R platforms with
  another long-double precision can also differ.

### Missing values and newdata

- Fitting with `na.pass` cannot send an unknown event status to a numerical
  routine. Cox, AFT, Aalen, curve, log-rank, survival-check and redistribution
  calls raise `ValueError("missing values in the response")` for such rows. Use
  `na.omit` or `na.exclude` to apply the model frame's row-removal policy.
- Native Cox fitting and prediction methods validate their inputs even when a
  Rust caller constructs or modifies public data fields directly. Row-aligned
  vectors must match the design, statuses must be binary, and follow-up intervals
  require `entry < time`. An invalid interval raises an error rather than producing
  a negative expected count or a survival probability above one. Prediction
  covariates and offsets may still contain NaN where the corresponding R method
  propagates missing values; curve construction requires finite values.
- Native expected-count predictions require a stratum per new row when the model
  has multiple baselines. Without it, selecting the first baseline would be
  ambiguous. Ordinary `survfit(newdata)` can omit strata because it returns a
  separate block for every baseline. Individual covariate paths retain R's
  default of using the first stratum when none is supplied.
- For ordinary Cox and AFT models under `na.pass` (predict's default), a missing
  covariate affects only the predictions that use it. Term predictions retain
  unaffected contributions and standard errors; an unknown offset leaves the
  linear-predictor standard error available. This includes ridge and P-spline
  terms. AFT location and term predictions do not require a known scale stratum.
  Expected-count predictions use follow-up times regardless of a missing
  event indicator. `na.omit` drops incomplete rows, `na.exclude` restores them
  as all-NaN rows, and `na.fail` rejects them.
- A Cox expected-count prediction with an unknown stratum remains NaN here;
  R leaves its initial value at zero and reports survival 1 for that row.
- For the redundant Cox model `~ age + g + strata(g, na.group = TRUE)`, expected
  counts use the estimable coefficients; R propagates its aliased coefficient's
  NA.
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
- Python responses represent R's numeric NA and NaN by the same NaN, so
  `unique_surv` and `duplicated_surv` treat them as a single missing value.
  Empty responses remain empty under repetition, reversal, head and tail; R's
  `1:nrow(x)` in those methods can instead produce an out-of-bounds error.

### Formulas and arguments

- Term labels are the formula's own text; R deparses them (R names
  `I(age*1e-3)` "I(age * 0.001)", `pspline(age, df=4)` "pspline(age, df = 4)",
  and the `pspline(penalty = FALSE)` columns after the deparsed call). They
  agree when the formula is written with R's spacing.
- Python callbacks and penalty constructors cannot recover R's unevaluated
  argument expressions: `r.ridge(..., column_names=...)` supplies complete
  coefficient labels, and callback splines are labeled from the variable name,
  such as `ps(x)3`. Native penalties do not carry R's optional `varname`
  replacement attribute.
- In a comparison `T` and `F` are column names (R falls back to TRUE/FALSE when
  the data has no such column); a bare word in an rmap string is a label
  (`race = "white"`), where R's `rmap = list(race = white)` is a variable
  lookup.
- An ordered comparison (`<`, `<=`, `>`, `>=`) with a plain string operand
  raises when neither operand is a factor; R orders strings by the locale's
  collation. Factor comparisons follow R's ordered and unordered factor methods,
  including warnings and missing values.
- Formula string escapes decode valid UTF-8 text, including paired Unicode
  surrogate escapes. Non-UTF-8 byte strings and unpaired Unicode surrogate
  escapes are explicitly refused; R can retain their raw bytes.
- `cut()` refuses an argument it does not know; R's `cut.default` swallows it.
- Formula expressions accept a fixed set of R functions (see
  [Formula interface](formula-interface.md#r-functions-in-terms)); other calls
  are refused. `pmin()`/`pmax()` refuse character arguments, `c()` refuses
  factors, and `factor(exclude = NULL)` is refused; R accepts all three.
- `ns()`, `bs()`, `poly()` and `nsk()` take a data column, not an expression,
  and `poly()` takes one variable and no `coefs`. `ns()` and `bs()` move interior
  knots that coincide with a boundary knot as R does, without R's warning.
  `scale()` is recomputed on new data, as in R.
- Python `subset` is zero-based and accepts integer positions or a full-length
  Boolean mask; negative and fractional positions are refused. The R bridge uses
  R's indexing rules.
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
- `survreg(offset=)` (a column name or vector) and `survSplit(id=)` with a
  character id are Python extensions; R uses `offset()` in the formula and
  errors when survSplit must add an id to a subset.
- The bare Cox fitters accept only the method labels `"efron"`, `"breslow"` and
  `"exact"`; arbitrary R string labels are refused.
- The bare AFT fitters use explicit callback distributions regardless of their
  display name, also in the R bridge's `survreg.fit`; R's C fitter selects a
  built-in density from the display name. Fractional stratum codes are refused
  instead of being truncated.
- `survpenal_fit` takes zero-based column indices in `pcols` and `assign` and
  returns a zero-based `assign2`; R's are one-based, as are the R bridge's.
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
- The native log-rank functions accept counting-process data; the formula
  interface refuses it, as R's `survdiff` does.
- `aggregate_survfit` callbacks must keep returning one numeric value when called
  on curve values; R checks only the preliminary index calls and can then build an
  incompatible curve shape. An R function name is resolved once when aggregation
  begins; R can resolve it again for a later `apply` pass. Empty time or state
  axes give shaped empty results, where R's `apply` probes callbacks with zeros
  and can fail, and zero data columns and missing grouping values are rejected
  before callbacks run.
- Curve quantiles take a scalar `scale`; R's implicit recycling of several scale
  values and complex scale arithmetic are not represented.
- `yates_setup` refuses full matrices for survival prediction, because R's
  `outer`/`cbind` recycling does not produce one row per predictor.
- Direct Python `pyears` calls accept response vectors and the
  `time`/`start`/`stop`/`event` keywords with rate tables; the R bridge's direct
  vector form still requires formula calls for rate tables. Python direct `rmap`
  vectors have no R expression-symbol names to add to retained models.
- `match_ratetable` rejects duplicate required column names instead of selecting
  one, and does not run arbitrary R rate-table summary callbacks or read R's
  legacy matrix attributes; named columns and native tables supply that
  information.
- `tmerge` with mapping inputs reads only the referenced `data2` columns, so
  unrelated columns may have other lengths; R's data-frame inputs always share a
  length.

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
- Character levels (covariates, strata, ids and cluster labels) are sorted by
  code point ("10" before "9", "B" before "a"); R's `as.factor` uses the
  locale's collation, which agrees for digits but not for letter case.
  Numeric, logical and factor values sort as in R. `neardate` compares Python
  character dates in Unicode code-point order; the R facade uses R's locale
  collation.
- `cch`'s `sc_ids` holds the id values themselves; R's rownames are their
  character form.
- `survfit.coxph` curve names: ids that differ but print alike are kept apart
  with `make.unique` ("0.3", "0.3.1"; R repeats the name); a newdata index with
  repeated or missing labels names the curves 1..m; a 0-based integer index
  names them by its values ("0", "2" where R's row names are "1", "3"); the
  columns of an unstratified newdata matrix are unnamed.
- Python/native quantiles of a stratified `survfit.coxph` object with several
  curves have one row per (stratum, curve), ordered as `summary()`'s table;
  R and the R bridge return a stratum x curve x probability array.
  Python/native quantiles and medians always return a `SurvfitQuantileResult`,
  including single-response and single-probability calls.
  Tolerance-induced interpolation ties use R's averaged indices
  without emitting its "collapsing to unique x values" warning.
- A Turnbull fit reports `cumhaz` and `std.chaz` (R's object has them only after
  `survfit0`).
- `aggregate_survfit` without `by` keeps a data axis of length 1 (the package
  convention); R returns a time x state matrix. Native and Python group labels
  list observed levels only; R's bare factor `newdata` can include unused
  declared levels, which the R bridge keeps. The `newdata` of multi-state Cox
  curves holds the rows actually used; R stores the user's data frame.
- Residuals on multi-state Cox curves raise TypeError; R returns Aalen-Johansen
  residuals of the data that ignore the Cox model.
- Direct Cox curves (`coxsurv_fit`) name individual trajectories by their ids,
  and unused factor levels, zero-column designs and empty trajectories have
  explicit zero-size outputs where R drops dimensions.
- `summary_pyears` returns the tables instead of printing them (R's print
  switches are not taken); for a `data_frame=True` result the restored table
  keeps every level the pyears call used.
- Expected-survival results and summaries, and `pyears` results, do not store R
  call expressions or manually attached `na.action` attributes.
- Native calendar dates reject nonfinite day counts and dates outside the `i32`
  year range; R can keep and print such numeric `Date` values.
- `model_term_names(aareg)` returns the formula's term labels (R's
  `labels.aareg` returns NULL); `model_frame(survreg)` still needs
  `model=TRUE`. The port's `model_frame` output contains raw columns, without
  R's evaluated-column schema and `terms` attribute.
- A `cox.zph` function transform is labeled by its name, or "user" without
  one, not by R's deparsed expression.
- The penalized Cox summary keeps a scalar `loglik` plus `null_loglik`;
  `anova.coxph.penal`, which R does not register, is not ported (anova of one
  penalized model follows R's reachable `anova.coxph`). Residuals of a null Cox
  model other than martingale and deviance raise "... residuals are not defined
  for a null model".
- The typed `SurvpenalFit` returns `penalty` as `c(0, P)` for every fit (R: a
  scalar for a sparse fit) and dense column means for sparse fits (R includes
  the group-code column). Penalty histories keep their column schema even when
  R omits names for an empty fixed-gamma history.
- Concordance rank tables contain event rows without R's model-frame row labels,
  and new-data stratum counts keep the fitted labels instead of R's integer codes.
- `Surv` and `Surv2` reject arithmetic, comparisons and reductions as R does, and
  Python additionally rejects NumPy ufuncs and implicit array conversion. Use
  `left.equals(right)` for structural comparison; response objects are
  unhashable and have no truth value.
- Rate tables canonicalize their axis titles in `dimid`, without R's distinction
  between unnamed axes and a separate `dimid` attribute.
- R exposes `n.risk` only for Ederer Cox cohorts; the native and Python
  expected-survival results also supply risk counts for the Hakulinen and
  conditional methods.
- Yates factorial and SAS populations keep compound strata as evaluated factors;
  their labels are not parsed back into the original source columns.
- Repeated stratum labels in selected multi-state Cox curves gain suffixes such
  as `b`, `b.1`, so the Python dictionary keeps every selected time block.
- Print functions return objects (`ModelPrint`, `SurvfitPrint`,
  `SurvivalTablePrint`, `ResponsePrint`, `RateTablePrint`) instead of printing and
  returning the fit invisibly. They omit R call expressions, which Python fits do
  not store, strip trailing spaces and read no global print options. Curve and
  expected-curve output does not reconstruct omission notices, and expected-curve
  output omits R's rate-table prose.
- `print_survreg_penal` prints the Call block as `survreg(formula = ...)`; R
  `dput()`s the whole call. Its `terms=TRUE` Wald rows keep R's p-value on 1 df
  while the DF column shows the term's df, as `summary_coxph_penal` does.
  Arbitrary R `printfun` closures are not retained or executed; callback
  penalties use built-in coefficient or sparse Wald rows.
- `print_survConcordance` is an explicit function because R 3.8-12 contains
  `print.survConcordance` but no longer registers it.
- `plotting.cox_zph_plot_data` returns every selected nonsingular term and works
  with standard errors disabled, unlike R's `plot = FALSE` branch.
  `cumhaz=True` on expected-survival curves is an extension to R's graphics.
  Cox survival results use the common curve renderer; R's obsolete
  `survfit.coxph` class has no separate Python representation. Matplotlib
  controls device layout, ticks, padding and fonts; base R graphics parameters
  are not emulated.

### Messages

- Formula literals retain R's values and types but omit its decimal or
  out-of-range `L`-suffix parser warnings (for example, `10.0L`, `1.5L` or
  `2147483648L`). The fixture records formula-parsing warnings separately;
  integer arithmetic itself emits R's overflow warning.
- Formula domain warnings name the expression, such as `NaNs produced in
  log(off)`, where R says `NaNs produced`. R's additional warning before AFT
  `model.matrix` rejects a numeric replacement for an ordinary factor is not
  reproduced.
- Expected-survival summaries use R's averaged interpolation indices for tied
  source times without emitting R's tie-collapse warning, as curve quantiles do.
- A `match.arg` miss reads `'arg' should be one of "response", "link", ...` with
  ASCII quotes; R in a UTF-8 locale prints curly quotes.
- tmerge's not-found message adds "in data2" to R's "object 'x' not found"; a
  missing cumevent increment at an event raises a dedicated message (R: "NAs
  are not allowed in subscripted assignments"); a logical tdc default is
  inserted as given (R's assignment turns the column numeric).
- Residuals and pseudo values of a KM fit whose `start.time` empties a stratum
  refuse with survfitAJ's "start.time has removed all the observations from at
  least one curve" (R: "strata k not matched").
- A Student-t bare AFT fit without `parms` raises a clear error; R's bare fitter
  fails with an unhelpful length-zero error.
- For an AFT `model.matrix` call on a frame without a `terms` attribute and
  without raw `g`, with the formula `age + strata(g) + strata(g):flag`, R fails
  with "number of variables != number of variable names"; the port reports the
  absent `g`. A missing raw Cox strata variable likewise raises the port's own
  missing-variable error.
- `untangle_specials` reports a missing factor matrix when removing all terms
  leaves a registered special; R errors.
- `match_ratetable` on empty input returns empty positions without R's warnings
  about extrema of empty vectors.

### R bridge (`r/survivalr`)

- Residual and pseudo-value arrays preserve character dimension labels, but
  names attached to the dimension-label vectors themselves are dropped during
  id conversion.
- The bridge follows 3.8-12 response semantics. `Surv2` requires logical,
  numeric or factor statuses; missing binary statuses remain missing, and
  factor censor labels survive construction. `Surv` conversion preserves
  coded status columns and censor labels directly, without reconstructing
  per-row factor strings. Legacy responses lacking a censor label retain
  their `+` formatting when converted. Arbitrary R input attributes are not
  retained by Python response objects.
- A penalized survreg fit keeps the bridge's `survival_py_survreg` class rather
  than R's `c("survreg.penal", "survreg")`, so survival's own methods do not
  dispatch on it.
- The `summary.survfitms` list of `survfit.coxphms` curves carries time, the
  counts, pstate, cumhaz, states, table, rmean.endtime, strata and newdata, not
  R's n, n.id, p0, transitions or call.
- `model.matrix` term assignments are integers; R's strata term-number
  adjustment sometimes promotes this attribute to double.
- Grouped Kaplan–Meier curves use bare group labels (`a`, `b`) where R uses
  `g=a`, `g=b`, including in curve quantiles.
- The R `coxsurv.fit` list form keeps R's quirk of naming individual
  trajectories by the training stratum labels.
- AFT distribution queries forward Python `RuntimeWarning` and `UserWarning` as
  R warnings; other callback warning categories go to Python's warning hook.
- `survivalr::yates` consumes R's global random stream and ignores
  `options$seed`, while Python's `r.yates` uses a per-call seed. The R wrapper
  does not support `trend`, stratified survival prediction, multi-state Cox
  models or functions passed as `predict`.
- Saved models embed Python pickle state tied to the installed package; they are
  not a portable archival format, and R's serialization version 2 is refused.

## R behavior reproduced deliberately

The port reproduces these R behaviors on purpose. Some look surprising, but they
are not port defects.

Formulas and model frames:

- Cox constructs the full model matrix before dropping the strata main effects of
  a strata interaction, while AFT removes those main effects before assigning
  contrasts; factor order and column labels follow each. Cox curves for these
  models require `newdata`.
- Each distinct `offset()` contributes once even if it appears in a removed term,
  as in R's `terms()`. NaN created by arithmetic inside `strata()` forms a
  distinct `NaN` level, following R's factor conversion.
- A supplied evaluated model frame bypasses `model.frame`: missing rows remain
  under every global action, supplied factor levels can differ from the fitted
  levels, and offsets or unused variables do not remove rows. Because the stock
  Cox standalone-strata branch misspells `contrasts.arg`, supplied frame contrasts
  apply in that branch, while other Cox and AFT branches use the fitted contrasts.
  (For fitted and new-data matrices the port corrects this typo; see
  [Formulas and model matrices](#formulas-and-model-matrices).)
- `model.matrix.survreg` renumbers assignments after removing strata columns, so
  `attrassign` with the fitted term labels names a later column after a strata
  term. With exactly one term, `untangle.specials` uses the summed factor code as
  the endpoint of `seq(ff)`, so an interaction-only special with code 2 gives
  indices `[1, 2]`.
- Factor levels absent before `strata()` is evaluated are dropped. Several
  separate `strata()` terms are recombined after row selection, keeping only
  observed combinations, while a single `strata(a, b)` term keeps its combinations
  from before selection.
- Automatic frailty selection follows R's reevaluated constructor, so a sparse
  term can become dense on small new data, while a fitted dense term that switches
  to sparse codes fails because its factor contrasts no longer apply. Explicit
  `na.action` arguments in `...` are ignored by new-data `model.matrix`, which
  uses `options("na.action")`.
- Variables removed by formula subtraction stay in the model frame and take part
  in training-row omission, while Cox and AFT cluster extraction rebuilds the
  formula and drops them. R's checks for multiple clusters and missing interaction
  margins are kept.

Predictions and residuals:

- AFT quantiles drop single-row and single-column dimensions before restoring
  excluded rows, so one retained row with three quantiles and three excluded rows
  gives a vector of length six. Fixed-scale errors repeat the retained row's name.
- R leaves some prediction names absent: an unstratified Cox stored linear
  predictor without errors, the errors of training expected counts, new-data
  expected counts, AFT stored response and link fits, and some untransformed
  quantile errors.
- An explicit `terms = NULL` selects no Cox term columns but all AFT columns.
- New-data linear, risk and term predictions from frailty-only Cox fits do not
  evaluate the frailty variable, so its missing values remove no rows.
- Schoenfeld and scaled Schoenfeld residuals ignore `collapse`.
- An aliased AFT coefficient makes new-data and term predictions that use it
  missing. AFT prediction types other than quantiles ignore `p`.

Low-level fitters:

- `coxph.fit` and `agreg.fit` use Efron for `method = "efron"` and Breslow for
  every other accepted label, including `"exact"`. `agexact.fit` always uses the exact
  likelihood, requires unit weights, returns a one-column matrix of linear
  predictors, labels its method `"coxph"` and has no class.
- `survfitKM`'s old `type` argument overrides `stype` and `ctype`, including
  otherwise invalid values; `time0` is accepted and unused; the unweighted
  `counts` table appears according to R's relative comparison of weights with one.
- `survreg.fit` uses the numeric columns of a `Surv` without recoding, so a
  left-censored `Surv` is fitted with 0/1 status; `dist = "weibull"` is rejected
  because its entry has no density; with `nstrat = 1` the stratum codes are
  ignored; variance names are dropped when covariate rescaling is undone; the
  score keeps the internal covariate scaling; `assign` is unused.
- `survpenal.fit` and `r.survpenal_fit` leave `df2` unpopulated and return a
  scalar penalty for sparse fits (the native result always has two elements).
- The t family's `n.eff` evaluates `variance(sigma^2)` as R does and can be
  negative (-111.527 on log lung time); only AIC-method psplines with
  `dist = "t"` use it.
- `coxsurv.fit` accepts and ignores `cluster`, `position` and `oldid`, and
  `survfitcoxph.fit` ignores `vartype`.
- `rsurvreg(n, ...)` draws as many uniforms as R's `runif` count rules give before
  recycling location and scale, so `rsurvreg(2, mean=1:3)` draws two uniforms and
  returns three values with a recycling warning. Distribution queries recycle
  location and scale separately at each arithmetic stage, as R does.
- `rep_surv` follows R's first-element warnings, nonfinite defaults, fractional
  truncation and argument precedence.
- Frailty penalties follow R's initial values: a gamma EM search ignores an `init`
  vector longer than one value, Gaussian REML keeps a vector, and AIC searches
  start from R's `c(0.1, 1)`. Ridge scaling uses the sample variance without
  NumPy arithmetic warnings, as R's `var()` does for very large or subnormal
  inputs.

Curves:

- Without `FUN`, ungrouped ordinary curves are aggregated with `rowMeans`, while
  an explicit `"mean"` uses R's residual-refined mean. Aggregation callbacks first
  receive each group's one-based prediction-row indices, and extra arguments are
  ignored.
- `survfit.matrix` defaults to the discrete method for both KM and Cox input; it
  caps the diagonal departure probability at one without rescaling off-diagonal
  increments, reuses the previous event count stepwise, and takes a state's risk
  count from the last departing transition in column-major order.
- Tied `ctype = 2` residual and pseudo-value diagnostics use R's approximation.
- With the default tolerance, an entirely censored curve has an undefined quantile
  even at probability zero (R's terminal-flat rule); infinite tolerances keep R's
  undefined results and zero-probability origin rules.
- Counting curves whose inserted `start.time` origin makes the times unsorted
  refuse requested summaries. Ordinary KM quantiles keep the zero origin even when
  fitting starts later. Multi-state quantiles and medians are refused.

Printing and graphics:

- `print_survfitms` marks the last column with an asterisk even when it is
  `se(rmean)`; one-row strata blocks use the named-vector layout; several Cox
  prediction curves have no confidence columns; `print_survexp(naprint=False)`
  drops rows with at least the number of columns minus two missing cells.
- A zero frailty variance gives R's NaN Wald test, in Cox and penalized AFT
  summaries alike.
- `print_coxph_null` ignores the precision argument and uses seven digits.
  `print_summary_coxph_penal` ignores `signif_stars` and does not truncate
  confidence-interval labels. Multi-state Cox output omits `se(coef)`.
- `print_survreg` applies an explicit `digits` to the coefficient vector and
  multiple named scales only. `print_aareg` and `print_summary_aareg` round to
  three significant figures before seven-digit formatting, and the direct printer
  keeps the fit's original test label in its footer after an override.
- The case-cohort summary printer ignores `digits` and keeps R's literal
  `x$method,` heading and single-coefficient row labels `[1,]` and `Value`.
  Concordance counts are rounded to two decimals before seven-digit formatting.
- Named `Surv` vectors align names and values to the right even with
  `right=False`. Rate-table vectors print every entry when only one would be
  omitted; zero-limit matrix displays keep row labels; named Unicode axis titles
  use byte-counted padding.
- Plotting a `Surv2` response, and line or point overlays of raw `Surv`
  responses, are refused with R's message.
- `lines_survexp` joins expected survival with straight lines. Confidence bars
  from `conf_times` use the next fitted limit, while `mark_time` marks use the
  preceding estimate. Zero values on a log probability axis are drawn at 80% of
  the smallest positive value.
- Aalen plot variances use squared influences or squared coefficient increments,
  each squared individually at tied times, and output starts at zero. Cox
  diagnostic plot knots use the event times and the prediction grid, so `nsmo`
  can change the curve.

Population, data preparation and marginal means:

- `finegray` multiplies user weights by source position within the stratum,
  indexing the full weight vector, so a stratified result's `(weights)` column
  need not equal its initial `fgwt`.
- Rate-table `survexp` keeps R's zero cohort survival and unit conditional
  survival for an empty sparse group.
- `model=True` takes precedence over `x` and `y` in `pyears` and `survexp`. Cox
  rate models with a sparse frailty refuse new cohort curves; individual
  predictions use zero frailty for new rows.
- A matrix-valued continuous `survobrien` term uses its first column, and a custom
  transformation is first probed with ten values before it is called once per
  column of each risk set.
- With a two-column numeric `pyears` response, the columns are time and event
  without a rate table, and start and stop (without events) with one.
- `yates` weights linear estimates for the data population but not nonlinear
  averages, and omits model offsets. R's default `yates_setup` method reads `type`
  while its Cox and GLM methods read `predict`; GLM `terms` and Cox
  `expected`/`terms` keep R's explicit errors.

## Reference differences retained deliberately

The Python fixture suite (`python/tests/test_r_fixtures.py`) has **37 expected
numerical differences**, with no missing-feature or API-error exemptions. None
of the historical reference JSON files were changed. The differences are:

| Checks | Fixture family | Reason |
| ---: | --- | --- |
| 7 | Aalen regression, veteran data | The reference continues through rank-deficient risk sets and contains missing coefficients and values around 10¹⁵. The port truncates at the last nonsingular risk set and returns finite estimates. An earlier singular risk set (common with (start, stop] data, such as `heart` before the first transplant) gives an aliased covariate no step and no time weight, R's `qr.coef` with the missing value as zero; R's own step there comes from rounding in its risk sums. |
| 6 | Cox concordance | Exact linear-predictor ties depend on floating-point/BLAS rounding. The reference and this platform classify a few tied pairs differently. |
| 1 | Exact counting-process Cox deviance residuals | The reference's unclassed fit dispatch returns martingale residuals for a deviance request. The port computes deviance residuals. |
| 18 | Interval-censored AFT residual derivatives | The reference has opposite signs in some scale derivatives. The port's likelihood derivatives are checked against finite differences for every censoring type and distribution family. |
| 4 | AFT prediction with a new-data offset | R's `predict.survreg` drops new-data offsets on the linear-predictor/response path; only its stratified quantile path extracts them. The port includes them in the linear predictor and transformed predictions. |
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

## Reference limitations

Features R itself does not implement stay refused with R's message: anova on
multi-state fits; `predict.coxphms` types expected, survival and terms and
`reference = "strata"`; `basehaz`, `royston` and `yates` on multi-state fits;
`survexp`/`pyears` with a multi-state rate table ("Invalid rate table").
Multistate Cox covariate paths also raise R's unsupported-path error. Offsets
in formula lists raise an explicit error directing callers to the `offset`
argument; stock R 3.8-12 fails with "termmatch failure 1" for those formulas.
Stratified and multi-state `yates_setup` survival predictions are unsupported,
as in R.

`pyears(ratetable=cox_fit)` also remains explicitly unsupported. R 3.8-12
recognizes the fit initially but then tries to coerce it to a numeric rate
table and fails with "'list' object cannot be coerced to type 'double'".
`survexp(ratetable=cox_fit)` is supported.

## Reproduce validation

```sh
cargo test --lib --tests --no-default-features --offline
PYO3_PYTHON=$PWD/.venv/bin/python PATH=$PWD/.venv/bin:$PATH \
  PYTHONPATH=$("$PWD/.venv/bin/python" -c 'import site; print(site.getsitepackages()[0])') \
  cargo test --lib --tests --all-features --offline
PYTHONPATH=python .venv/bin/python -m pytest python/tests -q
python3 scripts/generate_binding_manifest.py --check
PYTHONPATH=python .venv/bin/python scripts/generate_stubs.py --check
```

The normal fixture run includes the documented expected differences. Add
`--runxfail` to display their full discrepancies. R itself is needed to regenerate
the reference data; see [the fixture workflow](../test/r/README.md). Independent
current-stock references are generated by the `scripts/generate_*_reference.R`
scripts with R 4.5.3 and survival 3.8-12; each writes a JSON fixture under
`python/tests/fixtures/` that the corresponding tests read. Where a reference
records a stock R failure or defective output, it keeps the raw R result beside
the corrected oracle. CI regenerates the current-stock references and compares
them at `rtol=1e-9`, `atol=1e-12`; some AFT and ordered-contrast references use
their own documented tolerances. The skipped Python tests are cases where stock R
could not compute a reference.
