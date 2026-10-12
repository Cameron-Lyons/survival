# Formula interface

`survival.r` is the R-style interface to the package; `survival.r_api` re-exports
it, and the R bridge package `survivalr` drives it through reticulate. Formula
strings are parsed by a restricted parser: formula options are read as literals,
and Python or R code is never evaluated. This page describes formulas, survival
responses, model frames and model matrices. Predictions are described in
[Predictions and residuals](prediction.md); every known difference from R is
listed in [R compatibility](r-compatibility.md).

```python
from survival import datasets, r

lung = datasets.load_lung()
fit = r.coxph("Surv(time, status) ~ pspline(age, df=4) + sex", lung)
prediction = r.predict(fit, {"age": [50, 70], "sex": [1, 2]}, type="lp")
effective_df = r.degrees_freedom(fit)

cox = r.coxph("Surv(time, status) ~ age + sex", lung, model=True)
expected = r.survexp("~ sex", lung, ratetable=cox, times=[0, 100, 365])
```

## Formula terms

Supported terms include `+`, `.` expansion, `-` exclusions, backtick-quoted
names, `:`, `*`, `%in%` and `/`, powers of term groups, `factor()`,
`as.factor()`, `strata()`, `cluster()`, `offset()`, `tt()`, `cut()`, `tcut()`,
the transforms `log()`, `sqrt()` and `exp()`, `I()`/`identity()` arithmetic and
comparisons, and the R functions listed under
[R functions in terms](#r-functions-in-terms). Categorical variables use
treatment contrasts unless an ordered factor or an explicit contrast applies.
Term and coefficient labels are R's deparsed expressions, whatever the spacing of
the formula: `I(age>60)` is labelled `I(age > 60)TRUE` and `I(age*1e-3)` is
`I(age * 0.001)`.

Interaction identity ignores factor order: `age:sex` and `sex:age` name the same
term, so subtraction and duplicate removal follow R. Grouped nesting such as
`(a + b)/c` produces `a + b + a:b:c`; powers bind before interactions and
products, and intercept changes inside parentheses are retained. Categorical
contrasts account for margins contained in earlier interactions, even when those
margins are not explicit main effects.

`offset()` and `cluster()` can appear inside products, nesting and powers. Each
distinct offset contributes once even if it appears in a removed term, as in R's
`terms()`. A cluster main term supplies the robust-variance groups; its
interaction columns remain in the design. R's checks for multiple clusters and
missing interaction margins are kept. Cluster identifiers may be arithmetic
expressions, comparisons, numeric transforms or factor conversions, such as
`cluster(site + id)` and `cluster(group > 5)`; their evaluated values determine
robust-variance groups in Cox and AFT fits and in clustered survival curves.
Missing values in the source, and NaNs created by these expressions, take part in
row omission, including when the cluster main effect has been removed from the
coefficient design.

Variables removed by subtraction remain in the model frame and take part in
training-row omission. For new-data predictions, `na.pass` allows a missing
unused variable; `na.omit`, `na.exclude` and `na.fail` still check it. Cox and
AFT cluster extraction rebuilds the formula and drops such unused variables, as
R does.

Cox and AFT formulas support stratum-specific effects such as `age * strata(sex)`,
`age:strata(sex)` and interactions with compound strata or categorical
covariates. The strata still determine the Cox baseline hazards or the AFT
scales. Cox constructs the full model matrix before dropping the strata main
effects; AFT removes those main effects before assigning contrasts. The port
keeps this difference, including factor order and column labels. Cox survival
curves for these models require `newdata`, as in R.

Character columns are factors, as in R's `model.matrix`: a column of `"0"`,
`"1"`, `"2"` gives two treatment columns, not a slope, and string levels sort as
strings (`"10"` before `"9"`). New data matches string levels through R's
`as.character`, so numeric `0`/`1` predicts for a factor with levels `"0"` and
`"1"`. `cut()` is a factor term in every formula; data-dependent breaks such as
`cut(age, 3)` are computed before subsetting and missing-row removal, as
`model.frame` does. `survival.r.Factor(values, levels=None, ordered=False)` is
R's `factor()`: a `list` of labels that carries `categories` (and `levels`), so
it declares a level order without pandas; the bundled data sets return their
factor columns this way. Cox and AFT fits record R's `xlevels` and `contrasts`
(Cox also lists the `strata()` levels; a plain logical column has a contrast but
no levels).

## R functions in terms

These R functions can be used as terms, inside other calls and in `survfit`,
`survdiff` and `pyears` groupings, with R's argument matching (exact names,
unique prefixes, then positions) and R's result types:

- `factor(x, levels, labels, exclude, ordered)`, `as.factor()`, `relevel(x, ref)`
  and `interaction(..., drop, sep, lex.order)`. Levels given to `factor()` are
  matched through `as.character`, a single label is numbered (`labels = "e"`
  gives `e1`, `e2`, …), and `interaction()` varies the first factor fastest
  unless `lex.order = TRUE`.
- `ifelse(test, yes, no)`, whose result takes the type of the branches it uses
  (a character branch makes a factor term; a factor branch contributes its codes).
- `pmin()` and `pmax()` with `na.rm`; `round(x, digits)` (halves to even),
  `signif()`, `floor()`, `ceiling()`, `trunc()`, `abs()`, `sign()`, `log(x, base)`,
  `log1p()`, `expm1()`, `log2()` and `log10()`, with R's `NaNs produced` warning.
- `as.character()`, `as.integer()`, `as.logical()`, `is.na()`, `c()` of
  constants and `%in%` inside a call (`I(ph.ecog %in% c(2, 3))`). Outside calls
  `%in%` remains the nesting operator.

A character result is a factor with sorted levels, so
`ifelse(age > 60, "old", "young")` gives the column
`ifelse(age > 60, "old", "young")young`. New data are evaluated with the same
expressions and coded with the fitted levels (`xlevels`).

```python
fit = r.coxph(
    "Surv(time, status) ~ pmin(age, 70) + relevel(factor(ph.ecog), ref = '1')"
    " + ifelse(wt.loss > 10, 'loss', 'stable')",
    lung,
)
```

## Typed vector expressions

Expressions inside `I()`, `identity()`, `as.numeric()`, `offset()` and the other
supported calls follow R's precedence: powers bind before unary signs, then
multiplication and division, addition and subtraction, comparisons, logical
negation, `&` and `|`. Thus `!z > 0` means `!(z > 0)` and `-z^2` means `-(z^2)`.
Outside expression calls, formula operators describe model terms.

```python
fit = r.coxph(
    "Surv(time, status) ~ x + I(z > 0 & x <= 1) + offset(z > 0)",
    data,
    model=True,
)
```

- Logical operators use R's three-valued rules: `FALSE & NA` is `FALSE` and
  `TRUE | NA` is `TRUE`; the other uncertain cases are missing.
- `I()` and `identity()` preserve the input type and factor levels through nested
  calls. Logical terms use treatment contrasts with a `TRUE` column; arithmetic
  and `as.numeric()` produce numeric columns; numeric conversion of a factor uses
  its one-based level codes. Logical offsets keep their logical model-frame
  values and contrast metadata, while the fitter receives zero and one.
- Integer and logical operands keep integer results for unary signs, `+`, `-`
  and `*`. Values outside R's integer range become `NA` with R's overflow
  warning, so overflow takes part in row omission. A double operand promotes to
  double; `/`, `^` and `as.numeric()` produce doubles. The integer or double
  type survives empty and entirely missing vectors.
- Factor comparisons follow R's factor methods. Equality compares labels and
  requires matching level sets when both operands are factors; level order may
  differ. Ordered factors support all six comparisons in declared level order,
  and ordering two ordered factors requires the same levels in the same order. A
  plain operand is matched to the levels, with unknown labels becoming missing.
  Ordering an unordered factor warns and returns missing values; mixed ordered
  and unordered factors keep R's incompatible-method warning and integer-code
  comparison. `factor()` keeps an input factor's ordering and drops unused levels.
- Quoted literals decode R's control escapes, octal and hexadecimal byte escapes,
  and `\u`/`\U` Unicode escapes, including braced forms and paired surrogates.
  Invalid escapes, embedded NULs, mixed Unicode and byte escapes, and Unicode
  escapes inside backtick names are rejected.

Ordered pandas categoricals and R factors use R's default `contr.poly` coding:
columns `.L`, `.Q`, `.C`, then `^4`, `^5` and so on. Scores are the declared level
positions, independent of numeric labels or observed counts. The basis follows
R's limited-pivot LINPACK QR, applying only its numerically independent
Householder reflections, and supports up to 95 levels. Explicit R contrast
matrices, named contrast attributes and configured R contrast defaults take
precedence and keep their fitted metadata. Bare factors and identities keep
unused levels; `factor()` drops them before subsetting and omission. Fitted
matrices, predictions and pickled models keep the training basis even when new
data changes the category order or supplies plain character values.

Each model-frame variable is evaluated once, before subsetting and missing-row
removal, and its values and type metadata are then selected together. Removed
terms still evaluate their variables and signal their domain warnings, and a
warning is not lost when another missing value removes the same row. Omission
inspects the evaluated variables rather than the raw sources: `I(z^0 * w)` keeps
a row with missing `z` because `NA^0` is one, unless another term reads `z`.
Source `NA` and genuine `NaN` stay distinct, and an overflowing `exp()` gives
`Inf`, as in R. Prediction reuses the fitted levels and contrasts, including
after copying or saving a model; empty and entirely missing prediction vectors
keep the fitted logical or factor type.

## Strata

`strata()` arguments may be comparisons, arithmetic, numeric transforms, factors,
`cut()` calls or nested strata. Named arguments set grouping labels; `shortlabel`,
`sep` and `na.group` accept literal options. For example, `strata(sex == 1)`
groups by a comparison and `strata(g, na.group = TRUE)` keeps missing values as a
group. NaN created by arithmetic forms a distinct `NaN` level, following R's
factor conversion; NaN in the input represents R's `NA`. Evaluated strata are
reused within a model frame and subsetted with its rows, so data-dependent cut
points are determined before subsetting.

The Python `strata()` function returns a `StrataFactor`. Its `categories` give
the level order as an immutable snapshot, and it takes part in the same factor
protocol as pandas categoricals and factors received from R. Reusing it in
`Surv`, another `strata()` call or a model column therefore keeps its categories:
numeric-looking labels stay categorical rather than becoming a continuous
predictor. The literal label `"NA"`, including one produced by
`strata(..., na_group=True, shortlabel=True)`, is a valid event-state label and
stays distinct from a missing value; blank state labels are invalid.

Several separate `strata()` terms are recombined after row selection, so only
observed combinations remain; a single `strata(a, b)` term keeps its combinations
from before selection. Cox fits and curves use contiguous observed stratum codes.
AFT scale strata are described [below](#aft-scale-strata).

## Subsets and missing data

Formula calls accept `subset=` and R's `na_action`: `"na.omit"` (the default),
`"na.exclude"` (residuals and predictions padded with NaN at removed rows),
`"na.fail"` and `"na.pass"`. The action applies across formula columns and
row-aligned arguments such as `weights`, `offset` and `strata`.

Python `subset` is zero-based and takes integer positions or a full-length
Boolean mask; negative and floating-point positions are rejected. R's
`subset = c(1L, 3L)` corresponds to Python's `subset=[0, 2]`. The R bridge
evaluates `subset` in the data and formula environment and translates R's
selection: positive, negative and fractional numeric selectors, repeated and
reordered rows, recycled logical selectors and row names. Zero entries select
nothing; missing indices and unmatched row names create missing rows that reach
the missing-data action. Mixed positive and negative selectors and out-of-bounds
positions raise R's indexing errors, and an empty selection raises the no-rows
error. This applies to `coxph`, `survreg`, formula and direct-response `survfit`,
`survdiff`, `rttright`, `survcondense`, formula `concordance`, `survConcordance`
and direct-input `pyears`; the other wrappers use R's `model.frame` selection.
Formula strings use the calling environment for subset expressions; R formula
objects use their own environment, with data columns taking precedence. R formula
evaluation without a data argument has not been checked for every function.

The survival response is constructed before rows are selected or omitted, and the
normalized response determines which rows are missing. With status codes
`[1, 2, 1, 2]`, removing the second and fourth rows leaves two censored
observations. Invalid status codes become missing with the constructor's warning;
reversed counting intervals and invalid interval endpoints follow the same
constructor. An unused interval endpoint does not remove a row unless a
right-hand-side term reads that column. Warnings point to the first caller
outside the package and are emitted once per response evaluation. Matrix
covariates keep their column dimension when a missing row is inserted, and
factor metadata, aligned arguments and evaluated strata follow the same rows.

Under `na.pass`, fitting cannot send an unknown event status to a numerical
routine: Cox, AFT, Aalen, curve, log-rank, survival-check and redistribution
calls raise `ValueError("missing values in the response")`. Use `na.omit` or
`na.exclude` instead.

## Survival responses

`Surv(time, event)`, `Surv(start, stop, event)` and the interval forms accept R's
named aliases `time=`, `time1=`, `start=`, `time2=`, `stop=`, `event=` and
`status=`. As in R, named arguments are matched first and the positional ones
fill the remaining slots, so `Surv(time, event = status)` works; `origin` may
hold one value per observation. Numeric 1/2 status codes are recoded as in R; logical arrays, masked
logical values and nullable pandas logical columns use logical coding. A
factor-valued event gives a multi-state response. Declared factor levels are
kept in their declared order, including unused levels: for values
`[10, 2, NA, 30, 2, 10]` with levels `[30, 10, 2, 99]`, the censoring label is
`"30"`, the states are `("10", "2", "99")` and the codes are
`(1, 2, None, 0, 2, 1)`, as in R. `factor()` and `as.factor()` inside a formula
response create a categorical status; `as.factor` keeps unused levels and
`factor` drops them.

`Surv` and `Surv2` keep the censoring factor's first level in `clabel` and format
it as `:label`; objects without `clabel` keep the `+` marker. A one-shot event
input is consumed once, and the input values and categories are never modified.
Plain numeric and logical arrays keep their dtype during validation. In the R
bridge, native model frames use the same constructor: Date-valued times are
rejected as in R, `difftime` inputs, named arguments and factor levels are kept,
and constructor diagnostics are R warnings. `survcheck` reports problem rows by
their original observation numbers.

Row operations are available from `survival.r` and the package root:
`concat_surv`, `rep_surv`, `rev_surv`, `unique_surv`, `duplicated_surv`,
`transpose_surv`, `levels_surv`, `head_surv` and `tail_surv`. Concatenation
requires matching censoring types and multi-state levels. Row operations keep the
normalized status codes, state levels and censoring label, and never recode a
subset containing only status 2 as an ordinary 1/2 response.

- `rep_surv` follows `rep.Surv`. `each` first repeats rows; a scalar `times`
  then repeats the whole sequence and a vector `times` gives a count for each
  entry; a finite `length_out` (`length.out`) takes precedence over `times`.
  Counts truncate toward zero before the negativity check, so `-0.5` gives zero
  copies and `-1` is invalid. `each` and `length_out` use the first element of a
  vector with R's warning. A missing `each` means one, and a missing `length_out`
  leaves `times` in effect; Python `None` and nullable pandas missing scalars count
  as missing. Numeric character controls are accepted; an unconvertible `each` or
  `length_out` warns and is treated as missing, while an invalid `times` raises R's
  error. `length_out` is validated before `each`. Length-limited repetition builds
  only the required rows.
- `unique_surv` and `duplicated_surv` accept `from_last` (`fromLast`) and compare
  complete rows, including missing values, in expected linear time.
- `as_character_surv` returns labels without right padding; `format_surv` pads
  them to a common width. Both keep the common numeric precision and leading
  padding of the time columns. `transpose_surv` returns a plain matrix with one row
  per response column; `levels_surv` returns the event-state names or `None`.

`Surv` and `Surv2` reject arithmetic, comparisons (including `==` and `!=`),
logical operations, reductions, NumPy ufuncs and implicit array conversion with
`TypeError("Invalid operation on a survival time")`, as R's `Math`, `Ops` and
`Summary` groups do. Use `left.equals(right)` to compare response columns and
metadata (matching missing values compare equal); response objects are
unhashable, and truth-value tests are invalid. `copy`, `deepcopy`, pickle, `len`
and survival-aware `median`/`quantile` remain available; extract
`response.as_matrix()` or a single column for numeric work; `Surv2.as_matrix()`
returns the time and normalized status.

## Input containers

Formula data may be a mapping of columns, a pandas or Polars data frame, or a
fitted object's retained data. Columns may be lists, tuples, NumPy arrays, pandas
or Polars series, masked arrays or one-shot iterators.

- Each used one-shot column or row-aligned argument is read once per call; the
  missing-value checks, subset selection, expression evaluation, fitting and
  prediction then read the same prepared values. This also applies to the direct
  response arguments of `survfit`, `survdiff` and `survcheck`, to Aalen fits and
  to the formula-based data preparation functions. An iterator shared by a column
  and a direct argument, such as `weights`, is read once. Unused mapping columns
  are never read. The caller must supply a fresh iterator for another call.
- Declared factor levels, ordering and contrasts, and declared integer or double
  types, survive preparation. Ordinary lists, arrays and data frames keep their
  ownership.
- Used iterator values survive pickle and copy, and variables removed from the
  design, such as `z` in `x + z - z`, remain available when rebuilding a model
  frame. Unrelated iterators are not read during serialization; reading such a
  column from the restored object raises an input error.
- Rows come from evaluated formula variables and supplied row-aligned arguments:
  `~x` uses the length of `x` regardless of other mapping columns. A
  variable-free `~1` with no extra vectors has zero rows for a mapping and keeps
  a data frame's explicit row count; a weights vector supplies its row count, as
  in `stats::model.frame`.
- Masked cells in numeric response matrices, whether supplied as a masked array,
  an iterator of NumPy rows or mixed reusable rows, are R `NA` and follow the
  missing-row rules. Genuine NaN stays distinguishable from `NA`.

## Model matrices

`r.model_matrix(fit)` returns the stored design; `r.model_matrix(fit, data)`
builds the design for new data with the fitted contrasts. Python results are
mappings of lists with `assign` and column metadata. The R `model.matrix` methods
also return source row names and the fitted contrast declarations.

Python `model_matrix` accepts `na_action` (default `"na.omit"`, with the usual
short and dotted aliases; `None` applies no omission). R new-data calls resolve
`options("na.action")` and ignore an explicit `na.action` in `...`, as the stock
methods do. `na.omit` and `na.exclude` omit incomplete rows, `na.pass` keeps rows
and missing entries, and `na.fail` refuses incomplete frames. Stored matrices
keep their fitted omission masks. Under `na.pass`, unused transforms still warn
without propagating missing values into unrelated columns; ridge bases keep their
input columns independently, so a missing `age` leaves an observed `rx` intact in
`ridge(age, rx)`.

- **Labels.** Formula and basis column labels are separate from penalized
  coefficient labels: `ridge(age, theta = 2)` keeps the coefficient name
  `ridge(age)` but uses the formula call for its matrix column. Multi-column
  ridge terms append argument names; spline and dense frailty bases append
  numbered suffixes; named time-transform penalty matrices keep their basis
  suffixes.
- **Frailty.** Stored Cox matrices include sparse frailty columns in their
  formula positions with the original numeric codes. New-data Cox matrices
  evaluate the complete design. Sparse group coding precedes omission and accepts
  new groups, recoded locally; dense terms keep fitted levels and full indicator
  contrasts. The automatic frailty default can change a sparse term into a dense
  basis on a small new dataset; dense constructors reject fewer than two observed
  groups, and a fitted dense term switching to sparse codes fails, as in R.
  Returned matrices are copies; changing them cannot change the fit.
- **Logical covariates.** Unclassed logical variables are factors with `FALSE`
  and `TRUE` levels. Both levels stay available when subset or omission leaves one
  value. Main effects use `flagTRUE`; interaction-only terms such as `age:flag`
  produce `age:flagFALSE` and `age:flagTRUE`; AFT formulas without an intercept
  keep both dummy columns. Explicit factor constructors keep their single-level
  validation.
- **Strata.** Cox term assignments keep the original numbering across strata
  terms; Cox strata attributes keep missing labels in the matrix rows. AFT
  matrices omit standalone scale strata before evaluating new data, so a new frame
  may omit those columns, have missing values in them or use other labels. Strata
  in covariate interactions, variables also used as ordinary covariates, and
  offsets remain required and take part in omission. Responses and formula
  cluster variables are not needed. Intercept-only matrices keep the row count and
  labels of zero-column frames.
- **Time transforms.** Expanded `tt()` matrices use risk-set row labels and need
  their fitted time and risk-set context to be rebuilt for new data.
- **Null models.** R null Cox models return `NULL` coefficients, as in R.

### Evaluated R model frames

The R bridge treats a supplied data frame with a non-null `terms` attribute as an
evaluated model frame. Cox and AFT `model.matrix` then read its evaluated columns
directly: `age + log(z)` reads `age` and `log(z)` rather than raw `z`, and ridge
and spline terms use the supplied basis and its current width. As in stock
survival, `model.frame` is bypassed for such inputs, so missing rows remain under
every global action, supplied factor levels may differ from the fitted levels, and
offsets or unused variables do not remove rows. Required column names follow the
full or reduced formula; missing ones raise the stock mismatch error. A one-column
numeric matrix gets its variable name alone, even with a column suffix, while
wider matrices append the supplied names or one-based indices; assignments adapt
to changed basis widths; with one stratum, the supplied column keeps its
attributes, including custom contrasts. Logical
factor coding, interactions, assignments, row labels, contrasts and R `NA`/`NaN`
payloads are kept. The port's own `model.frame` output contains raw columns
without R's evaluated-column schema and `terms` attribute. Custom global
missing-data functions, tagged lists, ordered or nondefault fitted contrasts,
time-transform and multi-state frames and matrix-valued interactions have not been
checked against R for evaluated frames.

## Model-specific design rules

### AFT scale strata

`survreg` keeps the levels of an evaluated `strata()` term when a subset or
missing-value action removes every row of a group. Selecting groups `a` and `c`
from `a`, `b`, `c` keeps three scale positions, with a zero covariance row and
column for `b`. This applies to ordinary and penalized fits, robust covariance,
score and influence residuals, quantile predictions and their standard errors,
summaries, serialization and ANOVA refits. AFT interactions such as
`age * strata(g)` keep their empty design columns as aliased coefficients. Factor
levels absent before `strata()` is evaluated are still dropped.

An empty stratum provides no information about its scale: its scale keeps the
starting estimate and its zero variance marks a singularity. Predictions for that
stratum are supported for compatibility but are not identified. Explicit starting
values include every retained scale position. Penalized fits compute the effective
sample size from the observed scales only. The typed `regression.survreg_fit` and
`regression.survpenal_fit` accept an optional final `nstrat`; Rust provides
`survreg_fit_with_nstrata` and `SurvpenalFit::fit_with_nstrata`. Counts below the
largest observed code or too large for a covariance matrix are rejected before any
callback runs.

### Rank-deficient AFT designs

`survreg` keeps the complete design when a column is redundant. The generalized
Cholesky solve identifies the redundancy; the full fit marks the aliased location
coefficient as `NaN` (R's `NA`) with a zero covariance row and column, while the
compact `survreg_fit` result keeps the solver's numeric value. Degrees of freedom
keep R's parameter count. Constant numeric columns are not rescaled, including
constants other than zero or one, such as an unused Helmert contrast; nonconstant
columns are still centered and scaled when the design starts with an intercept and
no starting values are given. Training predictions and quantiles stay finite;
new-data and term predictions keep `predict.survreg`'s propagation of the aliased
`NA`.

### Penalized terms

Penalized Cox formulas support `ridge`, `pspline`, `frailty`, `frailty.gamma`,
`frailty.gaussian` and `frailty.t`; penalized `survreg` formulas support `ridge`
and `pspline` (frailty terms are refused, as in R). Basis construction and penalty
optimization run in Rust, and prediction reuses the training knots and factor
levels. `ridge(..., theta=..., scale=FALSE)` uses the penalty in the original
units; the default scales by each column's unweighted sample variance before
subset or omission. Omitting `theta` selects the penalty by effective degrees of
freedom. A penalty term inside an interaction raises "Penalty terms cannot be in
an interaction".

`predict(pspline(x), newx)` and `predict_pspline(basis, newx)` evaluate the basis
at new values with the original boundaries, degree, number of terms, intercept and
combined columns, extrapolating linearly beyond the boundaries. The result is a
`PsplineResult` with `penalty=False` and the matrix in `.basis`; omitting new
values returns the original object. The generic accepts `newdata` or R's `newx`.

### Time-transform terms

`coxph(..., tt=...)` rebuilds the fitted design after risk-set expansion. A
callback receives `(x, time, riskset, weights)`, with one-based risk-set ids and
`None` (`NULL` in R) for unweighted fits, and may return a numeric vector, a
numeric matrix, a logical vector or a factor. Python matrices can be nested rows,
NumPy arrays, data frames or mappings of column names to columns.

```python
import numpy as np
from survival import r

def time_basis(x, time, riskset, weights):
    return {"log": np.asarray(x) * np.log(time),
            "root": np.asarray(x) * np.sqrt(time)}

fit = r.coxph("Surv(time, status) ~ x + tt(x)", data, tt=time_basis)
```

The fitted columns here are `x`, `tt(x)log` and `tt(x)root`. An unnamed
multi-column matrix uses numbered suffixes; a one-column matrix keeps the plain
transform name. Declared factor levels survive the expansion, ordered pandas
categoricals use polynomial contrasts and R factors keep their contrasts. Each
distinct transform runs once, in formula variable order, even when it occurs only
in an interaction. The fit's response, ids, clusters and risk-set strata describe
the expanded rows; the original observation count and missing-value records remain
available. Prediction, curves and `model=TRUE` are not available for `tt()`
models.

A callback can return a penalized basis. Numeric vectors, matrices and factors
keep `coxph.penalty` attributes, and dense factor penalties use one indicator per
group. In Python, wrap the values in `r.CoxPenaltyBasis(x, penalty)` or return a
standalone constructor:

```python
from survival import regression

def sparse_frailty(x, time, riskset, weights):
    return r.CoxPenaltyBasis(
        x, regression.CoxPenalty.frailty(theta=.4, sparse=True, n=len(x))
    )

def ridge_penalty(x, time, riskset, weights):
    return r.ridge(x, theta=2, column_names=["ridge(x)"])

def smooth_effect(x, time, riskset, weights):
    return r.pspline(np.asarray(x) * np.log(time), df=3)
```

`r.ridge()`, `r.frailty()` (with `frailty_gamma`, `frailty_gaussian` and
`frailty_t` aliases) and `r.pspline()` results extend `CoxPenaltyBasis`.
`RidgeResult.scale_values` holds each column's sample variance before row
selection; several numeric arguments combine with R's `cbind` rules.
`FrailtyResult.levels` keeps the used levels in declared order and `codes` the
one-based group positions. Because Python cannot recover R's unevaluated argument
expressions, `column_names` supplies complete coefficient labels, and callback
splines are labeled from the variable name, such as `ps(x)3`. Boolean bases use
zero and one; a sparse penalty requires one numeric group column; penalty
interactions and multiple sparse penalties raise errors, and nonnumeric bases fail
before any penalty callback runs. Sparse callbacks without a custom formatter get
a group Wald summary. R callbacks may return
`survival::frailty()`, `survivalr::frailty()`, splines, ridge terms or custom
penalty objects; sparse formatting callbacks receive the frailties and their
variance diagonal with the third argument missing, as in R, and dense formatting
callbacks receive coefficient and covariance blocks. Controller histories and
formatter closures survive model serialization, and R frailty constructors keep
R's numeric and logical group labels.

## Multi-state Cox models

`coxph` on a factor-status `Surv` with `id=`, formula lists, `istate=` and
`statedata=` returns a `CoxphmsModel` with R's `cmap`, `smap`, `states` and
`transitions`. It supports `coef`, `vcov(matrix=True)` (a dict of `NamedMatrix`
keyed by transition; the R bridge returns R's three-dimensional array),
`predict`, `residuals`, `cox_zph`, `coxph_detail` and `survfit` (R's
`survfit.coxphms`). `survcheckallow` must name survcheck flags (overlap, gap, jump,
teleport); an empty value keeps them all. Offsets in formula lists raise an error
directing callers to the `offset` argument.

## Term metadata helpers

`r.attrassign` and `r.untangle_specials` expose R's helpers for prepared
model-matrix and formula metadata. `TermMetadata` holds the relevant R `terms`
attributes without observations or an environment; it copies inputs to immutable
tuples and supports pickle.

```python
terms = r.TermMetadata(["age", "treatment"])
groups = r.attrassign({"assign": [0, 2, 1, 2]}, terms)
assert groups == {"(Intercept)": [1], "treatment": [2, 4], "age": [3]}
```

Assignment codes are zero for the intercept and one-based positions in
`term_labels`. Returned column positions are **one-based**, as in R, unlike the
zero-based positions in a fitted model's internal `assign` dictionary. The matrix
input may be the mapping from `r.model_matrix(fit)`, a mapping with `assign`, or
an object with an `assign` vector. The terms input may be `TermMetadata`, a mapping
with `term_labels` (or `term.labels`), or a fitted model. Groups follow the first
occurrence of each label in column order; unused terms are absent; an empty
assignment vector returns `{}`; invalid codes are rejected.

```python
terms = r.TermMetadata(
    term_labels=["age", "strata(center)", "age:strata(center)"],
    variables=["Surv(time, status)", "age", "strata(center)"],
    factors=[[0, 0, 0], [1, 0, 1], [0, 1, 1]],
    order=[1, 1, 2],
    response=1,
    specials={"strata": [3]},
)
assert r.untangle_specials(terms, "strata", order=2) == {
    "vars": ["strata(center)"], "tvar": [2], "terms": [3]
}
```

| Field | Meaning |
| --- | --- |
| `term_labels` | Factor-matrix column labels |
| `variables` | Factor-matrix row labels, including a response when present |
| `factors` | Variable-by-term matrix of R's 0/1/2 factor codes |
| `order` | One positive interaction degree per term |
| `response` | Zero or one, subtracted from special variable positions |
| `specials` | Registered special names and their one-based variable positions |

Only labels are required for column grouping; selecting a present special needs
`factors` and `order`. An absent special returns `{"vars": [], "terms": []}`. For a
present special, `vars` lists its registered variables, `tvar` adjusts their
positions for the response, and `terms` selects matching degrees; `order` may be a
sequence. Returned lists are independent of the metadata. Selection uses the
supplied registration and does not infer specials from expression text.

`model_term_names` returns the original fitted term labels, including strata, as
R's `labels(fit)`. Because `model.matrix.survreg` renumbers assignments after
removing strata columns, applying `attrassign` with those labels can name a later
column after a strata term; the helper reproduces this literal mapping, so supply
metadata matching the matrix's assignment codes when grouping a transformed
matrix.

## Other helpers

`blogit`, `bprobit`, `bcloglog` and `blog` accept `inverse=True` for the inverse
link, with scalar or vector input.
