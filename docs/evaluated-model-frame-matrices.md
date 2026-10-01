# Matrices from evaluated model frames

The R bridge now recognizes a supplied data frame with a non-null `terms`
attribute as an evaluated model frame. Cox and AFT model.matrix read its
evaluated columns directly. For example, `age + log(z)` reads `age` and
`log(z)` rather than looking for raw z and running log again. Ridge and spline
terms use the supplied basis, including its current width and column names.

Stock survival bypasses model.frame construction for these inputs. Consequently,
missing rows remain even when the global action is `na.omit`, `na.exclude` or
`na.fail`; supplied factor levels can differ from fitted levels; and offsets or
unused variables do not remove rows. Required column names still follow the
full or reduced formula, including offsets. Missing required evaluated columns
raise the stock model-frame/formula mismatch cause. A non-null marker has the
same effect as a full terms object, matching stock's dispatch test.

One shared Python constructor rebuilds the design from these columns without
altering the fitted design. It preserves logical factor coding, interactions,
assignments, row labels, contrast metadata, stratum levels and distinct R NA/NaN
payloads. The R adapter transfers matrix columns in bulk and supplies small
factor contrast bases. With one stratum it preserves the supplied column's
attributes, including custom contrasts. Python interop uses the private
`_r_model_frame` factory;
ordinary Python mappings and pandas frames continue to represent raw inputs.
No public native binding or Rust fitting kernel changes.

Stock Cox's call for standalone strata misspells `contrasts.arg`, so supplied
frame contrasts apply in that branch. Other Cox/AFT branches override them with
fitted contrasts. The constructor reproduces that distinction. A one-column
numeric matrix gets its variable name alone, even when it has a column suffix;
wider matrices append supplied names or one-based indices. Assignments adapt
to changed basis widths rather than reusing the fitted matrix's assignments.

## Validation

Unmodified survival 3.8-12 generates 212 models: 28 Cox and 25 AFT formulas,
automatic or Unicode labels, and both cached and rebuilt matrix modes. AFT
does not support stock frailty fits, so those three formulas are Cox-only.
Twenty-eight inputs under four missing-data actions produce 23,744 calls:
21,888 matrices and 1,856 expected failures. Each action is executed in R;
identical expectation sets are pooled only when serializing the fixture.

Inputs cover missing numeric/logical/factor/stratum/offset/response values,
distinct NaN, removed columns, altered evaluated values and factor levels,
custom contrasts, factor/logical retyping, supplied matrix NA/NaN, changed
basis widths and Unicode names, all-missing and empty frames, one missing row,
plain non-null markers and untagged controls. List, NumPy and pandas containers
exercise the same stock fixture in 2,544 Python tests. Three further tests
check 11 fitted designs each for input/result ownership and unchanged stored
matrices. Live R tests compare whole values, dimensions, names, assignments,
contrasts, strata, NA/NaN masks, errors and warnings, and unchanged inputs and
stored coefficient/covariance/matrix outputs.

The stock fixture regenerates byte for byte: 7,630,624 bytes, SHA-256
`ae16b2b0f3f2faa27f8a4df6f3c898b993c66fe2194b184210ed4ee2f8cd4795`.
All reference matrix calls are warning-free; eight fit warnings are compared with
whitespace normalized. Existing integer assignment conventions remain.
Error comparisons retain the established Python cause for missing raw Cox
strata. One stock AFT untagged control with `age + strata(g) + strata(g):flag`
fails with `number of variables != number of variable names` before resolving
raw g; the port reports the absent raw g. That precise control is checked
explicitly without changing the fixture or successful matrix expectations.

Four transformed models saved by #713 and four saved by #714 restore with
unchanged stored matrices and predictions; their supplied evaluated matrices
match stock under all four actions. Older designs without `variable_labels`
use their existing variable metadata. Eight fresh models, including omitted
training rows, also restore in separate processes: 16 models and 64 evaluated
matrix calls in total, with unchanged stored coefficient/covariance/matrix
outputs and predictions.

The full Python suite passes 37,800 tests, with 48 skips and 37 existing
expected numerical differences. Pinned Ruff 0.16.9 lint/format, the 47-file type
check and generated stubs/manifest checks pass. The full R source archive
passes 468,368 checks with zero failures, warnings, skips or check notes,
adding 250,472 checks over #714. All 58 R source/test files match both the
archive and its checked sources. The new live tests are grouped by family
and formula rather than accumulating every case in one test.

Rust sources, Cargo inputs, native APIs and the release extension remain
unchanged from #707. Its native validation remains the latest native gate;
those checks are not rerun for this Python/R change.

## Complete public-call measurements

`scripts/benchmark_evaluated_model_frames.R` compares #714 (`80ed7116`), this
change and stock R in separate processes and libraries with the same release
extension. Fits use 20,000 rows; new inputs use 10,000 rows. Three designs cover
a scalar numeric covariate, 16 transformed covariates with a logical factor,
five strata and an offset, and a two-column ridge basis. Calls include stored
matrices, raw data and complete/incomplete evaluated frames. Every 59th row
is incomplete in the latter workload.

Stock outputs are collected before loading the bridge's registered methods.
Whole outputs and warnings are verified before timing. Incorrect predecessor
calls are verified but not timed: transformed/penalty frames require absent
raw source columns, while incomplete scalar frames lose rows retained by R.
Three warmups precede nine samples per workload. Runs execute sequentially
after the heavy validation gates. Timing includes the complete public matrix
call, input conversion, construction, metadata and warning/error capture;
fitting, input setup, option changes and explicit GC are excluded.

Milliseconds, median (minimum–maximum), on an Intel Core Ultra 5 325 with
Python 3.14.7, NumPy 2.4.6, R 4.5.3, survival 3.8-12 and reticulate 1.47.0.
The unchanged release extension was built with Rust 1.94.0. Stored calls omit
the data argument: stock AFT distinguishes omission from an explicit NULL.

| Family/design | Call | #714 | This change | Stock R |
| --- | --- | --- | --- | --- |
| Cox / scalar | Stored | 29 (27–37) | 27 (24–42) | 0 (0–0) |
| Cox / scalar | Raw frame | 17 (16–18) | 15 (14–20) | 1 (0–1) |
| Cox / scalar | Evaluated frame | 17 (16–19) | 15 (14–15) | 0 (0–1) |
| Cox / scalar | Evaluated frame with NA | Incorrect; not timed | 13 (13–14) | 0 (0–1) |
| Cox / transformed | Stored | 48 (41–132) | 37 (35–52) | 0 (0–1) |
| Cox / transformed | Raw frame | 42 (39–56) | 35 (35–37) | 4 (3–4) |
| Cox / transformed | Evaluated frame | Incorrect; not timed | 54 (53–56) | 1 (1–2) |
| Cox / transformed | Evaluated frame with NA | Incorrect; not timed | 53 (53–54) | 1 (1–2) |
| Cox / ridge | Stored | 28 (23–41) | 24 (23–25) | 0 (0–1) |
| Cox / ridge | Raw frame | 19 (17–26) | 15 (14–15) | 1 (1–2) |
| Cox / ridge | Evaluated frame | Incorrect; not timed | 19 (18–24) | 1 (0–1) |
| Cox / ridge | Evaluated frame with NA | Incorrect; not timed | 19 (18–19) | 0 (0–1) |
| AFT / scalar | Stored | 29 (28–45) | 24 (24–31) | 0 (0–1) |
| AFT / scalar | Raw frame | 17 (16–17) | 14 (13–14) | 1 (0–1) |
| AFT / scalar | Evaluated frame | 18 (17–42) | 13 (13–13) | 0 (0–1) |
| AFT / scalar | Evaluated frame with NA | Incorrect; not timed | 13 (13–14) | 0 (0–1) |
| AFT / transformed | Stored | 49 (46–52) | 42 (40–55) | 0 (0–0) |
| AFT / transformed | Raw frame | 39 (37–41) | 32 (32–32) | 2 (2–3) |
| AFT / transformed | Evaluated frame | Incorrect; not timed | 39 (39–40) | 1 (0–1) |
| AFT / transformed | Evaluated frame with NA | Incorrect; not timed | 41 (39–42) | 1 (0–1) |
| AFT / ridge | Stored | 33 (29–38) | 26 (26–27) | 0 (0–0) |
| AFT / ridge | Raw frame | 19 (17–22) | 16 (14–16) | 1 (1–2) |
| AFT / ridge | Evaluated frame | Incorrect; not timed | 20 (19–27) | 0 (0–1) |
| AFT / ridge | Evaluated frame with NA | Incorrect; not timed | 20 (19–20) | 0 (0–1) |

Scalar evaluated-frame medians are 15/13 ms for Cox/AFT, versus 17/18 ms in
#714. Raw/stored controls also have lower medians in this run, and many ranges
overlap, so no general speedup is claimed. Correct incomplete scalar calls
retain all 10,000 rows; #714 drops 170 rows and is not timed. Complete/incomplete
evaluated transformed calls take 54/53 ms for Cox and 39/41 ms for AFT. Ridge
calls take 19/19 ms and 20/20 ms. The predecessor rejects those transformed
and ridge frames; those calls are verified but not timed.

The new transformed-frame constructor remains slower than its current raw
counterpart (35/32 ms for Cox/AFT), despite skipping formula evaluation.
It rebuilds column kinds and contrast coding from supplied data. Repeated
numeric-column classification is a candidate for the next performance audit.
Stock remains faster throughout; results below 1 ms are beneath this timer’s
resolution. These measurements cover the stated scalar, transformed and ridge
designs, not every formula or penalty family.

```sh
# Select each implementation's R_LIBS and PYTHONPATH before its run.
Rscript scripts/benchmark_evaluated_model_frames.R 20000 9 predecessor
Rscript scripts/benchmark_evaluated_model_frames.R 20000 9 current
Rscript scripts/benchmark_evaluated_model_frames.R 20000 9 stock
Rscript scripts/generate_evaluated_matrix_reference.R
```

This change covers supplied evaluated data frames for the stated Cox/AFT
designs. A fresh probe confirms that the port's own Cox/AFT model.frame output
contains raw variables and no terms attribute, while stock returns evaluated
columns with terms. That output requires a separate change. Custom global
missing-data actions, general tagged lists, ordered/nondefault
fitted contrast functions, time-transform and multistate frames, matrix-valued
interactions, stored-method argument overrides and broader formula/penalty
families remain open. Full package parity and performance are not established.
