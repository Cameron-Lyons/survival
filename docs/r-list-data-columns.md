# R list data columns

Named R lists now use the same atomic-column adapter as data frames. Previously,
automatic list conversion could turn logical NA into an observed boolean and
discard declared factor levels. A fresh Cox matrix for `age + flag`, with
`flag = c(TRUE, NA, FALSE)`, kept all three rows under omission. It now keeps
rows one and three, matching stock R. Pass rules retain the incomplete row
and its missing matrix entries.

Unclassed double columns cross as one-dimensional NumPy arrays, retaining
source NA payloads separately from genuine NaN. Complete integer columns also
cross in bulk. Missing integer, logical and character values use explicit
missing markers; factors retain their declared levels. Empty and single-value
vectors keep their shape. Classed atomic columns follow the established scalar
adapter. Nested, NULL and dimensioned list inputs retain their existing path;
an empty data list crosses as an empty mapping.

The shared adapter serves formula fits, model matrices, predictions and other
formula APIs. Data-frame conversion continues through the same helper with
explicit row counts and labels. Numeric views remain read-only and retain
their storage after the original R inputs are mutated, removed and collected.

Variable-free new model frames distinguish lists from data frames. A list or
plain Python mapping cannot infer a row count from unused columns and produces
zero rows, following stock R. Data frames keep their explicit row count.
Removed formula transforms still determine frame sizes; response-only and
strata-only frames derive sizes from the variables actually evaluated.

## Independent checks and persistence

Live R checks compare list-based Cox/AFT fits and fresh matrices across 18
formulas and four missing-data options. Thirty-six unmodified stock fits yield
1,976 matrices and 184 expected failures across 15 input variants. Whole values,
dimensions, names, assignments, contrasts, strata factors, NA/NaN masks and
successful-call warnings are checked. Sixty further stock fits cover numeric,
logical, response and strata omissions under omit/exclude actions; 900
prediction calls compare values, standard errors, shapes and row metadata.
Curve, log-rank and concordance calls separately check missing logical groups
and fail actions.

Eight stock fits independently generate 32 list/data-frame cases containing
174 matrices and 50 expected failures. They cover null, removed-transform,
strata-only and offset-only designs with empty lists, empty frames, explicit
zero-column frames and incomplete values. All cases run with Python list,
NumPy and pandas columns, adding 96 Python tests. The fixture regenerates
byte-identically. Failure checks compare missing-column/strata and missing-value
causes; assignment types retain the established integer convention.

The full Python suite passes 34,411 tests, with 48 skips and 37 documented
expected numerical differences. Pinned Ruff 0.16.9 lint/format, the 47-file
type check and generated stubs/manifest checks pass. The R source archive
matches all 54 R source/test files and passes 192,987 checks with zero errors,
warnings or notes, adding 26,967 checks over #712.

Four #712 models and four fresh models restore in separate R processes with
identical stored matrices and predictions. Restored models produce whole
stock-compatible list matrices under pass rules, including null designs.
Four earlier logical/ridge Cox/AFT models also restore unchanged stored
outputs and whole incomplete-frame matrices with their metadata. Rust sources,
Cargo inputs, native APIs and the release extension are unchanged from #707; native gates
remain its validation and were not rerun.

Two stock behaviors are checked explicitly. AFT term predictions can label a
flag contribution `strata(g)` when a standalone stratum occurs between age
and flag. The port retains the ordinary term name `flag`; values and row
metadata agree. Stock Cox expected/survival predictions for null models fail
when multiplying by NULL coefficients. The port's predictions instead match
the independently computed stock baseline hazard and its exponential survival.
Neither discrepancy is hidden by rewriting the reference output.

## Complete-call measurements

`scripts/benchmark_list_data_columns.R` compares #712 (`d52f9ff3`), this change
and stock R in separate libraries using the same release extension. Each fit
has 20,000 rows, 16 numeric covariates, one logical covariate and five strata;
new inputs have 10,000 rows. Numeric or logical incomplete inputs have NA every
59 rows. Whole values and attributes, including missing-value masks, are
verified before timing. Incorrect predecessor logical-missing calls are
verified but not timed.

Three warmups precede nine samples per workload. Runs execute sequentially
after validation. Calls include conversion, omission, matrix construction and
metadata; fitting, input setup, option changes and explicit GC are excluded.

Milliseconds, median (minimum–maximum), measured on an Intel Core Ultra 5 325
with Python 3.14.7, NumPy 2.4.6, R 4.5.3, survival 3.8-12 and reticulate 1.47.0.
The unchanged release extension was built with Rust 1.94.0.

| Family | Call | #712 | This change | Stock R |
| --- | --- | --- | --- | --- |
| Cox | Stored | 34 (32–41) | 34 (32–43) | 0 (0–1) |
| Cox | Complete data frame | 25 (24–25) | 25 (24–25) | 3 (2–3) |
| Cox | Complete list | 25 (24–26) | 24 (23–24) | 3 (2–3) |
| Cox | Numeric NA list, omit | 30 (30–31) | 26 (26–27) | 3 (2–3) |
| Cox | Logical NA list, omit | Incorrect; not timed | 29 (28–29) | 3 (2–3) |
| Cox | Logical NA list, pass | Incorrect; not timed | 27 (26–27) | 2 (2–2) |
| AFT | Stored | 39 (38–51) | 40 (37–48) | 0 (0–0) |
| AFT | Complete data frame | 22 (21–23) | 22 (21–22) | 2 (1–2) |
| AFT | Complete list | 22 (21–22) | 21 (20–21) | 2 (1–2) |
| AFT | Numeric NA list, omit | 28 (27–29) | 22 (21–23) | 2 (1–2) |
| AFT | Logical NA list, omit | Incorrect; not timed | 25 (24–34) | 2 (1–2) |
| AFT | Logical NA list, pass | Incorrect; not timed | 23 (23–24) | 1 (0–1) |

Incomplete numeric-list calls have 4 ms lower Cox and 6 ms lower AFT medians
in this workload. Complete list/data-frame and stored ranges overlap; no
general speedup is claimed. Stock remains faster, with stored medians below
this timer's 1 ms resolution. Previous logical-NA calls wrongly kept all
10,000 rows under omission and lost all 170 missing matrix entries under pass.
The corrected calls omit 170 rows or preserve their missing entries according
to the selected action, matching stock. These timings cover ordinary numeric
and logical designs with strata, not every formula or penalty family.

```sh
# Select each implementation's R_LIBS and PYTHONPATH before its run.
Rscript scripts/benchmark_list_data_columns.R 20000 9 predecessor
Rscript scripts/benchmark_list_data_columns.R 20000 9 current
Rscript scripts/benchmark_list_data_columns.R 20000 9 stock
Rscript scripts/generate_list_frame_reference.R
```

The broader compatibility and performance audit remains open. Domain warnings
can still be suppressed when another missing covariate already excludes the
same row under omission. Already evaluated R model-frame inputs, stored-method
argument overrides, and general nested or dimensioned formula inputs remain
outside these checks. These results do not establish full package parity.
