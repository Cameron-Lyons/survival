# Term metadata helpers

`survival.r.attrassign` and `survival.r.untangle_specials` expose R survival's
helpers for prepared model-matrix and formula metadata. `TermMetadata` holds
the relevant R `terms` attributes without observations, a formula environment
or a fitted numerical model. It copies input sequences to immutable tuples,
protects the special-name mapping, and supports pickle.

## Grouping model-matrix columns

```python
from survival import r

terms = r.TermMetadata(["age", "treatment"])
groups = r.attrassign({"assign": [0, 2, 1, 2]}, terms)
assert groups == {"(Intercept)": [1], "treatment": [2, 4], "age": [3]}
```

Assignment codes are zero for the intercept and one-based positions in
`term_labels`. Returned column positions are also **one-based**, matching R's
`attrassign`. This differs from the zero-based column positions in a fitted
Python model's internal `assign` dictionary.

The matrix input may be the mapping returned by `r.model_matrix(fit)`, a
mapping containing just `assign`, or an object with an `assign` vector.
Only that vector is accessed. The terms input may be `TermMetadata`, a mapping
with `term_labels` (or R's `term.labels` spelling), or a fitted model supported
by `r.model_term_names`. Groups follow the first occurrence of each label in
matrix-column order, including reordered and repeated columns. Unused terms
are absent; an empty assignment vector returns `{}`. Invalid or out-of-range
codes are rejected.

`model_term_names` now retains strata labels for Cox and AFT models, matching
R's `labels(fit)`. These are the original fitted term labels. In R,
`model.matrix.survreg` renumbers assignments after removing strata columns;
applying `attrassign` with the original fitted terms can therefore label a
later column with a strata label. The prepared helper reproduces this literal
mapping. When grouping an independently transformed matrix, supply metadata
whose labels correspond to that matrix's assignment codes.

## Finding special variables and terms

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

The metadata fields correspond to R's terms attributes:

| Field | Meaning |
| --- | --- |
| `term_labels` | Factor-matrix column labels |
| `variables` | Factor-matrix row labels, including a response when present |
| `factors` | Variable-by-term matrix of R's 0/1/2 factor codes |
| `order` | One positive interaction degree per term |
| `response` | Zero or one, subtracted from special variable positions |
| `specials` | Registered special names and their one-based variable positions |

A mapping with these fields is also accepted. Only labels are required for
column grouping. Selecting a present special requires `factors` and `order`.
An absent special returns `{"vars": [], "terms": []}` and ignores the requested
order. For a present special, `vars` includes all its registered variables,
`tvar` adjusts their positions for the response, and `terms` selects matching
degrees. A sequence such as `[1, 2]` selects several degrees; `[]` selects none.
Returned lists are independent of the metadata.

Selection uses the supplied registration and factor codes. It does not infer
specials from expression text: nested calls, custom functions and namespace
qualification retain the meaning assigned by the original terms object.
The container does not implement R's general formula parser. R callers continue
to use their native `terms` objects through the R bridge's existing helpers.

Two R edge behaviors are retained. With exactly one term, R's `seq(ff)` uses
the summed factor code as its endpoint, so an interaction-only special with
code 2 produces indices `[1, 2]` even though there is one term. When removing
all terms leaves a registered special but no factor matrix, R errors; the
Python helper reports the missing factor matrix explicitly.

## Checks and column-grouping performance

`scripts/generate_term_helpers_reference.R` records stock survival 3.8-12
results for 24 term structures and 576 special selections (including eight
expected missing-matrix errors), 36 original/reversed/interleaved matrix
assignments, and 12 fitted Cox/AFT examples. Tests compare exact names,
indices, list order and absent fields. Additional cases check validation,
immutable ownership, pickle, repeated labels and excluded strata.

The public helper and penalized AFT setup share one column-grouping pass.
For P columns and T term labels, setup now takes O(P + T) work, replacing
O(P × T) repeated scans. Group indices require O(P) output storage; no model
matrix values are read or copied.

Run `PYTHONPATH=python .venv/bin/python scripts/benchmark_term_groups.py`.
The script compares the previous grouping loop with the current helper and
checks equivalent groups before timing. Seven warmed repetitions on an Intel
Core Ultra 5 325 with Python 3.14.7, ten columns per term, one intercept and
one excluded stratum gave these local medians:

| Terms | Matrix columns | Previous scans | One pass |
| ---: | ---: | ---: | ---: |
| 10 | 101 | 0.0138 ms | 0.0041 ms |
| 100 | 1,001 | 1.515 ms | 0.0467 ms |
| 1,000 | 10,001 | 183.506 ms | 0.4912 ms |

These measure metadata grouping only. They do not measure complete fits,
matrix construction, the public helper's input validation, or memory usage.
