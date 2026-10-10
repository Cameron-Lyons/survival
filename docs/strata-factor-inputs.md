# Reusable strata factors and one-shot inputs

`strata()` now consumes iterator columns once. Previously, inspecting a single
iterator to distinguish a vector from a collection of columns exhausted it;
discovering levels also exhausted iterator columns inside named or positional
collections. A nonempty input such as `strata(iter([1, 2, 1]))` consequently
returned an empty factor. Iterator materialization preserves declared factor
levels, including unused labels used to format compound strata. Reusable lists
and arrays retain their existing input path and remain unmodified.

`StrataFactor.categories` exposes its level order as an immutable snapshot.
Returned strata factors now participate in the same factor protocol as pandas
categoricals and factors received from R. This matters when reusing a factor in
`Surv`, another `strata()` call, or a model data column. Previously, `Surv`
rejected its string labels, and a model could interpret numeric-looking labels
as a continuous predictor. The three-group Gaussian AFT reference checks the
categorical design, coefficient names, fitted coefficients, covariance, scale,
and predictions against stock R. Grouped Kaplan-Meier curves also check row
counts, factor ordering, times and survival probabilities.

The literal string `"NA"` is a valid event-state label, including when
`strata(..., na_group=True, shortlabel=True)` produces it. It remains distinct
from a missing factor value. Blank state labels remain invalid.

`scripts/generate_strata_iterable_reference.R` freezes independent values from
R 4.5.3 and survival 3.8-12. Its 142 source cases cover numeric, logical,
character, declared-factor and compound inputs; missing-value groups; named
and positional arguments; and explicit and default label choices. Each case
includes factor codes, levels, labels and counts, a factor roundtrip, and right
and counting-process `Surv` metadata. Tests apply five equivalent Python input
forms and check ownership and single consumption. A bare Python sequence of
only missing values has no numeric or character dtype, so all-missing
positional source cases specify `shortlabel` explicitly.

Regenerate and verify with:

```sh
Rscript scripts/generate_strata_iterable_reference.R
PYTHONPATH=python .venv/bin/python -m pytest -q python/tests/test_strata_iterable_inputs.py
```
