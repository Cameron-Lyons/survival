# Missing observation rows in survival residuals

KM and Aalen–Johansen fits retain their model frame's missing-data action as
`fit.na_action`, using the same `NaAction` record as Cox and AFT models. Its
one-based positions refer to the data after any subset selection. `survfit0`,
deep copies, and pickle retain this metadata.

`survfit_residuals` and `pseudo` follow stock R's row rules:

| Output | `na.omit` | `na.exclude` |
| --- | --- | --- |
| Observation residual matrix or array | Retained rows | Original row count, omitted rows filled with NaN |
| Observation pseudo-value vector, matrix or array | Retained rows | Original row count, omitted rows filled with NaN |
| Collapsed subject output | Retained subjects | Retained subjects |
| Long table (`data_frame=True`) | Retained rows | Retained rows |

Without an explicit id, labels preserve the positions before omission. For
example, omitting observations 2 and 6 gives ids `[1, 3, 4, 5, 7, 8]`, rather
than renumbering them. Excluded matrix row labels use the omitted position,
as R's `naresid.exclude` does, including when retained ids are strings. The
corresponding `curve` entries are NaN. As in R, requesting `collapse=True`
with no repeated clusters or ids leaves the output at observation level and
restores excluded rows.

```python
from survival import r

fit = r.survfit(
    "Surv(time, status) ~ 1",
    {"time": [1, None, 3, 4], "status": [1, 0, 1, 0]},
    na_action="na.exclude",
)
result = r.survfit_residuals(fit, times=[2, 3])
# result.id == [1, 2, 3, 4]; result.resid[1] contains two NaNs.
```

`scripts/generate_survfit_missing_residual_reference.R` creates 96 references
with unmodified R 4.5.3 and survival 3.8-12. They cover KM and Aalen–Johansen
curves, weights, omitted response/weight rows, repeated counting-process
histories, both missing-data policies, collapse, all three residual kinds,
one or two times, array values, row labels, curve indices, and long tables.
`python/tests/test_survfit_missing_residuals.py` checks these values and also
checks bare `Surv` input, subset positions, copying, and persistence.
The R facade uses the retained positions for pseudo-value row names and
preserves integer curve indices with NA after exclusion.
`r/survivalr/tests/testthat/test-survfit-missing-residuals.R` compares live
stock R values, missingness, dimension labels, and tables for these paths.
R's omitted character ids can attach observation names to the dimension-label
vector itself. The facade preserves those label values but flattens this
nested names attribute, as it does for other Python label lists.

Stock grouped Aalen–Johansen single-time tables fail because R calls
`col()` on a three-dimensional array. The fixture retains these errors and
also stores the matching time's rows from an independent successful
multi-time call. The port returns those tables directly. Stock exclusion of
multi-time Aalen–Johansen residuals also loses array dimension labels; the
port retains its state/transition names and row metadata.
