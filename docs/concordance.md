# Concordance

`survival.r.concordance` computes concordance for survival responses, fitted Cox
and AFT models, and externally fitted linear models and GLMs. All paths use the
Rust `concordancefit` kernel for counts, variance, influence and event ranks.
Differences from R are listed in
[R compatibility](r-compatibility.md#concordance).

## Direct calls

```python
from survival import core

time = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0]
status = [1, 1, 0, 1, 1, 1, 0, 1]
risk = [0.5, 0.2, 0.5, 0.9, 0.2, 0.7, 0.1, 0.9]

# R's concordance(Surv(time, status) ~ risk, reverse = TRUE, influence = 1)
fit = core.concordancefit(
    core.SurvivalData(time, status),
    core.CovariateMatrix(risk, len(risk), 1),
    reverse=True,
    influence=1,
)
```

- An observation censored at an event time remains a risk comparator;
  simultaneous events contribute outcome ties. Counting-process responses are
  supported.
- Raw influence rows are derivatives with respect to case weights, holding
  time-weight multipliers fixed; `dfbeta` applies case weights and uses pooled
  counts across strata. Variance is available with every result, and `influence`
  selects the returned diagnostic rows. For several predictors,
  `result.covariance` and `vcov(result)` include the covariance between them.
- `timefix=True` (the default) groups near-tied times with R's `aeqSurv`
  tolerance; `timefix=False` keeps exact times. `ymin` clips exit times and `ymax`
  limits contributing event times. Strata with fewer than two original events use
  unit time multipliers, including with a horizon. With no comparable pairs,
  concordance and its variance are `NaN`.
- Direct `Surv` inputs accept `strata=labels`. For a single predictor,
  `keepstrata` controls the optional `stratum_labels` and `stratum_counts`; counts
  hold the five exclusive pair categories (concordant, discordant, tied
  predictors, tied outcomes, tied in both). Concordance and covariance pool across
  strata.
- A zero-weight event whose weighted risk set is empty contributes zero to the Cox
  variance numerator, matching deletion of that observation.
- `concordancefit` keeps a keyword-only `names` for the predictor columns. It also
  accepts plain numeric responses, which keep their exact ordering (no near-tie
  merging), and ordered or two-level categorical responses in declared level
  order; plain logical and character vectors are refused.
- Iterator cluster and strata columns are read once; declared categories keep
  their order, and unused categories never add influence rows.

## Fitted models

`r.concordance(fit, ...)` prepares each model with its own retained response,
stored linear predictor, strata, weights and applicable clusters, including
omitted training rows and `na.exclude` fits. AFT formula `cluster()` terms are not
implicit concordance clusters, and AFT training concordance needs a fit that kept
its response (`y=True`). Multi-state Cox fits are refused.

- New data: responses, predictors, offsets and strata are omitted together when
  incomplete; training weights and clusters do not carry over. A row missing only
  its response is dropped from both the frame and the predictions.
- An explicit `cluster` must match the retained rows and replaces fitted
  clusters. Fitted categorical clusters keep their declared order through row
  selection, omission, pickle and multi-state model objects; plain character
  labels sort lexically (`"10"` before `"2"`) and numeric labels numerically.
- Several models must have matching sample sizes, weights and cluster counts;
  differing responses warn. Joint covariance is the cross-product of the already
  weighted, optionally clustered influence vectors. Single-model results keep
  per-stratum count tables; joint results pool each model's strata into one count
  row. For joint models, `influence=1` returns `dfbeta`, `influence=2` the
  three-dimensional influence array and `influence=3` neither, following R's
  fitted-model method; a single model with `influence=3` returns both.
- Joint influence rows are aligned by cluster membership on the retained rows:
  renaming groups or reordering factor levels does not change the covariance, and
  models with different partitions are refused with "models must have identical
  clustering". Compatible fits use the first model's group order, which also
  orders the returned `dfbeta` rows; an unclustered fit is compatible with
  singleton clusters. Joint models still need their observations in matching
  order.

The R bridge's `concordance` method uses the same preparation and keeps R model
names, count labels and cluster ordering. Rank tables contain event rows without
R's model-frame row labels, and new-data stratum counts keep the fitted labels.

## External linear models and GLMs

`r.concordance` accepts external fits through the `YatesModel` adapter, which does
not refit the model:

```python
from survival import r

model = r.YatesModel(
    "y ~ x + group",
    retained_training_data,
    coefficients,
    covariance,
    weights=case_weights,
    linear_predictors=training_predictions,
)
result = r.concordance(model, influence=3, ranks=True)
comparison = r.concordance(model, another_model, influence=1)
validation = r.concordance(model, newdata=validation_data)
```

Coefficients follow the formula's design-column order, including the intercept;
the covariance is part of the adapter contract and unused here. Training data,
weights and optional stored predictors contain the rows retained by the external
fit, in order. Numeric responses, factors, interactions, formula offsets and the
parser's numeric transformations are supported, including bare response
transforms such as `log(y)`. Without `linear_predictors`, the adapter computes the
design product plus offsets, with NaN coefficients contributing zero. Supplying
the external fit's stored predictors keeps its exact ties (R's `lm` fitted values
can differ in the last bits); for a GLM supply the linear predictor, since
concordance never applies the inverse link. New-data calls rebuild the design with
the fitted levels, drop incomplete response, design and offset rows together and
use no training weights. Plain numeric responses keep near ties; a `Surv` response
or model `newdata` uses `timefix`.

The R bridge's `concordance` has an `lm` method that also accepts `glm` objects;
R evaluates the model frame and predictions, including arbitrary transforms and
contrasts, and the common implementation computes the result, with stored or
reconstructed frames, offsets, weights, new data, clusters and several fits. The R
`concordancefit` wrapper accepts numeric and orderable-factor responses and
forwards `std.err=FALSE`.
