# Concordance for external linear models

`survival.r.concordance` accepts externally fitted linear models and GLMs
through the existing `YatesModel` adapter. It uses the same Rust concordance
kernel and Python model-comparison assembly as Cox and AFT models. The adapter
does not fit the external model.

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

Coefficients follow the formula's design-column order, including the intercept.
The covariance is part of the existing adapter contract and is unused by
concordance. Training data, weights and optional stored predictors must contain
the rows retained by the external fit, in matching order. Numeric responses,
factors, interactions, formula offsets, and the formula parser's numeric
transformations are supported. Bare numeric response transforms such as
`log(y)` now use the same evaluator as arithmetic expressions such as
`log(y) + 0`.

Without `linear_predictors`, the adapter computes the design product plus
formula offsets. NaN coefficients mark aliased columns and contribute zero.
With stored predictors, it uses their exact training values. R's `lm` can
construct its fitted values from the response and QR residuals, leaving tiny
differences even for an intercept-only model. Since concordance compares
predictors by exact equality, supplying those values preserves their rankings
and ties. For a GLM, supply its **linear predictor**, rather than fitted response
means. Concordance never calls a family's inverse link.

New-data calls rebuild the design with the fitted factor levels, drop incomplete
response/design/offset rows together, and use no training weights. An explicit
cluster vector must match the retained rows. Multiple models require matching
sample sizes and weights; differing responses warn. Their joint covariance is
the cross-product of their already weighted, optionally clustered influence
vectors. This retains the existing correction to R's repeated multiplication
by case weights in `cord.work`.

Plain numeric responses retain their original ordering, including nearly equal
values. This applies both to training-model concordance and direct
`concordancefit(y, x)` calls. A `Surv` response or model `newdata` uses the
survival-time near-tie handling controlled by `timefix`. R's model dispatcher
forgets to forward `timefix=False` to its final fitter; this implementation
honors the supplied option.

Direct `concordancefit` also accepts ordered or two-level categorical responses,
using their declared level order. Plain logical and character vectors are
refused; extract or convert numeric outcomes explicitly when that is intended.

## R interface

`survivalr::concordance` now has an `lm` method, which also accepts `glm` objects.
R evaluates the original model frame and predictions, including arbitrary R
transforms and contrasts, then the common implementation computes concordance,
counts, variance, influence and ranks. It supports stored or reconstructed
model frames, offsets, weights, new data, clusters and multiple fits.

New-data predictions remain aligned when the response is missing but predictors
are present. Stock R's separate response and prediction omission can instead
produce mismatched lengths. Cluster labels retain R's ordering in the returned
influence vector. The direct R `concordancefit` wrapper now uses the bare Python
interface, accepts numeric and orderable-factor responses, and forwards
`std.err=False` to avoid computing unused variance and influence results.

## Validation and performance

`scripts/generate_linear_concordance_reference.R` generates 35 cases from
R 4.5.3 / survival 3.8-12. Each is checked with stored and reconstructed
predictors. The fixture retains raw R output and the corrected joint weighted
covariance separately. It covers full and reduced models, factors, interactions,
transforms, offsets, aliased coefficients, weights including zero, clusters,
GLMs, new data, bounds, ranks and all influence modes.

Additional tests check exact and near ties, covariance against independent
single-fit influence vectors, row omission, type/shape errors, NumPy ownership,
data frames, serialization and concurrent calls. R tests compare whole results
and exercise the wrapper with the reference numerical function disabled.

`scripts/benchmark_linear_concordance.R` measures complete R-facing calls
against stock R, including preparation, conversion and result assembly. Model
fitting, input construction and explicit garbage collection are excluded.
It verifies results before timing and alternates call order after three warmups.

One local release-build run with 50,000 rows, R 4.5.3, survival 3.8-12 and
Python 3.14.7 produced these medians and ranges across seven samples:

| Complete call | Stock R (ms) | Shared implementation (ms) |
| --- | ---: | ---: |
| One model, training rows | 42 (41–43) | 38 (38–39) |
| Weighted model, training rows | 40 (39–41) | 43 (42–43) |
| Weighted model, new data | 63 (63–65) | 42 (41–43) |
| Two models, joint covariance | 92 (91–92) | 94 (93–97) |

Performance varies by workload: new-data scoring is faster here, while weighted
training and joint scoring are slightly slower. These times include the
R/Python boundary and exclude model fitting. Peak memory was not measured.

```sh
RETICULATE_PYTHON=$PWD/.venv/bin/python PYTHONPATH=$PWD/python \
  Rscript scripts/benchmark_linear_concordance.R 50000 7
```
