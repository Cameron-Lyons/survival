# Multistate curves from transition curves

`survfit` accepts a square matrix whose cell `[i][j]` is a Kaplan–Meier or
Cox survival curve for a transition from state `i` to state `j`. Use `None`
for absent transitions. All nonempty cells must have the same curve class
and the same numbers of strata and prediction columns. At least two
transitions are required, as in R's `survfit.matrix`.

```python
from survival import Surv, survfit

healthy_to_ill = survfit(Surv([1, 2, 3, 4], [1, 0, 1, 0]))
ill_to_dead = survfit(Surv([2, 3, 4, 5], [0, 1, 0, 1]))
fit = survfit(
    [[None, healthy_to_ill, None],
     [None, None, ill_to_dead],
     [None, None, None]],
    p0={"healthy": 1.0, "ill": 0.0, "dead": 0.0},
    method="matexp",
)
print(fit.time, fit.pstate)
```

`p0` defaults to probability 1 in the first state. A vector is reused for
all curves; a matrix supplies one row per output curve, with strata varying
fastest and Cox prediction columns next. A mapping names the states;
otherwise names default to `"1"`, `"2"`, etc. `states=` can name them explicitly.

Both methods use increments of the input **cumulative hazards**, not changes
in survival probabilities. `discrete` multiplies by `I + dA`; `matexp`
multiplies by `exp(dA)`. The actual R default is `discrete` for both KM and
Cox input. As in R, the discrete method caps the total departure probability
on the diagonal at 1 without rescaling off-diagonal increments. This can
produce probabilities summing above 1 when outgoing increments sum above 1;
the matrix-exponential method handles such increments as transition rates.

The output time grid is the union of transition event times. `start_time`
excludes times at or before that value and applies `p0` there, subtracting
the earlier hazard increments when updating probabilities. The default
start is `min(0, event times)` for each curve. `time0=True` adds the initial
probability row. An empty event grid gives an empty curve with its `p0` kept.

The result is a `SurvfitMultiStateResult`. `summary_survfit`, `survfit0`,
subsetting, `as_data_frame` and pickling use the existing curve machinery.
Standard errors and subject-level influence are unavailable. R's risk and
event counts are preserved, including its stepwise reuse of the previous
event count when another transition jumps. If several transitions depart
from one state, the last in column-major order supplies its risk count.
The additional `cumhaz` field holds the original, unconditioned transition
hazards; `n_transition` holds their stepwise event counts. Censor counts
are zero and `n_id` is unset because the inputs do not identify joint
subject histories. The complete list of deliberate differences is in
[R compatibility](r-compatibility.md#results-and-labels).

## Rust API and complexity

`surv_analysis::survfit_matrix` accepts borrowed `SurvfitMatrixTransition`
values, state names, an optional initial-probability matrix,
`SurvfitMatrixMethod` and an optional start time. Each transition gives its
zero-based source and destination and a slice of `SurvfitKMResult` values,
one per prediction column. The return type is `SurvivalResult<SurvfitAJResult>`.
The Python domain function `surv_analysis.survfit_matrix` accepts the same
columns as nested lists of native KM results. It releases the GIL around
the numerical kernel.

For each output curve, let `L` be the total input rows across transitions,
`U` the number of distinct event times, `K` the transitions and `S` the states.
Sorting the event-time union takes `O(L log L)`. Cursors visit input rows
once; discrete updates take `O(L + U(K + S))`. Matrix-exponential updates
add up to `O(U S³)` work for dense state graphs. No time-by-transition jump
matrix is allocated: working storage beyond the output is `O(L + K + S)`
for discrete updates, with an additional `O(S²)` exponential workspace.
The result itself takes `O(U(K + S))` storage.

## Verification and benchmarks

The shared reference file contains 16 direct comparisons against R survival
3.8-12: grouped KM and Cox curves, two Cox prediction columns, both methods,
vector and per-curve initial distributions, and two start times. Both Rust
and Python read it; Python additionally fits the underlying KM/Cox models.
Closed-form irreversible and reversible chains independently check the
matrix exponential. Regenerate the references with R, survival, jsonlite
and expm installed:

```sh
Rscript scripts/generate_survfit_matrix_reference.R
cargo test --lib --no-default-features survfit_matrix
PYTHONPATH=python .venv/bin/python -m pytest python/tests/test_survfit_matrix.py -q
```

Reproducible benchmarks exclude fitting and measure three transitions with
staggered times and censoring. The Python benchmark includes conversion of
the native result into the R-style Python object:

```sh
cargo bench --bench survival_benchmarks -- matrix_curves
PYTHONPATH=python .venv/bin/python scripts/bench_survfit_matrix.py
```

On an Intel Core Ultra 5 325, Linux x86-64 and Python 3.14.7, the release
extension gave these medians over five warmed Python calls (including
construction of the Python result):

| Observations per transition | Output times | Discrete | Matrix exponential |
| ---: | ---: | ---: | ---: |
| 1,000 | 2,000 | 0.81 ms | 1.01 ms |
| 10,000 | 20,000 | 16.27 ms | 16.05 ms |
| 100,000 | 200,000 | 182.69 ms | 188.35 ms |

These are local measurements of the port, not comparisons against R. The
benchmark reports all samples and the extension SHA-256 to identify the
measured build. State graphs with simultaneous transitions can take the
general matrix-exponential path and cost more than these staggered curves.
