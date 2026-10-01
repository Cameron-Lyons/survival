# Shared smoothing-parameter searches

`survival.regression.PenaltyController` exposes the Rust searches used by
the Cox and parametric penalized fitters. Supported methods are `fixed`,
`gamma`, `df`, `aic`, and `gaussian` (REML). Each controller is immutable;
the caller supplies the previous `PenaltyControlState` and receives a new
state after each inner fit.

```python
from survival.regression import PenaltyController

search = PenaltyController(
    "df", target_df=1.7, eps=0.1, thetas=[0.0], dfs=[4.0], guess=0.7
)
state = search.initial()
# After fitting with state.theta, supply the observed degrees of freedom.
state = search.step(state, 1, df=2.3)
print(state.theta, state.done, search.columns, state.history)
```

`step` uses consecutive, one-based iterations. `plik` is the unpenalized
likelihood used by AIC; `loglik` is the penalized likelihood used by gamma
correction. `neff` is the effective sample size, `trh` the Hessian trace,
`coef` the frailty coefficients, and `events_by_group` the integer event
counts. A `df` or `aic` controller accepts `gamma_correction=True` to report
the corrected likelihood as well as its ordinary search history.

Array inputs are copied, and numerical steps release the GIL. Controllers
and states support pickle and independent concurrent searches. The previous
state and its history remain unchanged. A state reports the next proposed
theta, including the final proposal when `done` becomes true; the inner fit
just completed used the previous state's theta.

Rust callers construct a `PenaltyControl` configuration and pass it to
`PenaltyController::new`. `initial` and `step` return `SurvivalResult`; the
step takes a `PenaltyControlInput` borrowing coefficient and event slices.
All three types and `PenaltyControlState` are exported from `regression`,
including builds without Python.

The public interface validates configurations, history dimensions, iteration
numbers and numerical inputs. Gamma and Gaussian `init` need at least two
values. The gamma correction accepts at most ten million events in a single
group, bounding its histogram allocation to 80 MB. Nonfinite or negative
theta proposals are errors. In particular, a df search with no upper point
at an exact boundary returns an error where R can return an unusable value;
the shared fitter also avoids an out-of-bounds access in that case.

The R bridge's `ridge`, `pspline`, and `frailty` constructors use these
searches through local callbacks. They preserve R callback signatures,
initial lists, vector-to-matrix history transitions, column names, gamma
corrections, and trace output. Existing user-supplied R callbacks remain
supported. The numerical controller functions and gamma correction in the
installed `survival` namespace are no longer required by these constructors.
Each R callback crosses into Python once and returns only the newly appended
history row; R retains its existing history attributes.

`scripts/generate_penalty_control_reference.R` records standalone controller
trajectories from the installed R package. Python tests replay the recorded
inputs through the public API; R tests compare callback states and printed
traces directly and fit models with the reference controllers disabled.

`scripts/benchmark_penalty_controllers.R` compares complete shared AFT calls
on 3,000 rows, changing only the controller callback between stock R and
the native search. Seven alternating samples after two warmups measured
4 ms for both ridge searches (ranges 4–5 ms), and 6 ms for Gaussian REML
versus 5 ms with the stock controller (both ranges 5–6 ms). Calls include
conversion and result assembly and are checked for equal results before
timing. These local, millisecond-resolution measurements do not establish
a speedup; the change removes the numerical dependency while keeping
small-call overhead low.
