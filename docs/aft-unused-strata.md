# AFT scales after row removal

`survreg` now retains the levels of an evaluated `strata()` term when a subset
or missing-value action removes every row from a group. For example, selecting
only groups `a` and `c` from `a`, `b`, `c` keeps three scale positions, with a
zero covariance row and column for `b`. Previously, the shared formula path
renumbered the groups and returned two scales. Coefficients could agree with R
while covariance dimensions, scale labels and predictions differed.

This applies to ordinary and penalized fitting, robust covariance, score and
influence residuals, quantile predictions and their standard errors, model
summaries, Python serialization, and refits used by ANOVA. AFT interactions
such as `age * strata(g)` retain empty design columns and their aliased
coefficients as well as the scale columns. Factor levels absent
before `strata()` is evaluated are still dropped, following R. Several separate
`strata()` terms are recombined after selection, so only observed combinations
remain; a single `strata(a, b)` term preserves its preselection combinations.
Cox and curve methods continue to use contiguous observed stratum codes.

The formula cache retains the original levels with selected rows. AFT passes
the declared count to the existing numerical engines. Python's full
`regression.survreg_fit` and `regression.survpenal_fit` accept an optional final
`nstrat` argument, as their raw counterparts already did. Rust adds
`survreg_fit_with_nstrata` and `SurvpenalFit::fit_with_nstrata`; existing fitting
signatures remain available. Counts below the largest observed code or too
large for a representable covariance matrix are rejected before callbacks.

An empty stratum provides no information about its scale. Its scale value
retains the starting estimate and its zero variance is a singularity marker,
not evidence that the scale is known precisely. Predictions for that stratum
are supported for compatibility but inherit this lack of identification.
Explicit starting values include every retained scale position.

## R reference behavior

Stock survival 3.8-12 supports empty interior strata in ordinary fits. Other
paths fail before returning a usable result:

- A trailing empty level can cause a scale-name length mismatch.
- Quantile standard errors and influence residuals infer their column count
  from the largest observed code, omitting trailing scale positions.
- Penalized fitting tries to invert the intercept-only covariance including
  the empty scale's zero row and column.

`scripts/generate_aft_unused_strata_reference.R` records 78 cases over Weibull,
Gaussian and log-normal distributions, including weighted, clustered and ridge
fits and strata interactions. Thirty-six fits use stock R results; the other
42 record stock errors and
use the corrected reference. Comparison values for failures use local copies of R's
functions with explicit corrections: fit counts use the retained factor levels;
prediction and residual counts use the fitted scale vector; penalized effective
sample size averages observed scales and inverts only their covariance block.
The installed likelihood and optimization routines remain unchanged.

The Rust penalized engine uses that same observed block for effective sample
size. Empty strata therefore do not alter the effective sample size or penalty
strength. Tests compare the estimable results with compact R models while
preserving the original, pre-subset penalty basis and scaling. The earlier
strata-expression fixture now retains the missing-value scale for
`~ age + g + strata(g, na.group = TRUE)` instead of fitting complete cases with
that level removed.

## Verification

The new tests cover leading, interior and trailing empty strata, a single
observed group, subsets followed by omission, response/predictor/weight missing
values, `na.exclude`, explicit initial values, ANOVA refits, adaptive ridge and
P-spline penalties, and callback validation. Two new Rust tests exercise the
full typed APIs and their raw fitting engines. All 92 new Python cases and
196 new R checks pass. The full Python suite passes 15,117 tests, with 48 skips
and 37 documented expected differences. Rust passes 1,671 tests with all
features, 1,647 with ML and 1,415 without default features; documentation tests
contain no runnable examples. The R source archive passes 7,634 checks with
zero errors, warnings or notes. Strict Clippy, Rust documentation/formatting,
pinned Ruff, generated interfaces and Mypy across 47 files pass.

One run's existing four-thread Cox timing assertion missed its threshold while
a baseline build was compiling. The final complete Python run passed after
compilation stopped; its threshold and implementation were unchanged.

## Performance

`scripts/benchmark_aft_unused_strata.R` checks location coefficients and the
active covariance block against R before timing complete weighted R-facing
calls on 50,000 rows. Each path receives three warmups and seven samples,
alternating the shared implementation and stock R where R supports the case.
Input construction and explicit garbage collection are excluded. The previous
revision is `1b536acb`, built separately with its own Python sources and Rust
release extension. Builds and timing runs do not overlap.

R 4.5.3, survival 3.8-12 and Python 3.14.7 produced these medians and ranges in
milliseconds. The subset removes group `b`; missing predictor rows are omitted
in every case.

| Complete call | Previous | Current | Stock R, current run |
| --- | ---: | ---: | ---: |
| Ordinary AFT, all groups | 142 (139–170) | 130 (126–135) | 86 (84–89) |
| Ordinary AFT, empty group | 135 (134–149) | 130 (129–133) | 59 (58–60) |
| Ridge AFT, all groups | 146 (139–152) | 136 (133–137) | 68 (68–69) |
| Ridge AFT, empty group | 141 (140–144) | 130 (128–132) | Fails during inversion |
| Cox, empty group | 144 (143–152) | 137 (135–140) | 75 (74–78) |

These measurements do not establish a fitting speedup. Stock R also ran faster
in the later process; an earlier current run measured 139/139/151/170/146 ms
for these five calls, with overlapping or wider ranges. The current formula
calls remain slower than stock R. The previous AFT subset calls omitted the
empty scale, so their complete result dimensions differ even though the
estimable coefficients and covariance agree. R's ridge result is checked using
a compact stratum mapping that preserves its pre-subset penalty scaling, but
that different R call is not timed. Timer resolution is about one millisecond;
peak memory is not measured.
