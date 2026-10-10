# Aggregating survival curves

`aggregate_survfit` summarises the newdata columns of survival and multistate
probability curves at each time, following R's
[`aggregate.survfit`](https://github.com/therneau/survival/blob/master/R/aggregate.survfit.R).
It supports `mean`, `median`, `min`, `max`, `sum` and custom scalar summaries,
with the first grouping variable varying fastest. Only observed combinations
produce output columns.

```python
import numpy as np
from survival import r

totals = r.aggregate_survfit(curves, by=groups, FUN="sum")
spread = r.aggregate_survfit(curves, by=groups, FUN=lambda values: np.ptp(values))
```

Rust callers use `aggregate_survfit_with(surv, pstate, by, callback)`, where
the closure accepts `&[f64]` and returns `SurvivalResult<f64>`. Python's native
`aggregate_survfit(..., fun=callback)` accepts the same callbacks as the facade.
Every Python call receives a fresh one-dimensional `float64` NumPy array;
modifying or retaining that array cannot change the inputs or another call.
Built-in names release the interpreter lock; Python callbacks retain it.

Omitting `FUN` (or passing `None` in Python) preserves R's default distinction:
ungrouped ordinary survival curves use `rowMeans`, while explicit `"mean"`
uses R's residual-refined mean. A constant grouping vector also follows the
ungrouped path. Multistate probabilities and several observed groups always
use the mean reducer. These operations can differ for extreme cancellation;
for `[MAX, MAX, -MAX, -MAX, 1]`, the pinned R platform's default ordinary
average is `0.2`, while an explicit mean is `0.36`.

Following stock R, callbacks first receive each group's one-based prediction-row
indices. All these preliminary results are collected before checking that each
is a numeric scalar. Survival values then arrive in time/group order, followed
by state probabilities in state/time/group order. Values within a group retain
the original prediction-row order. A callback exception stops immediately and
retains its Python exception type and message. Extra keyword arguments are
accepted and ignored, matching R's unused `...`; a callback requiring another
argument consequently raises its own missing-argument error.

Python accepts real numeric scalars and numeric NumPy arrays containing one
element, including NaN and infinities. Boolean, string, list and vector results
are rejected. The result must remain scalar when called on curve values; a
callback that changes its return length receives a clear error. Stock R only
checks the preliminary calls and can subsequently produce an incompatible
curve shape from such a callback.

The R bridge accepts a function or function name and preserves the original
R group table, including factor order, column types and names, in `newdata`.
Its callback indices retain R's integer type, and ignored `...` expressions
are left unevaluated. Both ordinary Cox curves and multistate Cox curves retain
counts, times, strata and state metadata while dropping uncertainty components
and cumulative hazards. Stock R drops these components except `std.chaz`, which
incorrectly retains errors for the original prediction columns. After grouping,
those columns can describe different curves; the port clears them as well.
An R function name is resolved once when aggregation begins; rebinding that
name from inside a callback does not switch the function during later passes.
Stock R can resolve the name again when entering a subsequent `apply` call.

Mean and sum use compensated arithmetic for ordinary same-sign normal values.
Exceptional groups replay the pinned R platform's sequential arithmetic with
a 64-bit significand and a separate exponent. This preserves signed infinities,
subnormal values and cancellation across intermediate binary64 overflow.
For example, the sum of `[1e308, 1e308, -1e308]` is `1e308`, and a mean containing
one signed infinity retains it. Sequential extended arithmetic still rounds:
`sum([1e300, 1, -1e300])` returns zero on that R platform. Ordinary results may
differ in the final binary64 bits; R platforms with another long-double
precision can also differ.

An additional arithmetic audit compared 4,333 exponent-gap, cancellation,
subnormal, overflow and randomized inputs with the pinned R runtime. Sums
matched bit for bit. Mean and default row-mean differences were at most
`2.22e-16` relative in that corpus and remained within the ordinary bound
`8*EPSILON + 8*n*2^-64`. Exceptional cases retain sequential rounding instead
of substituting a mathematically exact sum.

Empty time/state axes retain the native API's shaped empty results. Stock R's
`apply` instead probes callbacks with zeros and sometimes errors while permuting
an empty multistate result. Zero data columns and missing grouping values are
rejected before callbacks. Native/Python group labels include observed levels;
R's bare factor `newdata` can include unused declared levels even when they do
not produce curve columns, and the R bridge retains that table.

The native kernel prepares stable positions for all members of each group once.
Every time and state then fills one reusable contiguous buffer containing exactly
one float per input column. Group boundaries select disjoint slices from that
buffer. Replacing separate growing group vectors avoids their repeated capacity
checks and preserves the original member order for mean calculations.

Medians select the upper middle value with
[`select_nth_unstable_by`](https://doc.rust-lang.org/std/primitive.slice.html#method.select_nth_unstable_by).
For an even group, the maximum of the lower partition supplies the lower middle
value. This takes linear work per group, replacing a full sort for each time and
state. NaN propagation, odd and even groups, ties and the existing non-finite
value behaviour remain covered by comparison against a sorted reference.

The public `GroupingFactor` fields are validated at each aggregate call. Invalid
codes and zero data columns return errors before indexing or selecting a median.
The usual group encoding checks the product of declared factor levels; when the
product exceeds `usize`, observed tuples are ranked directly. This retains valid
combinations and their ordering without overflowing or creating an array for
every possible combination.

The Rust benchmark measures complete native calls with twenty output times,
scrambled curve values, one or twenty groups, and three states for the multistate
case. Input matrices are prepared before timing. The Python benchmark checks
each output against NumPy before timing complete binding calls, including input
conversion. Its multistate cases use a prebuilt nested list, a C-contiguous NumPy
array and a Fortran-contiguous NumPy array.

Measurements below used this Linux x86-64 checkout with Rust 1.94.0, Python
3.14.7 and NumPy 2.4.6, with 100,000 columns and twenty output times. The
predecessor used sorted medians and separate group vectors.

The native benchmarks used fifteen one-call samples, with predecessor and current
executables run consecutively after compilation:

| Complete native call, 100,000 columns | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Median over all columns | 32.62 ms | 5.72 ms | 5.70× |
| Median in twenty groups | 26.53 ms | 6.55 ms | 4.05× |
| Three-state median in twenty groups | 81.76 ms | 20.17 ms | 4.05× |

The initial kernel comparison used release extensions with the former
nested-vector multistate boundary in both builds. The table gives medians of
seven samples:

| Complete Python call | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Median over all columns | 34.69 ms | 7.16 ms | 4.84× |
| Median in twenty groups | 26.55 ms | 7.92 ms | 3.35× |
| Three-state median in twenty groups | 308.40 ms | 223.37 ms | 1.38× |

These are local measurements of this implementation before and after the change.
Input conversion accounted for most of the time in that multistate comparison.

The multistate binding now accepts an owned typed three-dimensional float array.
An aligned `float64` NumPy array is copied directly into one C-layout Rust array;
other dtypes are normalised to `float64`, and unaligned storage is aligned before
Rust views are constructed. Fortran, transposed, sliced, reversed and broadcast
views are copied in logical axis order. The copy completes before the kernel
releases the GIL, so the kernel retains no references into Python storage.

Nested lists and tuples also fill one flat owned buffer, with a fast path for
Python floats and checked extraction for other scalar/sequence types. Wrong array
ranks, ragged margins and shapes too large for a Rust allocation report errors.
Empty nested sequence shapes remain supported. The Python multi-state Cox wrapper
passes its probability array directly to this boundary, avoiding `.tolist()`.

The following comparison isolates that input-boundary change. Both release
extensions use the improved median kernel above, and the inputs and expected
outputs are prepared before timing. Each entry is the median of seven complete
calls over 100,000 columns, twenty times, twenty groups and three states:

| Multistate input | Nested-vector boundary | Owned-array boundary | Speedup |
| --- | ---: | ---: | ---: |
| Prebuilt nested list | 237.58 ms | 66.13 ms | 3.59× |
| C-contiguous NumPy array | 913.74 ms | 25.65 ms | 35.62× |
| Fortran-contiguous NumPy array | 903.26 ms | 56.69 ms | 15.93× |

The NumPy path now avoids constructing a Python-to-Rust vector for each state
row and flattening all those vectors again. Fortran input still needs a logical
axis-order copy into the owned C-layout array. The final samples were collected
after compilation and the full test runs had finished.

The callback and arithmetic changes were compared separately with the preceding
release extension, using eleven alternating before/after pairs on CPU 0 after
the broad test jobs finished. Each sample includes the complete native Python
call, input conversion, grouping and output construction; outputs and group
labels were checked before timing. Both builds use the same owned-array boundary.
These inputs contain ordinary nonnegative values, not exceptional arithmetic.

| Complete Python call | Before | After |
| --- | ---: | ---: |
| Mean, 40,000 columns, eight times, twenty groups | 1.02 ms | 0.93 ms |
| Mean, 100,000 columns, twenty times, twenty groups | 5.28 ms | 5.03 ms |
| Three-state mean, 40,000 columns, eight times, twenty groups | 2.47 ms | 2.33 ms |
| Median control, 100,000 columns, twenty times, twenty groups | 6.23 ms | 6.37 ms |

The normal complete-call timings stayed close in this comparison; the new
arithmetic does not imply a general speedup. A separate reduction-only audit
over twenty million values measured a mean-loop cost increase of about 1–7%.
Exceptional groups use the slower sequential replay to preserve R's rounding.

The reproducible callback benchmark uses 40,000 columns, eight times, twenty
groups, and both survival and two-state probability components. The following
medians cover seven complete calls on the same CPU. Facade calls also prepare
the grouping vector. Every callback receives and returns its actual arrays;
all outputs and labels are validated before timing.

| Summary | Native built-in | Native NumPy callback | Facade built-in | Facade NumPy callback |
| --- | ---: | ---: | ---: | ---: |
| Mean | 2.27 ms | 2.56 ms | 3.70 ms | 3.88 ms |
| Sum | 2.37 ms | 2.14 ms | 3.99 ms | 3.94 ms |

These NumPy callbacks use their own vectorized reduction arithmetic, so their
timings and exceptional-value behavior need not match the named Rust reducers.

Reproduce the checks and measurements with:

```sh
cargo test --offline --lib surv_analysis::aggregate_survfit
cargo bench --offline --bench survival_benchmarks -- aggregate_curves \
  --sample-count 15 --sample-size 1
.venv/bin/python -m pytest python/tests/test_surv_analysis.py -k aggregate_survfit
.venv/bin/python -m pytest python/tests/test_array_inputs.py \
  -k 'aggregate_survfit or unaligned or noncanonical or finegray_normalizes'
.venv/bin/python -m pytest python/tests/test_survfit_coxphms.py -k aggregate
.venv/bin/python -m pytest python/tests/test_gil_release.py -k aggregate_multistate
.venv/bin/python -m pytest python/tests/test_aggregate_fun.py
.venv/bin/python scripts/benchmark_aggregate_survfit.py --columns 100000 --repeat 7
.venv/bin/python scripts/benchmark_aggregate_callbacks.py
```

The direct Rust tests compare medians across 1–257 members, sorted, reverse,
scrambled and tied inputs, signed zeros, infinities and NaNs. They also cover
uneven groups, unused levels, strided survival matrices, shared scratch across
multistate arrays, invalid public grouping fields, empty data margins and a
declared level product of `2^usize::BITS`. Python tests compare both numerical
components against NumPy for one or several groups and odd or even member
counts. The existing R fixture topic `km-aggregate_survfit` checks reported
survival curves, multistate probabilities and group labels.

The independent `aggregate_fun_reference.json` records stock-R callback values
and invocation traces for ordinary, multistate and combined components,
grouped and ungrouped curves, factor order, missing values, ignored arguments,
malformed returns and exceptional arithmetic. Its generator also records empty
axes and missing-group behavior separately so the deliberate validation and
shape differences remain reviewable. Python checks both input containers and
both interfaces, together with retained dataclass metadata, callback exceptions,
ownership and one-element numeric array returns. The R bridge tests compare
complete group metadata and curve components with stock R.

Typed-input Rust tests additionally verify array ownership before detaching,
NumPy dtype/layout conversion, unaligned buffers, empty axes, array-valued nested
rows and clear shape/type errors. Python transport tests cover all four summary
methods over float64, float32, int32, bool and byte-swapped arrays with six layouts,
along with array protocol objects, unaligned storage, malformed shapes and GIL
release during multistate aggregation.

The alignment check is shared with existing float/integer vectors, float matrices
and row-based matrices. Unaligned buffers are normalised before Rust array views
are constructed, and empty arrays return owned empty containers without views.
NumPy boolean vectors are converted numerically before Rust booleans are created:
NumPy buffers can contain nonzero bytes such as 2 or 255, while a Rust boolean
representation must be 0 or 1. Tests cover those buffers, sliced/offset views,
empty axes and ownership after the source array changes. Python regressions
exercise these conversions through Kaplan–Meier, curve aggregation, Yates
summaries and Fine–Gray interval expansion, comparing with aligned, normalised
inputs. These checks also run against installed wheels.
