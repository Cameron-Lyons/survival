# Aggregating survival curves

`aggregate_survfit` summarises the newdata columns of survival and multistate
probability curves at each time, following R's
[`aggregate.survfit`](https://github.com/therneau/survival/blob/master/R/aggregate.survfit.R).
It supports `mean`, `median`, `min` and `max`, with the first grouping variable
varying fastest. Only observed combinations produce output columns.

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
.venv/bin/python scripts/benchmark_aggregate_survfit.py --columns 100000 --repeat 7
```

The direct Rust tests compare medians across 1–257 members, sorted, reverse,
scrambled and tied inputs, signed zeros, infinities and NaNs. They also cover
uneven groups, unused levels, strided survival matrices, shared scratch across
multistate arrays, invalid public grouping fields, empty data margins and a
declared level product of `2^usize::BITS`. Python tests compare both numerical
components against NumPy for one or several groups and odd or even member
counts. The existing R fixture topic `km-aggregate_survfit` checks reported
survival curves, multistate probabilities and group labels.

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
