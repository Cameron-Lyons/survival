# Survival-response repetition

`survival.r.rep_surv` follows `rep.Surv`'s row repetition rules while retaining
the response type, normalized event codes, multistate levels, and censoring
label. It is also available at the package root.

`each` first repeats individual rows. A scalar `times` then repeats that whole
sequence; a vector `times` supplies a count for each entry of the expanded
sequence. A finite `length_out` (`length.out`) takes precedence over `times` and
recycles/truncates the expanded sequence to the requested length.

```python
from survival import r

x = r.Surv([1, 3, 5], [0, 1, 0])
r.rep_surv(x, each=2, length_out=5).time
# (1.0, 1.0, 3.0, 3.0, 5.0)

r.rep_surv(x, times=[0, 2, 1]).time
# (3.0, 3.0, 5.0)
```

Repetition counts truncate toward zero before checking for negativity, so
`-0.5` produces zero copies and `-1` is invalid. `each` and `length_out` use the
first element of vector arguments and issue R's first-element warning for
empty or multiple-element vectors. Missing/nonfinite values take the default:
missing `each` means one, while missing `length_out` leaves `times` in effect.
Numeric character controls are accepted; an unconvertible `each` or
`length_out` emits the coercion warning and uses that same missing-value rule.
Invalid `times` still raises R's error.

Validation and warning order follow base R, including validation of
`length_out` before `each` and ignoring `times` when a finite output length is
supplied. Python's `None` and nullable pandas missing scalars represent missing
controls. An empty response stays empty, retaining the existing documented
difference from R's `1:nrow(x)` sequence for zero rows.

## Allocation and implementation

Scalar whole-sequence repetition uses the immutable normalized column tuples
directly. Elementwise/vector-count repetition uses NumPy object storage, which
retains the existing float/int objects and missing statuses while constructing
each result column. This avoids an output-sized list of row indices and its
Python integer objects. No response is passed back through the `Surv`
constructor, so retained status 2 or 3 is never reinterpreted as ordinary 1/2
coding.

Length-limited repetition builds only the required complete cycles and prefix.
For example, `each=1e12, length_out=1` constructs one result row without expanding
a trillion-entry intermediate. Repeated responses are independent objects;
their shared tuples/scalars are immutable, and exported matrices remain owned
snapshots.

## Stock references and measurements

[`generate_surv_repetition_reference.R`](../scripts/generate_surv_repetition_reference.R)
records 392 independent R 4.5.3 / survival 3.8-12 calls across ordinary right,
missing right, left, counting-process, interval, and both multistate response
types. The fixture retains complete result matrices, metadata, warning sequences,
and errors. [`test_surv_repetition.py`](../python/tests/test_surv_repetition.py)
checks each case with Python lists and NumPy controls: 784 differential checks,
plus ownership, empty-response, and nullable-control regressions. The focused
response/vector suite passed 1,974 tests.

[`benchmark_surv_repetition.py`](../scripts/benchmark_surv_repetition.py) measures
complete calls on a 100,000-row counting-process response. Response and control
creation are excluded; coercion, repetition, and returned-response construction
are included. Seven samples follow three warmups, with alternating call order.
All result columns and metadata match the saved predecessor. Peak traced
allocations are measured separately from timing and include the returned
response, temporary Python containers, and NumPy buffers; they are not process
RSS. On Python 3.14.7 / NumPy 2.4.6:

| Operation | Output rows | Before, ms | After, ms | Before peak, MiB | After peak, MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scalar times = 3 | 300,000 | 35.524 | 2.566 | 28.21 | 6.87 |
| Scalar each = 3 | 300,000 | 18.209 | 9.720 | 22.12 | 9.16 |
| Per-row counts | 99,999 | 18.823 | 13.409 | 9.91 | 4.58 |
| Each = 2 and per-entry counts | 199,999 | 37.085 | 23.768 | 19.93 | 8.42 |
| Each = 3 with cycling length | 700,001 | 67.227 | 20.237 | 65.29 | 21.36 |
| Each = 3 with five-row prefix | 5 | 0.0044 | 0.0060 | 0.0019 | 0.0020 |

The measured large outputs take 29–93% less time and 54–76% fewer peak traced
allocations. The tiny prefix call adds about 1.6 microseconds for the fuller
argument handling. These measurements describe the specified cases, not every
response size or repetition pattern.

```sh
git show <previous-revision>:python/survival/r/_surv_vector.py > /tmp/previous-surv-vector.py
PYTHONPATH=python .venv/bin/python scripts/benchmark_surv_repetition.py \
  --baseline-source /tmp/previous-surv-vector.py
PYTHONPATH=python .venv/bin/python -m pytest \
  python/tests/test_surv_repetition.py python/tests/test_surv_vector_methods.py \
  python/tests/test_surv_operations.py python/tests/test_r_surv.py \
  python/tests/test_response_normalization.py -q
```
