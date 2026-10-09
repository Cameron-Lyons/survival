# Last value carried forward

`survival.r.lvcf` carries non-missing values forward within each subject and
returns a list in the input row order. When `time` is supplied, observations
are ordered within each subject with missing times last. Equal times retain
input order, matching [R's implementation](https://github.com/therneau/survival/blob/master/R/xtras.R).

```python
from survival import r

r.lvcf([1, 1, 1], [None, None, 1], time=[None, 1, 2])
# [1, 0, 1]

r.lvcf([1, 1, 1], [None, None, 1], time=[None, 1, 2], first=False)
# [1, None, 1]
```

The default `first=True` initializes an unknown first observation to zero for
logical and binary numeric columns. Other numeric values, character values
and factor-valued columns retain unknown first observations. Pandas categorical
columns and factors transferred from R preserve this distinction even when
their levels are numeric or logical. Inputs remain unchanged.

The Rust kernel sorts the rows once and computes the source observation for
each result. The wrapper reuses those source indices to identify missing
initial observations. It applies the zero initialization before copying results,
avoiding a second Python sort and keeping subject identities and time ordering
consistent across both steps. The complete call takes `O(n log n)` time and
`O(n)` auxiliary memory.

Run the focused regression checks and benchmark with the installed extension:

```sh
PYTHONPATH=python .venv/bin/python -m pytest python/tests/test_lvcf_ordering.py python/tests/test_r_data_prep.py -q
git show HEAD:python/survival/r/_data_prep.py > /tmp/lvcf-before.py
PYTHONPATH=python .venv/bin/python scripts/benchmark_lvcf.py --rows 100000 --baseline-source /tmp/lvcf-before.py
```

For a before/after comparison after committing, extract the source from the
revision preceding the change. The benchmark checks every output row before
timing and alternates the two wrappers in the same process and native extension.
It includes input conversion, native sorting and copying the returned values.

Local medians of seven samples on Python 3.14.7 and NumPy 2.4.6:

| 100,000 rows, eight observations per subject | Before | After |
| --- | ---: | ---: |
| Ordered rows, `first=True` | 88.59 ms | 41.66 ms |
| Shuffled rows, `first=True` | 145.70 ms | 48.57 ms |
| Ordered rows, `first=False` | 32.81 ms | 32.55 ms |
| Shuffled rows, `first=False` | 42.71 ms | 42.39 ms |

These measurements describe the listed binary-value workloads. Input generation
and output verification are excluded from timing. Missing-time, tied-time,
factor, logical, character and continuous-value cases are checked separately by
the regression tests.
