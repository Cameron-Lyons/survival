# Restricted-mean comparisons

`survival.validation.rmst_comparison` fits a Kaplan–Meier curve for each group,
computes its restricted mean and uncertainty up to `tau`, and compares the
groups with the smallest label as the reference. Rust callers use
`survival::validation::rmst_comparison` directly.

Observation vectors are checked for equal lengths before selecting group rows.
Malformed `status` or `weights` now produce an input error, including extra
trailing values that previously disappeared during subsetting.

The Rust kernel gathers group rows once in sorted label order, preserving their
original order within each group. This replaces sorting a copy of every label
and scanning all observations once per group. Both `rmst_comparison` and
`survmean_curves` accept NumPy vectors through checked bulk conversion, own their
buffers before calculation and release the Python interpreter lock.

The public Rust regressions compare grouped results with separately fitted
weighted curves, including tied times and nonconsecutive group labels. Python
checks misaligned input errors, list/NumPy agreement across contiguous,
negative-stride and strided arrays, and thread progress during both bindings.

## Complete-call measurements

`scripts/benchmark_rmst.py` excludes input creation and includes conversion,
grouping, curve fitting, comparisons and returned-result construction. Nine
samples follow a warmup. Complete result snapshots matched the saved previous
extension exactly on 600,000 rows:

| Groups | Before, ms | After, ms |
| ---: | ---: | ---: |
| 2 | 88.801 | 40.365 |
| 32 | 89.055 | 44.099 |
| 128 | 127.758 | 61.191 |

These calls improved by 51–55% on the local Python 3.14.7 release extension.
The change combines input conversion and grouping improvements; these timings
do not isolate either one. Allocation and system load affect results.

```sh
PYTHONPATH=python .venv/bin/python scripts/benchmark_rmst.py
PYTHONPATH=python .venv/bin/python scripts/benchmark_rmst.py \
  --extension /path/to/saved/_survival.so
PYTHONPATH=python .venv/bin/python -m pytest python/tests/test_rmst_comparison_inputs.py -q
cargo test --offline --no-default-features --test rmst_inputs
```
