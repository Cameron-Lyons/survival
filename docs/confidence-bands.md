# Survival confidence bands

`survival.surv_analysis.survfit_confint` computes confidence limits for a vector
of survival estimates. Rust callers use `survival::surv_analysis::survfit_confint`.
The R-style `survival.r_api.survfit_confint` also accepts numeric scalar estimates
and recycles a single standard error or lower-band standard error over the
estimates.

The Python facade preserves missing values in list, NumPy, nullable pandas,
Polars and masked-array inputs. For example:

```python
from survival import r_api

bands = r_api.survfit_confint(
    [0.8, None, 0.3], [0.1, 0.2, 0.15], conf_type="log-log"
)
```

Plain numeric arrays cross both Python interfaces in bulk. The native binding
owns each input buffer before releasing the Python interpreter lock. Strided,
negative-stride, unaligned and read-only arrays retain their logical order;
later changes to the caller's arrays do not affect the result. Returned lower
and upper bounds retain their existing list getters.

The five supported transforms are `plain`, `log`, `log-log`, `logit` and
`arcsin`. `logse=True` means standard errors are on the log-survival scale.
`selow` widens the lower bound, and `ulimit=False` removes the upper cap for
plain and log intervals. The standalone function rejects `conf_type="none"`,
including for an empty estimate vector. Curve fitters separately support
`conf_type="none"` to omit confidence bands.

`scripts/generate_confidence_band_reference.R` records 720 independent
survival 3.8-12 calls across all transforms, two confidence levels, both error
scales, three lower-band options and capped/uncapped limits. Inputs include
endpoints, zero errors, missing estimates/errors, nonfinite values, single
estimates and empty vectors. Tests compare every lower and upper bound through
both Python interfaces with list and NumPy inputs. The source also records R's
warnings for out-of-domain transform inputs. Array ownership, missing
containers, scalar calls and concurrent Python-thread progress have separate
regressions.

## Complete-call measurement

`scripts/benchmark_confidence_bands.py` excludes input creation and includes
input conversion, native calculation and both output-list getters. It records
a hash over every returned bound. Nine samples follow one warmup. Run before
and after a release build; `--extension` can also measure the native interface
with a saved extension of the same Python ABI.

On the local Python 3.14.7 release extension, the native interface took 51–57%
less time for 300,000 estimates and array-valued standard errors/lower errors.
Every returned bound matched the saved predecessor exactly:

| Transform | Before, ms | After, ms |
| --- | ---: | ---: |
| Plain | 29.481 | 12.608 |
| Log | 30.870 | 13.683 |
| Log-log | 35.809 | 17.379 |

These timings include both output-list conversions. They measure the native
Python interface; the facade also avoids per-element array materialization.
Other input layouts, sizes and system load affect timings.

```sh
PYTHONPATH=python .venv/bin/python scripts/benchmark_confidence_bands.py
PYTHONPATH=python .venv/bin/python scripts/benchmark_confidence_bands.py \
  --extension /path/to/saved/_survival.so
PYTHONPATH=python .venv/bin/python -m pytest \
  python/tests/test_confidence_band_reference.py -q
```
