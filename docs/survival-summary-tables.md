# Survival-summary table performance

`surv_analysis.survmean` computes the compact table used by ordinary curve
summaries and reports: records, risk counts, events, restricted means and their
standard errors, and medians with confidence limits. The Python binding releases
the interpreter lock while the Rust calculation runs.

Restricted-mean rectangles are read directly from the observed curve. The mean
accumulates forward and its variance accumulates backward, retaining the previous
floating-point addition order. A binary search finds the truncation point; median
and confidence-limit searches stop when they find the required crossing or
plateau endpoint. Time scaling is applied to the values being read.

The calculation no longer allocates a scaled copy of the full time vector, four
vectors per restricted mean, or vectors of candidate median rows. Scratch
storage is proportional to the number of curves. Output fields and the existing
R scaling and truncation conventions retain their behavior.

## Verification

`scripts/generate_survmean_reference.R` produces 400 independent stock survival
3.8-12 tables from 20 curve sources. The sources cover grouped, weighted and
counting-process fits, no confidence limits, wholly censored and wholly observed
events, conditional starts, negative times, inserted time-zero rows and plateau
or missing confidence bands. Four scales and five mean options check every
table field. The fixture regenerates byte for byte and is checked in CI.

The local benchmark's 36 configurations produced exactly the same numerical
values as the saved predecessor extension, including undefined quantiles. A
thread-progress test also verifies that the summary calculation releases the
interpreter lock.

## Measurements

`scripts/benchmark_survmean.py` excludes fitting and input construction. Timings
include the native call and result construction, with 11 samples after a warmup.
On a local Python 3.14.7 release extension with 300,000 observed rows and scale 1:

| Curves | Mean option | Before, ms | After, ms |
| ---: | --- | ---: | ---: |
| 1 | none | 1.133 | 0.468 |
| 1 | common | 4.817 | 1.553 |
| 1 | cutoff 30,000 | 1.530 | 0.754 |
| 100 | none | 1.052 | 0.451 |
| 100 | common | 2.521 | 1.509 |
| 100 | cutoff 300 | 1.599 | 0.719 |

These measured calls improved by 40–68%; scale 2.5 cases improved by 38–62%.
Allocation and system load affect timings. These results describe compact
ordinary tables, and exclude fitting, multistate tables and formula-level
event-row preparation.

```sh
PYTHONPATH=python .venv/bin/python scripts/benchmark_survmean.py
PYTHONPATH=python .venv/bin/python scripts/benchmark_survmean.py \
  --extension /path/to/saved/_survival.so
cargo bench --bench survival_benchmarks -- curve_table
Rscript scripts/generate_survmean_reference.R /tmp/survmean.json
PYTHONPATH=python .venv/bin/python -m pytest python/tests/test_survmean_reference.py -q
```
