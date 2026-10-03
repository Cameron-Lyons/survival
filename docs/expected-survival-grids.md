# Expected survival on dense time grids

`survival.population.survexp` and the rate-table branch of `survival.r.survexp`
select requested times with one forward scan of the fitted grid. Previously,
each request searched from the start of the grid. Selection now takes
O(fitted times + requests), rather than O(fitted times × requests), plus the
output copy for each group. Rate-table integration and its numerical conventions
are unchanged. Cox-based expected curves already use ordered or binary lookups.

Duplicate requests retain separate rows, including signed zeros. Observed
follow-up times can insert additional fitted rows between requests; selection
still finds the exact requested time without merging nearby distinct times.
Scaling divides reported times after selection.

Run the Python benchmark against a release extension:

```sh
PYTHONPATH=python .venv/bin/python scripts/benchmark_survexp_grid.py \
  --sizes 1000 4000 8000 16000 32000 --repeats 7
```

Add `--api formula` to include formula preparation and conversion to the
R-style result. These are separate complete-call measurements.

It measures complete native Python calls, including conversion and result
construction, for one subject and a constant population hazard. Each sample
checks every time, risk count and survival probability against the analytic
curve. It also checks repeated requests and controls with short grids and
100 or 1,000 subjects. Formula-frame preparation and later NumPy snapshots
are outside the timed call. This workload isolates time selection; workloads
dominated by population integration can show a smaller improvement.

Local release-extension measurements on Python 3.14.7, with medians of seven
samples, were:

| Native Python workload | Before | After |
| --- | ---: | ---: |
| 1,000 requested times | 0.50 ms | 0.17 ms |
| 4,000 requested times | 5.51 ms | 0.59 ms |
| 16,000 requested times | 80.16 ms | 2.19 ms |
| 32,000 requested times | 316.20 ms | 4.20 ms |
| 32,000 distinct times, each requested twice | 631.98 ms | 8.09 ms |
| 1,000 subjects, 25 requested times | 0.29 ms | 0.28 ms |

Short-grid controls with one or 100 subjects were also close to their previous
timings. The largest improvement comes from avoiding repeated grid searches;
result allocation and copying still grow with the number of requested rows.

Using `--api formula`, 32,000 requested times measured 290.27 ms before and
12.00 ms after; doubling each request measured 580.64 ms and 24.98 ms.
The one-subject, 25-time formula control measured 0.21 ms and 0.24 ms.

Rust benchmarks validate the same analytic curve before timing:

```sh
cargo bench --bench survival_benchmarks -- expected_survival_grid
```

`SurvExpResult.cumhaz` computes `-log(surv)` with R's numerical domain behavior:
zero survival gives positive infinity; missing or negative survival gives NaN.
In particular, Cox expected curves with an exhausted group retain their
undefined hazards through result access and serialization.

`scripts/generate_survexp_grid_reference.R` regenerates the stock-R grid
references; `scripts/generate_expected_cumhaz_reference.R` regenerates the
logarithm and exhausted-group references. Python tests check both the formula
interface and native inputs, with separate checks for summary and report
propagation of missing survival probabilities.
