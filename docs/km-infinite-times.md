# Infinite Kaplan–Meier times

The formula interface, response-based `survfit`, direct `survfitKM`, native
`survfitkm`, and Rust `SurvfitKMData` accept infinite time endpoints. Right
censoring permits either sign; counting-process intervals still require
`start < stop`. Missing times, invalid statuses and nonfinite weights remain
refused by the native fitter. The separate finite-time containers used for
Cox fitting and other routines retain their own contracts.

KM `start.time` also accepts infinities. Negative infinity retains all rows;
positive infinity retains only infinite stop times or reports that all rows
were removed. Missing origins remain invalid. This opt-in is confined to KM;
the shared Python start-time helper retains finite validation by default.

Ordinary curve summaries accept infinite query times, preserve their ordering
and repetitions, and apply the existing `extend` and `dosum` rules. Stacked
curve inputs retain infinite times and their nonmissing origins. Quantiles
keep infinite estimates and bounds, including signed zeros after scaling;
KM probability-zero quantiles retain R's zero origin. Summary tables preserve
IEEE arithmetic for infinite time-zero rows, including undefined areas from
`Inf - Inf`. Finite-origin tables still use the streaming path without
constructing a time-zero copy.

Counting curves with entry reporting can retain entry rows before a supplied
`start.time`. If inserting that origin makes the curve times unsorted, requested
summaries return an input error, as stock R does. This also applies to finite
origins; summaries do not search an invalid time grid.

Time normalization leaves nonfinite endpoints in their original rows while
snapping finite near ties. Stock survival 3.8-12's `aeqSurv` instead applies
finite cutpoints to every endpoint once it finds a near tie. Positive infinity
then becomes the final finite time; a negative-infinity zero index drops a
value before R recycles or reconstructs columns. This can change censoring
times, scramble response rows or produce a dimension error. The port preserves
the endpoints and aligned statuses. Finite inputs retain the existing
cutpoint and interval-collapse rules, with no additional scan.

`scripts/generate_km_infinite_reference.R` records independent R 4.5.3 and
survival 3.8-12 values for 144 fits, 1,296 requested-time summaries, 864 curve
quantiles, 432 summary tables, nine normalized responses and six start-time
calls. The fits cover nine right/counting inputs, ordinary and grouped curves,
unit and noninteger weights, KM/Nelson–Aalen and exponential/Fleming–Harrington
estimates, time fixing, entry counts and both influence matrices. Sixteen fits
need the infinite-endpoint correction. Their expected fits use stock
`aeqSurv` on finite entries only, then stock fitting with time fixing disabled;
raw unmodified stock outputs, errors and warnings remain in the fixture.

The references also retain three stock shape defects beside their explicit
corrections. A summary can lose fitted vectors when its first stratum selects
no times; corrected fields concatenate direct stock single-stratum summaries,
and entirely empty results retain numeric-vector shapes.
Stock `survfit0` leaves unweighted counts at their old row count, and sometimes
adds an unmatched influence column to a curve already at the origin. The
corrected references insert count rows with base R matrix operations and
remove only unmatched influence columns. Fitted estimates and uncertainty
come from stock kernels.

The Python regressions check complete fits through four interfaces, time-zero
curves, summaries through both Python and native paths, quantiles through the
facade/prepared/stacked paths, complete tables, normalization and rejected
missing inputs. Rust integration tests read the same independent references
and exercise public mutation checks. R bridge tests call stock methods directly
from the survival namespace, retain raw stock failures, and apply the documented
endpoint and empty-stratum corrections. The R infinity file checks 2,558
expectations. Full frame references separately check 24 finite/infinite endpoint
cases, preserving valid robust terminal zero uncertainty and confidence bounds.

```sh
Rscript scripts/generate_km_infinite_reference.R
PYTHONPATH=python .venv/bin/python -m pytest -q python/tests/test_km_infinite_times.py
cargo test --test km_infinite_inputs --no-default-features --offline
```
