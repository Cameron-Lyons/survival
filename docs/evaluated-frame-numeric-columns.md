# Numeric columns in evaluated model frames

Cox/AFT model.matrix now uses primitive array dtypes when classifying supplied
evaluated scalar columns. Integer, unsigned integer and floating arrays without
declared factor levels skip the per-value boolean and string scans. The shared
constructor still checks row counts, fitted contrasts and explicit logical or
character metadata, and uses the same value conversion and matrix assembly.
Boolean arrays remain categorical. Object, masked and factor-like inputs retain
their established conversion behavior.

The change is confined to Python type detection. It adds no public API or
serialization field, and does not change formula evaluation, omission, factor
coding, matrix bases, predictions or fitting. The R adapter and all Rust/Cargo
inputs are unchanged. It addresses the repeated classification identified in
#715; broader package parity and performance remain open.

## Validation

The unmodified survival 3.8-12 fixture from #715 remains unchanged. Its 212
models and 23,744 matrix reference calls cover 28 input variants, four standard
missing-data actions, numeric/logical/factor retyping, changed factor levels,
contrasts, interactions, strata, transforms, offsets, ridge/spline/frailty bases,
missing values, empty inputs and expected failures.

An additional 212 tests apply four array layouts to 53 Cox/AFT formulas with
Unicode row names and uncached fitted matrices: strided, big-endian, read-only,
and integer arrays for exactly integral finite columns. Numeric kind metadata
is removed so the source type determines classification. Complete nonmissing
logical columns also use boolean arrays without kind metadata. Under na.fail,
5,724 tagged-frame calls compare against the existing independent stock
outputs: 5,388 matrices and 336 errors. Comparisons retain whole values,
dimensions, names, assignments, contrasts, strata/levels, separate R NA/NaN
masks, warning text and error causes. Read-only inputs exercise ownership.

The full Python suite passes 38,012 tests, with 48 skips, 37 existing expected
numerical differences and 13 existing warnings. All 2,759 evaluated-frame tests
run in that suite. Pinned Ruff 0.16.9 lint/format, the 47-file type check and
generated stubs/manifest checks pass.

Sixteen transformed models saved by #713, #714 and #715 restore unchanged
stored matrices and predictions in a separate process. Their evaluated
matrices match whole stock outputs under all four actions: 64 calls, including
models fitted after omitting training rows. Stored matrices remain unchanged
after those calls.

The full R source archive check against this Python implementation passes
468,368 assertions with zero test failures, warnings or skips and no check
errors, warnings or notes. All 58 R source/test files match the archive and
its newly checked sources byte for byte.
The release extension is unchanged from #707, with SHA-256
`8a83b9ace4231f4f8fc7b78a8ab1be564cc5c1abfe4107962945c663014eb4fc`.
Its native gates remain the latest native validation and are not rerun here.

## Complete public calls

The existing `scripts/benchmark_evaluated_model_frames.R` compares #715
(`1ecae911`) and this change in separate R processes with the same R archive
and release extension. Fits use 20,000 rows; new inputs use 10,000 rows. Scalar,
16-transform/logical/stratum/offset and two-column ridge designs cover Cox and
AFT stored matrices, raw frames, evaluated frames and evaluated frames with
missing values. Whole outputs, attributes, warnings and NA/NaN masks are
checked against stock before every workload is timed.

Round one runs baseline/current; round two reverses that order after the full
validation gates. No validation runs concurrently with the timed processes.
Each workload uses three warmups and nine samples per round. Timing includes
the public matrix call, conversion, construction, metadata and warning/error capture;
fitting, input setup, option changes and explicit GC are excluded. Stored calls
omit the data argument, following stock AFT's omission versus explicit NULL
distinction.

Milliseconds, median (minimum–maximum), for two sequential rounds.

| Family/design | Call | #715, round 1 | This change, round 1 | #715, round 2 | This change, round 2 |
| --- | --- | --- | --- | --- | --- |
| Cox / scalar | Stored | 29 (27–36) | 27 (26–33) | 22 (21–29) | 22 (22–29) |
| Cox / scalar | Raw frame | 17 (17–18) | 16 (15–17) | 13 (13–14) | 13 (13–14) |
| Cox / scalar | Evaluated frame | 16 (15–16) | 14 (14–14) | 12 (12–13) | 11 (11–12) |
| Cox / scalar | Evaluated frame with NA | 16 (15–16) | 13 (13–14) | 12 (11–12) | 11 (11–12) |
| Cox / transformed | Stored | 45 (42–61) | 44 (41–58) | 35 (34–49) | 37 (35–53) |
| Cox / transformed | Raw frame | 42 (42–43) | 40 (39–43) | 32 (32–33) | 35 (34–36) |
| Cox / transformed | Evaluated frame | 64 (61–66) | 41 (40–43) | 49 (48–49) | 37 (36–37) |
| Cox / transformed | Evaluated frame with NA | 65 (64–65) | 42 (41–43) | 49 (48–59) | 36 (35–38) |
| Cox / ridge | Stored | 29 (28–30) | 27 (26–28) | 22 (22–23) | 23 (22–23) |
| Cox / ridge | Raw frame | 19 (17–19) | 17 (16–19) | 14 (14–14) | 14 (13–14) |
| Cox / ridge | Evaluated frame | 23 (23–31) | 21 (20–31) | 18 (18–24) | 18 (17–24) |
| Cox / ridge | Evaluated frame with NA | 23 (22–23) | 21 (19–23) | 18 (18–18) | 18 (17–18) |
| AFT / scalar | Stored | 30 (29–39) | 26 (24–34) | 23 (23–31) | 23 (23–29) |
| AFT / scalar | Raw frame | 17 (17–18) | 16 (16–17) | 13 (13–14) | 13 (13–14) |
| AFT / scalar | Evaluated frame | 16 (15–23) | 14 (13–14) | 13 (12–13) | 11 (11–12) |
| AFT / scalar | Evaluated frame with NA | 16 (15–17) | 14 (13–15) | 12 (12–13) | 12 (11–12) |
| AFT / transformed | Stored | 53 (50–67) | 48 (45–64) | 42 (40–56) | 41 (38–53) |
| AFT / transformed | Raw frame | 39 (38–40) | 36 (35–38) | 30 (30–32) | 31 (31–32) |
| AFT / transformed | Evaluated frame | 48 (47–50) | 27 (27–28) | 38 (37–39) | 23 (22–24) |
| AFT / transformed | Evaluated frame with NA | 48 (46–49) | 28 (27–30) | 38 (37–39) | 23 (23–24) |
| AFT / ridge | Stored | 31 (29–32) | 29 (28–30) | 24 (23–25) | 25 (24–25) |
| AFT / ridge | Raw frame | 18 (18–19) | 17 (16–19) | 14 (14–15) | 14 (13–14) |
| AFT / ridge | Evaluated frame | 24 (24–35) | 23 (22–31) | 19 (19–27) | 18 (18–26) |
| AFT / ridge | Evaluated frame with NA | 24 (24–25) | 23 (21–24) | 19 (18–20) | 19 (18–19) |

Complete transformed-frame medians fall from 64 to 41 ms for Cox and 48 to
27 ms for AFT in round one, and from 49 to 37 ms and 38 to 23 ms in the
reverse-order round. Incomplete transformed calls also improve in both rounds.
Their measured ranges do not overlap within either round. Across these four
workloads and two rounds, medians are 24–44% lower. This describes these supplied
evaluated-frame workloads, not package-wide performance. Raw/stored controls
vary too: round two's transformed raw calls take 32/30 ms for the baseline and
35/31 ms for this change. Scalar and ridge differences are small and often
overlap. Results should not be interpreted as a speedup for those controls.

Measurements use an Intel Core Ultra 5 325, Python 3.14.7, NumPy 2.4.6,
R 4.5.3, survival 3.8-12 and reticulate 1.47.0. The extension was built with
Rust 1.94.0. Smaller differences in scalar/ridge and raw/stored controls do not
establish a general speedup. Stock remains faster; its #715 timing table is in
[evaluated model-frame matrices](evaluated-model-frame-matrices.md).

```sh
# Select #715's Python source for the baseline and this source for current.
# Both runs use the unchanged checked R archive and release extension.
Rscript scripts/benchmark_evaluated_model_frames.R 20000 9 baseline
Rscript scripts/benchmark_evaluated_model_frames.R 20000 9 current
PYTHONPATH=python .venv/bin/python -m pytest python/tests -q
```

The port's own model.frame output still returns raw columns without stock's
evaluated-column schema and terms attribute. Custom global missing-data
actions, tagged lists, ordered/nondefault fitted contrasts, time-transform and
multistate frames, matrix-valued interactions, stored-method overrides and
broader formula/penalty families remain outside this change.
