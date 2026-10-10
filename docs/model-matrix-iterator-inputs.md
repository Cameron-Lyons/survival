# Model-matrix iterator inputs

Fresh Cox model matrices prepare iterator columns before inspecting frailty
groups. Previously, the frailty prepass consumed a group iterator before the
model-frame pass, which then reported an incorrect group length. Sparse and
dense frailty designs now use the same retained values throughout the call.
Used aliases retain their shared identity, categorical groups retain their
declared levels, and unrelated iterator columns remain unread.

Stored matrices and already evaluated model frames retain their existing
dispatch. Dense matrices retain fitted group contrasts; sparse groups are
recoded before omission. Fresh matrices follow the
[model-matrix missing-data rules](model-matrix-na-action.md).

`scripts/generate_model_matrix_iterator_reference.R` records 17 independent
stock survival 3.8-12 matrix controls, including a dense single-group error.
The 173 Python controls compare complete
values, dimensions, column and row names, assignments, contrasts and strata.
They cover sparse, dense and automatic frailty selection; numeric and
categorical groups; reordered and unused levels; incomplete rows; shared
columns; custom row names; lists, NumPy, pandas and counted iterators;
unused columns; retained and rebuilt frames; input-storage snapshots and serialization.
The fixture regenerates byte-identically.

## Complete-call measurements

The paired benchmark isolates the two-line model-matrix preparation change.
Both implementations use the same final formula, coercion, fitting and native
modules. Complete public calls include input preparation, omission, design
construction and reading the returned values and metadata. Fitting and input
setup are outside the timer. Complete output and stored-matrix parity are
checked before and throughout sampling.

Twenty controls cover sparse frailty, dense frailty and ordinary Cox designs
with list, NumPy and pandas inputs at 64 and 20,000 rows, plus sparse NumPy
frames with six unused columns. Each trial uses three warmups and nine
alternating before/after pairs with normal garbage collection; the second
trial reverses case and initial execution order. Incorrect iterator calls in
the predecessor are correctness controls rather than comparable timings.
