# Masked numeric response rows

Formula model frames and person-years calculations recognize masked cells
when a numeric response is supplied as an iterator of NumPy matrix rows or
as mixed reusable rows and scalar cells. Previously, masked constants could
bypass omission and `na.fail`, then emit a warning when converted to NaN.
These inputs now follow the same missing-row rules as a directly supplied
masked matrix or nullable lists. Subsetting precedes missing-row selection.

Numeric response conversion uses R's NA payload for masked cells without
the NumPy masked-to-NaN warning. Genuine numerical NaN remains distinguishable
from R NA. Other
warnings and conversion errors retain their existing behavior. NumPy masked
arrays containing one element retain their supported scalar conversion;
larger malformed scalar cells still fail when retained. Iterator preparation
keeps the existing materialization and stored-input behavior.

`scripts/generate_masked_matrix_formula_reference.R` records 48 independent
matrix cases using R 4.5.3's `stats::model.frame` and unmodified survival
3.8-12 `pyears`, plus eight vector model-frame controls. They cover three
response patterns, two subsets, four missingness actions and two grouping
designs.
The fixture regenerates byte-identically. The 783 Python controls include
direct and iterated masked rows, nullable lists, mixed scalars, masks over
float/integer/logical values, retained inputs, no-mask cases, missing-value
payloads, malformed cells and preservation of unrelated warnings and errors.

## Complete-call measurements

Paired before/after calls isolate the changes in `_coerce.py` and
`_formula.py`; every other Python module and the native extension are shared.
The public model-frame and `pyears(model=True, y=True)` calls include response
preparation, omission, retained results and numerical work. Inputs, including
fresh row iterators, are prepared outside the timer. All returned fields,
retained matrices and row counts are checked for parity, and weighted totals
are checked independently.

Thirty controls cover matrix lists, direct NumPy arrays and fresh row
iterators at 64, 20,000 and 100,000 rows, plus scalar list and NumPy controls for
both public functions. Each trial uses three warmups and nine alternating
before/after pairs with normal garbage collection. The second trial reverses
case and initial execution order. Timings use complete unmasked inputs whose
results agree; previously incorrect masked results are checked separately.
