# Typed formula expressions

The Python R interface evaluates a restricted set of R vector expressions in
formula calls, including `I()`, `identity()`, `as.numeric()`, and `offset()`.
For example:

```python
from survival import r

fit = r.coxph(
    "Surv(time, status) ~ x + I(z > 0 & x <= 1) + offset(z > 0)",
    data,
    model=True,
)
```

Expressions follow R's precedence: powers bind before unary signs, then
multiplication and division, addition and subtraction, comparisons, logical
negation, `&`, and `|`. Thus `!z > 0` means `!(z > 0)`, and `-z^2` means
`-(z^2)`. Outside expression calls, formula operators continue to describe
model terms and interactions.

Logical evaluation uses R's three-valued rules. `FALSE & NA` is `FALSE`, and
`TRUE | NA` is `TRUE`; the other uncertain cases remain missing. Missing-row
actions inspect the evaluated model variables. A missing source value does
not remove a row when its evaluated logical expression is known, unless
another model variable requires that source value.

`I()` and `identity()` preserve the input type and factor levels through nested
calls. Logical terms use treatment contrasts with a `TRUE` column; arithmetic
and `as.numeric()` produce numeric columns. Factor identity retains declared
levels, and numeric factor conversion uses their one-based level codes.
Logical offsets retain their logical model-frame values and contrast metadata,
while the numerical fitter receives their zero/one values. Empty and entirely
missing prediction vectors retain the fitted design's logical or factor type.

Factor comparisons follow [R's factor methods](https://stat.ethz.ch/R-manual/R-patched/library/base/html/factor.html).
Equality compares labels and requires matching declared level sets when both
operands are factors; level order may differ. Ordered factors support all six
comparisons using their declared level order, and ordering two ordered factors
requires the same levels in the same order. A plain comparison operand is
matched to those levels, with unknown labels becoming missing. Ordering an
unordered factor warns and returns missing values. Mixed ordered and unordered
factors retain R's incompatible-method warning and integer-code comparison.
`factor()` preserves an input factor's ordering while dropping unused levels.

Ordered pandas categoricals and R factors use R's default `contr.poly` design
coding: columns `.L`, `.Q`, `.C`, then `^4`, `^5`, and so on. Scores are the
declared level positions, independent of numeric labels or observed counts.
Bare factors and identities retain unused levels; `factor()` drops them before
subsetting and missing-row omission. Interactions retain R's margin rules, so
full factor coding uses indicator columns. Fitted matrices, predictions and
pickle keep the training basis even when new data changes category order or
supplies plain character values. Explicit R contrast matrices, named contrast
attributes, and configured R contrast defaults take precedence over polynomial
coding and retain their fitted output metadata. The default polynomial basis
supports up to 95 levels, as in R.

The basis follows R's [limited-pivot LINPACK QR](https://github.com/wch/r-source/blob/R-4-5-branch/src/appl/dqrdc2.f)
and applies only its numerically independent Householder reflections, as
[`contr.poly`](https://github.com/wch/r-source/blob/R-4-5-branch/src/library/stats/R/contr.poly.R)
does. This matters once high powers lose numerical rank: a complete LAPACK Q
can produce a different basis. Frozen base-R references cover two through 24
levels, including rank loss and column cycling. Higher degree columns are
ill conditioned and can differ substantially between BLAS implementations,
including changes in their signs and orthogonal complement. The port keeps a
finite orthonormal basis through 95 levels; exact high-degree coefficients
across platforms are not guaranteed. Explicit contrast matrices retain the
chosen numerical basis when that is required.

Integer and logical operands retain integer results for unary signs and
`+`, `-`, and `*`. Values outside R's integer range become `NA` with its
integer-overflow warning, so overflow participates in row omission and
exclusion. A double operand promotes arithmetic to double; `/`, `^`, and
`as.numeric()` produce doubles. The integer or double type survives empty and
entirely missing vectors and nested identities.

Quoted literals decode R's common control escapes, octal and hexadecimal byte
escapes, and `\u`/`\U` Unicode escapes, including braced forms and paired
surrogates. Escaped comparison literals also receive R's decoded matrix and
prediction labels. Invalid escapes, embedded NULs, mixed Unicode and byte
escapes, and Unicode escapes inside backtick names are rejected. R byte strings
that are not valid UTF-8 and unpaired surrogate escapes are explicitly
unsupported; the port refuses them instead of changing their values.

The parser accepts only the supported operators and named functions; it does
not execute Python or arbitrary R code. Dependency discovery and evaluation
share a cached expression tree. Each model-frame variable is evaluated before
subsetting and missing-row removal, then its values and type metadata are
selected together. Prediction reuses the fitted levels and contrasts, including
after copying or saving a model. Native evaluated model frames reuse their
supplied variable values.

`scripts/generate_typed_formula_expression_reference.R` records stock R
survival 3.8-12 expression values and types, model frames, matrices, coefficients,
covariances, and training and new-data predictions. Cases include all nine
logical input pairs, precedence, nested identities, logical and numeric offsets,
factor levels, repeated subset rows, missing-row actions, and empty prediction
data. Integer overflow cases check changed fitted rows and missing predictions;
escape cases check UTF-8 bytes, parser failures and fitted labels. Raw R errors,
warnings and unsupported byte-string values remain in the reference. Where stock AFT
prediction drops new-data offsets, the reference also derives the intended
prediction from the fitted design; see [R compatibility](r-compatibility.md).
Stock AFT term prediction errors on empty new data; the port retains the fitted
empty matrix schema recorded from a successful stock term prediction.

The R bridge retains missing raw logical source values as R `NA` when their
evaluated expressions recover the row. Its flattened source frames preserve
declared logical and factor types and fitted row labels, including entirely
missing logical columns. Live R tests also check matrices, predictions, and
model serialization against the unmodified stock package.

CI regenerates this reference with pinned R 4.5.3 and survival 3.8-12 and checks
every field within the documented numerical tolerance. The frozen references
also run against installed wheels on macOS and Windows.

`python/tests/test_factor_formula_comparisons.py` additionally checks these
factor-dispatch rules against the base R methods and compares complete Cox and
AFT fits, matrices, and predictions with equivalent explicit logical designs.
These additional checks use source-derived expectations, separate from the
generated stock-R fixture above.
`python/tests/test_ordered_factor_contrasts.py` uses independent closed-form
polynomial values for two through five levels, equivalent numeric Cox/AFT fits,
and checks of unused levels, subsetting, interactions, explicit contrasts and
saved models. The R bridge test `test-ordered-factor-contrasts.R` compares the
same behavior with stock survival when an R test environment is available.
`scripts/generate_ordered_factor_reference.R` regenerates the additional
base-R polynomial reference; high-level tests also check finiteness,
normalization, orthogonality and removal of the constant column.

Measure preparation independently of numerical fitting with:

```sh
.venv/bin/python scripts/benchmark_typed_formulas.py --rows 100000 --repeat 7
```

The script checks numeric matrices, declared types, offset values and missing
masks before reporting wall and process CPU times. An archived Python source
tree can be compared with `--python-path PATH --legacy`, using the same native
extension; `--cpu N` pins a Linux run to one allowed CPU.

In a local seven-sample comparison of 100,000 rows, process CPU medians for
additive NumPy frame construction and matrix preparation stayed within 1% of
the previous implementation. Comparison-term preparation fell from 148.85 ms
to 92.64 ms. Retaining the typed raw source frame separately took 8.25 ms versus
6.36 ms, reflecting its additional type metadata and list copies. Concurrent
builds affected wall timings, so these preparation measurements use process
CPU medians and do not claim faster complete model fitting.
