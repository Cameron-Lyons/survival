# Python matrix inputs

Cox and AFT fitting and prediction accept NumPy arrays, lists, and tuples.
Their `FloatMatrix` boundary copies values into an owned Rust matrix in row
order. NumPy arrays are read directly, including Fortran-contiguous and
strided layouts, without converting elements to Python lists. Other array
dtypes are normalized to float64 before copying.

Lists and tuples containing ordinary Python float rows now fill one matrix
allocation directly. They previously allocated a Rust vector for every row,
then flattened those vectors into another allocation. Integer values, custom
numeric objects, sequence subclasses, and array-valued rows continue through
the general conversion path. Invalid shapes retain their existing errors.
The direct path does not call custom conversion methods before falling back.

This changes input conversion only. Kernels receive the same owned values;
the NumPy input path and numerical algorithms are unchanged. The ownership
allows fits to retain their training data and run with the Python GIL released.

Run the release-build benchmark with:

```sh
PYTHONPATH=python .venv/bin/python scripts/bench_matrix_input.py
```

It measures `SurvregData` construction, including vector and matrix conversion
and validation. Inputs are prepared before timing, model fitting is excluded,
and each result is the median of 11 calls. It exercises list, tuple, NumPy,
Fortran, and strided inputs at several matrix sizes.

Local release measurements on Linux x86-64 with Python 3.14.7 and NumPy 2.4.6:

| Matrix shape | Input | Before | After |
| --- | --- | ---: | ---: |
| 100,000 × 4 | List of float rows | 8.318 ms | 2.370 ms |
| 100,000 × 4 | Tuple of float rows | 8.371 ms | 2.468 ms |
| 100,000 × 16 | List of float rows | 20.221 ms | 6.628 ms |
| 100,000 × 16 | Tuple of float rows | 21.503 ms | 6.621 ms |

These cases improved by 3–3.5 times. NumPy timings stayed near 0.55–0.59 ms
for the 100,000 × 4 contiguous case and 1.59–1.64 ms for 100,000 × 16.
Fortran and strided timings were also effectively unchanged. These are
before/after measurements of this implementation, not comparisons with R.

Tests verify matrix ordering and empty/ragged shapes, compare list and NumPy
model fits, and check custom numeric conversions and row iteration. The
complete R reference suite exercises the converted matrices through fitting
and prediction.
