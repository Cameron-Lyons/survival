"""R's ``rowsum`` groups for ``collapse=`` vectors (``_rowsum_groups``).

Plain integer labels take a numpy path; it must give the codes and names of the
general path, which sorts the distinct labels and names them ``as.character``.
"""

import struct

import numpy as np

from .helpers import setup_survival_import

setup_survival_import()

from survival.r._coerce import _rowsum_groups  # noqa: E402


def same(a, b, path="value"):
    """Equal element types and float bits all the way down."""

    assert type(a) is type(b), f"{path}: {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, float):
        assert struct.pack("<d", a) == struct.pack("<d", b), f"{path}: {a!r} != {b!r}"
    elif isinstance(a, list | tuple):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            same(x, y, f"{path}[{i}]")
    else:
        assert a == b, path


def test_integer_collapse_labels_group_like_the_general_path():
    rng = np.random.default_rng(9)
    for labels in [
        rng.integers(0, 40, 500).tolist(),
        (rng.integers(-5, 5, 300) * 1_000_003).tolist(),
        [7],
        [2**62, -(2**62), 3, 2**62],
    ]:
        fast = _rowsum_groups(labels, len(labels))
        general = _rowsum_groups([np.int64(value) for value in labels], len(labels))
        same(fast, general, "groups")
        assert fast[1] == [str(value) for value in sorted(set(labels))]
    huge = [2**70, 1, 2**70]
    assert _rowsum_groups(huge, 3) == ([1, 0, 1], ["1", str(2**70)])
    booleans = _rowsum_groups([True, False, True], 3)
    assert booleans == ([1, 0, 1], ["FALSE", "TRUE"])
