"""Data-preparation regressions against R 4.5.3 / survival 3.8-12 (values hard-coded)."""

import importlib

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
r_data_prep = importlib.import_module("survival.r._data_prep")


# --- survcondense -------------------------------------------------------------


def test_survcondense_compares_covariates_with_r_equality():
    # survcondense(Surv(t1, t2, s) ~ x, d, id = id) with x = c(0.3, 0.1 + 0.2, 0.1 + 0.2):
    # 0.3 != 0.1 + 0.2, so R keeps (0, 1] and (1, 3]
    data = {"id": [1, 1, 1], "t1": [0, 1, 2], "t2": [1, 2, 3], "s": [0, 0, 1]}
    data["x"] = [0.3, 0.1 + 0.2, 0.1 + 0.2]
    frame = r.survcondense("Surv(t1, t2, s) ~ x", data, id="id")
    assert frame["x"] == [0.3, 0.1 + 0.2]
    assert frame["t1"] == [0.0, 1.0]
    assert frame["t2"] == [1.0, 3.0]
    assert frame["s"] == [0, 1]


def test_survcondense_reads_the_kernel_start_times_once(monkeypatch):
    # the start column is gathered from one read of the kernel's result, not one per row
    reads = []
    kernel = r_data_prep._core.survcondense

    class Counting:
        def __init__(self, result):
            self.keep = result.keep
            self._start = result.start

        @property
        def start(self):
            reads.append(1)
            return self._start

    monkeypatch.setattr(
        r_data_prep._core,
        "survcondense",
        lambda *args: Counting(kernel(*args)),
        raising=False,
    )
    data = {
        "id": [1, 1, 1, 2, 2, 3],
        "t1": [0, 2, 5, 0, 3, 0],
        "t2": [2, 5, 8, 3, 6, 4],
        "s": [0, 0, 1, 0, 0, 1],
        "x": [1, 2, 3, 4, 5, 6],
    }
    frame = r.survcondense("Surv(t1, t2, s) ~ x", data, id="id")
    assert frame["t1"] == [0.0, 2.0, 5.0, 0.0, 3.0, 0.0]
    assert len(reads) == 1
