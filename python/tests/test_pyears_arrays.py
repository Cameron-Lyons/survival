"""Person-years fixed categories, bulk arrays and typed matrix inputs."""

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
core = survival.population


@pytest.mark.parametrize("layout", ["list", "c", "f", "strided", "readonly"])
@pytest.mark.parametrize("weighted", [False, True])
def test_fixed_categories_match_an_independent_time_bin(layout, weighted):
    categories = np.array([[1, 1], [2, 2], [3, 1], [1, 2], [3, 2]], dtype=float)
    stepped = np.column_stack((categories, np.zeros(5)))
    if layout == "f":
        categories = np.asfortranarray(categories)
    elif layout == "strided":
        categories = np.repeat(categories, 2, axis=1)[:, ::2]
    elif layout == "readonly":
        categories.flags.writeable = False
    elif layout == "list":
        categories = categories.tolist()
    args = {
        "stop": [10, 5, 3, 7, 9],
        "start": [0, 2, 3, 1, 8],
        "event": [1, 0, 1, 2, 1],
        "scale": 2,
    }
    if weighted:
        args["weights"] = [2, 0, 3, -1, 0.5]
    actual = core.pyears(
        **args, factors=[1, 1], dims=[3, 2], cuts=[[], []], categories_data=categories
    )
    expected = core.pyears(
        **args,
        factors=[1, 1, 0],
        dims=[3, 2, 1],
        cuts=[[], [], [-1000, 1000]],
        categories_data=stepped,
    )
    arrays = actual.to_arrays()
    for name in ["pyears", "n", "event"]:
        np.testing.assert_array_equal(arrays[name], getattr(expected, name))
    np.testing.assert_array_equal(arrays["dims"], [3, 2])
    assert arrays["observations"] == 5
    assert arrays["expected"] is None
    original = actual.pyears
    arrays["pyears"][0] = -999
    assert actual.pyears == original
    del actual
    assert arrays["pyears"][0] == -999


def test_scalar_cell_and_zero_time_keep_events_without_exposure():
    output = core.pyears([0, 3], event=[1, 2], weights=[2, 4], scale=1).to_arrays()
    np.testing.assert_array_equal(output["pyears"], [12])
    np.testing.assert_array_equal(output["event"], [10])
    np.testing.assert_array_equal(output["n"], [1])
    np.testing.assert_array_equal(output["dims"], [])
    assert output["offtable"] == 0


@pytest.mark.parametrize("dims", [[2**63, 4], [2**61, 1]])
def test_category_product_is_checked_before_allocation(dims):
    with pytest.raises(ValueError, match="addressable memory"):
        core.pyears([1], factors=[1, 1], dims=dims, cuts=[[], []], categories_data=[[1, 1]])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_rate_table_constructor_reads_owned_numpy_rates(dtype):
    source = np.repeat(np.array([0, 1], dtype=dtype), 2)[::2]
    source.flags.writeable = False
    table = core.RateTable([2], ["group"], [["a", "b"]], [None], [1], source)
    assert table.rates == [0, 1]
    source.flags.writeable = True
    source[0] = 7
    assert table.rates == [0, 1]
    output = core.pyears(
        [2, 3], ratetable=table, ratetable_positions=np.array([[1], [2]]), scale=1
    ).to_arrays()
    np.testing.assert_array_equal(output["expected"], [3])
    assert output["event"] is None
