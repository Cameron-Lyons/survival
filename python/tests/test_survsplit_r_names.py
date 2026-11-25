"""survSplit's column names and kept columns, pinned to survival 3.8-12's output."""

import pytest
from survival import r

VETERAN = {"time": [72.0, 411.0, 228.0], "status": [1.0, 1.0, 1.0], "age": [69.0, 64.0, 38.0]}


def test_episode_and_added_names_go_through_make_names():
    # survSplit(Surv(time, status) ~ age, data = veteran[1:3, ], cut = 100,
    #           episode = "my ep", added = "was added")
    split = r.survSplit(
        "Surv(time, status) ~ age", VETERAN, cut=[100], episode="my ep", added="was added"
    )
    assert list(split) == ["age", "tstart", "time", "status", "my.ep", "was.added"]
    assert split["my.ep"] == [1, 1, 2, 1, 2]
    assert split["was.added"] == [False, True, False, True, False]
    surv = r.Surv([72.0, 411.0], [1, 1])
    assert list(r.survSplit(surv, {"age": [69.0, 64.0]}, cut=[100], episode="1st")) == [
        "age",
        "tstart",
        "tstop",
        "event",
        "X1st",
    ]


def test_episode_and_added_must_be_strings():
    with pytest.raises(ValueError, match="episode must be a character string"):
        r.survSplit("Surv(time, status) ~ age", VETERAN, cut=[100], episode=1)
    with pytest.raises(ValueError, match="added must be a character string"):
        r.survSplit("Surv(time, status) ~ age", VETERAN, cut=[100], added=True)


def test_an_id_already_in_the_data_is_not_added():
    data = {**VETERAN, "sid": [5.0, 6.0, 7.0]}
    # R names: age tstart time status; the data's sid is not a model variable.
    split = r.survSplit("Surv(time, status) ~ age", data, cut=[100], id="sid")
    assert list(split) == ["age", "tstart", "time", "status"]
    # A '.' right-hand side keeps every data column, sid among them.
    split = r.survSplit("Surv(time, status) ~ .", data, cut=[100], id="sid")
    assert list(split) == ["time", "status", "age", "sid", "tstart"]
    assert split["sid"] == [5.0, 6.0, 6.0, 7.0, 7.0]
    # An id the data lacks is created, numbered by data row.
    split = r.survSplit("Surv(time, status) ~ age", VETERAN, cut=[100], id="newid")
    assert list(split) == ["age", "newid", "tstart", "time", "status"]
    assert split["newid"] == [1, 2, 2, 3, 3]


def test_names_follow_surv_arguments():
    lung = {"time": [306.0, 455.0], "status": [0.0, 1.0], "age": [74.0, 68.0]}
    split = r.survSplit("Surv(event=status, time=time) ~ age", lung, cut=[200])
    assert list(split) == ["age", "tstart", "time", "status"]
    assert split["time"] == [200.0, 306.0, 200.0, 455.0]
    assert split["status"] == [0, 0, 0, 1]
    assert list(r.survSplit("Surv(time, status == 1) ~ age", lung, cut=[200])) == [
        "age",
        "tstart",
        "time",
        "event",
    ]
    assert list(r.survSplit("Surv(time/30, status) ~ age", lung, cut=[10])) == [
        "age",
        "tstart",
        "tstop",
        "status",
    ]


def test_timefix_runs_before_a_zero_vector_becomes_the_start():
    data = {"time": [10.0, 20.0], "status": [1.0, 0.0]}
    # survSplit(Surv(time, status) ~ ., d, cut = 2 + 1e-12, zero = c(2, 0)): 4 rows.
    split = r.survSplit("Surv(time, status) ~ .", data, cut=[2 + 1e-12], zero=[2, 0])
    assert split["tstart"] == [2.0, 2 + 1e-12, 0.0, 2 + 1e-12]
    assert split["time"] == [2 + 1e-12, 10.0, 2 + 1e-12, 20.0]
    # zero = c(10 - 1e-12, 0) with cut = 15 is not merged into the event time.
    split = r.survSplit("Surv(time, status) ~ .", data, cut=[15], zero=[10 - 1e-12, 0])
    assert split["tstart"] == [10 - 1e-12, 0.0, 15.0]
    assert split["time"] == [10.0, 15.0, 20.0]


def test_zero_is_read_only_for_time_status_data():
    data = {"time": [10.0, 20.0], "status": [1.0, 0.0]}
    split = r.survSplit("Surv(time, status) ~ .", data, cut=[5], zero=float("-inf"))
    assert split["tstart"] == [float("-inf"), 5.0, float("-inf"), 5.0]
    with pytest.raises(ValueError, match="zero must not be missing"):
        r.survSplit("Surv(time, status) ~ .", data, cut=[5], zero=float("nan"))
    counting = {"start": [0.0, 0.0], "stop": [10.0, 20.0], "status": [1.0, 0.0]}
    for zero in (float("nan"), "abc", None):
        split = r.survSplit("Surv(start, stop, status) ~ .", counting, cut=[5], zero=zero)
        assert split["start"] == [0.0, 5.0, 0.0, 5.0]


def test_make_names_keeps_non_ascii_letters_and_numbers():
    # R's make.names(c("ⅷ", "١ep", "1st")): "ⅷ" "١ep" "X1st"
    data = {"time": [3.0, 7.0], "status": [1, 0], "x": [1.5, 2.5]}
    assert list(r.survSplit("Surv(time, status) ~ x", data, cut=[5], episode="ⅷ"))[-1] == "ⅷ"
    assert list(r.survSplit("Surv(time, status) ~ x", data, cut=[5], added="١ep"))[-1] == "١ep"
    assert list(r.survSplit("Surv(time, status) ~ x", data, cut=[5], episode="1st"))[-1] == "X1st"
