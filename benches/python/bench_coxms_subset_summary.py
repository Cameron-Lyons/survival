#!/usr/bin/env python3
"""Compare a prepared one-state summary with summarizing all states then selecting.

Fitting and prediction are excluded. Each timed call includes the public summary,
state projection and numeric output materialization. Subset preparation is reported
separately. Both paths return the same state probabilities, counts and mean table.
"""

import argparse
import json
import platform
import statistics
import time

import numpy as np
import pandas as pd
from survival import datasets, r


def _project(curve, state):
    summary = r.summary_survfit(curve, censored=True)
    ndata = summary.pstate.shape[1]
    return {
        "time": np.asarray(summary.time),
        "pstate": summary.pstate[:, :, [state]],
        "n_risk": np.asarray(summary.n_risk)[:, [state]],
        "n_event": np.asarray(summary.n_event)[:, [state]],
        "n_censor": np.asarray(summary.n_censor)[:, [state]],
        "table": np.asarray(summary.table.values)[state * ndata : (state + 1) * ndata].copy(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()
    if args.rows < 1 or args.repeat < 1:
        parser.error("rows and repeat must be positive")
    data = pd.DataFrame(datasets.load_mgus2())
    data["etime"] = np.where(data.pstat == 1, data.ptime, data.futime)
    data["event"] = pd.Categorical(
        np.where(data.pstat == 1, "pcm", np.where(data.death == 1, "death", "censor")),
        categories=["censor", "pcm", "death"],
    )
    model = r.coxph("Surv(etime, event) ~ age + sex", data, id="id")
    full = r.survfit(
        model,
        newdata={
            "age": np.linspace(50, 80, args.rows),
            "sex": ["F", "M"] * (args.rows // 2) + (["F"] if args.rows % 2 else []),
        },
    )
    selected = full.subset(states=["death"])
    calls = {
        "all_states_then_select": lambda: _project(full, 2),
        "prepared_state_subset": lambda: _project(selected, 0),
        "prepare_subset": lambda: full.subset(states=["death"]),
    }
    expected, actual = calls["all_states_then_select"](), calls["prepared_state_subset"]()
    for field, value in expected.items():
        np.testing.assert_allclose(actual[field], value, rtol=2e-14, atol=1e-15)
    for function in calls.values():
        for _ in range(3):
            function()
    samples = {name: [] for name in calls}
    for repeat in range(args.repeat):
        order = list(calls.items())
        if repeat % 2:
            order.reverse()
        for name, function in order:
            start = time.perf_counter()
            function()
            samples[name].append((time.perf_counter() - start) * 1000)
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "rows": args.rows,
                "times": len(full.time),
                "states": len(full.states),
                "samples": args.repeat,
                "warmups": 3,
                "milliseconds": {
                    name: {"median": statistics.median(v), "min": min(v), "max": max(v)}
                    for name, v in samples.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
