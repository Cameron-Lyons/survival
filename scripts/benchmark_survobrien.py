"""Complete Python O'Brien expansions, including custom transform callbacks.

Run from the repository with PYTHONPATH=python. Native NumPy snapshots provide
an independent geometry check before timing the formula-facing result.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from functools import partial

import numpy as np
from survival import r, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.rows < 1 or args.repeats < 1:
        parser.error("rows and repeats must be positive")
    rng = np.random.default_rng(712)
    data = {
        "time": rng.uniform(1, 1000, args.rows),
        "status": (np.arange(args.rows) % 3 == 0).astype(np.int32),
        "x": np.round(rng.normal(size=args.rows), 1),
        "z": rng.normal(size=args.rows),
    }
    formula = "Surv(time, status) ~ x+z"

    def centered(values):
        values = np.asarray(values)
        return values - values.mean()

    geometry = validation.survobrien(data["time"], data["status"], [], transform=False).to_arrays()
    results = {}
    for kind, transform in [("default", None), ("custom", centered)]:
        call = partial(r.survobrien, formula, data, transform=transform)
        result = call()
        np.testing.assert_array_equal(result[".id."], geometry["row"] + 1)
        np.testing.assert_array_equal(result["status"], geometry["status"])
        np.testing.assert_array_equal(result[".strata."], geometry["strata"])
        if kind == "custom":
            for start, end in zip(
                geometry["block_offsets"][:-1], geometry["block_offsets"][1:], strict=True
            ):
                for name in ("x", "z"):
                    np.testing.assert_allclose(
                        result[name][start:end], centered(data[name][geometry["row"][start:end]])
                    )
        del result
        for _ in range(2):
            call()
        samples = []
        for _ in range(args.repeats):
            gc.collect()
            started = time.perf_counter()
            result = call()
            samples.append(1000 * (time.perf_counter() - started))
            del result
        results[kind] = {"median_ms": statistics.median(samples), "samples_ms": samples}
    print(
        json.dumps(
            {
                "observations": args.rows,
                "expanded_rows": len(geometry["row"]),
                "risk_sets": len(geometry["event_times"]),
                "repeats": args.repeats,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
