"""Measure survival plot preparation and optional rendering, excluding fitting.

PYTHONPATH=python .venv/bin/python scripts/bench_survfit_plot.py --render
"""

import argparse
import json
import platform
from statistics import median
from time import perf_counter

import numpy as np
from survival import plotting, r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if min(*args.n, args.repeats) < 1:
        parser.error("sizes and repeats must be positive")
    if args.render:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

    results = []
    for n in args.n:
        for event_every in (0, 20, 1):
            time = np.arange(1, n + 1, dtype=float)
            status = (
                np.zeros(n, dtype=int) if not event_every else (time % event_every == 0).astype(int)
            )
            fit = r.survfit(r.Surv(time, status))
            plotting.survfit_plot_data(fit, conf_int=False)
            samples = []
            for _ in range(args.repeats):
                start = perf_counter()
                data = plotting.survfit_plot_data(fit, conf_int=False)
                coordinates = data.curves[0].step()
                samples.append(1000 * (perf_counter() - start))
            result = {
                "n": n,
                "event_every": event_every,
                "prepare_ms": median(samples),
                "path_vertices": len(coordinates[0]),
            }
            if args.render:
                samples = []
                for _ in range(args.repeats):
                    start = perf_counter()
                    rendered = plotting.plot_survfit(fit, conf_int=False, legend=False)
                    rendered.axes.figure.canvas.draw()
                    samples.append(1000 * (perf_counter() - start))
                    plt.close(rendered.axes.figure)
                result["render_ms"] = median(samples)
            results.append(result)
    print(json.dumps({"python": platform.python_version(), "results": results}, indent=2))


if __name__ == "__main__":
    main()
