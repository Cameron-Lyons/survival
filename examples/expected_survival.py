"""Export raw-response curves with a fitted Cox expected-survival reference."""

import argparse

import numpy as np
from matplotlib import pyplot as plt
from survival import datasets, plotting, r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="expected-survival.svg")
    args = parser.parse_args()
    lung = datasets.load_lung()
    response = r.Surv(lung["time"], lung["status"])
    reference = r.coxph("Surv(time, status) ~ age + sex", lung)
    expected = r.survexp(
        "~1", lung, ratetable=reference, times=np.linspace(1, max(lung["time"]), 100)
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, transform, title in zip(
        axes, ("identity", "cumhaz"), ("Survival probability", "Cumulative hazard"), strict=True
    ):
        options = {"ax": ax, "fun": transform, "xscale": 365.25, "xlabel": "Years"}
        plotting.plot(response, **options, colors="navy", conf_style="band", label="Observed")
        plotting.lines(expected, **options, colors="darkorange", label="Cox reference")
        ax.set_title(title)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
