"""Export cumulative Aalen coefficients with ordinary and robust bands."""

import argparse

from matplotlib import pyplot as plt
from survival import datasets, plotting, r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="aalen-curves.svg")
    args = parser.parse_args()
    data = datasets.load_ovarian()
    ordinary = r.aareg("Surv(futime, fustat) ~ age + ecog.ps", data)
    robust = r.aareg("Surv(futime, fustat) ~ age + ecog.ps", data, dfbeta=True)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for fit, panels, color, title in zip(
        (ordinary, robust),
        (axes[:, 0], axes[:, 1]),
        ("navy", "darkred"),
        ("Coefficient-increment bands", "Influence-based bands"),
        strict=True,
    ):
        plotting.plot_aareg(fit, var=["age", "ecog.ps"], ax=panels, colors=color, xlabel="Days")
        panels[0].set_title(title)
    figure.suptitle("Aalen cumulative coefficient curves")
    figure.savefig(args.output)
    plt.close(figure)


if __name__ == "__main__":
    main()
