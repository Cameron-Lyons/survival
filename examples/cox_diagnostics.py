"""Export coefficient and hazard-ratio diagnostics for a Cox model."""

import argparse

from matplotlib import pyplot as plt
from survival import datasets, plotting, r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="cox-diagnostics.svg")
    args = parser.parse_args()
    fit = r.coxph("Surv(time, status) ~ age + sex", datasets.load_lung())
    diagnostic = r.cox_zph(fit)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    plotting.plot_cox_zph(diagnostic, ax=axes[:, 0], colors=["navy", "steelblue"])
    plotting.plot_cox_zph(
        diagnostic, ax=axes[:, 1], hr=True, resid=False, colors=["darkred", "salmon"]
    )
    figure.suptitle("Cox proportional-hazards diagnostics")
    figure.savefig(args.output)
    plt.close(figure)


if __name__ == "__main__":
    main()
