"""Export survival plots: python examples/survival_curves.py curves.svg."""

import argparse

import numpy as np
from matplotlib import pyplot as plt
from survival import datasets, plotting, r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="survival-curves.svg")
    args = parser.parse_args()
    lung = datasets.load_lung()
    grouped = r.survfit("Surv(time, status) ~ sex", lung)
    cox = r.coxph("Surv(time, status) ~ age + sex", lung)
    predicted = r.survfit(cox, newdata={"age": [50, 70], "sex": [1, 1]})
    endpoint = ["censor", "ill", "dead", "ill"] * 6
    multistate = r.survfit(r.Surv(np.arange(1, 25), endpoint, type="mstate"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    plotting.plot_survfit(grouped, ax=axes[0, 0], conf_int=True, conf_style="band", mark_time=True)
    axes[0, 0].set_title("Kaplan–Meier curves by sex")
    plotting.plot_survfit(grouped, ax=axes[0, 1], fun="cumhaz")
    axes[0, 1].set_title("Cumulative hazards")
    result = plotting.plot_survfit(predicted, ax=axes[1, 0], conf_int=True, conf_style="band")
    for line, age in zip(result.lines, [50, 70], strict=True):
        line.set_label(f"Age {age}")
    axes[1, 0].legend()
    axes[1, 0].set_title("Cox survival predictions")
    plotting.plot_survfit(multistate, ax=axes[1, 1])
    axes[1, 1].set_title("State probabilities")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
