"""Minimal example showing the preferred module-oriented Python imports."""

from survival import core, datasets, validation


def main() -> None:
    lung = datasets.load_lung()
    print(f"lung rows: {len(lung['time'])}")
    print(f"lung columns: {list(lung)[:4]}")

    concordance = core.concordancefit(
        core.SurvivalData([1.0, 2.0, 3.0, 4.0, 5.0], [1, 1, 0, 1, 1]),
        core.CovariateMatrix([0.2, 0.1, 0.5, 0.4, 0.9], 5, 1),
    )
    print(f"concordance index: {concordance.concordance[0]:.3f}")

    wald = validation.wald_test([1.0], [[0.5]])
    print(f"wald test result type: {type(wald).__name__}")


if __name__ == "__main__":
    main()
