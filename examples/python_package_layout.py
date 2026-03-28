"""Minimal example showing the preferred module-oriented Python imports."""

from survival import core, datasets, validation


def main() -> None:
    lung = datasets.load_lung()
    columns = [name for name in lung if not name.startswith("_")]
    print(f"lung rows: {lung['_nrow']}")
    print(f"lung columns: {columns[:4]}")

    concordance = core.perform_concordance1_calculation(
        [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0, 1, 2, 3, 4],
        5,
    )
    print(f"concordance index: {concordance['concordance_index']:.3f}")

    wald = validation.wald_test_py([1.0], [0.5])
    print(f"wald test result type: {type(wald).__name__}")


if __name__ == "__main__":
    main()
