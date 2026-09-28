"""What a type checker sees of ``survival.r`` (checked by mypy in CI, not collected by pytest).

``survival.r`` ships no stub, so its inline annotations are the typed surface; these
``assert_type`` calls fail the lint workflow's mypy step when a public return type
regresses to ``Any`` or to the wrong class.
"""

from typing import Any, assert_type

from survival import r


def _check_public_return_types(data: dict[str, list[Any]], curves: r.CoxSurvfitResult) -> None:
    cox = r.coxph("Surv(time, status) ~ age", data)
    assert_type(cox, r.CoxphModel)
    assert_type(r.clogit("status ~ age + strata(inst)", data), r.ClogitModel)
    assert_type(r.survreg("Surv(time, status) ~ age", data), r.SurvregModelResult)
    assert_type(
        r.survfit("Surv(time, status) ~ sex", data),
        r.SurvfitResult
        | r.SurvfitMultiStateResult
        | r.CoxSurvfitResult
        | r.CoxSurvfitMultiStateResult,
    )
    assert_type(r.brier(cox), r.BrierResult)
    assert_type(r.concordance(cox), r.ConcordanceResult)
    y = r.Surv(data["time"], data["status"])
    assert_type(r.concordancefit(y, data["age"]), r.ConcordanceResult)
    assert_type(r.survConcordance("Surv(time, status) ~ age", data), r.SurvConcordanceResult)
    assert_type(r.aareg("Surv(time, status) ~ age", data), r.AaregModelResult)
    assert_type(r.aggregate_survfit(curves), r.CoxSurvfitResult)
    # were ``survival`` unresolvable, --ignore-missing-imports would make every r.* above
    # Any and those assertions vacuous; this one, against a concrete type, would still fail
    assert_type(r.cluster(data["inst"]), list[Any])
