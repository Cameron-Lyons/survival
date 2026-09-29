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
    assert_type(r.print_coxph(cox), r.ModelPrint)
    assert_type(r.print_summary_coxph(r.model_summary(cox)), r.ModelPrint)
    assert_type(r.clogit("status ~ age + strata(inst)", data), r.ClogitModel)
    conditional = r.clogit("status ~ age + strata(inst)", data)
    assert_type(r.print_clogit(conditional), r.ModelPrint)
    case_cohort = r.cch(
        "Surv(time, status) ~ age", data, subcoh="subcoh", id="id", cohort_size=1000
    )
    assert_type(r.print_cch(case_cohort), r.ModelPrint)
    assert_type(r.print_summary_cch(r.model_summary(case_cohort)), r.ModelPrint)
    assert_type(r.print_cox_zph(r.cox_zph(cox)), r.ModelPrint)
    assert_type(r.print_concordance(r.concordance(cox)), r.ModelPrint)
    legacy_concordance = r.survConcordance("Surv(time, status) ~ age", data)
    assert_type(r.print_survConcordance(legacy_concordance), r.ModelPrint)
    assert_type(r.print_survdiff(r.survdiff("Surv(time, status) ~ sex", data)), r.ModelPrint)
    assert_type(r.print_pyears(r.pyears("time ~ sex", data)), r.ModelPrint)
    assert_type(
        r.print_survcheck(r.survcheck("Surv(time, status) ~ 1", data, id="id")), r.ModelPrint
    )
    assert_type(r.print_yates(r.yates(cox, "age", levels=[40, 60, 80])), r.YatesPrint)
    aft = r.survreg("Surv(time, status) ~ age", data)
    assert_type(aft, r.SurvregModelResult)
    assert_type(r.print_survreg(aft), r.ModelPrint | r.SurvregPenalPrint)
    assert_type(r.print_summary_survreg(r.model_summary(aft)), r.ModelPrint)
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
    assert_type(r.print_surv(y), r.ResponsePrint)
    assert_type(r.print_surv2(r.Surv2(data["time"], data["status"])), r.ResponsePrint)
    assert_type(r.print_ratetable(r.survexp_us()), r.RateTablePrint)
    assert_type(r.concordancefit(y, data["age"]), r.ConcordanceResult)
    assert_type(r.survConcordance("Surv(time, status) ~ age", data), r.SurvConcordanceResult)
    additive = r.aareg("Surv(time, status) ~ age", data)
    assert_type(additive, r.AaregModelResult)
    assert_type(r.print_aareg(additive), r.ModelPrint)
    assert_type(r.print_summary_aareg(r.model_summary(additive)), r.ModelPrint)
    assert_type(r.aggregate_survfit(curves), r.CoxSurvfitResult)
    assert_type(r.print_survfit(curves), r.SurvfitPrint)
    assert_type(r.print_summary_survfit(r.summary_survfit(curves)), r.SurvivalTablePrint)
    # were ``survival`` unresolvable, --ignore-missing-imports would make every r.* above
    # Any and those assertions vacuous; this one, against a concrete type, would still fail
    assert_type(r.cluster(data["inst"]), list[Any])
