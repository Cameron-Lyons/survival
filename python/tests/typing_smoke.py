"""What a type checker sees of ``survival.r`` (checked by mypy in CI, not collected by pytest).

``survival.r`` ships no stub, so its inline annotations are the typed surface; these
``assert_type`` calls fail the lint workflow's mypy step when a public return type
regresses to ``Any`` or to the wrong class.
"""

from typing import Any, assert_type

import numpy as np
from survival import _survival, r


def _check_yates_prediction_types() -> None:
    result = _survival.yates_predict(
        [[[1.0]], [[2.0]]],
        [0.1],
        [[0.02]],
        lambda eta: np.column_stack((np.exp(eta), eta)),
        nsim=3,
        normal_draws=lambda n, p: np.zeros((n, p)),
    )
    assert_type(result, _survival.YatesResult)
    assert_type(result.prediction_mean, list[list[float]] | None)
    assert_type(result.prediction_variance, list[list[float]] | None)


def _check_public_return_types(data: dict[str, list[Any]], curves: r.CoxSurvfitResult) -> None:
    cox = r.coxph("Surv(time, status) ~ age", data)
    assert_type(cox, r.CoxphModel)
    setup = r.yates_setup(cox, "survival")
    assert_type(setup, r.YatesLinkPrediction | r.YatesSurvivalSetup | None)
    if isinstance(setup, r.YatesSurvivalSetup):
        assert_type(setup.summary([[1.0, 1.0]], [[0.0, 0.0]]), r.CoxSurvfitResult)
    terms = r.TermMetadata(["age"])
    assert_type(r.attrassign(r.model_matrix(cox), terms), dict[str, list[int]])
    assert_type(r.untangle_specials(terms, "strata"), r.SpecialTerms)
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
    glm = r.YatesModel(
        "y ~ age",
        data,
        [1.0, 0.1],
        [[0.2, 0.0], [0.0, 0.3]],
        family={"linkinv": lambda eta: eta},
        weights=[1.0] * len(data["y"]),
    )
    assert_type(glm, r.YatesModel)
    assert_type(r.yates(glm, "age", levels=[40, 60], predict="response"), r.YatesResult)
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
    raw_km = r.survfitKM(r.strata(data["sex"]), y)
    assert_type(raw_km, r.SurvfitKMResult)
    assert_type(raw_km.n, list[int])
    assert_type(raw_km.strata, dict[str, int] | None)
    direct = r.coxsurv_fit(y=y, x=data, x2=data, risk=data["age"], risk2=data["age"])
    assert_type(direct, r.CoxSurvFitResult | r.CoxSurvFitList)
    if isinstance(direct, r.CoxSurvFitResult):
        assert_type(direct.n, list[int])
        assert_type(direct.strata, dict[str, int] | None)
    else:
        assert_type(direct[0], r.CoxSurvFitResult)
    assert_type(r.survfitcoxph_fit(y, data), r.CoxSurvFitResult | r.CoxSurvFitList)
    bare = r.coxph_fit(data["age"], y)
    assert_type(bare, r.CoxFitResult)
    assert_type(bare.coefficients, list[float] | None)
    assert_type(bare.residuals, list[float] | None)
    assert_type(r.agreg_fit(data["age"], y), r.CoxFitResult)
    assert_type(r.agexact_fit(data["age"], y), r.CoxFitResult)
    bare_aft = r.survreg_fit(data, y)
    assert_type(bare_aft, r.SurvregFitResult)
    assert_type(bare_aft.coefficients, list[float])
    assert_type(bare_aft.df, int)
    bare_penal = r.survpenal_fit(data, y)
    assert_type(bare_penal, r.SurvpenalFitResult)
    assert_type(bare_penal.coefficients, list[float])
    assert_type(bare_penal.assign2, dict[str, list[int]])
    assert_type(bare_penal.frail, list[float] | None)
    assert_type(r.print_survreg_penal(bare_penal), r.SurvregPenalPrint)
    assert_type(r.print_surv(y), r.ResponsePrint)
    assert_type(r.print_surv2(r.Surv2(data["time"], data["status"])), r.ResponsePrint)
    assert_type(r.print_ratetable(r.survexp_us()), r.RateTablePrint)
    assert_type(r.match_ratetable(data, r.survexp_us()), r.RateTableMatch)
    assert_type(r.survexp_us().match_levels(1, ["m", "f"]), list[int])
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
