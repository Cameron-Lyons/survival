"""Differential tests against R's survival package.

One test per (topic, case, aspect) triple in ``test/r/fixtures/*.json``.  The
fixtures are produced by ``test/r/generate_fixtures.R``; the schema and the
regeneration recipe live in ``test/r/README.md``.

Tolerances (relative): 1e-8 for coefficients, curves, residuals and linear
predictors; 1e-6 for variances, standard errors, test statistics and
p-values; exact for counts.

Cases the Python API cannot reproduce yet are listed in ``KNOWN_FAILURES``
and marked ``xfail(strict=True)``: a fixed case turns into an XPASS failure
that forces its removal from the list.  Keys are either ``topic/case/aspect``
or ``topic/case`` (every aspect of the case).

Set ``R_FIXTURES_COLLECT=/path/file.jsonl`` to append one record per failing
test (kind + message) for building the burndown list.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

from .helpers import setup_survival_import
from .r_fixture_support import (
    RTOL_COEF,
    RTOL_VAR,
    FixtureMismatchError,
    RFactor,
    UnsupportedCaseError,
    array_scale,
    as_float_list,
    assert_close,
    assert_exact,
    assert_matrix_close,
    assert_named_values,
    case_data,
    case_id,
    cases,
    column,
    decode_frame,
    decode_vector,
    is_r_error,
    load_dataset,
    load_topic,
    newdata_frame,
    nrow,
    r_strata_value,
    record_outcome,
    topic_names,
    transpose,
)

survival = setup_survival_import()
r = survival.r_api

# ---------------------------------------------------------------------------
# Known failures (burndown list).  Reasons start with one of
#   "missing feature:" the Python API cannot express the case
#   "mismatch:"        Python runs but the numbers differ from R
#   "error:"           the Python call raises
# ---------------------------------------------------------------------------

KNOWN_FAILURES: dict[str, str] = {
    "aareg/kidney_age_sex/chisq": "missing feature: result has none of the attributes ('chisq',)",
    "aareg/lung_age_sex_nrisk/chisq": (
        "missing feature: result has none of the attributes ('chisq',)"
    ),
    "aareg/lung_age_sex_ph_ecog/chisq": (
        "missing feature: result has none of the attributes ('chisq',)"
    ),
    "aareg/lung_qrtol_taper/chisq": "missing feature: result has none of the attributes ('chisq',)",
    "aareg/lung_weighted/chisq": "missing feature: result has none of the attributes ('chisq',)",
    "aareg/ovarian_age_ecog_dfbeta/chisq": (
        "missing feature: result has none of the attributes ('chisq',)"
    ),
    "aareg/ovarian_age_ecog_dfbeta/dfbeta": "mismatch: dfbeta[0][3][0]: -0.010642 != 0.051349",
    "aareg/ovarian_age_rx_dfbeta_nrisk/chisq": (
        "missing feature: result has none of the attributes ('chisq',)"
    ),
    "aareg/ovarian_age_rx_dfbeta_nrisk/dfbeta": (
        "mismatch: dfbeta[0][3][0]: -0.00015217 != 0.049983"
    ),
    "aareg/veteran_karno_celltype/times": "mismatch: times: length 104 differs from expected 117",
    "aareg/veteran_karno_celltype/nrisk": "mismatch: nrisk: length 104 differs from expected 117",
    "aareg/veteran_karno_celltype/coefficient": (
        "mismatch: coefficient: length 104 differs from expected 117"
    ),
    "aareg/veteran_karno_celltype/test_statistic": (
        "mismatch: test_statistic[0]: expected NA/NaN, got 16.341"
    ),
    "aareg/veteran_karno_celltype/test_var": (
        "mismatch: test_var[0][0]: expected NA/NaN, got 8.6362"
    ),
    "aareg/veteran_karno_celltype/tweight": (
        "mismatch: tweight: length 104 differs from expected 117"
    ),
    "aareg/veteran_karno_celltype/chisq": (
        "missing feature: result has none of the attributes ('chisq',)"
    ),
    "cch/i_borgan": "error: ValueError: population is smaller than the sample in a stratum",
    "cch/ii_borgan": "error: ValueError: population is smaller than the sample in a stratum",
    "cch/lin_ying/coef": "mismatch: coef[0]: -0.60686 != 0.69266",
    "cch/lin_ying/var": "mismatch: var[0][0]: 0.034577 != 0.02653",
    "cch/lin_ying/naive_var": "mismatch: naive_var[0][0]: 0.034577 != 0.02653",
    "cch/prentice/coef": "mismatch: coef[0]: -0.64956 != 0.73457",
    "cch/prentice/var": "mismatch: var[0][0]: 0.042022 != 0.028391",
    "cch/prentice/naive_var": "mismatch: naive_var[0][0]: 0.042022 != 0.028391",
    "cch/self_prentice/coef": "mismatch: coef[0]: -0.65538 != 0.73624",
    "cch/self_prentice/var": "mismatch: var[0][0]: 0.042022 != 0.028391",
    "cch/self_prentice/naive_var": "mismatch: naive_var[0][0]: 0.042022 != 0.028391",
    "clogit/infert_age_spont_efron/linear_predictors": (
        "mismatch: linear_predictors[0]: 2.3537 != 1.6751"
    ),
    "clogit/infert_spont_induced_approximate/linear_predictors": (
        "mismatch: linear_predictors[0]: 5.3808 != 3.4289"
    ),
    "clogit/infert_spont_induced_breslow/linear_predictors": (
        "mismatch: linear_predictors[0]: 5.3808 != 3.4289"
    ),
    "clogit/infert_spont_induced_efron/linear_predictors": (
        "mismatch: linear_predictors[0]: 5.3808 != 3.4289"
    ),
    "clogit/infert_spont_induced_exact/linear_predictors": (
        "mismatch: linear_predictors[0]: 5.3808 != 3.4289"
    ),
    "clogit/infert_spont_induced_pooled_exact/linear_predictors": (
        "mismatch: linear_predictors[0]: 5.4823 != 3.4962"
    ),
    "concordance/aml_x_numeric": (
        "error: ValueError: as.numeric() formula term 'x' requires numeric values"
    ),
    "concordance/cgd_counting_age_cluster/concordance": (
        "mismatch: concordance[0]: 0.426 != 0.42669"
    ),
    "concordance/cgd_counting_age_cluster/count": "mismatch: count[0]: 3221 != 3120",
    "concordance/cgd_counting_age_cluster/var": "mismatch: var[0]: 0.00055802 != 0.0022452",
    "concordance/cgd_counting_age_id/concordance": "mismatch: concordance[0]: 0.426 != 0.42669",
    "concordance/cgd_counting_age_id/count": "mismatch: count[0]: 3221 != 3120",
    "concordance/cgd_counting_age_id/var": "mismatch: var[0]: 0.00055802 != 0.0022452",
    "concordance/coxph_survreg_fits": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "concordance/heart_counting_age/concordance": "mismatch: concordance[0]: 0.57577 != 0.57546",
    "concordance/heart_counting_age/count": "mismatch: count[0]: 2568.5 != 2600",
    "concordance/heart_counting_age/var": "mismatch: variance is None",
    "concordance/lung_age/concordance": "mismatch: concordance[0]: 0.4498 != 0.44976",
    "concordance/lung_age/count": "mismatch: count[0]: 8996.5 != 8706",
    "concordance/lung_age/var": "mismatch: variance is None",
    "concordance/lung_age_cluster_inst/concordance": "mismatch: concordance[0]: 0.55195 != 0.55202",
    "concordance/lung_age_cluster_inst/count": "mismatch: count[0]: 10688 != 10412",
    "concordance/lung_age_cluster_inst/var": "mismatch: var[0]: 0.00015624 != 0.00062592",
    "concordance/lung_age_influence/concordance": "mismatch: concordance[0]: 0.5502 != 0.55024",
    "concordance/lung_age_influence/count": "mismatch: count[0]: 11004 != 10717",
    "concordance/lung_age_influence/var": "mismatch: var[0]: 0.00015806 != 0.00063213",
    "concordance/lung_age_influence/dfbeta": "mismatch: dfbeta[0]: -0.00037466 != -0.00074924",
    "concordance/lung_age_ph_karno/concordance": "mismatch: concordance[0]: 0.44805 != 0.44798",
    "concordance/lung_age_ph_karno/count": (
        "missing feature: multi-column concordance counts are not exposed"
    ),
    "concordance/lung_age_ph_karno/var": "mismatch: variance is None",
    "concordance/lung_age_ranks/concordance": "mismatch: concordance[0]: 0.5502 != 0.55024",
    "concordance/lung_age_ranks/count": "mismatch: count[0]: 11004 != 10717",
    "concordance/lung_age_ranks/var": "mismatch: variance is None",
    "concordance/lung_age_reverse/concordance": "mismatch: concordance[0]: 0.5502 != 0.55024",
    "concordance/lung_age_reverse/count": "mismatch: count[0]: 11004 != 10717",
    "concordance/lung_age_reverse/var": "mismatch: variance is None",
    "concordance/lung_age_strata_sex/concordance": "mismatch: concordance[0]: 0.54578 != 0.5459",
    "concordance/lung_age_strata_sex/count": (
        "missing feature: multi-column concordance counts are not exposed"
    ),
    "concordance/lung_age_strata_sex/var": "mismatch: variance is None",
    "concordance/lung_age_timewt_I/concordance": "mismatch: concordance[0]: 0.5433 != 0.54339",
    "concordance/lung_age_timewt_I/count": "mismatch: count[0]: 87.781 != 85.348",
    "concordance/lung_age_timewt_I/var": "mismatch: variance is None",
    "concordance/lung_age_timewt_S/concordance": "mismatch: concordance[0]: 0.54962 != 0.54967",
    "concordance/lung_age_timewt_S/count": "mismatch: count[0]: 12258 != 11933",
    "concordance/lung_age_timewt_S/var": "mismatch: variance is None",
    "concordance/lung_age_timewt_SG/concordance": "mismatch: concordance[0]: 0.54916 != 0.54923",
    "concordance/lung_age_timewt_SG/count": "mismatch: count[0]: 14117 != 13742",
    "concordance/lung_age_timewt_SG/var": "mismatch: variance is None",
    "concordance/lung_age_timewt_SG/cvar": "mismatch: cvar[0]: 0.00046156 != 0.00046131",
    "concordance/lung_age_timewt_n/concordance": "mismatch: concordance[0]: 0.5502 != 0.55024",
    "concordance/lung_age_timewt_n/count": "mismatch: count[0]: 11004 != 10717",
    "concordance/lung_age_timewt_n/var": "mismatch: variance is None",
    "concordance/lung_age_timewt_nG2/concordance": "mismatch: concordance[0]: 0.54916 != 0.54923",
    "concordance/lung_age_timewt_nG2/count": "mismatch: count[0]: 14113 != 13742",
    "concordance/lung_age_timewt_nG2/var": "mismatch: variance is None",
    "concordance/lung_age_timewt_nG2/cvar": "mismatch: cvar[0]: 0.00046181 != 0.00046131",
    "concordance/lung_age_weighted/concordance": "mismatch: concordance[0]: 0.45501 != 0.45497",
    "concordance/lung_age_weighted/count": "mismatch: count[0]: 14336 != 13901",
    "concordance/lung_age_weighted/var": "mismatch: variance is None",
    "concordance/lung_numeric_y_age": (
        "missing feature: ValueError: formula response must be Surv(...)"
    ),
    "concordance/ovarian_age_influence/var": "mismatch: var[0]: 0.0017086 != 0.0068343",
    "concordance/ovarian_age_influence/dfbeta": "mismatch: dfbeta[0]: 0.007775 != 0.01555",
    "concordance/ovarian_age_influence/influence": "mismatch: influence[0][0]: 11.5 != 23",
    "concordance/synthetic_delayed_x/concordance": "mismatch: concordance[0]: 0.4 != 0.375",
    "concordance/synthetic_delayed_x/count": "mismatch: count[1]: 18 != 20",
    "concordance/synthetic_delayed_x/var": "mismatch: var[0]: 0.0043333 != 0.016357",
    "concordance/synthetic_delayed_x/dfbeta": "mismatch: dfbeta[0]: 0.046667 != 0.09375",
    "concordance/synthetic_ties_x/concordance": "mismatch: concordance[0]: 0.30952 != 0.31461",
    "concordance/synthetic_ties_x/count": "mismatch: count[0]: 26 != 28",
    "concordance/synthetic_ties_x/var": "mismatch: var[0]: 0.0016913 != 0.0072577",
    "concordance/synthetic_ties_x/influence": "mismatch: influence[0][0]: 2.5 != 5",
    "concordance/veteran_karno_age_multi/concordance": (
        "mismatch: concordance[0]: 0.29072 != 0.29072"
    ),
    "concordance/veteran_karno_age_multi/count": (
        "missing feature: multi-column concordance counts are not exposed"
    ),
    "concordance/veteran_karno_age_multi/var": "mismatch: variance is None",
    "coxph/aml_x/n": "mismatch: n (nobs): 18 != 23",
    "coxph/aml_x/residuals.martingale": "mismatch: residuals.martingale[4]: 0.65156 != 0.67626",
    "coxph/aml_x/residuals.deviance": "mismatch: residuals.deviance[4]: 0.89747 != 0.95033",
    "coxph/aml_x/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/aml_x/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/aml_x/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/aml_x/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/aml_x/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/aml_x/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/aml_x_breslow/n": "mismatch: n (nobs): 18 != 23",
    "coxph/aml_x_breslow/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/aml_x_breslow/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/aml_x_breslow/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/aml_x_breslow/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/aml_x_breslow/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/aml_x_breslow/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/aml_x_exact/n": "mismatch: n (nobs): 18 != 23",
    "coxph/aml_x_exact/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/aml_x_exact/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/aml_x_exact/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/aml_x_exact/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/aml_x_exact/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/aml_x_exact/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/bladder_ag/n": "mismatch: n (nobs): 47 != 85",
    "coxph/bladder_ag/linear_predictors": "mismatch: linear_predictors[0]: -0.078965 != 0.040542",
    "coxph/bladder_ag/residuals.martingale": (
        "mismatch: residuals.martingale[4]: 0.28453 != 0.33371"
    ),
    "coxph/bladder_ag/residuals.deviance": "mismatch: residuals.deviance[4]: 0.31712 != 0.38032",
    "coxph/bladder_ag/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/bladder_ag/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/bladder_ag/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/bladder_ag/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/bladder_ag/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/bladder_ag/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/bladder_wlw/n": "mismatch: n (nobs): 112 != 340",
    "coxph/bladder_wlw/linear_predictors": "mismatch: linear_predictors[0]: -0.52935 != -0.022133",
    "coxph/bladder_wlw/residuals.martingale": (
        "mismatch: residuals.martingale[16]: 0.21381 != 0.26783"
    ),
    "coxph/bladder_wlw/residuals.deviance": "mismatch: residuals.deviance[16]: 0.23129 != 0.29636",
    "coxph/bladder_wlw/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/bladder_wlw/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/bladder_wlw/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/bladder_wlw/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/bladder_wlw/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/bladder_wlw/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_strata_enum/coef": "mismatch: coef[0]: 0.90376 != -0.90376",
    "coxph/cgd_counting_strata_enum/coef_names": (
        "mismatch: coef: names ['treatplacebo', 'age'] != ['treatrIFN-g', 'age']"
    ),
    "coxph/cgd_counting_strata_enum/var": "mismatch: var[0][1]: -0.00029484 != 0.00029484",
    "coxph/cgd_counting_strata_enum/n": "mismatch: n (nobs): 76 != 203",
    "coxph/cgd_counting_strata_enum/linear_predictors": (
        "mismatch: linear_predictors[0]: -0.31188 != -0.85946"
    ),
    "coxph/cgd_counting_strata_enum/residuals.martingale": (
        "mismatch: residuals.martingale[51]: 0.68659 != 0.69502"
    ),
    "coxph/cgd_counting_strata_enum/residuals.deviance": (
        "mismatch: residuals.deviance[51]: 0.97329 != 0.99246"
    ),
    "coxph/cgd_counting_strata_enum/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_strata_enum/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_strata_enum/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_strata_enum/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_strata_enum/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_strata_enum/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_strata_enum/summary.coefficients": (
        "mismatch: coefficients[treatplacebo].coef: 0.90376 != -0.90376"
    ),
    "coxph/cgd_counting_strata_enum/summary.conf_int": (
        "mismatch: conf_int.lower[0]: 1.42 != 0.23297"
    ),
    "coxph/cgd_counting_strata_enum/wtest": "mismatch: wtest.solve[0]: 10.89 != -10.89",
    "coxph/cgd_counting_strata_enum/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/cgd_counting_treat_cluster_id/coef": "mismatch: coef[0]: 1.0722 != -1.0722",
    "coxph/cgd_counting_treat_cluster_id/coef_names": (
        "mismatch: coef: names ['treatplacebo', 'inheritX-linked', 'steroids'] != ['trea..."
    ),
    "coxph/cgd_counting_treat_cluster_id/var": "mismatch: var[0][2]: -0.015329 != 0.015329",
    "coxph/cgd_counting_treat_cluster_id/n": "mismatch: n (nobs): 76 != 203",
    "coxph/cgd_counting_treat_cluster_id/linear_predictors": (
        "mismatch: linear_predictors[0]: 0 != -0.89455"
    ),
    "coxph/cgd_counting_treat_cluster_id/naive_var": (
        "mismatch: naive_var[0][2]: -0.0060783 != 0.0060783"
    ),
    "coxph/cgd_counting_treat_cluster_id/residuals.martingale": (
        "mismatch: residuals.martingale[1]: 0.53933 != 0.57026"
    ),
    "coxph/cgd_counting_treat_cluster_id/residuals.deviance": (
        "mismatch: residuals.deviance[1]: 0.68665 != 0.74069"
    ),
    "coxph/cgd_counting_treat_cluster_id/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_treat_cluster_id/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_treat_cluster_id/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_treat_cluster_id/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_treat_cluster_id/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_treat_cluster_id/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_treat_cluster_id/summary.coefficients": (
        "mismatch: coefficients[treatplacebo].coef: 1.0722 != -1.0722"
    ),
    "coxph/cgd_counting_treat_cluster_id/summary.conf_int": (
        "mismatch: conf_int.lower[0]: 1.5857 != 0.18574"
    ),
    "coxph/cgd_counting_treat_cluster_id/wtest": "mismatch: wtest.solve[0]: 11.835 != -11.835",
    "coxph/cgd_counting_treat_id_robust/coef": "mismatch: coef[0]: 1.1201 != -1.1201",
    "coxph/cgd_counting_treat_id_robust/coef_names": (
        "mismatch: coef: names ['treatplacebo', 'age'] != ['treatrIFN-g', 'age']"
    ),
    "coxph/cgd_counting_treat_id_robust/var": "mismatch: var[0][1]: -0.00028726 != 0.00028726",
    "coxph/cgd_counting_treat_id_robust/n": "mismatch: n (nobs): 76 != 203",
    "coxph/cgd_counting_treat_id_robust/linear_predictors": (
        "mismatch: linear_predictors[0]: -0.36658 != -1.068"
    ),
    "coxph/cgd_counting_treat_id_robust/naive_var": (
        "mismatch: naive_var[0][1]: -0.00011551 != 0.00011551"
    ),
    "coxph/cgd_counting_treat_id_robust/residuals.martingale": (
        "mismatch: residuals.martingale[1]: 0.56231 != 0.59437"
    ),
    "coxph/cgd_counting_treat_id_robust/residuals.deviance": (
        "mismatch: residuals.deviance[1]: 0.72654 != 0.78478"
    ),
    "coxph/cgd_counting_treat_id_robust/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_treat_id_robust/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_treat_id_robust/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_treat_id_robust/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_treat_id_robust/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_treat_id_robust/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/cgd_counting_treat_id_robust/summary.coefficients": (
        "mismatch: coefficients[treatplacebo].coef: 1.1201 != -1.1201"
    ),
    "coxph/cgd_counting_treat_id_robust/summary.conf_int": (
        "mismatch: conf_int.lower[0]: 1.6697 != 0.17772"
    ),
    "coxph/cgd_counting_treat_id_robust/wtest": "mismatch: wtest.solve[0]: 11.269 != -11.269",
    "coxph/colon_death_rx_nodes_extent/coef": "mismatch: coef[0]: 0.37965 != -0.072874",
    "coxph/colon_death_rx_nodes_extent/coef_names": (
        "mismatch: coef: names ['rxObs', 'rxLev', 'nodes', 'extent', 'surg'] != ['rxLev'..."
    ),
    "coxph/colon_death_rx_nodes_extent/var": "mismatch: var[0][0]: 0.014567 != 0.012542",
    "coxph/colon_death_rx_nodes_extent/n": "mismatch: n (nobs): 441 != 911",
    "coxph/colon_death_rx_nodes_extent/linear_predictors": (
        "mismatch: linear_predictors[0]: 2.0812 != -0.19561"
    ),
    "coxph/colon_death_rx_nodes_extent/residuals.martingale": (
        "mismatch: residuals.martingale[18]: 0.89558 != 0.89611"
    ),
    "coxph/colon_death_rx_nodes_extent/residuals.deviance": (
        "mismatch: residuals.deviance[18]: 1.6515 != 1.6543"
    ),
    "coxph/colon_death_rx_nodes_extent/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/colon_death_rx_nodes_extent/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/colon_death_rx_nodes_extent/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/colon_death_rx_nodes_extent/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/colon_death_rx_nodes_extent/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/colon_death_rx_nodes_extent/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/colon_death_rx_nodes_extent/summary.coefficients": (
        "mismatch: coefficients[rxObs].coef: 0.37965 != -0.072874"
    ),
    "coxph/colon_death_rx_nodes_extent/summary.conf_int": (
        "mismatch: conf_int.lower[0]: 1.1538 != 0.74649"
    ),
    "coxph/colon_death_rx_nodes_extent/wtest": "mismatch: wtest.solve[0]: 16.565 != 16.252",
    "coxph/colon_death_rx_nodes_extent/anova": "mismatch: anova.df[1]: 3 != 1",
    "coxph/flchain_500_age_sex_kappa/n": "mismatch: n (nobs): 422 != 500",
    "coxph/flchain_500_age_sex_kappa/linear_predictors": (
        "mismatch: linear_predictors[0]: 14.114 != 2.5746"
    ),
    "coxph/flchain_500_age_sex_kappa/residuals.martingale": (
        "mismatch: residuals.martingale[0]: 0.34212 != 0.35247"
    ),
    "coxph/flchain_500_age_sex_kappa/residuals.deviance": (
        "mismatch: residuals.deviance[0]: 0.39144 != 0.40527"
    ),
    "coxph/flchain_500_age_sex_kappa/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/flchain_500_age_sex_kappa/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/flchain_500_age_sex_kappa/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/flchain_500_age_sex_kappa/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/flchain_500_age_sex_kappa/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/flchain_500_age_sex_kappa/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/flchain_500_age_sex_kappa/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/heart_counting_age_surgery_transplant/coef_names": (
        "mismatch: coef: names ['age', 'surgery', 'transplant'] != ['age', 'surgery', 't..."
    ),
    "coxph/heart_counting_age_surgery_transplant/n": "mismatch: n (nobs): 75 != 172",
    "coxph/heart_counting_age_surgery_transplant/linear_predictors": (
        "mismatch: linear_predictors[0]: -0.52386 != -0.44801"
    ),
    "coxph/heart_counting_age_surgery_transplant/residuals.martingale": (
        "mismatch: residuals.martingale[1]: 0.85598 != 0.86288"
    ),
    "coxph/heart_counting_age_surgery_transplant/residuals.deviance": (
        "mismatch: residuals.deviance[1]: 1.4709 != 1.4993"
    ),
    "coxph/heart_counting_age_surgery_transplant/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/heart_counting_age_surgery_transplant/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/heart_counting_age_surgery_transplant/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/heart_counting_age_surgery_transplant/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/heart_counting_age_surgery_transplant/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/heart_counting_age_surgery_transplant/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/heart_counting_age_surgery_transplant/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/heart_counting_breslow/coef_names": (
        "mismatch: coef: names ['age', 'surgery', 'transplant'] != ['age', 'surgery', 't..."
    ),
    "coxph/heart_counting_breslow/n": "mismatch: n (nobs): 75 != 172",
    "coxph/heart_counting_breslow/linear_predictors": (
        "mismatch: linear_predictors[0]: -0.52379 != -0.44795"
    ),
    "coxph/heart_counting_breslow/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/heart_counting_breslow/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/heart_counting_breslow/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/heart_counting_breslow/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/heart_counting_breslow/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/heart_counting_breslow/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/heart_counting_breslow/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/jasa_surgery_age/n": "mismatch: n (nobs): 75 != 103",
    "coxph/jasa_surgery_age/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.94619 != -0.43943"
    ),
    "coxph/jasa_surgery_age/residuals.martingale": (
        "mismatch: residuals.martingale[1]: 0.85575 != 0.86267"
    ),
    "coxph/jasa_surgery_age/residuals.deviance": "mismatch: residuals.deviance[1]: 1.47 != 1.4985",
    "coxph/jasa_surgery_age/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/jasa_surgery_age/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/jasa_surgery_age/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/jasa_surgery_age/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/jasa_surgery_age/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/jasa_surgery_age/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/jasa_surgery_age/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/kidney_age_sex/n": "mismatch: n (nobs): 58 != 76",
    "coxph/kidney_age_sex/linear_predictors": "mismatch: linear_predictors[0]: -0.77242 != 0.57918",
    "coxph/kidney_age_sex/residuals.martingale": (
        "mismatch: residuals.martingale[0]: 0.8813 != 0.89374"
    ),
    "coxph/kidney_age_sex/residuals.deviance": "mismatch: residuals.deviance[0]: 1.581 != 1.642",
    "coxph/kidney_age_sex/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/kidney_age_sex/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/kidney_age_sex/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/kidney_age_sex/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/kidney_age_sex/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/kidney_age_sex/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/kidney_age_sex/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/kidney_age_sex_disease/n": "mismatch: n (nobs): 58 != 76",
    "coxph/kidney_age_sex_disease/linear_predictors": (
        "mismatch: linear_predictors[0]: -1.3941 != 1.0429"
    ),
    "coxph/kidney_age_sex_disease/residuals.martingale": (
        "mismatch: residuals.martingale[0]: 0.84054 != 0.85746"
    ),
    "coxph/kidney_age_sex_disease/residuals.deviance": (
        "mismatch: residuals.deviance[0]: 1.411 != 1.4769"
    ),
    "coxph/kidney_age_sex_disease/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/kidney_age_sex_disease/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/kidney_age_sex_disease/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/kidney_age_sex_disease/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/kidney_age_sex_disease/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/kidney_age_sex_disease/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/kidney_age_sex_disease/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/lung_age_offset/n": "mismatch: n (nobs): 165 != 228",
    "coxph/lung_age_offset/linear_predictors": "mismatch: linear_predictors[0]: 2.6349 != 1.2552",
    "coxph/lung_age_offset/residuals.martingale": (
        "mismatch: residuals.martingale[6]: -0.22141 != -0.2116"
    ),
    "coxph/lung_age_offset/residuals.deviance": (
        "mismatch: residuals.deviance[6]: -0.2069 != -0.19828"
    ),
    "coxph/lung_age_offset/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_offset/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_offset/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_offset/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_offset/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_offset/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_breslow/n": "mismatch: n (nobs): 165 != 228",
    "coxph/lung_age_sex_breslow/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.74639 != 0.39887"
    ),
    "coxph/lung_age_sex_breslow/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_breslow/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_breslow/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_breslow/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_breslow/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_breslow/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_breslow/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/lung_age_sex_cluster_inst/n": "mismatch: n (nobs): 162 != 225",
    "coxph/lung_age_sex_cluster_inst/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.79178 != 0.40018"
    ),
    "coxph/lung_age_sex_cluster_inst/residuals.martingale": (
        "mismatch: residuals.martingale[6]: 0.43687 != 0.44188"
    ),
    "coxph/lung_age_sex_cluster_inst/residuals.deviance": (
        "mismatch: residuals.deviance[6]: 0.52417 != 0.5316"
    ),
    "coxph/lung_age_sex_cluster_inst/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_cluster_inst/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_cluster_inst/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_cluster_inst/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_cluster_inst/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_cluster_inst/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_efron/n": "mismatch: n (nobs): 165 != 228",
    "coxph/lung_age_sex_efron/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.74814 != 0.3995"
    ),
    "coxph/lung_age_sex_efron/residuals.martingale": (
        "mismatch: residuals.martingale[6]: 0.44267 != 0.44753"
    ),
    "coxph/lung_age_sex_efron/residuals.deviance": (
        "mismatch: residuals.deviance[6]: 0.53278 != 0.54005"
    ),
    "coxph/lung_age_sex_efron/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_efron/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_efron/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_efron/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_efron/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_efron/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_efron/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/lung_age_sex_eps/n": "mismatch: n (nobs): 165 != 228",
    "coxph/lung_age_sex_eps/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.74818 != 0.39944"
    ),
    "coxph/lung_age_sex_eps/residuals.martingale": (
        "mismatch: residuals.martingale[6]: 0.44262 != 0.44749"
    ),
    "coxph/lung_age_sex_eps/residuals.deviance": (
        "mismatch: residuals.deviance[6]: 0.53271 != 0.53999"
    ),
    "coxph/lung_age_sex_eps/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_eps/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_eps/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_eps/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_eps/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_eps/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_eps/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/lung_age_sex_exact/n": "mismatch: n (nobs): 165 != 228",
    "coxph/lung_age_sex_exact/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.7486 != 0.39993"
    ),
    "coxph/lung_age_sex_exact/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_exact/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_exact/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_exact/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_exact/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_exact/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_exact/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/lung_age_sex_init/n": "mismatch: n (nobs): 165 != 228",
    "coxph/lung_age_sex_init/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.74814 != 0.3995"
    ),
    "coxph/lung_age_sex_init/residuals.martingale": (
        "mismatch: residuals.martingale[6]: 0.44267 != 0.44753"
    ),
    "coxph/lung_age_sex_init/residuals.deviance": (
        "mismatch: residuals.deviance[6]: 0.53278 != 0.54005"
    ),
    "coxph/lung_age_sex_init/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_init/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_init/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_init/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_init/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_init/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_init/anova": "mismatch: anova.p[0]: got nan, expected 1",
    "coxph/lung_age_sex_init_iter0/n": "mismatch: n (nobs): 165 != 228",
    "coxph/lung_age_sex_init_iter0/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.98 != 0.42842"
    ),
    "coxph/lung_age_sex_init_iter0/residuals.martingale": (
        "mismatch: residuals.martingale[6]: 0.43028 != 0.43525"
    ),
    "coxph/lung_age_sex_init_iter0/residuals.deviance": (
        "mismatch: residuals.deviance[6]: 0.51445 != 0.52177"
    ),
    "coxph/lung_age_sex_init_iter0/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_init_iter0/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_init_iter0/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_init_iter0/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_init_iter0/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_init_iter0/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_init_iter0/anova": "mismatch: anova.p[0]: got nan, expected 1",
    "coxph/lung_age_sex_nocenter_null/n": "mismatch: n (nobs): 165 != 228",
    "coxph/lung_age_sex_nocenter_null/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.74814 != 0.3995"
    ),
    "coxph/lung_age_sex_nocenter_null/residuals.martingale": (
        "mismatch: residuals.martingale[6]: 0.44267 != 0.44753"
    ),
    "coxph/lung_age_sex_nocenter_null/residuals.deviance": (
        "mismatch: residuals.deviance[6]: 0.53278 != 0.54005"
    ),
    "coxph/lung_age_sex_nocenter_null/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_nocenter_null/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_nocenter_null/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_nocenter_null/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_nocenter_null/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_nocenter_null/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_nocenter_null/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/lung_age_sex_robust/n": "mismatch: n (nobs): 165 != 228",
    "coxph/lung_age_sex_robust/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.74814 != 0.3995"
    ),
    "coxph/lung_age_sex_robust/residuals.martingale": (
        "mismatch: residuals.martingale[6]: 0.44267 != 0.44753"
    ),
    "coxph/lung_age_sex_robust/residuals.deviance": (
        "mismatch: residuals.deviance[6]: 0.53278 != 0.54005"
    ),
    "coxph/lung_age_sex_robust/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_robust/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_robust/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_robust/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_robust/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_robust/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_weighted/n": "mismatch: n (nobs): 165 != 228",
    "coxph/lung_age_sex_weighted/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.1229 != 0.34273"
    ),
    "coxph/lung_age_sex_weighted/residuals.martingale": (
        "mismatch: residuals.martingale[6]: 0.4397 != 0.44453"
    ),
    "coxph/lung_age_sex_weighted/residuals.deviance": (
        "mismatch: residuals.deviance[6]: 0.52836 != 0.53556"
    ),
    "coxph/lung_age_sex_weighted/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_weighted/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_weighted/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_weighted/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_weighted/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_weighted/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_weighted_breslow/n": "mismatch: n (nobs): 165 != 228",
    "coxph/lung_age_sex_weighted_breslow/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.12189 != 0.34198"
    ),
    "coxph/lung_age_sex_weighted_breslow/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_weighted_breslow/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_weighted_breslow/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_weighted_breslow/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_weighted_breslow/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_weighted_breslow/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_x_true/n": "mismatch: n (nobs): 165 != 228",
    "coxph/lung_age_sex_x_true/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.74814 != 0.3995"
    ),
    "coxph/lung_age_sex_x_true/residuals.martingale": (
        "mismatch: residuals.martingale[6]: 0.44267 != 0.44753"
    ),
    "coxph/lung_age_sex_x_true/residuals.deviance": (
        "mismatch: residuals.deviance[6]: 0.53278 != 0.54005"
    ),
    "coxph/lung_age_sex_x_true/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_x_true/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_x_true/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_x_true/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_x_true/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_x_true/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_sex_x_true/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/lung_age_strata_sex/n": "mismatch: n (nobs): 162 != 225",
    "coxph/lung_age_strata_sex/linear_predictors": (
        "mismatch: linear_predictors[0]: 1.2671 != 0.15271"
    ),
    "coxph/lung_age_strata_sex/residuals.martingale": (
        "mismatch: residuals.martingale[18]: 0.85352 != 0.85754"
    ),
    "coxph/lung_age_strata_sex/residuals.deviance": (
        "mismatch: residuals.deviance[18]: 1.4611 != 1.4773"
    ),
    "coxph/lung_age_strata_sex/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_strata_sex/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_strata_sex/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_strata_sex/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_strata_sex/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_strata_sex/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_age_strata_sex/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/lung_factor_ph_ecog/coef": "mismatch: coef[2]: -0.40928 != 0.40928",
    "coxph/lung_factor_ph_ecog/coef_names": (
        "mismatch: coef: names ['age', 'sex', 'factor(ph.ecog)0', 'factor(ph.ecog)2', 'f..."
    ),
    "coxph/lung_factor_ph_ecog/var": "mismatch: var[0][2]: 7.5297e-05 != -7.5297e-05",
    "coxph/lung_factor_ph_ecog/n": "mismatch: n (nobs): 162 != 225",
    "coxph/lung_factor_ph_ecog/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.28322 != 0.75565"
    ),
    "coxph/lung_factor_ph_ecog/residuals.martingale": (
        "mismatch: residuals.martingale[6]: 0.13346 != 0.14147"
    ),
    "coxph/lung_factor_ph_ecog/residuals.deviance": (
        "mismatch: residuals.deviance[6]: 0.1399 != 0.14875"
    ),
    "coxph/lung_factor_ph_ecog/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_factor_ph_ecog/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_factor_ph_ecog/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_factor_ph_ecog/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_factor_ph_ecog/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_factor_ph_ecog/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_factor_ph_ecog/summary.coefficients": (
        "mismatch: coefficients[factor(ph.ecog)0].coef: -0.40928 != 0.40928"
    ),
    "coxph/lung_factor_ph_ecog/summary.conf_int": "mismatch: conf_int.lower[2]: 0.4491 != 1.0182",
    "coxph/lung_factor_ph_ecog/wtest": "mismatch: wtest.solve[2]: -15.666 != -4.9835",
    "coxph/lung_factor_ph_ecog/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/lung_interaction/n": "mismatch: n (nobs): 162 != 225",
    "coxph/lung_interaction/linear_predictors": "mismatch: linear_predictors[0]: 0.38308 != 0.2815",
    "coxph/lung_interaction/residuals.martingale": (
        "mismatch: residuals.martingale[6]: 0.37857 != 0.38414"
    ),
    "coxph/lung_interaction/residuals.deviance": (
        "mismatch: residuals.deviance[6]: 0.44083 != 0.44855"
    ),
    "coxph/lung_interaction/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_interaction/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_interaction/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_interaction/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_interaction/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_interaction/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_interaction/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/lung_log_transform/n": "mismatch: n (nobs): 165 != 228",
    "coxph/lung_log_transform/linear_predictors": (
        "mismatch: linear_predictors[0]: 3.8156 != 0.38466"
    ),
    "coxph/lung_log_transform/residuals.martingale": (
        "mismatch: residuals.martingale[6]: 0.44157 != 0.44644"
    ),
    "coxph/lung_log_transform/residuals.deviance": (
        "mismatch: residuals.deviance[6]: 0.53114 != 0.53841"
    ),
    "coxph/lung_log_transform/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_log_transform/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_log_transform/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_log_transform/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_log_transform/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_log_transform/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/lung_log_transform/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/mgus2_age_sex_mspike/n": "mismatch: n (nobs): 957 != 1373",
    "coxph/mgus2_age_sex_mspike/linear_predictors": (
        "mismatch: linear_predictors[0]: 5.4904 != 1.0709"
    ),
    "coxph/mgus2_age_sex_mspike/residuals.martingale": (
        "mismatch: residuals.martingale[0]: 0.5553 != 0.56184"
    ),
    "coxph/mgus2_age_sex_mspike/residuals.deviance": (
        "mismatch: residuals.deviance[0]: 0.71421 != 0.72571"
    ),
    "coxph/mgus2_age_sex_mspike/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/mgus2_age_sex_mspike/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/mgus2_age_sex_mspike/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/mgus2_age_sex_mspike/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/mgus2_age_sex_mspike/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/mgus2_age_sex_mspike/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/mgus2_age_sex_mspike/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/myeloid_trt_sex_flt3/coef": "mismatch: coef[0]: 0.37368 != -0.37368",
    "coxph/myeloid_trt_sex_flt3/coef_names": (
        "mismatch: coef: names ['trtA', 'sexm', 'flt3B', 'flt3A'] != ['trtB', 'sexm', 'f..."
    ),
    "coxph/myeloid_trt_sex_flt3/var": "mismatch: var[0][1]: 0.0016195 != -0.0016195",
    "coxph/myeloid_trt_sex_flt3/n": "mismatch: n (nobs): 320 != 646",
    "coxph/myeloid_trt_sex_flt3/linear_predictors": "mismatch: linear_predictors[0]: 0 != 0.43298",
    "coxph/myeloid_trt_sex_flt3/residuals.martingale": (
        "mismatch: residuals.martingale[0]: 0.8213 != 0.82233"
    ),
    "coxph/myeloid_trt_sex_flt3/residuals.deviance": (
        "mismatch: residuals.deviance[0]: 1.3422 != 1.3457"
    ),
    "coxph/myeloid_trt_sex_flt3/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/myeloid_trt_sex_flt3/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/myeloid_trt_sex_flt3/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/myeloid_trt_sex_flt3/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/myeloid_trt_sex_flt3/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/myeloid_trt_sex_flt3/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/myeloid_trt_sex_flt3/summary.coefficients": (
        "mismatch: coefficients[trtA].coef: 0.37368 != -0.37368"
    ),
    "coxph/myeloid_trt_sex_flt3/summary.conf_int": "mismatch: conf_int.lower[0]: 1.164 != 0.55131",
    "coxph/myeloid_trt_sex_flt3/wtest": "mismatch: wtest.solve[0]: 27.213 != -27.213",
    "coxph/myeloid_trt_sex_flt3/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/nafld1_1000_age_male_bmi/n": "mismatch: n (nobs): 50 != 726",
    "coxph/nafld1_1000_age_male_bmi/linear_predictors": (
        "mismatch: linear_predictors[0]: 6.3981 != 0.21565"
    ),
    "coxph/nafld1_1000_age_male_bmi/residuals.martingale": (
        "mismatch: residuals.martingale[96]: 0.97997 != 0.98092"
    ),
    "coxph/nafld1_1000_age_male_bmi/residuals.deviance": (
        "mismatch: residuals.deviance[96]: 2.421 != 2.4405"
    ),
    "coxph/nafld1_1000_age_male_bmi/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/nafld1_1000_age_male_bmi/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/nafld1_1000_age_male_bmi/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/nafld1_1000_age_male_bmi/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/nafld1_1000_age_male_bmi/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/nafld1_1000_age_male_bmi/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/nafld1_1000_age_male_bmi/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/ovarian_age_rx/n": "mismatch: n (nobs): 12 != 26",
    "coxph/ovarian_age_rx/linear_predictors": "mismatch: linear_predictors[0]: 9.8524 != 2.7837",
    "coxph/ovarian_age_rx/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/ovarian_age_rx/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/ovarian_age_rx/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/ovarian_age_rx/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/ovarian_age_rx/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/ovarian_age_rx/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/ovarian_age_rx/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/ovarian_age_rx_exact/n": "mismatch: n (nobs): 12 != 26",
    "coxph/ovarian_age_rx_exact/linear_predictors": (
        "mismatch: linear_predictors[0]: 9.8524 != 2.7837"
    ),
    "coxph/ovarian_age_rx_exact/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/ovarian_age_rx_exact/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/ovarian_age_rx_exact/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/ovarian_age_rx_exact/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/ovarian_age_rx_exact/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/ovarian_age_rx_exact/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/ovarian_age_rx_exact/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/pbc_trial_age_edema_strata_sex/n": "mismatch: n (nobs): 125 != 312",
    "coxph/pbc_trial_age_edema_strata_sex/linear_predictors": (
        "mismatch: linear_predictors[0]: 6.1042 != 3.6062"
    ),
    "coxph/pbc_trial_age_edema_strata_sex/residuals.martingale": (
        "mismatch: residuals.martingale[22]: 0.10142 != 0.13678"
    ),
    "coxph/pbc_trial_age_edema_strata_sex/residuals.deviance": (
        "mismatch: residuals.deviance[22]: 0.10506 != 0.14356"
    ),
    "coxph/pbc_trial_age_edema_strata_sex/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/pbc_trial_age_edema_strata_sex/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/pbc_trial_age_edema_strata_sex/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/pbc_trial_age_edema_strata_sex/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/pbc_trial_age_edema_strata_sex/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/pbc_trial_age_edema_strata_sex/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/pbc_trial_age_edema_strata_sex/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/pbc_trial_trt_factor/n": "mismatch: n (nobs): 125 != 312",
    "coxph/pbc_trial_trt_factor/linear_predictors": (
        "mismatch: linear_predictors[0]: -2.3562 != 1.8251"
    ),
    "coxph/pbc_trial_trt_factor/residuals.martingale": (
        "mismatch: residuals.martingale[22]: 0.87106 != 0.8752"
    ),
    "coxph/pbc_trial_trt_factor/residuals.deviance": (
        "mismatch: residuals.deviance[22]: 1.5345 != 1.5529"
    ),
    "coxph/pbc_trial_trt_factor/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/pbc_trial_trt_factor/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/pbc_trial_trt_factor/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/pbc_trial_trt_factor/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/pbc_trial_trt_factor/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/pbc_trial_trt_factor/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/pbc_trial_trt_factor/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/rats_rx_litter_cluster/n": "mismatch: n (nobs): 42 != 300",
    "coxph/rats_rx_litter_cluster/residuals.martingale": (
        "mismatch: residuals.martingale[31]: 0.88866 != 0.89109"
    ),
    "coxph/rats_rx_litter_cluster/residuals.deviance": (
        "mismatch: residuals.deviance[31]: 1.6165 != 1.6286"
    ),
    "coxph/rats_rx_litter_cluster/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/rats_rx_litter_cluster/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/rats_rx_litter_cluster/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/rats_rx_litter_cluster/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/rats_rx_litter_cluster/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/rats_rx_litter_cluster/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_delayed_x/n": "mismatch: n (nobs): 8 != 12",
    "coxph/synthetic_delayed_x/linear_predictors": (
        "mismatch: linear_predictors[0]: -0.61326 != -0.3884"
    ),
    "coxph/synthetic_delayed_x/residuals.martingale": (
        "mismatch: residuals.martingale[3]: 0.38673 != 0.48532"
    ),
    "coxph/synthetic_delayed_x/residuals.deviance": (
        "mismatch: residuals.deviance[3]: 0.45215 != 0.59815"
    ),
    "coxph/synthetic_delayed_x/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_delayed_x/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_delayed_x/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_delayed_x/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_delayed_x/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_delayed_x/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_delayed_x_breslow/n": "mismatch: n (nobs): 8 != 12",
    "coxph/synthetic_delayed_x_breslow/linear_predictors": (
        "mismatch: linear_predictors[0]: -0.59182 != -0.37482"
    ),
    "coxph/synthetic_delayed_x_breslow/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_delayed_x_breslow/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_delayed_x_breslow/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_delayed_x_breslow/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_delayed_x_breslow/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_delayed_x_breslow/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_delayed_x_exact/n": "mismatch: n (nobs): 8 != 12",
    "coxph/synthetic_delayed_x_exact/linear_predictors": (
        "mismatch: linear_predictors[0]: -0.62639 != -0.39671"
    ),
    "coxph/synthetic_delayed_x_exact/residuals.deviance": (
        "mismatch: residuals.deviance[0]: 1.4061 != 0.83923"
    ),
    "coxph/synthetic_delayed_x_exact/concordance.r_error": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_delayed_x_exact/summary.r_error": (
        "missing feature: unhandled summary aspect summary.r_error"
    ),
    "coxph/synthetic_ties_breslow/n": "mismatch: n (nobs): 12 != 16",
    "coxph/synthetic_ties_breslow/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.38227 != 0.14335"
    ),
    "coxph/synthetic_ties_breslow/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_ties_breslow/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_ties_breslow/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_ties_breslow/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_ties_breslow/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_ties_breslow/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_ties_breslow/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/synthetic_ties_efron/n": "mismatch: n (nobs): 12 != 16",
    "coxph/synthetic_ties_efron/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.38214 != 0.1433"
    ),
    "coxph/synthetic_ties_efron/residuals.martingale": (
        "mismatch: residuals.martingale[0]: 0.84016 != 0.8819"
    ),
    "coxph/synthetic_ties_efron/residuals.deviance": (
        "mismatch: residuals.deviance[0]: 1.4095 != 1.5839"
    ),
    "coxph/synthetic_ties_efron/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_ties_efron/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_ties_efron/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_ties_efron/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_ties_efron/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_ties_efron/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_ties_efron/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/synthetic_ties_exact/n": "mismatch: n (nobs): 12 != 16",
    "coxph/synthetic_ties_exact/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.47841 != 0.1794"
    ),
    "coxph/synthetic_ties_exact/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_ties_exact/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_ties_exact/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_ties_exact/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_ties_exact/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_ties_exact/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/synthetic_ties_exact/anova": "mismatch: anova.df[1]: 2 != 1",
    "coxph/synthetic_timefix_true/loglik": "mismatch: loglik: length 2 differs from expected 1",
    "coxph/synthetic_timefix_true/score": "mismatch: score: expected NA/NaN, got 0",
    "coxph/synthetic_timefix_true/iter": "mismatch: iter: length 1 differs from expected 0",
    "coxph/synthetic_timefix_true/n": "mismatch: n (nobs): 6 != 8",
    "coxph/synthetic_timefix_true/curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph/synthetic_timefix_true/curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.14522 != 0.18982"
    ),
    "coxph/transplant_ltx_age_abo/coef": "mismatch: coef[1]: 0.33684 != -0.33684",
    "coxph/transplant_ltx_age_abo/coef_names": (
        "mismatch: coef: names ['age', 'aboA', 'aboO', 'aboAB'] != ['age', 'aboB', 'aboA..."
    ),
    "coxph/transplant_ltx_age_abo/var": "mismatch: var[0][1]: 2.5513e-05 != -2.5513e-05",
    "coxph/transplant_ltx_age_abo/n": "mismatch: n (nobs): 618 != 797",
    "coxph/transplant_ltx_age_abo/linear_predictors": (
        "mismatch: linear_predictors[0]: -0.24204 != -0.31871"
    ),
    "coxph/transplant_ltx_age_abo/residuals.martingale": (
        "mismatch: residuals.martingale[1]: 0.84805 != 0.84992"
    ),
    "coxph/transplant_ltx_age_abo/residuals.deviance": (
        "mismatch: residuals.deviance[1]: 1.4396 != 1.4468"
    ),
    "coxph/transplant_ltx_age_abo/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/transplant_ltx_age_abo/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/transplant_ltx_age_abo/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/transplant_ltx_age_abo/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/transplant_ltx_age_abo/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/transplant_ltx_age_abo/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/transplant_ltx_age_abo/summary.coefficients": (
        "mismatch: coefficients[aboA].coef: 0.33684 != -0.33684"
    ),
    "coxph/transplant_ltx_age_abo/summary.conf_int": (
        "mismatch: conf_int.lower[1]: 1.084 != 0.55264"
    ),
    "coxph/transplant_ltx_age_abo/wtest": "mismatch: wtest.solve[1]: 69.615 != -5.4707",
    "coxph/transplant_ltx_age_abo/anova": "mismatch: anova.df[1]: 4 != 3",
    "coxph/veteran_celltype_karno_trt/n": "mismatch: n (nobs): 128 != 137",
    "coxph/veteran_celltype_karno_trt/linear_predictors": (
        "mismatch: linear_predictors[0]: -1.6145 != -0.17466"
    ),
    "coxph/veteran_celltype_karno_trt/residuals.martingale": (
        "mismatch: residuals.martingale[5]: 0.87029 != 0.87538"
    ),
    "coxph/veteran_celltype_karno_trt/residuals.deviance": (
        "mismatch: residuals.deviance[5]: 1.5311 != 1.5538"
    ),
    "coxph/veteran_celltype_karno_trt/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/veteran_celltype_karno_trt/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/veteran_celltype_karno_trt/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/veteran_celltype_karno_trt/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/veteran_celltype_karno_trt/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/veteran_celltype_karno_trt/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/veteran_celltype_karno_trt/anova": "mismatch: anova.df[1]: 4 != 3",
    "coxph/veteran_celltype_x_true/n": "mismatch: n (nobs): 128 != 137",
    "coxph/veteran_celltype_x_true/linear_predictors": (
        "mismatch: linear_predictors[0]: -1.8634 != -0.044431"
    ),
    "coxph/veteran_celltype_x_true/residuals.martingale": (
        "mismatch: residuals.martingale[5]: 0.84227 != 0.84848"
    ),
    "coxph/veteran_celltype_x_true/residuals.deviance": (
        "mismatch: residuals.deviance[5]: 1.4175 != 1.4412"
    ),
    "coxph/veteran_celltype_x_true/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/veteran_celltype_x_true/concordance.n": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/veteran_celltype_x_true/concordance.count": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/veteran_celltype_x_true/concordance.var": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/veteran_celltype_x_true/concordance.cvar": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/veteran_celltype_x_true/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/veteran_celltype_x_true/anova": "mismatch: anova.df[1]: 4 != 1",
    "coxph/veteran_tt_karno/n": "mismatch: n (nobs): 128 != 137",
    "coxph/veteran_tt_karno/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph/veteran_tt_strata/n": "mismatch: n (nobs): 128 != 137",
    "coxph/veteran_tt_strata/summary.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "coxph_diagnostics/aml_x/residuals.martingale": (
        "mismatch: residuals.martingale[4]: 0.65156 != 0.67626"
    ),
    "coxph_diagnostics/aml_x/residuals.deviance": (
        "mismatch: residuals.deviance[4]: 0.89747 != 0.95033"
    ),
    "coxph_diagnostics/aml_x/residuals.partial": (
        "mismatch: residuals.partial[4][0]: 0.65156 != 0.67626"
    ),
    "coxph_diagnostics/aml_x/zph.km_terms.table": "mismatch: zph[x].chisq: 0.0062974 != 0.0078751",
    "coxph_diagnostics/cgd_counting_treat_cluster_id/residuals.martingale": (
        "mismatch: residuals.martingale[1]: 0.53933 != 0.57026"
    ),
    "coxph_diagnostics/cgd_counting_treat_cluster_id/residuals.deviance": (
        "mismatch: residuals.deviance[1]: 0.68665 != 0.74069"
    ),
    "coxph_diagnostics/cgd_counting_treat_cluster_id/residuals.score": (
        "mismatch: residuals.score[0][0]: -0.60451 != 0.60451"
    ),
    "coxph_diagnostics/cgd_counting_treat_cluster_id/residuals.schoenfeld": (
        "mismatch: residuals.schoenfeld[0][0]: 0.24279 != -0.24279"
    ),
    "coxph_diagnostics/cgd_counting_treat_cluster_id/residuals.scaledsch": (
        "mismatch: residuals.scaledsch[0][0]: 2.8201 != -2.1942"
    ),
    "coxph_diagnostics/cgd_counting_treat_cluster_id/residuals.dfbeta": (
        "mismatch: residuals.dfbeta[0][0]: -0.042918 != 0.042918"
    ),
    "coxph_diagnostics/cgd_counting_treat_cluster_id/residuals.dfbetas": (
        "mismatch: residuals.dfbetas[0][0]: -0.16389 != 0.16389"
    ),
    "coxph_diagnostics/cgd_counting_treat_cluster_id/residuals.partial": (
        "mismatch: residuals.partial[0][0]: 0.81336 != -0.25886"
    ),
    "coxph_diagnostics/cgd_counting_treat_cluster_id/zph.km_terms.table": (
        "mismatch: zph[treat].chisq: 0.32862 != 0.36352"
    ),
    "coxph_diagnostics/cgd_counting_treat_cluster_id/zph.km_terms.y": (
        "mismatch: zph.y[0][0]: 2.8201 != -2.1942"
    ),
    "coxph_diagnostics/cgd_counting_treat_cluster_id/zph.km_terms.var": (
        "mismatch: zph.var[0][2]: -0.46195 != 0.46195"
    ),
    "coxph_diagnostics/cgd_counting_treat_cluster_id/detail.hazard": (
        "mismatch: detail.hazard[0]: 0.0042859 != 0.010484"
    ),
    "coxph_diagnostics/cgd_counting_treat_cluster_id/detail.varhaz": (
        "mismatch: detail.varhaz[0]: 1.8369e-05 != 0.00010992"
    ),
    "coxph_diagnostics/cgd_counting_treat_cluster_id/detail.wtrisk": (
        "mismatch: detail.wtrisk[0]: 233.33 != 95.382"
    ),
    "coxph_diagnostics/cgd_counting_treat_cluster_id/detail.score": (
        "mismatch: detail.score[0][0]: 0.24279 != -0.24279"
    ),
    "coxph_diagnostics/cgd_counting_treat_cluster_id/detail.means": (
        "mismatch: detail.means[0][0]: 0.75721 != 0.24279"
    ),
    "coxph_diagnostics/cgd_counting_treat_cluster_id/detail.imat": (
        "mismatch: detail.imat[0][0][2]: 0.0062121 != -0.0062121"
    ),
    "coxph_diagnostics/heart_counting_age_surgery_transplant/residuals.martingale": (
        "mismatch: residuals.martingale[1]: 0.85598 != 0.86288"
    ),
    "coxph_diagnostics/heart_counting_age_surgery_transplant/residuals.deviance": (
        "mismatch: residuals.deviance[1]: 1.4709 != 1.4993"
    ),
    "coxph_diagnostics/heart_counting_age_surgery_transplant/residuals.partial": (
        "mismatch: residuals.partial[1][0]: 1.049 != 1.0559"
    ),
    "coxph_diagnostics/heart_counting_age_surgery_transplant/zph.km_terms.table": (
        "mismatch: zph[age].chisq: 0.7564 != 0.89538"
    ),
    "coxph_diagnostics/heart_counting_age_surgery_transplant/zph.km_terms.y": (
        "mismatch: zph.y[1][0]: -0.028608 != 0.097083"
    ),
    "coxph_diagnostics/heart_counting_age_surgery_transplant/detail.wtrisk": (
        "mismatch: detail.wtrisk[0]: 90.158 != 97.262"
    ),
    "coxph_diagnostics/kidney_age_sex_disease/residuals.martingale": (
        "mismatch: residuals.martingale[0]: 0.84054 != 0.85746"
    ),
    "coxph_diagnostics/kidney_age_sex_disease/residuals.deviance": (
        "mismatch: residuals.deviance[0]: 1.411 != 1.4769"
    ),
    "coxph_diagnostics/kidney_age_sex_disease/residuals.partial": (
        "mismatch: residuals.partial[0][0]: 0.79061 != 0.80753"
    ),
    "coxph_diagnostics/kidney_age_sex_disease/zph.km_terms.table": (
        "mismatch: zph[age].chisq: 0.087152 != 0.10481"
    ),
    "coxph_diagnostics/kidney_age_sex_disease/zph.km_terms.y": (
        "mismatch: zph.y[0][0]: 0.047938 != 0.040218"
    ),
    "coxph_diagnostics/kidney_age_sex_disease/zph.km_terms.var": (
        "mismatch: zph.var[0][0]: 0.0072057 != 0.0047672"
    ),
    "coxph_diagnostics/kidney_age_sex_disease/detail.wtrisk": (
        "mismatch: detail.wtrisk[0]: 8.7029 != 99.548"
    ),
    "coxph_diagnostics/lung_age_offset/residuals.martingale": (
        "mismatch: residuals.martingale[6]: -0.22141 != -0.2116"
    ),
    "coxph_diagnostics/lung_age_offset/residuals.deviance": (
        "mismatch: residuals.deviance[6]: -0.2069 != -0.19828"
    ),
    "coxph_diagnostics/lung_age_offset/residuals.partial": (
        "mismatch: residuals.partial[6][0]: -0.098737 != -0.088918"
    ),
    "coxph_diagnostics/lung_age_offset/zph.km_terms.table": (
        "mismatch: zph[age].chisq: 0.69677 != 0.76759"
    ),
    "coxph_diagnostics/lung_age_offset/detail.hazard": (
        "mismatch: detail.hazard[0]: 0.0038511 != 0.00095469"
    ),
    "coxph_diagnostics/lung_age_offset/detail.varhaz": (
        "mismatch: detail.varhaz[0]: 1.4831e-05 != 9.1143e-07"
    ),
    "coxph_diagnostics/lung_age_offset/detail.wtrisk": (
        "mismatch: detail.wtrisk[0]: 4162.2 != 1047.5"
    ),
    "coxph_diagnostics/lung_age_sex_breslow/zph.km_terms.table": (
        "mismatch: zph[age].chisq: 0.12275 != 0.20796"
    ),
    "coxph_diagnostics/lung_age_sex_breslow/zph.km_noterms.table": (
        "mismatch: zph[age].chisq: 0.12275 != 0.20796"
    ),
    "coxph_diagnostics/lung_age_sex_breslow/zph.rank_terms.table": (
        "mismatch: zph[age].chisq: 0.27679 != 0.1363"
    ),
    "coxph_diagnostics/lung_age_sex_breslow/zph.rank_terms.x": "mismatch: zph.x[27]: 28 != 28.5",
    "coxph_diagnostics/lung_age_sex_breslow/zph.rank_noterms.table": (
        "mismatch: zph[age].chisq: 0.27679 != 0.1363"
    ),
    "coxph_diagnostics/lung_age_sex_breslow/zph.rank_noterms.x": "mismatch: zph.x[27]: 28 != 28.5",
    "coxph_diagnostics/lung_age_sex_breslow/zph.identity_terms.table": (
        "mismatch: zph[age].chisq: 0.071204 != 0.14062"
    ),
    "coxph_diagnostics/lung_age_sex_breslow/zph.identity_noterms.table": (
        "mismatch: zph[age].chisq: 0.071204 != 0.14062"
    ),
    "coxph_diagnostics/lung_age_sex_breslow/detail.wtrisk": (
        "mismatch: detail.wtrisk[0]: 337.95 != 238.74"
    ),
    "coxph_diagnostics/lung_age_sex_cluster_inst/residuals.martingale": (
        "mismatch: residuals.martingale[6]: 0.43687 != 0.44188"
    ),
    "coxph_diagnostics/lung_age_sex_cluster_inst/residuals.deviance": (
        "mismatch: residuals.deviance[6]: 0.52417 != 0.5316"
    ),
    "coxph_diagnostics/lung_age_sex_cluster_inst/residuals.scaledsch": (
        "mismatch: residuals.scaledsch[0][0]: 0.04493 != 0.039849"
    ),
    "coxph_diagnostics/lung_age_sex_cluster_inst/residuals.partial": (
        "mismatch: residuals.partial[6][0]: 0.53385 != 0.53885"
    ),
    "coxph_diagnostics/lung_age_sex_cluster_inst/zph.km_terms.table": (
        "mismatch: zph[age].chisq: 0.056506 != 0.2758"
    ),
    "coxph_diagnostics/lung_age_sex_cluster_inst/zph.km_terms.y": (
        "mismatch: zph.y[0][0]: 0.04493 != 0.039849"
    ),
    "coxph_diagnostics/lung_age_sex_cluster_inst/detail.wtrisk": (
        "mismatch: detail.wtrisk[0]: 348.22 != 235.39"
    ),
    "coxph_diagnostics/lung_age_sex_efron/residuals.martingale": (
        "mismatch: residuals.martingale[6]: 0.44267 != 0.44753"
    ),
    "coxph_diagnostics/lung_age_sex_efron/residuals.deviance": (
        "mismatch: residuals.deviance[6]: 0.53278 != 0.54005"
    ),
    "coxph_diagnostics/lung_age_sex_efron/residuals.partial": (
        "mismatch: residuals.partial[6][0]: 0.53731 != 0.54218"
    ),
    "coxph_diagnostics/lung_age_sex_efron/zph.km_terms.table": (
        "mismatch: zph[age].chisq: 0.1236 != 0.2092"
    ),
    "coxph_diagnostics/lung_age_sex_efron/zph.km_noterms.table": (
        "mismatch: zph[age].chisq: 0.1236 != 0.2092"
    ),
    "coxph_diagnostics/lung_age_sex_efron/zph.rank_terms.table": (
        "mismatch: zph[age].chisq: 0.278 != 0.13702"
    ),
    "coxph_diagnostics/lung_age_sex_efron/zph.rank_terms.x": "mismatch: zph.x[27]: 28 != 28.5",
    "coxph_diagnostics/lung_age_sex_efron/zph.rank_noterms.table": (
        "mismatch: zph[age].chisq: 0.278 != 0.13702"
    ),
    "coxph_diagnostics/lung_age_sex_efron/zph.rank_noterms.x": "mismatch: zph.x[27]: 28 != 28.5",
    "coxph_diagnostics/lung_age_sex_efron/zph.identity_terms.table": (
        "mismatch: zph[age].chisq: 0.071962 != 0.14175"
    ),
    "coxph_diagnostics/lung_age_sex_efron/zph.identity_noterms.table": (
        "mismatch: zph[age].chisq: 0.071962 != 0.14175"
    ),
    "coxph_diagnostics/lung_age_sex_efron/detail.wtrisk": (
        "mismatch: detail.wtrisk[0]: 338.37 != 238.77"
    ),
    "coxph_diagnostics/lung_age_sex_exact/zph.km_terms.table": (
        "mismatch: zph[age].chisq: 0.12337 != 0.20734"
    ),
    "coxph_diagnostics/lung_age_sex_exact/zph.km_terms.y": (
        "mismatch: zph.y[0][0]: 0.040446 != 0.040417"
    ),
    "coxph_diagnostics/lung_age_sex_exact/zph.km_terms.var": (
        "mismatch: zph.var[0][0]: 0.014071 != 0.014035"
    ),
    "coxph_diagnostics/lung_age_sex_nocenter_null/residuals.martingale": (
        "mismatch: residuals.martingale[6]: 0.44267 != 0.44753"
    ),
    "coxph_diagnostics/lung_age_sex_nocenter_null/residuals.deviance": (
        "mismatch: residuals.deviance[6]: 0.53278 != 0.54005"
    ),
    "coxph_diagnostics/lung_age_sex_nocenter_null/residuals.partial": (
        "mismatch: residuals.partial[6][0]: 0.53731 != 0.54218"
    ),
    "coxph_diagnostics/lung_age_sex_nocenter_null/zph.km_terms.table": (
        "mismatch: zph[age].chisq: 0.1236 != 0.2092"
    ),
    "coxph_diagnostics/lung_age_sex_nocenter_null/detail.wtrisk": (
        "mismatch: detail.wtrisk[0]: 338.37 != 238.77"
    ),
    "coxph_diagnostics/lung_age_sex_weighted/residuals.martingale": (
        "mismatch: residuals.martingale[6]: 0.4397 != 0.44453"
    ),
    "coxph_diagnostics/lung_age_sex_weighted/residuals.deviance": (
        "mismatch: residuals.deviance[6]: 0.52836 != 0.53556"
    ),
    "coxph_diagnostics/lung_age_sex_weighted/residuals.scaledsch": (
        "mismatch: residuals.scaledsch[0][0]: 0.029514 != 0.042998"
    ),
    "coxph_diagnostics/lung_age_sex_weighted/residuals.partial": (
        "mismatch: residuals.partial[6][0]: 0.49318 != 0.49802"
    ),
    "coxph_diagnostics/lung_age_sex_weighted/zph.km_terms.table": (
        "mismatch: zph[age].chisq: 1.4427 != 1.1098"
    ),
    "coxph_diagnostics/lung_age_sex_weighted/zph.km_terms.y": (
        "mismatch: zph.y[0][0]: 0.029514 != 0.042998"
    ),
    "coxph_diagnostics/lung_age_sex_weighted/detail.wtrisk": (
        "mismatch: detail.wtrisk[0]: 239.03 != 297.8"
    ),
    "coxph_diagnostics/lung_age_strata_sex/residuals.martingale": (
        "mismatch: residuals.martingale[18]: 0.85352 != 0.85754"
    ),
    "coxph_diagnostics/lung_age_strata_sex/residuals.deviance": (
        "mismatch: residuals.deviance[18]: 1.4611 != 1.4773"
    ),
    "coxph_diagnostics/lung_age_strata_sex/residuals.partial": (
        "mismatch: residuals.partial[18][0]: 0.79432 != 0.79834"
    ),
    "coxph_diagnostics/lung_age_strata_sex/zph.km_terms.table": (
        "mismatch: zph[age].chisq: 1.8506 != 0.17958"
    ),
    "coxph_diagnostics/lung_age_strata_sex/zph.km_terms.x": "mismatch: zph.x[109]: 0.93179 != 0",
    "coxph_diagnostics/lung_age_strata_sex/detail.wtrisk": (
        "mismatch: detail.wtrisk[0]: 442.89 != 145.32"
    ),
    "coxph_diagnostics/ovarian_age_rx/zph.km_terms.table": (
        "mismatch: zph[age].chisq: 0.084233 != 0.22427"
    ),
    "coxph_diagnostics/ovarian_age_rx/detail.wtrisk": (
        "mismatch: detail.wtrisk[0]: 1.0454e+05 != 89.002"
    ),
    "coxph_diagnostics/ovarian_age_rx_exact/zph.km_terms.table": (
        "mismatch: zph[age].chisq: 0.084233 != 0.22427"
    ),
    "coxph_diagnostics/pbc_trial_age_edema_strata_sex/residuals.martingale": (
        "mismatch: residuals.martingale[22]: 0.10142 != 0.13678"
    ),
    "coxph_diagnostics/pbc_trial_age_edema_strata_sex/residuals.deviance": (
        "mismatch: residuals.deviance[22]: 0.10506 != 0.14356"
    ),
    "coxph_diagnostics/pbc_trial_age_edema_strata_sex/residuals.schoenfeld": (
        "mismatch: residuals.schoenfeld[0][0]: 10.659 != 5.6303"
    ),
    "coxph_diagnostics/pbc_trial_age_edema_strata_sex/residuals.scaledsch": (
        "mismatch: residuals.scaledsch[0][0]: 0.13549 != 0.083277"
    ),
    "coxph_diagnostics/pbc_trial_age_edema_strata_sex/residuals.partial": (
        "mismatch: residuals.partial[22][0]: 0.31365 != 0.34901"
    ),
    "coxph_diagnostics/pbc_trial_age_edema_strata_sex/zph.km_terms.table": (
        "mismatch: zph[age].chisq: 0.65184 != 0.8026"
    ),
    "coxph_diagnostics/pbc_trial_age_edema_strata_sex/zph.km_terms.x": (
        "mismatch: zph.x[0]: 0 != 0.022436"
    ),
    "coxph_diagnostics/pbc_trial_age_edema_strata_sex/zph.km_terms.y": (
        "mismatch: zph.y[0][0]: 0.13549 != 0.083277"
    ),
    "coxph_diagnostics/pbc_trial_age_edema_strata_sex/detail.time": (
        "mismatch: detail.time[0]: 41 != 140"
    ),
    "coxph_diagnostics/pbc_trial_age_edema_strata_sex/detail.nevent": (
        "mismatch: detail.nevent[13]: 2 != 1"
    ),
    "coxph_diagnostics/pbc_trial_age_edema_strata_sex/detail.nrisk": (
        "mismatch: detail.nrisk[0]: 276 != 36"
    ),
    "coxph_diagnostics/pbc_trial_age_edema_strata_sex/detail.hazard": (
        "mismatch: detail.hazard[0]: 0.0011781 != 0.010902"
    ),
    "coxph_diagnostics/pbc_trial_age_edema_strata_sex/detail.varhaz": (
        "mismatch: detail.varhaz[0]: 1.3879e-06 != 0.00011885"
    ),
    "coxph_diagnostics/pbc_trial_age_edema_strata_sex/detail.wtrisk": (
        "mismatch: detail.wtrisk[0]: 10320 != 91.726"
    ),
    "coxph_diagnostics/pbc_trial_age_edema_strata_sex/detail.score": (
        "mismatch: detail.score[0][0]: 10.659 != 5.6303"
    ),
    "coxph_diagnostics/pbc_trial_age_edema_strata_sex/detail.means": (
        "mismatch: detail.means[0][0]: 55.225 != 63.747"
    ),
    "coxph_diagnostics/pbc_trial_age_edema_strata_sex/detail.imat": (
        "mismatch: detail.imat[0][0][0]: 94.839 != 104.37"
    ),
    "coxph_diagnostics/synthetic_delayed_x/residuals.martingale": (
        "mismatch: residuals.martingale[3]: 0.38673 != 0.48532"
    ),
    "coxph_diagnostics/synthetic_delayed_x/residuals.deviance": (
        "mismatch: residuals.deviance[3]: 0.45215 != 0.59815"
    ),
    "coxph_diagnostics/synthetic_delayed_x/residuals.partial": (
        "mismatch: residuals.partial[3][0]: 0.18231 != 0.2809"
    ),
    "coxph_diagnostics/synthetic_delayed_x/zph.km_terms.table": (
        "mismatch: zph[x].chisq: 0.70486 != 2.3044"
    ),
    "coxph_diagnostics/synthetic_delayed_x/zph.km_terms.y": (
        "mismatch: zph.y[4][0]: 1.2542 != -2.7088"
    ),
    "coxph_diagnostics/synthetic_delayed_x/zph.km_noterms.table": (
        "mismatch: zph[x].chisq: 0.70486 != 2.3044"
    ),
    "coxph_diagnostics/synthetic_delayed_x/zph.km_noterms.y": (
        "mismatch: zph.y[4][0]: 1.2542 != -2.7088"
    ),
    "coxph_diagnostics/synthetic_delayed_x/zph.rank_terms.table": (
        "mismatch: zph[x].chisq: 0.76444 != 2.2433"
    ),
    "coxph_diagnostics/synthetic_delayed_x/zph.rank_terms.x": "mismatch: zph.x[1]: 2 != 3",
    "coxph_diagnostics/synthetic_delayed_x/zph.rank_terms.y": (
        "mismatch: zph.y[4][0]: 1.2542 != -2.7088"
    ),
    "coxph_diagnostics/synthetic_delayed_x/zph.rank_noterms.table": (
        "mismatch: zph[x].chisq: 0.76444 != 2.2433"
    ),
    "coxph_diagnostics/synthetic_delayed_x/zph.rank_noterms.x": "mismatch: zph.x[1]: 2 != 3",
    "coxph_diagnostics/synthetic_delayed_x/zph.rank_noterms.y": (
        "mismatch: zph.y[4][0]: 1.2542 != -2.7088"
    ),
    "coxph_diagnostics/synthetic_delayed_x/zph.identity_terms.table": (
        "mismatch: zph[x].chisq: 0.79906 != 2.1392"
    ),
    "coxph_diagnostics/synthetic_delayed_x/zph.identity_terms.y": (
        "mismatch: zph.y[4][0]: 1.2542 != -2.7088"
    ),
    "coxph_diagnostics/synthetic_delayed_x/zph.identity_noterms.table": (
        "mismatch: zph[x].chisq: 0.79906 != 2.1392"
    ),
    "coxph_diagnostics/synthetic_delayed_x/zph.identity_noterms.y": (
        "mismatch: zph.y[4][0]: 1.2542 != -2.7088"
    ),
    "coxph_diagnostics/synthetic_delayed_x/detail.wtrisk": (
        "mismatch: detail.wtrisk[0]: 5.7596 != 7.2119"
    ),
    "coxph_diagnostics/synthetic_ties_breslow/zph.km_terms.table": (
        "mismatch: zph[x].chisq: 1.5369 != 1.9273"
    ),
    "coxph_diagnostics/synthetic_ties_breslow/zph.km_noterms.table": (
        "mismatch: zph[x].chisq: 1.5369 != 1.9273"
    ),
    "coxph_diagnostics/synthetic_ties_breslow/zph.rank_terms.table": (
        "mismatch: zph[x].chisq: 1.6578 != 2.2434"
    ),
    "coxph_diagnostics/synthetic_ties_breslow/zph.rank_terms.x": "mismatch: zph.x[0]: 1.5 != 2",
    "coxph_diagnostics/synthetic_ties_breslow/zph.rank_noterms.table": (
        "mismatch: zph[x].chisq: 1.6578 != 2.2434"
    ),
    "coxph_diagnostics/synthetic_ties_breslow/zph.rank_noterms.x": "mismatch: zph.x[0]: 1.5 != 2",
    "coxph_diagnostics/synthetic_ties_breslow/zph.identity_terms.table": (
        "mismatch: zph[x].chisq: 1.5375 != 2.1077"
    ),
    "coxph_diagnostics/synthetic_ties_breslow/zph.identity_noterms.table": (
        "mismatch: zph[x].chisq: 1.5375 != 2.1077"
    ),
    "coxph_diagnostics/synthetic_ties_breslow/detail.wtrisk": (
        "mismatch: detail.wtrisk[0]: 19.328 != 15.221"
    ),
    "coxph_diagnostics/synthetic_ties_efron/residuals.martingale": (
        "mismatch: residuals.martingale[0]: 0.84016 != 0.8819"
    ),
    "coxph_diagnostics/synthetic_ties_efron/residuals.deviance": (
        "mismatch: residuals.deviance[0]: 1.4095 != 1.5839"
    ),
    "coxph_diagnostics/synthetic_ties_efron/residuals.partial": (
        "mismatch: residuals.partial[0][0]: 0.98346 != 1.0252"
    ),
    "coxph_diagnostics/synthetic_ties_efron/zph.km_terms.table": (
        "mismatch: zph[x].chisq: 1.5905 != 1.9868"
    ),
    "coxph_diagnostics/synthetic_ties_efron/zph.km_noterms.table": (
        "mismatch: zph[x].chisq: 1.5905 != 1.9868"
    ),
    "coxph_diagnostics/synthetic_ties_efron/zph.rank_terms.table": (
        "mismatch: zph[x].chisq: 1.7168 != 2.3068"
    ),
    "coxph_diagnostics/synthetic_ties_efron/zph.rank_terms.x": "mismatch: zph.x[0]: 1.5 != 2",
    "coxph_diagnostics/synthetic_ties_efron/zph.rank_noterms.table": (
        "mismatch: zph[x].chisq: 1.7168 != 2.3068"
    ),
    "coxph_diagnostics/synthetic_ties_efron/zph.rank_noterms.x": "mismatch: zph.x[0]: 1.5 != 2",
    "coxph_diagnostics/synthetic_ties_efron/zph.identity_terms.table": (
        "mismatch: zph[x].chisq: 1.5879 != 2.1662"
    ),
    "coxph_diagnostics/synthetic_ties_efron/zph.identity_noterms.table": (
        "mismatch: zph[x].chisq: 1.5879 != 2.1662"
    ),
    "coxph_diagnostics/synthetic_ties_efron/detail.wtrisk": (
        "mismatch: detail.wtrisk[0]: 19.192 != 15.114"
    ),
    "coxph_diagnostics/synthetic_ties_exact/zph.km_terms.table": (
        "mismatch: zph[x].chisq: 0.72339 != 2.0276"
    ),
    "coxph_diagnostics/synthetic_ties_exact/zph.km_terms.y": (
        "mismatch: zph.y[0][0]: 1.5344 != 1.1395"
    ),
    "coxph_diagnostics/synthetic_ties_exact/zph.km_terms.var": (
        "mismatch: zph.var[0][0]: 4.534 != 3.8121"
    ),
    "coxph_diagnostics/veteran_celltype_karno_trt/residuals.martingale": (
        "mismatch: residuals.martingale[5]: 0.87029 != 0.87538"
    ),
    "coxph_diagnostics/veteran_celltype_karno_trt/residuals.deviance": (
        "mismatch: residuals.deviance[5]: 1.5311 != 1.5538"
    ),
    "coxph_diagnostics/veteran_celltype_karno_trt/residuals.partial": (
        "mismatch: residuals.partial[5][0]: 2.0764 != 2.0815"
    ),
    "coxph_diagnostics/veteran_celltype_karno_trt/zph.km_terms.table": (
        "mismatch: zph[karno].chisq: 8.3151 != 12.813"
    ),
    "coxph_diagnostics/veteran_celltype_karno_trt/zph.km_terms.y": (
        "mismatch: zph.y[0][0]: -0.12837 != -0.12943"
    ),
    "coxph_diagnostics/veteran_celltype_karno_trt/zph.km_terms.var": (
        "mismatch: zph.var[0][0]: 0.0034148 != 0.0032472"
    ),
    "coxph_diagnostics/veteran_celltype_karno_trt/detail.wtrisk": (
        "mismatch: detail.wtrisk[0]: 82.16 != 346.73"
    ),
    "coxph_penalized/cgd_frailty_gamma_id": (
        "missing feature: ValueError: unsupported formula term(s): frailty(id)"
    ),
    "coxph_penalized/kidney_frailty_gamma": (
        "missing feature: ValueError: unsupported formula term(s): frailty(id)"
    ),
    "coxph_penalized/kidney_frailty_gamma_theta_fixed": (
        "missing feature: ValueError: unsupported formula term(s): frailty(id, theta = 0.5)"
    ),
    "coxph_penalized/kidney_frailty_gaussian": (
        'missing feature: ValueError: unsupported formula term(s): frailty(id, dist = "gauss")'
    ),
    "coxph_penalized/kidney_frailty_gaussian_df": (
        'missing feature: ValueError: unsupported formula term(s): frailty(id, dist = "gauss", ...'
    ),
    "coxph_penalized/kidney_frailty_t": (
        'missing feature: ValueError: unsupported formula term(s): frailty(id, dist = "t")'
    ),
    "coxph_penalized/lung_pspline_age_df0_aic": (
        "missing feature: ValueError: unsupported formula term(s): pspline(age, df = 0)"
    ),
    "coxph_penalized/lung_pspline_age_df4": (
        "missing feature: ValueError: unsupported formula term(s): pspline(age, df = 4)"
    ),
    "coxph_penalized/lung_pspline_karno_df3_nterm6": (
        "missing feature: ValueError: unsupported formula term(s): pspline(ph.karno, df = 3, nt..."
    ),
    "coxph_penalized/lung_ridge_age_sex_theta1": (
        "missing feature: ValueError: unsupported formula term(s): ridge(age, sex, theta = 1)"
    ),
    "coxph_penalized/lung_ridge_age_sex_theta5_scaled": (
        "missing feature: ValueError: unsupported formula term(s): ridge(age, sex, theta = 5, s..."
    ),
    "coxph_penalized/lung_ridge_df2": (
        "missing feature: ValueError: unsupported formula term(s): ridge(age, sex, ph.karno, df..."
    ),
    "coxph_penalized/rats_frailty_gamma_litter": (
        "missing feature: ValueError: unsupported formula term(s): frailty(litter)"
    ),
    "coxph_penalized/rats_frailty_gaussian_litter": (
        'missing feature: ValueError: unsupported formula term(s): frailty(litter, dist = "gauss")'
    ),
    "coxph_predict/aml_x/basehaz_centered": "mismatch: basehaz.hazard[0]: 0.081247 != 0.050392",
    "coxph_predict/aml_x/survfit:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/aml_x/survfit:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.038252 != 0.040229"
    ),
    "coxph_predict/aml_x/predict.risk": "mismatch: predict.risk.se_fit[11]: 1.2789 != 0.80913",
    "coxph_predict/aml_x/predict.expected": "mismatch: predict.expected.fit[4]: 0.34844 != 0.32374",
    "coxph_predict/aml_x/predict_newdata.risk": (
        "mismatch: predict_newdata.risk.se_fit[1]: 1.2789 != 0.80913"
    ),
    "coxph_predict/aml_x/survfit_newdata:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/aml_x/survfit_newdata:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.038252 != 0.040229"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/basehaz_centered": (
        "mismatch: basehaz.hazard[0]: 0.0073974 != 0.010484"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/basehaz_uncentered": (
        "mismatch: basehaz.hazard[0]: 0.0042859 != 0.010484"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/survfit:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/survfit:curves.surv": (
        "mismatch: curve[0].surv[0]: 0.99572 != 0.98957"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/survfit:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0044013 != 0.010599"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/survfit:curves.cumhaz": (
        "mismatch: curve[0].cumhaz[0]: 0.0042859 != 0.010484"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/survfit:curves.std_chaz": (
        "mismatch: curve[0].std_chaz[0]: 0.0044202 != 0.010599"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/survfit:curves.conf": (
        "mismatch: curve[0].lower[0]: 0.98713 != 0.96923"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/predict.lp": (
        "mismatch: predict.lp.fit[0]: 0 != -0.89455"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/predict.risk": (
        "mismatch: predict.risk.fit[0]: 1 != 0.40879"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/predict.expected": (
        "mismatch: predict.expected.fit[1]: 0.46067 != 0.42974"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/predict.terms": (
        "mismatch: predict.terms.fit[0][0]: 0 != -1.0722"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/predict.lp_uncentered": (
        "mismatch: predict.lp_uncentered[0]: 0 != -0.89455"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/predict_newdata.lp": (
        "mismatch: predict_newdata.lp.fit[0]: 0.89455 != 0"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/predict_newdata.risk": (
        "mismatch: predict_newdata.risk.fit[0]: 2.4462 != 1"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/predict_newdata.terms": (
        "mismatch: predict_newdata.terms.fit[0][0]: 1.0722 != 0"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/predict_newdata.lp_uncentered": (
        "mismatch: predict_newdata.lp_uncentered[0]: 0.89455 != 0"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/survfit_newdata:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/survfit_newdata:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.010449 != 0.010599"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/survfit_newdata:curves.std_chaz": (
        "mismatch: curve[0].std_chaz[0]: 0.010559 != 0.010599"
    ),
    "coxph_predict/cgd_counting_treat_cluster_id/survfit_newdata:curves.conf": (
        "mismatch: curve[0].lower[0]: 0.9693 != 0.96923"
    ),
    "coxph_predict/heart_counting_age_surgery_transplant/basehaz_centered": (
        "mismatch: basehaz.hazard[0]: 0.0090831 != 0.010281"
    ),
    "coxph_predict/heart_counting_age_surgery_transplant/survfit:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/heart_counting_age_surgery_transplant/survfit:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.010185 != 0.01029"
    ),
    "coxph_predict/heart_counting_age_surgery_transplant/predict.risk": (
        "mismatch: predict.risk.se_fit[0]: 0.13022 != 0.16292"
    ),
    "coxph_predict/heart_counting_age_surgery_transplant/predict.expected": (
        "mismatch: predict.expected.fit[1]: 0.14402 != 0.13712"
    ),
    "coxph_predict/heart_counting_age_surgery_transplant/predict_newdata.risk": (
        "mismatch: predict_newdata.risk.se_fit[0]: 0.032369 != 0.033637"
    ),
    "coxph_predict/heart_counting_age_surgery_transplant/survfit_newdata:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/heart_counting_age_surgery_transplant/survfit_newdata:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0094551 != 0.0095455"
    ),
    "coxph_predict/kidney_age_sex_disease/basehaz_centered": (
        "mismatch: basehaz.hazard[0]: 0.009856 != 0.010045"
    ),
    "coxph_predict/kidney_age_sex_disease/survfit:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/kidney_age_sex_disease/survfit:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.010256 != 0.01036"
    ),
    "coxph_predict/kidney_age_sex_disease/predict.risk": (
        "mismatch: predict.risk.se_fit[0]: 0.87185 != 0.51758"
    ),
    "coxph_predict/kidney_age_sex_disease/predict.expected": (
        "mismatch: predict.expected.fit[0]: 0.15946 != 0.14254"
    ),
    "coxph_predict/kidney_age_sex_disease/predict_newdata.risk": (
        "mismatch: predict_newdata.risk.se_fit[0]: 1.5893 != 0.90005"
    ),
    "coxph_predict/kidney_age_sex_disease/survfit_newdata:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/kidney_age_sex_disease/survfit_newdata:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.031776 != 0.032787"
    ),
    "coxph_predict/lung_age_offset/basehaz_uncentered": (
        "mismatch: basehaz.hazard[0]: 0.00024025 != 0.00096917"
    ),
    "coxph_predict/lung_age_offset/survfit:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/lung_age_offset/survfit:curves.surv": (
        "mismatch: curve[0].surv[0]: 0.99905 != 0.99616"
    ),
    "coxph_predict/lung_age_offset/survfit:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.00095383 != 0.0038514"
    ),
    "coxph_predict/lung_age_offset/survfit:curves.cumhaz": (
        "mismatch: curve[0].cumhaz[0]: 0.00095469 != 0.0038511"
    ),
    "coxph_predict/lung_age_offset/survfit:curves.std_chaz": (
        "mismatch: curve[0].std_chaz[0]: 0.00095474 != 0.0038514"
    ),
    "coxph_predict/lung_age_offset/survfit:curves.conf": (
        "mismatch: curve[0].lower[0]: 0.99718 != 0.98867"
    ),
    "coxph_predict/lung_age_offset/predict.risk": (
        "mismatch: predict.risk.se_fit[0]: 0.092437 != 0.099115"
    ),
    "coxph_predict/lung_age_offset/predict.expected": (
        "mismatch: predict.expected.fit[6]: 1.2214 != 1.2116"
    ),
    "coxph_predict/lung_age_offset/predict.lp_uncentered": (
        "mismatch: predict.lp_uncentered[0]: 2.6349 != 1.2402"
    ),
    "coxph_predict/lung_age_offset/predict_newdata.risk": (
        "mismatch: predict_newdata.risk.se_fit[0]: 0.058609 != 0.081921"
    ),
    "coxph_predict/lung_age_offset/predict_newdata.lp_uncentered": (
        "mismatch: predict_newdata.lp_uncentered[0]: 2.1047 != 0.70994"
    ),
    "coxph_predict/lung_age_offset/survfit_newdata:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/lung_age_offset/survfit_newdata:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0019827 != 0.0019866"
    ),
    "coxph_predict/lung_age_sex_breslow/survfit:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/lung_age_sex_breslow/survfit:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0041724 != 0.0041899"
    ),
    "coxph_predict/lung_age_sex_breslow/predict.risk": (
        "mismatch: predict.risk.se_fit[0]: 0.18215 != 0.14921"
    ),
    "coxph_predict/lung_age_sex_breslow/survfit_censor_false:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/lung_age_sex_breslow/survfit_censor_false:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0041724 != 0.0041899"
    ),
    "coxph_predict/lung_age_sex_breslow/survfit_stype2:curves.time_counts": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_breslow/survfit_stype2:curves.surv": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_breslow/survfit_stype2:curves.std_err": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_breslow/survfit_stype2:curves.cumhaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_breslow/survfit_stype2:curves.std_chaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_breslow/survfit_stype2:curves.conf": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_breslow/survfit_ctype2:curves.time_counts": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_breslow/survfit_ctype2:curves.surv": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_breslow/survfit_ctype2:curves.std_err": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_breslow/survfit_ctype2:curves.cumhaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_breslow/survfit_ctype2:curves.std_chaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_breslow/survfit_ctype2:curves.conf": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_breslow/predict_newdata.risk": (
        "mismatch: predict_newdata.risk.se_fit[0]: 0.13432 != 0.13495"
    ),
    "coxph_predict/lung_age_sex_breslow/survfit_newdata:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/lung_age_sex_breslow/survfit_newdata:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0041723 != 0.0041896"
    ),
    "coxph_predict/lung_age_sex_breslow/survfit_newdata_loglog:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/lung_age_sex_breslow/survfit_newdata_loglog:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0041723 != 0.0041896"
    ),
    "coxph_predict/lung_age_sex_cluster_inst/survfit:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/lung_age_sex_cluster_inst/survfit:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0042315 != 0.0042489"
    ),
    "coxph_predict/lung_age_sex_cluster_inst/survfit:curves.std_chaz": (
        "mismatch: curve[0].std_chaz[0]: 0.0042495 != 0.0042489"
    ),
    "coxph_predict/lung_age_sex_cluster_inst/survfit:curves.conf": (
        "mismatch: curve[0].lower[1]: 0.96666 != 0.96666"
    ),
    "coxph_predict/lung_age_sex_cluster_inst/predict.risk": (
        "mismatch: predict.risk.se_fit[0]: 0.13568 != 0.11107"
    ),
    "coxph_predict/lung_age_sex_cluster_inst/predict.expected": (
        "mismatch: predict.expected.fit[6]: 0.56313 != 0.55812"
    ),
    "coxph_predict/lung_age_sex_cluster_inst/predict_newdata.risk": (
        "mismatch: predict_newdata.risk.se_fit[0]: 0.1103 != 0.11128"
    ),
    "coxph_predict/lung_age_sex_cluster_inst/survfit_newdata:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/lung_age_sex_cluster_inst/survfit_newdata:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0041975 != 0.0042011"
    ),
    "coxph_predict/lung_age_sex_cluster_inst/survfit_newdata:curves.std_chaz": (
        "mismatch: curve[0].std_chaz[0]: 0.0042151 != 0.0042011"
    ),
    "coxph_predict/lung_age_sex_cluster_inst/survfit_newdata:curves.conf": (
        "mismatch: curve[0].lower[0]: 0.98764 != 0.98767"
    ),
    "coxph_predict/lung_age_sex_efron/survfit:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/lung_age_sex_efron/survfit:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0041718 != 0.0041893"
    ),
    "coxph_predict/lung_age_sex_efron/predict.risk": (
        "mismatch: predict.risk.se_fit[0]: 0.1823 != 0.14929"
    ),
    "coxph_predict/lung_age_sex_efron/predict.expected": (
        "mismatch: predict.expected.fit[6]: 0.55733 != 0.55247"
    ),
    "coxph_predict/lung_age_sex_efron/predict.survival": (
        "mismatch: predict.survival.fit[6]: 0.57273 != 0.57553"
    ),
    "coxph_predict/lung_age_sex_efron/survfit_censor_false:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/lung_age_sex_efron/survfit_censor_false:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0041718 != 0.0041893"
    ),
    "coxph_predict/lung_age_sex_efron/survfit_stype2:curves.time_counts": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_efron/survfit_stype2:curves.surv": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_efron/survfit_stype2:curves.std_err": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_efron/survfit_stype2:curves.cumhaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_efron/survfit_stype2:curves.std_chaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_efron/survfit_stype2:curves.conf": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_efron/survfit_ctype2:curves.time_counts": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_efron/survfit_ctype2:curves.surv": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_efron/survfit_ctype2:curves.std_err": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_efron/survfit_ctype2:curves.cumhaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_efron/survfit_ctype2:curves.std_chaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_efron/survfit_ctype2:curves.conf": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/lung_age_sex_efron/predict_newdata.risk": (
        "mismatch: predict_newdata.risk.se_fit[0]: 0.1343 != 0.13495"
    ),
    "coxph_predict/lung_age_sex_efron/survfit_newdata:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/lung_age_sex_efron/survfit_newdata:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0041712 != 0.0041885"
    ),
    "coxph_predict/lung_age_sex_efron/survfit_newdata_loglog:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/lung_age_sex_efron/survfit_newdata_loglog:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0041712 != 0.0041885"
    ),
    "coxph_predict/lung_age_sex_nocenter_null/survfit:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/lung_age_sex_nocenter_null/survfit:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0041718 != 0.0041893"
    ),
    "coxph_predict/lung_age_sex_nocenter_null/predict.risk": (
        "mismatch: predict.risk.se_fit[0]: 0.1823 != 0.14929"
    ),
    "coxph_predict/lung_age_sex_nocenter_null/predict.expected": (
        "mismatch: predict.expected.fit[6]: 0.55733 != 0.55247"
    ),
    "coxph_predict/lung_age_sex_nocenter_null/predict_newdata.risk": (
        "mismatch: predict_newdata.risk.se_fit[0]: 0.1343 != 0.13495"
    ),
    "coxph_predict/lung_age_sex_nocenter_null/survfit_newdata:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/lung_age_sex_nocenter_null/survfit_newdata:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0041712 != 0.0041885"
    ),
    "coxph_predict/lung_age_sex_weighted/basehaz_centered": (
        "mismatch: basehaz.hazard[0]: 0.0033937 != 0.0033579"
    ),
    "coxph_predict/lung_age_sex_weighted/survfit:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/lung_age_sex_weighted/survfit:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0033474 != 0.0033591"
    ),
    "coxph_predict/lung_age_sex_weighted/survfit:curves.std_chaz": (
        "mismatch: curve[0].std_chaz[0]: 0.0033586 != 0.0033591"
    ),
    "coxph_predict/lung_age_sex_weighted/survfit:curves.conf": (
        "mismatch: curve[0].lower[1]: 0.97361 != 0.9736"
    ),
    "coxph_predict/lung_age_sex_weighted/predict.risk": (
        "mismatch: predict.risk.se_fit[0]: 0.19764 != 0.16652"
    ),
    "coxph_predict/lung_age_sex_weighted/predict.expected": (
        "mismatch: predict.expected.fit[6]: 0.5603 != 0.55547"
    ),
    "coxph_predict/lung_age_sex_weighted/predict_newdata.risk": (
        "mismatch: predict_newdata.risk.se_fit[0]: 0.15861 != 0.14947"
    ),
    "coxph_predict/lung_age_sex_weighted/survfit_newdata:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/lung_age_sex_weighted/survfit_newdata:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0037929 != 0.0038185"
    ),
    "coxph_predict/lung_age_sex_weighted/survfit_newdata:curves.std_chaz": (
        "mismatch: curve[0].std_chaz[0]: 0.0038073 != 0.0038185"
    ),
    "coxph_predict/lung_age_sex_weighted/survfit_newdata:curves.conf": (
        "mismatch: curve[0].lower[0]: 0.98882 != 0.9888"
    ),
    "coxph_predict/lung_age_strata_sex/basehaz_centered": (
        "mismatch: basehaz strata labels differ: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,..."
    ),
    "coxph_predict/lung_age_strata_sex/basehaz_uncentered": (
        "mismatch: basehaz strata labels differ: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,..."
    ),
    "coxph_predict/lung_age_strata_sex/survfit:curves.time_counts": (
        "mismatch: survfit.time: 1 python curves, R has 2"
    ),
    "coxph_predict/lung_age_strata_sex/survfit:curves.surv": (
        "mismatch: curve[0].surv: length 184 differs from expected 116"
    ),
    "coxph_predict/lung_age_strata_sex/survfit:curves.std_err": (
        "mismatch: curve[0].std_err: length 184 differs from expected 116"
    ),
    "coxph_predict/lung_age_strata_sex/survfit:curves.cumhaz": (
        "mismatch: curve[0].cumhaz: length 184 differs from expected 116"
    ),
    "coxph_predict/lung_age_strata_sex/survfit:curves.std_chaz": (
        "mismatch: curve[0].std_chaz: length 184 differs from expected 116"
    ),
    "coxph_predict/lung_age_strata_sex/survfit:curves.conf": (
        "mismatch: curve[0].lower: length 184 differs from expected 116"
    ),
    "coxph_predict/lung_age_strata_sex/predict.risk": (
        "mismatch: predict.risk.se_fit[0]: 0.11271 != 0.10508"
    ),
    "coxph_predict/lung_age_strata_sex/predict.expected": (
        "mismatch: predict.expected.fit[18]: 0.14648 != 0.14246"
    ),
    "coxph_predict/lung_age_strata_sex/predict_newdata.risk": (
        "mismatch: predict_newdata.risk.se_fit[0]: 0.084682 != 0.11328"
    ),
    "coxph_predict/lung_age_strata_sex/survfit_newdata:curves.time_counts": (
        "mismatch: survfit.time: 1 python curves, R has 2"
    ),
    "coxph_predict/lung_age_strata_sex/survfit_newdata:curves.surv": (
        "mismatch: curve[0].surv: length 184 differs from expected 116"
    ),
    "coxph_predict/lung_age_strata_sex/survfit_newdata:curves.std_err": (
        "mismatch: curve[0].std_err: length 184 differs from expected 116"
    ),
    "coxph_predict/lung_age_strata_sex/survfit_newdata:curves.cumhaz": (
        "mismatch: curve[0].cumhaz: length 184 differs from expected 116"
    ),
    "coxph_predict/lung_age_strata_sex/survfit_newdata:curves.std_chaz": (
        "mismatch: curve[0].std_chaz: length 184 differs from expected 116"
    ),
    "coxph_predict/lung_age_strata_sex/survfit_newdata:curves.conf": (
        "mismatch: curve[0].lower: length 184 differs from expected 116"
    ),
    "coxph_predict/lung_factor_ph_ecog/basehaz_centered": (
        "mismatch: basehaz.hazard[0]: 0.0039837 != 0.0026555"
    ),
    "coxph_predict/lung_factor_ph_ecog/basehaz_uncentered": (
        "mismatch: basehaz.hazard[0]: 0.0042591 != 0.0028286"
    ),
    "coxph_predict/lung_factor_ph_ecog/survfit:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/lung_factor_ph_ecog/survfit:curves.surv": (
        "mismatch: curve[0].surv[0]: 0.99601 != 0.99735"
    ),
    "coxph_predict/lung_factor_ph_ecog/survfit:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0039981 != 0.0026899"
    ),
    "coxph_predict/lung_factor_ph_ecog/survfit:curves.cumhaz": (
        "mismatch: curve[0].cumhaz[0]: 0.0039985 != 0.0026555"
    ),
    "coxph_predict/lung_factor_ph_ecog/survfit:curves.std_chaz": (
        "mismatch: curve[0].std_chaz[0]: 0.0040141 != 0.0026899"
    ),
    "coxph_predict/lung_factor_ph_ecog/survfit:curves.conf": (
        "mismatch: curve[0].lower[0]: 0.9882 != 0.9921"
    ),
    "coxph_predict/lung_factor_ph_ecog/predict.lp": (
        "mismatch: predict.lp.fit[0]: 0.34637 != 0.75565"
    ),
    "coxph_predict/lung_factor_ph_ecog/predict.risk": (
        "mismatch: predict.risk.fit[0]: 1.4139 != 2.129"
    ),
    "coxph_predict/lung_factor_ph_ecog/predict.expected": (
        "mismatch: predict.expected.fit[6]: 0.86654 != 0.85853"
    ),
    "coxph_predict/lung_factor_ph_ecog/predict.terms": (
        "mismatch: predict.terms.fit[0][2]: 0 != 0.40928"
    ),
    "coxph_predict/lung_factor_ph_ecog/predict.lp_uncentered": (
        "mismatch: predict.lp_uncentered[0]: 0.28322 != 0.69251"
    ),
    "coxph_predict/lung_factor_ph_ecog/predict_newdata.lp": (
        "mismatch: predict_newdata.lp.fit[0]: -0.33088 != 0.078404"
    ),
    "coxph_predict/lung_factor_ph_ecog/predict_newdata.risk": (
        "mismatch: predict_newdata.risk.fit[0]: 0.71829 != 1.0816"
    ),
    "coxph_predict/lung_factor_ph_ecog/predict_newdata.terms": (
        "mismatch: predict_newdata.terms.fit[0][2]: -0.40928 != 0"
    ),
    "coxph_predict/lung_factor_ph_ecog/predict_newdata.lp_uncentered": (
        "mismatch: predict_newdata.lp_uncentered[0]: -0.39402 != 0.01526"
    ),
    "coxph_predict/lung_factor_ph_ecog/survfit_newdata:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/lung_factor_ph_ecog/survfit_newdata:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0029195 != 0.0029279"
    ),
    "coxph_predict/ovarian_age_rx/survfit:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/ovarian_age_rx/survfit:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.013157 != 0.013305"
    ),
    "coxph_predict/ovarian_age_rx/predict.risk": (
        "mismatch: predict.risk.se_fit[0]: 12.326 != 3.0645"
    ),
    "coxph_predict/ovarian_age_rx/predict_newdata.risk": (
        "mismatch: predict_newdata.risk.se_fit[0]: 0.27604 != 0.35557"
    ),
    "coxph_predict/ovarian_age_rx/survfit_newdata:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/ovarian_age_rx/survfit_newdata:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0091882 != 0.0092507"
    ),
    "coxph_predict/pbc_trial_age_edema_strata_sex/basehaz_centered": (
        "mismatch: basehaz.time[0]: 41 != 140"
    ),
    "coxph_predict/pbc_trial_age_edema_strata_sex/basehaz_uncentered": (
        "mismatch: basehaz.time[0]: 41 != 140"
    ),
    "coxph_predict/pbc_trial_age_edema_strata_sex/survfit:curves.time_counts": (
        "mismatch: survfit.time: 1 python curves, R has 2"
    ),
    "coxph_predict/pbc_trial_age_edema_strata_sex/survfit:curves.surv": (
        "mismatch: curve[0].surv: length 301 differs from expected 36"
    ),
    "coxph_predict/pbc_trial_age_edema_strata_sex/survfit:curves.std_err": (
        "mismatch: curve[0].std_err: length 301 differs from expected 36"
    ),
    "coxph_predict/pbc_trial_age_edema_strata_sex/survfit:curves.cumhaz": (
        "mismatch: curve[0].cumhaz: length 301 differs from expected 36"
    ),
    "coxph_predict/pbc_trial_age_edema_strata_sex/survfit:curves.std_chaz": (
        "mismatch: curve[0].std_chaz: length 301 differs from expected 36"
    ),
    "coxph_predict/pbc_trial_age_edema_strata_sex/survfit:curves.conf": (
        "mismatch: curve[0].lower: length 301 differs from expected 36"
    ),
    "coxph_predict/pbc_trial_age_edema_strata_sex/predict.risk": (
        "mismatch: predict.risk.se_fit[0]: 11.539 != 1.8463"
    ),
    "coxph_predict/pbc_trial_age_edema_strata_sex/predict.expected": (
        "mismatch: predict.expected.fit[22]: 0.89858 != 0.86322"
    ),
    "coxph_predict/pbc_trial_age_edema_strata_sex/predict_newdata.risk": (
        "mismatch: predict_newdata.risk.se_fit[0]: 0.032229 != 0.063122"
    ),
    "coxph_predict/pbc_trial_age_edema_strata_sex/survfit_newdata:curves.time_counts": (
        "mismatch: survfit.time: 1 python curves, R has 2"
    ),
    "coxph_predict/pbc_trial_age_edema_strata_sex/survfit_newdata:curves.surv": (
        "mismatch: curve[0].surv: length 301 differs from expected 36"
    ),
    "coxph_predict/pbc_trial_age_edema_strata_sex/survfit_newdata:curves.std_err": (
        "mismatch: curve[0].std_err: length 301 differs from expected 36"
    ),
    "coxph_predict/pbc_trial_age_edema_strata_sex/survfit_newdata:curves.cumhaz": (
        "mismatch: curve[0].cumhaz: length 301 differs from expected 36"
    ),
    "coxph_predict/pbc_trial_age_edema_strata_sex/survfit_newdata:curves.std_chaz": (
        "mismatch: curve[0].std_chaz: length 301 differs from expected 36"
    ),
    "coxph_predict/pbc_trial_age_edema_strata_sex/survfit_newdata:curves.conf": (
        "mismatch: curve[0].lower: length 301 differs from expected 36"
    ),
    "coxph_predict/synthetic_delayed_x/survfit:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/synthetic_delayed_x/survfit:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.12652 != 0.14533"
    ),
    "coxph_predict/synthetic_delayed_x/predict.risk": (
        "mismatch: predict.risk.se_fit[0]: 0.30229 != 0.36708"
    ),
    "coxph_predict/synthetic_delayed_x/predict.expected": (
        "mismatch: predict.expected.fit[3]: 0.61327 != 0.51468"
    ),
    "coxph_predict/synthetic_delayed_x/predict.survival": (
        "mismatch: predict.survival.fit[3]: 0.54158 != 0.59769"
    ),
    "coxph_predict/synthetic_delayed_x/survfit_censor_false:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/synthetic_delayed_x/survfit_censor_false:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.12652 != 0.14533"
    ),
    "coxph_predict/synthetic_delayed_x/survfit_stype2:curves.time_counts": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_delayed_x/survfit_stype2:curves.surv": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_delayed_x/survfit_stype2:curves.std_err": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_delayed_x/survfit_stype2:curves.cumhaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_delayed_x/survfit_stype2:curves.std_chaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_delayed_x/survfit_stype2:curves.conf": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_delayed_x/survfit_ctype2:curves.time_counts": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_delayed_x/survfit_ctype2:curves.surv": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_delayed_x/survfit_ctype2:curves.std_err": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_delayed_x/survfit_ctype2:curves.cumhaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_delayed_x/survfit_ctype2:curves.std_chaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_delayed_x/survfit_ctype2:curves.conf": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_delayed_x/predict_newdata.risk": (
        "mismatch: predict_newdata.risk.se_fit[0]: 0.32314 != 0.28878"
    ),
    "coxph_predict/synthetic_delayed_x/survfit_newdata:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/synthetic_delayed_x/survfit_newdata:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.14618 != 0.17389"
    ),
    "coxph_predict/synthetic_delayed_x/survfit_newdata_loglog:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/synthetic_delayed_x/survfit_newdata_loglog:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.14618 != 0.17389"
    ),
    "coxph_predict/synthetic_ties_breslow/basehaz_centered": (
        "mismatch: basehaz.hazard[0]: 0.11254 != 0.1314"
    ),
    "coxph_predict/synthetic_ties_breslow/survfit:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/synthetic_ties_breslow/survfit:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.089684 != 0.10228"
    ),
    "coxph_predict/synthetic_ties_breslow/predict.risk": (
        "mismatch: predict.risk.se_fit[0]: 0.11665 != 0.10858"
    ),
    "coxph_predict/synthetic_ties_breslow/survfit_censor_false:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/synthetic_ties_breslow/survfit_censor_false:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.089684 != 0.10228"
    ),
    "coxph_predict/synthetic_ties_breslow/survfit_stype2:curves.time_counts": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_breslow/survfit_stype2:curves.surv": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_breslow/survfit_stype2:curves.std_err": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_breslow/survfit_stype2:curves.cumhaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_breslow/survfit_stype2:curves.std_chaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_breslow/survfit_stype2:curves.conf": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_breslow/survfit_ctype2:curves.time_counts": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_breslow/survfit_ctype2:curves.surv": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_breslow/survfit_ctype2:curves.std_err": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_breslow/survfit_ctype2:curves.cumhaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_breslow/survfit_ctype2:curves.std_chaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_breslow/survfit_ctype2:curves.conf": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_breslow/predict_newdata.risk": (
        "mismatch: predict_newdata.risk.se_fit[0]: 0.13265 != 0.14948"
    ),
    "coxph_predict/synthetic_ties_breslow/survfit_newdata:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/synthetic_ties_breslow/survfit_newdata:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.075482 != 0.083711"
    ),
    "coxph_predict/synthetic_ties_breslow/survfit_newdata_loglog:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/synthetic_ties_breslow/survfit_newdata_loglog:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.075482 != 0.083711"
    ),
    "coxph_predict/synthetic_ties_efron/basehaz_centered": (
        "mismatch: basehaz.hazard[0]: 0.11786 != 0.1385"
    ),
    "coxph_predict/synthetic_ties_efron/survfit:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/synthetic_ties_efron/survfit:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.093751 != 0.10768"
    ),
    "coxph_predict/synthetic_ties_efron/predict.risk": (
        "mismatch: predict.risk.se_fit[0]: 0.11572 != 0.10772"
    ),
    "coxph_predict/synthetic_ties_efron/predict.expected": (
        "mismatch: predict.expected.fit[0]: 0.15984 != 0.1181"
    ),
    "coxph_predict/synthetic_ties_efron/predict.survival": (
        "mismatch: predict.survival.fit[0]: 0.85228 != 0.88861"
    ),
    "coxph_predict/synthetic_ties_efron/survfit_censor_false:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/synthetic_ties_efron/survfit_censor_false:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.093751 != 0.10768"
    ),
    "coxph_predict/synthetic_ties_efron/survfit_stype2:curves.time_counts": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_efron/survfit_stype2:curves.surv": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_efron/survfit_stype2:curves.std_err": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_efron/survfit_stype2:curves.cumhaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_efron/survfit_stype2:curves.std_chaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_efron/survfit_stype2:curves.conf": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_efron/survfit_ctype2:curves.time_counts": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_efron/survfit_ctype2:curves.surv": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_efron/survfit_ctype2:curves.std_err": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_efron/survfit_ctype2:curves.cumhaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_efron/survfit_ctype2:curves.std_chaz": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_efron/survfit_ctype2:curves.conf": (
        "missing feature: ValueError: non-Kaplan-Meier survfit styles are only supported for Su..."
    ),
    "coxph_predict/synthetic_ties_efron/predict_newdata.risk": (
        "mismatch: predict_newdata.risk.se_fit[0]: 0.13161 != 0.1483"
    ),
    "coxph_predict/synthetic_ties_efron/survfit_newdata:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/synthetic_ties_efron/survfit_newdata:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.078769 != 0.087847"
    ),
    "coxph_predict/synthetic_ties_efron/survfit_newdata_loglog:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/synthetic_ties_efron/survfit_newdata_loglog:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.078769 != 0.087847"
    ),
    "coxph_predict/veteran_celltype_karno_trt/basehaz_centered": (
        "mismatch: basehaz.hazard[0]: 0.01049 != 0.0057904"
    ),
    "coxph_predict/veteran_celltype_karno_trt/survfit:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/veteran_celltype_karno_trt/survfit:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0042661 != 0.0042909"
    ),
    "coxph_predict/veteran_celltype_karno_trt/predict.risk": (
        "mismatch: predict.risk.se_fit[0]: 0.084221 != 0.091906"
    ),
    "coxph_predict/veteran_celltype_karno_trt/predict.expected": (
        "mismatch: predict.expected.fit[5]: 0.12971 != 0.12462"
    ),
    "coxph_predict/veteran_celltype_karno_trt/predict_newdata.risk": (
        "mismatch: predict_newdata.risk.se_fit[0]: 0.49115 != 0.35481"
    ),
    "coxph_predict/veteran_celltype_karno_trt/survfit_newdata:curves.time_counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "coxph_predict/veteran_celltype_karno_trt/survfit_newdata:curves.std_err": (
        "mismatch: curve[0].std_err[0]: 0.0079853 != 0.0080744"
    ),
    "pseudo/aml_1/pseudo_rmst": "mismatch: pseudo_rmst[3][1]: 17.165 != 17.224",
    "pseudo/aml_1/pseudo_auc": "mismatch: pseudo_auc[3][1]: 17.165 != 17.224",
    "pseudo/aml_1/pseudo_sojourn": "mismatch: pseudo_sojourn[3][1]: 17.165 != 17.224",
    "pseudo/aml_1/survfit0:curves.std_err": "mismatch: curve[1].std_err[1]: 0.058753 != 0.064349",
    "pseudo/aml_1_single_time/pseudo_rmst": "mismatch: pseudo_rmst[3]: 17.165 != 17.224",
    "pseudo/aml_1_single_time/pseudo_auc": "mismatch: pseudo_auc[3]: 17.165 != 17.224",
    "pseudo/aml_1_single_time/pseudo_sojourn": "mismatch: pseudo_sojourn[3]: 17.165 != 17.224",
    "pseudo/aml_1_single_time/survfit0:curves.std_err": (
        "mismatch: curve[1].std_err[1]: 0.058753 != 0.064349"
    ),
    "pseudo/aml_x/pseudo_pstate": "mismatch: pseudo_pstate: length 2 differs from expected 23",
    "pseudo/aml_x/pseudo_cumhaz": "mismatch: pseudo_cumhaz: length 2 differs from expected 23",
    "pseudo/aml_x/pseudo_rmst": "mismatch: pseudo_rmst: length 2 differs from expected 23",
    "pseudo/aml_x/pseudo_auc": "mismatch: pseudo_auc: length 2 differs from expected 23",
    "pseudo/aml_x/pseudo_sojourn": "mismatch: pseudo_sojourn: length 2 differs from expected 23",
    "pseudo/aml_x/pseudo_survival": "mismatch: pseudo_survival: length 2 differs from expected 23",
    "pseudo/aml_x/survfit0:curves.std_err": (
        "mismatch: curve[x=Maintained].std_err[1]: 0.086678 != 0.095346"
    ),
    "pseudo/aml_x/survfit0:curves.conf": (
        "mismatch: curve[x=Nonmaintained].lower[10]: expected NA/NaN, got 0"
    ),
    "pseudo/cgd_counting_id/survfit0:curves.std_err": (
        "mismatch: curve[1].std_err[2]: 0.010919 != 0.010876"
    ),
    "pseudo/cgd_counting_id/survfit0:curves.std_chaz": (
        "mismatch: curve[1].std_chaz[1]: 0.0078125 != 0.0077819"
    ),
    "pseudo/cgd_counting_id/survfit0:curves.conf": (
        "mismatch: curve[1].lower[2]: 0.96327 != 0.96335"
    ),
    "pseudo/lung_1/pseudo_rmst": "mismatch: pseudo_rmst[0][1]: 302.23 != 302.21",
    "pseudo/lung_1/pseudo_auc": "mismatch: pseudo_auc[0][1]: 302.23 != 302.21",
    "pseudo/lung_1/pseudo_sojourn": "mismatch: pseudo_sojourn[0][1]: 302.23 != 302.21",
    "pseudo/lung_1/survfit0:curves.std_err": (
        "mismatch: curve[1].std_err[1]: 0.0043763 != 0.0043956"
    ),
    "pseudo/lung_sex/pseudo_pstate": "mismatch: pseudo_pstate: length 2 differs from expected 228",
    "pseudo/lung_sex/pseudo_cumhaz": "mismatch: pseudo_cumhaz: length 2 differs from expected 228",
    "pseudo/lung_sex/pseudo_rmst": "mismatch: pseudo_rmst: length 2 differs from expected 228",
    "pseudo/lung_sex/pseudo_auc": "mismatch: pseudo_auc: length 2 differs from expected 228",
    "pseudo/lung_sex/pseudo_sojourn": (
        "mismatch: pseudo_sojourn: length 2 differs from expected 228"
    ),
    "pseudo/lung_sex/pseudo_survival": (
        "mismatch: pseudo_survival: length 2 differs from expected 228"
    ),
    "pseudo/lung_sex/survfit0:curves.std_err": (
        "mismatch: curve[sex=1].std_err[1]: 0.012414 != 0.01269"
    ),
    "pseudo/mgus2_150_mstate/pseudo_pstate": "error: KeyError: 0",
    "pseudo/mgus2_150_mstate/pseudo_cumhaz": "error: KeyError: 0",
    "pseudo/mgus2_150_mstate/residuals_cumhaz": "error: IndexError: list index out of range",
    "pseudo/mgus2_150_mstate/pseudo_sojourn": "error: KeyError: 0",
    "pseudo/synthetic_delayed/survfit0:curves.std_err": (
        "mismatch: curve[1].std_err[1]: 0.15215 != 0.18257"
    ),
    "pseudo/synthetic_delayed/survfit0:curves.conf": (
        "mismatch: curve[1].lower[9]: expected NA/NaN, got 0"
    ),
    "pseudo/synthetic_istate/pseudo_pstate": "error: KeyError: 0",
    "pseudo/synthetic_istate/pseudo_cumhaz": "error: KeyError: 0",
    "pseudo/synthetic_istate/residuals_cumhaz": (
        "mismatch: residuals_cumhaz[0][state 0]: length 3 differs from expected 5"
    ),
    "pseudo/synthetic_istate/pseudo_rmst": "error: KeyError: 0",
    "pseudo/synthetic_istate/residuals_rmst": (
        "mismatch: residuals_rmst[0][state 0][0]: 0.10938 != 0.046875"
    ),
    "pseudo/synthetic_istate/pseudo_auc": "error: KeyError: 0",
    "pseudo/synthetic_istate/residuals_auc": (
        "mismatch: residuals_auc[0][state 0][0]: 0.10938 != 0.046875"
    ),
    "pseudo/synthetic_istate/pseudo_sojourn": "error: KeyError: 0",
    "pseudo/synthetic_istate/residuals_sojourn": (
        "mismatch: residuals_sojourn[0][state 0][0]: 0.10938 != 0.046875"
    ),
    "pseudo/synthetic_istate/pseudo_survival": "error: KeyError: 0",
    "pseudo/synthetic_istate/survfit0:curves.time_counts": (
        "mismatch: curve[1].time: length 10 differs from expected 9"
    ),
    "pseudo/synthetic_istate/survfit0:curves.pstate": (
        "mismatch: curve[1].pstate: length 10 differs from expected 9"
    ),
    "pseudo/synthetic_istate/survfit0:curves.std_err": (
        "mismatch: curve[1].std_err: length 10 differs from expected 9"
    ),
    "pseudo/synthetic_istate/survfit0:curves.cumhaz": (
        "mismatch: curve[1].cumhaz: length 10 differs from expected 9"
    ),
    "pseudo/synthetic_istate/survfit0:curves.std_chaz": (
        "mismatch: curve[1].std_chaz: length 10 differs from expected 9"
    ),
    "pseudo/synthetic_istate/survfit0:curves.conf": (
        "mismatch: curve[1].lower: length 10 differs from expected 9"
    ),
    "pseudo/synthetic_istate/survfit0:curves.n_transition": (
        "mismatch: curve[1].n_transition: length 10 differs from expected 9"
    ),
    "pseudo/synthetic_ties_g/pseudo_pstate": (
        "mismatch: pseudo_pstate: length 2 differs from expected 16"
    ),
    "pseudo/synthetic_ties_g/pseudo_cumhaz": (
        "mismatch: pseudo_cumhaz: length 2 differs from expected 16"
    ),
    "pseudo/synthetic_ties_g/pseudo_rmst": (
        "mismatch: pseudo_rmst: length 2 differs from expected 16"
    ),
    "pseudo/synthetic_ties_g/pseudo_auc": "mismatch: pseudo_auc: length 2 differs from expected 16",
    "pseudo/synthetic_ties_g/pseudo_sojourn": (
        "mismatch: pseudo_sojourn: length 2 differs from expected 16"
    ),
    "pseudo/synthetic_ties_g/pseudo_survival": (
        "mismatch: pseudo_survival: length 2 differs from expected 16"
    ),
    "pseudo/synthetic_ties_g/survfit0:curves.std_err": (
        "mismatch: curve[g=a].std_err[1]: 0.11693 != 0.13363"
    ),
    "pseudo/synthetic_ties_g/survfit0:curves.conf": (
        "mismatch: curve[g=a].lower[7]: expected NA/NaN, got 0"
    ),
    "pseudo/synthetic_ties_mstate/pseudo_pstate": "error: KeyError: 0",
    "pseudo/synthetic_ties_mstate/pseudo_cumhaz": "error: KeyError: 0",
    "pseudo/synthetic_ties_mstate/residuals_cumhaz": "error: IndexError: list index out of range",
    "pseudo/synthetic_ties_mstate/pseudo_rmst": "error: KeyError: 0",
    "pseudo/synthetic_ties_mstate/pseudo_auc": "error: KeyError: 0",
    "pseudo/synthetic_ties_mstate/pseudo_sojourn": "error: KeyError: 0",
    "pseudo/synthetic_ties_mstate/pseudo_survival": "error: KeyError: 0",
    "pyears/hearta_age_surgery": (
        "missing feature: pyears formula with tcut()/cut() terms is not available"
    ),
    "pyears/hearta_age_surgery_data_frame": (
        "missing feature: pyears formula with tcut()/cut() terms is not available"
    ),
    "pyears/hearta_no_event": "missing feature: ValueError: formula response must be Surv(...)",
    "pyears/hearta_surgery_scale365/offtable": (
        "missing feature: result has none of the attributes ('off_table',)"
    ),
    "pyears/lung_tcut_age_sex_no_ratetable": (
        "missing feature: pyears formula with tcut()/cut() terms is not available"
    ),
    "pyears/lung_tcut_age_survexp_us": (
        "missing feature: pyears with a ratetable/rmap is not available"
    ),
    "pyears/lung_tcut_age_year_survexp_us": (
        "missing feature: pyears with a ratetable/rmap is not available"
    ),
    "pyears/lung_tcut_weighted": (
        "missing feature: pyears formula with tcut()/cut() terms is not available"
    ),
    "survSplit/aml_zero_cut": (
        "missing feature: TypeError: survSplit response must be a Surv object"
    ),
    "survSplit/cgd_counting_cut": (
        "missing feature: TypeError: survSplit response must be a Surv object"
    ),
    "survSplit/lung_50_cut": "missing feature: TypeError: survSplit response must be a Surv object",
    "survSplit/lung_50_cut_start_end_event": (
        "missing feature: TypeError: survSplit response must be a Surv object"
    ),
    "survSplit/synthetic_delayed_cut": (
        "missing feature: TypeError: survSplit response must be a Surv object"
    ),
    "survSplit/synthetic_mstate_cut": (
        "missing feature: TypeError: survSplit response must be a Surv object"
    ),
    "survcheck/cgd_counting/states": "missing feature: survcheck result has no states",
    "survcheck/cgd_counting/transitions": (
        "mismatch: transitions {'0 -> 1': 44, '1 -> 1': 32} != {'0 -> 0': 32}"
    ),
    "survcheck/cgd_counting/events": "missing feature: survcheck result has no events",
    "survcheck/cgd_counting/istate": "missing feature: survcheck result has no istate",
    "survcheck/heart_counting/states": "missing feature: survcheck result has no states",
    "survcheck/heart_counting/transitions": (
        "mismatch: transitions {'0 -> 1': 75} != {'0 -> 0': 75}"
    ),
    "survcheck/heart_counting/events": "missing feature: survcheck result has no events",
    "survcheck/heart_counting/istate": "missing feature: survcheck result has no istate",
    "survcheck/myeloid_ms_1/states": "missing feature: survcheck result has no states",
    "survcheck/myeloid_ms_1/events": "missing feature: survcheck result has no events",
    "survcheck/myeloid_ms_1/istate": "missing feature: survcheck result has no istate",
    "survcheck/myeloid_ms_trt/states": "missing feature: survcheck result has no states",
    "survcheck/myeloid_ms_trt/events": "missing feature: survcheck result has no events",
    "survcheck/myeloid_ms_trt/istate": "missing feature: survcheck result has no istate",
    "survcheck/synthetic_istate/states": "missing feature: survcheck result has no states",
    "survcheck/synthetic_istate/events": "missing feature: survcheck result has no events",
    "survcheck/synthetic_istate/istate": "missing feature: survcheck result has no istate",
    "survcheck/synthetic_overlap_gaps": "error: ValueError: start[10] must be less than stop[10]",
    "survcheck/synthetic_overlap_no_istate": (
        "error: ValueError: start[10] must be less than stop[10]"
    ),
    "survcondense/cgd_split_treat_age": (
        "mismatch: frame lacks column '1' (has ['treat', 'age', 'id', 'tstart', 'tstop',..."
    ),
    "survcondense/lung_split_age_sex": (
        "mismatch: frame lacks column '1' (has ['age', 'sex', 'subject', 'tstart', 'tsto..."
    ),
    "survcondense/lung_split_age_sex_epi": (
        "mismatch: frame lacks column '1' (has ['age', 'sex', 'epi', 'subject', 'tstart'..."
    ),
    "survcondense/lung_split_start_end": (
        "mismatch: frame lacks column '1' (has ['age', 'subject', 't1', 't2', 'died'])"
    ),
    "survdiff/colon_rx_death/counts": "mismatch: obs[0]: 161 != 168",
    "survdiff/colon_rx_death/var": "mismatch: var[0][0]: 98.79 != 99.579",
    "survdiff/kidney_disease/counts": "mismatch: obs[0]: 18 != 20",
    "survdiff/kidney_disease/var": "mismatch: var[0][0]: 10.167 != 13.167",
    "survdiff/veteran_celltype/counts": "mismatch: obs[0]: 26 != 31",
    "survdiff/veteran_celltype/var": "mismatch: var[0][0]: 12.966 != 26.338",
    "survdiff/veteran_celltype_rho1/counts": "mismatch: obs[0]: 16.066 != 13.39",
    "survdiff/veteran_celltype_rho1/var": "mismatch: var[0][0]: 6.5465 != 9.0471",
    "survexp/lung_conditional_us": "mismatch: surv[1]: 0.98783 != 0.99213",
    "survexp/lung_conditional_us_by_sex": (
        "missing feature: survexp grouped by a formula term is not available"
    ),
    "survexp/lung_coxph_ratetable": (
        "missing feature: survexp with a coxph fit as ratetable is not available"
    ),
    "survexp/lung_ederer_mn": (
        "missing feature: only survexp.us is supported by the direct survexp interface (survexp..."
    ),
    "survexp/lung_ederer_us": "mismatch: surv[1]: 0.98724 != 0.99155",
    "survexp/lung_ederer_us_by_sex": (
        "missing feature: survexp grouped by a formula term is not available"
    ),
    "survexp/lung_ederer_us_default_times": "mismatch: surv[0]: 0.99965 != 0.99977",
    "survexp/lung_ederer_usr": (
        "missing feature: only survexp.us is supported by the direct survexp interface (survexp..."
    ),
    "survexp/lung_hakulinen_us": "mismatch: surv[1]: 0.98789 != 0.99214",
    "survexp/lung_hakulinen_us_by_sex": (
        "missing feature: survexp grouped by a formula term is not available"
    ),
    "survexp/lung_individual_us": "mismatch: surv[0]: 0.95164 != 0.96031",
    "survexp/survexp_us_table": (
        "missing feature: result has none of the attributes ('dim', 'shape')"
    ),
    "survfit_interval/interval2_synthetic/time_surv": "mismatch: time[0]: 1 != 1.5",
    "survfit_interval/interval2_synthetic/counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "survfit_interval/interval2_synthetic/std_err": (
        "missing feature: result has none of the attributes ('std_err',)"
    ),
    "survfit_interval/interval2_synthetic/conf": "mismatch: lower[0]: 1 != 0.6072",
    "survfit_interval/interval2_synthetic/n": (
        "missing feature: result has none of the attributes ('n',)"
    ),
    "survfit_interval/interval2_synthetic_group": (
        "missing feature: result has none of the attributes ('time',)"
    ),
    "survfit_interval/interval_status_synthetic/time_surv": (
        "mismatch: time: length 6 differs from expected 7"
    ),
    "survfit_interval/interval_status_synthetic/counts": (
        "missing feature: result has none of the attributes ('n_risk',)"
    ),
    "survfit_interval/interval_status_synthetic/std_err": (
        "missing feature: result has none of the attributes ('std_err',)"
    ),
    "survfit_interval/interval_status_synthetic/conf": (
        "mismatch: lower: length 6 differs from expected 7"
    ),
    "survfit_interval/interval_status_synthetic/n": (
        "missing feature: result has none of the attributes ('n',)"
    ),
    "survfit_km/aml_fh2_ties/curves.std_err": (
        "mismatch: curve[x=Maintained].std_err[0]: 0.083009 != 0.090909"
    ),
    "survfit_km/aml_fh2_ties/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/aml_fh2_ties/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/aml_fh2_ties/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/aml_fh_ties/curves.std_err": (
        "mismatch: curve[x=Maintained].std_err[0]: 0.083009 != 0.090909"
    ),
    "survfit_km/aml_fh_ties/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/aml_fh_ties/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/aml_fh_ties/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/aml_x/curves.std_err": (
        "mismatch: curve[x=Maintained].std_err[0]: 0.086678 != 0.095346"
    ),
    "survfit_km/aml_x/curves.conf": (
        "mismatch: curve[x=Nonmaintained].lower[9]: expected NA/NaN, got 0"
    ),
    "survfit_km/aml_x/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/aml_x/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/aml_x/summary_std_err": "mismatch: summary_std_err[15]: expected NA/NaN, got 0",
    "survfit_km/aml_x/summary_times": "missing feature: no summary(fit, times=) equivalent",
    "survfit_km/aml_x/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/bladder_rx_enum1/curves.std_err": (
        "mismatch: curve[rx=1].std_err[0]: 0.021049 != 0.021507"
    ),
    "survfit_km/bladder_rx_enum1/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/bladder_rx_enum1/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/bladder_rx_enum1/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/cgd_counting_cluster/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/cgd_counting_cluster/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/cgd_counting_cluster/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/cgd_counting_treat_id/curves.std_err": (
        "mismatch: curve[treat=placebo].std_err[1]: 0.021257 != 0.02109"
    ),
    "survfit_km/cgd_counting_treat_id/curves.std_chaz": (
        "mismatch: curve[treat=placebo].std_chaz[0]: 0.015385 != 0.015266"
    ),
    "survfit_km/cgd_counting_treat_id/curves.conf": (
        "mismatch: curve[treat=placebo].lower[1]: 0.92869 != 0.929"
    ),
    "survfit_km/cgd_counting_treat_id/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/cgd_counting_treat_id/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/cgd_counting_treat_id/summary_std_err": (
        "mismatch: summary_std_err[1]: 0.021257 != 0.02109"
    ),
    "survfit_km/cgd_counting_treat_id/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/cgd_counting_treat_noid/curves.std_err": (
        "mismatch: curve[treat=placebo].std_err[0]: 0.015266 != 0.015504"
    ),
    "survfit_km/cgd_counting_treat_noid/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/cgd_counting_treat_noid/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/cgd_counting_treat_noid/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/colon_rx_death/curves.std_err": (
        "mismatch: curve[rx=Obs].std_err[0]: 0.0031696 != 0.0031797"
    ),
    "survfit_km/colon_rx_death/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/colon_rx_death/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/colon_rx_death/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/flchain_sex_500/curves.std_err": (
        "mismatch: curve[sex=F].std_err[0]: 0.0051239 != 0.0051614"
    ),
    "survfit_km/flchain_sex_500/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/flchain_sex_500/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/flchain_sex_500/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/heart_counting_transplant_id/curves.std_err": (
        "mismatch: curve[transplant=0].std_err[0]: 0.0096615 != 0.0097562"
    ),
    "survfit_km/heart_counting_transplant_id/fit.n": (
        "missing feature: survfit result does not report n"
    ),
    "survfit_km/heart_counting_transplant_id/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/heart_counting_transplant_id/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/jasa_surgery/curves.std_err": (
        "mismatch: curve[surgery=0].std_err[0]: 0.016067 != 0.016445"
    ),
    "survfit_km/jasa_surgery/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/jasa_surgery/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/jasa_surgery/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/kidney_sex/curves.std_err": (
        "mismatch: curve[sex=1].std_err[0]: 0.048734 != 0.051299"
    ),
    "survfit_km/kidney_sex/curves.conf": "mismatch: curve[sex=1].lower[17]: expected NA/NaN, got 0",
    "survfit_km/kidney_sex/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/kidney_sex/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/kidney_sex/summary_std_err": (
        "mismatch: summary_std_err[16]: expected NA/NaN, got 0"
    ),
    "survfit_km/kidney_sex/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_1/curves.std_err": "mismatch: curve[1].std_err[0]: 0.0043763 != 0.0043956",
    "survfit_km/lung_1/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_1/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_1/summary_times": "missing feature: no summary(fit, times=) equivalent",
    "survfit_km/lung_1/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_conf_arcsin/curves.std_err": (
        "mismatch: curve[1].std_err[0]: 0.0043763 != 0.0043956"
    ),
    "survfit_km/lung_conf_arcsin/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_conf_arcsin/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_conf_arcsin/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_conf_int_90/curves.std_err": (
        "mismatch: curve[sex=1].std_err[0]: 0.012414 != 0.01269"
    ),
    "survfit_km/lung_conf_int_90/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_conf_int_90/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_conf_int_90/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_conf_log/curves.std_err": (
        "mismatch: curve[1].std_err[0]: 0.0043763 != 0.0043956"
    ),
    "survfit_km/lung_conf_log/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_conf_log/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_conf_log/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_conf_logit/curves.std_err": (
        "mismatch: curve[1].std_err[0]: 0.0043763 != 0.0043956"
    ),
    "survfit_km/lung_conf_logit/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_conf_logit/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_conf_logit/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_conf_loglog/curves.std_err": (
        "mismatch: curve[1].std_err[0]: 0.0043763 != 0.0043956"
    ),
    "survfit_km/lung_conf_loglog/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_conf_loglog/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_conf_loglog/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_conf_lower_modified": "missing feature: survfit has no conf.lower argument",
    "survfit_km/lung_conf_lower_peto": "missing feature: survfit has no conf.lower argument",
    "survfit_km/lung_conf_none/curves.std_err": (
        "mismatch: curve[1].std_err[0]: 0.0043763 != 0.0043956"
    ),
    "survfit_km/lung_conf_none/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_conf_none/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_conf_plain/curves.std_err": (
        "mismatch: curve[1].std_err[0]: 0.0043763 != 0.0043956"
    ),
    "survfit_km/lung_conf_plain/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_conf_plain/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_conf_plain/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_reverse/curves.std_err": (
        "mismatch: curve[sex=1].std_err[33]: 0.010471 != 0.010582"
    ),
    "survfit_km/lung_reverse/curves.conf": (
        "mismatch: curve[sex=1].lower[118]: expected NA/NaN, got 0"
    ),
    "survfit_km/lung_reverse/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_reverse/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_reverse/summary_std_err": (
        "mismatch: summary_std_err[24]: expected NA/NaN, got 0"
    ),
    "survfit_km/lung_reverse/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_reverse_1/curves.std_err": (
        "mismatch: curve[1].std_err[19]: 0.0049627 != 0.0049875"
    ),
    "survfit_km/lung_reverse_1/curves.conf": (
        "mismatch: curve[1].lower[185]: expected NA/NaN, got 0"
    ),
    "survfit_km/lung_reverse_1/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_reverse_1/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_reverse_1/summary_std_err": (
        "mismatch: summary_std_err[59]: expected NA/NaN, got 0"
    ),
    "survfit_km/lung_reverse_1/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_se_fit_false/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_se_fit_false/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_se_fit_false/summary_std_err": (
        "error: ValueError: zip() argument 2 is longer than argument 1"
    ),
    "survfit_km/lung_sex/curves.std_err": "mismatch: curve[sex=1].std_err[0]: 0.012414 != 0.01269",
    "survfit_km/lung_sex/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_sex/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_sex/summary_times": "missing feature: no summary(fit, times=) equivalent",
    "survfit_km/lung_sex/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_sex_ph_ecog/curves.std_err": (
        "mismatch: curve[sex=1, ph.ecog=0].std_err[0]: 0.027389 != 0.028172"
    ),
    "survfit_km/lung_sex_ph_ecog/curves.conf": (
        "mismatch: curve[sex=1, ph.ecog=2].lower[26]: expected NA/NaN, got 0"
    ),
    "survfit_km/lung_sex_ph_ecog/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_sex_ph_ecog/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_sex_ph_ecog/summary_std_err": (
        "mismatch: summary_std_err[102]: expected NA/NaN, got 0"
    ),
    "survfit_km/lung_start_time_100/curves.std_err": (
        "mismatch: curve[sex=1].std_err[0]: 0.0087334 != 0.0088107"
    ),
    "survfit_km/lung_start_time_100/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_start_time_100/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_start_time_100/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_stype1_ctype2/curves.std_err": (
        "mismatch: curve[1].std_err[0]: 0.0043763 != 0.0043956"
    ),
    "survfit_km/lung_stype1_ctype2/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_stype1_ctype2/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_stype1_ctype2/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_stype2_ctype1/curves.std_err": (
        "mismatch: curve[1].std_err[0]: 0.0043668 != 0.004386"
    ),
    "survfit_km/lung_stype2_ctype1/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_stype2_ctype1/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_stype2_ctype1/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_stype2_ctype2/curves.std_err": (
        "mismatch: curve[1].std_err[0]: 0.0043668 != 0.004386"
    ),
    "survfit_km/lung_stype2_ctype2/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_stype2_ctype2/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_stype2_ctype2/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_time0/curves.time_counts": (
        "mismatch: curve[sex=1].time: length 120 differs from expected 119"
    ),
    "survfit_km/lung_time0/curves.surv": (
        "mismatch: curve[sex=1].surv: length 120 differs from expected 119"
    ),
    "survfit_km/lung_time0/curves.std_err": (
        "mismatch: curve[sex=1].std_err: length 120 differs from expected 119"
    ),
    "survfit_km/lung_time0/curves.cumhaz": (
        "mismatch: curve[sex=1].cumhaz: length 120 differs from expected 119"
    ),
    "survfit_km/lung_time0/curves.std_chaz": (
        "mismatch: curve[sex=1].std_chaz: length 120 differs from expected 119"
    ),
    "survfit_km/lung_time0/curves.conf": (
        "mismatch: curve[sex=1].lower: length 120 differs from expected 119"
    ),
    "survfit_km/lung_time0/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_time0/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_time0/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_type_fh/curves.std_err": (
        "mismatch: curve[sex=1].std_err[0]: 0.01237 != 0.012643"
    ),
    "survfit_km/lung_type_fh/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_type_fh/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_type_fh/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_type_fh2/curves.std_err": (
        "mismatch: curve[sex=1].std_err[0]: 0.01237 != 0.012643"
    ),
    "survfit_km/lung_type_fh2/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_type_fh2/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_type_fh2/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_weighted/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_weighted/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_weighted/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/lung_weighted_1/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/lung_weighted_1/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/lung_weighted_1/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/mgus2_sex/curves.std_err": (
        "mismatch: curve[sex=F].std_err[0]: 0.0069742 != 0.0072024"
    ),
    "survfit_km/mgus2_sex/curves.conf": "mismatch: curve[sex=M].lower[231]: expected NA/NaN, got 0",
    "survfit_km/mgus2_sex/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/mgus2_sex/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/mgus2_sex/summary_std_err": (
        "mismatch: summary_std_err[360]: expected NA/NaN, got 0"
    ),
    "survfit_km/mgus2_sex/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/myeloid_trt/curves.std_err": (
        "mismatch: curve[trt=A].std_err[2]: 0.0031796 != 0.0031898"
    ),
    "survfit_km/myeloid_trt/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/myeloid_trt/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/myeloid_trt/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/nafld1_male_1000/curves.std_err": (
        "mismatch: curve[male=0].std_err[0]: 0.0017841 != 0.0017873"
    ),
    "survfit_km/nafld1_male_1000/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/nafld1_male_1000/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/nafld1_male_1000/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/ovarian_rx/curves.std_err": (
        "mismatch: curve[rx=1].std_err[0]: 0.073905 != 0.080064"
    ),
    "survfit_km/ovarian_rx/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/ovarian_rx/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/ovarian_rx/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/pbc_trt/curves.std_err": (
        "mismatch: curve[trt=1].std_err[0]: 0.0063091 != 0.0063492"
    ),
    "survfit_km/pbc_trt/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/pbc_trt/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/pbc_trt/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/rats_rx/curves.std_err": "mismatch: curve[rx=0].std_err[2]: 0.0050377 != 0.0050633",
    "survfit_km/rats_rx/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/rats_rx/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/rats_rx/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/synthetic_delayed_entry/curves.std_err": (
        "mismatch: curve[1].std_err[0]: 0.15215 != 0.18257"
    ),
    "survfit_km/synthetic_delayed_entry/curves.conf": (
        "mismatch: curve[1].lower[8]: expected NA/NaN, got 0"
    ),
    "survfit_km/synthetic_delayed_entry/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/synthetic_delayed_entry/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/synthetic_delayed_entry/summary_std_err": (
        "mismatch: summary_std_err[6]: expected NA/NaN, got 0"
    ),
    "survfit_km/synthetic_delayed_entry/summary_times": (
        "missing feature: no summary(fit, times=) equivalent"
    ),
    "survfit_km/synthetic_delayed_entry/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/synthetic_ties/curves.std_err": (
        "mismatch: curve[g=a].std_err[0]: 0.11693 != 0.13363"
    ),
    "survfit_km/synthetic_ties/curves.conf": (
        "mismatch: curve[g=a].lower[6]: expected NA/NaN, got 0"
    ),
    "survfit_km/synthetic_ties/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/synthetic_ties/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/synthetic_ties/summary_std_err": (
        "mismatch: summary_std_err[5]: expected NA/NaN, got 0"
    ),
    "survfit_km/synthetic_ties/summary_times": (
        "missing feature: no summary(fit, times=) equivalent"
    ),
    "survfit_km/synthetic_ties/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/synthetic_ties_1/curves.std_err": (
        "mismatch: curve[1].std_err[0]: 0.08268 != 0.094491"
    ),
    "survfit_km/synthetic_ties_1/curves.conf": (
        "mismatch: curve[1].lower[8]: expected NA/NaN, got 0"
    ),
    "survfit_km/synthetic_ties_1/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/synthetic_ties_1/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/synthetic_ties_1/summary_std_err": (
        "mismatch: summary_std_err[7]: expected NA/NaN, got 0"
    ),
    "survfit_km/synthetic_ties_1/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/synthetic_timefix_false/curves.time_counts": (
        "mismatch: curve[1].time: length 6 differs from expected 8"
    ),
    "survfit_km/synthetic_timefix_false/curves.surv": (
        "mismatch: curve[1].surv: length 6 differs from expected 8"
    ),
    "survfit_km/synthetic_timefix_false/curves.std_err": (
        "mismatch: curve[1].std_err: length 6 differs from expected 8"
    ),
    "survfit_km/synthetic_timefix_false/curves.cumhaz": (
        "mismatch: curve[1].cumhaz: length 6 differs from expected 8"
    ),
    "survfit_km/synthetic_timefix_false/curves.std_chaz": (
        "mismatch: curve[1].std_chaz: length 6 differs from expected 8"
    ),
    "survfit_km/synthetic_timefix_false/curves.conf": (
        "mismatch: curve[1].lower: length 6 differs from expected 8"
    ),
    "survfit_km/synthetic_timefix_false/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/synthetic_timefix_false/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/synthetic_timefix_false/summary_std_err": (
        "mismatch: summary_std_err: length 5 differs from expected 6"
    ),
    "survfit_km/synthetic_timefix_false/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/synthetic_timefix_true/curves.std_err": (
        "mismatch: curve[1].std_err[0]: 0.15309 != 0.20412"
    ),
    "survfit_km/synthetic_timefix_true/curves.conf": (
        "mismatch: curve[1].lower[4]: expected NA/NaN, got 0"
    ),
    "survfit_km/synthetic_timefix_true/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/synthetic_timefix_true/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/synthetic_timefix_true/summary_std_err": (
        "mismatch: summary_std_err[3]: expected NA/NaN, got 0"
    ),
    "survfit_km/synthetic_timefix_true/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/transplant_abo_ltx/curves.std_err": (
        "mismatch: curve[abo=A].std_err[0]: 0.0080526 != 0.0082299"
    ),
    "survfit_km/transplant_abo_ltx/curves.conf": (
        "mismatch: curve[abo=B].lower[89]: expected NA/NaN, got 0"
    ),
    "survfit_km/transplant_abo_ltx/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/transplant_abo_ltx/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/transplant_abo_ltx/summary_std_err": (
        "mismatch: summary_std_err[228]: expected NA/NaN, got 0"
    ),
    "survfit_km/transplant_abo_ltx/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_km/veteran_celltype/curves.std_err": (
        "mismatch: curve[celltype=squamous].std_err[0]: 0.039235 != 0.041613"
    ),
    "survfit_km/veteran_celltype/curves.conf": (
        "mismatch: curve[celltype=squamous].lower[32]: expected NA/NaN, got 0"
    ),
    "survfit_km/veteran_celltype/fit.n": "missing feature: survfit result does not report n",
    "survfit_km/veteran_celltype/summary_table": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for summary_table"
    ),
    "survfit_km/veteran_celltype/summary_std_err": (
        "mismatch: summary_std_err[29]: expected NA/NaN, got 0"
    ),
    "survfit_km/veteran_celltype/quantile": (
        "missing feature: no summary.survfit/quantile.survfit equivalent for quantile"
    ),
    "survfit_multistate/mgus2_400_1/summary_times": (
        "missing feature: no summary(fit, times=) equivalent"
    ),
    "survfit_multistate/mgus2_sex/summary_times": (
        "missing feature: no summary(fit, times=) equivalent"
    ),
    "survfit_multistate/myeloid_ms_trt/summary_times": (
        "missing feature: no summary(fit, times=) equivalent"
    ),
    "survfit_multistate/synthetic_istate/influence_pstate": (
        "missing feature: survfit result has no influence_state"
    ),
    "survfit_multistate/synthetic_istate_x/curves.std_err": (
        "mismatch: curve[x=0].std_err[0][0]: 0.27217 != 0.18426"
    ),
    "survfit_multistate/synthetic_istate_x/curves.conf": (
        "mismatch: curve[x=0].lower[0][0]: 0.067278 != 0.11281"
    ),
    "survfit_multistate/synthetic_ties_influence/influence_pstate": (
        "missing feature: survfit result has no influence_state"
    ),
    "survfit_multistate/transplant_abo/summary_times": (
        "missing feature: no summary(fit, times=) equivalent"
    ),
    "survreg/interval2_synthetic_lognormal_g/coef": "mismatch: coef[0]: 1.3219 != 1.0957",
    "survreg/interval2_synthetic_lognormal_g/coef_names": "mismatch: coef[0]: 1.3219 != 1.0957",
    "survreg/interval2_synthetic_lognormal_g/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/interval2_synthetic_lognormal_g/scale": "mismatch: scale[0]: 0.41632 != 0.57263",
    "survreg/interval2_synthetic_lognormal_g/var": "mismatch: var[0][0]: 0.048476 != 0.075561",
    "survreg/interval2_synthetic_lognormal_g/loglik": (
        "mismatch: loglik[0]: got nan, expected -13.942"
    ),
    "survreg/interval2_synthetic_lognormal_g/iter": "mismatch: iter: 5 != 3",
    "survreg/interval2_synthetic_lognormal_g/df_residual": "mismatch: df_residual: 3 != 7",
    "survreg/interval2_synthetic_lognormal_g/linear_predictors": (
        "mismatch: linear_predictors: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_lognormal_g/residuals.response": (
        "mismatch: residuals.response: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_lognormal_g/residuals.deviance": (
        "mismatch: residuals.deviance: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_lognormal_g/residuals.dfbeta": (
        "mismatch: residuals.dfbeta: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_lognormal_g/residuals.dfbetas": (
        "mismatch: residuals.dfbetas: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_lognormal_g/residuals.working": (
        "mismatch: residuals.working: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_lognormal_g/residuals.ldcase": (
        "mismatch: residuals.ldcase: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_lognormal_g/residuals.ldresp": (
        "mismatch: residuals.ldresp: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_lognormal_g/residuals.ldshape": (
        "mismatch: residuals.ldshape: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_lognormal_g/residuals.matrix": (
        "mismatch: residuals.matrix: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_lognormal_g/predict.response": (
        "mismatch: predict.response.fit: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_lognormal_g/predict.lp": (
        "mismatch: predict.lp.fit: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_lognormal_g/predict.terms": (
        "mismatch: predict.terms.fit: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_lognormal_g/predict.quantile": (
        "mismatch: predict.quantile.fit: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_lognormal_g/predict.uquantile": (
        "mismatch: predict.uquantile.fit: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_lognormal_g/summary": (
        "mismatch: summary[(Intercept)].value: 1.3219 != 1.0957"
    ),
    "survreg/interval2_synthetic_lognormal_g/anova": (
        "missing feature: TypeError: anova requires fitted Cox model objects"
    ),
    "survreg/interval2_synthetic_lognormal_g/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "survreg/interval2_synthetic_weibull/coef": "mismatch: coef[0]: 1.5263 != 1.5711",
    "survreg/interval2_synthetic_weibull/coef_names": "mismatch: coef[0]: 1.5263 != 1.5711",
    "survreg/interval2_synthetic_weibull/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/interval2_synthetic_weibull/scale": "mismatch: scale[0]: 0.37204 != 0.53481",
    "survreg/interval2_synthetic_weibull/var": "mismatch: var[0][0]: 0.027907 != 0.038118",
    "survreg/interval2_synthetic_weibull/loglik": "mismatch: loglik[0]: got nan, expected -13.97",
    "survreg/interval2_synthetic_weibull/df_residual": "mismatch: df_residual: 4 != 8",
    "survreg/interval2_synthetic_weibull/linear_predictors": (
        "mismatch: linear_predictors: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_weibull/residuals.response": (
        "mismatch: residuals.response: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_weibull/residuals.deviance": (
        "mismatch: residuals.deviance: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_weibull/residuals.dfbeta": (
        "mismatch: residuals.dfbeta: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_weibull/residuals.dfbetas": (
        "mismatch: residuals.dfbetas: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_weibull/residuals.working": (
        "mismatch: residuals.working: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_weibull/residuals.ldcase": (
        "mismatch: residuals.ldcase: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_weibull/residuals.ldresp": (
        "mismatch: residuals.ldresp: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_weibull/residuals.ldshape": (
        "mismatch: residuals.ldshape: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_weibull/residuals.matrix": (
        "mismatch: residuals.matrix: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_weibull/predict.response": (
        "mismatch: predict.response.fit: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_weibull/predict.lp": (
        "mismatch: predict.lp.fit: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_weibull/predict.quantile": (
        "mismatch: predict.quantile.fit: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_weibull/predict.uquantile": (
        "mismatch: predict.uquantile.fit: length 6 differs from expected 10"
    ),
    "survreg/interval2_synthetic_weibull/summary": (
        "mismatch: summary[(Intercept)].value: 1.5263 != 1.5711"
    ),
    "survreg/interval2_synthetic_weibull/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "survreg/interval_status_synthetic_weibull/coef": "mismatch: coef[0]: 1.4798 != 1.5711",
    "survreg/interval_status_synthetic_weibull/coef_names": "mismatch: coef[0]: 1.4798 != 1.5711",
    "survreg/interval_status_synthetic_weibull/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/interval_status_synthetic_weibull/scale": "mismatch: scale[0]: 0.42808 != 0.53481",
    "survreg/interval_status_synthetic_weibull/var": "mismatch: var[0][0]: 0.044857 != 0.038118",
    "survreg/interval_status_synthetic_weibull/loglik": (
        "mismatch: loglik[0]: got nan, expected -13.97"
    ),
    "survreg/interval_status_synthetic_weibull/df_residual": "mismatch: df_residual: 3 != 8",
    "survreg/interval_status_synthetic_weibull/linear_predictors": (
        "mismatch: linear_predictors: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/residuals.response": (
        "mismatch: residuals.response: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/residuals.deviance": (
        "mismatch: residuals.deviance: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/residuals.dfbeta": (
        "mismatch: residuals.dfbeta: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/residuals.dfbetas": (
        "mismatch: residuals.dfbetas: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/residuals.working": (
        "mismatch: residuals.working: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/residuals.ldcase": (
        "mismatch: residuals.ldcase: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/residuals.ldresp": (
        "mismatch: residuals.ldresp: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/residuals.ldshape": (
        "mismatch: residuals.ldshape: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/residuals.matrix": (
        "mismatch: residuals.matrix: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/predict.response": (
        "mismatch: predict.response.fit: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/predict.lp": (
        "mismatch: predict.lp.fit: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/predict.quantile": (
        "mismatch: predict.quantile.fit: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/predict.uquantile": (
        "mismatch: predict.uquantile.fit: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/summary": (
        "mismatch: summary[(Intercept)].value: 1.4798 != 1.5711"
    ),
    "survreg/interval_status_synthetic_weibull/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "survreg/kidney_weibull_age_sex/coef": "mismatch: coef[0]: 1.3074 != 3.3176",
    "survreg/kidney_weibull_age_sex/coef_names": "mismatch: coef[0]: 1.3074 != 3.3176",
    "survreg/kidney_weibull_age_sex/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/kidney_weibull_age_sex/scale": "mismatch: scale[0]: 0.81263 != 1.1033",
    "survreg/kidney_weibull_age_sex/var": "mismatch: var[0][0]: 0.0034372 != 0.60543",
    "survreg/kidney_weibull_age_sex/loglik": "mismatch: loglik[0]: got nan, expected -340.94",
    "survreg/kidney_weibull_age_sex/iter": "mismatch: iter: 30 != 5",
    "survreg/kidney_weibull_age_sex/linear_predictors": (
        "mismatch: linear_predictors[0]: 1.2436 != 4.1701"
    ),
    "survreg/lung_age_sex_exponential/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/lung_age_sex_exponential/loglik": "mismatch: loglik[0]: got nan, expected -1162.3",
    "survreg/lung_age_sex_exponential/iter": "mismatch: iter: 10 != 4",
    "survreg/lung_age_sex_exponential/residuals.response": (
        "mismatch: residuals.response[6]: -212.94 != -212.94"
    ),
    "survreg/lung_age_sex_exponential/residuals.deviance": (
        "mismatch: residuals.deviance[49]: 0.10094 != 0.10094"
    ),
    "survreg/lung_age_sex_exponential/residuals.working": (
        "mismatch: residuals.working[0]: 0.039524 != 0.038022"
    ),
    "survreg/lung_age_sex_exponential/predict.response": (
        "mismatch: predict.response.fit[18]: 630.74 != 630.74"
    ),
    "survreg/lung_age_sex_exponential/predict.terms": (
        "mismatch: predict.terms.fit[0][1]: -0.18984 != -0.18984"
    ),
    "survreg/lung_age_sex_exponential/predict_newdata.response": (
        "mismatch: predict_newdata.response.fit[1]: 506.86 != 506.86"
    ),
    "survreg/lung_age_sex_exponential/predict_newdata.terms": (
        "mismatch: predict_newdata.terms.fit[0][0]: 0.19441 != 0.19441"
    ),
    "survreg/lung_age_sex_exponential/predict_newdata.quantile": (
        "mismatch: predict_newdata.quantile.fit[1][2]: 1167.1 != 1167.1"
    ),
    "survreg/lung_age_sex_exponential/summary": (
        "mismatch: summary[age].value: -0.015619 != -0.015619"
    ),
    "survreg/lung_age_sex_exponential/anova": (
        "missing feature: TypeError: anova requires fitted Cox model objects"
    ),
    "survreg/lung_age_sex_exponential/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "survreg/lung_age_sex_gaussian/coef": "mismatch: coef[0]: 4.3976 != 403.32",
    "survreg/lung_age_sex_gaussian/coef_names": "mismatch: coef[0]: 4.3976 != 403.32",
    "survreg/lung_age_sex_gaussian/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/lung_age_sex_gaussian/scale": "mismatch: scale[0]: 1.5344 != 240.76",
    "survreg/lung_age_sex_gaussian/var": "mismatch: var[0][0]: 0.66839 != 18514",
    "survreg/lung_age_sex_gaussian/loglik": "mismatch: loglik[0]: got nan, expected -1185.9",
    "survreg/lung_age_sex_gaussian/iter": "mismatch: iter: 10 != 3",
    "survreg/lung_age_sex_gaussian/linear_predictors": (
        "mismatch: linear_predictors[0]: -9.2916 != 287.14"
    ),
    "survreg/lung_age_sex_gaussian/residuals.response": (
        "mismatch: residuals.response[0]: 315.29 != 18.86"
    ),
    "survreg/lung_age_sex_gaussian/residuals.deviance": (
        "mismatch: residuals.deviance[0]: 0 != 0.078335"
    ),
    "survreg/lung_age_sex_gaussian/residuals.dfbeta": (
        "mismatch: residuals.dfbeta[0][0]: 0 != -0.77021"
    ),
    "survreg/lung_age_sex_gaussian/residuals.dfbetas": (
        "mismatch: residuals.dfbetas[0][0]: 0 != -0.0056606"
    ),
    "survreg/lung_age_sex_gaussian/residuals.working": "mismatch: residuals.working[0]: 0 != 18.86",
    "survreg/lung_age_sex_gaussian/residuals.ldcase": (
        "mismatch: residuals.ldcase[0]: 0 != 0.0030952"
    ),
    "survreg/lung_age_sex_gaussian/residuals.ldresp": (
        "mismatch: residuals.ldresp[0]: 0 != 0.014948"
    ),
    "survreg/lung_age_sex_gaussian/residuals.ldshape": (
        "mismatch: residuals.ldshape[0]: 0 != 0.00036446"
    ),
    "survreg/lung_age_sex_gaussian/residuals.matrix": (
        "mismatch: residuals.matrix[0][0]: -690 != -6.4058"
    ),
    "survreg/lung_age_sex_gaussian/predict.response": (
        "mismatch: predict.response.fit[0]: -9.2916 != 287.14"
    ),
    "survreg/lung_age_sex_gaussian/predict.lp": "mismatch: predict.lp.fit[0]: -9.2916 != 287.14",
    "survreg/lung_age_sex_gaussian/predict.terms": (
        "mismatch: predict.terms.fit[0][0]: -2.7555 != -35.716"
    ),
    "survreg/lung_age_sex_gaussian/predict.quantile": (
        "mismatch: predict.quantile.fit[0][0]: -11.258 != -21.407"
    ),
    "survreg/lung_age_sex_gaussian/predict.uquantile": (
        "mismatch: predict.uquantile.fit[0][0]: -11.258 != -21.407"
    ),
    "survreg/lung_age_sex_gaussian/predict_newdata.response": (
        "mismatch: predict_newdata.response.fit[0]: -3.5672 != 361.34"
    ),
    "survreg/lung_age_sex_gaussian/predict_newdata.lp": (
        "mismatch: predict_newdata.lp.fit[0]: -3.5672 != 361.34"
    ),
    "survreg/lung_age_sex_gaussian/predict_newdata.terms": (
        "mismatch: predict_newdata.terms.fit[0][0]: 2.9689 != 38.482"
    ),
    "survreg/lung_age_sex_gaussian/predict_newdata.quantile": (
        "mismatch: predict_newdata.quantile.fit[0][0]: -5.5337 != 52.79"
    ),
    "survreg/lung_age_sex_gaussian/predict_newdata.uquantile": (
        "mismatch: predict_newdata.uquantile.fit[0][0]: -5.5337 != 52.79"
    ),
    "survreg/lung_age_sex_gaussian/summary": (
        "mismatch: summary[(Intercept)].value: 4.3976 != 403.32"
    ),
    "survreg/lung_age_sex_gaussian/anova": (
        "missing feature: TypeError: anova requires fitted Cox model objects"
    ),
    "survreg/lung_age_sex_gaussian/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "survreg/lung_age_sex_logistic/coef": "mismatch: coef[0]: -0.014205 != 340.77",
    "survreg/lung_age_sex_logistic/coef_names": "mismatch: coef[0]: -0.014205 != 340.77",
    "survreg/lung_age_sex_logistic/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/lung_age_sex_logistic/scale": "mismatch: scale[0]: 1.0267 != 137.21",
    "survreg/lung_age_sex_logistic/var": "mismatch: var[0][0]: -673.85 != 17740",
    "survreg/lung_age_sex_logistic/loglik": "mismatch: loglik[0]: got nan, expected -1186.9",
    "survreg/lung_age_sex_logistic/iter": "mismatch: iter: 14 != 3",
    "survreg/lung_age_sex_logistic/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.0053443 != 263.41"
    ),
    "survreg/lung_age_sex_logistic/residuals.response": (
        "mismatch: residuals.response[0]: 305.99 != 42.585"
    ),
    "survreg/lung_age_sex_logistic/residuals.deviance": (
        "mismatch: residuals.deviance[0]: 24.358 != 0.21902"
    ),
    "survreg/lung_age_sex_logistic/residuals.working": (
        "mismatch: residuals.working[0]: 1 != 43.272"
    ),
    "survreg/lung_age_sex_logistic/predict.response": (
        "mismatch: predict.response.fit[0]: 0.0053443 != 263.41"
    ),
    "survreg/lung_age_sex_logistic/predict.lp": "mismatch: predict.lp.fit[0]: 0.0053443 != 263.41",
    "survreg/lung_age_sex_logistic/predict.terms": (
        "mismatch: predict.terms.fit[0][0]: 0.0026601 != -31.635"
    ),
    "survreg/lung_age_sex_logistic/predict_newdata.response": (
        "mismatch: predict_newdata.response.fit[0]: -0.00018196 != 329.13"
    ),
    "survreg/lung_age_sex_logistic/predict_newdata.lp": (
        "mismatch: predict_newdata.lp.fit[0]: -0.00018196 != 329.13"
    ),
    "survreg/lung_age_sex_logistic/predict_newdata.terms": (
        "mismatch: predict_newdata.terms.fit[0][0]: -0.0028661 != 34.085"
    ),
    "survreg/lung_age_sex_logistic/predict_newdata.quantile": (
        "mismatch: predict_newdata.quantile.fit[0][0]: -2.256 != 27.649"
    ),
    "survreg/lung_age_sex_logistic/predict_newdata.uquantile": (
        "mismatch: predict_newdata.uquantile.fit[0][0]: -2.256 != 27.649"
    ),
    "survreg/lung_age_sex_logistic/summary": (
        "mismatch: summary[(Intercept)].value: -0.014205 != 340.77"
    ),
    "survreg/lung_age_sex_logistic/anova": (
        "missing feature: TypeError: anova requires fitted Cox model objects"
    ),
    "survreg/lung_age_sex_logistic/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "survreg/lung_age_sex_loglogistic/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/lung_age_sex_loglogistic/loglik": "mismatch: loglik[0]: got nan, expected -1160.9",
    "survreg/lung_age_sex_loglogistic/iter": "mismatch: iter: 9 != 4",
    "survreg/lung_age_sex_loglogistic/residuals.working": (
        "mismatch: residuals.working[0]: 0.30804 != 0.38497"
    ),
    "survreg/lung_age_sex_loglogistic/predict.terms": (
        "mismatch: predict.terms.se_fit[0][0]: 0.089122 != 0.0012482"
    ),
    "survreg/lung_age_sex_loglogistic/predict_newdata.terms": (
        "mismatch: predict_newdata.terms.se_fit[0][0]: 0.096024 != 0.0013448"
    ),
    "survreg/lung_age_sex_loglogistic/anova": (
        "missing feature: TypeError: anova requires fitted Cox model objects"
    ),
    "survreg/lung_age_sex_loglogistic/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "survreg/lung_age_sex_lognormal/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/lung_age_sex_lognormal/loglik": "mismatch: loglik[0]: got nan, expected -1169.3",
    "survreg/lung_age_sex_lognormal/iter": "mismatch: iter: 23 != 3",
    "survreg/lung_age_sex_lognormal/residuals.working": (
        "mismatch: residuals.working[0]: 0.49846 != 0.52472"
    ),
    "survreg/lung_age_sex_lognormal/residuals.ldresp": (
        "mismatch: residuals.ldresp[5]: 0.050524 != 0.050524"
    ),
    "survreg/lung_age_sex_lognormal/predict.terms": (
        "mismatch: predict.terms.se_fit[0][0]: 0.096906 != 0.0022634"
    ),
    "survreg/lung_age_sex_lognormal/predict_newdata.terms": (
        "mismatch: predict_newdata.terms.se_fit[0][0]: 0.10441 != 0.0024387"
    ),
    "survreg/lung_age_sex_lognormal/summary": (
        "mismatch: summary[Log(scale)].value: 0.051335 != 0.051335"
    ),
    "survreg/lung_age_sex_lognormal/anova": (
        "missing feature: TypeError: anova requires fitted Cox model objects"
    ),
    "survreg/lung_age_sex_lognormal/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "survreg/lung_age_sex_rayleigh/coef": "mismatch: coef[0]: -27.845 != 6.2597",
    "survreg/lung_age_sex_rayleigh/coef_names": "mismatch: coef[0]: -27.845 != 6.2597",
    "survreg/lung_age_sex_rayleigh/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/lung_age_sex_rayleigh/var": "mismatch: var[0][0]: 0.017179 != 0.1037",
    "survreg/lung_age_sex_rayleigh/loglik": "mismatch: loglik[0]: got nan, expected -1181.8",
    "survreg/lung_age_sex_rayleigh/iter": "mismatch: iter: 7 != 5",
    "survreg/lung_age_sex_rayleigh/linear_predictors": (
        "mismatch: linear_predictors[0]: 1.9764 != 5.8767"
    ),
    "survreg/lung_age_sex_rayleigh/residuals.response": (
        "mismatch: residuals.response[0]: 298.78 != -50.614"
    ),
    "survreg/lung_age_sex_rayleigh/residuals.deviance": (
        "mismatch: residuals.deviance[0]: 0 != -0.29128"
    ),
    "survreg/lung_age_sex_rayleigh/residuals.working": (
        "mismatch: residuals.working[0]: 0 != -0.17909"
    ),
    "survreg/lung_age_sex_rayleigh/predict.response": (
        "mismatch: predict.response.fit[0]: 7.217 != 356.61"
    ),
    "survreg/lung_age_sex_rayleigh/predict.lp": "mismatch: predict.lp.fit[0]: 1.9764 != 5.8767",
    "survreg/lung_age_sex_rayleigh/predict.terms": (
        "mismatch: predict.terms.fit[0][0]: 4.6325 != -0.10087"
    ),
    "survreg/lung_age_sex_rayleigh/predict_newdata.response": (
        "mismatch: predict_newdata.response.fit[0]: 0.00047734 != 439.75"
    ),
    "survreg/lung_age_sex_rayleigh/predict_newdata.lp": (
        "mismatch: predict_newdata.lp.fit[0]: -7.6473 != 6.0862"
    ),
    "survreg/lung_age_sex_rayleigh/predict_newdata.terms": (
        "mismatch: predict_newdata.terms.fit[0][0]: -4.9912 != 0.10868"
    ),
    "survreg/lung_age_sex_rayleigh/predict_newdata.quantile": (
        "mismatch: predict_newdata.quantile.fit[0][0]: 0.00015494 != 142.74"
    ),
    "survreg/lung_age_sex_rayleigh/predict_newdata.uquantile": (
        "mismatch: predict_newdata.uquantile.fit[0][0]: -8.7725 != 4.961"
    ),
    "survreg/lung_age_sex_rayleigh/summary": (
        "mismatch: summary[(Intercept)].value: -27.845 != 6.2597"
    ),
    "survreg/lung_age_sex_rayleigh/anova": (
        "missing feature: TypeError: anova requires fitted Cox model objects"
    ),
    "survreg/lung_age_sex_rayleigh/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "survreg/lung_age_sex_t/coef": "mismatch: coef[0]: 307.34 != 307.34",
    "survreg/lung_age_sex_t/coef_names": "mismatch: coef[0]: 307.34 != 307.34",
    "survreg/lung_age_sex_t/icoef": "missing feature: result has none of the attributes ('icoef',)",
    "survreg/lung_age_sex_t/scale": "mismatch: scale[0]: 196.25 != 196.25",
    "survreg/lung_age_sex_t/loglik": "mismatch: loglik[0]: got nan, expected -1188",
    "survreg/lung_age_sex_t/iter": "mismatch: iter: 23 != 3",
    "survreg/lung_age_sex_t/linear_predictors": "mismatch: linear_predictors[0]: 253.09 != 253.09",
    "survreg/lung_age_sex_t/residuals.response": "mismatch: residuals.response[0]: 52.91 != 52.91",
    "survreg/lung_age_sex_t/residuals.deviance": (
        "mismatch: residuals.deviance[11]: 2.3902 != 2.3902"
    ),
    "survreg/lung_age_sex_t/residuals.working": "mismatch: residuals.working[0]: 54.871 != 54.869",
    "survreg/lung_age_sex_t/predict.response": (
        "mismatch: predict.response.fit[0]: 253.09 != 253.09"
    ),
    "survreg/lung_age_sex_t/predict.lp": "mismatch: predict.lp.fit[0]: 253.09 != 253.09",
    "survreg/lung_age_sex_t/predict.terms": "mismatch: predict.terms.fit[0][0]: -28.544 != -28.544",
    "survreg/lung_age_sex_t/predict_newdata.terms": (
        "mismatch: predict_newdata.terms.fit[0][0]: 30.755 != 30.755"
    ),
    "survreg/lung_age_sex_t/predict_newdata.quantile": (
        "mismatch: predict_newdata.quantile.fit[0][0]: 11.504 != 11.504"
    ),
    "survreg/lung_age_sex_t/predict_newdata.uquantile": (
        "mismatch: predict_newdata.uquantile.fit[0][0]: 11.504 != 11.504"
    ),
    "survreg/lung_age_sex_t/summary": "mismatch: summary[(Intercept)].value: 307.34 != 307.34",
    "survreg/lung_age_sex_t/anova": (
        "missing feature: TypeError: anova requires fitted Cox model objects"
    ),
    "survreg/lung_age_sex_t/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "survreg/lung_age_sex_t_df8/coef": "mismatch: coef[0]: 115.06 != 350.62",
    "survreg/lung_age_sex_t_df8/coef_names": "mismatch: coef[0]: 115.06 != 350.62",
    "survreg/lung_age_sex_t_df8/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/lung_age_sex_t_df8/scale": "mismatch: scale[0]: 35.69 != 214.57",
    "survreg/lung_age_sex_t_df8/var": "mismatch: var[0][0]: 3674.9 != 17581",
    "survreg/lung_age_sex_t_df8/loglik": "mismatch: loglik[0]: got nan, expected -1186.4",
    "survreg/lung_age_sex_t_df8/iter": "mismatch: iter: 30 != 3",
    "survreg/lung_age_sex_t_df8/linear_predictors": (
        "mismatch: linear_predictors[0]: 57.924 != 265.53"
    ),
    "survreg/lung_age_sex_weibull/coef": "mismatch: coef[0]: 0.65845 != 6.2749",
    "survreg/lung_age_sex_weibull/coef_names": "mismatch: coef[0]: 0.65845 != 6.2749",
    "survreg/lung_age_sex_weibull/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/lung_age_sex_weibull/scale": "mismatch: scale[0]: 0.90352 != 0.75405",
    "survreg/lung_age_sex_weibull/var": "mismatch: var[0][0]: 0.00083865 != 0.23171",
    "survreg/lung_age_sex_weibull/loglik": "mismatch: loglik[0]: got nan, expected -1153.9",
    "survreg/lung_age_sex_weibull/iter": "mismatch: iter: 30 != 5",
    "survreg/lung_age_sex_weibull/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.65955 != 5.7499"
    ),
    "survreg/lung_age_sex_weibull/residuals.response": (
        "mismatch: residuals.response[0]: 304.07 != -8.165"
    ),
    "survreg/lung_age_sex_weibull/residuals.deviance": (
        "mismatch: residuals.deviance[0]: 23.027 != -0.03472"
    ),
    "survreg/lung_age_sex_weibull/residuals.dfbeta": (
        "mismatch: residuals.dfbeta[0][0]: -0.028873 != 0.00029026"
    ),
    "survreg/lung_age_sex_weibull/residuals.dfbetas": (
        "mismatch: residuals.dfbetas[0][0]: -0.99703 != 0.00060299"
    ),
    "survreg/lung_age_sex_weibull/residuals.working": (
        "mismatch: residuals.working[0]: 270.72 != -0.026798"
    ),
    "survreg/lung_age_sex_weibull/residuals.ldcase": (
        "mismatch: residuals.ldcase[0]: 4.3555 != 0.0037504"
    ),
    "survreg/lung_age_sex_weibull/residuals.ldresp": (
        "mismatch: residuals.ldresp[0]: 4.6342 != 0.017443"
    ),
    "survreg/lung_age_sex_weibull/residuals.ldshape": (
        "mismatch: residuals.ldshape[0]: 191.27 != 8.6096e-05"
    ),
    "survreg/lung_age_sex_weibull/residuals.matrix": (
        "mismatch: residuals.matrix[0][0]: -266.02 != -0.71831"
    ),
    "survreg/lung_age_sex_weibull/predict.response": (
        "mismatch: predict.response.fit[0]: 1.9339 != 314.16"
    ),
    "survreg/lung_age_sex_weibull/predict.lp": "mismatch: predict.lp.fit[0]: 0.65955 != 5.7499",
    "survreg/lung_age_sex_weibull/predict.terms": (
        "mismatch: predict.terms.fit[0][0]: -0.00015159 != -0.1416"
    ),
    "survreg/lung_age_sex_weibull/predict.quantile": (
        "mismatch: predict.quantile.fit[0][0]: 0.25317 != 57.571"
    ),
    "survreg/lung_age_sex_weibull/predict.uquantile": (
        "mismatch: predict.uquantile.fit[0][0]: -1.3737 != 4.053"
    ),
    "survreg/lung_age_sex_weibull/predict_newdata.response": (
        "mismatch: predict_newdata.response.fit[0]: 1.9345 != 421.61"
    ),
    "survreg/lung_age_sex_weibull/predict_newdata.lp": (
        "mismatch: predict_newdata.lp.fit[0]: 0.65986 != 6.0441"
    ),
    "survreg/lung_age_sex_weibull/predict_newdata.terms": (
        "mismatch: predict_newdata.terms.fit[0][0]: 0.00016333 != 0.15257"
    ),
    "survreg/lung_age_sex_weibull/predict_newdata.quantile": (
        "mismatch: predict_newdata.quantile.fit[0][0]: 0.25325 != 77.261"
    ),
    "survreg/lung_age_sex_weibull/predict_newdata.uquantile": (
        "mismatch: predict_newdata.uquantile.fit[0][0]: -1.3734 != 4.3472"
    ),
    "survreg/lung_age_sex_weibull/summary": (
        "mismatch: summary[(Intercept)].value: 0.65845 != 6.2749"
    ),
    "survreg/lung_age_sex_weibull/anova": (
        "missing feature: TypeError: anova requires fitted Cox model objects"
    ),
    "survreg/lung_age_sex_weibull/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "survreg/lung_intercept_only_weibull/coef": "mismatch: coef[0]: 0.65847 != 6.0349",
    "survreg/lung_intercept_only_weibull/coef_names": "mismatch: coef[0]: 0.65847 != 6.0349",
    "survreg/lung_intercept_only_weibull/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/lung_intercept_only_weibull/scale": "mismatch: scale[0]: 0.9039 != 0.75939",
    "survreg/lung_intercept_only_weibull/var": "mismatch: var[0][0]: -9.1931e-05 != 0.0034971",
    "survreg/lung_intercept_only_weibull/loglik": "mismatch: loglik[0]: got nan, expected -1153.9",
    "survreg/lung_intercept_only_weibull/iter": "mismatch: iter: 30 != 6",
    "survreg/lung_intercept_only_weibull/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.65847 != 6.0349"
    ),
    "survreg/lung_lognormal_strata_sex/coef": "mismatch: coef[0]: 7.3677 != 7.3677",
    "survreg/lung_lognormal_strata_sex/coef_names": "mismatch: coef[0]: 7.3677 != 7.3677",
    "survreg/lung_lognormal_strata_sex/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/lung_lognormal_strata_sex/scale": "mismatch: scale: length 1 differs from expected 2",
    "survreg/lung_lognormal_strata_sex/loglik": "mismatch: loglik[0]: got nan, expected -1149.2",
    "survreg/lung_lognormal_strata_sex/iter": "mismatch: iter: 23 != 3",
    "survreg/lung_lognormal_strata_sex/linear_predictors": (
        "mismatch: linear_predictors[1]: 5.8935 != 5.8935"
    ),
    "survreg/lung_weibull_cluster_inst/coef": "mismatch: coef[0]: 0.65524 != 6.316",
    "survreg/lung_weibull_cluster_inst/coef_names": "mismatch: coef[0]: 0.65524 != 6.316",
    "survreg/lung_weibull_cluster_inst/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/lung_weibull_cluster_inst/scale": "mismatch: scale[0]: 0.904 != 0.7523",
    "survreg/lung_weibull_cluster_inst/var": "mismatch: var[0][0]: 0.16501 != 0.16692",
    "survreg/lung_weibull_cluster_inst/loglik": "mismatch: loglik[0]: got nan, expected -1134.5",
    "survreg/lung_weibull_cluster_inst/iter": "mismatch: iter: 30 != 5",
    "survreg/lung_weibull_cluster_inst/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.65671 != 5.7594"
    ),
    "survreg/lung_weibull_cluster_inst/naive_var": (
        "mismatch: naive_var[0][0]: 0.00083929 != 0.2338"
    ),
    "survreg/lung_weibull_factor_ph_ecog/coef": "mismatch: coef[0]: 0.65826 != 6.2576",
    "survreg/lung_weibull_factor_ph_ecog/coef_names": (
        "mismatch: coef: names ['(Intercept)', 'age', 'sex', 'factor(ph.ecog)0', 'factor..."
    ),
    "survreg/lung_weibull_factor_ph_ecog/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/lung_weibull_factor_ph_ecog/scale": "mismatch: scale[0]: 0.90412 != 0.7323",
    "survreg/lung_weibull_factor_ph_ecog/var": "mismatch: var[0][0]: 0.00088763 != 0.21433",
    "survreg/lung_weibull_factor_ph_ecog/loglik": "mismatch: loglik[0]: got nan, expected -1134.5",
    "survreg/lung_weibull_factor_ph_ecog/iter": "mismatch: iter: 30 != 5",
    "survreg/lung_weibull_factor_ph_ecog/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.65678 != 5.8036"
    ),
    "survreg/lung_weibull_factor_ph_ecog/residuals.response": (
        "mismatch: residuals.response[0]: 304.07 != -25.497"
    ),
    "survreg/lung_weibull_factor_ph_ecog/residuals.deviance": (
        "mismatch: residuals.deviance[0]: 23.019 != -0.10733"
    ),
    "survreg/lung_weibull_factor_ph_ecog/residuals.working": (
        "mismatch: residuals.working[0]: 270.55 != -0.08457"
    ),
    "survreg/lung_weibull_factor_ph_ecog/predict.response": (
        "mismatch: predict.response.fit[0]: 1.9286 != 331.5"
    ),
    "survreg/lung_weibull_factor_ph_ecog/predict.lp": (
        "mismatch: predict.lp.fit[0]: 0.65678 != 5.8036"
    ),
    "survreg/lung_weibull_factor_ph_ecog/predict.terms": (
        "mismatch: predict.terms.fit[0][0]: -0.00050052 != -0.087419"
    ),
    "survreg/lung_weibull_factor_ph_ecog/predict_newdata.response": (
        "mismatch: predict_newdata.response.fit[0]: 1.9223 != 531.32"
    ),
    "survreg/lung_weibull_factor_ph_ecog/predict_newdata.lp": (
        "mismatch: predict_newdata.lp.fit[0]: 0.65355 != 6.2754"
    ),
    "survreg/lung_weibull_factor_ph_ecog/predict_newdata.terms": (
        "mismatch: predict_newdata.terms.fit[0][0]: 0.00053782 != 0.093934"
    ),
    "survreg/lung_weibull_factor_ph_ecog/predict_newdata.quantile": (
        "mismatch: predict_newdata.quantile.fit[0][0]: 0.25131 != 102.25"
    ),
    "survreg/lung_weibull_factor_ph_ecog/predict_newdata.uquantile": (
        "mismatch: predict_newdata.uquantile.fit[0][0]: -1.381 != 4.6274"
    ),
    "survreg/lung_weibull_factor_ph_ecog/summary": (
        "mismatch: summary[(Intercept)].value: 0.65826 != 6.2576"
    ),
    "survreg/lung_weibull_factor_ph_ecog/anova": (
        "missing feature: TypeError: anova requires fitted Cox model objects"
    ),
    "survreg/lung_weibull_factor_ph_ecog/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "survreg/lung_weibull_init/coef": "mismatch: coef[0]: 6.3206 != 6.3206",
    "survreg/lung_weibull_init/coef_names": "mismatch: coef[0]: 6.3206 != 6.3206",
    "survreg/lung_weibull_init/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/lung_weibull_init/loglik": "mismatch: loglik[0]: got nan, expected -1157.3",
    "survreg/lung_weibull_init/iter": "mismatch: iter: 4 != 5",
    "survreg/lung_weibull_init/linear_predictors": (
        "mismatch: linear_predictors[6]: 6.2345 != 6.2345"
    ),
    "survreg/lung_weibull_robust/coef": "mismatch: coef[0]: 0.65845 != 6.2749",
    "survreg/lung_weibull_robust/coef_names": "mismatch: coef[0]: 0.65845 != 6.2749",
    "survreg/lung_weibull_robust/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/lung_weibull_robust/scale": "mismatch: scale[0]: 0.90352 != 0.75405",
    "survreg/lung_weibull_robust/var": "mismatch: var[0][0]: 0.19628 != 0.24068",
    "survreg/lung_weibull_robust/loglik": "mismatch: loglik[0]: got nan, expected -1153.9",
    "survreg/lung_weibull_robust/iter": "mismatch: iter: 30 != 5",
    "survreg/lung_weibull_robust/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.65955 != 5.7499"
    ),
    "survreg/lung_weibull_robust/naive_var": "mismatch: naive_var[0][0]: 0.00083865 != 0.23171",
    "survreg/lung_weibull_scale_fixed/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/lung_weibull_scale_fixed/loglik": "mismatch: loglik[0]: got nan, expected -1162.3",
    "survreg/lung_weibull_scale_fixed/iter": "mismatch: iter: 10 != 4",
    "survreg/lung_weibull_scale_fixed/residuals.response": (
        "mismatch: residuals.response[6]: -212.94 != -212.94"
    ),
    "survreg/lung_weibull_scale_fixed/residuals.deviance": (
        "mismatch: residuals.deviance[49]: 0.10094 != 0.10094"
    ),
    "survreg/lung_weibull_scale_fixed/residuals.working": (
        "mismatch: residuals.working[0]: 0.039524 != 0.038022"
    ),
    "survreg/lung_weibull_scale_fixed/predict.response": (
        "mismatch: predict.response.fit[18]: 630.74 != 630.74"
    ),
    "survreg/lung_weibull_scale_fixed/predict.terms": (
        "mismatch: predict.terms.fit[0][1]: -0.18984 != -0.18984"
    ),
    "survreg/lung_weibull_scale_fixed/predict_newdata.response": (
        "mismatch: predict_newdata.response.fit[1]: 506.86 != 506.86"
    ),
    "survreg/lung_weibull_scale_fixed/predict_newdata.terms": (
        "mismatch: predict_newdata.terms.fit[0][0]: 0.19441 != 0.19441"
    ),
    "survreg/lung_weibull_scale_fixed/predict_newdata.quantile": (
        "mismatch: predict_newdata.quantile.fit[1][2]: 1167.1 != 1167.1"
    ),
    "survreg/lung_weibull_scale_fixed/summary": (
        "mismatch: summary[age].value: -0.015619 != -0.015619"
    ),
    "survreg/lung_weibull_scale_fixed/anova": (
        "missing feature: TypeError: anova requires fitted Cox model objects"
    ),
    "survreg/lung_weibull_scale_fixed/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "survreg/lung_weibull_strata_sex/coef": "mismatch: coef[0]: 0.74662 != 6.2102",
    "survreg/lung_weibull_strata_sex/coef_names": "mismatch: coef[0]: 0.74662 != 6.2102",
    "survreg/lung_weibull_strata_sex/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/lung_weibull_strata_sex/scale": "mismatch: scale: length 1 differs from expected 2",
    "survreg/lung_weibull_strata_sex/var": "mismatch: var[0][0]: -9.097e-05 != 0.22198",
    "survreg/lung_weibull_strata_sex/loglik": "mismatch: loglik[0]: got nan, expected -1152.5",
    "survreg/lung_weibull_strata_sex/iter": "mismatch: iter: 30 != 5",
    "survreg/lung_weibull_strata_sex/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.65218 != 5.7518"
    ),
    "survreg/lung_weibull_strata_sex/residuals.response": (
        "mismatch: residuals.response[0]: 304.08 != -8.7681"
    ),
    "survreg/lung_weibull_strata_sex/residuals.deviance": (
        "mismatch: residuals.deviance[0]: 23.111 != -0.034958"
    ),
    "survreg/lung_weibull_strata_sex/residuals.working": (
        "mismatch: residuals.working[0]: 272.67 != -0.028754"
    ),
    "survreg/lung_weibull_strata_sex/predict.response": (
        "mismatch: predict.response.fit[0]: 1.9197 != 314.77"
    ),
    "survreg/lung_weibull_strata_sex/predict.lp": "mismatch: predict.lp.fit[0]: 0.65218 != 5.7518",
    "survreg/lung_weibull_strata_sex/predict.terms": (
        "mismatch: predict.terms.fit[0][0]: -0.00022044 != -0.12939"
    ),
    "survreg/lung_weibull_strata_sex/predict_newdata.response": (
        "mismatch: predict_newdata.response.fit[0]: 1.9206 != 411.84"
    ),
    "survreg/lung_weibull_strata_sex/predict_newdata.lp": (
        "mismatch: predict_newdata.lp.fit[0]: 0.65264 != 6.0206"
    ),
    "survreg/lung_weibull_strata_sex/predict_newdata.terms": (
        "mismatch: predict_newdata.terms.fit[0][0]: 0.00023751 != 0.13941"
    ),
    "survreg/lung_weibull_strata_sex/predict_newdata.quantile": (
        "mismatch: predict_newdata.quantile.fit[0][0]: 0.25133 != 67.532"
    ),
    "survreg/lung_weibull_strata_sex/predict_newdata.uquantile": (
        "mismatch: predict_newdata.uquantile.fit[0][0]: -1.381 != 4.2126"
    ),
    "survreg/lung_weibull_strata_sex/summary": (
        "mismatch: summary[(Intercept)].value: 0.74662 != 6.2102"
    ),
    "survreg/lung_weibull_strata_sex/anova": (
        "missing feature: TypeError: anova requires fitted Cox model objects"
    ),
    "survreg/lung_weibull_strata_sex/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "survreg/lung_weibull_weighted/coef": "mismatch: coef[0]: 0.67273 != 5.7756",
    "survreg/lung_weibull_weighted/coef_names": "mismatch: coef[0]: 0.67273 != 5.7756",
    "survreg/lung_weibull_weighted/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/lung_weibull_weighted/scale": "mismatch: scale[0]: 0.90034 != 0.75929",
    "survreg/lung_weibull_weighted/var": "mismatch: var[0][0]: 0.00068709 != 0.18183",
    "survreg/lung_weibull_weighted/loglik": "mismatch: loglik[0]: got nan, expected -1429.1",
    "survreg/lung_weibull_weighted/iter": "mismatch: iter: 30 != 5",
    "survreg/lung_weibull_weighted/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.67886 != 5.7535"
    ),
    "survreg/lung_weibull_weighted/residuals.response": (
        "mismatch: residuals.response[0]: 304.03 != -9.2823"
    ),
    "survreg/lung_weibull_weighted/residuals.deviance": (
        "mismatch: residuals.deviance[0]: 23.007 != -0.0391"
    ),
    "survreg/lung_weibull_weighted/residuals.working": (
        "mismatch: residuals.working[0]: 270.27 != -0.030479"
    ),
    "survreg/lung_weibull_weighted/predict.response": (
        "mismatch: predict.response.fit[0]: 1.9716 != 315.28"
    ),
    "survreg/lung_weibull_weighted/predict.lp": "mismatch: predict.lp.fit[0]: 0.67886 != 5.7535",
    "survreg/lung_weibull_weighted/predict.terms": (
        "mismatch: predict.terms.fit[0][0]: 0.00066727 != -0.070379"
    ),
    "survreg/lung_weibull_weighted/predict_newdata.response": (
        "mismatch: predict_newdata.response.fit[0]: 1.9689 != 364.92"
    ),
    "survreg/lung_weibull_weighted/predict_newdata.lp": (
        "mismatch: predict_newdata.lp.fit[0]: 0.67747 != 5.8997"
    ),
    "survreg/lung_weibull_weighted/predict_newdata.terms": (
        "mismatch: predict_newdata.terms.fit[0][0]: -0.00071895 != 0.07583"
    ),
    "survreg/lung_weibull_weighted/predict_newdata.quantile": (
        "mismatch: predict_newdata.quantile.fit[0][0]: 0.25959 != 66.088"
    ),
    "survreg/lung_weibull_weighted/predict_newdata.uquantile": (
        "mismatch: predict_newdata.uquantile.fit[0][0]: -1.3486 != 4.191"
    ),
    "survreg/lung_weibull_weighted/summary": (
        "mismatch: summary[(Intercept)].value: 0.67273 != 5.7756"
    ),
    "survreg/lung_weibull_weighted/anova": (
        "missing feature: TypeError: anova requires fitted Cox model objects"
    ),
    "survreg/lung_weibull_weighted/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "survreg/lung_weibull_x_true/coef": "mismatch: coef[0]: 0.65845 != 6.2749",
    "survreg/lung_weibull_x_true/coef_names": "mismatch: coef[0]: 0.65845 != 6.2749",
    "survreg/lung_weibull_x_true/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/lung_weibull_x_true/scale": "mismatch: scale[0]: 0.90352 != 0.75405",
    "survreg/lung_weibull_x_true/var": "mismatch: var[0][0]: 0.00083865 != 0.23171",
    "survreg/lung_weibull_x_true/loglik": "mismatch: loglik[0]: got nan, expected -1153.9",
    "survreg/lung_weibull_x_true/iter": "mismatch: iter: 30 != 5",
    "survreg/lung_weibull_x_true/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.65955 != 5.7499"
    ),
    "survreg/ovarian_exponential_ecog_rx/coef": "mismatch: coef[1]: -0.43313 != -0.43313",
    "survreg/ovarian_exponential_ecog_rx/coef_names": "mismatch: coef[1]: -0.43313 != -0.43313",
    "survreg/ovarian_exponential_ecog_rx/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/ovarian_exponential_ecog_rx/loglik": "mismatch: loglik[0]: got nan, expected -98.032",
    "survreg/ovarian_exponential_ecog_rx/iter": "mismatch: iter: 11 != 4",
    "survreg/ovarian_exponential_ecog_rx/linear_predictors": (
        "mismatch: linear_predictors[3]: 7.6917 != 7.6917"
    ),
    "survreg/ovarian_loglogistic_age/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/ovarian_loglogistic_age/loglik": "mismatch: loglik[0]: got nan, expected -97.355",
    "survreg/ovarian_loglogistic_age/iter": "mismatch: iter: 10 != 5",
    "survreg/ovarian_weibull_ecog_rx/coef": "mismatch: coef[0]: 0.74639 != 6.8967",
    "survreg/ovarian_weibull_ecog_rx/coef_names": "mismatch: coef[0]: 0.74639 != 6.8967",
    "survreg/ovarian_weibull_ecog_rx/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/ovarian_weibull_ecog_rx/scale": "mismatch: scale[0]: 0.88955 != 0.88387",
    "survreg/ovarian_weibull_ecog_rx/var": "mismatch: var[0][0]: 0.0018539 != 1.3866",
    "survreg/ovarian_weibull_ecog_rx/loglik": "mismatch: loglik[0]: got nan, expected -97.954",
    "survreg/ovarian_weibull_ecog_rx/iter": "mismatch: iter: 30 != 5",
    "survreg/ovarian_weibull_ecog_rx/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.74647 != 7.0403"
    ),
    "survreg/ovarian_weibull_ecog_rx/residuals.response": (
        "mismatch: residuals.response[0]: 56.89 != -1082.7"
    ),
    "survreg/ovarian_weibull_ecog_rx/residuals.deviance": (
        "mismatch: residuals.deviance[0]: 8.666 != -2.185"
    ),
    "survreg/ovarian_weibull_ecog_rx/residuals.working": (
        "mismatch: residuals.working[0]: 41.295 != -24.36"
    ),
    "survreg/ovarian_weibull_ecog_rx/predict.response": (
        "mismatch: predict.response.fit[0]: 2.1096 != 1141.7"
    ),
    "survreg/ovarian_weibull_ecog_rx/predict.lp": "mismatch: predict.lp.fit[0]: 0.74647 != 7.0403",
    "survreg/ovarian_weibull_ecog_rx/predict.terms": (
        "mismatch: predict.terms.fit[0][0]: 0.0013416 != 0.17771"
    ),
    "survreg/ovarian_weibull_ecog_rx/predict_newdata.response": (
        "mismatch: predict_newdata.response.fit[0]: 2.1096 != 1141.7"
    ),
    "survreg/ovarian_weibull_ecog_rx/predict_newdata.lp": (
        "mismatch: predict_newdata.lp.fit[0]: 0.74647 != 7.0403"
    ),
    "survreg/ovarian_weibull_ecog_rx/predict_newdata.terms": (
        "mismatch: predict_newdata.terms.fit[0][0]: 0.0013416 != 0.17771"
    ),
    "survreg/ovarian_weibull_ecog_rx/predict_newdata.quantile": (
        "mismatch: predict_newdata.quantile.fit[0][0]: 0.28498 != 156.22"
    ),
    "survreg/ovarian_weibull_ecog_rx/predict_newdata.uquantile": (
        "mismatch: predict_newdata.uquantile.fit[0][0]: -1.2553 != 5.0513"
    ),
    "survreg/ovarian_weibull_ecog_rx/summary": (
        "mismatch: summary[(Intercept)].value: 0.74639 != 6.8967"
    ),
    "survreg/ovarian_weibull_ecog_rx/anova": (
        "missing feature: TypeError: anova requires fitted Cox model objects"
    ),
    "survreg/ovarian_weibull_ecog_rx/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "survreg/pbc_trial_weibull/coef": "mismatch: coef[0]: 0.50626 != 10.057",
    "survreg/pbc_trial_weibull/coef_names": "mismatch: coef[0]: 0.50626 != 10.057",
    "survreg/pbc_trial_weibull/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/pbc_trial_weibull/scale": "mismatch: scale[0]: 0.9308 != 0.65115",
    "survreg/pbc_trial_weibull/var": "mismatch: var[0][0]: 0.0030413 != 0.10627",
    "survreg/pbc_trial_weibull/loglik": "mismatch: loglik[0]: got nan, expected -1188.8",
    "survreg/pbc_trial_weibull/iter": "mismatch: iter: 21 != 6",
    "survreg/pbc_trial_weibull/linear_predictors": (
        "mismatch: linear_predictors[0]: 0.44552 != 5.9846"
    ),
    "survreg/synthetic_ties_weibull/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/synthetic_ties_weibull/loglik": "mismatch: loglik[0]: got nan, expected -31.806",
    "survreg/synthetic_ties_weibull/iter": "mismatch: iter: 9 != 7",
    "survreg/synthetic_ties_weibull/residuals.working": (
        "mismatch: residuals.working[0]: -0.94748 != -9.915"
    ),
    "survreg/synthetic_ties_weibull/residuals.ldresp": (
        "mismatch: residuals.ldresp[8]: 0.49917 != 0.49917"
    ),
    "survreg/synthetic_ties_weibull/residuals.ldshape": (
        "mismatch: residuals.ldshape[11]: 1.0199 != 1.0199"
    ),
    "survreg/synthetic_ties_weibull/residuals.matrix": (
        "mismatch: residuals.matrix[0][2]: -0.17389 != -0.17389"
    ),
    "survreg/synthetic_ties_weibull/predict.terms": (
        "mismatch: predict.terms.se_fit[0][0]: 0.052886 != 0.021055"
    ),
    "survreg/synthetic_ties_weibull/anova": (
        "missing feature: TypeError: anova requires fitted Cox model objects"
    ),
    "survreg/synthetic_ties_weibull/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "survreg/tobin_gaussian_left": "error: ValueError: time[0] must be positive",
    "survreg/tobin_logistic_left": "error: ValueError: time[0] must be positive",
    "survreg/veteran_lognormal_celltype_karno/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/veteran_lognormal_celltype_karno/loglik": (
        "mismatch: loglik[0]: got nan, expected -749.47"
    ),
    "survreg/veteran_lognormal_celltype_karno/iter": "mismatch: iter: 18 != 4",
    "survreg/veteran_weibull_celltype_karno/coef": "mismatch: coef[0]: 1.1634 != 3.4806",
    "survreg/veteran_weibull_celltype_karno/coef_names": "mismatch: coef[0]: 1.1634 != 3.4806",
    "survreg/veteran_weibull_celltype_karno/icoef": (
        "missing feature: result has none of the attributes ('icoef',)"
    ),
    "survreg/veteran_weibull_celltype_karno/scale": "mismatch: scale[0]: 0.82514 != 0.93782",
    "survreg/veteran_weibull_celltype_karno/var": "mismatch: var[0][0]: 0.0010972 != 0.11593",
    "survreg/veteran_weibull_celltype_karno/loglik": (
        "mismatch: loglik[0]: got nan, expected -748.09"
    ),
    "survreg/veteran_weibull_celltype_karno/iter": "mismatch: iter: 30 != 5",
    "survreg/veteran_weibull_celltype_karno/linear_predictors": (
        "mismatch: linear_predictors[0]: 1.1267 != 5.2312"
    ),
    "survreg/veteran_weibull_celltype_karno/residuals.response": (
        "mismatch: residuals.response[0]: 68.915 != -115.01"
    ),
    "survreg/veteran_weibull_celltype_karno/residuals.deviance": (
        "mismatch: residuals.deviance[0]: 9.0192 != -0.87085"
    ),
    "survreg/veteran_weibull_celltype_karno/residuals.working": (
        "mismatch: residuals.working[0]: 44.49 != -1.6572"
    ),
    "survreg/veteran_weibull_celltype_karno/predict.response": (
        "mismatch: predict.response.fit[0]: 3.0854 != 187.01"
    ),
    "survreg/veteran_weibull_celltype_karno/predict.lp": (
        "mismatch: predict.lp.fit[0]: 1.1267 != 5.2312"
    ),
    "survreg/veteran_weibull_celltype_karno/predict.terms": (
        "mismatch: predict.terms.fit[0][0]: 0.0066626 != 0.53003"
    ),
    "survreg/veteran_weibull_celltype_karno/predict_newdata.response": (
        "mismatch: predict_newdata.response.fit[0]: 3.0854 != 187.01"
    ),
    "survreg/veteran_weibull_celltype_karno/predict_newdata.lp": (
        "mismatch: predict_newdata.lp.fit[0]: 1.1267 != 5.2312"
    ),
    "survreg/veteran_weibull_celltype_karno/predict_newdata.terms": (
        "mismatch: predict_newdata.terms.fit[0][0]: 0.0066626 != 0.53003"
    ),
    "survreg/veteran_weibull_celltype_karno/predict_newdata.quantile": (
        "mismatch: predict_newdata.quantile.fit[0][0]: 0.48181 != 22.663"
    ),
    "survreg/veteran_weibull_celltype_karno/predict_newdata.uquantile": (
        "mismatch: predict_newdata.uquantile.fit[0][0]: -0.7302 != 3.1208"
    ),
    "survreg/veteran_weibull_celltype_karno/summary": (
        "mismatch: summary[(Intercept)].value: 1.1634 != 3.4806"
    ),
    "survreg/veteran_weibull_celltype_karno/anova": (
        "missing feature: TypeError: anova requires fitted Cox model objects"
    ),
    "survreg/veteran_weibull_celltype_karno/concordance.concordance": (
        "missing feature: concordance(fit) for model objects: concordance response must be a Su..."
    ),
    "utilities/bounded_links/blogit_linkinv": (
        "missing feature: bounded link inverse functions are not exposed"
    ),
    "utilities/bounded_links/bprobit_linkinv": (
        "missing feature: bounded link inverse functions are not exposed"
    ),
    "utilities/bounded_links/bcloglog_linkinv": (
        "missing feature: bounded link inverse functions are not exposed"
    ),
    "utilities/bounded_links/blog_linkinv": (
        "missing feature: bounded link inverse functions are not exposed"
    ),
    "utilities/statefig/layout_1_2_column": "mismatch: layout_1_2_column[0][0]: 0.25 != 0.5",
    "utilities/statefig/layout_1_2_1_column": "mismatch: layout_1_2_1_column[0][0]: 0.16667 != 0.5",
    "utilities/surv_types/interval": (
        "error: TypeError: float() argument must be a string or a real number, not 'N..."
    ),
    "utilities/surv_types/interval2": "mismatch: interval2.type: 'interval2' != 'interval'",
    "utilities/surv_types/is_na": (
        "error: TypeError: float() argument must be a string or a real number, not 'N..."
    ),
    "utilities/surv_types/format_counting": "mismatch: format_counting[0]: '(0, 3] ' != '(0,3] '",
    "yates/lung_ph_ecog_factor": (
        "missing feature: yates(fit, term, population=...) formula interface is not available"
    ),
    "yates/lung_ph_ecog_factor_pop_data": (
        "missing feature: yates(fit, term, population=...) formula interface is not available"
    ),
    "yates/veteran_celltype_linear": (
        "missing feature: yates(fit, term, population=...) formula interface is not available"
    ),
    "yates/veteran_celltype_lm": (
        "missing feature: yates(fit, term, population=...) formula interface is not available"
    ),
    "yates/veteran_celltype_pop_data": (
        "missing feature: yates(fit, term, population=...) formula interface is not available"
    ),
    "yates/veteran_celltype_pop_factorial": (
        "missing feature: yates(fit, term, population=...) formula interface is not available"
    ),
    "yates/veteran_celltype_pop_sas": (
        "missing feature: yates(fit, term, population=...) formula interface is not available"
    ),
    "yates/veteran_celltype_predict_risk": (
        "missing feature: yates(fit, term, population=...) formula interface is not available"
    ),
    "yates/veteran_trt_factor": (
        "missing feature: yates(fit, term, population=...) formula interface is not available"
    ),
}


def _known_failure_reason(test_id: str) -> str | None:
    if test_id in KNOWN_FAILURES:
        return KNOWN_FAILURES[test_id]
    topic, name, _aspect = test_id.split("/", 2)
    return KNOWN_FAILURES.get(f"{topic}/{name}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FIT_CACHE: dict[str, Any] = {}

_ARG_NAMES = {
    "conf.type": "conf_type",
    "conf.int": "conf_int",
    "conf.lower": "conf_lower",
    "se.fit": "se_fit",
    "start.time": "start_time",
    "cohort.size": "cohort_size",
    "data.frame": "data_frame",
    "na.action": "na_action",
}


def _kwargs(
    args: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    drop: Sequence[str] = (),
    na_omit: bool = True,
) -> dict:
    """Translate R call arguments into Python keyword arguments.

    R's model functions drop rows with missing values (``na.action = na.omit``)
    by default, so ``na_action="omit"`` is passed unless the case says otherwise.
    """

    out: dict[str, Any] = {"na_action": "omit"} if na_omit else {}
    for name, value in args.items():
        if name in drop:
            continue
        key = _ARG_NAMES.get(name, name)
        if isinstance(value, list):
            value = decode_vector(value)
        out[key] = value
    return out


def _cached(key: str, build: Callable[[], Any]) -> Any:
    if key not in _FIT_CACHE:
        _FIT_CACHE[key] = build()
    value = _FIT_CACHE[key]
    if isinstance(value, BaseException):
        raise value
    return value


def _fit_key(family: str, case: Mapping[str, Any]) -> str:
    """Cache key for a fit: the same inputs fitted by the same function."""

    return "|".join(
        [
            family,
            str(case.get("dataset")),
            str(case.get("data_ref")),
            str(case.get("rows")),
            str(case.get("formula")),
            str(case.get("args")),
        ]
    )


def _expect(case: Mapping[str, Any], aspect: str) -> Any:
    """Look up ``expected[aspect]`` where aspect may be dotted."""

    value: Any = case["expected"]
    for part in aspect.split("."):
        value = value[part]
    if is_r_error(value):
        pytest.skip(f"R cannot compute this aspect: {value['r_error']}")
    return value


def _aspects_from_expected(expected: Mapping[str, Any], expand: Sequence[str]) -> list[str]:
    out: list[str] = []
    for key, value in expected.items():
        if key in ("newdata", "cox_formula"):
            continue
        if key in expand and isinstance(value, Mapping) and not is_r_error(value):
            out.extend(f"{key}.{sub}" for sub in value)
        else:
            out.append(key)
    return out


def _formula_is_null_model(formula: str) -> bool:
    return formula.split("~", 1)[1].strip() == "1"


def _attr(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    raise UnsupportedCaseError(f"result has none of the attributes {names}")


def _concordance_of_fit(fit: Any, **kwargs: Any) -> Any:
    try:
        return r.concordance(fit, **kwargs)
    except TypeError as exc:
        raise UnsupportedCaseError(f"concordance(fit) for model objects: {exc}") from exc


def _tt_function(source: str) -> Callable[..., Any]:
    if source.replace(" ", "") == "function(x,t,...)x*log(t+20)":
        return lambda x, t, *args: [xi * math.log(ti + 20) for xi, ti in zip(x, t, strict=True)]
    raise UnsupportedCaseError(f"no Python translation for tt = {source!r}")


# ---------------------------------------------------------------------------
# survfit curve comparison (shared by KM, Cox and multistate curves)
# ---------------------------------------------------------------------------

_CURVE_FIELDS = {
    # r field -> (python attribute candidates, rtol or None for exact)
    "time": (("time",), 0.0),
    "n_risk": (("n_risk",), 0.0),
    "n_event": (("n_event",), 0.0),
    "n_censor": (("n_censor",), 0.0),
    "n_enter": (("n_enter",), 0.0),
    "surv": (("surv", "estimate"), RTOL_COEF),
    "std_err": (("std_err",), RTOL_VAR),
    "cumhaz": (("cumhaz",), RTOL_COEF),
    "std_chaz": (("std_chaz",), RTOL_VAR),
    "lower": (("conf_lower", "lower"), RTOL_VAR),
    "upper": (("conf_upper", "upper"), RTOL_VAR),
    "pstate": (("pstate",), RTOL_COEF),
    "std_err0": (("std_err0",), RTOL_VAR),
    "n_transition": (("n_transition",), 0.0),
}

_CURVE_ASPECTS = {
    "curves.time_counts": ("time", "n_risk", "n_event", "n_censor"),
    "curves.n_enter": ("n_enter",),
    "curves.surv": ("surv",),
    "curves.pstate": ("pstate",),
    "curves.std_err": ("std_err", "std_err0"),
    "curves.cumhaz": ("cumhaz",),
    "curves.std_chaz": ("std_chaz",),
    "curves.conf": ("lower", "upper"),
    "curves.n_transition": ("n_transition",),
}


def _curve_aspects(fit_expected: Mapping[str, Any]) -> list[str]:
    present: set[str] = set()
    for curve in fit_expected["curves"]:
        present.update(curve)
    out = []
    for aspect, fields in _CURVE_ASPECTS.items():
        if any(field in present for field in fields):
            out.append(aspect)
    return out


def _python_curves(result: Any, expected_curves: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Align Python curve objects with the R strata order."""

    if isinstance(result, Mapping):
        keys = list(result)
        if len(keys) != len(expected_curves):
            raise FixtureMismatchError(f"{len(keys)} python curves, R has {len(expected_curves)}")
        by_label = {str(key): key for key in keys}
        aligned = []
        for curve in expected_curves:
            label = r_strata_value(curve["name"])
            if label in by_label:
                aligned.append(result[by_label[label]])
            else:
                aligned.append(result[keys[len(aligned)]])
        return aligned
    if len(expected_curves) != 1:
        raise FixtureMismatchError(f"python returned one curve, R has {len(expected_curves)}")
    return [result]


def _compare_curve_fields(
    python_curve: Any, expected_curve: Mapping[str, Any], fields: Sequence[str], path: str
) -> None:
    for field in fields:
        if field not in expected_curve:
            continue
        candidates, rtol = _CURVE_FIELDS[field]
        actual = _attr(python_curve, *candidates)
        expected = expected_curve[field]
        if actual is None:
            raise FixtureMismatchError(f"{path}.{field}: python value is None")
        if expected and isinstance(expected[0], list):
            assert_matrix_close(actual, expected, rtol=rtol, path=f"{path}.{field}")
        else:
            assert_close(as_float_list(actual), expected, rtol=rtol, path=f"{path}.{field}")


def _check_curves(result: Any, fit_expected: Mapping[str, Any], aspect: str) -> None:
    fields = _CURVE_ASPECTS[aspect]
    for python_curve, expected_curve in zip(
        _python_curves(result, fit_expected["curves"]), fit_expected["curves"], strict=True
    ):
        _compare_curve_fields(
            python_curve, expected_curve, fields, f"curve[{expected_curve['name']}]"
        )


# ---------------------------------------------------------------------------
# Topic handlers
# ---------------------------------------------------------------------------

HANDLERS: dict[str, TopicHandler] = {}


class TopicHandler:
    topic: str = ""
    expand: tuple[str, ...] = ()

    def __init_subclass__(cls) -> None:
        if cls.topic:
            HANDLERS[cls.topic] = cls()

    def aspects(self, case: Mapping[str, Any]) -> list[str]:
        return _aspects_from_expected(case["expected"], self.expand)

    def check(self, case: Mapping[str, Any], aspect: str) -> None:
        raise NotImplementedError


# --- datasets ---------------------------------------------------------------


class DatasetsHandler(TopicHandler):
    topic = "datasets"

    def aspects(self, case):
        return ["shape", "values"]

    def check(self, case, aspect):
        expected = case["expected"]
        data = load_dataset(case["dataset"])
        if aspect == "shape":
            assert_exact(nrow(data), expected["nrow"], path="nrow")
            r_names = [col["name"] for col in expected["columns"]]
            if list(data) != r_names:
                raise FixtureMismatchError(f"columns {list(data)} != {r_names}")
            return
        for col in expected["columns"]:
            values = data.get(col["name"])
            if values is None:
                raise FixtureMismatchError(f"column {col['name']} missing")
            missing = sum(
                1 for v in values if v is None or (isinstance(v, float) and math.isnan(v))
            )
            assert_exact(missing, col["n_missing"], path=f"{col['name']}.n_missing")
            if col["type"] in ("factor", "character"):
                present = sorted({str(v) for v in values if v is not None})
                if col["type"] == "factor":
                    if present and not set(present) <= set(col["levels"]):
                        raise FixtureMismatchError(
                            f"{col['name']}: values {present[:5]} are not factor levels "
                            f"{col['levels']}"
                        )
                elif present != list(col["levels"]):
                    raise FixtureMismatchError(f"{col['name']}: distinct values differ from R")
            elif col["type"] == "logical":
                total = sum(1 for v in values if v)
                assert_exact(total, col["sum"], path=f"{col['name']}.sum")
            elif col["type"] == "date":
                continue
            else:
                total = sum(
                    float(v)
                    for v in values
                    if not (v is None or (isinstance(v, float) and math.isnan(v)))
                )
                assert_close(total, col["sum"], rtol=1e-10, atol=1e-8, path=f"{col['name']}.sum")


# --- survfit (KM) -----------------------------------------------------------


def _mstate_formula(formula: str, data: Mapping[str, Any]) -> str:
    """Add ``type = "mstate"`` when the status column is an R factor.

    R infers a multi-state response from a factor status column; the Python
    formula parser needs the explicit type hint (see the
    ``mstate_from_factor`` aspect, which checks the inference itself).
    """

    lhs, _, rhs = formula.partition("~")
    inner = lhs.strip()
    if not inner.startswith("Surv(") or "type" in inner:
        return formula
    args = [arg.strip() for arg in inner[5:-1].split(",")]
    status = args[-1]
    if isinstance(data.get(status), RFactor):
        return f'Surv({", ".join(args)}, type = "mstate") ~{rhs}'
    return formula


def _survfit_call(
    topic: str,
    case: Mapping[str, Any],
    *,
    drop: Sequence[str] = (),
    infer_mstate: bool = True,
    keep_model: bool = False,
) -> Any:
    def build():
        data = case_data(topic, case)
        kwargs = _kwargs(case.get("args", {}), data, drop=drop)
        if "conf_lower" in kwargs:
            raise UnsupportedCaseError("survfit has no conf.lower argument")
        if keep_model:
            kwargs["model"] = True
        formula = _mstate_formula(case["formula"], data) if infer_mstate else case["formula"]
        return r.survfit(formula, data, **kwargs)

    family = "survfit" if infer_mstate else "survfit-nohint"
    if keep_model:
        family += "-model"
    return _cached(_fit_key(family, case), build)


class SurvfitKMHandler(TopicHandler):
    topic = "survfit_km"

    def aspects(self, case):
        expected = case["expected"]
        out = _curve_aspects(expected["fit"])
        out.append("fit.n")
        out.extend(key for key in expected if key != "fit")
        return out

    def check(self, case, aspect):
        fit = _survfit_call(self.topic, case)
        expected = case["expected"]
        if aspect.startswith("curves."):
            _check_curves(fit, expected["fit"], aspect)
        elif aspect == "fit.n":
            curves = _python_curves(fit, expected["fit"]["curves"])
            n_values = []
            for c in curves:
                n_value = getattr(c, "n", None)
                if n_value is None:
                    raise UnsupportedCaseError("survfit result does not report n")
                n_values.append(n_value)
            assert_exact(n_values, expected["fit"]["n"], path="n")
        elif aspect == "summary_std_err":
            # summary(fit)$std.err = surv-scale standard error at event times only
            curves = _python_curves(fit, expected["fit"]["curves"])
            actual: list[float] = []
            for c in curves:
                std_err = as_float_list(_attr(c, "std_err"))
                n_event = as_float_list(_attr(c, "n_event"))
                actual.extend(se for se, d in zip(std_err, n_event, strict=True) if d > 0)
            assert_close(actual, expected["summary_std_err"], rtol=RTOL_VAR, path="summary_std_err")
        elif aspect in ("summary_table", "quantile"):
            raise UnsupportedCaseError(
                f"no summary.survfit/quantile.survfit equivalent for {aspect}"
            )
        elif aspect == "summary_times":
            raise UnsupportedCaseError("no summary(fit, times=) equivalent")
        else:
            raise UnsupportedCaseError(f"unhandled survfit aspect {aspect}")


# --- survfit multistate -----------------------------------------------------


class SurvfitMultistateHandler(TopicHandler):
    topic = "survfit_multistate"

    def aspects(self, case):
        expected = case["expected"]
        out = _curve_aspects(expected["fit"])
        out.extend(["fit.states", "fit.p0", "fit.transitions", "mstate_from_factor"])
        out.extend(key for key in expected if key != "fit")
        return out

    def check(self, case, aspect):
        expected = case["expected"]
        if aspect == "mstate_from_factor":
            fit = _survfit_call(self.topic, case, infer_mstate=False)
            first = _python_curves(fit, expected["fit"]["curves"])[0]
            if not hasattr(first, "pstate"):
                raise FixtureMismatchError(
                    "factor status column was not treated as a multi-state response"
                )
            return
        fit = _survfit_call(self.topic, case)
        if aspect.startswith("curves."):
            _check_curves(fit, expected["fit"], aspect)
        elif aspect == "fit.states":
            first = _python_curves(fit, expected["fit"]["curves"])[0]
            assert_exact(list(_attr(first, "states")), expected["fit"]["states"], path="states")
        elif aspect == "fit.p0":
            curves = _python_curves(fit, expected["fit"]["curves"])
            p0 = expected["fit"]["p0"]
            if p0 and isinstance(p0[0], list):
                assert_matrix_close([list(_attr(c, "p0")) for c in curves], p0, path="p0")
            else:
                assert_close(list(_attr(curves[0], "p0")), p0, path="p0")
        elif aspect == "fit.transitions":
            first = _python_curves(fit, expected["fit"]["curves"])[0]
            actual = _attr(first, "transitions")
            table = expected["fit"]["transitions"]
            states = list(_attr(first, "states"))
            counts: dict[tuple[str, str], float] = {}
            for row_name, row in zip(table["rownames"], table["values"], strict=True):
                for col_name, value in zip(table["colnames"], row, strict=True):
                    counts[(row_name, col_name)] = value
            # Python reports transitions as (from, to) state index pairs.
            actual_pairs = {(states[a], states[b]) for a, b in actual}
            expected_pairs = {
                key for key, value in counts.items() if value > 0 and key[1] != "(censored)"
            }
            if actual_pairs != expected_pairs:
                raise FixtureMismatchError(
                    f"transitions {sorted(actual_pairs)} != {sorted(expected_pairs)}"
                )
        elif aspect == "summary_times":
            raise UnsupportedCaseError("no summary(fit, times=) equivalent")
        elif aspect in ("influence_pstate", "influence_chaz"):
            first = _python_curves(fit, expected["fit"]["curves"])[0]
            attr = "influence_state" if aspect == "influence_pstate" else "influence_chaz"
            actual = getattr(first, attr, None)
            if actual is None:
                raise UnsupportedCaseError(f"survfit result has no {attr}")
            # R stores subjects x times x states; Python stores a flat per-subject layout.
            layers = expected[aspect]
            n_subjects = len(layers[0])
            n_times = len(layers[0][0])
            n_states = len(layers)
            if len(actual) != n_subjects:
                raise FixtureMismatchError(f"influence rows {len(actual)} != subjects {n_subjects}")
            for subject in range(n_subjects):
                row = list(actual[subject])
                if len(row) != n_times * n_states:
                    raise FixtureMismatchError(
                        f"influence row length {len(row)} != times*states {n_times * n_states}"
                    )
                for state in range(n_states):
                    actual_col = row[state * n_times : (state + 1) * n_times]
                    assert_close(
                        actual_col,
                        layers[state][subject],
                        rtol=RTOL_VAR,
                        path=f"{aspect}[{subject}][state {state}]",
                    )
        else:
            raise UnsupportedCaseError(f"unhandled multistate aspect {aspect}")


# --- survfit interval -------------------------------------------------------


class SurvfitIntervalHandler(TopicHandler):
    topic = "survfit_interval"

    def aspects(self, case):
        expected = case["expected"]
        if "fit" in expected:
            return _curve_aspects(expected["fit"])
        return ["time_surv", "counts", "std_err", "conf", "n"]

    def check(self, case, aspect):
        fit = _survfit_call(self.topic, case)
        expected = case["expected"]
        if aspect.startswith("curves."):
            _check_curves(fit, expected["fit"], aspect)
            return
        if aspect == "time_surv":
            assert_close(
                as_float_list(_attr(fit, "time_points", "time")), expected["time"], path="time"
            )
            assert_close(
                as_float_list(_attr(fit, "survival", "surv")), expected["surv"], path="surv"
            )
        elif aspect == "counts":
            assert_exact(as_float_list(_attr(fit, "n_risk")), expected["n_risk"], path="n_risk")
            assert_exact(as_float_list(_attr(fit, "n_event")), expected["n_event"], path="n_event")
        elif aspect == "std_err":
            assert_close(
                as_float_list(_attr(fit, "std_err")),
                expected["std_err"],
                rtol=RTOL_VAR,
                path="std_err",
            )
        elif aspect == "conf":
            assert_close(
                as_float_list(_attr(fit, "survival_lower", "conf_lower")),
                expected["lower"],
                rtol=RTOL_VAR,
                path="lower",
            )
            assert_close(
                as_float_list(_attr(fit, "survival_upper", "conf_upper")),
                expected["upper"],
                rtol=RTOL_VAR,
                path="upper",
            )
        elif aspect == "n":
            assert_exact(_attr(fit, "n"), expected["n"], path="n")


# --- survdiff ---------------------------------------------------------------


class SurvdiffHandler(TopicHandler):
    topic = "survdiff"

    def aspects(self, case):
        return ["counts", "var", "chisq", "pvalue"]

    def check(self, case, aspect):
        expected = case["expected"]

        def build():
            data = case_data(self.topic, case)
            kwargs = _kwargs(case.get("args", {}), data, drop=("expect",))
            if "expect" in case.get("args", {}):
                data = dict(data)
                data["expect"] = decode_vector(case["args"]["expect"])
            return r.survdiff(case["formula"], data, **kwargs)

        result = _cached(_fit_key("survdiff", case), build)
        if aspect == "counts":
            n = _attr(result, "n", "n_group", "group_sizes") if hasattr(result, "n") else None
            if n is not None:
                assert_exact(as_float_list(n), expected["n"], path="n")
            # with strata R reports group x stratum matrices; compare the totals
            totals = lambda value: (  # noqa: E731
                [sum(row) for row in value] if value and isinstance(value[0], list) else value
            )
            assert_close(
                as_float_list(_attr(result, "observed")),
                totals(expected["obs"]),
                rtol=1e-12,
                path="obs",
            )
            assert_close(
                as_float_list(_attr(result, "expected")),
                totals(expected["exp"]),
                rtol=RTOL_COEF,
                path="exp",
            )
        elif aspect == "var":
            variance = _attr(result, "variance", "var")
            if hasattr(variance, "tolist"):
                variance = variance.tolist()
            if isinstance(expected["var"][0], list):
                if isinstance(variance, (list, tuple)):
                    assert_matrix_close(variance, expected["var"], rtol=RTOL_VAR, path="var")
                else:
                    # a scalar variance is the (1, 1) element of R's k x k matrix
                    assert_close(variance, expected["var"][0][0], rtol=RTOL_VAR, path="var[0][0]")
            else:
                assert_close(variance, expected["var"], rtol=RTOL_VAR, path="var")
        elif aspect == "chisq":
            assert_close(
                _attr(result, "statistic", "chisq"), expected["chisq"], rtol=RTOL_VAR, path="chisq"
            )
        elif aspect == "pvalue":
            assert_close(
                _attr(result, "p_value", "pvalue"), expected["pvalue"], rtol=RTOL_VAR, path="pvalue"
            )


# --- coxph ------------------------------------------------------------------


def _coxph_fit(topic: str, case: Mapping[str, Any]) -> Any:
    def build():
        data = case_data(topic, case)
        args = dict(case.get("args", {}))
        kwargs = _kwargs(args, data, drop=("control", "tt", "init"))
        if "control" in args:
            control = args["control"]
            if "iter.max" in control:
                kwargs["max_iter"] = int(control["iter.max"])
            if "eps" in control:
                kwargs["eps"] = float(control["eps"])
            if "toler.chol" in control:
                kwargs["toler"] = float(control["toler.chol"])
        if "init" in args:
            kwargs["init"] = decode_vector(args["init"])
        if "tt" in args:
            kwargs["tt"] = _tt_function(args["tt"])
        if "nocenter" in args and args["nocenter"] is None:
            kwargs["nocenter"] = None
        return r.coxph(case["formula"], data, **kwargs)

    return _cached(_fit_key("coxph", case), build)


def _coef_names(fit: Any) -> list[str] | None:
    try:
        return list(r.coef_names(fit))
    except Exception:  # noqa: BLE001 - names are optional for the comparison
        return None


def _r_wald_test(fit: Any, init: Sequence[float] | None = None) -> float:
    """R's ``fit$wald.test``: ``coxph.wtest(var, coef - init)$test`` (R/coxph.R)."""

    coef = as_float_list(r.coef(fit))
    if init is not None:
        coef = [c - i for c, i in zip(coef, as_float_list(init)[: len(coef)], strict=True)]
    wald = r.coxph_wtest(r.vcov(fit), coef)
    return as_float_list(_attr(wald, "test"))[0]


def _case_init(case: Mapping[str, Any]) -> list[float] | None:
    init = (case.get("args") or {}).get("init")
    return None if init is None else decode_vector(init)


def _check_summary(
    fit: Any, aspect: str, expected: Mapping[str, Any], init: Sequence[float] | None = None
) -> None:
    summary = r.model_summary(fit)
    if aspect == "summary.logtest":
        loglik = _attr(summary, "loglik")
        null_loglik = _attr(summary, "null_loglik")
        assert_close(
            2 * (loglik - null_loglik), expected["test"], rtol=RTOL_VAR, path="logtest.test"
        )
        assert_exact(_attr(summary, "df"), expected["df"], path="logtest.df")
    elif aspect == "summary.sctest":
        assert_close(
            _attr(summary, "score_test"), expected["test"], rtol=RTOL_VAR, path="sctest.test"
        )
    elif aspect == "summary.waldtest":
        # summary.coxph reports round(fit$wald.test, 2) (R/summary.coxph.R):
        # compare within half a unit of the last reported digit (the extra
        # 1e-9 absorbs R's round-half-even at an exact .xx5); the unrounded
        # statistic is checked by the wald_test aspect.
        wald = r.coxph_wtest(r.vcov(fit), r.coef(fit))
        assert_close(
            _r_wald_test(fit, init),
            expected["test"],
            rtol=0.0,
            atol=0.005 + 1e-9,
            path="waldtest.test",
        )
        assert_exact(_attr(wald, "df"), expected["df"], path="waldtest.df")
    elif aspect == "summary.coefficients":
        rows = _attr(summary, "coefficients")
        r_cols = expected["colnames"]
        for row, r_row in zip(rows, expected["values"], strict=True):
            mapping = {
                "coef": "coef",
                "exp(coef)": "exp_coef",
                "se(coef)": "se",
                "z": "z",
                "Pr(>|z|)": "p",
                "robust se": "se",
            }
            for col_name, value in zip(r_cols, r_row, strict=True):
                key = mapping.get(col_name)
                if key is None or key not in row:
                    continue
                if col_name == "se(coef)" and "robust se" in r_cols:
                    key = "naive_se"
                rtol = RTOL_COEF if col_name in ("coef", "exp(coef)") else RTOL_VAR
                assert_close(
                    row[key], value, rtol=rtol, path=f"coefficients[{row['name']}].{col_name}"
                )
    elif aspect == "summary.conf_int":
        ci = r.confint(fit)
        lower = [_attr(row, "lower") if isinstance(row, Mapping) else row[0] for row in ci]
        upper = [_attr(row, "upper") if isinstance(row, Mapping) else row[1] for row in ci]
        r_cols = expected["colnames"]
        lo_idx = r_cols.index("lower .95")
        hi_idx = r_cols.index("upper .95")
        assert_close(
            [math.exp(v) for v in lower],
            column(expected["values"], lo_idx),
            rtol=RTOL_VAR,
            path="conf_int.lower",
        )
        assert_close(
            [math.exp(v) for v in upper],
            column(expected["values"], hi_idx),
            rtol=RTOL_VAR,
            path="conf_int.upper",
        )
    elif aspect == "summary.concordance":
        cc = _concordance_of_fit(fit)
        values = list(expected.values())
        assert_close(_attr(cc, "concordance"), values[0], rtol=RTOL_VAR, path="concordance")
        variance = _attr(cc, "variance", "var")
        if variance is None:
            raise FixtureMismatchError("concordance variance is None")
        se = math.sqrt(variance if not isinstance(variance, list) else variance[0][0])
        assert_close(se, values[1], rtol=RTOL_VAR, path="concordance.se")
    elif aspect == "summary.rsq":
        n = _attr(summary, "n")
        loglik = _attr(summary, "loglik")
        null_loglik = _attr(summary, "null_loglik")
        logtest = -2 * (null_loglik - loglik)
        rsq = 1 - math.exp(-logtest / n)
        maxrsq = 1 - math.exp(2 * null_loglik / n)
        assert_close([rsq, maxrsq], list(expected.values()), rtol=RTOL_VAR, path="rsq")
    elif aspect in ("summary.n", "summary.nevent"):
        key = "n" if aspect == "summary.n" else "n_event"
        assert_exact(_attr(summary, key), expected, path=aspect)
    elif aspect == "summary.used_robust":
        assert_exact(bool(_attr(summary, "robust")), expected, path=aspect)
    else:
        raise UnsupportedCaseError(f"unhandled summary aspect {aspect}")


def _check_concordance_result(cc: Any, expected: Mapping[str, Any], aspect: str) -> None:
    if aspect.endswith("concordance") and not aspect.endswith(".concordance"):
        raise AssertionError("internal: use the dotted aspect")
    sub = aspect.rsplit(".", 1)[1] if "." in aspect else aspect
    if sub == "concordance":
        actual = _attr(cc, "concordance")
        assert_close(actual, expected["concordance"], rtol=RTOL_COEF, path="concordance")
    elif sub == "n":
        assert_exact(_attr(cc, "n"), expected["n"], path="n")
    elif sub == "count":
        count = expected["count"]
        if "values" in count:
            raise UnsupportedCaseError("multi-column concordance counts are not exposed")
        tied_x = _attr(cc, "tied_x")
        tied_y = _attr(cc, "tied_y")
        tied_xy = _attr(cc, "tied_xy")
        concordant = _attr(cc, "concordant")
        comparable = _attr(cc, "comparable")
        discordant = comparable - concordant - tied_x
        actual = [concordant, discordant, tied_x, tied_y, tied_xy]
        assert_close(actual, list(count.values()), rtol=1e-12, path="count")
    elif sub == "var":
        variance = _attr(cc, "variance", "var")
        if variance is None:
            raise FixtureMismatchError("variance is None")
        assert_close(variance, expected["var"], rtol=RTOL_VAR, path="var")
    elif sub == "cvar":
        assert_close(
            _attr(cc, "conditional_variance", "cvar"), expected["cvar"], rtol=RTOL_VAR, path="cvar"
        )
    elif sub == "dfbeta":
        actual = _attr(cc, "dfbeta")
        if actual is None:
            raise FixtureMismatchError("dfbeta is None")
        assert_close(actual, expected["dfbeta"], rtol=RTOL_VAR, path="dfbeta")
    elif sub == "influence":
        actual = _attr(cc, "influence")
        if actual is None:
            raise FixtureMismatchError("influence is None")
        assert_matrix_close(actual, expected["influence"], rtol=RTOL_VAR, path="influence")
    elif sub == "timewt":
        return
    else:
        raise UnsupportedCaseError(f"unhandled concordance aspect {sub}")


class CoxphHandler(TopicHandler):
    topic = "coxph"
    expand = ("residuals", "summary", "concordance", "anova", "wtest", "survfit")

    def aspects(self, case):
        expected = case["expected"]
        if "formulas" in case:
            return ["anova_nested"]
        out = []
        for key, value in expected.items():
            if key in ("method", "coef_names"):
                continue
            if key == "wald_test" and value is None:
                continue  # R has no wald.test for a null model
            if key == "coef":
                out.extend(["coef", "coef_names"])
            elif key in ("residuals", "summary", "concordance") and isinstance(value, Mapping):
                out.extend(f"{key}.{sub}" for sub in value if not is_r_error(value[sub]))
            elif key == "survfit":
                out.extend(_curve_aspects(value))
            else:
                out.append(key)
        return out

    def check(self, case, aspect):
        if aspect == "anova_nested":
            self._check_anova_nested(case)
            return
        fit = _coxph_fit(self.topic, case)
        expected = case["expected"]
        if aspect == "coef":
            assert_named_values(_coef_names(fit), r.coef(fit), expected["coef"], path="coef")
        elif aspect == "coef_names":
            assert_named_values(
                _coef_names(fit), r.coef(fit), expected["coef"], check_names=True, path="coef"
            )
        elif aspect == "var":
            assert_matrix_close(r.vcov(fit), expected["var"], rtol=RTOL_VAR, path="var")
        elif aspect == "naive_var":
            naive = getattr(fit, "naive_var", None) or getattr(fit, "naive_variance", None)
            if naive is None:
                raise FixtureMismatchError("fit has no naive variance")
            assert_matrix_close(naive, expected["naive_var"], rtol=RTOL_VAR, path="naive_var")
        elif aspect == "loglik":
            loglik = _attr(fit, "log_likelihood")
            assert_close(as_float_list(loglik), expected["loglik"], rtol=RTOL_COEF, path="loglik")
        elif aspect == "score":
            assert_close(_attr(fit, "score_test"), expected["score"], rtol=RTOL_VAR, path="score")
        elif aspect == "iter":
            assert_exact([_attr(fit, "iterations")], expected["iter"][:1], path="iter")
        elif aspect == "wald_test":
            assert_close(
                _r_wald_test(fit, _case_init(case)),
                expected["wald_test"],
                rtol=RTOL_VAR,
                path="wald_test",
            )
        elif aspect == "n":
            assert_exact(r.nobs(fit), expected["n"], path="n (nobs)")
        elif aspect == "nevent":
            assert_exact(_attr(r.model_summary(fit), "n_event"), expected["nevent"], path="nevent")
        elif aspect == "means":
            assert_close(
                as_float_list(_attr(fit, "means")), expected["means"], rtol=RTOL_COEF, path="means"
            )
        elif aspect == "nocenter":
            assert_close(
                as_float_list(_attr(fit, "nocenter")), expected["nocenter"], path="nocenter"
            )
        elif aspect == "linear_predictors":
            assert_close(
                as_float_list(_attr(fit, "linear_predictors")),
                expected["linear_predictors"],
                rtol=RTOL_COEF,
                path="linear_predictors",
            )
        elif aspect == "x":
            x = _attr(fit, "x")
            assert_matrix_close(x, expected["x"]["values"], rtol=RTOL_COEF, path="x")
        elif aspect.startswith("residuals."):
            _check_cox_residual(fit, aspect.split(".", 1)[1], _expect(case, aspect))
        elif aspect.startswith("summary."):
            _check_summary(fit, aspect, _expect(case, aspect), _case_init(case))
        elif aspect.startswith("concordance."):
            _check_concordance_result(
                _concordance_of_fit(fit), _expect(case, "concordance"), aspect
            )
        elif aspect == "wtest":
            wtest = _expect(case, "wtest")
            actual = r.coxph_wtest(r.vcov(fit), r.coef(fit))
            assert_close(_attr(actual, "test"), wtest["test"], rtol=RTOL_VAR, path="wtest.test")
            assert_exact(_attr(actual, "df"), wtest["df"], path="wtest.df")
            assert_close(
                as_float_list(_attr(actual, "solve")),
                wtest["solve"],
                rtol=RTOL_VAR,
                path="wtest.solve",
            )
        elif aspect == "anova":
            _check_anova(r.anova(fit), _expect(case, "anova"))
        elif aspect.startswith("curves."):
            _check_cox_curves(r.survfit(fit), expected["survfit"], aspect)
        else:
            raise UnsupportedCaseError(f"unhandled coxph aspect {aspect}")

    def _check_anova_nested(self, case):
        data = case_data(self.topic, case)
        fits = [r.coxph(formula, data) for formula in case["formulas"]]
        _check_anova(r.anova(*fits), case["expected"]["anova"], nested=True)


def _check_anova(result: Any, expected: Mapping[str, Any], nested: bool = False) -> None:
    rows = _attr(result, "rows", "models")
    loglik = [_attr(row, "loglik") for row in rows]
    assert_close(loglik, expected["loglik"], rtol=RTOL_COEF, path="anova.loglik")
    chisq = [getattr(row, "chisq", None) for row in rows]
    df = [getattr(row, "df", None) for row in rows]
    p = [getattr(row, "p_value", getattr(row, "p", None)) for row in rows]
    assert_close(chisq[1:], expected["chisq"][1:], rtol=RTOL_VAR, path="anova.chisq")
    assert_close(p[1:], expected["p"][1:], rtol=RTOL_VAR, path="anova.p")
    if not nested:
        assert_exact(df[1:], [int(v) for v in expected["df"][1:]], path="anova.df")


def _check_cox_residual(fit: Any, kind: str, expected: Any) -> None:
    actual = r.residuals(fit, type=kind)
    rtol = RTOL_COEF
    if kind in ("schoenfeld", "scaledsch"):
        assert_matrix_close(actual, expected["values"], rtol=rtol, path=f"residuals.{kind}")
        return
    if expected and isinstance(expected[0], list):
        assert_matrix_close(actual, expected, rtol=rtol, path=f"residuals.{kind}")
    else:
        assert_close(as_float_list(actual), expected, rtol=rtol, path=f"residuals.{kind}")


class CoxphPredictHandler(TopicHandler):
    topic = "coxph_predict"

    def aspects(self, case):
        expected = case["expected"]
        out = []
        for key, value in expected.items():
            if key in ("coef", "newdata"):
                continue
            if key.startswith("survfit"):
                if is_r_error(value):
                    out.append(key)
                else:
                    out.extend(f"{key}:{aspect}" for aspect in _curve_aspects(value))
            elif key.startswith("predict"):
                out.extend(f"{key}.{sub}" for sub in value)
            else:
                out.append(key)
        return out

    def check(self, case, aspect):
        fit = _coxph_fit("coxph", case)
        expected = case["expected"]
        newdata = newdata_frame(expected)
        if aspect.startswith("basehaz_"):
            centered = aspect == "basehaz_centered"
            bh = r.basehaz(fit, centered=centered)
            exp = _expect(case, aspect)
            assert_close(as_float_list(_attr(bh, "time")), exp["time"], path="basehaz.time")
            assert_close(
                as_float_list(_attr(bh, "hazard")),
                exp["hazard"],
                rtol=RTOL_COEF,
                path="basehaz.hazard",
            )
            if "strata" in exp:
                labels = _attr(bh, "strata_labels", "strata")
                if labels is None or [str(v) for v in labels] != [
                    r_strata_value(v) for v in exp["strata"]
                ]:
                    raise FixtureMismatchError(
                        f"basehaz strata labels differ: {labels} vs {exp['strata'][:3]}"
                    )
        elif aspect.startswith("survfit"):
            key, curve_aspect = aspect.split(":", 1)
            exp = _expect(case, key)
            kwargs: dict[str, Any] = {}
            if key.endswith("censor_false"):
                kwargs["censor"] = False
            elif key.endswith("stype2"):
                kwargs.update(stype=2, ctype=1)
            elif key.endswith("ctype2"):
                kwargs.update(stype=2, ctype=2)
            if key.startswith("survfit_newdata"):
                kwargs["newdata"] = newdata
                if key.endswith("loglog"):
                    kwargs["conf_type"] = "log-log"
            result = r.survfit(fit, **kwargs)
            _check_cox_curves(result, exp, curve_aspect)
        elif aspect.startswith("predict"):
            key, kind = aspect.split(".", 1)
            exp = _expect(case, aspect)
            nd = newdata if key == "predict_newdata" else None
            if kind == "lp_uncentered":
                actual = r.predict(fit, nd, type="lp", reference="zero")
                assert_close(as_float_list(actual), exp, rtol=RTOL_COEF, path=aspect)
                return
            result = r.predict(fit, nd, type=kind, se_fit=True)
            fit_values = _attr(result, "fit")
            se_values = _attr(result, "se_fit")
            rtol = RTOL_COEF
            if kind == "terms":
                assert_matrix_close(fit_values, exp["fit"], rtol=rtol, path=f"{aspect}.fit")
                assert_matrix_close(
                    se_values, exp["se_fit"], rtol=RTOL_VAR, path=f"{aspect}.se_fit"
                )
            else:
                assert_close(as_float_list(fit_values), exp["fit"], rtol=rtol, path=f"{aspect}.fit")
                assert_close(
                    as_float_list(se_values), exp["se_fit"], rtol=RTOL_VAR, path=f"{aspect}.se_fit"
                )
        else:
            raise UnsupportedCaseError(f"unhandled coxph_predict aspect {aspect}")


def _check_cox_curves(result: Any, exp: Mapping[str, Any], curve_aspect: str) -> None:
    """Compare a Cox survfit result with R curves.

    R stores one block per stratum; inside a block a field is a vector (one
    curve) or a matrix with one column per newdata row.  Python stores one
    vector per (stratum, newdata row) curve; both are flattened to the same
    strata-major order before comparing.
    """

    fields = _CURVE_ASPECTS[curve_aspect]
    for field in fields:
        if not any(field in curve for curve in exp["curves"]):
            continue
        candidates, rtol = _CURVE_FIELDS[field]
        value = _attr(result, *candidates)
        if value is None:
            raise FixtureMismatchError(f"survfit.{field} is None")
        if hasattr(value, "tolist"):
            value = value.tolist()
        expected_columns: list[list[float]] = []
        for curve in exp["curves"]:
            column_values = curve.get(field)
            if column_values is None:
                continue
            if column_values and isinstance(column_values[0], list):
                expected_columns.extend(transpose(column_values))
            else:
                expected_columns.append(list(column_values))
        if value and isinstance(value[0], (list, tuple)):
            python_curves = [list(item) for item in value]
            if len(python_curves) != len(expected_columns) and len(python_curves[0]) == len(
                expected_columns
            ):
                python_curves = transpose(python_curves)
        else:
            python_curves = [list(value)]
        if len(python_curves) != len(expected_columns):
            raise FixtureMismatchError(
                f"survfit.{field}: {len(python_curves)} python curves, "
                f"R has {len(expected_columns)}"
            )
        for idx, (actual, expected) in enumerate(zip(python_curves, expected_columns, strict=True)):
            assert_close(as_float_list(actual), expected, rtol=rtol, path=f"curve[{idx}].{field}")


class CoxphDiagnosticsHandler(TopicHandler):
    topic = "coxph_diagnostics"

    def aspects(self, case):
        expected = case["expected"]
        out = [
            f"residuals.{sub}"
            for sub, value in expected["residuals"].items()
            if not is_r_error(value)
        ]
        for key, value in expected["zph"].items():
            if is_r_error(value):
                continue
            out.extend(f"zph.{key}.{sub}" for sub in ("table", "x", "y", "var"))
        if not is_r_error(expected["detail"]):
            out.extend(f"detail.{sub}" for sub in expected["detail"] if sub != "strata")
        return out

    def check(self, case, aspect):
        fit = _coxph_fit("coxph", case)
        parts = aspect.split(".")
        if parts[0] == "residuals":
            _check_cox_residual(fit, parts[1], _expect(case, aspect))
        elif parts[0] == "zph":
            transform, terms = parts[1].rsplit("_", 1)
            exp = _expect(case, f"zph.{parts[1]}")
            z = r.cox_zph(fit, transform=transform, terms=(terms == "terms"))
            sub = parts[2]
            if sub == "table":
                table = _attr(z, "table")
                rows = {row["name"]: row for row in table}
                for name, values in zip(
                    exp["table"]["rownames"], exp["table"]["values"], strict=True
                ):
                    if name not in rows:
                        raise FixtureMismatchError(f"zph table has no row {name!r} ({list(rows)})")
                    row = rows[name]
                    assert_close(row["chisq"], values[0], rtol=RTOL_VAR, path=f"zph[{name}].chisq")
                    assert_exact(row["df"], values[1], path=f"zph[{name}].df")
                    assert_close(row["p"], values[2], rtol=RTOL_VAR, path=f"zph[{name}].p")
            elif sub == "x":
                assert_close(as_float_list(_attr(z, "x")), exp["x"], rtol=RTOL_COEF, path="zph.x")
            elif sub == "y":
                assert_matrix_close(_attr(z, "y"), exp["y"], rtol=RTOL_COEF, path="zph.y")
            elif sub == "var":
                assert_matrix_close(_attr(z, "var"), exp["var"], rtol=RTOL_VAR, path="zph.var")
        elif parts[0] == "detail":
            exp = _expect(case, "detail")
            detail = r.coxph_detail(fit)
            sub = parts[1]
            mapping = {
                "time": (("time",), 0.0),
                "nevent": (("nevent", "n_event"), 0.0),
                "nrisk": (("nrisk", "n_risk"), 0.0),
                "hazard": (("hazard",), RTOL_COEF),
                "varhaz": (("varhaz", "var_hazard"), RTOL_VAR),
                "wtrisk": (("wtrisk",), RTOL_COEF),
                "score": (("score",), RTOL_COEF),
                "means": (("means",), RTOL_COEF),
                "imat": (("imat",), RTOL_VAR),
            }
            candidates, rtol = mapping[sub]
            actual = _attr(detail, *candidates)
            expected = exp[sub]
            if sub == "imat":
                # R: nvar x nvar x ntime; Python: per time list of matrices
                if isinstance(expected[0], list) and isinstance(expected[0][0], list):
                    # one R array, compared layer by layer: scale the
                    # absolute floor to the whole of it
                    atol = rtol * array_scale(expected)
                    for t, layer in enumerate(expected):
                        assert_matrix_close(
                            actual[t], layer, rtol=rtol, atol=atol, path=f"detail.imat[{t}]"
                        )
                else:
                    assert_close(
                        as_float_list(
                            [
                                m[0][0] if isinstance(m, list) and isinstance(m[0], list) else m
                                for m in actual
                            ]
                        ),
                        expected,
                        rtol=rtol,
                        path="detail.imat",
                    )
            elif expected and isinstance(expected[0], list):
                assert_matrix_close(actual, expected, rtol=rtol, path=f"detail.{sub}")
            else:
                assert_close(as_float_list(actual), expected, rtol=rtol, path=f"detail.{sub}")


class CoxphPenalizedHandler(TopicHandler):
    topic = "coxph_penalized"

    def aspects(self, case):
        expected = case["expected"]
        out = []
        for key, value in expected.items():
            if key in ("method", "penalty", "pterms", "nocenter", "coef_names"):
                continue
            if key == "residuals":
                out.extend(f"residuals.{sub}" for sub in value)
            elif key == "survfit":
                out.extend(_curve_aspects(value)) if not is_r_error(value) else out.append(key)
            elif key == "concordance":
                out.append("concordance.concordance")
            else:
                out.append(key)
        return out

    def check(self, case, aspect):
        fit = _coxph_fit(self.topic, case)
        expected = case["expected"]
        if aspect == "coef":
            assert_named_values(_coef_names(fit), r.coef(fit), expected["coef"], path="coef")
        elif aspect == "var":
            assert_matrix_close(r.vcov(fit), expected["var"], rtol=RTOL_VAR, path="var")
        elif aspect == "var2":
            assert_matrix_close(_attr(fit, "var2"), expected["var2"], rtol=RTOL_VAR, path="var2")
        elif aspect == "loglik":
            assert_close(
                as_float_list(_attr(fit, "log_likelihood")),
                expected["loglik"],
                rtol=RTOL_COEF,
                path="loglik",
            )
        elif aspect == "iter":
            assert_exact(as_float_list(_attr(fit, "iterations")), expected["iter"], path="iter")
        elif aspect == "wald_test":
            assert_close(
                _r_wald_test(fit, _case_init(case)),
                expected["wald_test"],
                rtol=RTOL_VAR,
                path="wald_test",
            )
        elif aspect in ("df", "df2"):
            assert_close(
                as_float_list(_attr(fit, aspect)), expected[aspect], rtol=RTOL_VAR, path=aspect
            )
        elif aspect == "history":
            history = _attr(fit, "history")
            for (name, exp_item), item in zip(expected["history"].items(), history, strict=True):
                assert_close(
                    as_float_list(_attr(item, "theta")),
                    exp_item["theta"],
                    rtol=RTOL_VAR,
                    path=f"history[{name}].theta",
                )
        elif aspect in ("frail", "fvar"):
            assert_close(
                as_float_list(_attr(fit, aspect)), expected[aspect], rtol=RTOL_COEF, path=aspect
            )
        elif aspect == "means":
            assert_close(
                as_float_list(_attr(fit, "means")), expected["means"], rtol=RTOL_COEF, path="means"
            )
        elif aspect == "linear_predictors":
            assert_close(
                as_float_list(_attr(fit, "linear_predictors")),
                expected["linear_predictors"],
                rtol=RTOL_COEF,
                path="linear_predictors",
            )
        elif aspect in ("n", "nevent", "score"):
            assert_close(
                _attr(fit, {"n": "n", "nevent": "nevent", "score": "score_test"}[aspect]),
                expected[aspect],
                rtol=RTOL_VAR,
                path=aspect,
            )
        elif aspect.startswith("residuals."):
            _check_cox_residual(fit, aspect.split(".", 1)[1], _expect(case, aspect))
        elif aspect == "concordance.concordance":
            _check_concordance_result(
                _concordance_of_fit(fit), _expect(case, "concordance"), aspect
            )
        elif aspect in ("predict_lp", "predict_risk"):
            kind = aspect.split("_")[1]
            assert_close(
                as_float_list(r.predict(fit, type=kind)),
                _expect(case, aspect),
                rtol=RTOL_COEF,
                path=aspect,
            )
        elif aspect.startswith("curves."):
            _check_cox_curves(r.survfit(fit), _expect(case, "survfit"), aspect)
        elif aspect == "basehaz_centered":
            exp = _expect(case, aspect)
            bh = r.basehaz(fit, centered=True)
            assert_close(
                as_float_list(_attr(bh, "hazard")), exp["hazard"], rtol=RTOL_COEF, path="basehaz"
            )
        else:
            raise UnsupportedCaseError(f"unhandled penalized aspect {aspect}")


# --- survreg ----------------------------------------------------------------


def _survreg_fit(topic: str, case: Mapping[str, Any]) -> Any:
    def build():
        data = case_data(topic, case)
        kwargs = _kwargs(case.get("args", {}), data)
        return r.survreg(case["formula"], data, **kwargs)

    return _cached(_fit_key("survreg", case), build)


class SurvregHandler(TopicHandler):
    topic = "survreg"

    def aspects(self, case):
        expected = case["expected"]
        if case["name"] == "distribution_functions":
            return [f"dist.{key}" for key in expected]
        out = []
        for key, value in expected.items():
            if key in ("dist", "newdata", "means", "n", "coef_names"):
                continue
            if key == "coef":
                out.extend(["coef", "coef_names"])
            elif key in ("residuals", "predict", "predict_newdata"):
                out.extend(f"{key}.{sub}" for sub, item in value.items() if not is_r_error(item))
            elif key == "concordance":
                out.append("concordance.concordance")
            else:
                out.append(key)
        return out

    def check(self, case, aspect):
        expected = case["expected"]
        if aspect.startswith("dist."):
            key = aspect.split(".", 1)[1]
            _check_survreg_distribution(key, expected[key])
            return
        fit = _survreg_fit(self.topic, case)
        if aspect == "coef":
            assert_named_values(_coef_names(fit), r.coef(fit), expected["coef"], path="coef")
        elif aspect == "coef_names":
            assert_named_values(
                _coef_names(fit), r.coef(fit), expected["coef"], check_names=True, path="coef"
            )
        elif aspect == "icoef":
            assert_close(
                as_float_list(_attr(fit, "icoef")), expected["icoef"], rtol=RTOL_COEF, path="icoef"
            )
        elif aspect == "scale":
            assert_close(
                as_float_list(_attr(fit, "scale")), expected["scale"], rtol=RTOL_COEF, path="scale"
            )
        elif aspect == "var":
            assert_matrix_close(r.vcov(fit), expected["var"], rtol=RTOL_VAR, path="var")
        elif aspect == "naive_var":
            assert_matrix_close(
                _attr(fit, "naive_var", "naive_variance"),
                expected["naive_var"],
                rtol=RTOL_VAR,
                path="naive_var",
            )
        elif aspect == "loglik":
            loglik = _attr(fit, "log_likelihood")
            if not isinstance(loglik, (list, tuple)):
                loglik = [math.nan, loglik]
            assert_close(as_float_list(loglik), expected["loglik"], rtol=RTOL_COEF, path="loglik")
        elif aspect == "iter":
            assert_exact(_attr(fit, "iterations"), expected["iter"], path="iter")
        elif aspect == "df":
            assert_exact(r.degrees_freedom(fit), expected["df"], path="df")
        elif aspect == "df_residual":
            assert_exact(r.df_residual(fit), expected["df_residual"], path="df_residual")
        elif aspect == "parms":
            assert_close(
                as_float_list(_attr(fit, "distribution_parameters", "parms")),
                expected["parms"],
                rtol=RTOL_COEF,
                path="parms",
            )
        elif aspect == "linear_predictors":
            assert_close(
                as_float_list(_attr(fit, "linear_predictors")),
                expected["linear_predictors"],
                rtol=RTOL_COEF,
                path="linear_predictors",
            )
        elif aspect == "x":
            assert_matrix_close(_attr(fit, "x"), expected["x"]["values"], rtol=RTOL_COEF, path="x")
        elif aspect.startswith("residuals."):
            kind = aspect.split(".", 1)[1]
            exp = _expect(case, aspect)
            actual = r.residuals(fit, type=kind)
            if isinstance(exp, Mapping):
                assert_matrix_close(actual, exp["values"], rtol=RTOL_COEF, path=aspect)
            else:
                assert_close(as_float_list(actual), exp, rtol=RTOL_COEF, path=aspect)
        elif aspect.startswith("predict"):
            key, kind = aspect.split(".", 1)
            exp = _expect(case, aspect)
            nd = newdata_frame(expected) if key == "predict_newdata" else None
            kwargs: dict[str, Any] = {"type": kind, "se_fit": True}
            if kind in ("quantile", "uquantile"):
                kwargs["p"] = exp["p"]
            result = r.predict(fit, nd, **kwargs)
            fit_values = _attr(result, "fit")
            se_values = _attr(result, "se_fit")
            if isinstance(exp["fit"][0], list):
                assert_matrix_close(fit_values, exp["fit"], rtol=RTOL_COEF, path=f"{aspect}.fit")
                assert_matrix_close(
                    se_values, exp["se_fit"], rtol=RTOL_VAR, path=f"{aspect}.se_fit"
                )
            else:
                assert_close(
                    as_float_list(fit_values), exp["fit"], rtol=RTOL_COEF, path=f"{aspect}.fit"
                )
                assert_close(
                    as_float_list(se_values), exp["se_fit"], rtol=RTOL_VAR, path=f"{aspect}.se_fit"
                )
        elif aspect == "summary":
            exp = _expect(case, "summary")
            summary = r.model_summary(fit)
            rows = _attr(summary, "coefficients")
            cols = exp["table"]["colnames"]
            for row, values in zip(rows, exp["table"]["values"], strict=True):
                assert_close(
                    row["coef"],
                    values[cols.index("Value")],
                    rtol=RTOL_COEF,
                    path=f"summary[{row['name']}].value",
                )
                assert_close(
                    row["se"],
                    values[cols.index("Std. Error")],
                    rtol=RTOL_VAR,
                    path=f"summary[{row['name']}].se",
                )
                assert_close(
                    row["p"],
                    values[cols.index("p")],
                    rtol=RTOL_VAR,
                    path=f"summary[{row['name']}].p",
                )
        elif aspect == "anova":
            exp = _expect(case, "anova")
            result = r.anova(fit)
            rows = _attr(result, "models", "rows")
            loglik = [_attr(row, "loglik") for row in rows]
            assert_close([-2 * v for v in loglik], exp["loglik"], rtol=RTOL_COEF, path="anova.-2LL")
        elif aspect == "concordance.concordance":
            _check_concordance_result(
                _concordance_of_fit(fit), _expect(case, "concordance"), aspect
            )
        else:
            raise UnsupportedCaseError(f"unhandled survreg aspect {aspect}")


def _check_survreg_distribution(key: str, exp: Mapping[str, Any]) -> None:
    """dsurvreg/psurvreg/qsurvreg samples; ``key`` is ``<dist>_scale<s>`` or ``t_df4_scale1``."""

    dist = "t" if key.startswith("t_df4") else key.rpartition("_scale")[0]
    kwargs: dict[str, Any] = {"mean": exp["mean"], "scale": exp["scale"], "distribution": dist}
    if dist == "t":
        kwargs["parms"] = 4
    assert_close(
        as_float_list(r.dsurvreg(exp["x"], **kwargs)), exp["d"], rtol=RTOL_COEF, path=f"{key}.d"
    )
    assert_close(
        as_float_list(r.psurvreg(exp["x"], **kwargs)), exp["p_"], rtol=RTOL_COEF, path=f"{key}.p"
    )
    assert_close(
        as_float_list(r.qsurvreg(exp["p"], **kwargs)), exp["q"], rtol=RTOL_COEF, path=f"{key}.q"
    )


# --- concordance ------------------------------------------------------------


class ConcordanceHandler(TopicHandler):
    topic = "concordance"

    def aspects(self, case):
        expected = case["expected"]
        if case["name"] == "coxph_survreg_fits":
            return [f"{fit}.concordance" for fit in expected]
        return [key for key in expected if key != "timewt"]

    def check(self, case, aspect):
        expected = case["expected"]
        if case["name"] == "coxph_survreg_fits":
            data = case_data(self.topic, case)
            key = aspect.split(".")[0]
            if key == "coxph":
                cc = _concordance_of_fit(r.coxph(case["formula"], data))
            elif key == "coxph_timewt_S":
                cc = _concordance_of_fit(r.coxph(case["formula"], data), timewt="S")
            elif key == "survreg":
                cc = _concordance_of_fit(r.survreg(case["formula"], data))
            else:
                raise UnsupportedCaseError("concordance of several fits at once")
            _check_concordance_result(cc, expected[key], aspect)
            return

        def build():
            data = case_data(self.topic, case)
            kwargs = _kwargs(case.get("args", {}), data)
            return r.concordance(case["formula"], data, **kwargs)

        cc = _cached(_fit_key("concordance", case), build)
        _check_concordance_result(cc, expected, f"x.{aspect}")


# --- aareg --------------------------------------------------------------------


class AaregHandler(TopicHandler):
    topic = "aareg"

    def aspects(self, case):
        return [key for key in case["expected"] if key not in ("test", "n")]

    def check(self, case, aspect):
        expected = case["expected"]

        def build():
            data = case_data(self.topic, case)
            kwargs = _kwargs(case.get("args", {}), data)
            if "dfbeta" in expected:
                kwargs["dfbeta"] = True
            return r.aareg(case["formula"], data, **kwargs)

        fit = _cached(_fit_key("aareg", case), build)
        if aspect == "times":
            assert_close(as_float_list(_attr(fit, "times")), expected["times"], path="times")
        elif aspect == "nrisk":
            assert_exact(
                as_float_list(_attr(fit, "nrisk", "n_risk")), expected["nrisk"], path="nrisk"
            )
        elif aspect == "coefficient":
            assert_matrix_close(
                _attr(fit, "coefficient", "coefficients"),
                expected["coefficient"]["values"],
                rtol=RTOL_COEF,
                path="coefficient",
            )
        elif aspect == "test_statistic":
            assert_close(
                as_float_list(_attr(fit, "test_statistic")),
                list(expected["test_statistic"].values()),
                rtol=RTOL_VAR,
                path="test_statistic",
            )
        elif aspect == "test_var":
            assert_matrix_close(
                _attr(fit, "test_var"), expected["test_var"], rtol=RTOL_VAR, path="test_var"
            )
        elif aspect == "test_var2":
            assert_matrix_close(
                _attr(fit, "test_var2"), expected["test_var2"], rtol=RTOL_VAR, path="test_var2"
            )
        elif aspect == "tweight":
            actual = _attr(fit, "tweight")
            if isinstance(expected["tweight"][0], list):
                assert_matrix_close(actual, expected["tweight"], rtol=RTOL_COEF, path="tweight")
            else:
                assert_close(
                    as_float_list(actual), expected["tweight"], rtol=RTOL_COEF, path="tweight"
                )
        elif aspect == "chisq":
            exp = _expect(case, "chisq")
            assert_close(_attr(fit, "chisq"), exp["chisq"], rtol=RTOL_VAR, path="chisq")
        elif aspect == "dfbeta":
            # Python: subjects x nvar x times (R's array layout); R fixture: one
            # subjects x nvar matrix per time.
            actual = _attr(fit, "dfbeta")
            for t, layer in enumerate(expected["dfbeta"]):
                actual_layer = [[row[k][t] for k in range(len(row))] for row in actual]
                assert_matrix_close(actual_layer, layer, rtol=RTOL_COEF, path=f"dfbeta[{t}]")
        else:
            raise UnsupportedCaseError(f"unhandled aareg aspect {aspect}")


# --- cch ----------------------------------------------------------------------


class CchHandler(TopicHandler):
    topic = "cch"

    def aspects(self, case):
        return [
            key
            for key in ("coef", "var", "naive_var", "subcohort_size")
            if case["expected"].get(key) is not None
        ]

    def check(self, case, aspect):
        expected = case["expected"]

        def build():
            data = case_data(self.topic, case)
            args = case["args"]
            kwargs = {
                "subcoh": args["subcoh"],
                "id": args["id"],
                "method": args["method"],
                "na_action": "omit",
            }
            size = args["cohort.size"]
            kwargs["cohort_size"] = size if not isinstance(size, Mapping) else dict(size)
            if "stratum" in args:
                kwargs["stratum"] = args["stratum"]
            return r.cch(case["formula"], data, **kwargs)

        fit = _cached(_fit_key("cch", case), build)
        if aspect == "coef":
            assert_named_values(_coef_names(fit), r.coef(fit), expected["coef"], path="coef")
        elif aspect == "var":
            assert_matrix_close(r.vcov(fit), expected["var"], rtol=RTOL_VAR, path="var")
        elif aspect == "naive_var":
            assert_matrix_close(
                _attr(fit, "naive_var", "naive_variance"),
                expected["naive_var"],
                rtol=RTOL_VAR,
                path="naive_var",
            )
        elif aspect == "subcohort_size":
            assert_exact(
                as_float_list(_attr(fit, "subcohort_size")),
                expected["subcohort_size"],
                path="subcohort_size",
            )


# --- clogit -------------------------------------------------------------------


class ClogitHandler(TopicHandler):
    topic = "clogit"

    def aspects(self, case):
        return ["coef", "var", "loglik", "iter", "linear_predictors", "summary.coefficients"]

    def check(self, case, aspect):
        expected = case["expected"]

        def build():
            data = case_data(self.topic, case)
            return r.clogit(case["formula"], data, method=case["args"]["method"], na_action="omit")

        fit = _cached(_fit_key("clogit", case), build)
        if aspect == "coef":
            assert_named_values(_coef_names(fit), r.coef(fit), expected["coef"], path="coef")
        elif aspect == "var":
            assert_matrix_close(r.vcov(fit), expected["var"], rtol=RTOL_VAR, path="var")
        elif aspect == "loglik":
            assert_close(
                as_float_list(_attr(fit, "log_likelihood")),
                expected["loglik"],
                rtol=RTOL_COEF,
                path="loglik",
            )
        elif aspect == "iter":
            assert_exact(_attr(fit, "iterations"), expected["iter"], path="iter")
        elif aspect == "linear_predictors":
            assert_close(
                as_float_list(_attr(fit, "linear_predictors")),
                expected["linear_predictors"],
                rtol=RTOL_COEF,
                path="linear_predictors",
            )
        elif aspect == "summary.coefficients":
            _check_summary(fit, aspect, _expect(case, aspect))


# --- finegray -----------------------------------------------------------------


class FinegrayHandler(TopicHandler):
    topic = "finegray"

    def aspects(self, case):
        out = ["frame"]
        if "coxph" in case["expected"]:
            out.extend(["coxph.coef", "coxph.loglik"])
        return out

    def check(self, case, aspect):
        expected = case["expected"]

        def build():
            data = case_data(self.topic, case)
            kwargs = _kwargs(case.get("args", {}), data, na_omit=False)
            return r.finegray(_mstate_formula(case["formula"], data), data, **kwargs)

        frame = _cached(_fit_key("finegray", case), build)
        r_frame = decode_frame(expected["frame"])
        if aspect == "frame":
            if nrow(frame) != nrow(r_frame):
                raise FixtureMismatchError(f"finegray rows {nrow(frame)} != {nrow(r_frame)}")
            for name, values in r_frame.items():
                if name not in frame:
                    raise FixtureMismatchError(
                        f"finegray frame lacks column {name!r} ({list(frame)})"
                    )
                actual = list(frame[name])
                if isinstance(values, RFactor) or (values and isinstance(values[0], str)):
                    if [str(v) for v in actual] != [str(v) for v in values]:
                        raise FixtureMismatchError(f"column {name} differs")
                else:
                    assert_close(
                        as_float_list(actual), values, rtol=RTOL_COEF, path=f"frame.{name}"
                    )
        else:
            fit = r.coxph(expected["cox_formula"], frame, weights="fgwt")
            if aspect == "coxph.coef":
                assert_named_values(
                    _coef_names(fit), r.coef(fit), expected["coxph"]["coef"], path="coef"
                )
            else:
                assert_close(
                    as_float_list(_attr(fit, "log_likelihood")),
                    expected["coxph"]["loglik"],
                    rtol=RTOL_COEF,
                    path="loglik",
                )


# --- survobrien ---------------------------------------------------------------


class SurvobrienHandler(TopicHandler):
    topic = "survobrien"

    def aspects(self, case):
        return ["frame", "coxph_coef"]

    def check(self, case, aspect):
        expected = case["expected"]
        data = case_data(self.topic, case)
        frame = r.survobrien(case["formula"], data=data)
        r_frame = decode_frame(expected["frame"])
        if aspect == "frame":
            for name in (".strata.", ".id."):
                if name not in frame:
                    raise FixtureMismatchError(f"survobrien frame lacks {name}: {list(frame)}")
            for name, values in r_frame.items():
                if name not in frame:
                    continue
                if values and isinstance(values[0], str):
                    continue
                assert_close(
                    as_float_list(frame[name]), values, rtol=RTOL_COEF, path=f"frame.{name}"
                )
        else:
            fit = r.coxph(expected["cox_formula"], frame)
            assert_named_values(_coef_names(fit), r.coef(fit), expected["coxph_coef"], path="coef")


# --- yates --------------------------------------------------------------------


class YatesHandler(TopicHandler):
    topic = "yates"

    def aspects(self, case):
        return ["estimate", "test"]

    def check(self, case, aspect):
        raise UnsupportedCaseError(
            "yates(fit, term, population=...) formula interface is not available"
        )


# --- royston / brier ----------------------------------------------------------


class RoystonBrierHandler(TopicHandler):
    topic = "royston_brier"

    def aspects(self, case):
        return list(case["expected"])

    def check(self, case, aspect):
        expected = case["expected"]
        fit = _coxph_fit(self.topic, {**case, "args": {}})
        if aspect.startswith("royston"):
            result = r.royston(fit, adjust=aspect.endswith("adjust"))
            exp = expected[aspect]
            for name, value in exp.items():
                if name not in result:
                    raise FixtureMismatchError(f"royston result lacks {name!r}")
                assert_close(result[name], value, rtol=RTOL_VAR, path=f"{aspect}.{name}")
        else:
            exp = expected[aspect]
            kwargs: dict[str, Any] = {}
            if aspect != "brier_default_times":
                kwargs["times"] = exp["times"]
            if aspect == "brier_ties_false":
                kwargs["ties"] = False
            result = r.brier(fit, **kwargs)
            assert_close(as_float_list(result["times"]), exp["times"], path=f"{aspect}.times")
            assert_close(
                as_float_list(result["brier"]), exp["brier"], rtol=RTOL_VAR, path=f"{aspect}.brier"
            )
            assert_close(
                as_float_list(result["rsquared"]),
                exp["rsquared"],
                rtol=RTOL_VAR,
                path=f"{aspect}.rsquared",
            )


# --- rttright -----------------------------------------------------------------


class RttrightHandler(TopicHandler):
    topic = "rttright"

    def aspects(self, case):
        return ["weights"]

    def check(self, case, aspect):
        expected = case["expected"]
        data = case_data(self.topic, case)
        kwargs = _kwargs(case.get("args", {}), data, na_omit=False)
        result = r.rttright(_mstate_formula(case["formula"], data), data=data, **kwargs)
        if hasattr(result, "tolist"):
            result = result.tolist()
        if "times" in expected:
            assert_matrix_close(result, expected["weights"], rtol=RTOL_COEF, path="weights")
        else:
            assert_close(as_float_list(result), expected["weights"], rtol=RTOL_COEF, path="weights")


# --- pseudo / residuals.survfit / survfit0 -------------------------------------


class PseudoHandler(TopicHandler):
    topic = "pseudo"

    def aspects(self, case):
        out = []
        for key, value in case["expected"].items():
            if key == "times" or is_r_error(value):
                continue
            if key == "survfit0":
                out.extend(f"survfit0:{aspect}" for aspect in _curve_aspects(value))
            else:
                out.append(key)
        return out

    def check(self, case, aspect):
        expected = case["expected"]
        # residuals.survfit re-evaluates the model frame; keep it on the fit
        fit = _survfit_call(self.topic, case, drop=("times",), keep_model=True)
        times = expected["times"]
        if aspect.startswith("survfit0"):
            curve_aspect = aspect.split(":", 1)[1]
            result = r.survfit0(fit)
            _check_curves(result, expected["survfit0"], curve_aspect)
            return
        kind, type_name = aspect.split("_", 1)
        exp = expected[aspect]
        if kind == "pseudo":
            result = r.pseudo(fit, times=times, type=type_name)
        else:
            result = r.survfit_residuals(fit, times=times, type=type_name)
            if isinstance(result, Mapping):
                result = result["resid"]
        if hasattr(result, "tolist"):
            result = result.tolist()
        if exp and isinstance(exp[0], list) and isinstance(exp[0][0], list):
            # R: subjects x times x states
            for state, layer in enumerate(exp):
                for subject, row in enumerate(layer):
                    actual = result[subject]
                    if isinstance(actual[0], (list, tuple)):
                        actual_row = [actual[t][state] for t in range(len(times))]
                    else:
                        actual_row = actual[state * len(times) : (state + 1) * len(times)]
                    assert_close(
                        actual_row, row, rtol=RTOL_VAR, path=f"{aspect}[{subject}][state {state}]"
                    )
        elif exp and isinstance(exp[0], list):
            assert_matrix_close(result, exp, rtol=RTOL_VAR, path=aspect)
        else:
            assert_close(as_float_list(result), exp, rtol=RTOL_VAR, path=aspect)


# --- survcheck ----------------------------------------------------------------


class SurvcheckHandler(TopicHandler):
    topic = "survcheck"

    def aspects(self, case):
        return ["states", "transitions", "events", "flag", "istate", "n"]

    def check(self, case, aspect):
        expected = case["expected"]

        def build():
            data = case_data(self.topic, case)
            kwargs = _kwargs(case.get("args", {}), data, na_omit=False)
            return r.survcheck(_mstate_formula(case["formula"], data), data, **kwargs)

        result = _cached(_fit_key("survcheck", case), build)
        expected = case["expected"]
        if aspect in ("states", "istate", "events"):
            value = getattr(result, aspect, None)
            if value is None:
                raise UnsupportedCaseError(f"survcheck result has no {aspect}")
            if aspect == "states":
                assert_exact(list(value), expected["states"], path="states")
            elif aspect == "istate":
                assert_exact([str(v) for v in value], expected["istate"], path="istate")
            else:
                assert_matrix_close(value, expected["events"]["values"], rtol=0.0, path="events")
        elif aspect == "transitions":
            # Python codes states by the position in the event factor levels
            # (0 = the censoring level, doubling as the "(s0)" initial state).
            table = expected["transitions"]
            data = case_data(self.topic, case)
            status_column = case["formula"].split("~")[0].strip()[5:-1].split(",")[-1].strip()
            levels = list(getattr(data[status_column], "categories", ()))

            def code(name: str) -> int:
                return levels.index(name) if name in levels else 0

            r_counts: dict[str, float] = {}
            for row_name, row in zip(table["rownames"], table["values"], strict=True):
                for col_name, value in zip(table["colnames"], row, strict=True):
                    if col_name == "(censored)" or value == 0:
                        continue
                    r_counts[f"{code(row_name)} -> {code(col_name)}"] = value
            actual = _attr(result, "transitions")
            actual_counts = {key: float(value) for key, value in dict(actual).items() if value}
            if actual_counts != r_counts:
                raise FixtureMismatchError(f"transitions {actual_counts} != {r_counts}")
        elif aspect == "flag":
            for name, value in expected["flag"].items():
                rows = getattr(result, f"{name}_rows", None)
                if rows is None:
                    if name == "duplicate":
                        continue
                    raise UnsupportedCaseError(f"survcheck result has no {name}_rows")
                assert_exact(len(rows), value, path=f"flag.{name}")
        elif aspect == "n":
            actual = [
                _attr(result, "n_subjects"),
                _attr(result, "n_observations"),
                _attr(result, "n_transitions"),
            ]
            assert_exact(actual, list(expected["n"].values()), path="n")


# --- survSplit / survcondense --------------------------------------------------


def _compare_frames(
    actual: Mapping[str, Any],
    expected_frame: Mapping[str, Any],
    *,
    rtol: float = RTOL_COEF,
    columns: Sequence[str] | None = None,
) -> None:
    r_frame = decode_frame(expected_frame)
    if nrow(actual) != nrow(r_frame):
        raise FixtureMismatchError(f"rows {nrow(actual)} != {nrow(r_frame)}")
    for name, values in r_frame.items():
        if columns is not None and name not in columns:
            continue
        if name not in actual:
            raise FixtureMismatchError(f"frame lacks column {name!r} (has {list(actual)})")
        actual_values = list(actual[name])
        if isinstance(values, RFactor) or (values and isinstance(values[0], str)):
            if [str(v) for v in actual_values] != [str(v) for v in values]:
                raise FixtureMismatchError(f"column {name!r} differs")
        else:
            assert_close(as_float_list(actual_values), values, rtol=rtol, path=name)


class SurvSplitHandler(TopicHandler):
    topic = "survSplit"

    def aspects(self, case):
        return ["frame"]

    def check(self, case, aspect):
        data = case_data(self.topic, case)
        kwargs = _kwargs(case.get("args", {}), data, na_omit=False)
        frame = r.survSplit(case["formula"], data, **kwargs)
        _compare_frames(frame, case["expected"]["frame"])


class SurvcondenseHandler(TopicHandler):
    topic = "survcondense"

    def aspects(self, case):
        return ["frame"]

    def check(self, case, aspect):
        data = case_data(self.topic, case)
        kwargs = _kwargs(case.get("args", {}), data, na_omit=False)
        frame = r.survcondense(case["formula"], data, **kwargs)
        _compare_frames(frame, case["expected"]["frame"])


# --- tmerge -------------------------------------------------------------------


class TmergeHandler(TopicHandler):
    topic = "tmerge"

    def aspects(self, case):
        return [
            key for key in case["expected"] if key not in ("tcount", "tcount_final", "matches_cgd")
        ]

    def check(self, case, aspect):
        expected = case["expected"]
        if case["name"] == "cgd0_vignette":
            cgd0 = load_dataset("cgd0")
            base = {name: cgd0[name] for name in list(cgd0)[:13]}
            frame = r.tmerge(base, cgd0, id="id", tstop="futime")
            if aspect == "after_base":
                return _compare_frames(frame, expected["after_base"])
            for k in range(1, 8):
                frame = r.tmerge(frame, cgd0, id="id", infect=r.event(f"etime{k}"))
            if aspect == "after_events":
                return _compare_frames(frame, expected["after_events"])
            frame = r.tmerge(frame, frame, id="id", enum=r.cumtdc("tstart"))
            return _compare_frames(frame, expected["final"])
        doc_data = case_data(self.topic, {"data_ref": case["data_ref"]})
        long = decode_frame(load_topic(self.topic)["data"][case["data_ref2"]])
        if case["name"] == "pbcseq_20_vignette":
            death = [int(value == 2) for value in doc_data["status"]]
            frame = r.tmerge(doc_data, doc_data, id="id", death=r.event("time", death))
            frame = r.tmerge(
                frame,
                long,
                id="id",
                bili=r.tdc("day", "bili"),
                albumin=r.tdc("day", "albumin"),
                protime=r.tdc("day", "protime"),
                edema=r.tdc("day", "edema"),
            )
            return _compare_frames(frame, expected["frame"])
        d1 = r.tmerge(doc_data, doc_data, id="id", death=r.event("futime", "death"))
        if aspect == "step1_death_event":
            return _compare_frames(d1, expected[aspect])
        if aspect == "tdc_init":
            return _compare_frames(
                r.tmerge(d1, long, id="id", lab=r.tdc("time", "lab", init=0.5)), expected[aspect]
            )
        if aspect == "tdc_tdcstart":
            return _compare_frames(
                r.tmerge(d1, long, id="id", lab=r.tdc("time", "lab"), options={"tdcstart": -1}),
                expected[aspect],
            )
        d2 = r.tmerge(d1, long, id="id", lab=r.tdc("time", "lab"))
        if aspect == "step2_lab_tdc":
            return _compare_frames(d2, expected[aspect])
        d3 = r.tmerge(d2, long, id="id", nlab=r.cumtdc("time"))
        if aspect == "step3_nlab_cumtdc":
            return _compare_frames(d3, expected[aspect])
        d4 = r.tmerge(d3, long, id="id", infect=r.event("time", "infection"))
        if aspect == "step4_infect_event":
            return _compare_frames(d4, expected[aspect])
        d5 = r.tmerge(d4, long, id="id", ninfect=r.cumevent("time", "infection"))
        return _compare_frames(d5, expected["step5_ninfect_cumevent"])


# --- neardate -----------------------------------------------------------------


class NeardateHandler(TopicHandler):
    topic = "neardate"

    def aspects(self, case):
        return [key for key in case["expected"] if key != "after_dates"]

    def check(self, case, aspect):
        args = case["args"]
        best, _, nomatch = aspect.partition("_")
        kwargs: dict[str, Any] = {"best": best}
        if nomatch == "nomatch0":
            kwargs["nomatch"] = 0
        result = r.neardate(args["id1"], args["id2"], args["y1"], args["y2"], **kwargs)
        assert_exact(list(result), case["expected"][aspect], path=aspect)


# --- pyears -------------------------------------------------------------------


class PyearsHandler(TopicHandler):
    topic = "pyears"

    def aspects(self, case):
        expected = case["expected"]
        if case["name"] == "tcut_basis":
            return ["tcut"]
        return [key for key in ("pyears", "n", "event", "expected", "offtable") if key in expected]

    def check(self, case, aspect):
        expected = case["expected"]
        if case["name"] == "tcut_basis":
            args = case["args"]
            tc = r.tcut(args["x"], args["breaks"])
            assert_exact(as_float_list(_attr(tc, "values")), expected["values"], path="values")
            assert_exact(
                as_float_list(_attr(tc, "breaks", "cutpoints")),
                expected["cutpoints"],
                path="cutpoints",
            )
            return
        if "ratetable" in case.get("args", {}):
            raise UnsupportedCaseError("pyears with a ratetable/rmap is not available")
        if "tcut(" in case["formula"] or "cut(" in case["formula"]:
            raise UnsupportedCaseError("pyears formula with tcut()/cut() terms is not available")
        data = case_data(self.topic, case)
        kwargs = _kwargs(case.get("args", {}), data, na_omit=False)
        result = r.pyears(case["formula"], data, **kwargs)
        actual = _attr(result, aspect if aspect != "offtable" else "off_table")
        exp = expected[aspect]
        if isinstance(exp, list) and exp and isinstance(exp[0], list):
            assert_matrix_close(actual, exp, rtol=RTOL_COEF, path=aspect)
        else:
            assert_close(actual, exp, rtol=RTOL_COEF, path=aspect)


# --- survexp ------------------------------------------------------------------


class SurvexpHandler(TopicHandler):
    topic = "survexp"

    def aspects(self, case):
        name = case["name"]
        if name == "ratetableDate":
            return ["from_date", "from_numeric"]
        if name == "survexp_us_table":
            return ["dim", "sample"]
        if name == "lung_coxph_ratetable":
            return ["by_sex", "overall", "individual"]
        return ["surv"]

    def check(self, case, aspect):
        expected = case["expected"]
        name = case["name"]
        if name == "ratetableDate":
            args = case["args"]
            if aspect == "from_numeric":
                result = r.ratetableDate(args["numeric"])
            else:
                import datetime

                dates = [datetime.date.fromisoformat(value) for value in args["dates"]]
                result = r.ratetableDate(dates)
            assert_close(as_float_list(result), expected[aspect], rtol=1e-12, path=aspect)
            return
        if name == "survexp_us_table":
            table = r.survexp_us()
            if aspect == "dim":
                dims = _attr(table, "dim", "shape")
                assert_exact(list(dims), expected["dim"], path="dim")
            else:
                raise UnsupportedCaseError("rate table cell access by dimnames is not exposed")
            return
        if name == "lung_coxph_ratetable":
            raise UnsupportedCaseError("survexp with a coxph fit as ratetable is not available")
        args = case["args"]
        if args.get("ratetable") != "survexp.us" or "race" in args.get("rmap", ""):
            raise UnsupportedCaseError(
                "only survexp.us is supported by the direct survexp interface "
                f"({args.get('ratetable')})"
            )
        if "~ sex" in case["formula"]:
            raise UnsupportedCaseError("survexp grouped by a formula term is not available")
        data = case_data(self.topic, case)
        method = args.get("method", "ederer")
        if args.get("cohort") is False:
            result = r.survexp_individual(
                data["time"], data["agedays"], data["entry"], r.survexp_us(), sex=data["sex"]
            )
            assert_close(as_float_list(result), expected["surv"], rtol=RTOL_VAR, path="surv")
            return
        times = args.get("times")
        result = r.survexp(
            data["time"],
            data["agedays"],
            data["entry"],
            r.survexp_us(),
            sex=data["sex"],
            times=times,
            method=method,
        )
        assert_close(as_float_list(_attr(result, "time")), expected["time"], path="time")
        assert_close(
            as_float_list(_attr(result, "surv", "survival")),
            expected["surv"],
            rtol=RTOL_VAR,
            path="surv",
        )


# --- utilities ----------------------------------------------------------------


class UtilitiesHandler(TopicHandler):
    topic = "utilities"

    def aspects(self, case):
        return list(case["expected"])

    def check(self, case, aspect):
        name = case["name"]
        expected = case["expected"][aspect]
        args = case["args"]
        if name == "cipoisson":
            if aspect.startswith("scalar"):
                result = r.cipoisson(5) if aspect == "scalar_k5" else r.cipoisson(0, time=2)
                assert_close(list(result), expected, rtol=RTOL_VAR, path=aspect)
                return
            method, _, p = aspect.partition("_")
            result = r.cipoisson(
                args["k"], time=args["time"], p=0.90 if p == "p90" else 0.95, method=method
            )
            assert_matrix_close([list(row) for row in result], expected, rtol=RTOL_VAR, path=aspect)
        elif name == "bounded_links":
            x = args["x"]
            edge = 0.05
            if aspect.endswith("_linkinv"):
                raise UnsupportedCaseError("bounded link inverse functions are not exposed")
            link, _, suffix = aspect.partition("_")
            if suffix == "edge01":
                edge = 0.1
            elif suffix == "edge001":
                edge = 0.01
            fn = getattr(r, link)
            assert_close(as_float_list(fn(x, edge)), expected, rtol=RTOL_VAR, path=aspect)
        elif name == "nsk_basis":
            if aspect == "lung_age_df3":
                x = load_dataset("lung")["age"]
                basis = r.nsk(x, df=3)
            elif aspect == "df4":
                basis = r.nsk(args["x"], df=4)
            elif aspect == "knots_35_50_65":
                basis = r.nsk(args["x"], knots=[35, 50, 65], Boundary_knots=[20, 80])
            elif aspect == "df3_intercept":
                basis = r.nsk(args["x"], df=3, intercept=True)
            else:
                basis = r.nsk(args["x"], df=4, b=0.1)
            values = _basis_rows(basis)
            assert_matrix_close(values, expected["values"], rtol=RTOL_COEF, path=aspect)
            assert_close(
                as_float_list(_attr(basis, "knots")),
                expected["knots"],
                rtol=RTOL_COEF,
                path=f"{aspect}.knots",
            )
        elif name == "pspline_basis":
            x = load_dataset("lung")["age"]
            kwargs = {
                "df4": {"df": 4},
                "df4_nterm8": {"df": 4, "nterm": 8},
                "degree2_nterm6_df3": {"degree": 2, "nterm": 6, "df": 3},
                "df0": {"df": 0},
                "theta05": {"theta": 0.5},
            }[aspect]
            basis = r.pspline(x, **kwargs)
            values = _basis_rows(basis)
            assert_matrix_close(values, expected["values"], rtol=RTOL_COEF, path=aspect)
            assert_exact(_attr(basis, "nterm"), expected["nterm"], path=f"{aspect}.nterm")
        elif name == "aeqSurv":
            if aspect == "synthetic_timefix":
                data = case_data(self.topic, case)
                surv = r.Surv(data["time"], data["status"])
            elif aspect.startswith("right"):
                surv = r.Surv(args["time2"], args["status2"])
            else:
                surv = r.Surv(args["start3"], args["stop3"], args["status3"])
            tol = 1e-8 if aspect.endswith("tol_1e8") else None
            result = r.aeqSurv(surv, tolerance=tol) if tol else r.aeqSurv(surv)
            assert_matrix_close(_surv_matrix(result), expected["values"], rtol=1e-15, path=aspect)
        elif name == "surv_types":
            self._check_surv(aspect, expected)
        elif name == "statefig":
            connect = args["connect3"] if "1_2_1" not in aspect else args["connect4"]
            layout = [1, 2] if "1_2_1" not in aspect else [1, 2, 1]
            states = ["A", "B", "C"] if "1_2_1" not in aspect else ["A", "B", "C", "D"]
            result = r.statefig(layout, connect, states=states)
            coords = _attr(result, "coordinates", "positions", "xy")
            assert_matrix_close(coords, expected, rtol=RTOL_VAR, path=aspect)
        else:
            raise UnsupportedCaseError(f"unhandled utilities case {name}")

    @staticmethod
    def _check_surv(aspect: str, expected: Any) -> None:
        if aspect.startswith("format_"):
            kind = aspect.split("_", 1)[1]
            surv = {
                "right": lambda: r.Surv([1, 2, 3], [1, 0, 1]),
                "counting": lambda: r.Surv([0, 1], [3, 4], [1, 0]),
                "interval2": lambda: r.Surv([1, None, 3], [2, 3, None], type="interval2"),
                "mstate": lambda: r.Surv([1, 2], RFactor(["a", "censor"], ["censor", "a"])),
            }[kind]()
            assert_exact(list(r.format_surv(surv)), expected, path=aspect)
            return
        if aspect == "is_na":
            surv = r.Surv([1, None, 3], [1, 0, None])
            assert_exact([bool(v) for v in r.is_na_surv(surv)], expected, path=aspect)
            return
        builders = {
            "right": lambda: r.Surv([1, 2, 3, 4], [1, 0, 1, 1]),
            "right_12": lambda: r.Surv([1, 2, 3, 4], [2, 1, 2, 2]),
            "right_logical": lambda: r.Surv([1, 2, 3, 4], [True, False, True, True]),
            "left": lambda: r.Surv([1, 2, 3, 4], [1, 0, 1, 1], type="left"),
            "interval": lambda: r.Surv(
                [1, 2, 3, 4, 5], [2, None, 6, None, 7], [3, 0, 3, 1, 2], type="interval"
            ),
            "interval2": lambda: r.Surv([1, None, 3, 4, 5], [2, 3, None, 4, 8], type="interval2"),
            "counting": lambda: r.Surv([0, 1, 2, 0], [3, 4, 5, 2], [1, 0, 1, 1]),
            "counting_type": lambda: r.Surv(
                [0, 1, 2, 0], [3, 4, 5, 2], [1, 0, 1, 1], type="counting"
            ),
            "mstate": lambda: r.Surv(
                [1, 2, 3, 4],
                RFactor(["a", "censor", "b", "a"], ["censor", "a", "b"]),
                type="mstate",
            ),
            "mstate_factor": lambda: r.Surv(
                [1, 2, 3, 4], RFactor(["a", "censor", "b", "a"], ["censor", "a", "b"])
            ),
            "mcounting": lambda: r.Surv(
                [0, 1, 2, 0], [3, 4, 5, 2], RFactor(["a", "censor", "b", "a"], ["censor", "a", "b"])
            ),
            "origin": lambda: r.Surv([3, 4, 5], [1, 0, 1], origin=2),
        }
        surv = builders[aspect]()
        assert_exact(surv.type, expected["type"], path=f"{aspect}.type")
        assert_matrix_close(
            _surv_matrix(surv), expected["values"], rtol=1e-15, path=f"{aspect}.values"
        )
        if "states" in expected:
            assert_exact(list(_attr(surv, "states")), expected["states"], path=f"{aspect}.states")


def _basis_rows(basis: Any) -> list[list[float]]:
    """Rows of a spline basis returned either as nested lists or flat row-major."""

    values = _attr(basis, "basis", "values", "matrix")
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], (list, tuple)):
        return [list(row) for row in values]
    n_cols = _attr(basis, "n_cols")
    return [list(values[i : i + n_cols]) for i in range(0, len(values), n_cols)]


def _surv_matrix(surv: Any) -> list[list[Any]]:
    """R's Surv matrix columns for a Python Surv object."""

    time = list(surv.time)
    event = list(surv.event)
    if surv.type in ("right", "left"):
        return [[t, e] for t, e in zip(time, event, strict=True)]
    if surv.type in ("counting", "mcounting"):
        return [[s, t, e] for s, t, e in zip(surv.start, time, event, strict=True)]
    if surv.type in ("mright",):
        return [[t, e] for t, e in zip(time, event, strict=True)]
    if surv.type in ("interval", "interval2"):
        time2 = list(surv.time2)
        return [[t, t2, e] for t, t2, e in zip(time, time2, event, strict=True)]
    raise UnsupportedCaseError(f"unhandled Surv type {surv.type}")


# ---------------------------------------------------------------------------
# Parametrisation
# ---------------------------------------------------------------------------


# Cases are looked up by name inside the test: parametrising over the case
# dictionaries themselves makes pytest's failure reports (which repr every
# argument) prohibitively slow for the large fixture cases.
_CASES: dict[tuple[str, str], Mapping[str, Any]] = {}


def _collect_params() -> list[Any]:
    params = []
    for topic in topic_names():
        handler = HANDLERS.get(topic)
        for case in cases(topic):
            _CASES[(topic, case["name"])] = case
            aspects = handler.aspects(case) if handler is not None else ["(no handler)"]
            for aspect in aspects:
                test_id = case_id(topic, case, aspect)
                marks = []
                reason = _known_failure_reason(test_id)
                if reason is not None:
                    marks.append(pytest.mark.xfail(reason=reason, strict=True))
                params.append(pytest.param(topic, case["name"], aspect, id=test_id, marks=marks))
    return params


@pytest.mark.parametrize(("topic", "name", "aspect"), _collect_params())
def test_r_fixture(topic: str, name: str, aspect: str) -> None:
    __tracebackhide__ = True
    handler = HANDLERS.get(topic)
    case = _CASES[(topic, name)]
    test_id = case_id(topic, case, aspect)
    if handler is None:
        record_outcome(test_id, "missing feature", f"no handler for topic {topic}")
        raise UnsupportedCaseError(f"no handler for topic {topic}")
    # Every failure is re-raised from this frame without the original
    # traceback: pytest re-parses the whole source file of every frame it
    # prints (the 20k-line r_api.py included), which multiplies the runtime of
    # the ~2700 expected failures by ~20.  The message keeps the innermost
    # location; set R_FIXTURES_FULL_TRACEBACK=1 to debug with full tracebacks.
    full_traceback = bool(os.environ.get("R_FIXTURES_FULL_TRACEBACK"))
    try:
        handler.check(case, aspect)
        record_outcome(test_id, "pass", "")
    except UnsupportedCaseError as exc:
        record_outcome(test_id, "missing feature", str(exc))
        if full_traceback:
            raise
        raise UnsupportedCaseError(str(exc)) from None
    except FixtureMismatchError as exc:
        record_outcome(test_id, "mismatch", str(exc))
        if full_traceback:
            raise
        raise FixtureMismatchError(str(exc)) from None
    except pytest.skip.Exception:
        raise
    except Exception as exc:  # noqa: BLE001 - every failure kind feeds the burndown list
        message = f"{type(exc).__name__}: {exc} [{_innermost_location(exc)}]"
        record_outcome(test_id, "error", message)
        if full_traceback:
            raise
        raise PythonApiError(message) from None


class PythonApiError(Exception):
    """An exception raised by the Python API while running a fixture case."""


def _innermost_location(exc: BaseException) -> str:
    tb = exc.__traceback__
    location = "?"
    while tb is not None:
        code = tb.tb_frame.f_code
        location = f"{code.co_filename.rsplit('/', 1)[-1]}:{tb.tb_lineno}"
        tb = tb.tb_next
    return location


# ---------------------------------------------------------------------------
# Harness self-tests
# ---------------------------------------------------------------------------


def test_assert_close_floor_is_scaled_to_the_vector() -> None:
    # Noise around zero inside a vector whose other entries are O(1) is
    # within rtol * max|expected| and is not a mismatch ...
    assert_close([1.0, 0.0, -2.0], [1.0, -1.2e-17, -2.0], rtol=1e-8)
    assert_close([[0.0, 1.0], [1e-15, 0.0]], [[0.0, 1.0], [0.0, 0.0]], rtol=1e-8)
    # ... while a real difference of the same absolute size in a vector of
    # tiny values, or against a scalar zero, still is.
    with pytest.raises(FixtureMismatchError):
        assert_close([1e-17, 0.0], [1e-17, 1e-17], rtol=1e-8)
    with pytest.raises(FixtureMismatchError):
        assert_close(1e-17, 0.0, rtol=1e-8)
    with pytest.raises(FixtureMismatchError):
        assert_close([1.0, 1e-7], [1.0, 0.0], rtol=1e-8)
    # Exact comparisons keep a zero floor.
    with pytest.raises(FixtureMismatchError):
        assert_exact([3, 1e-17], [3, 0])


def test_array_scale_skips_non_numbers_and_mappings() -> None:
    assert array_scale([None, "NaN", "Inf", -3.0, [2.0, True]]) == 3.0
    assert array_scale([{"a": 100.0}, 0.5]) == 0.5
    assert array_scale([]) == 0.0
