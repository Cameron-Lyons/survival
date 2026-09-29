//! survpenal.fit against R 4.5.3 / survival 3.8-12 on the `kidney` data:
//! `survreg(Surv(time, status) ~ ..., kidney)` with `ridge()`, `pspline()`
//! and (through the alias `fr <- frailty`, which survreg() does not
//! recognise by name) sparse `frailty()` terms.  The reference values come
//! from R and are hard-coded; so are the 76 rows of `kidney`.

use super::kernel::{Case, JjPenalty, add_penalty, ensure_jj, survreg7};
use super::*;
use crate::core::pspline::pspline_basis;
use crate::regression::parametric_survival::survreg_fit;
use crate::regression::penalized::cholesky3::cholesky3;
use crate::regression::penalized::{FrailtyFamily, PenaltyTerm};
use crate::regression::survregc1::BlockLikelihood;

const TIME: [f64; 76] = [
    8.0, 16.0, 23.0, 13.0, 22.0, 28.0, 447.0, 318.0, 30.0, 12.0, 24.0, 245.0, 7.0, 9.0, 511.0,
    30.0, 53.0, 196.0, 15.0, 154.0, 7.0, 333.0, 141.0, 8.0, 96.0, 38.0, 149.0, 70.0, 536.0, 25.0,
    17.0, 4.0, 185.0, 177.0, 292.0, 114.0, 22.0, 159.0, 15.0, 108.0, 152.0, 562.0, 402.0, 24.0,
    13.0, 66.0, 39.0, 46.0, 12.0, 40.0, 113.0, 201.0, 132.0, 156.0, 34.0, 30.0, 2.0, 25.0, 130.0,
    26.0, 27.0, 58.0, 5.0, 43.0, 152.0, 30.0, 190.0, 5.0, 119.0, 8.0, 54.0, 16.0, 6.0, 78.0, 63.0,
    8.0,
];
const STATUS: [i32; 76] = [
    1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0,
    1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
    1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0,
];
const AGE: [f64; 76] = [
    28.0, 28.0, 48.0, 48.0, 32.0, 32.0, 31.0, 32.0, 10.0, 10.0, 16.0, 17.0, 51.0, 51.0, 55.0, 56.0,
    69.0, 69.0, 51.0, 52.0, 44.0, 44.0, 34.0, 34.0, 35.0, 35.0, 42.0, 42.0, 17.0, 17.0, 60.0, 60.0,
    60.0, 60.0, 43.0, 44.0, 53.0, 53.0, 44.0, 44.0, 46.0, 47.0, 30.0, 30.0, 62.0, 63.0, 42.0, 43.0,
    43.0, 43.0, 57.0, 58.0, 10.0, 10.0, 52.0, 52.0, 53.0, 53.0, 54.0, 54.0, 56.0, 56.0, 50.0, 51.0,
    57.0, 57.0, 44.0, 45.0, 22.0, 22.0, 42.0, 42.0, 52.0, 52.0, 60.0, 60.0,
];
const SEX: [f64; 76] = [
    1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0,
    1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0,
    1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0,
];
const ID: [f64; 76] = [
    1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 10.0,
    10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0, 14.0, 14.0, 15.0, 15.0, 16.0, 16.0, 17.0, 17.0, 18.0,
    18.0, 19.0, 19.0, 20.0, 20.0, 21.0, 21.0, 22.0, 22.0, 23.0, 23.0, 24.0, 24.0, 25.0, 25.0, 26.0,
    26.0, 27.0, 27.0, 28.0, 28.0, 29.0, 29.0, 30.0, 30.0, 31.0, 31.0, 32.0, 32.0, 33.0, 33.0, 34.0,
    34.0, 35.0, 35.0, 36.0, 36.0, 37.0, 37.0, 38.0, 38.0,
];

/// Asserts `actual` is within `rtol` of `expected` relative to
/// `max(|expected|, 1)`.
fn close(actual: f64, expected: f64, rtol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= rtol * expected.abs().max(1.0),
        "{what}: expected {expected}, got {actual}"
    );
}

fn all_close(actual: &[f64], expected: &[f64], rtol: f64, what: &str) {
    assert!(actual.len() >= expected.len(), "{what}: too short");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{what}[{i}]: expected NaN, got {a}");
        } else {
            close(*a, *e, rtol, &format!("{what}[{i}]"));
        }
    }
}

/// `pspline(age, df = 3)`: `nterm = round(2.5 * 3) = 8` intervals, cubic,
/// the first basis column dropped.
fn pspline_columns() -> Vec<Vec<f64>> {
    let lower = AGE.iter().copied().fold(f64::INFINITY, f64::min);
    let upper = AGE.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    pspline_basis(&AGE, 8, 3, (lower, upper))
        .unwrap()
        .basis
        .into_iter()
        .map(|row| row[1..].to_vec())
        .collect()
}

fn ridge() -> PenaltyTerm {
    PenaltyTerm::ridge(Some(1.0), None, 0.1, true, None).unwrap()
}

fn pspline() -> PenaltyTerm {
    PenaltyTerm::pspline(3.0, None, None, None, None, false).unwrap()
}

fn frailty(distribution: FrailtyFamily, theta: Option<f64>) -> PenaltyTerm {
    PenaltyTerm::frailty(
        distribution,
        true,
        theta,
        None,
        None,
        None,
        false,
        None,
        None,
    )
    .unwrap()
}

/// `kidney` with the design `columns` (an intercept first) and the model
/// `terms`.
fn kidney(
    columns: Vec<Vec<f64>>,
    terms: Vec<(Vec<usize>, Option<PenaltyTerm>)>,
    strata: bool,
) -> SurvpenalData {
    let covariates = Array2::from_shape_fn((76, columns.len() + 1), |(i, j)| match j {
        0 => 1.0,
        _ => columns[j - 1][i],
    });
    let strata = strata.then(|| SEX.iter().map(|&s| s as usize - 1).collect());
    let survreg = SurvregData::try_new(
        TIME.to_vec(),
        STATUS.to_vec(),
        covariates,
        None,
        None,
        None,
        strata,
        None,
    )
    .unwrap();
    let terms = terms
        .into_iter()
        .map(|(columns, penalty)| ModelTerm { columns, penalty })
        .collect();
    SurvpenalData::try_new(survreg, terms).unwrap()
}

/// `~ ridge(age, sex, theta = 1)`.
fn ridge_data() -> SurvpenalData {
    kidney(
        vec![AGE.to_vec(), SEX.to_vec()],
        vec![(vec![0], None), (vec![1, 2], Some(ridge()))],
        false,
    )
}

/// `~ sex + fr(id, ...)`.
fn frailty_data(term: PenaltyTerm) -> SurvpenalData {
    kidney(
        vec![SEX.to_vec(), ID.to_vec()],
        vec![(vec![0], None), (vec![1], None), (vec![2], Some(term))],
        false,
    )
}

/// `~ pspline(age, df = 3) + ...`: the basis in columns 1-10, `extra`
/// columns after it.
fn pspline_data(extra: Vec<Vec<f64>>, strata: bool) -> SurvpenalData {
    let basis = pspline_columns();
    let mut columns: Vec<Vec<f64>> = (0..10)
        .map(|j| basis.iter().map(|row| row[j]).collect())
        .collect();
    let mut terms = vec![(vec![0], None), ((1..=10).collect(), Some(pspline()))];
    for _ in &extra {
        terms.push((vec![columns.len() + 1], None));
    }
    columns.extend(extra);
    kidney(columns, terms, strata)
}

fn fit(data: &SurvpenalData, dist: &str, init: Option<Vec<f64>>) -> SurvpenalFit {
    let distribution = SurvregDistribution::from_name(dist, None).unwrap();
    let options = SurvpenalOptions {
        init,
        ..SurvpenalOptions::default()
    };
    SurvpenalFit::fit(data, &distribution, &options).unwrap()
}

fn variance_diagonal(fit: &SurvpenalFit) -> Vec<f64> {
    let var = &fit.survreg.variance_matrix;
    (0..var.len()).map(|i| var[i][i]).collect()
}

const RTOL: f64 = 1e-10;

#[test]
fn ridge_weibull_matches_r() {
    // Takes a golden-section step.
    let fit = fit(&ridge_data(), "weibull", None);
    let survreg = &fit.survreg;
    all_close(
        &survreg.coefficients,
        &[3.35054402986042, -0.00403459051341299, 0.946258145220318],
        RTOL,
        "coef",
    );
    all_close(&survreg.scale, &[1.10194190729782], RTOL, "scale");
    all_close(
        &[
            survreg.intercept_only_log_likelihood,
            survreg.log_likelihood,
        ],
        &[-340.937439453268, -336.555943285084],
        RTOL,
        "loglik",
    );
    assert_eq!(fit.iter, [1, 5]);
    all_close(
        &fit.df,
        &[0.976646840165056, 1.95723476197612, 0.999002597959565],
        RTOL,
        "df",
    );
    all_close(&fit.penalty, &[0.0, 0.0897380002703507], RTOL, "penalty");
    all_close(
        &survreg.icoef,
        &[4.85228317790347, 0.11812571381677],
        RTOL,
        "icoef",
    );
    all_close(
        &variance_diagonal(&fit),
        &[
            0.590416102186386,
            0.000103219076159984,
            0.103504584045543,
            0.00877472120709039,
        ],
        1e-10,
        "var",
    );
    close(fit.var2[(0, 0)], 0.576628020582902, RTOL, "var2");
    all_close(
        &survreg.linear_predictors,
        &[4.18383364070518, 4.18383364070518, 5.04939997565723],
        RTOL,
        "lp",
    );
    assert_eq!(fit.assign2, vec![vec![0], vec![1, 2], vec![3]]);
    assert_eq!(fit.pterms, vec![0, 1]);
    assert_eq!(fit.history.len(), 1);
    assert!(fit.history[0].done);
    assert_eq!(fit.history[0].theta, 1.0);
    assert!(fit.inner_failures.is_empty());
    close(survreg.df, fit.df.iter().sum(), 0.0, "sum(df)");
    close(survreg.df_residual, 76.0 - survreg.df, 0.0, "df.residual");
}

#[test]
fn pspline_plus_sex_matches_r() {
    // Takes golden-section steps.
    let fit = fit(&pspline_data(vec![SEX.to_vec()], false), "weibull", None);
    let survreg = &fit.survreg;
    assert_eq!(survreg.coefficients.len(), 13);
    all_close(
        &survreg.coefficients,
        &[2.44911705184861, 0.245947048809651, 0.45441898774683],
        RTOL,
        "coef",
    );
    close(survreg.coefficients[11], 1.07738007210454, RTOL, "sex");
    all_close(&survreg.scale, &[1.07840186162129], RTOL, "scale");
    close(survreg.log_likelihood, -335.142251835951, RTOL, "loglik");
    assert_eq!(fit.iter, [3, 10]);
    all_close(
        &fit.df,
        &[
            0.687098664185854,
            3.09430661106354,
            0.961597379470448,
            0.98900093172608,
        ],
        RTOL,
        "df",
    );
    close(fit.penalty[1], 0.23125117125333, RTOL, "penalty");
    close(fit.history[0].theta, 0.770373533504044, RTOL, "theta");
    all_close(
        &variance_diagonal(&fit),
        &[1.31442307030826, 0.494729106615561, 1.09770027561664],
        1e-10,
        "var",
    );
}

#[test]
fn sparse_gamma_frailty_matches_r() {
    // sex + fr(id): theta 0 flags the frailty at outer 1; Fisher steps, a
    // clamped golden-section step and an abject failure at outer 2; the
    // search is still running after outer.max = 10.
    let fit = fit(
        &frailty_data(frailty(FrailtyFamily::Gamma, None)),
        "weibull",
        None,
    );
    let survreg = &fit.survreg;
    all_close(
        &survreg.coefficients,
        &[3.11964882533375, 1.01248162811182],
        RTOL,
        "coef",
    );
    all_close(&survreg.scale, &[0.508562226907739], RTOL, "scale");
    close(survreg.log_likelihood, -296.432927752673, RTOL, "loglik");
    assert_eq!(fit.iter, [10, 33]);
    assert_eq!(fit.inner_failures, vec![2]);
    all_close(
        &fit.df,
        &[
            0.148888552517186,
            0.14945996127935,
            31.4467135088117,
            0.978496566736456,
        ],
        1e-10,
        "df",
    );
    close(fit.penalty[1], 8.63299889479804, RTOL, "penalty");
    all_close(
        fit.frail.as_ref().unwrap(),
        &[-1.52324962070511, -1.69924724335024, -0.853841439512139],
        RTOL,
        "frail",
    );
    all_close(
        fit.fvar.as_ref().unwrap(),
        &[0.31591834977036, 0.412352326819021, 0.301871162912768],
        1e-10,
        "fvar",
    );
    assert_eq!(fit.frail.as_ref().unwrap().len(), 38);
    close(fit.history[0].theta, 1.45218822470589, RTOL, "theta");
    assert!(!fit.history[0].done);
    assert!(fit.history[0].c_loglik.is_some());
    all_close(
        &survreg.linear_predictors,
        &[2.60888083274047, 2.60888083274047, 3.44536483820716],
        RTOL,
        "lp",
    );
    assert_eq!(fit.pterms, vec![0, 0, 2]);
    assert_eq!(fit.score.len(), 38 + 3);
}

#[test]
fn sparse_gaussian_frailty_matches_r() {
    // fr(id, dist = "gauss", theta = 0.5): a clamped golden-section step.
    let fit = fit(
        &frailty_data(frailty(FrailtyFamily::Gaussian, Some(0.5))),
        "weibull",
        None,
    );
    let survreg = &fit.survreg;
    all_close(
        &survreg.coefficients,
        &[2.16193567275493, 1.38337767950422],
        RTOL,
        "coef",
    );
    all_close(&survreg.scale, &[0.594046084448677], RTOL, "scale");
    close(survreg.log_likelihood, -303.326574650928, RTOL, "loglik");
    assert_eq!(fit.iter, [1, 7]);
    all_close(
        &fit.df,
        &[
            0.308937188867406,
            0.315785495300452,
            22.3796999709095,
            0.788973265516946,
        ],
        1e-10,
        "df",
    );
    close(fit.penalty[1], 35.5609883086048, RTOL, "penalty");
    all_close(
        fit.frail.as_ref().unwrap(),
        &[-0.702569658736247, -0.66690980534923, -0.233024628355921],
        RTOL,
        "frail",
    );
}

#[test]
fn pspline_with_scale_strata_matches_r() {
    let fit = fit(&pspline_data(Vec::new(), true), "lognormal", None);
    let survreg = &fit.survreg;
    all_close(
        &survreg.coefficients,
        &[4.38078504735036, 0.0721393949887948, 0.132352660962305],
        RTOL,
        "coef",
    );
    all_close(
        &survreg.scale,
        &[1.73660299742425, 1.12944033764675],
        RTOL,
        "scale",
    );
    all_close(
        &[
            survreg.intercept_only_log_likelihood,
            survreg.log_likelihood,
        ],
        &[-338.190507819404, -337.3856549376],
        RTOL,
        "loglik",
    );
    assert_eq!(fit.iter, [3, 9]);
    all_close(
        &fit.df,
        &[0.551391161188717, 3.07368769521557, 1.98219641429485],
        RTOL,
        "df",
    );
    all_close(
        &survreg.icoef,
        &[4.38012760600967, 0.545540266699129, 0.142054177622766],
        RTOL,
        "icoef",
    );
    assert_eq!(fit.assign2.last().unwrap(), &vec![11, 12]);
}

#[test]
fn a_far_start_converges_through_fisher_and_clamped_golden_steps() {
    // init = c(3, 0.05, 1, -2): Fisher steps at inner 1-22, a golden step
    // with the log-sigma clamp at 23, converged at 28.
    let fit = fit(&ridge_data(), "weibull", Some(vec![3.0, 0.05, 1.0, -2.0]));
    let survreg = &fit.survreg;
    all_close(
        &survreg.coefficients,
        &[3.35054402959599, -0.0040345904941353, 0.946258144725036],
        RTOL,
        "coef",
    );
    all_close(&survreg.scale, &[1.10194190708316], RTOL, "scale");
    close(survreg.log_likelihood, -336.555943285193, RTOL, "loglik");
    assert_eq!(fit.iter, [1, 28]);
    all_close(
        &fit.df,
        &[0.976646840177965, 1.9572347620063, 0.999002597961186],
        RTOL,
        "df",
    );
    all_close(&fit.penalty, &[0.0, 0.0897380001613628], RTOL, "penalty");
    assert!(fit.inner_failures.is_empty());
}

#[test]
fn an_abject_failure_returns_the_failed_step_information() {
    // init = c(6, 0, 0, -3): Fisher steps, a golden step at 18 and an
    // abject failure at 19; the information of the failed step zeroes the
    // sigma pivot, so the sigma df is 0/0.
    let fit = fit(&ridge_data(), "weibull", Some(vec![6.0, 0.0, 0.0, -3.0]));
    let survreg = &fit.survreg;
    let rtol = 1e-10;
    all_close(
        &survreg.coefficients,
        &[5.80537118486388, 0.000652952540412277, 0.0400329069119316],
        rtol,
        "coef",
    );
    all_close(&survreg.scale, &[0.0688589485747097], rtol, "scale");
    close(survreg.log_likelihood, -2264.89057824932, rtol, "loglik");
    assert_eq!(fit.iter, [1, 19]);
    assert!(!survreg.converged);
    assert_eq!(fit.inner_failures, vec![1]);
    all_close(
        &fit.df,
        &[0.999984909888272, 1.99998202496015, f64::NAN],
        rtol,
        "df",
    );
    assert!(survreg.df.is_nan() && survreg.df_residual.is_nan());
    close(fit.penalty[1], 0.000203770178077095, rtol, "penalty");
    all_close(
        &variance_diagonal(&fit),
        &[
            0.000254732492152532,
            4.79856632448092e-08,
            3.84174385719695e-05,
            0.0,
        ],
        rtol,
        "var",
    );
}

/// What `SurvpenalFit::fit` sets up before the outer loop, for driving
/// `survreg7` directly: the response on the log scale and the dense design.
struct Inner {
    distribution: SurvregDistribution,
    y1: Vec<f64>,
    status: Vec<i32>,
    xx: Array2<f64>,
    assign2: Vec<Vec<usize>>,
    frailx: Option<Vec<usize>>,
    nfrail: usize,
    ones: Vec<f64>,
    zeros: Vec<f64>,
    strata: Vec<usize>,
}

impl Inner {
    fn new(data: &SurvpenalData) -> Self {
        let distribution = SurvregDistribution::from_name("weibull", None).unwrap();
        let response = fitting_response(&data.survreg, &distribution, &[1.0; 76]).unwrap();
        let (xx, assign2, frailx, nfrail) =
            drop_sparse_column(data.survreg.design().view(), &data.terms);
        Self {
            distribution,
            y1: response.y1,
            status: response.status,
            xx,
            assign2,
            frailx,
            nfrail,
            ones: vec![1.0; 76],
            zeros: vec![0.0; 76],
            strata: vec![0; 76],
        }
    }

    fn kernel(&self) -> SurvregKernel<'_> {
        SurvregKernel {
            y1: &self.y1,
            y2: &self.y1,
            status: &self.status,
            covariates: self.xx.view(),
            weights: &self.ones,
            offset: &self.zeros,
            strata: &self.strata,
            nstrat: 1,
            distribution: &self.distribution,
        }
    }

    fn frailty(&self) -> Option<SparseFrailty<'_>> {
        self.frailx.as_deref().map(|group| SparseFrailty {
            group,
            nf: self.nfrail,
        })
    }

    fn composer<'a>(&self, data: &'a SurvpenalData) -> (Composer<'a>, PenaltyShape) {
        let (_, _, shape) = penalty_shape(&data.terms);
        let terms = build_term_states(
            &data.terms,
            &self.assign2,
            &self.xx,
            self.frailx.as_deref(),
            self.nfrail,
            76,
            &self.status,
            1e-9f64.sqrt(),
        )
        .unwrap();
        let composer = Composer::new(terms, self.nfrail, self.xx.ncols(), shape.full_imat, true);
        (composer, shape)
    }
}

/// `JJ` built after the fact equals, bit for bit, `JJ` accumulated in the
/// evaluation with the penalty added.
fn assert_lazy_jj_is_eager(data: &SurvpenalData, beta: &[f64]) {
    let inner = Inner::new(data);
    let kernel = inner.kernel();
    let frailty = inner.frailty();
    let (mut composer, shape) = inner.composer(data);
    let nf = inner.nfrail;
    let nvar = inner.xx.ncols();

    let mut lazy = BlockLikelihood::new(nf, nvar + 1);
    let mut jj_penalty = JjPenalty::default();
    let mut lazy_beta = beta.to_vec();
    kernel
        .evaluate_blocks(&lazy_beta, frailty.as_ref(), false, &mut lazy)
        .unwrap();
    add_penalty(
        Case::Full,
        nf,
        nvar,
        &mut lazy,
        &mut lazy_beta,
        shape,
        &mut composer,
        &mut jj_penalty,
    )
    .unwrap();
    ensure_jj(
        &kernel,
        frailty.as_ref(),
        &mut lazy,
        &mut None,
        &jj_penalty,
        shape,
    )
    .unwrap();

    let mut eager = BlockLikelihood::new(nf, nvar + 1);
    let mut eager_beta = beta.to_vec();
    kernel
        .evaluate_blocks(&eager_beta, frailty.as_ref(), true, &mut eager)
        .unwrap();
    add_penalty(
        Case::Full,
        nf,
        nvar,
        &mut eager,
        &mut eager_beta,
        shape,
        &mut composer,
        &mut JjPenalty::default(),
    )
    .unwrap();
    assert!(lazy.has_jj);
    assert_eq!(lazy.jj, eager.jj);
    assert_eq!(lazy.jdiag, eager.jdiag);
    assert_eq!(lazy_beta, eager_beta);
}

#[test]
fn jj_built_for_a_fisher_step_is_the_eager_jj() {
    // The first step from this far start is a Fisher step.
    let data = ridge_data();
    let beta = [3.0, 0.05, 1.0, -2.0];
    let inner = Inner::new(&data);
    let mut lik = BlockLikelihood::new(0, 4);
    inner
        .kernel()
        .evaluate_blocks(&beta, None, false, &mut lik)
        .unwrap();
    let (mut composer, shape) = inner.composer(&data);
    let mut start = beta.to_vec();
    add_penalty(
        Case::Full,
        0,
        3,
        &mut lik,
        &mut start,
        shape,
        &mut composer,
        &mut JjPenalty::default(),
    )
    .unwrap();
    assert!(cholesky3(&mut lik.hmat, 0, &mut lik.fdiag, 1e-10) < 0);
    assert_lazy_jj_is_eager(&data, &beta);

    // A sparse gamma frailty flagged at theta = 0, and one that is not.
    let mut beta: Vec<f64> = (0..38).map(|g| 0.01 * (g % 5) as f64 - 0.02).collect();
    beta.extend([4.8, 0.1, 0.1]);
    assert_lazy_jj_is_eager(&frailty_data(frailty(FrailtyFamily::Gamma, None)), &beta);
    assert_lazy_jj_is_eager(
        &frailty_data(frailty(FrailtyFamily::Gamma, Some(0.5))),
        &beta,
    );
}

#[test]
fn zero_iterations_return_the_start() {
    let data = ridge_data();
    let inner = Inner::new(&data);
    let (mut composer, shape) = inner.composer(&data);
    let start = vec![4.0, 0.01, 0.2, 0.1];
    let fit = survreg7(
        &inner.kernel(),
        None,
        0,
        start.clone(),
        1e-9,
        1e-10,
        shape,
        &mut composer,
    )
    .unwrap();
    assert_eq!(fit.iter, 1);
    assert_eq!(fit.beta, start);
    assert!(!fit.converged);
}

#[test]
fn accepted_steps_never_lower_the_penalised_loglik() {
    // survreg7 run for k iterations stops after its k-th accepted step, so
    // the penalised log likelihood grows with k, through the Fisher and
    // golden-section steps of both far starts and the abject failure.
    let data = ridge_data();
    let inner = Inner::new(&data);
    for (start, last) in [
        (vec![3.0, 0.05, 1.0, -2.0], 28),
        (vec![6.0, 0.0, 0.0, -3.0], 19),
    ] {
        let mut previous = f64::NEG_INFINITY;
        for maxiter in 0..=last {
            let (mut composer, shape) = inner.composer(&data);
            let fit = survreg7(
                &inner.kernel(),
                None,
                maxiter,
                start.clone(),
                1e-9,
                1e-10,
                shape,
                &mut composer,
            )
            .unwrap();
            assert!(fit.loglik >= previous, "iteration {maxiter}");
            previous = fit.loglik;
        }
    }
}

#[test]
fn a_negligible_ridge_is_the_unpenalised_fit() {
    let data = kidney(
        vec![AGE.to_vec(), SEX.to_vec()],
        vec![
            (vec![0], None),
            (
                vec![1, 2],
                Some(PenaltyTerm::ridge(Some(1e-10), None, 0.1, true, None).unwrap()),
            ),
        ],
        false,
    );
    let penalised = fit(&data, "weibull", None);
    let weibull = SurvregDistribution::from_name("weibull", None).unwrap();
    let plain = survreg_fit(
        &data.survreg,
        &weibull,
        None,
        0.0,
        &SurvregControl::default(),
        false,
    )
    .unwrap();
    for (a, b) in penalised
        .survreg
        .coefficients
        .iter()
        .zip(&plain.coefficients)
    {
        close(*a, *b, 1e-6, "coef");
    }
    close(
        penalised.survreg.log_likelihood,
        plain.log_likelihood,
        1e-8,
        "loglik",
    );
    close(penalised.survreg.df, 4.0, 1e-6, "df");
}

#[test]
fn invalid_requests_are_refused() {
    let data = ridge_data();
    let weibull = SurvregDistribution::from_name("weibull", None).unwrap();
    let refused = |options: SurvpenalOptions| {
        SurvpenalFit::fit(&data, &weibull, &options)
            .unwrap_err()
            .to_string()
    };
    let outer_max = SurvregControl {
        outer_max: 0,
        ..SurvregControl::default()
    };
    assert!(
        refused(SurvpenalOptions {
            control: outer_max,
            ..SurvpenalOptions::default()
        })
        .contains("invalid value for outer.max")
    );
    assert!(
        refused(SurvpenalOptions {
            init: Some(vec![6.0, 0.0]),
            ..SurvpenalOptions::default()
        })
        .contains("Wrong length for inital values")
    );
    // A sparse frailty has no robust variance, predictions or residuals.
    let sparse = frailty_data(frailty(FrailtyFamily::Gaussian, Some(0.5)));
    let error = SurvpenalFit::fit(
        &sparse,
        &weibull,
        &SurvpenalOptions {
            robust: true,
            ..SurvpenalOptions::default()
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("sparse frailty"));
    let fit = fit(&sparse, "weibull", None);
    assert!(fit.check_not_sparse("no").is_err());
}
