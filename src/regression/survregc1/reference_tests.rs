//! The likelihood sweeps against a per-observation dispatch over the
//! reference formulas: every result must agree bit for bit.

use super::*;
use crate::internal::rng::Rng;
use crate::regression::survreg_density::test_support::Mixture;
use crate::regression::survreg_kernels::reference::{
    KernelCase, builtin_distributions, kernel, same_bits,
};

fn reference_contribution(
    model: &SurvregKernel<'_>,
    distribution: &SurvregDistribution,
    person: usize,
    eta: f64,
    scales: &StratumScales,
    stratum: usize,
) -> Contribution {
    let sigma = scales.sigma[stratum];
    let sz = model.y1[person] - eta;
    let z = sz / sigma;
    let censored = |z| kernel(distribution, z, KernelCase::Distribution);
    match model.status[person] {
        1 => Contribution::exact_from(
            kernel(distribution, z, KernelCase::Density),
            z,
            sz,
            sigma,
            scales.log_sigma[stratum],
        ),
        0 => Contribution::right_from(censored(z), z, sz, sigma),
        2 => Contribution::left_from(censored(z), z, sz, sigma),
        _ => {
            let zu = (model.y2[person] - eta) / sigma;
            Contribution::interval_from(censored(z), censored(zu), z, zu, sigma)
        }
    }
}

fn reference_evaluate(model: &SurvregKernel<'_>, beta: &[f64], with_jj: bool) -> SurvregLikelihood {
    let nvar = model.nvar();
    let nvar2 = model.nvar2();
    let design = model.covariates.as_standard_layout();
    let design = design.as_slice().expect("standard layout");
    let scales = model.scales(beta, 0);
    let contributions = model
        .prepare_contributions(beta, None, design, &scales)
        .unwrap();
    let mut loglik = 0.0;
    let mut u = vec![0.0; nvar2];
    let mut imat = vec![0.0; nvar2 * nvar2];
    let mut jj = with_jj.then(|| vec![0.0; nvar2 * nvar2]);

    for person in 0..model.n() {
        let stratum = model.stratum(person);
        let x = &design[person * nvar..(person + 1) * nvar];
        let contribution = match &contributions {
            Contributions::Builtin(distribution) => {
                let eta = model.offset[person]
                    + x.iter().zip(&beta[..nvar]).map(|(x, b)| x * b).sum::<f64>();
                reference_contribution(model, distribution, person, eta, &scales, stratum)
            }
            Contributions::Batch(rows) => rows[person],
        };
        let w = model.weights[person];
        loglik += contribution.g * w;
        let scale = (model.nstrat != 0).then_some(nvar + stratum);
        contribution.accumulate(x, scale, w, &mut u, &mut imat, jj.as_deref_mut());
    }

    SurvregLikelihood {
        loglik,
        u,
        imat: symmetric_from_lower(nvar2, imat),
        jj: jj.map(|jj| symmetric_from_lower(nvar2, jj)),
    }
}

fn reference_blocks(
    model: &SurvregKernel<'_>,
    beta: &[f64],
    frailty: Option<&SparseFrailty<'_>>,
    with_jj: bool,
) -> BlockLikelihood {
    let nvar = model.nvar();
    let nf = frailty.map_or(0, |f| f.nf);
    let width = nf + model.nvar2();
    let design = model.covariates.as_standard_layout();
    let design = design.as_slice().expect("standard layout");
    let scales = model.scales(beta, nf);
    let contributions = model
        .prepare_contributions(beta, frailty, design, &scales)
        .unwrap();
    let mut out = BlockLikelihood::new(nf, model.nvar2());
    out.at.extend_from_slice(beta);
    out.has_jj = with_jj;
    let u = &mut out.u;
    let fdiag = &mut out.fdiag;
    let jdiag = &mut out.jdiag;
    let hmat = out.hmat.as_slice_mut().expect("standard layout");
    let jj = out.jj.as_slice_mut().expect("standard layout");
    let mut loglik = 0.0;
    for person in 0..model.n() {
        let x = &design[person * nvar..(person + 1) * nvar];
        let stratum = model.stratum(person);
        let c = match &contributions {
            Contributions::Builtin(distribution) => {
                let eta = model.block_eta(person, x, beta, frailty);
                reference_contribution(model, distribution, person, eta, &scales, stratum)
            }
            Contributions::Batch(rows) => rows[person],
        };
        let w = model.weights[person];
        loglik += c.g * w;
        let group = frailty.map(|f| f.group[person]);
        if let Some(g) = group {
            u[g] += c.dg * w;
            fdiag[g] -= c.ddg * w;
            if with_jj {
                jdiag[g] += c.dg * c.dg * w;
            }
        }
        for (i, &xi) in x.iter().enumerate() {
            let temp = c.dg * xi * w;
            u[nf + i] += temp;
            let row = i * width;
            for (j, &xj) in x[..=i].iter().enumerate() {
                hmat[row + nf + j] -= xi * xj * c.ddg * w;
                if with_jj {
                    jj[row + nf + j] += temp * xj * c.dg;
                }
            }
            if let Some(g) = group {
                hmat[row + g] -= xi * c.ddg * w;
                if with_jj {
                    jj[row + g] += temp * c.dg;
                }
            }
        }
        if model.nstrat != 0 {
            let k = stratum + nvar;
            let row = k * width;
            u[nf + k] += w * c.dsig;
            for (i, &xi) in x.iter().enumerate() {
                hmat[row + nf + i] -= c.dsg * xi * w;
                if with_jj {
                    jj[row + nf + i] += c.dsig * xi * c.dg * w;
                }
            }
            hmat[row + nf + k] -= c.ddsig * w;
            if with_jj {
                jj[row + nf + k] += c.dsig * c.dsig * w;
            }
            if let Some(g) = group {
                hmat[row + g] -= c.dsg * w;
                if with_jj {
                    jj[row + g] += c.dsig * c.dg * w;
                }
            }
        }
    }
    out.loglik = loglik;
    out
}

fn reference_loglik(
    model: &SurvregKernel<'_>,
    beta: &[f64],
    frailty: Option<&SparseFrailty<'_>>,
) -> f64 {
    let nvar = model.nvar();
    let design = model.covariates.as_standard_layout();
    let design = design.as_slice().expect("standard layout");
    let scales = model.scales(beta, frailty.map_or(0, |f| f.nf));
    let contributions = model
        .prepare_contributions(beta, frailty, design, &scales)
        .unwrap();
    let mut loglik = 0.0;
    for person in 0..model.n() {
        let x = &design[person * nvar..(person + 1) * nvar];
        let c = match &contributions {
            Contributions::Builtin(distribution) => {
                let eta = model.block_eta(person, x, beta, frailty);
                reference_contribution(
                    model,
                    distribution,
                    person,
                    eta,
                    &scales,
                    model.stratum(person),
                )
            }
            Contributions::Batch(rows) => rows[person],
        };
        loglik += c.g * model.weights[person];
    }
    loglik
}

/// Random data with every censoring code (zero-width intervals included),
/// weights with exact zeros, offsets, strata and frailty groups; `extreme`
/// also places responses far into the tails and at infinity.
struct Sample {
    y1: Vec<f64>,
    y2: Vec<f64>,
    status: Vec<i32>,
    x: Array2<f64>,
    weights: Vec<f64>,
    offset: Vec<f64>,
    strata: Vec<usize>,
    group: Vec<usize>,
}

const NF: usize = 4;

impl Sample {
    fn new(seed: u64, n: usize, nvar: usize, nstrata: usize, extreme: bool) -> Self {
        let mut rng = Rng::with_seed(seed);
        let mut uniform = |low: f64, high: f64| low + (high - low) * rng.f64();
        let mut y1 = Vec::with_capacity(n);
        let mut y2 = Vec::with_capacity(n);
        let mut status = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        let mut offset = Vec::with_capacity(n);
        for i in 0..n {
            let code = (uniform(0.0, 4.0) as i32).min(3);
            let mut lower = uniform(-3.0, 3.0);
            if extreme && i % 7 == 0 {
                lower = [300.0, -300.0, 1e4, -1e4, 900.0, -900.0][i % 6];
            }
            if extreme && i % 29 == 0 {
                lower = if i % 2 == 0 {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                };
            }
            let width = if uniform(0.0, 1.0) < 0.2 {
                0.0
            } else {
                uniform(0.0, 2.0)
            };
            y1.push(lower);
            y2.push(lower + width);
            status.push(code);
            weights.push(if uniform(0.0, 1.0) < 0.1 {
                0.0
            } else {
                uniform(0.0, 3.0)
            });
            offset.push(if uniform(0.0, 1.0) < 0.5 {
                0.0
            } else {
                uniform(-0.3, 0.3)
            });
        }
        let x = Array2::from_shape_fn(
            (n, nvar),
            |(_, j)| {
                if j == 0 { 1.0 } else { uniform(-2.0, 2.0) }
            },
        );
        let strata = (0..n)
            .map(|_| (uniform(0.0, nstrata.max(1) as f64) as usize).min(nstrata.max(1) - 1))
            .collect();
        let group = (0..n)
            .map(|_| (uniform(0.0, NF as f64) as usize).min(NF - 1))
            .collect();
        Self {
            y1,
            y2,
            status,
            x,
            weights,
            offset,
            strata,
            group,
        }
    }

    fn kernel<'a>(
        &'a self,
        nstrat: usize,
        distribution: &'a dyn SurvregDensitySource,
    ) -> SurvregKernel<'a> {
        SurvregKernel {
            y1: &self.y1,
            y2: &self.y2,
            status: &self.status,
            covariates: self.x.view(),
            weights: &self.weights,
            offset: &self.offset,
            strata: &self.strata,
            nstrat,
            distribution,
        }
    }

    fn frailty(&self) -> SparseFrailty<'_> {
        SparseFrailty {
            group: &self.group,
            nf: NF,
        }
    }
}

fn beta(seed: u64, len: usize) -> Vec<f64> {
    let mut rng = Rng::with_seed(seed);
    (0..len).map(|_| 0.6 * rng.f64() - 0.3).collect()
}

fn assert_same(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(same_bits(*a, *e), "{context} [{index}]: {a} != {e}");
    }
}

fn assert_same_likelihood(actual: &SurvregLikelihood, expected: &SurvregLikelihood, context: &str) {
    assert_same(&[actual.loglik], &[expected.loglik], context);
    assert_same(&actual.u, &expected.u, context);
    assert_same(
        actual.imat.as_slice().unwrap(),
        expected.imat.as_slice().unwrap(),
        context,
    );
    assert_eq!(actual.jj.is_some(), expected.jj.is_some(), "{context}");
    if let (Some(a), Some(e)) = (&actual.jj, &expected.jj) {
        assert_same(a.as_slice().unwrap(), e.as_slice().unwrap(), context);
    }
}

fn assert_same_blocks(actual: &BlockLikelihood, expected: &BlockLikelihood, context: &str) {
    assert_same(&actual.at, &expected.at, context);
    assert_same(&[actual.loglik], &[expected.loglik], context);
    assert_same(&actual.u, &expected.u, context);
    assert_same(&actual.fdiag, &expected.fdiag, context);
    assert_same(
        actual.hmat.as_slice().unwrap(),
        expected.hmat.as_slice().unwrap(),
        context,
    );
    assert_eq!(actual.has_jj, expected.has_jj, "{context}");
    if expected.has_jj {
        assert_same(&actual.jdiag, &expected.jdiag, context);
        assert_same(
            actual.jj.as_slice().unwrap(),
            expected.jj.as_slice().unwrap(),
            context,
        );
    }
}

/// Every sweep of `model` at a random `beta` against the reference sweeps,
/// with and without the squared scores and the sparse frailty.
fn check_sweeps(model: &SurvregKernel<'_>, sample: &Sample, seed: u64, context: &str) {
    let nscale = model.nstrat.max(1);
    let beta0 = beta(seed, model.nvar() + nscale);
    for with_jj in [false, true] {
        assert_same_likelihood(
            &model.evaluate(&beta0, with_jj).unwrap(),
            &reference_evaluate(model, &beta0, with_jj),
            &format!("{context} evaluate jj={with_jj}"),
        );
    }
    let frailty = sample.frailty();
    for frailty in [None, Some(&frailty)] {
        let nf = frailty.map_or(0, |f| f.nf);
        let beta = beta(seed + 1, nf + model.nvar() + nscale);
        for with_jj in [false, true] {
            let context = format!("{context} blocks frailty={} jj={with_jj}", nf > 0);
            let mut blocks = BlockLikelihood::new(nf, model.nvar2());
            model
                .evaluate_blocks(&beta, frailty, with_jj, &mut blocks)
                .unwrap();
            assert_same_blocks(
                &blocks,
                &reference_blocks(model, &beta, frailty, with_jj),
                &context,
            );
        }
        assert_same(
            &[model.loglik_at(&beta, frailty).unwrap()],
            &[reference_loglik(model, &beta, frailty)],
            &format!("{context} loglik_at frailty={}", nf > 0),
        );
    }
}

#[test]
fn builtin_sweeps_match_the_reference_dispatch_bit_for_bit() {
    let distributions = builtin_distributions();
    for (case, &(nvar, nstrata, nstrat, extreme)) in [
        (3, 1, 1, false),
        (1, 1, 1, false),
        (4, 3, 3, false),
        (2, 1, 0, false),
        (3, 3, 3, true),
        (2, 1, 1, true),
    ]
    .iter()
    .enumerate()
    {
        let seed = 100 + case as u64;
        let sample = Sample::new(seed, 160, nvar, nstrata, extreme);
        for distribution in &distributions {
            let model = sample.kernel(nstrat, distribution);
            let context = format!(
                "{} nvar={nvar} nstrat={nstrat} extreme={extreme}",
                distribution.name
            );
            check_sweeps(&model, &sample, seed, &context);
        }
    }
}

#[test]
fn batch_sweeps_match_the_reference_dispatch_bit_for_bit() {
    let source = Mixture::default();
    for (case, &(nvar, nstrata, nstrat)) in [(3, 1, 1), (2, 3, 3), (2, 1, 0)].iter().enumerate() {
        let seed = 200 + case as u64;
        let sample = Sample::new(seed, 120, nvar, nstrata, false);
        let model = sample.kernel(nstrat, &source);
        check_sweeps(&model, &sample, seed, &format!("mixture nstrat={nstrat}"));
    }
}
