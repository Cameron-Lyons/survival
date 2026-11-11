//! A resolved dot-product correction must not share storage with risk mass.
//! Kept separate while the shared logarithmic representation is extended.

use ndarray::array;
use survival::regression::{CoxphData, CoxphFitResult, CoxphOptions, TieMethod};

#[test]
fn duplicate_extreme_covariate_risks_retain_their_event_probability() {
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        for counting in [false, true] {
            let data = CoxphData::try_new(
                vec![1., 1., 2., 2.],
                counting.then(|| vec![0., 0., 1.5, 1.5]),
                vec![1, 0, 1, 0],
                array![[1., 1.], [1., 1.], [0., 0.], [0., 0.]],
                None,
                None,
                None,
            )
            .unwrap();
            let fit = CoxphFitResult::fit(
                data,
                CoxphOptions {
                    method,
                    init: Some(vec![1e308, 1e292]),
                    iter_max: 0,
                    ..CoxphOptions::default()
                },
                false,
            )
            .unwrap();
            // At each event the two relevant rows have identical covariates,
            // hence each event probability is 1/2. In the right-censored case
            // the late pair's contribution to the first risk set is below the
            // representable range; in the counting case it has not entered.
            let expected = -2.0 * 2.0_f64.ln();
            for loglik in fit.loglik {
                assert!(
                    (loglik - expected).abs() < 1e-12,
                    "method {method:?}, counting {counting}: {loglik} != {expected}"
                );
            }
            assert_eq!(fit.first, [0.0, 0.0]);
            assert!(fit.var.iter().all(|&value| value == 0.0));
        }
    }
}
