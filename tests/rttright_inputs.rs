use survival::data_prep::{RttrightInput, rttright};

#[test]
fn sparse_stratum_labels_preserve_redistribution_and_subject_normalization() {
    let time = [1.0, 3.0, 2.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let start = [0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 3.0];
    let status = [0, 1, 0, 1, 0, 0, 0, 1];
    let id = [1i64, 1, 2, 2, 3, 3, 4, 4];
    let weights = [1.0, 1.0, 2.0, 2.0, 0.5, 0.5, 3.0, 3.0];
    let compact = [1, 0, 1, 0, 1, 1, 0, 0];
    let sparse = compact.map(|code| if code == 0 { usize::MAX } else { 7 });
    let queries = [0.0, 1.0, 2.5, 5.0, 7.0];
    for start in [None, Some(start.as_slice())] {
        for renorm in [false, true] {
            for times in [None, Some(queries.as_slice())] {
                for timefix in [false, true] {
                    let fit = |strata| {
                        rttright(RttrightInput {
                            start,
                            time: &time,
                            status: &status,
                            strata: Some(strata),
                            weights: Some(&weights),
                            id: Some(&id),
                            times,
                            timefix,
                            renorm,
                        })
                        .unwrap()
                    };
                    let expected = fit(compact.as_slice());
                    let actual = fit(sparse.as_slice());
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}

#[test]
fn empty_stratified_and_unstratified_inputs_have_no_weights() {
    for strata in [None, Some([].as_slice())] {
        for renorm in [false, true] {
            let result = rttright::<i64>(RttrightInput {
                start: None,
                time: &[],
                status: &[],
                strata,
                weights: None,
                id: None,
                times: Some(&[1.0, 2.0]),
                timefix: false,
                renorm,
            })
            .unwrap();
            assert!(result.weights.is_empty());
            assert_eq!(result.times, vec![1.0, 2.0]);
        }
    }
}
