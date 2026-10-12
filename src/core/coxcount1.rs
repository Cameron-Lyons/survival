use crate::core::strata_order::{first_of_run, order_within_strata, validate_intervals};
use crate::error::SurvivalResult;
use crate::internal::typed_inputs::{CountingProcessData, SurvivalData};
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_length, validate_non_empty,
};
use pyo3::prelude::*;

#[doc = " The expanded risk sets: `time[k]` and `nrisk[k]` describe the `k`-th"]
#[doc = " unique event time; the next `nrisk[k]` entries of `index`/`status` are"]
#[doc = " its members (zero-based rows of the input) and their event indicators."]
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CoxCountOutput {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub nrisk: Vec<usize>,
    #[pyo3(get)]
    pub index: Vec<usize>,
    #[pyo3(get)]
    pub status: Vec<i32>,
}

pub fn coxcount1(
    survival: &SurvivalData,
    strata: Option<&[i32]>,
) -> SurvivalResult<CoxCountOutput> {
    let n = survival.len();
    let time = &survival.time;
    let status = &survival.status;
    SurvivalData::validate_parts(time, status)?;
    validate_binary_i32(status, "status")?;
    let zero = vec![0; n];
    let strata = strata.map_or(Ok(&zero[..]), |s| {
        validate_length(n, s.len(), "strata").map(|()| s)
    })?;
    let order = order_within_strata(strata, |a, b| {
        time[b]
            .total_cmp(&time[a])
            .then_with(|| status[a].cmp(&status[b]))
    });
    let sorted_time: Vec<f64> = order.iter().map(|&i| time[i]).collect();
    let sorted_status: Vec<i32> = order.iter().map(|&i| status[i]).collect();
    let sorted_strata: Vec<i32> = order.iter().map(|&i| strata[i]).collect();
    let first = first_of_run(&sorted_strata);

    let mut out = CoxCountOutput {
        time: Vec::new(),
        nrisk: Vec::new(),
        index: Vec::new(),
        status: Vec::new(),
    };
    let mut stratastart = 0;
    let mut i = 0;
    while i < n {
        if first[i] {
            stratastart = i;
        }
        if sorted_status[i] == 1 {
            let dtime = sorted_time[i];
            let mut j = i + 1;
            while j < n && sorted_status[j] == 1 && sorted_time[j] == dtime && !first[j] {
                j += 1;
            }
            let last = j - 1;
            out.status.extend(std::iter::repeat_n(0, i - stratastart));
            out.status.extend(std::iter::repeat_n(1, last + 1 - i));
            out.index.extend(order[stratastart..=last].iter().copied());
            out.time.push(dtime);
            out.nrisk.push(last + 1 - stratastart);
            i = last;
        }
        i += 1;
    }
    Ok(out)
}

pub fn coxcount2(
    counting: &CountingProcessData,
    strata: Option<&[i32]>,
) -> SurvivalResult<CoxCountOutput> {
    let n = counting.len();
    let time1 = &counting.start;
    let time2 = &counting.stop;
    let status = &counting.event;
    validate_non_empty(time1, "start")?;
    validate_length(n, time2.len(), "stop")?;
    validate_length(n, status.len(), "event")?;
    validate_finite(time1, "start")?;
    validate_finite(time2, "stop")?;
    validate_binary_i32(status, "event")?;
    validate_intervals(time1, time2)?;
    let zero = vec![0; n];
    let strata = strata.map_or(Ok(&zero[..]), |s| {
        validate_length(n, s.len(), "strata").map(|()| s)
    })?;
    let sort2 = order_within_strata(strata, |a, b| {
        time2[b]
            .total_cmp(&time2[a])
            .then_with(|| status[a].cmp(&status[b]))
    });
    let sort1 = order_within_strata(strata, |a, b| time1[b].total_cmp(&time1[a]));
    let sorted_strata: Vec<i32> = sort2.iter().map(|&i| strata[i]).collect();
    let first = first_of_run(&sorted_strata);

    let mut out = CoxCountOutput {
        time: Vec::new(),
        nrisk: Vec::new(),
        index: Vec::new(),
        status: Vec::new(),
    };
    let mut who: Vec<usize> = Vec::with_capacity(n);
    let mut atrisk = vec![0usize; n];
    let mut j = 0;
    let mut i = 0;
    while i < n {
        let iptr = sort2[i];
        if first[i] {
            who.clear();
            j = i;
        }
        if status[iptr] == 0 {
            atrisk[iptr] = who.len();
            who.push(iptr);
            i += 1;
            continue;
        }
        let dtime = time2[iptr];
        while j < i && time1[sort1[j]] >= dtime {
            let jptr = sort1[j];
            let k = atrisk[jptr];
            let last = who.pop().expect("a departing subject is in the risk set");
            if k < who.len() {
                who[k] = last;
                atrisk[last] = k;
            }
            j += 1;
        }
        out.status.extend(std::iter::repeat_n(0, who.len()));
        out.index.extend(who.iter().copied());
        let mut deaths = 1;
        atrisk[iptr] = who.len();
        who.push(iptr);
        out.index.push(iptr);
        i += 1;
        while i < n && !first[i] && time2[sort2[i]] == dtime {
            let tied = sort2[i];
            out.index.push(tied);
            atrisk[tied] = who.len();
            who.push(tied);
            deaths += 1;
            i += 1;
        }
        out.status.extend(std::iter::repeat_n(1, deaths));
        out.time.push(dtime);
        out.nrisk.push(who.len());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn right_censored_risk_sets_follow_r() {
        let out = coxcount1(
            &SurvivalData::try_new(vec![3.0, 1.0, 2.0, 2.0], vec![1, 1, 0, 1]).unwrap(),
            None,
        )
        .unwrap();
        assert_eq!(out.time, vec![3.0, 2.0, 1.0]);
        assert_eq!(out.nrisk, vec![1, 3, 4]);
        assert_eq!(out.index, vec![0, 0, 2, 3, 0, 2, 3, 1]);
        assert_eq!(out.status, vec![1, 0, 0, 1, 0, 0, 0, 1]);
    }

    #[test]
    fn tied_deaths_share_a_risk_set_and_strata_reset() {
        let out = coxcount1(
            &SurvivalData::try_new(vec![1.0, 1.0, 2.0, 5.0, 5.0], vec![1, 1, 0, 1, 1]).unwrap(),
            Some(&[1, 1, 1, 2, 2]),
        )
        .unwrap();
        assert_eq!(out.time, vec![1.0, 5.0]);
        assert_eq!(out.nrisk, vec![3, 2]);
        assert_eq!(out.index, vec![2, 0, 1, 3, 4]);
        assert_eq!(out.status, vec![0, 1, 1, 1, 1]);
    }

    #[test]
    fn counting_process_risk_sets_drop_late_entries_and_early_exits() {
        let out = coxcount2(
            &CountingProcessData::try_new(
                vec![0.0, 0.0, 1.0, 2.0, 4.0],
                vec![4.0, 6.0, 5.0, 7.0, 9.0],
                vec![1, 0, 1, 1, 0],
            )
            .unwrap(),
            None,
        )
        .unwrap();
        assert_eq!(out.time, vec![7.0, 5.0, 4.0]);
        assert_eq!(out.nrisk, vec![2, 4, 4]);
        assert_eq!(out.index, vec![4, 3, 4, 3, 1, 2, 2, 3, 1, 0]);
        assert_eq!(out.status, vec![0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
    }

    #[test]
    fn stratified_outputs_match_r() {
        let out = coxcount1(
            &SurvivalData::try_new(
                vec![5.0, 3.0, 3.0, 8.0, 1.0, 3.0, 8.0, 2.0, 6.0, 6.0],
                vec![1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            )
            .unwrap(),
            Some(&[1, 1, 1, 1, 2, 2, 2, 2, 2, 2]),
        )
        .unwrap();
        assert_eq!(out.time, vec![8.0, 5.0, 3.0, 6.0, 3.0, 2.0]);
        assert_eq!(out.nrisk, vec![1, 2, 4, 3, 4, 5]);
        assert_eq!(
            out.index,
            vec![3, 3, 0, 3, 0, 2, 1, 6, 8, 9, 6, 8, 9, 5, 6, 8, 9, 5, 7]
        );
        assert_eq!(
            out.status,
            vec![1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        );

        let out = coxcount2(
            &CountingProcessData::try_new(
                vec![0.0, 1.0, 0.0, 2.0, 4.0, 0.0, 3.0, 1.0, 0.0, 5.0],
                vec![4.0, 6.0, 5.0, 7.0, 9.0, 3.0, 8.0, 6.0, 2.0, 9.0],
                vec![1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
            )
            .unwrap(),
            Some(&[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
        )
        .unwrap();
        assert_eq!(out.time, vec![7.0, 5.0, 4.0, 9.0, 8.0, 6.0, 3.0]);
        assert_eq!(out.nrisk, vec![2, 4, 4, 1, 2, 3, 2]);
        assert_eq!(
            out.index,
            vec![4, 3, 4, 3, 1, 2, 2, 3, 1, 0, 9, 9, 6, 9, 6, 7, 7, 5]
        );
        assert_eq!(
            out.status,
            vec![0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
        );
    }

    #[test]
    fn rejects_bad_inputs() {
        let survival = SurvivalData::try_new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        assert!(coxcount1(&survival, None).is_err());
        let survival = SurvivalData::try_new(vec![1.0, 2.0], vec![1, 0]).unwrap();
        assert!(coxcount1(&survival, Some(&[1])).is_err());
    }
}
