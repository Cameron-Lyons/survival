#![no_main]
use libfuzzer_sys::fuzz_target;
use survival::validation::{BrierInput, brier};

fuzz_target!(|data: &[u8]| {
    if data.len() < 24 {
        return;
    }
    let n = (data.len() / 24).min(1000);
    let mut time = Vec::with_capacity(n);
    let mut status = Vec::with_capacity(n);
    let mut phat = Vec::with_capacity(n);

    for i in 0..n {
        let offset = i * 24;
        let t = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        let p = f64::from_le_bytes(data[offset + 8..offset + 16].try_into().unwrap());
        let s_raw = i64::from_le_bytes(data[offset + 16..offset + 24].try_into().unwrap());

        if !t.is_finite() || t < 0.0 || !p.is_finite() {
            return;
        }

        time.push(t);
        phat.push(p.clamp(0.0, 1.0));
        status.push(i32::from(s_raw > 0));
    }

    let max_time = time.iter().cloned().fold(0.0_f64, f64::max);
    let times: Vec<f64> = (1..=4).map(|k| max_time * k as f64 / 5.0).collect();
    let phat: Vec<Vec<f64>> = times.iter().map(|_| phat.clone()).collect();
    let _ = brier(&BrierInput {
        start: None,
        time: &time,
        status: &status,
        weights: None,
        times: &times,
        phat: &phat,
        ties: true,
        efron: false,
        timefix: true,
    });
});
