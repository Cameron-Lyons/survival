#![no_main]
use libfuzzer_sys::fuzz_target;
use survival::concordance::{ConcordanceOptions, concordancefit};
use survival::core::SurvResponse;
use survival::data_types::SurvivalData;

fuzz_target!(|data: &[u8]| {
    if data.len() < 20 {
        return;
    }
    let n = (data.len() / 20).min(500);
    let mut time = Vec::with_capacity(n);
    let mut status = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);

    for i in 0..n {
        let offset = i * 20;
        let t = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        let s = f64::from_le_bytes(data[offset + 8..offset + 16].try_into().unwrap());
        let w_raw = u32::from_le_bytes(data[offset + 16..offset + 20].try_into().unwrap());

        if !t.is_finite() || !s.is_finite() {
            return;
        }

        time.push(t.abs());
        status.push(i32::from(s > 0.0));
        weights.push((w_raw as f64 / u32::MAX as f64).max(0.01));
        x.push(s);
    }

    let Ok(survival) = SurvivalData::try_new(time, status) else {
        return;
    };
    let x = ndarray::Array2::from_shape_vec((n, 1), x).expect("n x 1");
    let _ = concordancefit(
        SurvResponse::Right(&survival),
        x.view(),
        Some(&weights),
        None,
        None,
        &ConcordanceOptions::default(),
    );
});
