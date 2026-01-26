#![no_main]
use libfuzzer_sys::fuzz_target;
use survival::{KaplanMeierConfig, compute_survfitkm};

fuzz_target!(|data: &[u8]| {
    if data.len() < 24 {
        return;
    }
    let n = (data.len() / 24).min(1000);
    let mut time = Vec::with_capacity(n);
    let mut status = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);

    for i in 0..n {
        let offset = i * 24;
        let t = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        let s = f64::from_le_bytes(data[offset + 8..offset + 16].try_into().unwrap());
        let w = f64::from_le_bytes(data[offset + 16..offset + 24].try_into().unwrap());

        if t.is_nan() || t.is_infinite() || t < 0.0 {
            return;
        }
        if w.is_nan() || w.is_infinite() || w < 0.0 {
            return;
        }

        time.push(t);
        status.push(if s > 0.0 { 1.0 } else { 0.0 });
        weights.push(w.max(0.01));
    }

    let position = vec![0i32; n];
    let config = KaplanMeierConfig::default();
    let _ = compute_survfitkm(&time, &status, &weights, None, &position, &config);
});
