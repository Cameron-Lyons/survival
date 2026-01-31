#![no_main]
use libfuzzer_sys::fuzz_target;
use survival::compute_brier;

fuzz_target!(|data: &[u8]| {
    if data.len() < 12 {
        return;
    }
    let n = (data.len() / 12).min(1000);
    let mut predictions = Vec::with_capacity(n);
    let mut outcomes = Vec::with_capacity(n);

    for i in 0..n {
        let offset = i * 12;
        let p = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        let o_raw = i32::from_le_bytes(data[offset + 8..offset + 12].try_into().unwrap());

        if p.is_nan() || p.is_infinite() {
            return;
        }

        predictions.push(p.clamp(0.0, 1.0));
        outcomes.push(if o_raw > 0 { 1 } else { 0 });
    }

    let _ = compute_brier(&predictions, &outcomes, None);
});
