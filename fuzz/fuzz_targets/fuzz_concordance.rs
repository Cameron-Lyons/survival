#![no_main]
use libfuzzer_sys::fuzz_target;
use survival::concordance1;

fuzz_target!(|data: &[u8]| {
    if data.len() < 20 {
        return;
    }
    let n = (data.len() / 20).min(500);
    let mut y = Vec::with_capacity(2 * n);
    let mut weights = Vec::with_capacity(n);

    for i in 0..n {
        let offset = i * 20;
        let t = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        let s = f64::from_le_bytes(data[offset + 8..offset + 16].try_into().unwrap());
        let w_raw = u32::from_le_bytes(data[offset + 16..offset + 20].try_into().unwrap());

        if t.is_nan() || t.is_infinite() {
            return;
        }

        y.push(t.abs());
        weights.push((w_raw as f64 / u32::MAX as f64).max(0.01));
        // status stored in second half of y
        let _ = s; // will be pushed after time
    }

    for i in 0..n {
        let offset = i * 20;
        let s = f64::from_le_bytes(data[offset + 8..offset + 16].try_into().unwrap());
        y.push(if s > 0.0 { 1.0 } else { 0.0 });
    }

    let ntree = 1i32;
    let indx = vec![0i32; n];
    let _ = concordance1(&y, &weights, &indx, ntree);
});
