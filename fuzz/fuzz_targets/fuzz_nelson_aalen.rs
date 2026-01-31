#![no_main]
use libfuzzer_sys::fuzz_target;
use survival::nelson_aalen;

fuzz_target!(|data: &[u8]| {
    if data.len() < 12 {
        return;
    }
    let n = (data.len() / 12).min(1000);
    let mut time = Vec::with_capacity(n);
    let mut status = Vec::with_capacity(n);

    for i in 0..n {
        let offset = i * 12;
        let t = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        let s = i32::from_le_bytes(data[offset + 8..offset + 12].try_into().unwrap());

        if t.is_nan() || t.is_infinite() || t < 0.0 {
            return;
        }

        time.push(t);
        status.push(if s > 0 { 1 } else { 0 });
    }

    let _ = nelson_aalen(&time, &status, None, 0.95);
});
