#[allow(dead_code)]
pub fn agsurv4(ndeath: &[i32], risk: &[f64], wt: &[f64], sn: usize, denom: &[f64], km: &mut [f64]) {
    let n = sn;
    let mut j = 0;

    for i in 0..n {
        match ndeath[i] {
            0 => km[i] = 1.0,
            1 => {
                let numerator = wt[j] * risk[j];
                km[i] = (1.0 - numerator / denom[i]).powf(1.0 / risk[j]);
                j += 1;
            }
            _ => {
                let mut guess: f64 = 0.5;
                let mut inc = 0.25;
                let death_count = ndeath[i] as usize;
                let range = j..(j + death_count);

                for _ in 0..35 {
                    let mut sumt = 0.0;
                    for k in range.clone() {
                        let term = wt[k] * risk[k] / (1.0 - guess.powf(risk[k]));
                        sumt += term;
                    }

                    if sumt < denom[i] {
                        guess += inc;
                    } else {
                        guess -= inc;
                    }
                    inc /= 2.0;
                }

                km[i] = guess;
                j += death_count;
            }
        }
    }
}
