pub struct Decomposition {
    pub d: Vec<f64>,
    pub a: Vec<f64>,
    pub ainv: Vec<f64>,
    pub p: Vec<f64>,
}

pub fn c_decomp(r: &[f64], time: f64) -> Decomposition {
    let nc = (r.len() as f64).sqrt() as usize;
    assert_eq!(nc * nc, r.len(), "R must be a square matrix");

    let mut d = vec![0.0; nc];
    let mut a = vec![0.0; nc * nc];
    let mut ainv = vec![0.0; nc * nc];
    let mut p = vec![0.0; nc * nc];

    // Compute eigenvectors (matrix A) and eigenvalues (vector d)
    for i in 0..nc {
        let diag_idx = i * nc + i;
        d[i] = r[diag_idx];
        a[diag_idx] = 1.0;

        for j in (0..i).rev() {
            let mut temp = 0.0;
            for k in j..=i {
                temp += r[j + k * nc] * a[k + i * nc];
            }
            a[j + i * nc] = temp / (d[i] - r[j + j * nc]);
        }
    }

    // Compute inverse of A (ainv)
    for i in 0..nc {
        let diag_idx = i * nc + i;
        ainv[diag_idx] = 1.0;

        for j in (0..i).rev() {
            let mut temp = 0.0;
            for k in (j + 1)..=i {
                temp += a[j + k * nc] * ainv[k + i * nc];
            }
            ainv[j + i * nc] = -temp;
        }
    }

    // Compute matrix P = A * diag(exp(time*d)) * ainv
    let ediag: Vec<f64> = d.iter().map(|&val| (time * val).exp()).collect();

    for i in 0..nc {
        let p_diag_idx = i * nc + i;
        p[p_diag_idx] = ediag[i];

        for j in 0..i {
            let mut temp = 0.0;
            for k in j..nc {
                temp += a[j + k * nc] * ainv[k + i * nc] * ediag[k];
            }
            p[j + i * nc] = temp;
        }
    }

    Decomposition { d, a, ainv, p }
}
