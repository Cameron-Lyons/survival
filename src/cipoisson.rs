// Confidence limits for the Poisson

use statrs::distribution::{Gamma, Normal, Univariate};

enum Method {
    Exact,
    Anscombe,
}

fn cipoisson_exact(k: u32, time: f64, p: f64) -> Result<(f64, f64), &'static str> {
    if time <= 0.0 || p <= 0.0 || p >= 1.0 {
        return Err("Invalid input values");
    }

    let alpha_low = p / 2.0;
    let alpha_high = 1.0 - alpha_low;

    let gamma_low = Gamma::new(k as f64, 1.0).map_err(|_| "Error creating Gamma distribution")?;
    let gamma_high =
        Gamma::new((k + 1) as f64, 1.0).map_err(|_| "Error creating Gamma distribution")?;

    let lower_bound = gamma_low.inverse_cdf(alpha_low);
    let upper_bound = gamma_high.inverse_cdf(alpha_high);

    Ok((lower_bound / time, upper_bound / time))
}

fn cipoisson_anscombe(k: u32, time: f64, p: f64) -> Result<(f64, f64), &'static str> {
    if time <= 0.0 || p <= 0.0 || p >= 1.0 {
        return Err("Invalid input values");
    }

    let transformed_k = (k as f64 + 3.0 / 8.0).sqrt();
    let z = Normal::new(0.0, 1.0)
        .map_err(|_| "Error creating Normal distribution")?
        .inverse_cdf(p / 2.0);
    let variance = 1.0 / 4.0; // The variance of sqrt(k + 3/8) is approximately 1/4

    let lower_bound = transformed_k - z * (variance.sqrt());
    let upper_bound = transformed_k + z * (variance.sqrt());

    // Re-transforming the bounds back to the Poisson scale
    let lower_bound_poisson = (lower_bound.powi(2) - 3.0 / 8.0).max(0.0) / time;
    let upper_bound_poisson = (upper_bound.powi(2) - 3.0 / 8.0) / time;

    Ok((lower_bound_poisson, upper_bound_poisson))
}

fn cipoisson(k: u32, time: f64, p: f64, method: Method) -> Result<(f64, f64), &'static str> {
    match method {
        Method::Exact => cipoisson_exact(k, time, p),
        Method::Anscombe => cipoisson_anscombe(k, time, p),
    }
}
