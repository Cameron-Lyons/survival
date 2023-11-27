// Confidence limits for the Poisson

enum Method {
    Exact,
    Anscombe,
}

fn cipoisson(k: u32, time: f64, p: f64, method: Method) -> Result<(f64, f64), &'static str> {
    match method {
        Method::Exact => {
            let mut lower = 0.0;
            let mut upper = 0.0;
            let mut sum = 0.0;
            let mut term = 1.0;
            let mut i = 0;
            while i <= k {
                sum += term;
                if i == k {
                    lower = sum;
                }
                if sum >= p && upper == 0.0 {
                    upper = sum;
                }
                term *= time / (i + 1) as f64;
                i += 1;
            }
            Ok((lower, upper))
        }
        Method::Anscombe => {
            let mut lower = 0.0;
            let mut upper = 0.0;
            let mut sum = 0.0;
            let mut term = 1.0;
            let mut i = 0;
            while i <= k {
                sum += term;
                if i == k {
                    lower = sum;
                }
                if sum >= p && upper == 0.0 {
                    upper = sum;
                }
                term *= time / (i + 1) as f64;
                i += 1;
            }
            Ok((lower, upper))
        }
    }
}
