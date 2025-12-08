const LARGE: f64 = 22.0;
const SMALL: f64 = -200.0;

pub fn coxsafe(x: f64) -> f64 {
    if x < SMALL {
        SMALL
    } else if x > LARGE {
        LARGE
    } else {
        x
    }
}
