const LARGE: f64 = 22.0;
const SMALL: f64 = -200.0;

pub(crate) fn coxsafe(x: f64) -> f64 {
    x.clamp(SMALL, LARGE)
}
