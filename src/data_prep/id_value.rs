//! Subject identifiers for the data-preparation routines.
//!
//! R lets `id`, `strata` and state variables be integer, double, character
//! or factor; its `match`/`order` semantics are what the kernels need.
//! [`SubjectId`] extracts a comparable, hashable key from whatever the
//! caller holds, so every routine has one generic implementation.  Rust
//! callers use `i64`, `String` or `&str`; the Python bindings accept
//! [`IdValue`], into which an `int`, `float` or `str` converts.

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// A value usable as a subject identifier or categorical state.
pub trait SubjectId: Clone {
    /// The identity used for equality, hashing and ordering.
    type Key: Clone + Eq + Hash + Ord;

    fn key(&self) -> Self::Key;

    /// The label R would print for the value.
    fn label(&self) -> String;
}

impl SubjectId for i64 {
    type Key = i64;

    fn key(&self) -> i64 {
        *self
    }

    fn label(&self) -> String {
        self.to_string()
    }
}

impl SubjectId for usize {
    type Key = usize;

    fn key(&self) -> usize {
        *self
    }

    fn label(&self) -> String {
        self.to_string()
    }
}

impl SubjectId for String {
    type Key = String;

    fn key(&self) -> String {
        self.clone()
    }

    fn label(&self) -> String {
        self.clone()
    }
}

impl SubjectId for &str {
    type Key = String;

    fn key(&self) -> String {
        (*self).to_string()
    }

    fn label(&self) -> String {
        (*self).to_string()
    }
}

/// An identifier as it arrives from Python: an integer, a float or a
/// string.  Integral floats equal the corresponding integers (`1 == 1.0`,
/// as in R), other floats compare by value, and numbers sort before
/// strings.  `NaN` is rejected at the boundary because R's `id` cannot be
/// missing.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
pub enum IdValue {
    #[cfg_attr(feature = "python", pyo3(transparent))]
    Int(i64),
    #[cfg_attr(feature = "python", pyo3(transparent))]
    Float(f64),
    #[cfg_attr(feature = "python", pyo3(transparent))]
    Str(String),
}

/// The normalised identity of an [`IdValue`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IdKey {
    Int(i64),
    /// A non-integral float, kept as its bit pattern (`-0.0` is folded
    /// into `0`, so it never occurs here).
    Float(u64),
    Str(String),
}

impl IdKey {
    fn numeric(&self) -> Option<f64> {
        match self {
            Self::Int(v) => Some(*v as f64),
            Self::Float(bits) => Some(f64::from_bits(*bits)),
            Self::Str(_) => None,
        }
    }
}

impl Ord for IdKey {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => a.cmp(b),
            (Self::Str(a), Self::Str(b)) => a.cmp(b),
            (Self::Str(_), _) => Ordering::Greater,
            (_, Self::Str(_)) => Ordering::Less,
            _ => self
                .numeric()
                .unwrap_or(f64::NAN)
                .total_cmp(&other.numeric().unwrap_or(f64::NAN)),
        }
    }
}

impl PartialOrd for IdKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl IdValue {
    /// Whether the value is a missing number, which R does not allow.
    pub fn is_missing(&self) -> bool {
        matches!(self, Self::Float(v) if v.is_nan())
    }
}

impl SubjectId for IdValue {
    type Key = IdKey;

    fn key(&self) -> IdKey {
        match self {
            Self::Int(v) => IdKey::Int(*v),
            Self::Float(v) => {
                if v.fract() == 0.0 && v.abs() < 9.007_199_254_740_992e15 {
                    IdKey::Int(*v as i64)
                } else {
                    IdKey::Float(v.to_bits())
                }
            }
            Self::Str(s) => IdKey::Str(s.clone()),
        }
    }

    fn label(&self) -> String {
        match self {
            Self::Int(v) => v.to_string(),
            Self::Float(v) => v.to_string(),
            Self::Str(s) => s.clone(),
        }
    }
}

impl Hash for IdValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}

/// R's `match(x, unique(x))` minus one: zero-based codes in order of first
/// appearance, together with one representative per code.
pub fn first_appearance_codes<I: SubjectId>(values: &[I]) -> (Vec<usize>, Vec<I>) {
    let mut seen: std::collections::HashMap<I::Key, usize> = std::collections::HashMap::new();
    let mut representatives = Vec::new();
    let codes = values
        .iter()
        .map(|value| {
            *seen.entry(value.key()).or_insert_with(|| {
                representatives.push(value.clone());
                representatives.len() - 1
            })
        })
        .collect();
    (codes, representatives)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integral_floats_equal_integers_and_numbers_sort_before_strings() {
        assert_eq!(IdValue::Float(1.0).key(), IdValue::Int(1).key());
        assert_ne!(IdValue::Float(1.5).key(), IdValue::Int(1).key());
        assert_eq!(IdValue::Float(-0.0).key(), IdValue::Int(0).key());
        assert!(IdValue::Int(2).key() < IdValue::Float(2.5).key());
        assert!(IdValue::Float(2.5).key() < IdValue::Int(3).key());
        assert!(IdValue::Int(9).key() < IdValue::Str("1".into()).key());
        assert!(IdValue::Str("a".into()).key() < IdValue::Str("b".into()).key());
        assert!(IdValue::Float(f64::NAN).is_missing());
        assert_eq!(IdValue::Str("x".into()).label(), "x");
        assert_eq!(IdValue::Float(2.5).label(), "2.5");
        assert_eq!("k".label(), "k");
    }

    #[test]
    fn first_appearance_codes_follow_r_match_unique() {
        let (codes, reps) = first_appearance_codes(&[3i64, 1, 3, 2, 1]);
        assert_eq!(codes, vec![0, 1, 0, 2, 1]);
        assert_eq!(reps, vec![3, 1, 2]);
        let (codes, _) = first_appearance_codes(&[
            IdValue::Int(1),
            IdValue::Float(1.0),
            IdValue::Str("1".into()),
        ]);
        assert_eq!(codes, vec![0, 0, 1]);
    }
}
