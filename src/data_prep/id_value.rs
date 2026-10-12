use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

pub trait SubjectId: Clone {
    type Key: Clone + Eq + Hash + Ord;

    fn key(&self) -> Self::Key;

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

#[derive(Debug, Clone, PartialEq)]
pub enum IdValue {
    Int(i64),
    Float(f64),
    Str(String),
}

#[cfg(feature = "python")]
impl<'py> pyo3::FromPyObject<'_, 'py> for IdValue {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'_, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::prelude::*;
        use pyo3::types::{PyFloat, PyString};

        if let Ok(text) = obj.cast::<PyString>() {
            return Ok(Self::Str(text.to_str()?.to_owned()));
        }
        if let Ok(number) = obj.cast::<PyFloat>() {
            return Ok(Self::Float(number.value()));
        }
        if let Ok(value) = obj.extract::<i64>() {
            return Ok(Self::Int(value));
        }
        if let Ok(value) = obj.extract::<f64>() {
            return Ok(Self::Float(value));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "an id must be an int, float or str, not {}",
            obj.get_type().name()?
        )))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IdKey {
    Int(i64),
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
