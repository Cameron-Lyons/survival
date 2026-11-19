use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

pub trait SubjectId: Clone {
    type Key: Clone + Eq + Hash + Ord;

    fn key(&self) -> Self::Key;

    fn label(&self) -> String;

    /// The key as an integer, for ids that have one: two such ids have equal keys exactly
    /// when these integers are equal.  Ids that are all integers in a narrow range (the
    /// codes callers usually pass) are numbered through a table instead of a hash map.
    fn integer_key(&self) -> Option<i64> {
        None
    }
}

impl SubjectId for i64 {
    type Key = i64;

    fn key(&self) -> i64 {
        *self
    }

    fn label(&self) -> String {
        self.to_string()
    }

    fn integer_key(&self) -> Option<i64> {
        Some(*self)
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

    fn integer_key(&self) -> Option<i64> {
        i64::try_from(*self).ok()
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

fn cmp_int_float(integer: i64, bits: u64) -> Ordering {
    let float = f64::from_bits(bits);
    (integer as f64).total_cmp(&float).then_with(|| {
        // The cast of a large integer can round onto a neighboring float.
        // Use i128 here because i64::MAX itself rounds up to 2^63.
        i128::from(integer)
            .cmp(&(float as i128))
            // Public callers can construct a noncanonical Float key directly.
            .then(Ordering::Less)
    })
}

impl Ord for IdKey {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => a.cmp(b),
            (Self::Float(a), Self::Float(b)) => f64::from_bits(*a).total_cmp(&f64::from_bits(*b)),
            (Self::Int(a), Self::Float(b)) => cmp_int_float(*a, *b),
            (Self::Float(a), Self::Int(b)) => cmp_int_float(*b, *a).reverse(),
            (Self::Str(a), Self::Str(b)) => a.cmp(b),
            (Self::Str(_), _) => Ordering::Greater,
            (_, Self::Str(_)) => Ordering::Less,
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
                if v.fract() == 0.0 && (i64::MIN as f64..-(i64::MIN as f64)).contains(v) {
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

    fn integer_key(&self) -> Option<i64> {
        match self {
            Self::Str(_) => None,
            number => match number.key() {
                IdKey::Int(v) => Some(v),
                _ => None,
            },
        }
    }
}

impl Hash for IdValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}

/// R's `match(x, unique(x))` as 0-based codes, with the first id of each code.
pub fn first_appearance_codes<I: SubjectId>(values: &[I]) -> (Vec<usize>, Vec<I>) {
    dense_first_appearance_codes(values).unwrap_or_else(|| hashed_first_appearance_codes(values))
}

/// [`first_appearance_codes`] through a table indexed by the integer keys, for ids whose
/// keys span fewer than twice as many integers as there are ids; `None` otherwise.
fn dense_first_appearance_codes<I: SubjectId>(values: &[I]) -> Option<(Vec<usize>, Vec<I>)> {
    let (mut low, mut high) = (i64::MAX, i64::MIN);
    for value in values {
        let key = value.integer_key()?;
        low = low.min(key);
        high = high.max(key);
    }
    let span = usize::try_from(high.checked_sub(low)?).ok()?;
    if span >= values.len().saturating_mul(2) {
        return None;
    }
    let mut table = vec![usize::MAX; span + 1];
    let mut representatives = Vec::new();
    let codes = values
        .iter()
        .map(|value| {
            let key = value
                .integer_key()
                .expect("every key was checked to be an integer");
            let code = &mut table[(key - low) as usize];
            if *code == usize::MAX {
                *code = representatives.len();
                representatives.push(value.clone());
            }
            *code
        })
        .collect();
    Some((codes, representatives))
}

/// [`first_appearance_codes`] through a hash map.  The map starts with room for at most
/// 65536 ids and grows from there: room for every row would allocate many times what it
/// holds when the ids repeat, as they do over the rows of a subject's intervals.
fn hashed_first_appearance_codes<I: SubjectId>(values: &[I]) -> (Vec<usize>, Vec<I>) {
    let mut seen: std::collections::HashMap<I::Key, usize> =
        std::collections::HashMap::with_capacity(values.len().min(1 << 16));
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

/// [`first_appearance_codes`] of several subsets of the codes `0..levels`, one subset at a
/// time, through one table that is cleared after each: a subset then costs time in
/// proportion to its own length however many levels there are (one curve's share of the
/// clusters of a fit with many strata).
pub(crate) struct SubsetRenumbering {
    table: Vec<usize>,
}

impl SubsetRenumbering {
    pub(crate) fn new(levels: usize) -> Self {
        Self {
            table: vec![usize::MAX; levels],
        }
    }

    /// The new code of each of `codes`, and the original code of each new one.
    pub(crate) fn renumber(
        &mut self,
        codes: impl IntoIterator<Item = usize>,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut unique = Vec::new();
        let renumbered = codes
            .into_iter()
            .map(|code| {
                let slot = &mut self.table[code];
                if *slot == usize::MAX {
                    *slot = unique.len();
                    unique.push(code);
                }
                *slot
            })
            .collect();
        for &code in &unique {
            self.table[code] = usize::MAX;
        }
        (renumbered, unique)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_numeric_ids_match_only_exactly_represented_integers() {
        let large = 1_i64 << 53;
        for value in [i64::MIN, -large, large, large + 2, i64::MAX - 1023] {
            assert_eq!(
                IdValue::Float(value as f64).key(),
                IdValue::Int(value).key()
            );
        }
        assert_ne!(
            IdValue::Float(large as f64).key(),
            IdValue::Int(large + 1).key()
        );
        assert_ne!(
            IdValue::Float(i64::MAX as f64).key(),
            IdValue::Int(i64::MAX).key()
        );

        // A sorted map and a hash map must distinguish the same public keys,
        // even when a caller constructs an integral Float key directly.
        let keys = [
            IdKey::Int(large),
            IdKey::Float((large as f64).to_bits()),
            IdKey::Int(large + 1),
            IdKey::Int(i64::MAX),
            IdKey::Float((i64::MAX as f64).to_bits()),
        ];
        for (i, left) in keys.iter().enumerate() {
            for (j, right) in keys.iter().enumerate() {
                assert_eq!(left.cmp(right), i.cmp(&j));
            }
        }
    }

    /// The hash-map numbering every id type had before the table paths.
    fn reference_codes<I: SubjectId>(values: &[I]) -> (Vec<usize>, Vec<I>) {
        let mut seen = std::collections::HashMap::new();
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

    fn lcg(seed: u64) -> impl FnMut() -> u64 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state >> 33
        }
    }

    #[test]
    fn table_numbering_matches_hashed_numbering() {
        let mut next = lcg(7);
        for case in 0..60u64 {
            let n = (next() % 50) as usize;
            let spread = [1, 3, n as u64 + 1, 2 * n as u64 + 2, 1 << 40][case as usize % 5];
            let offset = [0i64, -17, i64::MIN / 2, i64::MAX - (1 << 41)][case as usize % 4];
            let ints: Vec<i64> = (0..n)
                .map(|_| offset + (next() % spread.max(1)) as i64)
                .collect();
            assert_eq!(first_appearance_codes(&ints), reference_codes(&ints));
            let codes: Vec<usize> = ints.iter().map(|&v| (v - offset) as usize).collect();
            assert_eq!(first_appearance_codes(&codes), reference_codes(&codes));
            let mixed: Vec<IdValue> = ints
                .iter()
                .map(|&v| match next() % 4 {
                    0 => IdValue::Float(v as f64),
                    1 if v % 3 == 0 => IdValue::Float(v as f64 + 0.5),
                    2 if v % 5 == 0 => IdValue::Str(v.to_string()),
                    _ => IdValue::Int(v),
                })
                .collect();
            assert_eq!(first_appearance_codes(&mixed), reference_codes(&mixed));
        }
        let extremes = [i64::MIN, i64::MAX, 0, i64::MIN, -1];
        assert_eq!(
            first_appearance_codes(&extremes),
            reference_codes(&extremes)
        );
        let wide = [usize::MAX, 0, usize::MAX, 3];
        assert_eq!(first_appearance_codes(&wide), reference_codes(&wide));
        assert_eq!(first_appearance_codes::<i64>(&[]), (Vec::new(), Vec::new()));
    }

    #[test]
    fn subset_renumbering_matches_first_appearance_codes() {
        let mut next = lcg(11);
        let levels = 40;
        let mut renumbering = SubsetRenumbering::new(levels);
        for _ in 0..50 {
            let n = (next() % 30) as usize;
            let subset: Vec<usize> = (0..n).map(|_| (next() % levels as u64) as usize).collect();
            assert_eq!(
                renumbering.renumber(subset.iter().copied()),
                first_appearance_codes(&subset)
            );
        }
        assert_eq!(renumbering.renumber([]), (Vec::new(), Vec::new()));
    }

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
