//! Column-typed parsing of the bundled R `survival` datasets.
//!
//! Every dataset is embedded as the CSV produced from R's data frame and
//! described by a [`Dataset`] entry whose schema records R's storage mode
//! for each column, so the Python dictionaries round-trip R's types:
//! `integer` -> `int`, `double` -> `float`, `factor`/`character`/`Date` ->
//! `str` (dates as ISO-8601 `YYYY-MM-DD`), `logical` -> `bool`, and `NA` ->
//! `None` in every column. Doubles are written as the shortest decimal that a
//! correctly rounded parser (such as `str::parse::<f64>`) reads back as the
//! identical double, so every value is a bit-exact copy of R's.

use super::parser::{Field, parse_header, split_line};
use crate::error::{SurvivalError, SurvivalResult};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// R storage mode of a dataset column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ColType {
    /// R `double`.
    Float,
    /// R `integer`.
    Int,
    /// R `factor`, `character`, or `Date` (formatted `YYYY-MM-DD`).
    Str,
    /// R `logical`.
    Bool,
}

/// One bundled dataset: its R name, embedded CSV text, and column schema in
/// R's column order.
pub(crate) struct Dataset {
    /// The name used by R, e.g. `rhDNase`.
    pub(crate) name: &'static str,
    pub(crate) csv: &'static str,
    pub(crate) schema: &'static [(&'static str, ColType)],
}

/// A parsed column; `None` marks R's `NA`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Column {
    Float(Vec<Option<f64>>),
    Int(Vec<Option<i32>>),
    Str(Vec<Option<String>>),
    Bool(Vec<Option<bool>>),
}

impl Column {
    fn with_capacity(col_type: ColType, capacity: usize) -> Self {
        match col_type {
            ColType::Float => Self::Float(Vec::with_capacity(capacity)),
            ColType::Int => Self::Int(Vec::with_capacity(capacity)),
            ColType::Str => Self::Str(Vec::with_capacity(capacity)),
            ColType::Bool => Self::Bool(Vec::with_capacity(capacity)),
        }
    }

    fn push(&mut self, field: &Field<'_>) -> Result<(), String> {
        match self {
            Self::Float(values) => values.push(field.parse_f64()?),
            Self::Int(values) => values.push(field.parse_i32()?),
            Self::Str(values) => values.push(field.parse_str()),
            Self::Bool(values) => values.push(field.parse_bool()?),
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Float(values) => values.len(),
            Self::Int(values) => values.len(),
            Self::Str(values) => values.len(),
            Self::Bool(values) => values.len(),
        }
    }

    #[cfg(test)]
    pub(crate) fn na_count(&self) -> usize {
        match self {
            Self::Float(values) => values.iter().filter(|v| v.is_none()).count(),
            Self::Int(values) => values.iter().filter(|v| v.is_none()).count(),
            Self::Str(values) => values.iter().filter(|v| v.is_none()).count(),
            Self::Bool(values) => values.iter().filter(|v| v.is_none()).count(),
        }
    }

    fn to_pylist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        match self {
            Self::Float(values) => PyList::new(py, values),
            Self::Int(values) => PyList::new(py, values),
            Self::Str(values) => PyList::new(py, values.iter().map(Option::as_deref)),
            Self::Bool(values) => PyList::new(py, values),
        }
    }
}

/// A column-oriented copy of an R data frame.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DataFrame {
    pub(crate) nrow: usize,
    pub(crate) columns: Vec<(&'static str, Column)>,
}

impl DataFrame {
    pub(crate) fn ncol(&self) -> usize {
        self.columns.len()
    }

    #[cfg(test)]
    pub(crate) fn column(&self, name: &str) -> Option<&Column> {
        self.columns
            .iter()
            .find(|(col, _)| *col == name)
            .map(|(_, column)| column)
    }
}

impl Dataset {
    /// Parse the embedded CSV into typed columns following the schema.
    pub(crate) fn parse(&self) -> SurvivalResult<DataFrame> {
        let mut lines = self.csv.lines();
        let header = lines
            .next()
            .map(parse_header)
            .ok_or_else(|| self.error("empty CSV"))?;
        let ncol = header.len();

        let mut indices = Vec::with_capacity(self.schema.len());
        for &(name, _) in self.schema {
            let idx = header
                .iter()
                .position(|h| h == name)
                .ok_or_else(|| self.error(format!("column '{name}' not found in CSV")))?;
            indices.push(idx);
        }

        let capacity = self.csv.len() / ncol.max(1) / 4;
        let mut columns: Vec<(&'static str, Column)> = self
            .schema
            .iter()
            .map(|&(name, col_type)| (name, Column::with_capacity(col_type, capacity)))
            .collect();

        let mut fields = Vec::with_capacity(ncol);
        let mut nrow = 0;
        for (line_no, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            split_line(line, &mut fields);
            if fields.len() != ncol {
                return Err(self.error(format!(
                    "line {}: expected {ncol} fields, got {}",
                    line_no + 2,
                    fields.len()
                )));
            }
            for (&idx, (name, column)) in indices.iter().zip(columns.iter_mut()) {
                column.push(&fields[idx]).map_err(|message| {
                    self.error(format!("line {}, column '{name}': {message}", line_no + 2))
                })?;
            }
            nrow += 1;
        }

        Ok(DataFrame { nrow, columns })
    }

    /// Build the Python dictionary of column lists plus `_nrow`/`_ncol`.
    pub(crate) fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let frame = self.parse()?;
        let dict = PyDict::new(py);
        for (name, column) in &frame.columns {
            dict.set_item(*name, column.to_pylist(py)?)?;
        }
        dict.set_item("_nrow", frame.nrow)?;
        dict.set_item("_ncol", frame.ncol())?;
        Ok(dict.into())
    }

    fn error(&self, message: impl std::fmt::Display) -> SurvivalError {
        SurvivalError::invalid_input(format!("dataset '{}': {message}", self.name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOY: Dataset = Dataset {
        name: "toy",
        csv: "\"id\",\"score\",\"label\",\"flag\"\n1,1.5,\"a\",TRUE\n2,NA,NA,FALSE\n3,-2,\"NA\",NA\n",
        schema: &[
            ("flag", ColType::Bool),
            ("id", ColType::Int),
            ("score", ColType::Float),
            ("label", ColType::Str),
        ],
    };

    #[test]
    fn parses_typed_columns_in_schema_order() {
        let frame = TOY.parse().unwrap();
        assert_eq!(frame.nrow, 3);
        assert_eq!(frame.ncol(), 4);
        let names: Vec<&str> = frame.columns.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, ["flag", "id", "score", "label"]);
        assert_eq!(
            frame.column("flag"),
            Some(&Column::Bool(vec![Some(true), Some(false), None]))
        );
        assert_eq!(
            frame.column("id"),
            Some(&Column::Int(vec![Some(1), Some(2), Some(3)]))
        );
        assert_eq!(
            frame.column("score"),
            Some(&Column::Float(vec![Some(1.5), None, Some(-2.0)]))
        );
        assert_eq!(
            frame.column("label"),
            Some(&Column::Str(vec![
                Some("a".to_string()),
                None,
                Some("NA".to_string())
            ]))
        );
        assert_eq!(frame.column("score").unwrap().na_count(), 1);
        assert_eq!(frame.column("missing"), None);
    }

    #[test]
    fn rejects_missing_column_and_bad_values() {
        let missing = Dataset {
            name: "toy",
            csv: "a\n1\n",
            schema: &[("b", ColType::Int)],
        };
        assert!(missing.parse().is_err());

        let bad_int = Dataset {
            name: "toy",
            csv: "a\n1.5\n",
            schema: &[("a", ColType::Int)],
        };
        let err = bad_int.parse().unwrap_err().to_string();
        assert!(err.contains("line 2, column 'a'"), "{err}");

        let ragged = Dataset {
            name: "toy",
            csv: "a,b\n1\n",
            schema: &[("a", ColType::Int)],
        };
        assert!(ragged.parse().is_err());
    }
}
