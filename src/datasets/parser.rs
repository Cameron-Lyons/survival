//! Simple CSV parser for embedded survival datasets
//!
//! Handles:
//! - Header row detection
//! - NA/missing value handling
//! - Numeric and string columns

/// Parse CSV data into rows of string values
pub fn parse_csv(data: &str) -> Result<(Vec<String>, Vec<Vec<String>>), String> {
    let mut lines = data.lines().peekable();

    let header_line = lines.next().ok_or("Empty CSV")?;
    let headers: Vec<String> = header_line
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .collect();

    let num_cols = headers.len();
    let mut rows = Vec::new();

    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let cols: Vec<String> = parse_csv_line(trimmed);

        if cols.len() != num_cols {
            return Err(format!(
                "Column count mismatch: expected {}, got {} in line: {}",
                num_cols,
                cols.len(),
                trimmed
            ));
        }
        rows.push(cols);
    }

    Ok((headers, rows))
}

/// Parse a single CSV line, handling quoted fields
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '"' => {
                if in_quotes {
                    if chars.peek() == Some(&'"') {
                        current.push('"');
                        chars.next();
                    } else {
                        in_quotes = false;
                    }
                } else {
                    in_quotes = true;
                }
            }
            ',' if !in_quotes => {
                result.push(current.trim().to_string());
                current = String::new();
            }
            _ => {
                current.push(c);
            }
        }
    }
    result.push(current.trim().to_string());

    result
}

/// Extract a column by index from parsed rows
#[allow(dead_code)]
pub fn extract_column(rows: &[Vec<String>], index: usize) -> Vec<String> {
    rows.iter()
        .map(|row| row.get(index).cloned().unwrap_or_default())
        .collect()
}

/// Parse a string value to f64, treating NA as NaN
pub fn parse_f64(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("na") || s.eq_ignore_ascii_case("nan") {
        None
    } else {
        s.parse().ok()
    }
}

/// Parse a string value to i32, treating NA as None
pub fn parse_i32(s: &str) -> Option<i32> {
    let s = s.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("na") || s.eq_ignore_ascii_case("nan") {
        None
    } else {
        s.parse()
            .ok()
            .or_else(|| s.parse::<f64>().ok().map(|f| f as i32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_csv() {
        let csv = "a,b,c\n1,2,3\n4,5,6";
        let (headers, rows) = parse_csv(csv).unwrap();
        assert_eq!(headers, vec!["a", "b", "c"]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec!["1", "2", "3"]);
    }

    #[test]
    fn test_parse_na_values() {
        assert_eq!(parse_f64("NA"), None);
        assert_eq!(parse_f64("1.5"), Some(1.5));
        assert_eq!(parse_i32("NA"), None);
        assert_eq!(parse_i32("42"), Some(42));
    }
}
