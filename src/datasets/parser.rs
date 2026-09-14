//! Minimal CSV tokenizer for the bundled dataset files.
//!
//! The files are written by R's `write.table(sep = ",", na = "NA")` with
//! string columns quoted, so the only syntax needed is: comma separators,
//! optional double-quoted fields with `""` as an escaped quote, and an
//! unquoted `NA` token for missing values of any type.

use std::borrow::Cow;

/// One CSV field, borrowed from the input line when no unescaping was needed.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Field<'a> {
    pub(super) text: Cow<'a, str>,
    pub(super) quoted: bool,
}

impl Field<'_> {
    /// R's `na = "NA"` writes missing values unquoted; a quoted `"NA"` is the
    /// literal string.
    pub(super) fn is_na(&self) -> bool {
        !self.quoted && (self.text.is_empty() || self.text == "NA")
    }

    pub(super) fn parse_f64(&self) -> Result<Option<f64>, String> {
        if self.is_na() {
            return Ok(None);
        }
        self.text
            .parse::<f64>()
            .map(Some)
            .map_err(|_| format!("expected a number, got {:?}", self.text))
    }

    pub(super) fn parse_i32(&self) -> Result<Option<i32>, String> {
        if self.is_na() {
            return Ok(None);
        }
        self.text
            .parse::<i32>()
            .map(Some)
            .map_err(|_| format!("expected an integer, got {:?}", self.text))
    }

    pub(super) fn parse_bool(&self) -> Result<Option<bool>, String> {
        if self.is_na() {
            return Ok(None);
        }
        match &*self.text {
            "TRUE" => Ok(Some(true)),
            "FALSE" => Ok(Some(false)),
            other => Err(format!("expected TRUE/FALSE, got {other:?}")),
        }
    }

    pub(super) fn parse_str(&self) -> Option<String> {
        if self.is_na() {
            None
        } else {
            Some(self.text.to_string())
        }
    }
}

/// Split one line into fields, appending to `out` (cleared first so the
/// caller can reuse the allocation across lines).
pub(super) fn split_line<'a>(line: &'a str, out: &mut Vec<Field<'a>>) {
    out.clear();
    let bytes = line.as_bytes();
    let mut pos = 0;
    loop {
        if bytes.get(pos) == Some(&b'"') {
            let (field, next) = quoted_field(line, pos + 1);
            out.push(field);
            pos = match bytes.get(next) {
                Some(&b',') => next + 1,
                // Anything else after the closing quote is malformed; skip to
                // the next comma rather than silently merging fields.
                _ => match line[next..].find(',') {
                    Some(offset) => next + offset + 1,
                    None => break,
                },
            };
        } else {
            let end = line[pos..].find(',').map_or(line.len(), |o| pos + o);
            out.push(Field {
                text: Cow::Borrowed(line[pos..end].trim()),
                quoted: false,
            });
            if end == line.len() {
                break;
            }
            pos = end + 1;
        }
    }
}

/// Parse a quoted field whose opening quote sits just before `start`.
/// Returns the field and the index just past the closing quote.
fn quoted_field(line: &str, start: usize) -> (Field<'_>, usize) {
    let bytes = line.as_bytes();
    let mut owned: Option<String> = None;
    let mut seg_start = start;
    let mut pos = start;
    while pos < bytes.len() {
        if bytes[pos] == b'"' {
            if bytes.get(pos + 1) == Some(&b'"') {
                let owned = owned.get_or_insert_with(String::new);
                owned.push_str(&line[seg_start..=pos]);
                pos += 2;
                seg_start = pos;
                continue;
            }
            let text = match owned {
                Some(mut s) => {
                    s.push_str(&line[seg_start..pos]);
                    Cow::Owned(s)
                }
                None => Cow::Borrowed(&line[seg_start..pos]),
            };
            return (Field { text, quoted: true }, pos + 1);
        }
        pos += 1;
    }
    // Unterminated quote: take the rest of the line.
    let text = match owned {
        Some(mut s) => {
            s.push_str(&line[seg_start..]);
            Cow::Owned(s)
        }
        None => Cow::Borrowed(&line[seg_start..]),
    };
    (Field { text, quoted: true }, line.len())
}

/// Parse the header line into column names (quotes stripped).
pub(super) fn parse_header(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    split_line(line, &mut fields);
    fields.into_iter().map(|f| f.text.into_owned()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields(line: &str) -> Vec<Field<'_>> {
        let mut out = Vec::new();
        split_line(line, &mut out);
        out
    }

    #[test]
    fn splits_plain_and_quoted_fields() {
        let f = fields(r#"1,"female",NA,"NA",,"a,b","say ""hi""""#);
        let texts: Vec<&str> = f.iter().map(|x| &*x.text).collect();
        assert_eq!(texts, ["1", "female", "NA", "NA", "", "a,b", "say \"hi\""]);
        let quoted: Vec<bool> = f.iter().map(|x| x.quoted).collect();
        assert_eq!(quoted, [false, true, false, true, false, true, true]);
        assert!(matches!(f[6].text, Cow::Owned(_)));
        assert!(matches!(f[5].text, Cow::Borrowed(_)));
    }

    #[test]
    fn na_is_only_the_unquoted_token() {
        let f = fields(r#"NA,"NA",,"""#);
        assert!(f[0].is_na());
        assert!(!f[1].is_na());
        assert!(f[2].is_na());
        assert!(!f[3].is_na());
        assert_eq!(f[1].parse_str(), Some("NA".to_string()));
        assert_eq!(f[3].parse_str(), Some(String::new()));
    }

    #[test]
    fn typed_parsing() {
        let f = fields("1.5,42,TRUE,NA,x");
        assert_eq!(f[0].parse_f64(), Ok(Some(1.5)));
        assert_eq!(f[1].parse_i32(), Ok(Some(42)));
        assert_eq!(f[2].parse_bool(), Ok(Some(true)));
        assert_eq!(f[3].parse_f64(), Ok(None));
        assert_eq!(f[3].parse_i32(), Ok(None));
        assert_eq!(f[3].parse_bool(), Ok(None));
        assert!(f[4].parse_f64().is_err());
        assert!(f[4].parse_i32().is_err());
        assert!(f[4].parse_bool().is_err());
        assert!(f[0].parse_i32().is_err());
    }

    #[test]
    fn header_strips_quotes() {
        assert_eq!(
            parse_header(r#""id","ph.ecog",time"#),
            ["id", "ph.ecog", "time"]
        );
    }

    #[test]
    fn trailing_empty_field_is_kept() {
        assert_eq!(fields("1,2,").len(), 3);
        assert_eq!(fields(r#""a","#).len(), 2);
    }
}
