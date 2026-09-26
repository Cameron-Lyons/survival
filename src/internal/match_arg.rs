//! R's `match.arg(arg, choices)` for a single character argument.

use crate::error::{SurvivalError, SurvivalResult};

/// `match.arg(arg, choices)`: the index of the choice `arg` selects under
/// `pmatch` rules.  Matching is case sensitive; an exact match wins,
/// otherwise `arg` must be a prefix of exactly one choice (an ambiguous
/// prefix matches nothing), and the empty string never matches.  A miss is
/// R's error, `'arg' should be one of "a", "b", ...`.
pub(crate) fn match_arg(arg: &str, choices: &[&str]) -> SurvivalResult<usize> {
    if let Some(index) = choices.iter().position(|choice| *choice == arg) {
        return Ok(index);
    }
    let mut prefixed = choices
        .iter()
        .enumerate()
        .filter(|(_, choice)| !arg.is_empty() && choice.starts_with(arg))
        .map(|(index, _)| index);
    match (prefixed.next(), prefixed.next()) {
        (Some(index), None) => Ok(index),
        _ => Err(SurvivalError::invalid_input(format!(
            "'arg' should be one of {}",
            choices
                .iter()
                .map(|choice| format!("\"{choice}\""))
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TYPES: [&str; 7] = [
        "response",
        "link",
        "lp",
        "linear",
        "terms",
        "quantile",
        "uquantile",
    ];

    #[test]
    fn an_exact_match_wins_over_a_longer_choice_it_prefixes() {
        // residuals.survreg: "dfbeta" is itself a prefix of "dfbetas".
        let residuals = ["response", "deviance", "dfbeta", "dfbetas", "working"];
        assert_eq!(match_arg("dfbeta", &residuals).unwrap(), 2);
        assert_eq!(match_arg("dfbetas", &residuals).unwrap(), 3);
        assert!(match_arg("dfb", &residuals).is_err());
        assert_eq!(match_arg("lp", &TYPES).unwrap(), 2);
    }

    #[test]
    fn a_unique_prefix_matches() {
        assert_eq!(match_arg("q", &TYPES).unwrap(), 5);
        assert_eq!(match_arg("resp", &TYPES).unwrap(), 0);
        assert_eq!(match_arg("line", &TYPES).unwrap(), 3);
    }

    #[test]
    fn ambiguous_empty_and_case_folded_arguments_fail_with_r_message() {
        // f <- function(type = c("response", "link", "lp", "linear", "terms",
        //   "quantile", "uquantile")) match.arg(type); f("l"), f("lin"),
        // f(""), f("Response") all stop().
        for arg in ["l", "lin", "", "Response", "mystery"] {
            let err = match_arg(arg, &TYPES).unwrap_err();
            assert_eq!(
                err.to_string(),
                "'arg' should be one of \"response\", \"link\", \"lp\", \"linear\", \
                 \"terms\", \"quantile\", \"uquantile\"",
                "{arg}"
            );
        }
    }
}
