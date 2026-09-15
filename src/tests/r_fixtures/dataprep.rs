//! dataprep: fixture checks for the data-preparation and population
//! kernels (`neardate`, `aeqSurv`, `tcut`, `pyears`, `survexp` and the
//! rate tables, `tmerge`, `survSplit`, `survcondense`, `rttright`).
//!
//! The R-level bookkeeping these functions do before reaching the kernels
//! (formula evaluation, data-frame assembly) is reproduced here just far
//! enough for the fixture cases.

use super::*;
use crate::data_prep::{
    NeardateBest, Repeated, RttrightInput, SurvSplitResponse, TmergeBase, TmergeKind,
    TmergeOptions, TmergeUpdate, aeq_surv, neardate, rttright, surv2counting, survcondense,
    survsplit, tcut, tmerge_step,
};
use crate::population::{
    PyearsCategories, PyearsExpect, PyearsFollowup, PyearsRatetable, RateTable, RatetableColumn,
    SurvexpInput, SurvexpMethod, match_ratetable, pyears, ratetable_date, survexp,
    survexp_mn_table, survexp_us_table, survexp_usr_table,
};
use ndarray::Array2;

/// Numeric column of a frame.
fn column(frame: &Frame, name: &str) -> Result<Vec<f64>, String> {
    frame.get(name)?.numeric(name)
}

/// Zero-based codes of a column's levels (R's `as.numeric(factor(x)) - 1`).
fn level_codes(frame: &Frame, name: &str) -> Result<(Vec<usize>, Vec<String>), String> {
    let col = frame.get(name)?;
    let levels = col.levels();
    let codes = (0..col.len())
        .map(|i| {
            let label = col.label(i).ok_or_else(|| format!("{name}[{i}] is NA"))?;
            levels
                .iter()
                .position(|l| *l == label)
                .ok_or_else(|| format!("{name}: level {label:?} unknown"))
        })
        .collect::<Result<_, _>>()?;
    Ok((codes, levels))
}

/// Parse `c(a, b, c)`, optionally followed by `* factor`.
fn r_numeric_vector(expr: &str) -> Result<Vec<f64>, String> {
    let (inner, factor) = match expr.split_once(") *") {
        Some((head, factor)) => (
            head.trim(),
            factor.trim().parse::<f64>().map_err(|e| e.to_string())?,
        ),
        None => (expr.trim().trim_end_matches(')'), 1.0),
    };
    let inner = inner
        .trim()
        .strip_prefix("c(")
        .ok_or_else(|| format!("not a c() vector: {expr}"))?;
    inner
        .split(',')
        .map(|v| {
            v.trim()
                .parse::<f64>()
                .map(|x| x * factor)
                .map_err(|e| e.to_string())
        })
        .collect()
}

/// Parse `as.Date(c("1995-01-01", ...))` into days since 1970.
fn r_date_vector(expr: &str) -> Result<Vec<f64>, String> {
    let inner = expr
        .trim()
        .strip_prefix("as.Date(c(")
        .and_then(|rest| rest.strip_suffix("))"))
        .ok_or_else(|| format!("not an as.Date(c()) vector: {expr}"))?;
    inner
        .split(',')
        .map(|d| {
            let d = d.trim().trim_matches('"');
            let mut parts = d.split('-').map(|p| p.parse::<i64>());
            let (Some(Ok(y)), Some(Ok(m)), Some(Ok(day))) =
                (parts.next(), parts.next(), parts.next())
            else {
                return Err(format!("bad date {d}"));
            };
            ratetable_date(y as i32, m as u32, day as u32).map_err(|e| e.to_string())
        })
        .collect()
}

/// Split the top-level arguments of `f(a, b, c)`.
fn call_arguments(term: &str, function: &str) -> Option<Vec<String>> {
    let inner = term
        .trim()
        .strip_prefix(function)?
        .strip_prefix('(')?
        .strip_suffix(')')?;
    let mut args = Vec::new();
    let mut depth = 0;
    let mut current = String::new();
    let mut quoted = false;
    for ch in inner.chars() {
        match ch {
            '"' => quoted = !quoted,
            '(' if !quoted => depth += 1,
            ')' if !quoted => depth -= 1,
            ',' if depth == 0 && !quoted => {
                args.push(current.trim().to_string());
                current.clear();
                continue;
            }
            _ => {}
        }
        current.push(ch);
    }
    args.push(current.trim().to_string());
    Some(args)
}

/// The right-hand-side terms of a formula, split on top-level `+` only
/// (a `+` inside a call or a string belongs to the term).
fn formula_terms(formula: &str) -> Vec<String> {
    let rhs = formula.split_once('~').map_or("", |(_, rhs)| rhs);
    let mut terms = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    let mut quoted = false;
    for ch in rhs.chars() {
        match ch {
            '"' => quoted = !quoted,
            '(' if !quoted => depth += 1,
            ')' if !quoted => depth -= 1,
            '+' if depth == 0 && !quoted => {
                terms.push(current.trim().to_string());
                current.clear();
                continue;
            }
            _ => {}
        }
        current.push(ch);
    }
    terms.push(current.trim().to_string());
    terms.into_iter().filter(|t| !t.is_empty()).collect()
}

/// Evaluate `name` or `name + constant`.
fn arithmetic_column(frame: &Frame, expr: &str) -> Result<Vec<f64>, String> {
    if let Some((name, offset)) = expr.split_once('+') {
        let offset: f64 = offset
            .trim()
            .parse()
            .map_err(|e: std::num::ParseFloatError| e.to_string())?;
        return Ok(column(frame, name.trim())?
            .iter()
            .map(|v| v + offset)
            .collect());
    }
    if let Some((name, divisor)) = expr.split_once('/') {
        let divisor: f64 = divisor
            .trim()
            .parse()
            .map_err(|e: std::num::ParseFloatError| e.to_string())?;
        return Ok(column(frame, name.trim())?
            .iter()
            .map(|v| v / divisor)
            .collect());
    }
    column(frame, expr.trim())
}

// ---------------------------------------------------------------------------
// neardate
// ---------------------------------------------------------------------------

#[test]
fn r_fixtures_neardate() {
    let mut report = Report::new("neardate");
    let doc = load_topic("neardate");
    for case in doc["cases"].as_array().expect("cases") {
        let name = text(&case["name"]).expect("case name");
        let args = &case["args"];
        for (aspect, best, nomatch) in [
            ("after", NeardateBest::After, f64::NAN),
            ("prior", NeardateBest::Prior, f64::NAN),
            ("after_nomatch0", NeardateBest::After, 0.0),
            ("prior_nomatch0", NeardateBest::Prior, 0.0),
            ("after_dates", NeardateBest::After, f64::NAN),
        ] {
            let result = (|| -> Result<(), String> {
                let id1: Vec<i64> = nums(&args["id1"])?.into_iter().map(|v| v as i64).collect();
                let id2: Vec<i64> = nums(&args["id2"])?.into_iter().map(|v| v as i64).collect();
                let out = neardate(&id1, &nums(&args["y1"])?, &id2, &nums(&args["y2"])?, best)
                    .map_err(|err| format!("neardate: {err}"))?;
                // R returns 1-based row numbers, and `nomatch` where there is none.
                let actual: Vec<f64> = out
                    .iter()
                    .map(|idx| idx.map_or(nomatch, |i| i as f64 + 1.0))
                    .collect();
                assert_vec(&actual, &nums(&case["expected"][aspect])?, 0.0, aspect)
            })();
            report.record(name, aspect, result);
        }
    }
    report.finish();
}

// ---------------------------------------------------------------------------
// aeqSurv (utilities topic)
// ---------------------------------------------------------------------------

#[test]
fn r_fixtures_aeqsurv() {
    let mut report = Report::new("utilities");
    let doc = load_topic("utilities");
    for case in doc["cases"].as_array().expect("cases") {
        let name = text(&case["name"]).expect("case name");
        if name != "aeqSurv" {
            continue;
        }
        let args = &case["args"];
        let expected_columns = |aspect: &str| -> Result<Vec<Vec<f64>>, String> {
            let rows = matrix(&case["expected"][aspect]["values"])?;
            let ncol = rows.first().map_or(0, Vec::len);
            Ok((0..ncol)
                .map(|j| rows.iter().map(|r| r[j]).collect())
                .collect())
        };
        let result = (|| -> Result<(), String> {
            let frame = case_frame(&doc, case)?;
            let out = aeq_surv(&column(&frame, "time")?, None, None).map_err(|e| e.to_string())?;
            assert_vec(
                &out.time,
                &expected_columns("synthetic_timefix")?[0],
                1e-15,
                "time",
            )
        })();
        report.record(name, "synthetic_timefix", result);
        for (aspect, tolerance) in [("right_1e9", None), ("right_1e9_tol_1e8", Some(1e-8))] {
            let result = (|| -> Result<(), String> {
                let out =
                    aeq_surv(&nums(&args["time2"])?, None, tolerance).map_err(|e| e.to_string())?;
                assert_vec(&out.time, &expected_columns(aspect)?[0], 1e-15, aspect)
            })();
            report.record(name, aspect, result);
        }
        let result = (|| -> Result<(), String> {
            let out = aeq_surv(&nums(&args["start3"])?, Some(&nums(&args["stop3"])?), None)
                .map_err(|e| e.to_string())?;
            let expected = expected_columns("counting")?;
            assert_vec(&out.time, &expected[0], 1e-15, "start")?;
            assert_vec(&out.time2.unwrap_or_default(), &expected[1], 1e-15, "stop")
        })();
        report.record(name, "counting", result);
    }
    report.finish();
}

// ---------------------------------------------------------------------------
// pyears and tcut
// ---------------------------------------------------------------------------

/// One right-hand-side term of a `pyears` formula as a category dimension.
struct PyearsTerm {
    factor: i32,
    dim: usize,
    cuts: Vec<f64>,
    values: Vec<f64>,
    labels: Vec<String>,
}

fn pyears_term(term: &str, frame: &Frame) -> Result<PyearsTerm, String> {
    if let Some(args) = call_arguments(term, "cut") {
        // R's cut(): right-closed intervals (a, b], labelled "(a,b]".
        let values = arithmetic_column(frame, &args[0])?;
        let breaks = r_numeric_vector(&args[1])?;
        let labels: Vec<String> = breaks
            .windows(2)
            .map(|w| format!("({},{}]", format_r_number(w[0]), format_r_number(w[1])))
            .collect();
        let codes = values
            .iter()
            .map(|&v| {
                let k = breaks.partition_point(|&b| b < v);
                if k == 0 || k > breaks.len() - 1 {
                    Err(format!("cut(): value {v} outside the breaks"))
                } else {
                    Ok(k as f64)
                }
            })
            .collect::<Result<Vec<f64>, String>>()?;
        return Ok(PyearsTerm {
            factor: 1,
            dim: labels.len(),
            cuts: Vec::new(),
            values: codes,
            labels,
        });
    }
    if let Some(args) = call_arguments(term, "tcut") {
        let values = column(frame, &args[0])?;
        let breaks = if args[1].starts_with("as.Date") {
            r_date_vector(&args[1])?
        } else {
            r_numeric_vector(&args[1])?
        };
        let labels = args.get(2).and_then(|labels| {
            let inner = labels
                .trim()
                .strip_prefix("labels = c(")?
                .strip_suffix(')')?;
            Some(
                inner
                    .split(',')
                    .map(|l| l.trim().trim_matches('"').to_string())
                    .collect::<Vec<String>>(),
            )
        });
        let cut = tcut(&values, &breaks, labels.as_deref(), 1.0).map_err(|e| e.to_string())?;
        return Ok(PyearsTerm {
            factor: 0,
            dim: cut.cutpoints.len() - 1,
            cuts: cut.cutpoints,
            values: cut.values,
            labels: cut.labels,
        });
    }
    let (codes, labels) = level_codes(frame, term)?;
    Ok(PyearsTerm {
        factor: 1,
        dim: labels.len(),
        cuts: Vec::new(),
        values: codes.iter().map(|&c| c as f64 + 1.0).collect(),
        labels,
    })
}

/// The `rmap = list(age = agedays, ...)` argument as `match_ratetable` columns.
fn rmap_columns(rmap: &str, frame: &Frame) -> Result<(Vec<String>, Vec<RatetableColumn>), String> {
    let args = call_arguments(rmap, "list").ok_or("rmap is not a list()")?;
    let n = frame.nrow();
    let mut names = Vec::new();
    let mut columns = Vec::new();
    for arg in args {
        let (name, expr) = arg.split_once('=').ok_or("rmap entry without =")?;
        names.push(name.trim().to_string());
        let expr = expr.trim();
        columns.push(if let Some(label) = expr.strip_prefix('"') {
            RatetableColumn::Labels(vec![label.trim_end_matches('"').to_string(); n])
        } else {
            RatetableColumn::Numeric(column(frame, expr)?)
        });
    }
    Ok((names, columns))
}

fn ratetable_by_name(name: &str) -> Result<&'static RateTable, String> {
    match name {
        "survexp.us" => Ok(survexp_us_table()),
        "survexp.usr" => Ok(survexp_usr_table()),
        "survexp.mn" => Ok(survexp_mn_table()),
        other => unsupported(format!("ratetable {other}")),
    }
}

/// Compare a column-major table against the fixture's row-major encoding
/// (a vector for one dimension, nested rows for two).
fn assert_table(
    actual: &[f64],
    dims: &[usize],
    expected: &Value,
    rtol: f64,
    path: &str,
) -> Result<(), String> {
    match dims.len() {
        1 => assert_vec(actual, &nums(expected)?, rtol, path),
        2 => {
            let rows: Vec<Vec<f64>> = (0..dims[0])
                .map(|i| (0..dims[1]).map(|j| actual[i + dims[0] * j]).collect())
                .collect();
            assert_matrix(&rows, &matrix(expected)?, rtol, path)
        }
        _ => unsupported("harness: tables with more than two dimensions"),
    }
}

#[test]
fn r_fixtures_pyears() {
    let mut report = Report::new("pyears");
    let doc = load_topic("pyears");
    for case in doc["cases"].as_array().expect("cases") {
        let name = text(&case["name"]).expect("case name");
        let expected = &case["expected"];
        if name == "tcut_basis" {
            let args = &case["args"];
            let result = (|| -> Result<(), String> {
                let x = nums(&args["x"])?;
                let breaks = nums(&args["breaks"])?;
                let plain = tcut(&x, &breaks, None, 1.0).map_err(|e| e.to_string())?;
                assert_vec(&plain.values, &nums(&expected["values"])?, 0.0, "values")?;
                assert_vec(
                    &plain.cutpoints,
                    &nums(&expected["cutpoints"])?,
                    0.0,
                    "cutpoints",
                )?;
                if plain.labels != names_of(&expected["labels"]) {
                    return Err(format!(
                        "labels: {:?} != {:?}",
                        plain.labels,
                        names_of(&expected["labels"])
                    ));
                }
                let labels: Vec<String> = names_of(&expected["labelled"])
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                let labelled = tcut(&x, &breaks, Some(&labels), 1.0).map_err(|e| e.to_string())?;
                if labelled.labels != labels {
                    return Err("labelled: explicit labels not kept".to_string());
                }
                let three = tcut(&x, &[3.0], None, 1.0).map_err(|e| e.to_string())?;
                assert_vec(
                    &three.cutpoints,
                    &nums(&expected["scalar_breaks_cutpoints"])?,
                    1e-12,
                    "scalar_breaks_cutpoints",
                )?;
                assert_vec(
                    &three.values,
                    &nums(&expected["scalar_breaks_values"])?,
                    0.0,
                    "scalar_breaks_values",
                )
            })();
            report.record(name, "tcut", result);
            continue;
        }

        let prepared = (|| -> Result<_, String> {
            let frame = case_frame(&doc, case)?;
            let formula = text(&case["formula"]).ok_or("no formula")?;
            let args = &case["args"];
            let (lhs, _) = formula.split_once('~').ok_or("formula without ~")?;
            let lhs = lhs.trim();
            let followup = if let Some(surv) = call_arguments(lhs, "Surv") {
                let status = status_column(&surv[1], &frame)?;
                PyearsFollowup {
                    start: None,
                    stop: arithmetic_column(&frame, &surv[0])?,
                    event: Some(status.iter().map(|&s| f64::from(s)).collect()),
                }
            } else {
                PyearsFollowup {
                    start: None,
                    stop: arithmetic_column(&frame, lhs)?,
                    event: None,
                }
            };
            let terms: Vec<PyearsTerm> = formula_terms(formula)
                .iter()
                .map(|term| pyears_term(term, &frame))
                .collect::<Result<_, _>>()?;
            let n = frame.nrow();
            let mut data = Array2::<f64>::zeros((n, terms.len()));
            for (j, term) in terms.iter().enumerate() {
                for i in 0..n {
                    data[[i, j]] = term.values[i];
                }
            }
            let categories = PyearsCategories {
                factors: terms.iter().map(|t| t.factor).collect(),
                dims: terms.iter().map(|t| t.dim).collect(),
                cuts: terms.iter().map(|t| t.cuts.clone()).collect(),
                data,
            };
            let ratetable = match text(&args["ratetable"]) {
                Some(table_name) => {
                    let table = ratetable_by_name(table_name)?;
                    let (names, columns) =
                        rmap_columns(text(&args["rmap"]).ok_or("no rmap")?, &frame)?;
                    let positions =
                        match_ratetable(table, &names, &columns).map_err(|e| e.to_string())?;
                    Some(PyearsRatetable { table, positions })
                }
                None => None,
            };
            let weights = arg_nums(case, "weights");
            let scale = arg_nums(case, "scale").map_or(365.25, |s| s[0]);
            let out = pyears(
                &followup,
                weights.as_deref(),
                &categories,
                ratetable,
                PyearsExpect::Event,
                scale,
            )
            .map_err(|e| format!("pyears: {e}"))?;
            Ok((out, terms))
        })();
        let (out, terms) = match prepared {
            Ok(value) => value,
            Err(message) => {
                for aspect in [
                    "pyears",
                    "n",
                    "event",
                    "expected",
                    "offtable",
                    "observations",
                    "dimnames",
                ] {
                    if !expected[aspect].is_null() {
                        report.record(name, aspect, Err(message.clone()));
                    }
                }
                continue;
            }
        };
        let data_frame = is_true(&case["args"]["data.frame"]);
        if data_frame {
            // The data.frame output keeps the cells with person-years, in
            // column-major order.
            let keep: Vec<usize> = (0..out.pyears.len())
                .filter(|&c| out.pyears[c] > 0.0)
                .collect();
            let df = &expected["data"]["columns"];
            for (aspect, values) in [("pyears", &out.pyears), ("n", &out.n)] {
                let actual: Vec<f64> = keep.iter().map(|&c| values[c]).collect();
                let result = nums(&df[aspect]).and_then(|e| assert_vec(&actual, &e, 1e-8, aspect));
                report.record(name, aspect, result);
            }
            if let Some(event) = &out.event {
                let actual: Vec<f64> = keep.iter().map(|&c| event[c]).collect();
                let result = nums(&df["event"]).and_then(|e| assert_vec(&actual, &e, 0.0, "event"));
                report.record(name, "event", result);
            }
            report.record(
                name,
                "offtable",
                assert_scalar(
                    out.offtable,
                    num(&expected["offtable"]).unwrap_or(f64::NAN),
                    1e-8,
                    "offtable",
                ),
            );
            report.record(
                name,
                "observations",
                assert_scalar(
                    out.observations as f64,
                    num(&expected["observations"]).unwrap_or(f64::NAN),
                    0.0,
                    "observations",
                ),
            );
            continue;
        }
        report.record(
            name,
            "pyears",
            assert_table(&out.pyears, &out.dims, &expected["pyears"], 1e-8, "pyears"),
        );
        report.record(
            name,
            "n",
            assert_table(&out.n, &out.dims, &expected["n"], 0.0, "n"),
        );
        if !expected["event"].is_null() {
            let result = match &out.event {
                Some(event) => assert_table(event, &out.dims, &expected["event"], 0.0, "event"),
                None => Err("no event table".to_string()),
            };
            report.record(name, "event", result);
        }
        if !expected["expected"].is_null() {
            let result = match &out.expected {
                Some(values) => {
                    assert_table(values, &out.dims, &expected["expected"], 1e-8, "expected")
                }
                None => Err("no expected table".to_string()),
            };
            report.record(name, "expected", result);
        }
        report.record(
            name,
            "offtable",
            assert_scalar(
                out.offtable,
                num(&expected["offtable"]).unwrap_or(f64::NAN),
                1e-8,
                "offtable",
            ),
        );
        report.record(
            name,
            "observations",
            assert_scalar(
                out.observations as f64,
                num(&expected["observations"]).unwrap_or(f64::NAN),
                0.0,
                "observations",
            ),
        );
        if let Some(dimnames) = expected["dimnames"].as_object() {
            let result = (|| -> Result<(), String> {
                let dims = nums(&expected["dim"])?;
                if dims.len() != out.dims.len()
                    || dims.iter().zip(&out.dims).any(|(e, a)| *e as usize != *a)
                {
                    return Err(format!("dim: {:?} != {dims:?}", out.dims));
                }
                // JSON objects do not keep R's order: match terms by name.
                let formula = text(&case["formula"]).unwrap_or("");
                for (term, term_text) in terms.iter().zip(formula_terms(formula)) {
                    let labels = dimnames
                        .get(&term_text)
                        .ok_or_else(|| format!("dimnames: no entry for {term_text:?}"))?;
                    let expected_labels = names_of(labels);
                    if term.labels != expected_labels {
                        return Err(format!(
                            "dimnames: {:?} != {expected_labels:?}",
                            term.labels
                        ));
                    }
                }
                Ok(())
            })();
            report.record(name, "dimnames", result);
        }
    }
    report.finish();
}

// ---------------------------------------------------------------------------
// survexp, ratetableDate, and the census tables
// ---------------------------------------------------------------------------

#[test]
fn r_fixtures_survexp() {
    let mut report = Report::new("survexp");
    let doc = load_topic("survexp");
    for case in doc["cases"].as_array().expect("cases") {
        let name = text(&case["name"]).expect("case name");
        let expected = &case["expected"];
        let args = &case["args"];
        match name {
            "ratetableDate" => {
                let result = (|| -> Result<(), String> {
                    let dates = names_of(&args["dates"]);
                    let actual: Vec<f64> = dates
                        .iter()
                        .map(|d| {
                            let parts: Vec<i64> =
                                d.split('-').map(|p| p.parse().unwrap_or(0)).collect();
                            ratetable_date(parts[0] as i32, parts[1] as u32, parts[2] as u32)
                                .map_err(|e| e.to_string())
                        })
                        .collect::<Result<_, _>>()?;
                    assert_vec(&actual, &nums(&expected["from_date"])?, 0.0, "from_date")?;
                    assert_vec(
                        &nums(&args["numeric"])?,
                        &nums(&expected["from_numeric"])?,
                        0.0,
                        "from_numeric",
                    )
                })();
                report.record(name, "from_date", result);
                continue;
            }
            "survexp_us_table" => {
                let table = survexp_us_table();
                report.record(
                    name,
                    "dim",
                    nums(&expected["dim"]).and_then(|d| {
                        assert_vec(
                            &table.dims.iter().map(|&x| x as f64).collect::<Vec<_>>(),
                            &d,
                            0.0,
                            "dim",
                        )
                    }),
                );
                report.record(
                    name,
                    "type",
                    nums(&expected["type"]).and_then(|t| {
                        assert_vec(
                            &table
                                .types
                                .iter()
                                .map(|x| f64::from(x.code()))
                                .collect::<Vec<_>>(),
                            &t,
                            0.0,
                            "type",
                        )
                    }),
                );
                let check_dimnames = |table: &RateTable, names: &Value| -> Result<(), String> {
                    let names = names.as_object().ok_or("no dimnames")?;
                    if names.len() != table.ndim() {
                        return Err(format!(
                            "{} dimnames, expected {}",
                            table.ndim(),
                            names.len()
                        ));
                    }
                    for (d, id) in table.dimid.iter().enumerate() {
                        let labels = names
                            .get(id)
                            .ok_or_else(|| format!("no dimnames for {id}"))?;
                        if table.dimnames[d] != names_of(labels) {
                            return Err(format!("dimnames[{d}] differ"));
                        }
                    }
                    Ok(())
                };
                report.record(
                    name,
                    "dimnames",
                    check_dimnames(table, &expected["dimnames"]),
                );
                let result = (|| -> Result<(), String> {
                    let cuts = expected["cutpoints"].as_array().ok_or("no cutpoints")?;
                    for (d, cut) in cuts.iter().enumerate() {
                        match (&table.cutpoints[d], cut) {
                            (None, Value::Null) => {}
                            (Some(actual), values) => {
                                assert_vec(actual, &nums(values)?, 0.0, &format!("cutpoints[{d}]"))?
                            }
                            _ => return Err(format!("cutpoints[{d}]: factor/continuous mismatch")),
                        }
                    }
                    Ok(())
                })();
                report.record(name, "cutpoints", result);
                let result = (|| -> Result<(), String> {
                    let sample = &expected["sample"];
                    let at = |labels: &[&str]| -> f64 {
                        let index: Vec<usize> = labels
                            .iter()
                            .enumerate()
                            .map(|(d, l)| table.level(d, l).unwrap_or(usize::MAX))
                            .collect();
                        table.rate(&index).unwrap_or(f64::NAN)
                    };
                    // The fixture stores 15 significant digits.
                    assert_scalar(
                        at(&["0", "male", "1990"]),
                        num(&sample["age0_male_1990"])?,
                        1e-14,
                        "age0_male_1990",
                    )?;
                    assert_scalar(
                        at(&["50", "female", "2000"]),
                        num(&sample["age50_female_2000"])?,
                        1e-14,
                        "age50_female_2000",
                    )?;
                    assert_scalar(
                        at(&["100", "male", "1940"]),
                        num(&sample["age100_male_1940"])?,
                        1e-14,
                        "age100_male_1940",
                    )?;
                    assert_scalar(table.rates.iter().sum(), num(&sample["sum"])?, 1e-10, "sum")
                })();
                report.record(name, "sample", result);
                let result = (|| -> Result<(), String> {
                    let summary = &expected["summary"];
                    let usr = survexp_usr_table();
                    let mn = survexp_mn_table();
                    assert_vec(
                        &usr.dims.iter().map(|&x| x as f64).collect::<Vec<_>>(),
                        &nums(&summary["usr_dim"])?,
                        0.0,
                        "usr_dim",
                    )?;
                    assert_vec(
                        &mn.dims.iter().map(|&x| x as f64).collect::<Vec<_>>(),
                        &nums(&summary["mn_dim"])?,
                        0.0,
                        "mn_dim",
                    )?;
                    assert_scalar(
                        usr.rates.iter().sum(),
                        num(&summary["usr_sum"])?,
                        1e-10,
                        "usr_sum",
                    )?;
                    assert_scalar(
                        mn.rates.iter().sum(),
                        num(&summary["mn_sum"])?,
                        1e-10,
                        "mn_sum",
                    )?;
                    check_dimnames(usr, &summary["usr_dimnames"])?;
                    check_dimnames(mn, &summary["mn_dimnames"])
                })();
                report.record(name, "summary", result);
                continue;
            }
            "lung_coxph_ratetable" => {
                for aspect in ["by_sex", "overall", "individual"] {
                    report.record(
                        name,
                        aspect,
                        unsupported("survexp with a coxph fit as ratetable is Python-side"),
                    );
                }
                continue;
            }
            _ => {}
        }

        let individual = name.contains("individual");
        let computed = (|| -> Result<_, String> {
            let frame = case_frame(&doc, case)?;
            let formula = text(&case["formula"]).ok_or("no formula")?;
            let (lhs, rhs) = formula.split_once('~').ok_or("formula without ~")?;
            let lhs = lhs.trim();
            let y = if lhs.is_empty() {
                None
            } else if let Some(surv) = call_arguments(lhs, "Surv") {
                Some(column(&frame, &surv[0])?)
            } else {
                Some(column(&frame, lhs)?)
            };
            let rhs = rhs.trim();
            let group = if rhs == "1" {
                None
            } else {
                Some(level_codes(&frame, rhs)?.0)
            };
            let table = ratetable_by_name(text(&args["ratetable"]).ok_or("no ratetable")?)?;
            let (names, columns) = rmap_columns(text(&args["rmap"]).ok_or("no rmap")?, &frame)?;
            let positions = match_ratetable(table, &names, &columns).map_err(|e| e.to_string())?;
            let times = arg_nums(case, "times");
            let method = text(&args["method"])
                .map(SurvexpMethod::parse)
                .transpose()
                .map_err(|e| e.to_string())?;
            let cohort = args["cohort"].as_bool().unwrap_or(true);
            survexp(
                table,
                SurvexpInput {
                    positions: &positions,
                    y: y.as_deref(),
                    group: group.as_deref(),
                    times: times.as_deref(),
                    method,
                    cohort,
                    conditional: false,
                    scale: 1.0,
                },
            )
            .map_err(|e| format!("survexp: {e}"))
        })();
        let out = match computed {
            Ok(out) => out,
            Err(message) => {
                let aspects: &[&str] = if individual {
                    &["surv"]
                } else {
                    &["time", "n", "surv", "n_risk", "method"]
                };
                for aspect in aspects {
                    report.record(name, aspect, Err(message.clone()));
                }
                continue;
            }
        };
        if individual {
            let actual: Vec<f64> = out.surv.iter().map(|row| row[0]).collect();
            report.record(
                name,
                "surv",
                nums(&expected["surv"]).and_then(|e| assert_vec(&actual, &e, 1e-8, "surv")),
            );
            continue;
        }
        report.record(
            name,
            "time",
            nums(&expected["time"]).and_then(|e| assert_vec(&out.time, &e, 1e-12, "time")),
        );
        // n is the column-major flattening of the ntime x ngroup matrix.
        let n_groups = out.surv.first().map_or(1, Vec::len);
        let n_flat: Vec<f64> = (0..n_groups)
            .flat_map(|g| out.n_risk.iter().map(move |row| row[g]))
            .collect();
        report.record(
            name,
            "n",
            nums(&expected["n"]).and_then(|e| assert_vec(&n_flat, &e, 0.0, "n")),
        );
        let surv_result = if n_groups == 1 {
            let actual: Vec<f64> = out.surv.iter().map(|row| row[0]).collect();
            nums(&expected["surv"]).and_then(|e| assert_vec(&actual, &e, 1e-8, "surv"))
        } else {
            matrix(&expected["surv"]).and_then(|e| assert_matrix(&out.surv, &e, 1e-8, "surv"))
        };
        report.record(name, "surv", surv_result);
        let risk_result = if n_groups == 1 {
            let actual: Vec<f64> = out.n_risk.iter().map(|row| row[0]).collect();
            nums(&expected["n_risk"]).and_then(|e| assert_vec(&actual, &e, 0.0, "n_risk"))
        } else {
            matrix(&expected["n_risk"]).and_then(|e| assert_matrix(&out.n_risk, &e, 0.0, "n_risk"))
        };
        report.record(name, "n_risk", risk_result);
        let method_result = match text(&expected["method"]) {
            Some(m) if m == out.method => Ok(()),
            other => Err(format!("method: {:?} != {other:?}", out.method)),
        };
        report.record(name, "method", method_result);
    }
    report.finish();
}

// ---------------------------------------------------------------------------
// tmerge
// ---------------------------------------------------------------------------

/// The state R's `tmerge` keeps in `newdata`: which row of the base data
/// each row came from, the interval, and the variables tmerge created.
#[derive(Clone)]
struct TmergeData {
    row: Vec<usize>,
    id: Vec<f64>,
    start: Vec<f64>,
    stop: Vec<f64>,
    /// Created variables (numeric; NaN for NA) and whether they are events.
    variables: Vec<(String, Vec<f64>, bool)>,
}

impl TmergeData {
    fn expand(&self, rows: &[usize], start: &[f64], stop: &[f64], censor_rows: &[usize]) -> Self {
        let mut variables = Vec::new();
        for (name, values, is_event) in &self.variables {
            let mut expanded: Vec<f64> = rows.iter().map(|&r| values[r]).collect();
            if *is_event {
                for &r in censor_rows {
                    expanded[r] = 0.0;
                }
            }
            variables.push((name.clone(), expanded, *is_event));
        }
        TmergeData {
            row: rows.iter().map(|&r| self.row[r]).collect(),
            id: rows.iter().map(|&r| self.id[r]).collect(),
            start: start.to_vec(),
            stop: stop.to_vec(),
            variables,
        }
    }

    fn variable(&self, name: &str) -> Option<&Vec<f64>> {
        self.variables
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, v, _)| v)
    }

    fn set_variable(&mut self, name: &str, values: Vec<f64>, is_event: bool) {
        if let Some(entry) = self.variables.iter_mut().find(|(n, _, _)| n == name) {
            entry.1 = values;
            entry.2 = is_event;
        } else {
            self.variables.push((name.to_string(), values, is_event));
        }
    }

    /// Apply one `name = kind(time, value)` argument, as R's loop does.
    fn apply(
        &mut self,
        name: &str,
        kind: TmergeKind,
        update: (&[f64], &[f64], Option<&[f64]>),
        default: f64,
        tcount: &mut Vec<Vec<usize>>,
    ) -> Result<(), String> {
        self.apply_with(
            name,
            kind,
            update,
            default,
            &TmergeOptions::default(),
            tcount,
        )
    }

    /// [`Self::apply`] with explicit `tmerge` options (`na.rm`, `delay`).
    fn apply_with(
        &mut self,
        name: &str,
        kind: TmergeKind,
        update: (&[f64], &[f64], Option<&[f64]>),
        default: f64,
        options: &TmergeOptions,
        tcount: &mut Vec<Vec<usize>>,
    ) -> Result<(), String> {
        let (update_id, update_time, update_value) = update;
        let prior = self.variable(name).cloned();
        let ids: Vec<i64> = self.id.iter().map(|&v| v as i64).collect();
        let update_ids: Vec<i64> = update_id.iter().map(|&v| v as i64).collect();
        let step = tmerge_step(
            &TmergeBase {
                id: &ids,
                start: &self.start,
                stop: &self.stop,
            },
            &TmergeUpdate {
                id: &update_ids,
                time: update_time,
                value: update_value,
                missing: None,
            },
            kind,
            options,
            if kind == TmergeKind::Cumtdc {
                prior.as_deref()
            } else {
                None
            },
            default,
        )
        .map_err(|e| format!("tmerge_step({name}): {e}"))?;
        tcount.push(step.tcount.clone());
        let mut expanded = self.expand(&step.row, &step.start, &step.stop, &step.censor_rows);
        let n = expanded.row.len();
        match kind {
            TmergeKind::Tdc => {
                let values = step
                    .source
                    .iter()
                    .map(|s| match (s, update_value) {
                        (Some(k), Some(v)) => v[*k],
                        (Some(_), None) => 1.0,
                        (None, Some(_)) => default,
                        (None, None) => 0.0,
                    })
                    .collect();
                expanded.set_variable(name, values, false);
            }
            TmergeKind::Cumtdc => expanded.set_variable(name, step.cumulative.clone(), false),
            TmergeKind::Event | TmergeKind::Cumevent => {
                let mut values = match expanded.variable(name) {
                    Some(existing) => existing.clone(),
                    None => vec![0.0; n],
                };
                for (r, v) in step.event_row.iter().zip(&step.event_value) {
                    values[*r] = *v;
                }
                expanded.set_variable(name, values, true);
            }
        }
        *self = expanded;
        Ok(())
    }
}

/// Compare a `TmergeData` against a fixture frame: base columns through
/// the row map, the interval, and the created variables.
fn check_tmerge_frame(
    data: &TmergeData,
    base: &Frame,
    expected: &Value,
    path: &str,
) -> Result<(), String> {
    let columns = expected["columns"]
        .as_object()
        .ok_or("frame without columns")?;
    let nrow = expected["nrow"].as_u64().unwrap_or(0) as usize;
    if data.row.len() != nrow {
        return Err(format!("{path}: {} rows, expected {nrow}", data.row.len()));
    }
    for (name, values) in columns {
        let actual: Vec<f64> = match name.as_str() {
            "tstart" => data.start.clone(),
            "tstop" => data.stop.clone(),
            _ => {
                if let Some(created) = data.variable(name) {
                    created.clone()
                } else {
                    match base.get(name)? {
                        Column::Num(v) => data.row.iter().map(|&r| v[r]).collect(),
                        Column::Str(..) => continue,
                    }
                }
            }
        };
        let expected_values: Vec<f64> = values
            .as_array()
            .ok_or("column is not an array")?
            .iter()
            .map(|v| if v.is_string() { Ok(f64::NAN) } else { num(v) })
            .collect::<Result<_, _>>()?;
        if values
            .as_array()
            .is_some_and(|items| items.iter().all(Value::is_string))
        {
            continue;
        }
        assert_vec(&actual, &expected_values, 1e-12, &format!("{path}.{name}"))?;
    }
    Ok(())
}

fn check_tcount(tcount: &[Vec<usize>], expected: &Value, path: &str) -> Result<(), String> {
    let actual: Vec<Vec<f64>> = tcount
        .iter()
        .map(|row| row.iter().map(|&v| v as f64).collect())
        .collect();
    assert_matrix(&actual, &matrix(&expected["values"])?, 0.0, path)
}

/// The three stages of the cgd0 vignette and its tcount rows.
type CgdSteps = (TmergeData, TmergeData, TmergeData, Vec<Vec<usize>>);

/// R's first `tmerge` call: `data1` rows with `(tstart, tstop)`; without
/// `tstop` the first event time of each id sets the range.
fn tmerge_first_call(
    base: &Frame,
    id: &[f64],
    tstop: Option<&[f64]>,
) -> Result<TmergeData, String> {
    let n = base.nrow();
    let tstop = tstop
        .map(<[f64]>::to_vec)
        .ok_or("tstop is required for the harness")?;
    Ok(TmergeData {
        row: (0..n).collect(),
        id: id.to_vec(),
        start: vec![0.0; n],
        stop: tstop,
        variables: Vec::new(),
    })
}

#[test]
fn r_fixtures_tmerge() {
    let mut report = Report::new("tmerge");
    let doc = load_topic("tmerge");
    for case in doc["cases"].as_array().expect("cases") {
        let name = text(&case["name"]).expect("case name");
        let expected = &case["expected"];
        match name {
            "cgd0_vignette" => {
                let run = (|| -> Result<CgdSteps, String> {
                    let cgd0 = load_dataset("cgd0")?;
                    let id = column(&cgd0, "id")?;
                    let mut data = tmerge_first_call(&cgd0, &id, Some(&column(&cgd0, "futime")?))?;
                    let base = data.clone();
                    let mut tcount = Vec::new();
                    for k in 1..=7 {
                        let etime = column(&cgd0, &format!("etime{k}"))?;
                        data.apply(
                            "infect",
                            TmergeKind::Event,
                            (&id, &etime, None),
                            f64::NAN,
                            &mut tcount,
                        )?;
                    }
                    let after_events = data.clone();
                    let update_id = data.id.clone();
                    let update_time = data.start.clone();
                    data.apply(
                        "enum",
                        TmergeKind::Cumtdc,
                        (&update_id, &update_time, None),
                        f64::NAN,
                        &mut tcount,
                    )?;
                    Ok((base, after_events, data, tcount))
                })();
                match run {
                    Ok((base, after_events, final_data, tcount)) => {
                        let cgd0 = load_dataset("cgd0").expect("cgd0");
                        report.record(
                            name,
                            "after_base",
                            check_tmerge_frame(&base, &cgd0, &expected["after_base"], "after_base"),
                        );
                        report.record(
                            name,
                            "after_events",
                            check_tmerge_frame(
                                &after_events,
                                &cgd0,
                                &expected["after_events"],
                                "after_events",
                            ),
                        );
                        report.record(
                            name,
                            "final",
                            check_tmerge_frame(&final_data, &cgd0, &expected["final"], "final"),
                        );
                        report.record(
                            name,
                            "tcount",
                            check_tcount(&tcount, &expected["tcount"], "tcount"),
                        );
                    }
                    Err(message) => {
                        for aspect in ["after_base", "after_events", "final", "tcount"] {
                            report.record(name, aspect, Err(message.clone()));
                        }
                    }
                }
            }
            "synthetic_tdc_cumtdc_event_cumevent" => {
                let run = (|| -> Result<Vec<(&str, TmergeData)>, String> {
                    let base = inline_frame(&doc["data"]["tmerge_base"])?;
                    let long = inline_frame(&doc["data"]["tmerge_long"])?;
                    let id = column(&base, "id")?;
                    let long_id = column(&long, "id")?;
                    let time = column(&long, "time")?;
                    let lab = column(&long, "lab")?;
                    let infection = column(&long, "infection")?;
                    let mut tcount = Vec::new();
                    // death = event(futime, death): the first call, range from the event.
                    let mut d1 = tmerge_first_call(&base, &id, Some(&column(&base, "futime")?))?;
                    d1.apply(
                        "death",
                        TmergeKind::Event,
                        (
                            &id,
                            &column(&base, "futime")?,
                            Some(&column(&base, "death")?),
                        ),
                        f64::NAN,
                        &mut tcount,
                    )?;
                    let mut d2 = d1.clone();
                    d2.apply(
                        "lab",
                        TmergeKind::Tdc,
                        (&long_id, &time, Some(&lab)),
                        f64::NAN,
                        &mut tcount,
                    )?;
                    let mut d3 = d2.clone();
                    d3.apply(
                        "nlab",
                        TmergeKind::Cumtdc,
                        (&long_id, &time, None),
                        f64::NAN,
                        &mut tcount,
                    )?;
                    let mut d4 = d3.clone();
                    d4.apply(
                        "infect",
                        TmergeKind::Event,
                        (&long_id, &time, Some(&infection)),
                        f64::NAN,
                        &mut tcount,
                    )?;
                    let mut d5 = d4.clone();
                    d5.apply(
                        "ninfect",
                        TmergeKind::Cumevent,
                        (&long_id, &time, Some(&infection)),
                        f64::NAN,
                        &mut tcount,
                    )?;
                    let mut d6 = d1.clone();
                    let mut ignored = Vec::new();
                    d6.apply(
                        "lab",
                        TmergeKind::Tdc,
                        (&long_id, &time, Some(&lab)),
                        0.5,
                        &mut ignored,
                    )?;
                    let mut d7 = d1.clone();
                    d7.apply(
                        "lab",
                        TmergeKind::Tdc,
                        (&long_id, &time, Some(&lab)),
                        -1.0,
                        &mut ignored,
                    )?;
                    let tcount_data = TmergeData {
                        row: Vec::new(),
                        id: Vec::new(),
                        start: Vec::new(),
                        stop: Vec::new(),
                        variables: tcount
                            .iter()
                            .map(|row| {
                                (
                                    "tcount".to_string(),
                                    row.iter().map(|&v| v as f64).collect(),
                                    false,
                                )
                            })
                            .collect(),
                    };
                    Ok(vec![
                        ("step1_death_event", d1),
                        ("step2_lab_tdc", d2),
                        ("step3_nlab_cumtdc", d3),
                        ("step4_infect_event", d4),
                        ("step5_ninfect_cumevent", d5),
                        ("tdc_init", d6),
                        ("tdc_tdcstart", d7),
                        ("tcount_final", tcount_data),
                    ])
                })();
                match run {
                    Ok(steps) => {
                        let base = inline_frame(&doc["data"]["tmerge_base"]).expect("base");
                        for (aspect, data) in steps {
                            let result = if aspect == "tcount_final" {
                                let rows: Vec<Vec<usize>> = data
                                    .variables
                                    .iter()
                                    .map(|(_, v, _)| v.iter().map(|&x| x as usize).collect())
                                    .collect();
                                check_tcount(&rows, &expected[aspect], aspect)
                            } else {
                                check_tmerge_frame(&data, &base, &expected[aspect], aspect)
                            };
                            report.record(name, aspect, result);
                        }
                    }
                    Err(message) => {
                        for aspect in [
                            "step1_death_event",
                            "step2_lab_tdc",
                            "step3_nlab_cumtdc",
                            "step4_infect_event",
                            "step5_ninfect_cumevent",
                            "tdc_init",
                            "tdc_tdcstart",
                            "tcount_final",
                        ] {
                            report.record(name, aspect, Err(message.clone()));
                        }
                    }
                }
            }
            "pbcseq_20_vignette" => {
                let run = (|| -> Result<(TmergeData, Vec<Vec<usize>>), String> {
                    let pbc = inline_frame(&doc["data"]["pbc_20"])?;
                    let seq = inline_frame(&doc["data"]["pbcseq_20"])?;
                    let id = column(&pbc, "id")?;
                    let seq_id = column(&seq, "id")?;
                    let day = column(&seq, "day")?;
                    let mut tcount = Vec::new();
                    let mut data = tmerge_first_call(&pbc, &id, Some(&column(&pbc, "time")?))?;
                    let death: Vec<f64> = column(&pbc, "status")?
                        .iter()
                        .map(|&s| f64::from(s == 2.0))
                        .collect();
                    data.apply(
                        "death",
                        TmergeKind::Event,
                        (&id, &column(&pbc, "time")?, Some(&death)),
                        f64::NAN,
                        &mut tcount,
                    )?;
                    for variable in ["bili", "albumin", "protime", "edema"] {
                        let values = column(&seq, variable)?;
                        data.apply(
                            variable,
                            TmergeKind::Tdc,
                            (&seq_id, &day, Some(&values)),
                            f64::NAN,
                            &mut tcount,
                        )?;
                    }
                    Ok((data, tcount))
                })();
                match run {
                    Ok((data, tcount)) => {
                        let pbc = inline_frame(&doc["data"]["pbc_20"]).expect("pbc_20");
                        report.record(
                            name,
                            "frame",
                            check_tmerge_frame(&data, &pbc, &expected["frame"], "frame"),
                        );
                        report.record(
                            name,
                            "tcount",
                            check_tcount(&tcount, &expected["tcount"], "tcount"),
                        );
                    }
                    Err(message) => {
                        report.record(name, "frame", Err(message.clone()));
                        report.record(name, "tcount", Err(message));
                    }
                }
            }
            other => report.record(other, "frame", unsupported("harness: unknown tmerge case")),
        }
    }
    report.finish();
}

/// `tcount` rows as a `TmergeData` so the aspects of a case share one type.
fn tcount_data(tcount: &[Vec<usize>]) -> TmergeData {
    TmergeData {
        row: Vec::new(),
        id: Vec::new(),
        start: Vec::new(),
        stop: Vec::new(),
        variables: tcount
            .iter()
            .map(|row| {
                (
                    "tcount".to_string(),
                    row.iter().map(|&v| v as f64).collect(),
                    false,
                )
            })
            .collect(),
    }
}

/// Record every aspect of a `dataprep-tmerge` case: frames are compared
/// against the base data, `tcount*` aspects against the tcount matrix.
fn record_tmerge_steps(
    report: &mut Report,
    name: &str,
    base: &Frame,
    expected: &Value,
    run: Result<Vec<(&str, TmergeData)>, String>,
    aspects: &[&str],
) {
    match run {
        Ok(steps) => {
            for (aspect, data) in steps {
                let result = if aspect.starts_with("tcount") {
                    let rows: Vec<Vec<usize>> = data
                        .variables
                        .iter()
                        .map(|(_, v, _)| v.iter().map(|&x| x as usize).collect())
                        .collect();
                    check_tcount(&rows, &expected[aspect], aspect)
                } else {
                    check_tmerge_frame(&data, base, &expected[aspect], aspect)
                };
                report.record(name, aspect, result);
            }
        }
        Err(message) => {
            for aspect in aspects {
                report.record(name, aspect, Err(message.clone()));
            }
        }
    }
}

/// Last value carried forward across a subject's later intervals
/// (tmerge2's `k--`): tdc arguments whose update times miss some interval
/// starts, after an event split and with NA rows dropped by `na.rm`.
#[test]
fn r_fixtures_dataprep_tmerge() {
    let mut report = Report::new("dataprep-tmerge");
    let doc = load_topic("dataprep-tmerge");
    let na_kept = TmergeOptions {
        na_rm: false,
        ..TmergeOptions::default()
    };
    for case in doc["cases"].as_array().expect("cases") {
        let name = text(&case["name"]).expect("case name");
        let expected = &case["expected"];
        match name {
            "lvcf_after_event_split" => {
                let base = inline_frame(&doc["data"]["tmerge_lvcf_base"]).expect("base");
                let run = (|| -> Result<Vec<(&str, TmergeData)>, String> {
                    let long = inline_frame(&doc["data"]["tmerge_lvcf_long"])?;
                    let id = column(&base, "id")?;
                    let futime = column(&base, "futime")?;
                    let long_id = column(&long, "id")?;
                    let time = column(&long, "time")?;
                    let x = column(&long, "x")?;
                    let visit = column(&long, "visit")?;
                    let mut tcount = Vec::new();
                    let mut d1 = tmerge_first_call(&base, &id, Some(&futime))?;
                    d1.apply(
                        "death",
                        TmergeKind::Event,
                        (&id, &futime, Some(&column(&base, "death")?)),
                        f64::NAN,
                        &mut tcount,
                    )?;
                    let mut d2 = d1.clone();
                    d2.apply(
                        "visit",
                        TmergeKind::Event,
                        (&long_id, &time, Some(&visit)),
                        f64::NAN,
                        &mut tcount,
                    )?;
                    let mut d3 = d2.clone();
                    d3.apply(
                        "x",
                        TmergeKind::Tdc,
                        (&long_id, &time, Some(&x)),
                        f64::NAN,
                        &mut tcount,
                    )?;
                    let mut d4 = d3.clone();
                    d4.apply(
                        "nx",
                        TmergeKind::Cumtdc,
                        (&long_id, &time, Some(&x)),
                        f64::NAN,
                        &mut tcount,
                    )?;
                    let mut d5 = d4.clone();
                    d5.apply(
                        "seen",
                        TmergeKind::Tdc,
                        (&long_id, &time, None),
                        f64::NAN,
                        &mut tcount,
                    )?;
                    let mut ignored = Vec::new();
                    let mut d6 = d2.clone();
                    d6.apply_with(
                        "x",
                        TmergeKind::Tdc,
                        (&long_id, &time, Some(&x)),
                        f64::NAN,
                        &na_kept,
                        &mut ignored,
                    )?;
                    let mut d7 = d2.clone();
                    d7.apply(
                        "x",
                        TmergeKind::Tdc,
                        (&long_id, &time, Some(&x)),
                        0.0,
                        &mut ignored,
                    )?;
                    Ok(vec![
                        ("step2_visit_event", d2),
                        ("step3_x_tdc", d3),
                        ("step4_nx_cumtdc", d4),
                        ("step5_seen_tdc", d5),
                        ("x_tdc_na_kept", d6),
                        ("x_tdc_init_0", d7),
                        ("tcount_step5", tcount_data(&tcount)),
                    ])
                })();
                record_tmerge_steps(
                    &mut report,
                    name,
                    &base,
                    expected,
                    run,
                    &[
                        "step2_visit_event",
                        "step3_x_tdc",
                        "step4_nx_cumtdc",
                        "step5_seen_tdc",
                        "x_tdc_na_kept",
                        "x_tdc_init_0",
                        "tcount_step5",
                    ],
                );
            }
            "pbcseq_20_chol_na" => {
                let pbc = inline_frame(&doc["data"]["pbc_20"]).expect("pbc_20");
                let run = (|| -> Result<Vec<(&str, TmergeData)>, String> {
                    let seq = inline_frame(&doc["data"]["pbcseq_20_chol"])?;
                    let id = column(&pbc, "id")?;
                    let time = column(&pbc, "time")?;
                    let seq_id = column(&seq, "id")?;
                    let day = column(&seq, "day")?;
                    let mut tcount = Vec::new();
                    let mut data = tmerge_first_call(&pbc, &id, Some(&time))?;
                    let death: Vec<f64> = column(&pbc, "status")?
                        .iter()
                        .map(|&s| f64::from(s == 2.0))
                        .collect();
                    data.apply(
                        "death",
                        TmergeKind::Event,
                        (&id, &time, Some(&death)),
                        f64::NAN,
                        &mut tcount,
                    )?;
                    let mut chol_na_kept = data.clone();
                    let mut ignored = Vec::new();
                    chol_na_kept.apply_with(
                        "chol",
                        TmergeKind::Tdc,
                        (&seq_id, &day, Some(&column(&seq, "chol")?)),
                        f64::NAN,
                        &na_kept,
                        &mut ignored,
                    )?;
                    for variable in ["bili", "chol", "ascites", "hepato"] {
                        let values = column(&seq, variable)?;
                        data.apply(
                            variable,
                            TmergeKind::Tdc,
                            (&seq_id, &day, Some(&values)),
                            f64::NAN,
                            &mut tcount,
                        )?;
                    }
                    Ok(vec![
                        ("frame", data),
                        ("tcount", tcount_data(&tcount)),
                        ("chol_na_kept", chol_na_kept),
                    ])
                })();
                record_tmerge_steps(
                    &mut report,
                    name,
                    &pbc,
                    expected,
                    run,
                    &["frame", "tcount", "chol_na_kept"],
                );
            }
            other => report.record(other, "frame", unsupported("harness: unknown tmerge case")),
        }
    }
    report.finish();
}

// ---------------------------------------------------------------------------
// survSplit
// ---------------------------------------------------------------------------

#[test]
fn r_fixtures_survsplit() {
    let mut report = Report::new("survSplit");
    let doc = load_topic("survSplit");
    for case in doc["cases"].as_array().expect("cases") {
        let name = text(&case["name"]).expect("case name");
        let args = &case["args"];
        let expected = &case["expected"]["frame"];
        let result = (|| -> Result<(), String> {
            let frame = case_frame(&doc, case)?;
            let formula = text(&case["formula"]).ok_or("no formula")?;
            let lhs = formula.split('~').next().ok_or("formula without ~")?.trim();
            let surv = call_arguments(lhs, "Surv").ok_or("response is not Surv()")?;
            // Multi-state status: factor codes with 0 = censor.
            let status_name = surv.last().ok_or("empty Surv")?;
            let status: Vec<f64> = match frame.get(status_name)? {
                Column::Num(_) => status_column(status_name, &frame)?
                    .iter()
                    .map(|&s| f64::from(s))
                    .collect(),
                Column::Str(..) => level_codes(&frame, status_name)?
                    .0
                    .iter()
                    .map(|&c| c as f64)
                    .collect(),
            };
            let cut = arg_nums(case, "cut").ok_or("no cut")?;
            let zero = arg_nums(case, "zero").map_or(0.0, |z| z[0]);
            let (start_name, stop_name) = match surv.len() {
                2 => (None, surv[0].clone()),
                3 => (Some(surv[0].clone()), surv[1].clone()),
                _ => return unsupported("harness: Surv() arity"),
            };
            let stop = column(&frame, &stop_name)?;
            let start = start_name.as_ref().map(|s| column(&frame, s)).transpose()?;
            let response: SurvSplitResponse<'_, i64> = match &start {
                Some(start) => SurvSplitResponse::Counting {
                    start,
                    stop: &stop,
                    status: &status,
                },
                None => SurvSplitResponse::Right {
                    time: &stop,
                    status: &status,
                },
            };
            let out =
                survsplit(response, &cut, zero, true).map_err(|e| format!("survsplit: {e}"))?;

            let nrow = expected["nrow"].as_u64().unwrap_or(0) as usize;
            if out.row.len() != nrow {
                return Err(format!("{} rows, expected {nrow}", out.row.len()));
            }
            // R names the output columns after the Surv() variables, except
            // that (time, status) data gets a new "tstart".
            let default_start = start_name.as_deref().unwrap_or("tstart");
            let out_start_name = text(&args["start"]).unwrap_or(default_start);
            let out_stop_name = text(&args["end"]).unwrap_or(&stop_name);
            let out_event_name = text(&args["event"]).unwrap_or(status_name);
            let columns = expected["columns"]
                .as_object()
                .ok_or("frame without columns")?;
            for (col_name, values) in columns {
                let actual: Vec<f64> = if col_name == out_start_name {
                    out.start.clone()
                } else if col_name == out_stop_name {
                    out.end.clone()
                } else if col_name == out_event_name {
                    out.status.clone()
                } else if Some(col_name.as_str()) == text(&args["episode"]) {
                    out.interval.iter().map(|&i| i as f64 + 1.0).collect()
                } else if Some(col_name.as_str()) == text(&args["id"]) {
                    out.row.iter().map(|&r| r as f64 + 1.0).collect()
                } else {
                    match frame.get(col_name)? {
                        Column::Num(v) => out.row.iter().map(|&r| v[r]).collect(),
                        Column::Str(..) => {
                            let (codes, _) = level_codes(&frame, col_name)?;
                            let labels = names_of(values);
                            let expected_codes = level_codes_of_labels(&frame, col_name, &labels)?;
                            let actual_codes: Vec<f64> =
                                out.row.iter().map(|&r| codes[r] as f64).collect();
                            assert_vec(&actual_codes, &expected_codes, 0.0, col_name)?;
                            continue;
                        }
                    }
                };
                let expected_values: Vec<f64> = if values
                    .as_array()
                    .is_some_and(|items| items.iter().any(Value::is_string))
                {
                    // A factor status column: map the labels back to codes.
                    level_codes_of_labels(&frame, status_name, &names_of(values))?
                } else {
                    nums(values)?
                };
                assert_vec(&actual, &expected_values, 1e-12, col_name)?;
            }
            Ok(())
        })();
        report.record(name, "frame", result);
    }
    report.finish();
}

/// Codes of `labels` within the levels of a frame's string column.
fn level_codes_of_labels(frame: &Frame, name: &str, labels: &[&str]) -> Result<Vec<f64>, String> {
    let levels = frame.get(name)?.levels();
    labels
        .iter()
        .map(|label| {
            levels
                .iter()
                .position(|l| l == label)
                .map(|p| p as f64)
                .ok_or_else(|| format!("{name}: level {label:?} unknown"))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// survcondense
// ---------------------------------------------------------------------------

#[test]
fn r_fixtures_survcondense() {
    let mut report = Report::new("survcondense");
    let doc = load_topic("survcondense");
    for case in doc["cases"].as_array().expect("cases") {
        let name = text(&case["name"]).expect("case name");
        let args = &case["args"];
        let expected = &case["expected"]["frame"];
        let result = (|| -> Result<(), String> {
            let frame = case_frame(&doc, case)?;
            let formula = text(&case["formula"]).ok_or("no formula")?;
            let lhs = formula.split('~').next().ok_or("formula without ~")?.trim();
            let surv = call_arguments(lhs, "Surv").ok_or("response is not Surv()")?;
            let id_name = text(&args["id"]).ok_or("no id")?;
            let id: Vec<i64> = column(&frame, id_name)?.iter().map(|&v| v as i64).collect();
            let start = column(&frame, &surv[0])?;
            let stop = column(&frame, &surv[1])?;
            let status = column(&frame, &surv[2])?;
            // The covariates (and id) decide what may merge; the status does not.
            let terms = rhs_terms(formula);
            let mut signature: Vec<Vec<String>> = vec![Vec::new(); frame.nrow()];
            for term in terms.iter().chain(std::iter::once(&id_name.to_string())) {
                let col = frame.get(term)?;
                for (i, row) in signature.iter_mut().enumerate() {
                    row.push(col.label(i).unwrap_or_else(|| "NA".to_string()));
                }
            }
            let mut codes: std::collections::HashMap<Vec<String>, i64> =
                std::collections::HashMap::new();
            let row_code: Vec<i64> = signature
                .iter()
                .map(|s| {
                    let next = codes.len() as i64;
                    *codes.entry(s.clone()).or_insert(next)
                })
                .collect();
            let out = survcondense(&id, &start, &stop, &row_code)
                .map_err(|e| format!("survcondense: {e}"))?;

            let nrow = expected["nrow"].as_u64().unwrap_or(0) as usize;
            if out.keep.len() != nrow {
                return Err(format!("{} rows, expected {nrow}", out.keep.len()));
            }
            let start_name = text(&args["start"]).unwrap_or("tstart");
            let stop_name = text(&args["end"]).unwrap_or(&surv[1]);
            let event_name = text(&args["event"]).unwrap_or(&surv[2]);
            let columns = expected["columns"]
                .as_object()
                .ok_or("frame without columns")?;
            for (col_name, values) in columns {
                let actual: Vec<f64> = if col_name == start_name {
                    out.keep.iter().map(|&k| out.start[k]).collect()
                } else if col_name == stop_name {
                    out.keep.iter().map(|&k| stop[k]).collect()
                } else if col_name == event_name {
                    out.keep.iter().map(|&k| status[k]).collect()
                } else if col_name == "1" {
                    // R names the id column after the first id value (a do.call quirk).
                    out.keep.iter().map(|&k| id[k] as f64).collect()
                } else {
                    match frame.get(col_name)? {
                        Column::Num(v) => out.keep.iter().map(|&k| v[k]).collect(),
                        Column::Str(..) => {
                            let (codes, _) = level_codes(&frame, col_name)?;
                            let expected_codes =
                                level_codes_of_labels(&frame, col_name, &names_of(values))?;
                            let actual_codes: Vec<f64> =
                                out.keep.iter().map(|&k| codes[k] as f64).collect();
                            assert_vec(&actual_codes, &expected_codes, 0.0, col_name)?;
                            continue;
                        }
                    }
                };
                assert_vec(&actual, &nums(values)?, 1e-12, col_name)?;
            }
            Ok(())
        })();
        report.record(name, "frame", result);
    }
    report.finish();
}

// ---------------------------------------------------------------------------
// rttright
// ---------------------------------------------------------------------------

#[test]
fn r_fixtures_rttright() {
    let mut report = Report::new("rttright");
    let doc = load_topic("rttright");
    for case in doc["cases"].as_array().expect("cases") {
        let name = text(&case["name"]).expect("case name");
        let expected = &case["expected"];
        let result = (|| -> Result<(), String> {
            let frame = case_frame(&doc, case)?;
            let formula = text(&case["formula"]).ok_or("no formula")?;
            let (lhs, rhs) = formula.split_once('~').ok_or("formula without ~")?;
            let surv = call_arguments(lhs.trim(), "Surv").ok_or("response is not Surv()")?;
            let time = column(&frame, &surv[0])?;
            let status: Vec<i32> = match frame.get(&surv[1])? {
                Column::Num(_) => status_column(&surv[1], &frame)?,
                Column::Str(..) => level_codes(&frame, &surv[1])?
                    .0
                    .iter()
                    .map(|&c| c as i32)
                    .collect(),
            };
            let rhs = rhs.trim();
            let strata = if rhs == "1" {
                None
            } else {
                Some(level_codes(&frame, rhs)?.0)
            };
            let weights = arg_nums(case, "weights");
            let times = arg_nums(case, "times");
            let out = rttright::<i64>(RttrightInput {
                start: None,
                time: &time,
                status: &status,
                strata: strata.as_deref(),
                weights: weights.as_deref(),
                id: None,
                times: times.as_deref(),
                timefix: true,
                renorm: true,
            })
            .map_err(|e| format!("rttright: {e}"))?;
            if times.is_some() {
                assert_vec(&out.times, &nums(&expected["times"])?, 0.0, "times")?;
                assert_matrix(
                    &out.weights,
                    &matrix(&expected["weights"])?,
                    1e-8,
                    "weights",
                )
            } else {
                let actual: Vec<f64> = out.weights.iter().map(|row| row[0]).collect();
                assert_vec(&actual, &nums(&expected["weights"])?, 1e-8, "weights")
            }
        })();
        report.record(name, "weights", result);
    }
    report.finish();
}

// ---------------------------------------------------------------------------
// Surv2 timeline data: no fixture topic yet, so a documented round trip.
// ---------------------------------------------------------------------------

#[test]
fn surv2counting_round_trip_holds_for_the_fixture_datasets() {
    let cgd = load_dataset("cgd").expect("cgd");
    let id: Vec<i64> = column(&cgd, "id")
        .unwrap()
        .iter()
        .map(|&v| v as i64)
        .collect();
    let tstart = column(&cgd, "tstart").unwrap();
    let tstop = column(&cgd, "tstop").unwrap();
    let status: Vec<i32> = column(&cgd, "status")
        .unwrap()
        .iter()
        .map(|&s| s as i32)
        .collect();
    let istate = vec![1; id.len()];
    let timeline = crate::data_prep::totimeline(&id, &tstart, &tstop, &status, &istate).unwrap();
    let ids: Vec<i64> = timeline.time_row.iter().map(|&r| id[r]).collect();
    let states: Vec<Option<i32>> = timeline.state.iter().map(|&s| Some(s)).collect();
    let back = surv2counting(&ids, &timeline.time, &states, true, Repeated::Yes, &[]).unwrap();
    assert_eq!(back.tstart, tstart);
    assert_eq!(back.tstop, tstop);
    let back_status: Vec<i32> = back.status.iter().map(|s| s.unwrap_or(0)).collect();
    assert_eq!(back_status, status);
    assert!(back.counting);
}
