//! Box coordinates and arrows of a multi-state diagram: the layout half of
//! R's `statefig` (`R/statefig.R`).  R draws the figure; this returns what
//! it computes, the centre of each state's box (`statefig`'s invisible
//! return value) and the arrows implied by the connection matrix, so any
//! plotting front end can draw it.

use crate::error::{SurvivalError, SurvivalResult};
use pyo3::prelude::*;

/// The `layout` argument of `statefig`.
#[derive(Debug, Clone, PartialEq)]
pub enum StateFigLayout {
    /// A vector: `counts[k]` boxes in the `k`-th column, columns left to
    /// right, boxes top to bottom within a column.
    LeftToRight(Vec<usize>),
    /// A one-column matrix: `counts[k]` boxes in the `k`-th row, rows top
    /// to bottom, boxes left to right within a row.
    TopToBottom(Vec<usize>),
    /// A two-column matrix of `(x, y)` centres in `[0, 1]`, one per state.
    Coordinates(Vec<(f64, f64)>),
}

/// One arrow of the diagram.  `curvature` is `connect[from, to] - 1`: 0 for
/// a straight line, positive for an arc bending counter-clockwise, negative
/// for clockwise; `offset` marks the arrows that R shifts sideways because
/// the reverse arrow is drawn with the mirrored curvature.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct StateFigArrow {
    #[pyo3(get)]
    pub from_state: usize,
    #[pyo3(get)]
    pub to_state: usize,
    #[pyo3(get)]
    pub curvature: f64,
    #[pyo3(get)]
    pub offset: bool,
}

/// The diagram: box centres (`x`, `y` in `[0, 1]`) and arrows, in the
/// order R draws them (by destination state, then origin).
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct StateFigResult {
    #[pyo3(get)]
    pub states: Vec<String>,
    #[pyo3(get)]
    pub x: Vec<f64>,
    #[pyo3(get)]
    pub y: Vec<f64>,
    #[pyo3(get)]
    pub arrows: Vec<StateFigArrow>,
}

/// Centres of `n` boxes spread over `[0, 1]`: `(1:n - .5) / n`.
fn space(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect()
}

/// Port of the layout logic of `statefig`.  `connect` is the square
/// connection matrix (`0` = no arrow, `1` = straight, `1 + d` = an arc of
/// height `d`); `states` names its rows.
pub fn statefig(
    layout: &StateFigLayout,
    connect: &[Vec<f64>],
    states: &[String],
) -> SurvivalResult<StateFigResult> {
    let nstate = connect.len();
    if nstate == 0 || connect.iter().any(|row| row.len() != nstate) {
        return Err(SurvivalError::invalid_input(
            "connect must be a square matrix",
        ));
    }
    if states.len() != nstate {
        return Err(SurvivalError::invalid_input(
            "connect must have the state names as dimnames",
        ));
    }
    if let Some(value) = connect.iter().flatten().find(|v| !v.is_finite()) {
        return Err(SurvivalError::invalid_input(format!(
            "connect must be numeric; got {value}"
        )));
    }
    let (x, y) = match layout {
        StateFigLayout::Coordinates(centres) => {
            if centres.len() != nstate {
                return Err(SurvivalError::invalid_input(
                    "layout matrix should have one row per state",
                ));
            }
            if centres
                .iter()
                .any(|&(x, y)| !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y))
            {
                return Err(SurvivalError::invalid_input(
                    "layout coordinates must be between 0 and 1",
                ));
            }
            centres.iter().copied().unzip()
        }
        StateFigLayout::LeftToRight(counts) | StateFigLayout::TopToBottom(counts) => {
            if counts.contains(&0) {
                return Err(SurvivalError::invalid_input(
                    "non-integer number of states in layout argument",
                ));
            }
            if counts.iter().sum::<usize>() != nstate {
                return Err(SurvivalError::invalid_input(
                    "number of boxes != number of states",
                ));
            }
            let groups = space(counts.len());
            let mut x = Vec::with_capacity(nstate);
            let mut y = Vec::with_capacity(nstate);
            for (group, &count) in counts.iter().enumerate() {
                for within in space(count) {
                    match layout {
                        StateFigLayout::LeftToRight(_) => {
                            x.push(groups[group]);
                            y.push(1.0 - within);
                        }
                        _ => {
                            x.push(within);
                            y.push(1.0 - groups[group]);
                        }
                    }
                }
            }
            (x, y)
        }
    };
    let mut arrows = Vec::new();
    for j in 0..nstate {
        for (i, row) in connect.iter().enumerate() {
            if i != j && row[j] != 0.0 {
                arrows.push(StateFigArrow {
                    from_state: i,
                    to_state: j,
                    curvature: row[j] - 1.0,
                    offset: row[j] == 2.0 - connect[j][i],
                });
            }
        }
    }
    Ok(StateFigResult {
        states: states.to_vec(),
        x,
        y,
        arrows,
    })
}

/// Python binding of [`statefig`].  `layout` is R's vector form (boxes per
/// column, left to right), `column = True` turns it into the one-column
/// matrix form (boxes per row, top to bottom), and `coordinates` gives the
/// centres directly.
#[pyfunction(name = "statefig")]
#[pyo3(signature = (connect, states, layout=None, column=false, coordinates=None))]
pub fn statefig_py(
    connect: Vec<Vec<f64>>,
    states: Vec<String>,
    layout: Option<Vec<usize>>,
    column: bool,
    coordinates: Option<Vec<Vec<f64>>>,
) -> PyResult<StateFigResult> {
    let layout = match (layout, coordinates) {
        (None, Some(coordinates)) => {
            let centres = coordinates
                .iter()
                .map(|row| match row.as_slice() {
                    [x, y] => Ok((*x, *y)),
                    _ => Err(SurvivalError::invalid_input(
                        "coordinates must have two columns",
                    )),
                })
                .collect::<SurvivalResult<Vec<_>>>()?;
            StateFigLayout::Coordinates(centres)
        }
        (Some(counts), None) if column => StateFigLayout::TopToBottom(counts),
        (Some(counts), None) => StateFigLayout::LeftToRight(counts),
        _ => {
            return Err(SurvivalError::invalid_input("give either layout or coordinates").into());
        }
    };
    Ok(statefig(&layout, &connect, &states)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    fn connect3() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0],
        ]
    }

    #[test]
    fn matches_r_layouts() {
        // statefig(c(1, 2), connect) and statefig(matrix(c(1, 2), ncol = 1), connect)
        let row = statefig(
            &StateFigLayout::LeftToRight(vec![1, 2]),
            &connect3(),
            &names(&["A", "B", "C"]),
        )
        .unwrap();
        assert_eq!(row.x, vec![0.25, 0.75, 0.75]);
        assert_eq!(row.y, vec![0.5, 0.75, 0.25]);
        let column = statefig(
            &StateFigLayout::TopToBottom(vec![1, 2]),
            &connect3(),
            &names(&["A", "B", "C"]),
        )
        .unwrap();
        assert_eq!(column.x, vec![0.5, 0.25, 0.75]);
        assert_eq!(column.y, vec![0.75, 0.25, 0.25]);
        let connect4 = vec![
            vec![0.0, 1.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ];
        let four = statefig(
            &StateFigLayout::LeftToRight(vec![1, 2, 1]),
            &connect4,
            &names(&["A", "B", "C", "D"]),
        )
        .unwrap();
        assert!((four.x[0] - 1.0 / 6.0).abs() < 1e-12);
        assert_eq!(four.y, vec![0.5, 0.75, 0.25, 0.5]);
        let four_column = statefig(
            &StateFigLayout::TopToBottom(vec![1, 2, 1]),
            &connect4,
            &names(&["A", "B", "C", "D"]),
        )
        .unwrap();
        assert_eq!(four_column.x, vec![0.5, 0.25, 0.75, 0.5]);
        assert!((four_column.y[3] - 1.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn arrows_follow_the_connection_matrix() {
        let mut connect = connect3();
        connect[1][0] = 1.5; // B -> A as an arc, opposite of A -> B's 1.0
        connect[0][1] = 0.5;
        let out = statefig(
            &StateFigLayout::LeftToRight(vec![1, 2]),
            &connect,
            &names(&["A", "B", "C"]),
        )
        .unwrap();
        let arrows: Vec<(usize, usize, f64, bool)> = out
            .arrows
            .iter()
            .map(|a| (a.from_state, a.to_state, a.curvature, a.offset))
            .collect();
        assert_eq!(
            arrows,
            vec![
                (1, 0, 0.5, true),
                (0, 1, -0.5, true),
                (0, 2, 0.0, false),
                (1, 2, 0.0, false)
            ]
        );
        let coords = statefig(
            &StateFigLayout::Coordinates(vec![(0.1, 0.9), (0.5, 0.5), (0.9, 0.1)]),
            &connect3(),
            &names(&["A", "B", "C"]),
        )
        .unwrap();
        assert_eq!(coords.x, vec![0.1, 0.5, 0.9]);
    }

    #[test]
    fn rejects_bad_layouts() {
        let states = names(&["A", "B", "C"]);
        assert!(
            statefig(
                &StateFigLayout::LeftToRight(vec![1, 1]),
                &connect3(),
                &states
            )
            .is_err()
        );
        assert!(
            statefig(
                &StateFigLayout::LeftToRight(vec![0, 3]),
                &connect3(),
                &states
            )
            .is_err()
        );
        assert!(
            statefig(
                &StateFigLayout::Coordinates(vec![(0.0, 2.0); 3]),
                &connect3(),
                &states
            )
            .is_err()
        );
        assert!(
            statefig(
                &StateFigLayout::LeftToRight(vec![3]),
                &connect3(),
                &names(&["A"])
            )
            .is_err()
        );
        assert!(
            statefig(
                &StateFigLayout::LeftToRight(vec![3]),
                &vec![vec![0.0; 2]; 3],
                &states
            )
            .is_err()
        );
    }
}
