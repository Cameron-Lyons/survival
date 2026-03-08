use chrono::{NaiveDate, Duration};
use std::cmp::Ordering;

#[derive(Debug, Clone)]
enum DimensionKind {
    Discrete,
    Continuous,
}

#[derive(Debug, Clone)]
struct Dimension {
    kind: DimensionKind,
    cutpoints: Vec<f64>,
}

#[derive(Debug)]
struct Ratetable {
    dim_sizes: Vec<usize>,
    dimensions: Vec<Dimension>,
    rates: Vec<f64>,
}

#[derive(Debug, PartialEq)]
struct RateWalkResult {
    cells: Vec<Vec<usize>>,
    days: Vec<f64>,
    hazards: Vec<f64>,
}

fn compute_linear_index(cell: &[usize], dim_sizes: &[usize]) -> Result<usize, String> {
    if cell.len() != dim_sizes.len() {
        return Err("Cell length doesn't match dimensions".to_string());
    }
    
    let mut index = 0;
    let mut stride = 1;
    for (i, (&c, &d)) in cell.iter().zip(dim_sizes.iter()).enumerate().rev() {
        if c >= d {
            return Err(format!("Cell index {} out of bounds for dimension {} (size {})", c, i, d));
        }
        index += c * stride;
        stride *= d;
    }
    Ok(index)
}

fn ratewalk(start: &[f64], mut futime: f64, ratetable: &Ratable) -> Result<RateWalkResult, String> {
    if start.len() != ratetable.dimensions.len() {
        return Err("Start length doesn't match ratetable dimensions".to_string());
    }
    if futime <= 0.0 {
        return Err("futime must be positive".to_string());
    }

    const EPS: f64 = 1e-8;
    let mut current_start = start.to_vec();
    let mut cells = Vec::new();
    let mut days = Vec::new();
    let mut hazards = Vec::new();

    while futime > 0.0 {
        let mut cell = vec![0; ratetable.dimensions.len()];
        let mut edge = futime;

        for (i, dim) in ratetable.dimensions.iter().enumerate() {
            match dim.kind {
                DimensionKind::Discrete => {
                    cell[i] = current_start[i] as usize;
                }
                DimensionKind::Continuous => {
                    let adj_value = current_start[i] + EPS;
                    let count = dim.cutpoints.partition_point(|cp| cp <= &adj_value);
                    cell[i] = count;
                    
                    if count < dim.cutpoints.len() {
                        let time_to_next = dim.cutpoints[count] - current_start[i];
                        if time_to_next < edge {
                            edge = time_to_next;
                        }
                    }
                }
            }
        }

        let linear_index = compute_linear_index(&cell, &ratetable.dim_sizes)?;
        let rate = *ratetable.rates.get(linear_index).unwrap_or(&0.0);
        let hazard = edge * rate;

        days.push(edge);
        hazards.push(hazard);
        cells.push(cell.clone());

        for (i, dim) in ratetable.dimensions.iter().enumerate() {
            if let DimensionKind::Continuous = dim.kind {
                current_start[i] += edge;
            }
        }
        futime -= edge;
    }

    Ok(RateWalkResult { cells, days, hazards })
}

fn mdy_date(m: u32, d: u32, y: i32) -> NaiveDate {
    let y = if y < 100 { y + 1900 } else { y };
    NaiveDate::from_ymd_opt(y, m, d).expect("Invalid date")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ratewalk_simple() {
        let age_cutpoints: Vec<f64> = (0..=100).map(|i| i as f64 * 365.25).collect();
        let year_cutpoints: Vec<f64> = (1960..=2000)
            .map(|y| {
                let date = NaiveDate::from_ymd_opt(y, 1, 1).unwrap();
                (date - NaiveDate::from_ymd_opt(1960, 1, 1).unwrap()).num_days() as f64
            })
            .collect();

        let dim_sizes = vec![
            age_cutpoints.len() + 1,
            2,
            year_cutpoints.len() + 1,
        ];

        let dimensions = vec![
            Dimension {
                kind: DimensionKind::Continuous,
                cutpoints: age_cutpoints,
            },
            Dimension {
                kind: DimensionKind::Discrete,
                cutpoints: vec![],
            },
            Dimension {
                kind: DimensionKind::Continuous,
                cutpoints: year_cutpoints,
            },
        ];

        let mut rates = vec![0.0; dim_sizes.iter().product()];
        
        let age_index = 20; 
        let sex_index = 0;  
        let year_index = 0; 
        let idx = compute_linear_index(&[age_index, sex_index, year_index], &dim_sizes).unwrap();
        rates[idx] = - (1.0 - 0.00169).ln() / 365.25; 

        let ratetable = Ratetable {
            dim_sizes,
            dimensions,
            rates,
        };

        let birth_date = mdy_date(1, 1, 36);
        let entry_date = mdy_date(9, 7, 60);
        let age_days = (entry_date - birth_date).num_days() as f64;
        let year_days = (entry_date - NaiveDate::from_ymd_opt(1960, 1, 1).unwrap()).num_days() as f64;
        let start = vec![age_days, 1.0, year_days];
        
        let result = ratewalk(&start, 200.0, &ratetable).unwrap();

        assert_eq!(result.days.len(), 2);
        assert!((result.days[0] - 116.0).abs() < 1e-6);
        assert!((result.days[1] - 84.0).abs() < 1e-6);

        let expected_hazard = -(116.0 / 365.25) * (1.0 - 0.00169).ln();
        assert!((result.hazards[0] - expected_hazard).abs() < 1e-6);
    }
}
