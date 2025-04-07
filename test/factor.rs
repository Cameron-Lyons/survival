use anyhow::Result;
use csv::Reader;
use linfa::Dataset;
use linfa_survival::CoxPhParams;
use ndarray::{Array1, Array2, Axis};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct LungRecord {
    #[serde(rename = "time")]
    time: f64,
    #[serde(rename = "status")]
    status: u32,
    #[serde(rename = "age")]
    age: f64,
    #[serde(rename = "ph.ecog")]
    ph_ecog: Option<i32>,
}

fn main() -> Result<()> {
    let mut rdr = Reader::from_path("lung.csv")?;
    let records: Vec<LungRecord> = rdr.deserialize().filter_map(|r| r.ok()).collect();

    let (features, times, statuses, complete_indices) = create_features(&records);

    let dataset = Dataset::new(features, (times, statuses));

    let model = CoxPhParams::default().fit(&dataset)?;

    let p1_complete = model.predict_risk(&dataset);

    let p1 = create_full_predictions(p1_complete, &complete_indices, records.len());

    let keep = create_keep_mask(&records);
    let p1_subset = get_subset_predictions(&p1, &keep);

    let p2 = predict_subset(&model, &records, &keep);

    assert!(aeq(&p1_subset, &p2, 1e-6));

    Ok(())
}

fn create_features(records: &[LungRecord]) -> (Array2<f64>, Array1<f64>, Array1<u32>, Vec<usize>) {
    let mut features = Vec::new();
    let mut times = Vec::new();
    let mut statuses = Vec::new();
    let mut complete_indices = Vec::new();

    for (i, record) in records.iter().enumerate() {
        if let Some(ph_ecog) = record.ph_ecog {
            let dummies = [
                record.age,
                (ph_ecog == 1) as i32 as f64,
                (ph_ecog == 2) as i32 as f64,
                (ph_ecog == 3) as i32 as f64,
            ];
            features.push(dummies);
            times.push(record.time);
            statuses.push(record.status);
            complete_indices.push(i);
        }
    }

    let features = Array2::from_shape_vec((features.len(), 4), features.concat()).unwrap();
    (features, Array1::from(times), Array1::from(statuses), complete_indices)
}

fn create_full_predictions(
    p1_complete: Array1<f64>,
    complete_indices: &[usize],
    total_records: usize,
) -> Vec<Option<f64>> {
    let mut p1 = vec![None; total_records];
    for (i, &idx) in complete_indices.iter().enumerate() {
        p1[idx] = Some(p1_complete[i]);
    }
    p1
}

fn create_keep_mask(records: &[LungRecord]) -> Vec<bool> {
    records
        .iter()
        .map(|r| r.ph_ecog.map_or(true, |e| e != 1))
        .collect()
}

fn get_subset_predictions(p1: &[Option<f64>], keep: &[bool]) -> Vec<Option<f64>> {
    p1.iter()
        .zip(keep)
        .filter(|(_, &k)| k)
        .map(|(p, _)| *p)
        .collect()
}

fn predict_subset(model: &CoxPhParams, records: &[LungRecord], keep: &[bool]) -> Vec<Option<f64>> {
    records
        .iter()
        .zip(keep)
        .filter(|(_, &k)| k)
        .map(|(r, _)| {
            r.ph_ecog
                .map(|e| {
                    let features = Array1::from(vec![
                        r.age,
                        (e == 1) as i32 as f64,
                        (e == 2) as i32 as f64,
                        (e == 3) as i32 as f64,
                    ])
                    .insert_axis(Axis(0));
                    model.predict_risk(&features)[0]
                })
        })
        .collect()
}

fn aeq(x: &[Option<f64>], y: &[Option<f64>], tolerance: f64) -> bool {
    x.len() == y.len()
        && x.iter().zip(y).all(|(a, b)| match (a, b) {
            (Some(a_val), Some(b_val)) => (a_val - b_val).abs() < tolerance,
            (None, None) => true,
            _ => false,
        })
}
