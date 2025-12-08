use polars::prelude::*;
use survival::coxph::CoxPHModel;
use survival::SurvivalData;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut tdata = CsvReader::from_path("lung.csv")?
        .finish()?
        .lazy()
        .filter(col("ph.ecog").lt(3))
        .collect()?;
    
    tdata.apply("status", |s| {
        s.categorical()
            .map(|cat| cat.set_categories(&["censor", "death"]).unwrap())
    })?;
    tdata.apply("ph.ecog", |s| {
        s.categorical()
            .map(|cat| cat.set_categories(&["ph0", "ph1", "ph2"]).unwrap())
    })?;

    let formula = "Surv(time, status) ~ age + sex + ph.ecog";
    let mut model = CoxPHModel::new();
    model.fit(&tdata, formula)?;
    
    Ok(())
}
