use polars::prelude::*;
use survival::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pbcseq = LazyFrame::scan_parquet("pbcseq.parquet", Default::default())?;
    
    let pbc1 = pbcseq
        .with_column(
            when()
                .then(lit("normal"))
                .when(col("bili").lt_eq(1))
                .then(lit("1-2x"))
                .when(col("bili").lt_eq(2))
                .then(lit("2-4x"))
                .when(col("bili").lt_eq(4))
                .then(lit(">4"))
                .otherwise(lit(">4"))
                .alias("bili4")
        );
    
    let ptemp = pbc1.clone()
        .group_by(["id"])
        .agg([col("*").first()])
        .collect()?;
    
    let pbc2 = survival::tdc::merge(
        ptemp,
        pbc1.collect()?,
        "id",
        vec![
            tdc_event("death", "futime", "status".eq(2)),
            tdc_covar("bili", "day", "bili"),
            tdc_covar("bili4", "day", "bili4"),
            tdc_state("bstat", "day", col("bili4").cast(DataType::Float64))
        ]
    )?;
    
    let btemp = pbc2.column("bstat")?
        .map(|bstat| if pbc2.column("death")? { 5 } else { bstat });
    let b2 = btemp.zip(pbc2.column("bili4")?)
        .map(|(b, bl)| if b == bl.as_f32() { 0 } else { b });
    
    let bstat = b2.cast(DataType::Categorical(Some(vec![
        "censor", "normal", "1-2x", "2-4x", ">4", "death"
    ]))?;
    
    let check1 = survival::check(
        SurvivalData::new("tstart", "tstop", "bstat"),
        InitialState::new("bili4"),
        &pbc2
    )?;
    
    let fit1 = CoxPH::new()
        .formula("Surv(tstart, tstop, death) ~ age + bili4")
        .ties("breslow")
        .fit(&pbc2)?;
    
    let fit2 = MultiStateCox::new()
        .add_transition(1, 5)
        .add_transition(2, 5)
        .add_transition(3, 5)
        .add_transition(4, 5)
        .covariates("age / common + shared")
        .fit(&pbc2)?;
    
    let aeq = |x: &[f64], y: &[f64]| x.iter().zip(y).all(|(a, b)| (a - b).abs() < 1e-6);
    
    assert!(aeq(fit1.coefficients(), fit2.coefficients()));
    
    Ok(())
}
