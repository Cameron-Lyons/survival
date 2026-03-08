use survival::dataset::ovarian;
use survival::prelude::*;

fn main() {
    let data = ovarian::load();

    let km = KaplanMeier::new()
        .time(&data.futime)
        .event(&data.fustat)
        .fit();
    println!("Kaplan-Meier summary:\n{:?}", km.summary());

    let models = vec![
        CoxModel::new()
            .time(&data.futime)
            .event(&data.fustat)
            .add_feature("age", &data.age)
            .fit(),
        CoxModel::new()
            .time(&data.futime)
            .event(&data.fustat)
            .add_feature("resid.ds", &data.resid_ds)
            .fit(),
    ];

    for (i, model) in models.iter().enumerate() {
        println!("Model {} coefficients: {:?}", i + 1, model.coefficients());
    }

    let full_model = CoxModel::new()
        .time(&data.futime)
        .event(&data.fustat)
        .add_feature("age", &data.age)
        .add_feature("resid.ds", &data.resid_ds)
        .add_feature("rx", &data.rx)
        .add_feature("ecog.ps", &data.ecog_ps)
        .fit();

    let residuals = full_model.residuals(ResidualType::Deviance);
    println!("Deviance residuals: {:?}", residuals);

    let stratified_model = CoxModel::new()
        .time(&data.futime)
        .event(&data.fustat)
        .add_feature("age", &data.age)
        .add_feature("ecog.ps", &data.ecog_ps)
        .stratify(&data.rx)
        .fit();

    let new_data = vec![
        FeatureSet::new().add("age", 30.0).add("ecog.ps", 2.0),
        FeatureSet::new().add("age", 70.0).add("ecog.ps", 3.0),
    ];

    let survival_curves = stratified_model.predict_survival(&new_data);
    println!("Predicted survival curves:\n{:?}", survival_curves);

    let model1 = CoxModel::new()
        .time(&data.futime)
        .event(&data.fustat)
        .add_feature("age", &data.age)
        .add_feature("rx", &data.rx)
        .fit();

    let model2 = CoxModel::new()
        .time(&data.futime)
        .event(&data.fustat)
        .add_feature("age", &data.age)
        .add_offset("rx", &data.rx, model1.coefficient("rx"))
        .fit();

    assert!((model1.coefficient("age") - model2.coefficient("age")).abs() < 1e-8);
}
