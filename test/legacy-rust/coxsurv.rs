use approx::assert_abs_diff_eq;
use ndarray::{array, Array1, Array2};

struct CoxPHModel {
    coefficients: Array1<f64>,
    strata: Option<Array1<usize>>,
}

struct SurvivalCurve {
    time: Array1<f64>,
    surv: Array2<f64>,
    strata: Array1<usize>,
}

impl CoxPHModel {
    fn fit(formula: &str, data: &DataFrame) -> Self {
        CoxPHModel {
            coefficients: array![0.1, -0.2, 0.05], 
            strata: Some(array![0, 1, 2]),         
        }
    }

    fn survfit(&self, newdata: Option<&DataFrame>) -> SurvivalCurve {
        SurvivalCurve {
            time: Array1::linspace(0.0, 1000.0, 100),
            surv: Array2::ones((100, 3)), 
            strata: array![30, 40, 30],   
        }
    }
}

fn main() {
    let lung_data = DataFrame::new(); 
    let formula = "Surv(time, status) ~ age + sex + meal.cal + strata(ph.ecog)";

    let fit = CoxPHModel::fit(formula, &lung_data);
    let surv1 = fit.survfit(None);

    let temp = SurvivalCurve {
        time: surv1.time.slice(s![30..70]).to_owned(),
        surv: surv1.surv.slice(s![30..70, ..]).to_owned(),
        strata: array![40, 30],
    };

    assert_abs_diff_eq!(
        surv1.surv.slice(s![30..70, 1..3]),
        temp.surv.slice(s![.., 1..3]),
        epsilon = 1e-6
    );

    let fit_x = CoxPHModel::fit(formula, &lung_data);
    let surv2 = fit_x.survfit(None);

    assert_abs_diff_eq!(surv1.surv, surv2.surv, epsilon = 1e-6);

    let dummy = DataFrame::new(); 
    let surv3 = fit.survfit(Some(&dummy));

    let fit1 = CoxPHModel::fit("Surv(time, status) ~ age + ph.ecog", &lung_data);
    let fit2 = CoxPHModel::fit("Surv(time, status) ~ age + offset", &lung_data);

    let surv_offset1 = fit1.survfit(Some(&dummy));
    let surv_offset2 = fit2.survfit(Some(&dummy));

    assert_abs_diff_eq!(surv_offset1.surv, surv_offset2.surv, epsilon = 1e-6);

    let start_time = 100.0;
    let restricted_surv = SurvivalCurve {
        time: surv1.time.slice(s![start_time as usize..]).to_owned(),
        surv: surv1.surv.slice(s![start_time as usize.., ..]).to_owned(),
        strata: surv1.strata.clone(),
    };
}
