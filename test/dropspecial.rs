use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
struct Terms {
    factors_rows: Vec<String>,
    factors_cols: Vec<String>,
    term_labels: Vec<String>,
    variables: Vec<String>,
    data_classes: HashMap<String, String>,
    predvars: Vec<String>,
    specials: HashMap<String, Vec<usize>>,
    offset: Option<usize>,
}

impl Terms {
    fn drop_special(&self, indices: &[usize]) -> Self {
        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_unstable_by(|a, b| b.cmp(a)); 

        let mut new_cols = self.factors_cols.clone();
        let mut new_labels = self.term_labels.clone();

        for &idx in &sorted_indices {
            new_cols.remove(idx);
            new_labels.remove(idx);
        }

        let mut new_specials = self.specials.clone();
        for spec_indices in new_specials.values_mut() {
            spec_indices.retain(|i| !indices.contains(i));
            for i in spec_indices.iter_mut() {
                let num_removed = indices.iter().filter(|&&x| x < *i).count();
                *i -= num_removed;
            }
        }

        Terms {
            factors_rows: self.factors_rows.clone(),
            factors_cols: new_cols,
            term_labels: new_labels,
            variables: self.variables.clone(),
            data_classes: self.data_classes.clone(),
            predvars: self.predvars.clone(),
            specials: new_specials,
            offset: self.offset,
        }
    }

    fn ccheck(&self) -> [bool; 5] {
        let vname = self.variables == self.factors_rows;

        let labels = self.term_labels == self.factors_cols;

        let data = {
            let mut keys: Vec<_> = self.data_classes.keys().collect();
            let mut rows: Vec<_> = self.factors_rows.iter().collect();
            keys.sort();
            rows.sort();
            keys == rows
        };

        let predvars = {
            let p: Vec<_> = self
                .predvars
                .iter()
                .map(|s| s.chars().take(8).collect::<String>())
                .collect();
            let v: Vec<_> = self
                .variables
                .iter()
                .map(|s| s.chars().take(8).collect::<String>())
                .collect();
            p == v
        };

        let strata = self.term_labels.iter().any(|tl| tl.contains("strata"));

        [vname, labels, data, predvars, strata]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maplit::hashmap;

    fn create_test_terms(
        factors_rows: &[&str],
        factors_cols: &[&str],
        term_labels: &[&str],
        variables: &[&str],
        data_classes: HashMap<String, String>,
        predvars: &[&str],
        specials: HashMap<String, Vec<usize>>,
    ) -> Terms {
        Terms {
            factors_rows: factors_rows.iter().map(|s| s.to_string()).collect(),
            factors_cols: factors_cols.iter().map(|s| s.to_string()).collect(),
            term_labels: term_labels.iter().map(|s| s.to_string()).collect(),
            variables: variables.iter().map(|s| s.to_string()).collect(),
            data_classes,
            predvars: predvars.iter().map(|s| s.to_string()).collect(),
            specials,
            offset: None,
        }
    }

    const ANS1: [bool; 5] = [true, true, true, true, false];
    const ANS2: [bool; 5] = [true, true, true, true, true];

    #[test]
    fn test0() {
        let terms = create_test_terms(
            &["age", "wt.loss", "sex"],
            &["age", "ns(wt.loss)", "strata(sex)"],
            &["age", "ns(wt.loss)", "strata(sex)"],
            &["age", "wt.loss", "sex"],
            hashmap! {
                "age".into() => "numeric".into(),
                "wt.loss".into() => "numeric".into(),
                "sex".into() => "factor".into(),
            },
            &["age", "ns(wt.loss)", "sex"],
            hashmap! {"strata".into() => vec![2]},
        );

        let modified = terms.drop_special(&[2]);
        assert_eq!(modified.ccheck(), ANS1);
    }

    #[test]
    fn test3() {
        let terms = create_test_terms(
            &["age", "sex", "ph.ecog", "wt.loss"],
            &["age", "strata(sex)", "ph.ecog:strata(sex)", "ns(wt.loss)"],
            &["age", "strata(sex)", "ph.ecog:strata(sex)", "ns(wt.loss)"],
            &["age", "sex", "ph.ecog", "wt.loss"],
            hashmap! {
                "age".into() => "numeric".into(),
                "sex".into() => "factor".into(),
                "ph.ecog".into() => "numeric".into(),
                "wt.loss".into() => "numeric".into(),
            },
            &["age", "sex", "ph.ecog", "ns(wt.loss)"],
            hashmap! {"strata".into() => vec![1]},
        );

        let modified = terms.drop_special(&[1]);
        assert_eq!(modified.ccheck(), ANS2);
    }
}
