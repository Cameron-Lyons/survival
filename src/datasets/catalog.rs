//! The catalog of R `survival` datasets bundled with this crate.
//!
//! Each entry mirrors the data frame exported from R (`data(package =
//! "survival")`, survival 3.8-11): the R name, the CSV written from that data
//! frame, and the R storage mode of every column in R's column order. One
//! `#[pyfunction]` loader per dataset is generated from the same table so the
//! Python surface stays `load_<name>()` while the data lives in one place.

use super::common::{ColType, Dataset};
use pyo3::prelude::*;
use pyo3::types::PyDict;

macro_rules! datasets {
    (
        $(
            $const:ident : $loader:ident = $name:literal [
                $( ($col:literal, $ty:ident) ),* $(,)?
            ];
        )*
    ) => {
        $(
            pub(crate) static $const: Dataset = Dataset {
                name: $name,
                csv: include_str!(concat!("data/", $name, ".csv")),
                schema: &[ $( ($col, ColType::$ty) ),* ],
            };

            #[doc = concat!("Load R's `survival::", $name, "` data frame as a dictionary of column lists.")]
            #[pyfunction]
            pub(crate) fn $loader(py: Python<'_>) -> PyResult<Py<PyDict>> {
                $const.to_pydict(py)
            }
        )*

        /// Every bundled dataset, in the order of R's `data(package = "survival")`.
        #[cfg(test)]
        pub(crate) static DATASETS: &[&Dataset] = &[ $( &$const ),* ];
    };
}

datasets! {
    AML: load_aml = "aml" [
        ("time", Float),
        ("status", Float),
        ("x", Str),
    ];
    BLADDER: load_bladder = "bladder" [
        ("id", Int),
        ("rx", Float),
        ("number", Int),
        ("size", Int),
        ("stop", Int),
        ("event", Float),
        ("enum", Int),
    ];
    BLADDER1: load_bladder1 = "bladder1" [
        ("id", Int),
        ("treatment", Str),
        ("number", Int),
        ("size", Int),
        ("recur", Int),
        ("start", Int),
        ("stop", Int),
        ("status", Float),
        ("rtumor", Str),
        ("rsize", Str),
        ("enum", Float),
    ];
    BLADDER2: load_bladder2 = "bladder2" [
        ("id", Int),
        ("rx", Float),
        ("number", Int),
        ("size", Int),
        ("start", Int),
        ("stop", Int),
        ("event", Float),
        ("enum", Float),
    ];
    BRAKING: load_braking = "braking" [
        ("locomotive", Int),
        ("batch", Int),
        ("day1", Float),
        ("day2", Int),
        ("status", Int),
    ];
    CAPACITOR: load_capacitor = "capacitor" [
        ("temperature", Float),
        ("voltage", Float),
        ("fail", Bool),
        ("time", Float),
        ("status", Int),
    ];
    CGD: load_cgd = "cgd" [
        ("id", Int),
        ("center", Str),
        ("random", Str),
        ("treat", Str),
        ("sex", Str),
        ("age", Int),
        ("height", Float),
        ("weight", Float),
        ("inherit", Str),
        ("steroids", Float),
        ("propylac", Float),
        ("hos.cat", Str),
        ("tstart", Int),
        ("enum", Int),
        ("tstop", Int),
        ("status", Int),
    ];
    CGD0: load_cgd0 = "cgd0" [
        ("id", Int),
        ("center", Int),
        ("random", Int),
        ("treat", Int),
        ("sex", Int),
        ("age", Int),
        ("height", Float),
        ("weight", Float),
        ("inherit", Int),
        ("steroids", Int),
        ("propylac", Int),
        ("hos.cat", Int),
        ("futime", Int),
        ("etime1", Int),
        ("etime2", Int),
        ("etime3", Int),
        ("etime4", Int),
        ("etime5", Int),
        ("etime6", Int),
        ("etime7", Int),
    ];
    COLON: load_colon = "colon" [
        ("id", Float),
        ("study", Float),
        ("rx", Str),
        ("sex", Float),
        ("age", Float),
        ("obstruct", Float),
        ("perfor", Float),
        ("adhere", Float),
        ("nodes", Float),
        ("status", Float),
        ("differ", Float),
        ("extent", Float),
        ("surg", Float),
        ("node4", Float),
        ("time", Float),
        ("etype", Float),
    ];
    CRACKS: load_cracks = "cracks" [
        ("days", Float),
        ("fail", Int),
    ];
    DIABETIC: load_diabetic = "diabetic" [
        ("id", Int),
        ("laser", Str),
        ("age", Int),
        ("eye", Str),
        ("trt", Int),
        ("risk", Int),
        ("time", Float),
        ("status", Int),
    ];
    FLCHAIN: load_flchain = "flchain" [
        ("age", Float),
        ("sex", Str),
        ("sample.yr", Float),
        ("kappa", Float),
        ("lambda", Float),
        ("flc.grp", Float),
        ("creatinine", Float),
        ("mgus", Float),
        ("futime", Int),
        ("death", Float),
        ("chapter", Str),
    ];
    GBSG: load_gbsg = "gbsg" [
        ("pid", Int),
        ("age", Int),
        ("meno", Int),
        ("size", Int),
        ("grade", Int),
        ("nodes", Int),
        ("pgr", Int),
        ("er", Int),
        ("hormon", Int),
        ("rfstime", Int),
        ("status", Int),
    ];
    GENFAN: load_genfan = "genfan" [
        ("hours", Float),
        ("status", Int),
    ];
    HEART: load_heart = "heart" [
        ("start", Float),
        ("stop", Float),
        ("event", Float),
        ("age", Float),
        ("year", Float),
        ("surgery", Float),
        ("transplant", Str),
        ("id", Float),
    ];
    HOEL: load_hoel = "hoel" [
        ("trt", Str),
        ("days", Float),
        ("outcome", Str),
        ("id", Int),
    ];
    IFLUID: load_ifluid = "ifluid" [
        ("time", Float),
        ("voltage", Float),
    ];
    IMOTOR: load_imotor = "imotor" [
        ("temp", Int),
        ("time", Int),
        ("status", Int),
    ];
    JASA: load_jasa = "jasa" [
        ("birth.dt", Str),
        ("accept.dt", Str),
        ("tx.date", Str),
        ("fu.date", Str),
        ("fustat", Float),
        ("surgery", Float),
        ("age", Float),
        ("futime", Float),
        ("wait.time", Float),
        ("transplant", Float),
        ("mismatch", Float),
        ("hla.a2", Float),
        ("mscore", Float),
        ("reject", Float),
    ];
    JASA1: load_jasa1 = "jasa1" [
        ("id", Float),
        ("start", Float),
        ("stop", Float),
        ("event", Float),
        ("transplant", Float),
        ("age", Float),
        ("year", Float),
        ("surgery", Float),
    ];
    KIDNEY: load_kidney = "kidney" [
        ("id", Float),
        ("time", Float),
        ("status", Float),
        ("age", Float),
        ("sex", Float),
        ("disease", Str),
        ("frail", Float),
    ];
    LOGAN: load_logan = "logan" [
        ("occupation", Str),
        ("focc", Str),
        ("education", Int),
        ("race", Str),
    ];
    LUNG: load_lung = "lung" [
        ("inst", Float),
        ("time", Float),
        ("status", Float),
        ("age", Float),
        ("sex", Float),
        ("ph.ecog", Float),
        ("ph.karno", Float),
        ("pat.karno", Float),
        ("meal.cal", Float),
        ("wt.loss", Float),
    ];
    MGUS: load_mgus = "mgus" [
        ("id", Float),
        ("age", Float),
        ("sex", Str),
        ("dxyr", Float),
        ("pcdx", Str),
        ("pctime", Float),
        ("futime", Float),
        ("death", Float),
        ("alb", Float),
        ("creat", Float),
        ("hgb", Float),
        ("mspike", Float),
    ];
    MGUS1: load_mgus1 = "mgus1" [
        ("id", Float),
        ("age", Float),
        ("sex", Str),
        ("dxyr", Float),
        ("pcdx", Str),
        ("alb", Float),
        ("creat", Float),
        ("hgb", Float),
        ("mspike", Float),
        ("stop", Float),
        ("status", Float),
        ("event", Str),
        ("start", Float),
        ("enum", Float),
    ];
    MGUS2: load_mgus2 = "mgus2" [
        ("id", Float),
        ("age", Float),
        ("sex", Str),
        ("dxyr", Float),
        ("hgb", Float),
        ("creat", Float),
        ("mspike", Float),
        ("ptime", Float),
        ("pstat", Float),
        ("futime", Float),
        ("death", Float),
    ];
    MYELOID: load_myeloid = "myeloid" [
        ("id", Int),
        ("trt", Str),
        ("sex", Str),
        ("flt3", Str),
        ("futime", Float),
        ("death", Float),
        ("txtime", Float),
        ("crtime", Float),
        ("rltime", Float),
    ];
    MYELOMA: load_myeloma = "myeloma" [
        ("id", Int),
        ("year", Int),
        ("entry", Int),
        ("futime", Int),
        ("death", Int),
    ];
    NAFLD1: load_nafld1 = "nafld1" [
        ("id", Int),
        ("age", Int),
        ("male", Int),
        ("weight", Float),
        ("height", Int),
        ("bmi", Float),
        ("case.id", Int),
        ("futime", Int),
        ("status", Int),
    ];
    NAFLD2: load_nafld2 = "nafld2" [
        ("id", Int),
        ("days", Int),
        ("test", Str),
        ("value", Float),
    ];
    NAFLD3: load_nafld3 = "nafld3" [
        ("id", Int),
        ("days", Int),
        ("event", Str),
    ];
    NWTCO: load_nwtco = "nwtco" [
        ("seqno", Int),
        ("instit", Int),
        ("histol", Int),
        ("stage", Int),
        ("study", Int),
        ("rel", Int),
        ("edrel", Int),
        ("age", Int),
        ("in.subcohort", Bool),
    ];
    OVARIAN: load_ovarian = "ovarian" [
        ("futime", Float),
        ("fustat", Float),
        ("age", Float),
        ("resid.ds", Float),
        ("rx", Float),
        ("ecog.ps", Float),
    ];
    PBC: load_pbc = "pbc" [
        ("id", Int),
        ("time", Int),
        ("status", Int),
        ("trt", Int),
        ("age", Float),
        ("sex", Str),
        ("ascites", Int),
        ("hepato", Int),
        ("spiders", Int),
        ("edema", Float),
        ("bili", Float),
        ("chol", Int),
        ("albumin", Float),
        ("copper", Int),
        ("alk.phos", Float),
        ("ast", Float),
        ("trig", Int),
        ("platelet", Int),
        ("protime", Float),
        ("stage", Int),
    ];
    PBCSEQ: load_pbcseq = "pbcseq" [
        ("id", Int),
        ("futime", Int),
        ("status", Int),
        ("trt", Int),
        ("age", Float),
        ("sex", Str),
        ("day", Int),
        ("ascites", Int),
        ("hepato", Int),
        ("spiders", Int),
        ("edema", Float),
        ("bili", Float),
        ("chol", Int),
        ("albumin", Float),
        ("alk.phos", Int),
        ("ast", Float),
        ("platelet", Int),
        ("protime", Float),
        ("stage", Int),
    ];
    RATS: load_rats = "rats" [
        ("litter", Int),
        ("rx", Float),
        ("time", Float),
        ("status", Float),
        ("sex", Str),
    ];
    RATS2: load_rats2 = "rats2" [
        ("id", Int),
        ("trt", Int),
        ("obs", Int),
        ("time1", Int),
        ("time2", Int),
        ("status", Int),
    ];
    RETINOPATHY: load_retinopathy = "retinopathy" [
        ("id", Int),
        ("laser", Str),
        ("eye", Str),
        ("age", Int),
        ("type", Str),
        ("trt", Int),
        ("futime", Float),
        ("status", Int),
        ("risk", Int),
    ];
    RHDNASE: load_rhdnase = "rhDNase" [
        ("id", Int),
        ("inst", Int),
        ("trt", Int),
        ("entry.dt", Str),
        ("end.dt", Str),
        ("fev", Float),
        ("ivstart", Float),
        ("ivstop", Float),
    ];
    ROTTERDAM: load_rotterdam = "rotterdam" [
        ("pid", Int),
        ("year", Int),
        ("age", Int),
        ("meno", Int),
        ("size", Str),
        ("grade", Int),
        ("nodes", Int),
        ("pgr", Int),
        ("er", Int),
        ("hormon", Int),
        ("chemo", Int),
        ("rtime", Float),
        ("recur", Int),
        ("dtime", Float),
        ("death", Int),
    ];
    SOLDER: load_solder = "solder" [
        ("Opening", Str),
        ("Solder", Str),
        ("Mask", Str),
        ("PadType", Str),
        ("Panel", Str),
        ("skips", Float),
    ];
    STANFORD2: load_stanford2 = "stanford2" [
        ("id", Float),
        ("time", Float),
        ("status", Float),
        ("age", Float),
        ("t5", Float),
    ];
    TOBIN: load_tobin = "tobin" [
        ("durable", Float),
        ("age", Float),
        ("quant", Int),
    ];
    TRANSPLANT: load_transplant = "transplant" [
        ("age", Float),
        ("sex", Str),
        ("abo", Str),
        ("year", Float),
        ("futime", Float),
        ("event", Str),
    ];
    TURBINE: load_turbine = "turbine" [
        ("hours", Int),
        ("inspected", Int),
        ("failed", Int),
    ];
    UDCA: load_udca = "udca" [
        ("id", Int),
        ("trt", Int),
        ("entry.dt", Str),
        ("last.dt", Str),
        ("stage", Int),
        ("bili", Float),
        ("riskscore", Float),
        ("death.dt", Str),
        ("tx.dt", Str),
        ("hprogress.dt", Str),
        ("varices.dt", Str),
        ("ascites.dt", Str),
        ("enceph.dt", Str),
        ("double.dt", Str),
        ("worsen.dt", Str),
    ];
    UDCA1: load_udca1 = "udca1" [
        ("id", Int),
        ("trt", Int),
        ("stage", Int),
        ("bili", Float),
        ("riskscore", Float),
        ("futime", Float),
        ("status", Float),
    ];
    UDCA2: load_udca2 = "udca2" [
        ("id", Int),
        ("trt", Int),
        ("stage", Int),
        ("bili", Float),
        ("riskscore", Float),
        ("futime", Float),
        ("status", Float),
        ("endpoint", Str),
    ];
    VALVESEAT: load_valveseat = "valveSeat" [
        ("id", Float),
        ("time", Float),
        ("status", Float),
    ];
    VETERAN: load_veteran = "veteran" [
        ("trt", Float),
        ("celltype", Str),
        ("time", Float),
        ("status", Float),
        ("karno", Float),
        ("diagtime", Float),
        ("age", Float),
        ("prior", Float),
    ];
}

/// R's `cancer` is the same data frame as `lung` (`identical(cancer, lung)`).
#[pyfunction]
pub(crate) fn load_cancer(py: Python<'_>) -> PyResult<Py<PyDict>> {
    LUNG.to_pydict(py)
}

/// R's `leukemia` is the same data frame as `aml` (`identical(leukemia, aml)`).
#[pyfunction]
pub(crate) fn load_leukemia(py: Python<'_>) -> PyResult<Py<PyDict>> {
    AML.to_pydict(py)
}

#[cfg(test)]
mod tests {
    use super::super::common::{Column, DataFrame};
    use super::*;

    /// Per-dataset shape, column kinds, NA counts, checksums and bit-pattern
    /// hashes computed in R.
    const R_REFERENCE: &str = include_str!("r_reference.tsv");

    fn dataset(name: &str) -> &'static Dataset {
        DATASETS
            .iter()
            .copied()
            .find(|d| d.name == name)
            .unwrap_or_else(|| panic!("no dataset named {name}"))
    }

    fn frame(name: &str) -> DataFrame {
        dataset(name).parse().unwrap()
    }

    fn checksum(column: &Column) -> f64 {
        match column {
            Column::Float(values) => values.iter().flatten().sum(),
            Column::Int(values) => values.iter().flatten().map(|&v| f64::from(v)).sum(),
            Column::Bool(values) => values.iter().flatten().filter(|&&v| v).count() as f64,
            Column::Str(values) => values
                .iter()
                .flatten()
                .map(|s| s.chars().count() as f64)
                .sum(),
        }
    }

    /// Order-dependent hash of the IEEE-754 bit patterns of the non-NA values
    /// (`(h * 31 + low32) mod 2^32`, then the same with the high word), as
    /// computed by the R script that wrote `r_reference.tsv`. Unlike the
    /// tolerant checksum it catches a value that is off by a single ulp.
    fn bits_hash(values: &[Option<f64>]) -> u64 {
        values.iter().flatten().fold(0, |h, v| {
            let bits = v.to_bits();
            let h = (h * 31 + (bits & 0xffff_ffff)) % (1 << 32);
            (h * 31 + (bits >> 32)) % (1 << 32)
        })
    }

    fn kind(column: &Column) -> &'static str {
        match column {
            Column::Float(_) => "float",
            Column::Int(_) => "int",
            Column::Bool(_) => "logical",
            Column::Str(_) => "string",
        }
    }

    fn float(column: &Column, row: usize) -> f64 {
        match column {
            Column::Float(values) => values[row].unwrap(),
            other => panic!("not a float column: {other:?}"),
        }
    }

    fn int(column: &Column, row: usize) -> i32 {
        match column {
            Column::Int(values) => values[row].unwrap(),
            other => panic!("not an int column: {other:?}"),
        }
    }

    fn is_na(column: &Column, row: usize) -> bool {
        match column {
            Column::Float(values) => values[row].is_none(),
            Column::Int(values) => values[row].is_none(),
            Column::Str(values) => values[row].is_none(),
            Column::Bool(values) => values[row].is_none(),
        }
    }

    fn string(column: &Column, row: usize) -> &str {
        match column {
            Column::Str(values) => values[row].as_deref().unwrap(),
            other => panic!("not a string column: {other:?}"),
        }
    }

    #[test]
    fn catalog_names_are_unique_and_match_files() {
        let mut names: Vec<&str> = DATASETS.iter().map(|d| d.name).collect();
        assert_eq!(names.len(), 50);
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 50);
        for d in DATASETS {
            assert!(!d.schema.is_empty(), "{} has an empty schema", d.name);
        }
    }

    #[test]
    fn bits_hash_matches_the_r_definition() {
        // aml$time, hashed in R with the same recurrence.
        let time = frame("aml").column("time").unwrap().clone();
        let Column::Float(values) = time else {
            unreachable!()
        };
        assert_eq!(bits_hash(&values), 327_819_264);
        assert_eq!(bits_hash(&[]), 0);
        // 1.0 is 0x3ff0_0000_0000_0000: a zero low word, then the high word.
        assert_eq!(bits_hash(&[Some(1.0)]), 0x3ff0_0000);
        assert_ne!(
            bits_hash(&[Some(1.0)]),
            bits_hash(&[Some(1.0 + f64::EPSILON)])
        );
    }

    #[test]
    fn every_dataset_matches_r_shape_types_na_counts_and_checksums() {
        let mut checked = 0;
        let mut current: Option<(&str, DataFrame, usize)> = None;
        for line in R_REFERENCE.lines().filter(|l| !l.starts_with('#')) {
            let fields: Vec<&str> = line.split('\t').collect();
            if !fields[0].is_empty() {
                let name = fields[0];
                let df = frame(name);
                assert_eq!(df.nrow, fields[1].parse::<usize>().unwrap(), "{name} nrow");
                assert_eq!(
                    df.ncol(),
                    fields[2].parse::<usize>().unwrap(),
                    "{name} ncol"
                );
                current = Some((name, df, 0));
                checked += 1;
                continue;
            }
            let (name, df, col_index) = current.as_mut().unwrap();
            let (col_name, column) = &df.columns[*col_index];
            *col_index += 1;
            let context = format!("{name}${col_name}");
            assert_eq!(*col_name, fields[1], "{context}: column order");
            assert_eq!(kind(column), fields[2], "{context}: column kind");
            assert_eq!(column.len(), df.nrow, "{context}: length");
            assert_eq!(
                column.na_count(),
                fields[3].parse::<usize>().unwrap(),
                "{context}: NA count"
            );
            let expected: f64 = fields[4].parse().unwrap();
            let actual = checksum(column);
            let tolerance = 1e-9 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{context}: checksum {actual} != R {expected}"
            );
            if let Column::Float(values) = column {
                let expected: u64 = fields[5].parse().unwrap();
                assert_eq!(
                    bits_hash(values),
                    expected,
                    "{context}: values are not bit-exact copies of R's"
                );
            } else {
                assert!(fields[5].is_empty(), "{context}: unexpected bits hash");
            }
        }
        if let Some((name, df, col_index)) = current {
            assert_eq!(col_index, df.ncol(), "{name}: unchecked columns");
        }
        assert_eq!(checked, DATASETS.len(), "every dataset has an R reference");
    }

    #[test]
    fn aml_has_r_column_names_and_values() {
        let df = frame("aml");
        assert_eq!((df.nrow, df.ncol()), (23, 3));
        let time = df.column("time").unwrap();
        assert_eq!((float(time, 0), float(time, 22)), (9.0, 45.0));
        assert_eq!(float(df.column("status").unwrap(), 0), 1.0);
        let x = df.column("x").unwrap();
        assert_eq!(
            (string(x, 0), string(x, 22)),
            ("Maintained", "Nonmaintained")
        );
    }

    #[test]
    fn lung_first_row_and_missing_values() {
        let df = frame("lung");
        assert_eq!(float(df.column("inst").unwrap(), 0), 3.0);
        assert_eq!(float(df.column("time").unwrap(), 0), 306.0);
        assert_eq!(float(df.column("meal.cal").unwrap(), 0), 1175.0);
        assert!(is_na(df.column("wt.loss").unwrap(), 0));
        assert_eq!(df.column("meal.cal").unwrap().na_count(), 47);
        assert_eq!(df.column("wt.loss").unwrap().na_count(), 14);
    }

    #[test]
    fn rats_is_the_300_litter_table() {
        let df = frame("rats");
        assert_eq!((df.nrow, df.ncol()), (300, 5));
        assert_eq!(int(df.column("litter").unwrap(), 299), 100);
        assert_eq!(float(df.column("rx").unwrap(), 299), 0.0);
        assert_eq!(float(df.column("time").unwrap(), 299), 102.0);
        assert_eq!(float(df.column("time").unwrap(), 0), 101.0);
        assert_eq!(string(df.column("sex").unwrap(), 299), "m");
        assert_eq!(string(df.column("sex").unwrap(), 0), "f");
    }

    #[test]
    fn hoel_myeloma_rhdnase_have_r_content() {
        let hoel = frame("hoel");
        assert_eq!((hoel.nrow, hoel.ncol()), (181, 4));
        assert_eq!(string(hoel.column("trt").unwrap(), 180), "Germ-free");
        assert_eq!(float(hoel.column("days").unwrap(), 180), 1019.0);
        assert_eq!(
            string(hoel.column("outcome").unwrap(), 0),
            "thymic lymphoma"
        );
        assert_eq!(int(hoel.column("id").unwrap(), 180), 181);

        let myeloma = frame("myeloma");
        assert_eq!((myeloma.nrow, myeloma.ncol()), (3882, 5));
        assert_eq!(int(myeloma.column("id").unwrap(), 3881), 3914);
        assert_eq!(int(myeloma.column("futime").unwrap(), 0), 1431);
        assert_eq!(int(myeloma.column("futime").unwrap(), 3881), 498);

        let rhdnase = frame("rhDNase");
        assert_eq!((rhdnase.nrow, rhdnase.ncol()), (767, 8));
        assert_eq!(string(rhdnase.column("entry.dt").unwrap(), 0), "1992-03-20");
        assert_eq!(string(rhdnase.column("end.dt").unwrap(), 766), "1992-09-11");
        assert_eq!(float(rhdnase.column("fev").unwrap(), 766), 92.8);
        assert_eq!(rhdnase.column("ivstart").unwrap().na_count(), 400);
        assert_eq!(float(rhdnase.column("ivstart").unwrap(), 2), 65.0);
        assert_eq!(float(rhdnase.column("ivstop").unwrap(), 2), 75.0);
    }

    #[test]
    fn dates_logicals_and_computed_doubles_round_trip() {
        let jasa = frame("jasa");
        assert_eq!(string(jasa.column("birth.dt").unwrap(), 0), "1937-01-10");
        assert_eq!(jasa.column("tx.date").unwrap().na_count(), 34);
        assert!(is_na(jasa.column("tx.date").unwrap(), 0));
        assert_eq!(float(jasa.column("age").unwrap(), 0), 30.844626967830255);

        let heart = frame("heart");
        assert_eq!(float(heart.column("age").unwrap(), 0), -17.15537303216975);
        assert_eq!(string(heart.column("transplant").unwrap(), 0), "0");

        let pbc = frame("pbc");
        assert_eq!(float(pbc.column("age").unwrap(), 0), 58.76522929500342);
        assert_eq!(string(pbc.column("sex").unwrap(), 0), "f");

        let genfan = frame("genfan");
        let Column::Float(hours) = genfan.column("hours").unwrap() else {
            unreachable!()
        };
        assert!(hours.contains(&Some(459.99999999999994)));

        let nwtco = frame("nwtco");
        let Column::Bool(sub) = nwtco.column("in.subcohort").unwrap() else {
            panic!("in.subcohort must be logical")
        };
        assert_eq!(sub[0], Some(false));
        assert_eq!(sub.iter().flatten().filter(|&&v| v).count(), 668);

        let capacitor = frame("capacitor");
        assert_eq!(capacitor.column("fail").unwrap().na_count(), 64);
        assert_eq!(float(capacitor.column("time").unwrap(), 63), 455.0);
    }

    #[test]
    fn large_and_new_tables_have_r_edges() {
        let nafld2 = frame("nafld2");
        assert_eq!((nafld2.nrow, nafld2.ncol()), (400123, 4));
        assert_eq!(int(nafld2.column("days").unwrap(), 0), -459);
        assert_eq!(string(nafld2.column("test").unwrap(), 0), "hdl");
        assert_eq!(int(nafld2.column("id").unwrap(), 400122), 17566);
        assert_eq!(string(nafld2.column("test").unwrap(), 400122), "chol");
        assert_eq!(float(nafld2.column("value").unwrap(), 400122), 47.0);

        let nafld3 = frame("nafld3");
        assert_eq!((nafld3.nrow, nafld3.ncol()), (34345, 3));
        assert_eq!(nafld3.column("days").unwrap().na_count(), 18);

        let bladder1 = frame("bladder1");
        assert_eq!(string(bladder1.column("rtumor").unwrap(), 293), ".");
        assert_eq!(int(bladder1.column("id").unwrap(), 293), 118);

        let udca1 = frame("udca1");
        assert_eq!(float(udca1.column("futime").unwrap(), 169), 791.0);
        let udca2 = frame("udca2");
        assert_eq!(
            string(udca2.column("endpoint").unwrap(), 1359),
            "doubling of bilirubin"
        );

        let mgus1 = frame("mgus1");
        assert_eq!(string(mgus1.column("event").unwrap(), 304), "death");
        assert_eq!(float(mgus1.column("start").unwrap(), 304), 3766.0);

        let valve = frame("valveSeat");
        assert_eq!(float(valve.column("time").unwrap(), 88), 582.0);
        let cracks = frame("cracks");
        assert_eq!(int(cracks.column("fail").unwrap(), 7), 17);
        let turbine = frame("turbine");
        assert_eq!(int(turbine.column("failed").unwrap(), 10), 21);
    }

    #[test]
    fn factors_are_labels_and_dates_are_iso_strings() {
        let cgd = frame("cgd");
        assert_eq!(
            string(cgd.column("center").unwrap(), 0),
            "Scripps Institute"
        );
        assert_eq!(string(cgd.column("random").unwrap(), 0), "1989-06-07");
        assert_eq!(string(cgd.column("treat").unwrap(), 0), "rIFN-g");
        let solder = frame("solder");
        assert_eq!(string(solder.column("Panel").unwrap(), 899), "3");
        let transplant = frame("transplant");
        assert_eq!(string(transplant.column("event").unwrap(), 814), "censored");
        assert!(is_na(transplant.column("age").unwrap(), 71));
    }
}
