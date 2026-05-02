#[path = "aeq_surv.rs"]
pub(crate) mod aeq_surv_module;
#[path = "cluster.rs"]
pub(crate) mod cluster_module;
#[path = "collapse.rs"]
pub(crate) mod collapse_module;
#[path = "neardate.rs"]
pub(crate) mod neardate_module;
#[path = "rttright.rs"]
pub(crate) mod rttright_module;
#[path = "strata.rs"]
pub(crate) mod strata_module;
#[path = "surv2data.rs"]
pub(crate) mod surv2data_module;
#[path = "survcondense.rs"]
pub(crate) mod survcondense_module;
#[path = "survsplit.rs"]
pub(crate) mod survsplit_module;
#[path = "tcut.rs"]
pub(crate) mod tcut_module;
pub(crate) mod timeline;
#[path = "tmerge.rs"]
pub(crate) mod tmerge_module;

// Public facade exports
pub use aeq_surv_module::{AeqSurvResult, aeq_surv};
pub use cluster_module::{ClusterResult, cluster, cluster_str};
pub use collapse_module::collapse;
pub use neardate_module::{NearDateResult, neardate, neardate_str};
pub use rttright_module::{RttrightResult, rttright, rttright_stratified};
pub use strata_module::{StrataResult, strata, strata_str};
pub use surv2data_module::{Surv2DataResult, surv2data};
pub use survcondense_module::{CondenseResult, survcondense};
pub use survsplit_module::{SplitResult, survsplit};
pub use tcut_module::{TcutResult, tcut, tcut_expand};
pub use timeline::{IntervalResult, TimelineResult, from_timeline, to_timeline};
pub use tmerge_module::{tmerge, tmerge2, tmerge3};
