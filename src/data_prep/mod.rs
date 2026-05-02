pub(crate) mod aeq_surv;
pub(crate) mod cluster;
pub(crate) mod collapse;
pub(crate) mod neardate;
pub(crate) mod rttright;
pub(crate) mod strata;
pub(crate) mod surv2data;
pub(crate) mod survcondense;
pub(crate) mod survsplit;
pub(crate) mod tcut;
pub(crate) mod timeline;
pub(crate) mod tmerge;

// Public facade exports
pub use aeq_surv::{AeqSurvResult, aeq_surv};
pub use cluster::{ClusterResult, cluster, cluster_str};
pub use collapse::collapse;
pub use neardate::{NearDateResult, neardate, neardate_str};
pub use rttright::{RttrightResult, rttright, rttright_stratified};
pub use strata::{StrataResult, strata, strata_str};
pub use surv2data::{Surv2DataResult, surv2data};
pub use survcondense::{CondenseResult, survcondense};
pub use survsplit::{SplitResult, survsplit};
pub use tcut::{TcutResult, tcut, tcut_expand};
pub use timeline::{IntervalResult, TimelineResult, from_timeline, to_timeline};
pub use tmerge::{tmerge, tmerge2, tmerge3};
