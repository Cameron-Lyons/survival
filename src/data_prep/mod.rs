//! Data-preparation helpers of R's `survival` package: `aeqSurv`,
//! `cluster`, `strata`, `lvcf`, `neardate`, `nostutter`, `rttright`,
//! `surv2counting`/`totimeline` (timeline data), `survcondense`,
//! `survSplit`, `tcut` and `tmerge`.
//!
//! Every row index that enters or leaves these routines is zero based
//! (the first observation is row 0), including the rows R reports as
//! one-based subscripts (`survsplit`'s `row`, `neardate`'s match,
//! `tmerge`'s value sources); a caller that needs R's numbering adds one.
//! Subject identifiers are generic over [`SubjectId`], so integer and
//! string ids share one implementation; the Python bindings accept
//! [`IdValue`].

pub(crate) mod aeq_surv;
pub(crate) mod cluster;
pub(crate) mod id_value;
pub(crate) mod lvcf;
pub(crate) mod neardate;
pub(crate) mod nostutter;
pub(crate) mod rttright;
pub(crate) mod strata;
pub(crate) mod surv2counting;
pub(crate) mod survcondense;
pub(crate) mod survsplit;
pub(crate) mod tcut;
pub(crate) mod tmerge;
pub(crate) mod totimeline;

pub use aeq_surv::{AeqSurvResult, DEFAULT_TOLERANCE, aeq_surv, aeq_surv_py, aeq_times};
pub use cluster::{ClusterResult, cluster, cluster_py};
pub use id_value::{IdKey, IdValue, SubjectId, first_appearance_codes};
pub use lvcf::{lvcf, lvcf_py};
pub use neardate::{NeardateBest, neardate, neardate_py};
pub use nostutter::{nostutter, nostutter_py};
pub use rttright::{RttrightInput, RttrightResult, rttright, rttright_py};
pub use strata::{StrataResult, StrataVariable, strata, strata_py};
pub use surv2counting::{Repeated, Surv2CountingResult, surv2counting, surv2counting_py};
pub use survcondense::{SurvcondenseResult, survcondense, survcondense_py};
pub use survsplit::{
    SurvSplitResponse, SurvSplitResult, survsplit, survsplit_intervals, survsplit_py,
};
pub use tcut::{TcutResult, format_numbers, tcut, tcut_py};
pub use tmerge::{
    TCOUNT_NAMES, TmergeBase, TmergeKind, TmergeOptions, TmergeStep, TmergeUpdate,
    tmerge_carry_forward, tmerge_cumulative, tmerge_lookup, tmerge_step, tmerge_step_py,
};
pub use totimeline::{TotimelineResult, totimeline, totimeline_py};
