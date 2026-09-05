# Regenerate with R survival 3.8.11 and jsonlite:
# Rscript python/tests/fixtures/concordance_boundary_r3811.R > \
#   python/tests/fixtures/concordance_boundary_r3811.json
library(survival)
stopifnot(as.character(packageVersion("survival")) == "3.8.11")

dataset <- function(time, status, score = seq_along(time), start = rep(0, length(time)),
                    w = rep(1, length(time))) {
    data.frame(start = start, time = time, status = status, score = score, w = w)
}
datasets <- list(
    one_original_event = dataset(c(1,2,3), c(1,0,0), c(2,1,3), w=c(2,1,3)),
    one_event_after_censor = dataset(c(1,2,3), c(0,1,0), c(2,1,3), w=c(2,1,3)),
    two_original_events = dataset(c(1,2,3), c(1,1,0), c(2,1,3), w=c(2,1,3)),
    two_events_after_censor = dataset(c(1,2,3,4), c(0,1,1,0), c(2,2,1,3), w=c(2,1,2,3)),
    zero_weight_simultaneous_event = dataset(c(1,1,2), c(1,1,0), c(2,1,3), w=c(1,0,1)),
    all_censored = dataset(c(1,2,3), c(0,0,0)),
    event_after_censors = dataset(c(1,2,3), c(0,0,1)),
    simultaneous_events = dataset(c(1,1), c(1,1)),
    simultaneous_events_tied_scores = dataset(c(1,1), c(1,1), c(2,2)),
    disjoint_counting_events = dataset(c(1,2), c(1,1), start=c(0,1)),
    all_events = dataset(c(1,2,3), c(1,1,1), c(2,1,3)),
    bound_limits = dataset(c(1,2,3), c(1,1,0), c(3,1,2)),
    all_zero_weights = dataset(c(1,2,3), c(1,1,0), w=c(0,0,0)),
    ymin_floor = dataset(c(1,2,3,4), c(1,0,1,1), c(2,1,3,2), w=c(2,3,1,2)),
    ymin_after_timefix = dataset(c(1,3,4), c(1,1,0), c(2,1,3), start=c(0,2-5e-13,0)),
    aeq_absolute = dataset(c(.01,.01+5e-9,.01+2.5e-8,1), c(1,0,1,1), c(2,1,3,2), w=c(1,2,1,1)),
    aeq_relative = dataset(1e9+c(0,5,20,100), c(1,0,1,1), c(2,1,3,2), start=rep(1e9-100,4)),
    aeq_adjacent_chain = dataset(c(1,1+1e-8,1+2e-8,2), c(1,0,1,1), c(2,1,3,2)),
    negative_times = dataset(c(-3,-2,-1,1), c(1,0,1,1), c(2,1,3,2), start=c(-5,-4,-3,-1)),
    near_zero_interval = dataset(c(1,1+5e-9,3), c(1,1,0), c(2,1,3), start=c(0,1,0))
)
datasets$sparse_strata <- rbind(
    dataset(c(1,2,3,4), c(1,0,0,0), c(2,1,4,3), w=c(1,2,1,.5)),
    dataset(c(1,3,2,4), c(1,1,0,0), c(3,1,2,4), w=c(2,1,.5,2)),
    dataset(c(1,2,3,4), c(0,0,0,0), c(1,4,3,2), w=c(1,2,3,1))
)
datasets$sparse_strata$group <- rep(c("A","B","C"), each=4)
datasets$sparse_strata$cluster <- rep(c("a","b","a","c"),3)

cases <- list()
errors <- list()
add <- function(name, response="right", timewt="n", timefix=FALSE,
                ymin=NULL, ymax=NULL, ranks=TRUE, cluster=FALSE, suffix="") {
    d <- datasets[[name]]
    y <- if (response == "right") Surv(d$time,d$status) else Surv(d$start,d$time,d$status)
    args <- list(y=y, x=d$score, weights=d$w, timewt=timewt, timefix=timefix,
                 ymin=ymin, ymax=ymax, influence=3, ranks=ranks, keepstrata=TRUE)
    if ("group" %in% names(d)) args$strata <- d$group
    if (cluster) args$cluster <- d$cluster
    # Formula concordance in 3.8.11 omits timefix on forwarding. Use its
    # numeric engine directly, with the explicit flag, for an exact oracle.
    fit <- tryCatch(do.call(concordancefit,args), error=function(e)e)
    id <- paste(name,response,timewt,if(timefix) "fix" else "exact",suffix,sep="_")
    inputs <- list(name=id,dataset=name,response=response,timewt=timewt,timefix=timefix,
                   ymin=ymin,ymax=ymax,check_ranks=ranks,cluster=cluster)
    if (inherits(fit,"error")) {
        errors[[length(errors)+1]] <<- c(inputs,list(error=conditionMessage(fit)))
        return(invisible(NULL))
    }
    rank_rows <- NULL
    if (ranks) {
        if (nrow(fit$ranks) == 0L) {
            rank_rows <- list()
        } else {
            stopifnot(identical(names(fit$ranks),c("time","rank","timewt","casewt")))
            rank_rows <- lapply(seq_len(nrow(fit$ranks)),function(i)unname(as.numeric(fit$ranks[i,])))
        }
    }
    counts <- if(is.matrix(fit$count)) colSums(fit$count) else fit$count
    cases[[length(cases)+1]] <<- c(inputs,list(
        count=unname(counts),concordance=unname(fit$concordance),
        variance=unname(fit$var),cvar=unname(fit$cvar),
        dfbeta=unname(fit$dfbeta),influence=unname(fit$influence),ranks=rank_rows))
}

for (response in c("right","counting")) {
    timeweights <- if(response == "right") c("n","S","S/G","n/G2","I") else c("n","S","I")
    for (tw in timeweights) {
        add("one_original_event",response,tw,ranks=FALSE)
        add("one_event_after_censor",response,tw,ranks=FALSE)
        add("two_original_events",response,tw,ranks=FALSE,ymax=1,suffix="cutoff")
        add("two_events_after_censor",response,tw,ranks=FALSE,ymax=2,suffix="cutoff")
        add("sparse_strata",response,tw,ranks=FALSE)
        add("sparse_strata",response,tw,ranks=FALSE,ymax=2,suffix="cutoff")
        add("sparse_strata",response,tw,ranks=FALSE,ymax=2,cluster=TRUE,suffix="cutoff_cluster")
        add("ymin_floor",response,tw,ymin=2,suffix="lower")
    }
    add("zero_weight_simultaneous_event",response,"I",ranks=FALSE)
    for (name in c("all_censored","simultaneous_events","simultaneous_events_tied_scores","all_zero_weights")) {
        add(name,response)
    }
    add("event_after_censors",response,ranks=FALSE)
    add("all_events",response,ymax=.5,suffix="none_before_upper")
    add("all_events",response,ymin=3,ymax=2,suffix="empty_window")
    add("bound_limits",response)
    add("bound_limits",response,ymin=3,ymax=2,suffix="reversed")
    add("bound_limits",response,ymin=-Inf,ymax=Inf,suffix="infinite_limits")
    add("bound_limits",response,ymin=1,ymax=Inf,suffix="infinite_upper")
    add("bound_limits",response,ymin=Inf,suffix="infinite_lower")
    add("bound_limits",response,ymin=Inf,ymax=Inf,suffix="both_positive_infinite")
    add("bound_limits",response,ymin=Inf,ymax=2,suffix="empty_after_infinite_lower")
    add("bound_limits",response,ymax=-Inf,suffix="negative_infinite_upper")
    for (name in c("aeq_absolute","aeq_relative","aeq_adjacent_chain","negative_times")) {
        for (fix in c(FALSE,TRUE)) add(name,response,timefix=fix)
    }
    add("negative_times",response,ymin=-2.5,ymax=0,ranks=TRUE,suffix="negative_bounds")
}
add("disjoint_counting_events","counting")
for (fix in c(FALSE,TRUE)) {
    add("ymin_after_timefix","counting",timefix=fix,ymin=2)
    add("near_zero_interval","counting",timefix=fix)
}

# Nonfinite outputs are strings so NaN and +/-Inf are distinguished from null.
limitations <- c(
    "Single retained rank rows lose matrix dimensions in R 3.8.11; these cases use ranks=FALSE.",
    "A single observation can trigger an R dimension-drop error; no n=1 numeric reference is used.",
    "R can read misaligned time weights at unique event times with zero death weight; zero-weight event fixture shares a positive-weight event time instead.",
    "R pooled rank assembly fails for censored strata; stratified cases use ranks=FALSE.",
    "R keepstrata=FALSE can drop count dimensions before pooled colSums; references retain strata counts and sum only those counts afterward.",
    "R keepstrata=FALSE optimization for >10 strata bypasses per-stratum one-event fallback; this representation-dependent bug is excluded.",
    "All-zero weights are referenced with timewt=n; R non-n weights can fail with an internal program error."
)
cat(jsonlite::toJSON(list(survival_version=as.character(packageVersion("survival")),
    datasets=lapply(datasets,as.list),cases=cases,error_cases=errors,reference_limitations=limitations),
    auto_unbox=TRUE,digits=17,pretty=TRUE,na="string",null="null"))
