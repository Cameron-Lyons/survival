# Regenerate with R survival 3.8.11 and jsonlite:
# Rscript python/tests/fixtures/concordance_tied_censoring_r3811.R > \
#   python/tests/fixtures/concordance_tied_censoring_r3811.json
library(survival)
stopifnot(as.character(packageVersion("survival")) == "3.8.11")

datasets <- list()
for (orientation in c("ascending", "descending", "tied")) {
    scores <- switch(orientation, ascending=c(1,2), descending=c(2,1), tied=c(1,1))
    datasets[[paste0("pair_", orientation)]] <- data.frame(
        start=c(0,0), time=c(1,1), status=c(1,0), score=scores, w=c(1,1))
}
datasets$weighted_minimal <- data.frame(
    start=c(0,0,0), time=c(1,1,2), status=c(1,0,1),
    score=c(2,1,3), w=c(2,3,1))
datasets$untied <- data.frame(
    start=c(0,0,0), time=c(1,2,3), status=c(1,1,1),
    score=c(2,1,3), w=c(1,1,1))
datasets$zero_censor_weight <- data.frame(
    start=c(0,0,0,0), time=c(1,1,2,3), status=c(1,0,1,1),
    score=c(2,1,3,2), w=c(2,0,1,3))
datasets$zero_event_weight <- data.frame(
    start=c(0,0,0,0), time=c(1,1,2,3), status=c(1,0,1,1),
    score=c(2,1,3,2), w=c(0,2,1,3))
datasets$mixed <- data.frame(
    start=rep(0,12), time=c(1,1,1,1,2,2,2,3,3,4,5,5),
    status=c(1,1,1,0,0,0,1,1,0,1,0,1),
    score=c(2,2,1,2,1,3,2,3,2,1,2,3),
    w=c(.5,1.25,2,.75,1.5,0,2.25,.5,1.75,1,2,1.25))
datasets$entry <- datasets$mixed
datasets$entry$start <- c(0,0,.5,0,1,0,1,2,1,3,4,2)
datasets$stratified <- rbind(datasets$mixed, datasets$mixed)
datasets$stratified$group <- rep(c("A", "B"), each=12)
datasets$stratified$score[13:24] <- 5 - datasets$stratified$score[13:24]
datasets$stratified$w[13:24] <- (rev(datasets$stratified$w[13:24]) + .25) * .75
datasets$stratified$cluster <- rep(c("a", "b", "a", "c", "d", "e"),4)
datasets$near <- data.frame(
    start=c(0,0,0,0,0), time=c(1+5e-10,1,2,3,4),
    status=c(1,0,1,0,1), score=c(2,1,3,2,1), w=c(2,3,1,2,1))
datasets$near_entry <- data.frame(
    start=c(0,1-5e-10,0,1,0), time=c(1,2,2,3,4),
    status=c(1,0,1,0,1), score=c(2,1,3,2,1), w=c(2,3,1,2,1))

cases <- list()
rank_matrix <- function(f) {
    # R drops the one-event residual matrix to a four-row, one-column frame.
    if (ncol(f$ranks) == 1) matrix(f$ranks[[1]],nrow=1) else
        as.matrix(f$ranks[c("time","rank","timewt","casewt")])
}
add <- function(dataset, response="right", timewt="n", timefix=FALSE,
                ymin=NULL, ymax=NULL, cluster=FALSE, suffix="") {
    d <- datasets[[dataset]]
    response_data <- function(data) {
        if(response == "right") Surv(data$time,data$status) else
            Surv(data$start,data$time,data$status)
    }
    # concordance.formula in 3.8.11 omits timefix when forwarding to this
    # engine. Call it directly so timefix=FALSE fixtures really use exact times.
    args <- list(y=response_data(d), x=d$score,
                 weights=d$w, timewt=timewt, timefix=timefix,
                 influence=3, ranks=!("group" %in% names(d)), keepstrata=TRUE)
    if ("group" %in% names(d)) args$strata <- d$group
    if (!is.null(ymin)) args$ymin <- ymin
    if (!is.null(ymax)) args$ymax <- ymax
    if (cluster) args$cluster <- d$cluster
    f <- tryCatch(do.call(survival:::concordancefit, args), error=function(e)
        stop(paste(dataset,response,timewt,suffix,conditionMessage(e))))
    # survival 3.8.11's pooled ranks assignment fails with censored strata.
    # Rank rows are stratum-local, so obtain these from separate R fits;
    # all count/influence/variance references still use the pooled R fit.
    rank_rows <- if (!("group" %in% names(d))) rank_matrix(f) else {
        do.call(rbind,lapply(split(d,d$group),function(group_data) {
            rank_args <- args
            rank_args$y <- response_data(group_data)
            rank_args$x <- group_data$score
            rank_args$strata <- NULL
            rank_args$weights <- group_data$w
            rank_args$cluster <- NULL
            rank_args$ranks <- TRUE
            rank_matrix(do.call(survival:::concordancefit,rank_args))
        }))
    }
    cases[[length(cases)+1]] <<- list(
        name=paste(dataset,response,gsub("/","_",timewt),suffix,sep="_"),
        dataset=dataset, response=response, timewt=timewt, timefix=timefix,
        ymin=ymin, ymax=ymax, cluster=cluster,
        concordance=unname(f$concordance),
        count=unname(if(is.matrix(f$count)) colSums(f$count) else f$count),
        variance=unname(f$var), cvar=unname(f$cvar),
        dfbeta=unname(f$dfbeta), influence=unname(f$influence),
        ranks=unname(rank_rows)
    )
}
for (dataset in c("pair_ascending","pair_descending","pair_tied",
                  "weighted_minimal","untied","zero_censor_weight","zero_event_weight")) {
    for (response in c("right","counting")) add(dataset,response)
}
for (dataset in c("mixed","stratified")) {
    for (tw in c("n","I","S","S/G","n/G2")) add(dataset,"right",tw)
    for (tw in c("n","I","S")) add(dataset,"counting",tw)
}
for (tw in c("n","I","S")) add("entry","counting",tw)
for (response in c("right","counting")) {
    for (fix in c(FALSE,TRUE)) add("near",response,timefix=fix,suffix=paste0("fix",fix))
    for (tw in c("n","S")) {
        add("mixed",response,tw,ymax=1,suffix="upper_at_tie")
        add("mixed",response,tw,ymin=2,ymax=3,suffix="window_at_ties")
    }
}
for (fix in c(FALSE,TRUE)) add("near_entry","counting",timefix=fix,suffix=paste0("fix",fix))
for (response in c("right","counting")) {
    add("stratified",response,cluster=TRUE,suffix="cluster")
}

multiscore_cases <- list()
for (dataset in c("entry","stratified")) {
    d <- datasets[[dataset]]
    d$score2 <- d$score + .75 * rep(c(1,-1,0,1,0,-1,1,-1,0,1,-1,0),length.out=nrow(d))
    d$cluster <- rep(c("a","b","a","c","d","e"),length.out=nrow(d))
    name <- paste0("multi_",dataset)
    datasets[[name]] <- d
    for (response in c("right","counting")) {
        for (tw in c("n","S")) for (cluster in c(FALSE,TRUE)) {
            y <- if(response == "right") Surv(d$time,d$status) else
                Surv(d$start,d$time,d$status)
            X <- as.matrix(d[c("score","score2")])
            args <- list(y=y,x=X,weights=d$w,timewt=tw,timefix=FALSE,
                         influence=3,ranks=FALSE,keepstrata=TRUE)
            if(cluster) args$cluster <- d$cluster
            if("group" %in% names(d)) {
                # R 3.8.11's joint-score stratified result fails dimname assembly.
                # Each single-score fit pools its strata correctly; the joint IJ
                # covariance is crossprod of their aligned (clustered) dfbeta.
                args$strata <- d$group
                fits <- lapply(seq_len(ncol(X)),function(j) {
                    single_args <- args
                    single_args$x <- X[,j]
                    do.call(survival:::concordancefit,single_args)
                })
                dfbeta <- do.call(cbind,lapply(fits,function(f) f$dfbeta))
                covariance <- crossprod(dfbeta)
                count <- do.call(rbind,lapply(fits,function(f) colSums(f$count)))
                concordance <- vapply(fits,function(f) f$concordance,numeric(1))
                cvar <- vapply(fits,function(f) f$cvar,numeric(1))
                influence <- lapply(fits,function(f) unname(f$influence))
                reference <- "crossprod of pooled single-score R dfbeta"
            } else {
                f <- do.call(survival:::concordancefit,args)
                dfbeta <- f$dfbeta
                covariance <- f$var
                count <- f$count
                concordance <- f$concordance
                cvar <- f$cvar
                influence <- lapply(seq_len(ncol(X)),function(j) unname(f$influence[,,j]))
                reference <- "joint R concordancefit"
            }
            stopifnot(all(diag(covariance)>0),abs(covariance[1,2])>1e-8,
                      det(covariance)>0)
            multiscore_cases[[length(multiscore_cases)+1]] <- list(
                name=paste(name,response,tw,if(cluster) "cluster" else "unclustered",sep="_"),
                dataset=name,response=response,timewt=tw,cluster=cluster,reference=reference,
                concordance=unname(concordance),count=unname(count),
                variance=unname(diag(covariance)),covariance=unname(covariance),
                cvar=unname(cvar),dfbeta=unname(t(dfbeta)),influence=influence)
        }
    }
}
cat(jsonlite::toJSON(list(survival_version=as.character(packageVersion("survival")),
                         datasets=lapply(datasets,as.list),cases=cases,
                         multiscore_cases=multiscore_cases),
                     auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null"))
