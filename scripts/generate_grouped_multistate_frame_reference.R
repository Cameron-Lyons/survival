#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/grouped_multistate_frame_reference.json"
data <- data.frame(
    time=c(1,2,4,1,3,3,5,8,2,6),
    event=factor(c("a","censor","b","b","a","censor","c","censor","c","a"),
                 levels=c("censor","a","b","c","unused")),
    group=factor(c(rep("first",3),rep("second",5),rep("third",2)),
                 levels=c("third","first","second","unused")),
    weight=c(1,1.5,2,.5,1,2,1.5,1,2,.5)
)
fields <- c("n.risk","n.event","n.censor","pstate","std.err","lower","upper")
raw_snapshot <- function(fit) {
    result <- list(time=I(fit$time), states=I(fit$states))
    for (name in fields) if (!is.null(fit[[name]])) result[[name]] <- unname(fit[[name]])
    result
}
cases <- list()
for (se_fit in c(TRUE,FALSE)) {
    for (initial_row in c(FALSE,TRUE)) {
        fit <- survfit(Surv(time,event) ~ group, data, weights=weight, se.fit=se_fit)
        if (initial_row) fit <- survival:::survfit0(fit)
        for (selection_name in c("all","reversed","repeated","mixed")) {
            selection <- switch(selection_name, all=seq_along(fit$states),
                reversed=rev(seq_along(fit$states)), repeated=c(2L,2L), mixed=c(3L,1L,3L,2L))
            groups <- list()
            frame <- list()
            labels <- sub("^group=", "", names(fit$strata))
            for (i in seq_along(fit$strata)) {
                source <- fit[i, ]
                selected <- source[, selection]
                groups[[i]] <- list(label=labels[[i]], raw_selected=raw_snapshot(selected))
            }
            for (state_position in seq_along(selection)) {
                for (i in seq_along(fit$strata)) {
                    source <- fit[i, ]
                    selected <- source[, selection]
                    block <- list(time=source$time)
                    for (name in fields) {
                        if (is.null(selected[[name]])) next
                        # Stock [.survfitms leaves n.censor unsliced. Its summary
                        # consequently fails or reports the wrong censor column.
                        # Keep raw selected matrices above as provenance, and
                        # select this count from the original matrix by position.
                        block[[name]] <- if (name == "n.censor") source[[name]][,selection[[state_position]]]
                                         else selected[[name]][,state_position]
                    }
                    block$strata <- rep(labels[[i]],length(source$time))
                    block$state <- rep(selected$states[[state_position]],length(source$time))
                    for (name in names(block)) frame[[name]] <- c(frame[[name]],block[[name]])
                }
            }
            cases[[length(cases)+1L]] <- list(name=paste(selection_name,se_fit,initial_row,sep="_"),
                se_fit=se_fit,initial_row=initial_row,selection=I(selection-1L),
                groups=groups,expected=lapply(frame,I))
        }
    }
}
reference <- list(metadata=list(generator="scripts/generate_grouped_multistate_frame_reference.R",
    r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival")),
    note="Raw selected matrices preserve repeated state columns. Expected n.censor is selected from the original fit because stock [.survfitms does not subset it and summary data.frame can fail."),
    data=list(time=I(data$time),event=I(as.character(data$event)),group=I(as.character(data$group)),
              weight=I(data$weight)),levels=list(event=I(levels(data$event)),group=I(levels(data$group))),
    cases=cases)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null")
cat(length(cases),"grouped multi-state frame cases written to",output,"\n")
