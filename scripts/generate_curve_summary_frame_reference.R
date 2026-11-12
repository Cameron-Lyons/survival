#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/curve_summary_frame_reference.json"

# Share the small, independently generated count-selection dataset.
source <- fromJSON("python/tests/fixtures/summary_counts_reference.json", simplifyVector=FALSE)
data <- as.data.frame(lapply(source$data, unlist), stringsAsFactors=FALSE)
data$event <- factor(data$event, levels=unlist(source$event_levels))
specs <- list(
    km=list(kind="formula", formula="Surv(time,status) ~ 1"),
    km_groups=list(kind="formula", formula="Surv(time,status) ~ group"),
    aj=list(kind="formula", formula="Surv(time,event) ~ 1", fit_args=list(id="id")),
    aj_none=list(kind="formula", formula="Surv(time,event) ~ 1",
                 fit_args=list(id="id", conf.type="none")),
    cox=list(kind="cox", formula="Surv(time,status) ~ x",
             newdata=list(x=I(c(0,2)))),
    cox_single=list(kind="cox", formula="Surv(time,status) ~ x",
                    newdata=list(x=I(1))),
    cox_mean=list(kind="cox", formula="Surv(time,status) ~ x"),
    cox_none=list(kind="cox", formula="Surv(time,status) ~ x",
                  newdata=list(x=I(c(0,2))), fit_args=list(conf.type="none")),
    cox_no_se=list(kind="cox", formula="Surv(time,status) ~ x",
                   newdata=list(x=I(c(0,2))), fit_args=list(se.fit=FALSE)),
    cox_groups=list(kind="cox", formula="Surv(time,status) ~ x + strata(group)",
                    newdata=list(x=I(c(0,2)))),
    cox_matched=list(kind="cox", formula="Surv(time,status) ~ x + strata(group)",
                     newdata=list(x=I(c(0,2)), group=I(c("a","b")))),
    cox_collision=list(kind="cox", formula="Surv(time,status) ~ x",
                       newdata=list(x=I(c(0,2)), time=I(c(99,88)), label=I(c("a","b")))),
    coxms=list(kind="coxms", formula="Surv(time,event) ~ x",
               newdata=list(x=I(c(0,2)))),
    coxms_single=list(kind="coxms", formula="Surv(time,event) ~ x",
                      newdata=list(x=I(1)))
)
fits <- lapply(specs, function(spec) {
    if (spec$kind == "formula") {
        arguments <- spec$fit_args
        if (!is.null(arguments$id)) arguments$id <- data[[arguments$id]]
        return(do.call(survfit, c(list(formula=as.formula(spec$formula), data=data), arguments)))
    }
    arguments <- list(formula=as.formula(spec$formula), data=data)
    if (spec$kind == "coxms") arguments$id <- data$id
    model <- do.call(coxph, arguments)
    arguments <- list(formula=model)
    if (!is.null(spec$newdata)) arguments$newdata <- as.data.frame(spec$newdata)
    do.call(survfit, c(arguments, spec$fit_args))
})
queries <- list(
    events=list(),
    censored=list(censored=TRUE),
    requested=list(times=I(c(2,4,7)), scale=2),
    unordered=list(times=I(c(7,2,2,0,10)), extend=TRUE, dosum=FALSE),
    single=list(times=I(4)),
    empty=list(times=I(20))
)
cases <- list()
for (fit in names(fits)) {
    for (query in names(queries)) {
        if (query == "empty" && fit %in% c("aj", "aj_none", "coxms", "coxms_single")) next
        arguments <- queries[[query]]
        if (!is.null(arguments$times)) arguments$times <- as.numeric(arguments$times)
        value <- tryCatch(do.call(summary, c(list(object=fits[[fit]], data.frame=TRUE), arguments)),
                          error=function(e) list(error=conditionMessage(e)))
        stock_error <- NULL
        if (fit == "aj_none" && !is.null(value$error)) {
            stock_error <- value$error
            # Stock summary.survfitms binds NULL confidence bands with the SE
            # column. The same fit with bands has identical probabilities/SEs.
            value <- do.call(summary, c(list(object=fits$aj, data.frame=TRUE), arguments))
            value$lower <- value$upper <- NULL
        }
        stock_columns <- names(value)
        if (query == "empty") {
            # Preserve the original fitted schema on empty output, rather than
            # R's NULL removal of every component except time and n.risk.
            template <- summary(fits[[fit]], data.frame=TRUE)
            value <- template[integer(0), , drop=FALSE]
        }
        if (length(specs[[fit]]$newdata) == 1L) {
            # [.data.frame drops a one-column newdata to an unnamed vector in R.
            # Keep the actual covariate name, as the existing Python frames do.
            unnamed <- grep("^fit\\$newdata\\[", names(value))
            if (length(unnamed)) names(value)[unnamed] <- names(specs[[fit]]$newdata)
        }
        expected <- if (!is.null(value$error)) value else
            lapply(value, function(column) I(if (is.factor(column)) as.character(column) else column))
        cases[[length(cases)+1L]] <- list(name=paste(fit,query,sep="/"), fit=fit,
                                         arguments=queries[[query]], expected=expected,
                                         stock_error=stock_error, stock_columns=I(stock_columns))
    }
}
serialize <- function(value) toJSON(value, auto_unbox=TRUE, digits=NA, na="null", null="null")
header <- serialize(list(survival_version=as.character(packageVersion("survival")), fits=specs))
writeLines(c(substring(header,1L,nchar(header)-1L), ',"cases":[',
    vapply(seq_along(cases),function(i) paste0(serialize(cases[[i]]),
        if (i < length(cases)) "," else ""), ""), "]}"), output, useBytes=TRUE)
cat(length(cases), "curve-summary references written with", R.version.string, "\n")
