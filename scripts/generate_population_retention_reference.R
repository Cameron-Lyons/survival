#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/population_retention_reference.json"
data <- data.frame(time=c(100,200,400,800,NA,300,500,600),
    start=c(0,50,100,200,0,100,200,100), status=c(1,0,1,0,1,1,0,1),
    agey=c(40,50,60,70,80,55,65,75), sex=c(1,2,1,2,1,2,1,2),
    entry=as.Date(rep("2000-01-01",8)),
    grp=factor(c("a","b","a","b","a","b","a","b"),levels=c("b","a","unused")),
    z=c(2,1,3,2,3,1,2,3),score=c(1,0,2,1,2,0,1,2))
data$band <- tcut(data$agey,c(0,60,100),scale=365.25)
training <- data.frame(time=c(2,3,4,6,7,9,10,12),status=c(1,0,1,1,0,1,0,1),
                       z=c(1,2,3,1,2,3,1,2))
cox <- coxph(Surv(time,status)~z,training)
snapshot <- function(value) {
    if (is.null(value)) return(NULL)
    if (inherits(value,"Surv")) return(list(kind="surv",values=unname(as.matrix(value)),
                                           surv_type=attr(value,"type")))
    if (inherits(value,"tcut")) return(list(kind="tcut",values=I(as.numeric(value)),
        cutpoints=I(attr(value,"cutpoints")),levels=I(attr(value,"labels"))))
    if (is.factor(value)) return(list(kind="factor",values=I(as.character(value)),
                                      levels=I(levels(value)),codes=I(as.integer(value))))
    if (inherits(value,"Date")) return(list(kind="date",values=I(as.character(value))))
    if (is.matrix(value)) return(list(kind="matrix",values=unname(value)))
    list(kind="vector",values=I(unname(value)))
}
cases <- list()
add <- function(name,function_name,formula,options=list(),table="none",subset=NULL,
                weights=FALSE,na_action="omit") {
    arguments <- c(list(formula=as.formula(formula),data=data,na.action=get(paste0("na.",na_action))),options)
    if (!is.null(subset)) arguments$subset <- subset
    if (weights) arguments$weights <- c(1,2,1,3,1,2,1,1)
    if (table=="population") {
        arguments$ratetable <- survexp.us
        arguments$rmap <- quote(list(age=agey*365.25,sex=sex,year=entry))
    } else if (table=="cox") {
        arguments$ratetable <- cox
        arguments$rmap <- quote(list(z=score+1))
    }
    fit <- suppressWarnings(do.call(function_name,arguments))
    if (is.list(fit)) {
        expected <- list(model=if (is.null(fit$model)) NULL else lapply(fit$model,snapshot),
                         x=snapshot(fit$x),y=snapshot(fit$y))
        if (function_name=="pyears") expected$values <- as.vector(fit$pyears)
        else expected$values <- as.vector(fit$surv)
    } else expected <- list(individual=I(unname(fit)))
    cases[[length(cases)+1L]] <<- list(name=name,function_name=function_name,formula=formula,
        term_labels=I(attr(terms(as.formula(formula)),"term.labels")),
        options=options,table=table,subset=if(is.null(subset)) NULL else I(subset-1L),
        weights=weights,na_action=na_action,expected=expected)
}
for (function_name in c("pyears","survexp")) {
    table <- if(function_name=="pyears") "none" else "population"
    for (flags in list(list(),list(x=TRUE),list(y=TRUE),list(x=TRUE,y=TRUE),
                      list(model=TRUE),list(model=TRUE,x=TRUE,y=TRUE))) {
        suffix <- if(length(flags)) paste(names(flags),collapse="_") else "default"
        add(paste(function_name,suffix,sep="_"),function_name,"Surv(time, status) ~ grp + sex",
            flags,table,subset=c(8,2,5,1,8,6),weights=TRUE,na_action="exclude")
    }
    add(paste0(function_name,"_precedence"),function_name,"time ~ 1",
        list(model=TRUE,x=NA,y=NA),table)
    add(paste0(function_name,"_ones"),function_name,"time ~ 1",list(x=TRUE,y=TRUE),table)
    add(paste0(function_name,"_expression_model"),function_name,"time ~ factor(z) + I(sex + 1)",
        list(model=TRUE),table,subset=c(1,3,5,7))
}
add("pyears_counting","pyears","Surv(start, time, status) ~ grp",list(x=TRUE,y=TRUE))
add("pyears_counting_model","pyears","Surv(start, time, status) ~ grp",list(model=TRUE),"population")
for (kind in c("xy","model")) {
    flags <- if(kind=="xy") list(x=TRUE,y=TRUE) else list(model=TRUE)
    add(paste0("pyears_tcut_",kind),"pyears",
        "time ~ tcut(agey, c(0, 60, 100), scale = 365.25) + grp",flags,subset=c(7,2,5,1))
    add(paste0("pyears_band_",kind),"pyears","time ~ band",flags,subset=c(7,2,5,1))
    add(paste0("pyears_cut_",kind),"pyears",
        "time ~ cut(agey, c(45, 65, 90))",flags,subset=c(7,2,5,1))
    add(paste0("pyears_cut_codes_",kind),"pyears",
        "time ~ cut(agey, c(45, 65, 90), labels = FALSE)",flags)
    add(paste0("survexp_no_response_",kind),"survexp","~ grp",
        c(flags,list(times=c(50,200,800),scale=10)),"population")
    add(paste0("survexp_cox_",kind),"survexp","time ~ grp",
        c(flags,list(times=c(2,5,10))),"cox",subset=c(7,2,5,1),weights=TRUE)
    add(paste0("survexp_cox_no_response_",kind),"survexp","~ grp",
        c(flags,list(times=c(2,5,10))),"cox")
}
for (method in c("individual.s","individual.h")) {
    for (action in c("omit","exclude")) {
        add(paste("survexp",method,action,sep="_"),"survexp","time ~ 1",
            list(method=method,model=NA,x=NA,y=NA),"population",subset=c(8,5,2,8,1),na_action=action)
    }
}
reference <- list(metadata=list(generator="scripts/generate_population_retention_reference.R",
    r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival"))),
    data=lapply(data,snapshot),training=training,cases=cases)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null",dataframe="columns")
cat(length(cases),"population retention cases written to",output,"\n")
